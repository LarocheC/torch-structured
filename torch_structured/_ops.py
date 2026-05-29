"""Single dispatch point for kernel-backed ops (DISP-01..05, D-04..D-08).

This module is the only place where backend selection happens. It reads the
``TORCH_STRUCTURED_BACKEND`` environment variable at import time, runs
``_resolve()`` to pick a backend (one of ``triton``, ``cuda``, ``torch``,
``auto``), and binds module-level callable attributes (``butterfly_multiply``,
``hadamard_transform``, ``diag_mult``) to the chosen implementation. The
``set_backend()`` API re-runs the resolver and re-binds the same names so tests
can switch backends without per-call branching.

Call-site contract (D-05) — load-bearing for Phase 5+ consumer migration
-----------------------------------------------------------------------

Consumers (nn.Module forwards, tests, etc.) MUST call via attribute access on
this module so that ``set_backend()`` re-bindings take effect for already-loaded
callers.

CORRECT — attribute access (re-reads binding on each call)::

    import torch_structured

    def some_function(twiddle, x, ...):
        return torch_structured._ops.butterfly_multiply(twiddle, x, ...)

    # Equivalent attribute-access form (also correct):
    from torch_structured import _ops
    def some_function(twiddle, x, ...):
        return _ops.butterfly_multiply(twiddle, x, ...)

WRONG — captures the CURRENT object at import time::

    from torch_structured._ops import butterfly_multiply

    def some_function(twiddle, x, ...):
        butterfly_multiply(twiddle, x, ...)   # set_backend() rebind invisible

Python's ``from X import Y`` binds ``Y`` in the caller's namespace at import
time; subsequent reassignments to ``X.Y`` are invisible to the caller.
``X.Y`` (attribute access) re-reads the binding on every call. Phase 5 onward
enforces this in consumer plans.
"""
import importlib
import json
import logging
import math
import os
import pathlib
import warnings

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

log = logging.getLogger("torch_structured")

# ── Public, module-level callables ──────────────────────────────────────
# These are rebound by ``_resolve()`` at import time and by every call to
# ``set_backend()``. Consumers must use attribute access (see module docstring).
butterfly_multiply = None      # type: ignore[assignment]
hadamard_transform = None      # type: ignore[assignment]
diag_mult = None               # type: ignore[assignment]

_BACKEND = "uninitialized"


def _has_triton() -> bool:
    """Return True iff the ``triton`` package is importable AND a CUDA device is present."""
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


_HAS_CUDA_LEGACY_BUTTERFLY_RUNTIME_OK: "bool | None" = None  # one-shot cache


def _has_cuda_legacy() -> bool:
    """Return True iff the compiled C++ butterfly op is registered AND runtime-dispatchable.

    The .so is loaded as a side effect of importing
    ``torch_structured.butterfly`` (see butterfly/__init__.py:22-39). The probe
    checks BOTH whether the registration succeeded AND whether the CUDA
    dispatch actually works at runtime (one-shot cached per process — Phase 9
    09-01 honest-probe tightening per CHECKER B3 / D-21).

    Why the runtime check matters: a build with CUDA-version mismatch (or a
    CPU-only build with the schema still registered) leaves
    ``hasattr(torch.ops.torch_structured, "butterfly_multiply")`` True but the
    CUDA dispatch raises ``RuntimeError("Not compiled with CUDA support")`` on
    first invocation. Without the runtime check, the conftest cuda axis runs
    and fails with a verbose stack trace; with the runtime check the cuda
    axis is honestly skipped per D-62 (matches the
    ``_has_cuda_legacy_diag_mult/hadamard`` sentinel-based pattern at
    ``_cuda_legacy/diag_mult.py:24-29`` and ``_cuda_legacy/hadamard.py``).

    The runtime check is a tiny ``log_n=2`` CUDA call (n=4) cached for the
    lifetime of the process; one-shot cost bounded at first probe.
    """
    global _HAS_CUDA_LEGACY_BUTTERFLY_RUNTIME_OK
    if not hasattr(torch.ops.torch_structured, "butterfly_multiply"):
        return False
    if not torch.cuda.is_available():
        return False
    if _HAS_CUDA_LEGACY_BUTTERFLY_RUNTIME_OK is None:
        try:
            log_n = 2
            n = 1 << log_n
            tw = torch.zeros(1, 1, log_n, n // 2, 2, 2, device="cuda", dtype=torch.float32)
            x = torch.zeros(1, 1, n, device="cuda", dtype=torch.float32)
            torch.ops.torch_structured.butterfly_multiply(tw, x, True, n)
            _HAS_CUDA_LEGACY_BUTTERFLY_RUNTIME_OK = True
        except RuntimeError:
            _HAS_CUDA_LEGACY_BUTTERFLY_RUNTIME_OK = False
    return _HAS_CUDA_LEGACY_BUTTERFLY_RUNTIME_OK


def _has_cuda_legacy_diag_mult() -> bool:
    """Per-op honest probe (CHECKER B3) for the legacy ``_diag_mult_cuda`` extension.

    Symmetric to ``_has_cuda_legacy()`` but checks the pybind11 ``_diag_mult_cuda``
    extension (D-22). Returns the ``HAS_CUDA_LEGACY_DIAG_MULT`` sentinel from
    ``_cuda_legacy/diag_mult.py`` — True iff the ``.so`` was built and the
    top-of-module try-import succeeded. Never raises; returns a clean bool.

    Phase 10 D-74b: wrapped in ``warnings.catch_warnings()`` +
    ``simplefilter("ignore", DeprecationWarning)`` so the probe stays SILENT.
    The user-facing DeprecationWarning from ``_cuda_legacy/__init__.py`` is
    reserved for explicit ``set_backend('cuda')`` invocations, not for the
    silent probe path consumed by the Phase 9 backend fixture.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            from torch_structured._cuda_legacy.diag_mult import HAS_CUDA_LEGACY_DIAG_MULT
            return HAS_CUDA_LEGACY_DIAG_MULT
        except ImportError:
            return False


def _has_cuda_legacy_hadamard() -> bool:
    """Per-op honest probe (CHECKER B3) for the legacy ``_hadamard_cuda`` extension.

    Symmetric to ``_has_cuda_legacy_diag_mult()``; returns the ``HAS_CUDA_LEGACY_HADAMARD``
    sentinel from ``_cuda_legacy/hadamard.py``. Never raises; returns a clean bool.

    Phase 10 D-74b: wrapped in ``warnings.catch_warnings()`` +
    ``simplefilter("ignore", DeprecationWarning)`` so the probe stays SILENT.
    The user-facing DeprecationWarning from ``_cuda_legacy/__init__.py`` is
    reserved for explicit ``set_backend('cuda')`` invocations, not for the
    silent probe path consumed by the Phase 9 backend fixture.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            from torch_structured._cuda_legacy.hadamard import HAS_CUDA_LEGACY_HADAMARD
            return HAS_CUDA_LEGACY_HADAMARD
        except ImportError:
            return False


def _has_cuda_legacy_for_op(op_name: str) -> bool:
    """Per-op cuda-legacy probe — multiplexes the existing per-op probes by name.

    Phase 9 D-62a — symmetric to ``_has_triton_kernel(op_name)`` (Phase 5/6/7
    per-op honest probe). Consumer: ``tests/conftest.py`` ``backend`` fixture
    (Phase 9 D-62b) uses ``@pytest.mark.op('<op_name>')`` to bind a test to
    a specific op, then calls this probe to decide whether to skip the
    ``cuda`` axis when that op's compiled ``.so`` is missing.

    Branches:
    - ``"butterfly_multiply"`` → ``_has_cuda_legacy()`` (the eagerly-loaded
      ``_butterfly.so`` symbol, verified at this module's lines 72-79)
    - ``"diag_mult"`` → ``_has_cuda_legacy_diag_mult()``
    - ``"hadamard_transform"`` → ``_has_cuda_legacy_hadamard()``
    - anything else → ``False`` (explicit branch; no try/except per CLAUDE.md
      ``no try/except in core lib`` — the per-op probes own all the
      try/except surface, D-21 sanctioned site)

    Never raises; returns a clean bool. Threat T-09-01 (CONTEXT.md threat
    model): unknown op_names are accepted explicitly without dynamic import,
    so a typo in ``@pytest.mark.op('butterly_multipl')`` skips the cuda
    axis safely instead of crashing the test collector.

    Example::

        if not torch_structured._ops._has_cuda_legacy_for_op("butterfly_multiply"):
            pytest.skip("No CUDA legacy .so for butterfly_multiply")
    """
    if op_name == "butterfly_multiply":
        return _has_cuda_legacy()
    if op_name == "diag_mult":
        return _has_cuda_legacy_diag_mult()
    if op_name == "hadamard_transform":
        return _has_cuda_legacy_hadamard()
    return False


# Phase 7 name-asymmetry map: the op name (``butterfly_multiply``) doesn't
# match the Triton-package name (``butterfly``). Phase 5 + Phase 6 ops are
# symmetric (their package name equals their op name) and fall through the
# ``.get(op_name, op_name)`` default in ``_has_triton_kernel``.
_TRITON_PACKAGE_NAMES = {
    "butterfly_multiply": "butterfly",
}


def _has_triton_kernel(op_name: str) -> bool:
    """Per-op probe — True only when a real Triton kernel ships for ``op_name``.

    Probes ``torch_structured._triton.<package>.op`` (where ``<package>`` is
    derived from ``op_name`` — see ``_TRITON_PACKAGE_NAMES`` above for the
    name-asymmetry map) and checks that ``<op_name>`` is defined on it.

    Phase 5 lights up ``_has_triton_kernel("diag_mult")`` (package: ``diag_mult``).
    Phase 6 lights up ``_has_triton_kernel("hadamard_transform")`` (package: ``hadamard_transform``).
    Phase 7 lights up ``_has_triton_kernel("butterfly_multiply")`` (package: ``butterfly``).

    Distinguishing this from ``_has_triton()`` is per CHECKER B3: the resolver
    must be honest about backend availability so it never silently lies about
    binding to Triton.
    """
    package_name = _TRITON_PACKAGE_NAMES.get(op_name, op_name)
    try:
        mod = importlib.import_module(f"torch_structured._triton.{package_name}.op")
    except (ImportError, AttributeError):
        return False
    return hasattr(mod, op_name)


def _has_any_triton_kernel() -> bool:
    """Per-op honest probe (CHECKER B3): True iff ANY per-op Triton kernel is installed.

    Lights up progressively across phases (5: diag_mult, 6: hadamard_transform,
    7: butterfly_multiply). The widened predicate used by ``_resolve()`` Step 1
    so the ``auto`` and ``triton`` branches reach ``actual="triton"`` as soon
    as the first per-op Triton kernel ships — without requiring butterfly
    (Phase 7) to land first. Never raises; returns a clean bool.
    """
    for op_name in ("butterfly_multiply", "diag_mult", "hadamard_transform"):
        if _has_triton_kernel(op_name):
            return True
    return False


def _resolve(name: str) -> str:
    """Pick a backend, bind module-level names, return the **actual** chosen name.

    The returned string reflects the ACTUAL binding, not the requested name
    (per CHECKER B3): ``_BACKEND`` always agrees with the function objects that
    are now bound to ``butterfly_multiply`` / ``hadamard_transform`` /
    ``diag_mult``. Observers cannot be deceived about what's bound.

    Validates ``name`` against the exact set ``{triton, cuda, torch, auto}``
    and raises ``ValueError`` otherwise (T-04-01 mitigation — explicit reject
    of arbitrary env-var values; NO dynamic import of ``name`` is performed).
    """
    global butterfly_multiply, hadamard_transform, diag_mult, _BACKEND
    name = (name or "auto").lower()
    if name not in ("triton", "cuda", "torch", "auto"):
        raise ValueError(f"Unknown backend {name!r}; expected triton|cuda|torch|auto")

    # ── Step 1: pick actual backend ─────────────────────────────────────
    # Honest per-op probe (CHECKER B3): _has_any_triton_kernel() is True iff at
    # least one per-op Triton kernel ships. In Phase 5 only diag_mult lights up;
    # Phase 6 adds hadamard_transform; Phase 7 adds butterfly_multiply.
    if name == "auto":
        if _has_any_triton_kernel() and torch.cuda.is_available():
            actual = "triton"
        elif _has_cuda_legacy():
            actual = "cuda"
        else:
            actual = "torch"
    elif name == "triton":
        if _has_any_triton_kernel() and torch.cuda.is_available():
            actual = "triton"
        elif _has_cuda_legacy():
            actual = "cuda"
            log.warning(
                "set_backend('triton') requested but no Triton kernel installed; "
                "falling back to %s",
                actual,
            )
        else:
            actual = "torch"
            log.warning(
                "set_backend('triton') requested but no Triton kernel installed; "
                "falling back to %s",
                actual,
            )
    elif name == "cuda":
        if _has_cuda_legacy():
            actual = "cuda"
        else:
            actual = "torch"
            log.warning(
                "set_backend('cuda') requested but compiled C++ backend is not "
                "available; falling back to %s",
                actual,
            )
    else:  # name == "torch"
        actual = "torch"

    # ── Step 2: bind module-level callables for the actual backend ──────
    if actual == "triton":
        # In Phase 4 this branch is unreachable (no Triton kernels yet); Phase 5+
        # will route through `from torch_structured._triton.butterfly.op import
        # butterfly_multiply` here. Phase 7 lands the butterfly Triton kernel.
        if _has_triton_kernel("butterfly_multiply"):
            from torch_structured._triton.butterfly.op import (  # type: ignore[import-not-found]
                butterfly_multiply as _triton_bm,
            )
            # Phase 9 09-03 D-66b: wrap the Triton kernel binding with a
            # routing closure that consults ``_should_route_to_cuda`` on each
            # call. When the routing table marks (op, log_n, dtype,
            # 'forward') as route_to_cuda AND ``_has_cuda_legacy()`` is True,
            # route to the CUDA path; otherwise fall through to the Triton
            # kernel. When CUDA is unavailable and a cell is marked, emit a
            # one-shot warning (D-61b) and fall back to Triton.
            #
            # The forward direction is the only one the resolver sees here.
            # Backward direction is correctly handled transitively: if the
            # forward routes to CUDA, the autograd graph attached to the
            # CUDA-produced output drives the CUDA backward; if it stays on
            # Triton, the Phase 8 register_autograd _backward callback handles
            # it. No separate backward-direction hook is needed at the
            # resolver level.
            if _has_cuda_legacy():
                # DEPR-02-leak: suppress the leak-site import's warning. This
                # binding runs on the DEFAULT triton backend (routing-fallback
                # closure), so importing _cuda_legacy here must NOT surface the
                # user-facing DeprecationWarning — that is reserved for explicit
                # set_backend('cuda') (D-74b). Mirrors the per-op probe
                # suppression at _ops.py:133-139. (The warning is now an
                # explicit warn_cuda_deprecation() call, so this suppressed
                # import is also side-effect-free and does not consume the
                # _WARNED gate against a later explicit-cuda selection.)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    from torch_structured._cuda_legacy import (
                        butterfly_multiply as _cuda_bm_for_route,
                    )

                def _routed_butterfly_multiply(twiddle, input_, *args, **kwargs):
                    if _should_route_to_cuda(
                        "butterfly_multiply",
                        tuple(input_.shape),
                        input_.dtype,
                        "forward",
                    ):
                        return _cuda_bm_for_route(twiddle, input_, *args, **kwargs)
                    return _triton_bm(twiddle, input_, *args, **kwargs)

                butterfly_multiply = _routed_butterfly_multiply
            else:
                # D-61b: when CUDA is unavailable AND the routing table marks
                # any cell route_to_cuda, log a one-shot warning at first
                # invocation. Bind a thin closure that emits the warning the
                # first time it sees a routed cell, then falls back to
                # Triton.
                def _triton_with_cuda_missing_warning(twiddle, input_, *args, **kwargs):
                    if _ROUTING_TABLE and not _ROUTING_DISABLED:
                        log_n = _shape_to_log_n(tuple(input_.shape))
                        dtype = input_.dtype
                        dtype_str = "fp32" if dtype == torch.float32 else (
                            "complex64" if dtype == torch.complex64 else str(dtype)
                        )
                        key = f"butterfly_multiply::{log_n}::{dtype_str}::forward"
                        cell = _ROUTING_TABLE.get(key)
                        if cell is not None and cell.get("route_to_cuda", False):
                            warn_key = ("butterfly_multiply", log_n, dtype_str, "forward")
                            if warn_key not in _routing_warning_emitted:
                                _routing_warning_emitted.add(warn_key)
                                log.warning(
                                    "torch_structured: routing table marks "
                                    "butterfly_multiply(log_n=%d, dtype=%s) as "
                                    "route_to_cuda, but CUDA legacy backend is "
                                    "unavailable; falling back to Triton (D-61b)",
                                    log_n, dtype_str,
                                )
                    return _triton_bm(twiddle, input_, *args, **kwargs)

                butterfly_multiply = _triton_with_cuda_missing_warning
        elif _has_cuda_legacy():
            from torch_structured._cuda_legacy import butterfly_multiply as _cuda_bm
            butterfly_multiply = _cuda_bm
        else:
            from torch_structured._torch_ref.butterfly import butterfly_multiply_torch
            butterfly_multiply = butterfly_multiply_torch
    elif actual == "cuda":
        # DEPR-02-leak / D-74b: explicit cuda selection — emit the user-facing
        # DeprecationWarning here (at-most-once per process via the _WARNED gate
        # inside warn_cuda_deprecation()).
        from torch_structured._cuda_legacy import (
            butterfly_multiply as _cuda_bm,
            warn_cuda_deprecation,
        )
        warn_cuda_deprecation()
        butterfly_multiply = _cuda_bm
    else:  # actual == "torch"
        from torch_structured._torch_ref.butterfly import butterfly_multiply_torch
        butterfly_multiply = butterfly_multiply_torch

    # diag_mult per-op binding (D-22 — asymmetric fallback). The coarse `actual`
    # signals the user's intent; the per-op binding uses ``_has_triton_kernel`` /
    # ``_has_cuda_legacy_diag_mult`` to honor honest availability. ``_diag_mult_backend``
    # is local — the only consumer is the log.info line below; the module-level
    # ``_BACKEND`` global stays coarse per D-22a recommendation A.
    if actual == "triton" and _has_triton_kernel("diag_mult"):
        from torch_structured._triton.diag_mult.op import diag_mult as _triton_dm
        diag_mult = _triton_dm
        _diag_mult_backend = "triton"
    elif actual == "cuda" and _has_cuda_legacy_diag_mult():
        # DEPR-02-leak / D-74b: explicit cuda selection — emit (idempotent).
        from torch_structured._cuda_legacy import warn_cuda_deprecation
        from torch_structured._cuda_legacy.diag_mult import diag_mult as _cuda_dm
        warn_cuda_deprecation()
        diag_mult = _cuda_dm
        _diag_mult_backend = "cuda"
    else:
        from torch_structured._torch_ref.diag_mult import diag_mult as _torch_dm
        diag_mult = _torch_dm
        _diag_mult_backend = "torch"
        if actual == "cuda":
            log.warning(
                "set_backend('cuda') requested but _diag_mult_cuda not built; "
                "falling back to torch_ref for diag_mult (D-22)"
            )

    # hadamard_transform per-op binding (D-22 / D-36 — same shape as diag_mult above).
    if actual == "triton" and _has_triton_kernel("hadamard_transform"):
        from torch_structured._triton.hadamard_transform.op import hadamard_transform as _triton_ht
        hadamard_transform = _triton_ht
        _hadamard_transform_backend = "triton"
    elif actual == "cuda" and _has_cuda_legacy_hadamard():
        # DEPR-02-leak / D-74b: explicit cuda selection — emit (idempotent).
        from torch_structured._cuda_legacy import warn_cuda_deprecation
        from torch_structured._cuda_legacy.hadamard import hadamard_transform as _cuda_ht
        warn_cuda_deprecation()
        hadamard_transform = _cuda_ht
        _hadamard_transform_backend = "cuda"
    else:
        from torch_structured._torch_ref.hadamard import hadamard_transform_torch as _torch_ht
        hadamard_transform = _torch_ht
        _hadamard_transform_backend = "torch"
        if actual == "cuda":
            log.warning(
                "set_backend('cuda') requested but _hadamard_cuda not built; "
                "falling back to torch_ref for hadamard_transform (D-22)"
            )

    log.info(
        "torch_structured: per-op bindings: butterfly_multiply=%s, diag_mult=%s, hadamard_transform=%s",
        actual, _diag_mult_backend, _hadamard_transform_backend,
    )

    # ── Step 3: D-08 heads-up log ──────────────────────────────────────
    # Per CHECKER B3 tightened condition: emit ONLY when the ACTUAL binding is
    # triton AND a legacy .so is also detected (upgrade-signal heads-up). In
    # Phase 4 this never fires because actual can never be "triton" yet. Phase
    # 5 will exercise this the first time a real Triton kernel ships on a host
    # that also has the legacy .so. Does NOT use ``warnings.warn`` — that is
    # reserved for DEPR-02 in Phase 10 per D-15.
    if actual == "triton" and _has_cuda_legacy():
        log.info(
            "torch_structured: selecting Triton backend; the compiled CUDA "
            "backend is still available via TORCH_STRUCTURED_BACKEND=cuda. "
            "See README for the deprecation timeline."
        )

    _BACKEND = actual
    return actual


def set_backend(name: str) -> str:
    """Public API: switch backend at runtime (primarily for tests, per D-06).

    Returns the **actual** resolved backend name (which may differ from
    ``name`` when a requested backend isn't available — e.g.,
    ``set_backend('triton')`` returns ``'torch'`` in Phase 4 because no real
    Triton kernel exists yet, and emits a ``log.warning`` recording the
    fallback).
    """
    actual = _resolve(name)
    log.info("torch_structured: backend=%s (set_backend)", actual)
    return actual


# ── Phase 9 09-02 (D-63 + D-63a + D-63b + D-63c) ────────────────────────
# Module-level deterministic-mode flag and helpers. Mirrors the shape of
# ``set_backend`` — a single global mutated via a top-level setter that
# returns the PREVIOUS value (save/restore pattern).
_DETERMINISTIC: bool = False


def set_deterministic(value: bool) -> bool:
    """Toggle deterministic-mode for torch_structured Triton kernels (D-63).

    Returns the PREVIOUS value (save/restore pattern, mirrors ``set_backend``).

    Under deterministic mode, the butterfly_multiply backward routes through
    ``butterfly_multiply_torch`` (the pure-PyTorch oracle) — slower but
    deterministic because there is no atomicAdd reduction. Forward is
    unaffected (no atomicAdd in forward; deterministic-by-construction via
    the multi-launch tile structure).

    Composes additively with ``torch.use_deterministic_algorithms(True)``
    (D-63b): the deterministic path is active when EITHER
    ``torch_structured._ops._DETERMINISTIC`` OR
    ``torch.are_deterministic_algorithms_enabled()`` is True. Setting this
    flag does NOT propagate to PyTorch's global determinism flag — users opt
    into PyTorch's global determinism separately.

    Implementation note (D-63a): the gate lives at the WRAPPER LEVEL in
    ``torch_structured/_triton/butterfly/op.py:_backward`` — clones the
    small-N fallback shape verbatim (``twiddle.detach().requires_grad_(True)``
    + ``torch.enable_grad()`` + ``torch.autograd.grad`` against
    ``_butterfly_multiply_torch``). No kernel-side constexpr specialization.

    Example::

        prev = torch_structured.set_deterministic(True)
        try:
            ...  # deterministic backward
        finally:
            torch_structured.set_deterministic(prev)
    """
    global _DETERMINISTIC
    prev = _DETERMINISTIC
    _DETERMINISTIC = bool(value)
    return prev


def _is_deterministic_mode_active() -> bool:
    """Helper consumed by ``_triton/butterfly/op.py:_backward`` (D-63a + D-63b).

    Returns True iff EITHER the library-level ``_DETERMINISTIC`` flag OR
    PyTorch's global ``torch.are_deterministic_algorithms_enabled()`` is
    True. Additive OR composition per D-63b — the two flags are independent;
    activating either one routes the backward through the oracle.

    Inlined as a one-liner to keep the call-path cheap (the gate is
    consulted on every backward call; no measurable overhead at the
    Python-attribute-access level).
    """
    return _DETERMINISTIC or torch.are_deterministic_algorithms_enabled()


# ── Phase 9 09-03 runtime selector (D-66, D-66a, D-66b, D-66c, D-66d) ───
# Static routing table baked from ``07-BASELINE.json`` by
# ``scripts/regenerate_routing_table.py``. The resolver hook in ``_resolve()``
# wraps the bound Triton kernel with a closure that consults
# ``_should_route_to_cuda`` per-invocation; cells where Triton trails CUDA by
# >40% (ratio > 1.67) route to the legacy CUDA path transparently. Users with
# different hardware regenerate locally via ``scripts/regenerate_routing_table.py``.
#
# Test-time override: ``set_routing_enabled(False)`` disables routing for the
# Phase 8 SC#4 reconciliation (D-61a + D-66c). Internal API — NOT exported
# in ``torch_structured/__init__.py``.
_ROUTING_DISABLED: bool = False

# Per-process set tracking which (op, log_n, dtype, direction) cells have
# emitted the "first-route" log message (D-66d). Avoids log spam in tight
# kernel-call loops.
_routing_log_emitted: "set[tuple]" = set()

# Per-process set tracking which (op, log_n, dtype, direction) cells have
# emitted the "CUDA missing — falling back to Triton" warning (D-61b).
_routing_warning_emitted: "set[tuple]" = set()


def _load_routing_table() -> dict:
    """Load the static routing table from ``torch_structured/_routing.json``.

    Phase 9 D-66 — read the keyed-object table at module import time. When the
    file does not exist (e.g., a fresh checkout before
    ``scripts/regenerate_routing_table.py`` has run), return an empty dict —
    the selector then never routes, and ``butterfly_multiply`` behaves as the
    pre-Phase-9-09-03 binding.

    Pre-generation graceful fallback (D-66b): the file is committed to the
    repo as part of Plan 09-03 Task 2, so users see the dev-host bake by
    default. Custom hardware regen produces a new ``_routing.json``.

    No try/except per CLAUDE.md ``no try/except in core lib`` — the
    ``exists()`` check is the gate; the read itself raises only on malformed
    JSON, which would be a packaging bug.
    """
    routing_path = pathlib.Path(__file__).parent / "_routing.json"
    if not routing_path.exists():
        log.warning(
            "torch_structured: _routing.json not found at %s; "
            "runtime selector will not route any cells (Phase 9 D-66 fallback)",
            routing_path,
        )
        return {}
    data = json.loads(routing_path.read_text())
    return data.get("rules", {})


_ROUTING_TABLE: dict = _load_routing_table()


def _shape_to_log_n(shape: tuple) -> int:
    """Derive ``log_n`` from the input tensor shape ``(batch, nstacks, n)``.

    Phase 9 D-66b: when consumers call ``butterfly_multiply``, the natural
    shape to consult at the resolver call site is the input — ``(batch,
    nstacks, n)``. Twiddle shape ``(nstacks, nblocks, log_n, n//2, 2, 2)``
    contains ``log_n`` directly as axis 2 but is less convenient at the
    resolver call site (and is typed differently in the dispatch graph).

    The last axis of the input is ``n`` (always a power of 2); return its
    integer log_2.
    """
    n = shape[-1]
    return int(math.log2(n))


def _should_route_to_cuda(
    op_name: str,
    shape: tuple,
    dtype: "torch.dtype",
    direction: str,
) -> bool:
    """Consult the static routing table — should this call route to CUDA?

    Phase 9 D-66a: returns True iff
        (a) ``_ROUTING_DISABLED`` is False (D-66c — test override), AND
        (b) the cell ``(op_name, log_n, dtype, direction)`` is present in
            ``_ROUTING_TABLE`` AND has ``route_to_cuda: true``.

    Emits a one-shot log.info per cell on the first True return (D-66d) so
    a user inspecting logs can see which shapes are being routed without
    log spam in tight inner loops.

    The dtype string is "fp32" for fp32, "complex64" for complex64, else
    ``str(dtype)``. The routing JSON only contains the first two; unknown
    dtypes fall through with no routing.
    """
    if _ROUTING_DISABLED:
        return False
    if not _ROUTING_TABLE:
        return False

    log_n = _shape_to_log_n(shape)
    if dtype == torch.float32:
        dtype_str = "fp32"
    elif dtype == torch.complex64:
        dtype_str = "complex64"
    else:
        dtype_str = str(dtype)

    key = f"{op_name}::{log_n}::{dtype_str}::{direction}"
    cell = _ROUTING_TABLE.get(key)
    if cell is None or not cell.get("route_to_cuda", False):
        return False

    # D-66d: emit first-route log once per (op, log_n, dtype, direction).
    log_key = (op_name, log_n, dtype_str, direction)
    if log_key not in _routing_log_emitted:
        _routing_log_emitted.add(log_key)
        log.info(
            "torch_structured: routing %s(log_n=%d, dtype=%s, direction=%s) "
            "to CUDA legacy backend (Triton/CUDA ratio %.2f > 1.67 per "
            "_routing.json)",
            op_name, log_n, dtype_str, direction,
            cell.get("triton_cuda_ratio_p50") or cell.get("triton_torch_ref_ratio_p50") or float("nan"),
        )
    return True


def set_routing_enabled(value: bool) -> bool:
    """Test-time override: enable/disable the runtime routing selector (D-66c).

    Returns the PREVIOUS value (save/restore pattern, mirrors ``set_backend``
    and ``set_deterministic``).

    Internal API — NOT exported in ``torch_structured/__init__.py``. The
    Phase 8 SC#4 reconciliation test (D-61a) uses this to ensure the
    monkey-patch shim is not bypassed by the routing closure.

    Example::

        prev = torch_structured._ops.set_routing_enabled(False)
        try:
            ...  # routing-bypass test body
        finally:
            torch_structured._ops.set_routing_enabled(prev)
    """
    global _ROUTING_DISABLED
    prev = not _ROUTING_DISABLED  # invert: prev was 'enabled?'
    _ROUTING_DISABLED = not bool(value)
    return prev


# ── Import-time resolution (DISP-03, DISP-05) ───────────────────────────
_initial = os.environ.get("TORCH_STRUCTURED_BACKEND", "auto")
_resolve(_initial)
log.info("torch_structured: backend=%s (import)", _BACKEND)
