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
import logging
import os

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


def _has_cuda_legacy() -> bool:
    """Return True iff the compiled C++ butterfly op is registered.

    The .so is loaded as a side effect of importing
    ``torch_structured.butterfly`` (see butterfly/__init__.py:22-39). This
    probe simply checks whether the registration succeeded.
    """
    return hasattr(torch.ops.torch_structured, "butterfly_multiply")


def _has_cuda_legacy_diag_mult() -> bool:
    """Per-op honest probe (CHECKER B3) for the legacy ``_diag_mult_cuda`` extension.

    Symmetric to ``_has_cuda_legacy()`` but checks the pybind11 ``_diag_mult_cuda``
    extension (D-22). Returns the ``HAS_CUDA_LEGACY_DIAG_MULT`` sentinel from
    ``_cuda_legacy/diag_mult.py`` — True iff the ``.so`` was built and the
    top-of-module try-import succeeded. Never raises; returns a clean bool.
    """
    try:
        from torch_structured._cuda_legacy.diag_mult import HAS_CUDA_LEGACY_DIAG_MULT
        return HAS_CUDA_LEGACY_DIAG_MULT
    except ImportError:
        return False


def _has_cuda_legacy_hadamard() -> bool:
    """Per-op honest probe (CHECKER B3) for the legacy ``_hadamard_cuda`` extension.

    Symmetric to ``_has_cuda_legacy_diag_mult()``; returns the ``HAS_CUDA_LEGACY_HADAMARD``
    sentinel from ``_cuda_legacy/hadamard.py``. Never raises; returns a clean bool.
    """
    try:
        from torch_structured._cuda_legacy.hadamard import HAS_CUDA_LEGACY_HADAMARD
        return HAS_CUDA_LEGACY_HADAMARD
    except ImportError:
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
            butterfly_multiply = _triton_bm
        elif _has_cuda_legacy():
            from torch_structured._cuda_legacy import butterfly_multiply as _cuda_bm
            butterfly_multiply = _cuda_bm
        else:
            from torch_structured._torch_ref.butterfly import butterfly_multiply_torch
            butterfly_multiply = butterfly_multiply_torch
    elif actual == "cuda":
        from torch_structured._cuda_legacy import butterfly_multiply as _cuda_bm
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
        from torch_structured._cuda_legacy.diag_mult import diag_mult as _cuda_dm
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
        from torch_structured._cuda_legacy.hadamard import hadamard_transform as _cuda_ht
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


# ── Import-time resolution (DISP-03, DISP-05) ───────────────────────────
_initial = os.environ.get("TORCH_STRUCTURED_BACKEND", "auto")
_resolve(_initial)
log.info("torch_structured: backend=%s (import)", _BACKEND)
