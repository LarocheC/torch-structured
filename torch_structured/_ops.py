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


def _has_triton_kernel(op_name: str) -> bool:
    """Per-op probe — True only when a real Triton kernel ships for ``op_name``.

    Probes ``torch_structured._triton.<op_name>.op`` and checks that
    ``<op_name>`` is defined on it. In Phase 4 the ``_triton/`` package is
    empty (Plan 04-02 Task 1 creates the placeholder package; no submodules),
    so this returns False for every op. Phase 5 lights up
    ``_has_triton_kernel("diag_mult")`` when the first real kernel ships.

    Distinguishing this from ``_has_triton()`` is per CHECKER B3: the resolver
    must be honest about backend availability so it never silently lies about
    binding to Triton.
    """
    try:
        mod = importlib.import_module(f"torch_structured._triton.{op_name}.op")
    except (ImportError, AttributeError):
        return False
    return hasattr(mod, op_name)


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
    # Honest per-op probe (CHECKER B3): in Phase 4 the _triton package is empty,
    # so _has_triton_kernel(*) is always False and `auto`/`triton` never resolve
    # to "triton". Phase 5 lights up the first real Triton kernel.
    if name == "auto":
        if _has_triton_kernel("butterfly_multiply") and torch.cuda.is_available():
            actual = "triton"
        elif _has_cuda_legacy():
            actual = "cuda"
        else:
            actual = "torch"
    elif name == "triton":
        if _has_triton_kernel("butterfly_multiply") and torch.cuda.is_available():
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
        # butterfly_multiply` here.
        from torch_structured._triton.butterfly.op import (  # type: ignore[import-not-found]
            butterfly_multiply as _triton_bm,
        )
        butterfly_multiply = _triton_bm
    elif actual == "cuda":
        from torch_structured._cuda_legacy import butterfly_multiply as _cuda_bm
        butterfly_multiply = _cuda_bm
    else:  # actual == "torch"
        from torch_structured._torch_ref.butterfly import butterfly_multiply_torch
        butterfly_multiply = butterfly_multiply_torch

    # hadamard_transform / diag_mult: Phase 6 / Phase 5 populate; stay None for now.

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


# ─── Phase 4 demonstrator op (deleted at start of Phase 5 per D-13) ──────
#
# Demonstrator op (D-13, D-14): proves the triton_op + register_autograd +
# register_fake pipeline survives torch.compile(fullgraph=True) and gradcheck.
# Deleted at the start of Phase 5 once diag_mult exercises the same pattern on
# a real kernel. The register_fake decorator below is the literal 260419-p27
# fix — DO NOT remove (RESEARCH.md Pitfall 1).


@triton.jit
def _demo_identity_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Identity copy kernel — no-op proof-of-concept for the wrapper pattern."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


@triton_op("torch_structured::_demo_identity", mutates_args={})
def _demo_identity_op(x: torch.Tensor) -> torch.Tensor:
    """No-op identity demonstrator: exercises the triton_op + wrap_triton path.

    Handles complex64 via the view_as_real/view_as_complex wrapper boundary so
    that Phase 7's critical path (TRI-06, D-01) is proven sound before Phase 7
    starts. See 04-COMPLEX-LAYOUT.md for the canonical pattern.

    Parameters:
        x: input tensor; dtype may be a real float type or torch.complex64.

    Returns:
        Element-wise copy of ``x`` with the same dtype, shape, and device.
    """
    is_complex = x.is_complex()
    if is_complex:
        # Pitfall 3 (04-RESEARCH.md): view_as_real on a non-contiguous complex
        # tensor inherits the transposed/strided pattern; the kernel would
        # read garbage. Force contiguity at the wrapper boundary.
        assert x.is_contiguous(), (
            "complex input must be contiguous before view_as_real "
            "(Pitfall 3 / 04-COMPLEX-LAYOUT.md)"
        )
        x_work = torch.view_as_real(x)  # zero-copy trailing-2 real view
    else:
        x_work = x
    out_work = torch.empty_like(x_work)
    n_elements = x_work.numel()
    # BLOCK_SIZE is a constexpr passed at the call site (no @triton.heuristics:
    # wrap_triton rejects Heuristics-wrapped kernels in PyTorch 2.6+, only
    # plain @triton.jit or @triton.autotune are accepted). We pick a fixed
    # power-of-2 BLOCK_SIZE — the demonstrator is a single-pass identity copy,
    # autotuning is not needed.
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    wrap_triton(_demo_identity_kernel)[grid](x_work, out_work, n_elements, BLOCK_SIZE)
    if is_complex:
        # view_as_complex requires the last dim to be size 2 with stride 1 and
        # all other strides to be even — `out_work = empty_like(x_work)` is
        # already contiguous, so the .contiguous() call is a no-op here but
        # documents the invariant for future kernel changes.
        return torch.view_as_complex(out_work.contiguous())
    return out_work


def _setup_context(ctx, inputs, output):
    """Identity op needs nothing saved for backward."""
    # (x,) = inputs  # noqa: F841 — kept here for shape reference
    pass


def _backward(ctx, grad):
    """Identity op gradient is the input gradient pass-through (d(out)/d(in) = 1)."""
    return grad


_demo_identity_op.register_autograd(_backward, setup_context=_setup_context)


@_demo_identity_op.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    """Meta kernel — THE 260419-p27 fix.

    Without this, ``torch.compile``'s dynamo fake-tensor trace cannot infer
    the output shape/dtype/device of ``_demo_identity_op`` and raises:
    ``"The tensor has a non-zero number of elements, but its data is not
    allocated yet"``. With this, dynamo traces through cleanly.
    """
    return torch.empty_like(x)
