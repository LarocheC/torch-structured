"""Back-compat shim — Phase 6 (TRI-02).

Per D-33d, the pure-PyTorch reference ``hadamard_transform_torch`` has been
relocated to ``torch_structured._torch_ref.hadamard`` (the cross-cutting oracle
home). This module re-exports it so that the existing import surface in
``tests/structured/test_hadamard.py:8`` and ``tests/structured/test_imports.py:6-13``
continues to work without edits.

The legacy autograd Function class, the unnormalized-CUDA-wrapper, and the
module-level conditional binding were deleted per D-33 / D-33a / D-33b —
``torch_structured._ops.hadamard_transform`` now handles dispatch + autograd
plumbing via ``register_autograd`` (Task 3). The legacy try-import was also
deleted per D-33c — the new ``_cuda_legacy/hadamard.py`` (Task 2) owns the
honest-probe pattern.

This module additionally exposes a ``hadamard_transform`` callable for
back-compat (D-33d) that re-reads ``torch_structured._ops.hadamard_transform``
on every call (D-05 attribute access — rebind-safe across ``set_backend()``).
"""
import torch_structured  # noqa: F401 — needed so the shim below can attribute-access _ops

from torch_structured._torch_ref.hadamard import hadamard_transform_torch  # noqa: F401 — back-compat shim per D-33d


def hadamard_transform(*args, **kwargs):
    """Back-compat shim — delegates to ``torch_structured._ops.hadamard_transform``.

    Preserves the historical ``structured.hadamard.hadamard_transform`` import
    surface used by ``tests/structured/test_imports.py:6-13`` while honoring the
    D-05 attribute-access contract (rebind-safe across ``set_backend()`` — the
    binding is re-read on every call, so backend switches take effect
    transparently).
    """
    return torch_structured._ops.hadamard_transform(*args, **kwargs)
