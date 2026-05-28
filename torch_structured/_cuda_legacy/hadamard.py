"""Pass-through wrapper for the legacy ``_hadamard_cuda`` pybind11 extension.

Same asymmetry rationale as ``_cuda_legacy/diag_mult.py`` — the butterfly ``.so``
is loaded eagerly into ``torch.ops.torch_structured.*`` by
``butterfly/__init__.py:22-39`` (via ``torch.ops.load_library``), so the butterfly
passthrough is a thin ``torch.ops.torch_structured.butterfly_multiply(...)`` call.
The hadamard ``.so`` is a *pybind11 extension* imported by name
(``torch_structured._hadamard_cuda``) — it is not registered into the
``torch.ops`` namespace, so this module top-imports the extension under a
try-import to honestly report its absence via the ``HAS_CUDA_LEGACY_HADAMARD``
sentinel (D-21, D-22 honest probe).

Per CLAUDE.md §"Error Handling", core library code does not use try/except.
This is the documented exception: ``RuntimeError`` is the right exception type
for *environmental absence* (the ``.so`` was not built — distinct from a
precondition violation that ``assert`` would surface), and the resolver's
``_has_cuda_legacy_hadamard()`` probe is meant to prevent this module from
being bound at all when the ``.so`` is missing. The ``RuntimeError`` in
``hadamard_transform()`` is defensive — it should never fire in practice.
"""
from typing import Optional

import torch

try:
    from torch_structured import _hadamard_cuda as _hadamard_cuda_module
except (ImportError, RuntimeError):
    _hadamard_cuda_module = None  # type: ignore[assignment]

HAS_CUDA_LEGACY_HADAMARD: bool = _hadamard_cuda_module is not None


def hadamard_transform(u: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """Pass-through to the compiled pybind11 ``hadamard_transform`` op.

    Parameters:
        u: Tensor of shape (..., n) where n is a power of 2
        normalize: if True, divide the result by 2^{m/2} where m = log_2(n).
    Returns:
        product: Tensor of shape (..., n) — same shape as input

    Raises ``RuntimeError`` if the extension was not built — callers should
    probe ``HAS_CUDA_LEGACY_HADAMARD`` (or ``_ops._has_cuda_legacy_hadamard()``)
    first per the D-22 honest-fallback contract.
    """
    if _hadamard_cuda_module is None:
        raise RuntimeError(
            "_hadamard_cuda not built — caller should use "
            "_has_cuda_legacy_hadamard() probe (D-22)"
        )
    out = _hadamard_cuda_module.hadamard_transform(u)
    if normalize:
        n = u.shape[-1]
        m = n.bit_length() - 1
        assert n == 1 << m, 'n must be a power of 2'
        out = out / 2 ** (m / 2)
    return out
