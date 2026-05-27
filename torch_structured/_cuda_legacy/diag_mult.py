"""Pass-through wrapper for the legacy ``_diag_mult_cuda`` pybind11 extension.

Asymmetry vs ``_cuda_legacy/butterfly.py``: the butterfly ``.so`` is loaded
eagerly into ``torch.ops.torch_structured.*`` by ``butterfly/__init__.py:22-39``
(via ``torch.ops.load_library``), so the butterfly passthrough is a thin
``torch.ops.torch_structured.butterfly_multiply(...)`` call. The diag_mult
``.so`` is a *pybind11 extension* imported by name (``torch_structured._diag_mult_cuda``)
— it is not registered into the ``torch.ops`` namespace, so this module
top-imports the extension under a try-import to honestly report its absence via
the ``HAS_CUDA_LEGACY_DIAG_MULT`` sentinel (D-21, D-22 honest probe).

Per CLAUDE.md §"Error Handling", core library code does not use try/except.
This is the documented exception: ``RuntimeError`` is the right exception type
for *environmental absence* (the ``.so`` was not built — distinct from a
precondition violation that ``assert`` would surface), and the resolver's
``_has_cuda_legacy_diag_mult()`` probe is meant to prevent this module from
being bound at all when the ``.so`` is missing. The ``RuntimeError`` in
``diag_mult()`` is defensive — it should never fire in practice.
"""
from typing import Optional

import torch

try:
    from torch_structured import _diag_mult_cuda as _diag_mult_cuda_module
except (ImportError, RuntimeError):
    _diag_mult_cuda_module = None  # type: ignore[assignment]

HAS_CUDA_LEGACY_DIAG_MULT: bool = _diag_mult_cuda_module is not None


def diag_mult(subdiag: torch.Tensor, v: torch.Tensor, shift_subdiag: int,
              shift_v: int) -> torch.Tensor:
    """Pass-through to the compiled pybind11 ``cycle_mult`` op.

    Raises ``RuntimeError`` if the extension was not built — callers should
    probe ``HAS_CUDA_LEGACY_DIAG_MULT`` (or ``_ops._has_cuda_legacy_diag_mult()``)
    first per the D-22 honest-fallback contract.
    """
    if _diag_mult_cuda_module is None:
        raise RuntimeError(
            "_diag_mult_cuda not built — caller should use "
            "_has_cuda_legacy_diag_mult() probe (D-22)"
        )
    return _diag_mult_cuda_module.cycle_mult(subdiag, v, shift_subdiag, shift_v)
