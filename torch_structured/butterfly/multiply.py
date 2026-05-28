"""§0 LANDMINE fix — Phase 9 09-01 / D-05 attribute-access delegators.

Before this rewrite, the top-level ``butterfly_multiply`` here was a
``@torch.jit.script``-decorated shim that called
``torch.ops.torch_structured.butterfly_multiply`` (the C++ op from
``csrc/butterfly.cpp``) DIRECTLY. ``torch_structured/butterfly/butterfly.py:8``
imports the symbol as ``from .multiply import butterfly_multiply`` and binds it
into its namespace at module-load time. Consequently, every
``Butterfly(...).forward(x)`` / ``ButterflyBmm(...).forward(x)`` call routed
through the C++ op regardless of ``set_backend('triton')`` — the nn.Module
surface silently bypassed the ``_ops`` dispatcher (the §0 LANDMINE recorded in
``.planning/phases/09/09-RESEARCH.md`` lines 108-141).

Resolution (Phase 9 D-05 / D-67 — inherits Phase 4 D-05; analog at
``torch_structured/structured/hadamard.py:25-34``): rewrite the three top-level
callables as plain-Python delegators. The ``butterfly_multiply`` delegator now
re-reads ``torch_structured._ops.butterfly_multiply`` on every call so
``set_backend()`` rebindings take effect through the existing import-time
binding in ``butterfly/butterfly.py``. ``butterfly_multiply_fw`` /
``butterfly_multiply_bw`` stay as direct callers of the C++ ``_fw``/``_bw``
ops (no Triton equivalent exists for those entry points) but lose the
``@torch.jit.script`` decorator so Phase 8 SC#4's monkey-patch
(``tests/test_butterfly_triton.py:756-757`) keeps working.
"""
import math
from typing import Tuple, Optional

import torch
from torch.nn import functional as F

import torch_structured  # noqa: F401 — needed so the delegator below can attribute-access _ops

# Phase 4: butterfly_multiply_torch now lives in _torch_ref/. Re-export here
# so existing test imports (torch_structured.butterfly.multiply.butterfly_multiply_torch)
# keep working unchanged.
from torch_structured._torch_ref.butterfly import butterfly_multiply_torch  # noqa: F401


def butterfly_multiply_fw(twiddle: torch.Tensor, input: torch.Tensor, increasing_stride: bool,
                          output_size: Optional[int] = None) -> torch.Tensor:
    """Direct C++ ``_fw`` entry point — no Triton equivalent at this surface.

    @torch.jit.script REMOVED so Phase 8 SC#4's monkey-patch
    (``tests/test_butterfly_triton.py:756-757``) can rebind this module attribute
    to a raising sentinel and verify that under BACKEND=triton the C++ ``_fw``
    op is NOT invoked. ScriptFunction objects are immutable and would defeat
    that mechanism.
    """
    return torch.ops.torch_structured.butterfly_multiply_fw(twiddle, input, increasing_stride,
                                                            output_size)


def butterfly_multiply_bw(twiddle: torch.Tensor, input: torch.Tensor, grad: torch.Tensor,
                          increasing_stride: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Direct C++ ``_bw`` entry point — no Triton equivalent at this surface.

    @torch.jit.script REMOVED — see ``butterfly_multiply_fw`` docstring above.
    """
    return torch.ops.torch_structured.butterfly_multiply_bw(twiddle, input, grad, increasing_stride)


# Phase 9 D-05/D-67 — Import-binding semantics note:
# torch_structured/butterfly/butterfly.py:8 does `from .multiply import butterfly_multiply`
# which binds the NAME `butterfly_multiply` in butterfly.py's module namespace to THIS
# delegator object at module-load time. The binding is to the OBJECT (this function),
# not to the attribute access path. Because the delegator BODY re-reads
# torch_structured._ops.butterfly_multiply on every call, subsequent set_backend()
# rebindings of _ops.butterfly_multiply take effect THROUGH this delegator without
# any modification to butterfly/butterfly.py. This is what makes the §0 LANDMINE fix
# work end-to-end: a single rewrite of multiply.py propagates to all four call sites
# in butterfly.py (lines 124, 128, 239, 243) without further edits.
def butterfly_multiply(*args, **kwargs):
    """Back-compat shim — delegates to ``torch_structured._ops.butterfly_multiply``.

    Preserves the historical ``butterfly.multiply.butterfly_multiply`` import
    surface (used by ``butterfly/butterfly.py:8`` and downstream user code)
    while honoring the D-05 attribute-access contract: the binding is re-read
    on every call so ``set_backend()`` switches take effect transparently
    through the existing import-time binding in ``butterfly.py``.
    """
    return torch_structured._ops.butterfly_multiply(*args, **kwargs)
