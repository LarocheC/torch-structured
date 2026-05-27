"""Pass-through wrapper for the compiled C++ butterfly_multiply op.

The .so is already loaded by butterfly/__init__.py at package import time. This
wrapper exposes the registered op as a plain Python callable (no
``@torch.jit.script``) so it composes cleanly with ``torch.compile`` / Inductor.
TorchScript is deprecated as of PyTorch 2.10 and composes poorly with the
post-2.6 compile path; the dispatch wrapper in ``torch_structured/_ops.py`` may
invoke this callable from inside a compiled graph. Phase 10 may absorb this
into the deprecation-warning module per ``04-DEPRECATION-PLAN.md``.
"""
from typing import Optional

import torch


def butterfly_multiply(twiddle: torch.Tensor, input: torch.Tensor, increasing_stride: bool,
                       output_size: Optional[int] = None) -> torch.Tensor:
    """Pass-through to the compiled C++ op (already loaded by butterfly/__init__.py)."""
    return torch.ops.torch_structured.butterfly_multiply(twiddle, input, increasing_stride,
                                                          output_size)
