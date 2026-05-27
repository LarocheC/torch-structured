import math
from typing import Tuple, Optional

import torch
from torch.nn import functional as F

# Phase 4: butterfly_multiply_torch now lives in _torch_ref/. Re-export here
# so existing test imports (torch_structured.butterfly.multiply.butterfly_multiply_torch)
# keep working unchanged.
from torch_structured._torch_ref.butterfly import butterfly_multiply_torch  # noqa: F401


@torch.jit.script
def butterfly_multiply_fw(twiddle: torch.Tensor, input: torch.Tensor, increasing_stride: bool,
                          output_size: Optional[int] = None) -> torch.Tensor:
    return torch.ops.torch_structured.butterfly_multiply_fw(twiddle, input, increasing_stride,
                                                            output_size)


@torch.jit.script
def butterfly_multiply_bw(twiddle: torch.Tensor, input: torch.Tensor, grad: torch.Tensor,
                          increasing_stride: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_structured.butterfly_multiply_bw(twiddle, input, grad, increasing_stride)


@torch.jit.script
def butterfly_multiply(twiddle: torch.Tensor, input: torch.Tensor, increasing_stride: bool,
                       output_size: Optional[int] = None) -> torch.Tensor:
    return torch.ops.torch_structured.butterfly_multiply(twiddle, input, increasing_stride,
                                                          output_size)
