"""Pure-PyTorch reference implementation of the cycle_mult primitive (D-19, D-26).

This is the pure-PyTorch oracle for the generic ``cycle_mult`` primitive whose
C++/CUDA implementation lives at ``csrc/diag_mult/diag_mult_cuda_kernel.cu:8``:

    out[pos] = subdiag[(pos + shift_subdiag + N) % N] * v[(pos + shift_v + N) % N]

The closed-form PyTorch equivalent uses ``torch.roll`` (sign verified in
``05-RESEARCH.md`` §"torch.roll Direction Sanity Check"):

    out = torch.roll(subdiag, -shift_subdiag, dims=-1) * torch.roll(v, -shift_v, dims=-1)

Three roles per the Phase 5 plan:

1. **Gradcheck oracle.** ``torch.autograd.gradcheck`` runs against this fp64-capable
   pure-PyTorch implementation; the Triton kernel's ``register_autograd`` backward
   callback also invokes this function (per D-26).
2. **Runtime fallback** for the D-22 asymmetric path: when
   ``TORCH_STRUCTURED_BACKEND=cuda`` is requested but ``_diag_mult_cuda.so`` is not
   built, the resolver binds ``_ops.diag_mult`` to this function and emits a
   ``log.warning``.
3. **Backend = ``torch``** path: this is the implementation used when
   ``set_backend("torch")`` is in effect (no GPU required).

Per D-20c, ``torch.roll`` operates natively on complex tensors — no
``view_as_real`` games are needed at this layer (those live in the Triton
wrapper at ``torch_structured/_triton/diag_mult/op.py``).
"""
import torch


def diag_mult(subdiag: torch.Tensor, v: torch.Tensor, shift_subdiag: int,
              shift_v: int) -> torch.Tensor:
    """Pure-PyTorch cycle_mult: pointwise multiply of cyclically-shifted subdiag and v.

    Parameters:
        subdiag: 1-D tensor of shape ``(N,)`` (broadcasts over batch) or full
            shape ``(*batch, N)`` (matches v elementwise, like the C++
            ``batchedSubdiag`` path).
        v: tensor of shape ``(*batch, N)``.
        shift_subdiag: integer cyclic shift applied to subdiag along the last dim.
        shift_v: integer cyclic shift applied to v along the last dim.

    Returns:
        Tensor of shape ``v.shape`` with the same dtype as v.
    """
    # D-20 mixed-dtype reject: assert per CLAUDE.md §"Error Handling" convention.
    assert subdiag.dtype == v.dtype, (
        f"subdiag.dtype ({subdiag.dtype}) must equal v.dtype ({v.dtype})"
    )
    # D-19b trailing-dim agreement.
    assert subdiag.size(-1) == v.size(-1), (
        f"subdiag.size(-1) ({subdiag.size(-1)}) must equal v.size(-1) ({v.size(-1)})"
    )
    return torch.roll(subdiag, -shift_subdiag, dims=-1) * torch.roll(v, -shift_v, dims=-1)
