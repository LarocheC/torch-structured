"""Triton kernel + ``@triton_op`` wrapper for the cycle_mult / diag_mult primitive.

Implements the formula from ``csrc/diag_mult/diag_mult_cuda_kernel.cu:8``::

    out[pos] = subdiag[(pos + shift_subdiag + N) % N] * v[(pos + shift_v + N) % N]

over a 2-D Triton grid ``(n_batch, cdiv(N, BLOCK_SIZE))``. Real / complex64 are
both handled via a single kernel source gated on ``IS_COMPLEX: tl.constexpr``
(per ``04-COMPLEX-LAYOUT.md`` — the canonical wrapper + 4-FMA template).

The wrapper boundary follows ``04-COMPLEX-LAYOUT.md``: complex64 inputs are
reinterpreted via ``torch.view_as_real`` (asserting contiguity per Pitfall 3),
the kernel sees trailing-2 real layout, and the result is returned via
``torch.view_as_complex``.

Backward (``register_autograd``) is the Wirtinger-correct ``.conj()`` formula
derived and numerically verified in ``05-RESEARCH.md`` §"Backward Gradient
Formula"; ``.conj()`` is a no-op for real so a single code path handles both
real and complex gradcheck (D-26).
"""
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from torch_structured._torch_ref.diag_mult import diag_mult as _diag_mult_torch  # backward oracle (D-26)


@triton.jit
def _cycle_mult_kernel(
    subdiag_ptr,
    v_ptr,
    out_ptr,
    n_batch,
    N,
    subdiag_batch_stride,
    shift_subdiag,
    shift_v,
    IS_COMPLEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Cycle_mult Triton kernel (2-D grid).

    Maps each ``(bid, pid)`` to a tile of ``BLOCK_SIZE`` elements in row ``bid``
    of ``v`` and computes ``out[bid, pos] = subdiag[bid, sub_idx] * v[bid, v_idx]``
    where ``sub_idx = (pos + shift_subdiag + N) % N`` and ``v_idx = (pos + shift_v + N) % N``.

    For broadcast subdiag (1-D), ``subdiag_batch_stride == 0`` so all batch rows
    share the same subdiag slice. For batched subdiag (matching v.numel),
    ``subdiag_batch_stride == N`` (real) or ``2 * N`` (complex, trailing-2 real view).

    For complex inputs (IS_COMPLEX == True), the pointers are into the view_as_real
    trailing-2 layout: each element occupies 2 consecutive floats (re, im) so
    every byte offset is doubled and the inner branch applies the 4-FMA complex
    multiply ``(a + bi)(c + di) = (ac - bd) + (ad + bc)i``.
    """
    bid = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    pos = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pos < N
    sub_idx = (pos + shift_subdiag + N) % N
    v_idx = (pos + shift_v + N) % N

    if IS_COMPLEX:
        # view_as_real layout: each logical element = 2 consecutive floats.
        # Row stride for v/out is 2*N; subdiag uses the supplied stride.
        v_row_base = bid * N * 2
        sub_row_base = bid * subdiag_batch_stride

        v_off_re = v_row_base + v_idx * 2
        v_off_im = v_off_re + 1
        sub_off_re = sub_row_base + sub_idx * 2
        sub_off_im = sub_off_re + 1
        out_off_re = v_row_base + pos * 2
        out_off_im = out_off_re + 1

        a_re = tl.load(subdiag_ptr + sub_off_re, mask=mask)
        a_im = tl.load(subdiag_ptr + sub_off_im, mask=mask)
        c_re = tl.load(v_ptr + v_off_re, mask=mask)
        c_im = tl.load(v_ptr + v_off_im, mask=mask)

        # (a + bi)(c + di) = (ac - bd) + (ad + bc)i  — 4-FMA per 04-COMPLEX-LAYOUT.md
        out_re = a_re * c_re - a_im * c_im
        out_im = a_re * c_im + a_im * c_re

        tl.store(out_ptr + out_off_re, out_re, mask=mask)
        tl.store(out_ptr + out_off_im, out_im, mask=mask)
    else:
        v_row_base = bid * N
        sub_row_base = bid * subdiag_batch_stride

        a = tl.load(subdiag_ptr + sub_row_base + sub_idx, mask=mask)
        c = tl.load(v_ptr + v_row_base + v_idx, mask=mask)
        tl.store(out_ptr + v_row_base + pos, a * c, mask=mask)


@triton_op("torch_structured::diag_mult", mutates_args={})
def diag_mult(subdiag: torch.Tensor, v: torch.Tensor, shift_subdiag: int,
              shift_v: int) -> torch.Tensor:
    """Triton-backed cycle_mult (D-19).

    Parameters:
        subdiag: 1-D tensor of shape ``(N,)`` (broadcasts over batch) or full
            shape ``(*batch, N)`` matching ``v.numel()`` (batched subdiag —
            mirrors the csrc ``batchedSubdiag = subdiag.numel() == v.numel()``
            flag).
        v: tensor of shape ``(*batch, N)`` — real (float32) or complex64.
        shift_subdiag: integer cyclic shift on subdiag (last dim).
        shift_v: integer cyclic shift on v (last dim).

    Returns:
        Tensor of shape ``v.shape`` and same dtype as v.
    """
    # D-20 mixed-dtype reject (Pitfall 5: tl.load otherwise reads garbage).
    assert subdiag.dtype == v.dtype, (
        f"subdiag.dtype ({subdiag.dtype}) must equal v.dtype ({v.dtype})"
    )
    # Pitfall 3: contiguity needed before view_as_real and for kernel pointer math.
    assert v.is_contiguous(), "v must be contiguous before view_as_real (Pitfall 3)"
    assert subdiag.is_contiguous(), \
        "subdiag must be contiguous before view_as_real (Pitfall 3)"
    # D-19b trailing-dim agreement (prevents OOB index arithmetic in kernel).
    assert subdiag.size(-1) == v.size(-1), (
        f"subdiag.size(-1) ({subdiag.size(-1)}) must equal v.size(-1) ({v.size(-1)})"
    )

    N = v.size(-1)
    n_batch = v.numel() // N
    is_batched_subdiag = (subdiag.numel() == v.numel())
    is_complex = v.is_complex()

    if is_complex:
        v_work = torch.view_as_real(v)
        subdiag_work = torch.view_as_real(subdiag)
        subdiag_batch_stride = 2 * N if is_batched_subdiag else 0
    else:
        v_work = v
        subdiag_work = subdiag
        subdiag_batch_stride = N if is_batched_subdiag else 0

    out_work = torch.empty_like(v_work)
    BLOCK_SIZE = 1024
    # wrap_triton in PyTorch 2.6+ accepts only plain @triton.jit (no
    # @triton.heuristics); BLOCK_SIZE is a fixed power-of-2 — the kernel is
    # pointwise and not block-size sensitive.
    grid = lambda meta: (n_batch, triton.cdiv(N, meta["BLOCK_SIZE"]))
    wrap_triton(_cycle_mult_kernel)[grid](
        subdiag_work,
        v_work,
        out_work,
        n_batch,
        N,
        subdiag_batch_stride,
        shift_subdiag,
        shift_v,
        IS_COMPLEX=is_complex,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    if is_complex:
        return torch.view_as_complex(out_work.contiguous())
    return out_work


def _setup_context(ctx, inputs, output):
    """Save subdiag/v + shifts for the Wirtinger backward (D-26)."""
    subdiag, v, shift_subdiag, shift_v = inputs
    ctx.save_for_backward(subdiag, v)
    ctx.shift_subdiag = shift_subdiag
    ctx.shift_v = shift_v


def _backward(ctx, grad_out):
    """Wirtinger-correct backward — formula derived + numerically verified in 05-RESEARCH.md.

    ``.conj()`` on the OTHER operand is the load-bearing Wirtinger correction.
    For real inputs ``.conj()`` is a no-op so this single code path covers both
    real and complex gradcheck. Without ``.conj()`` complex64 gradcheck fails
    with errors ~2.0 (verified in RESEARCH lines 404-516).
    """
    subdiag, v = ctx.saved_tensors
    s_sub, s_v = ctx.shift_subdiag, ctx.shift_v
    grad_subdiag = _diag_mult_torch(grad_out, v.conj(), -s_sub, s_v - s_sub)
    grad_v = _diag_mult_torch(subdiag.conj(), grad_out, s_sub - s_v, -s_v)
    # Pitfall 6: 1-D subdiag broadcast → sum gradient over the broadcast (leading) dims.
    if subdiag.shape != grad_subdiag.shape:
        ndims_to_sum = grad_subdiag.dim() - subdiag.dim()
        if ndims_to_sum > 0:
            grad_subdiag = grad_subdiag.sum(dim=tuple(range(ndims_to_sum)))
    return grad_subdiag, grad_v, None, None


diag_mult.register_autograd(_backward, setup_context=_setup_context)


@diag_mult.register_fake
def _diag_mult_fake(subdiag, v, shift_subdiag, shift_v):
    """Meta kernel — Phase 4 D-12 mandate (the literal 260419-p27 fix).

    Without this, ``torch.compile``'s dynamo fake-tensor trace cannot infer
    the output shape/dtype/device of ``diag_mult`` and raises:
    ``"The tensor has a non-zero number of elements, but its data is not
    allocated yet"``.
    """
    return torch.empty_like(v)
