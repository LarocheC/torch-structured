"""Triton kernel + ``@triton_op`` wrapper for the Walsh-Hadamard transform (D-31).

Implements the single-pass shared-memory fast Walsh-Hadamard transform analogous
to the CUDA reference at ``csrc/hadamard/hadamard_cuda_kernel.cu:24-86``
(``fwtBatch1Kernel``): for each ``stride`` in ``(1, 2, 4, ..., N/2)``, pairs at
indices ``(i, i + stride)`` are replaced by
``(x[i] + x[i + stride], x[i] - x[i + stride])``. The stage loop is unrolled at
JIT time via ``tl.static_range(LOG_N)``.

Phase 6 ships the single-pass kernel covering ``log_n <= 12`` (ROADMAP SC#1).
The mixed-radix two-pass ``fwtBatch2Kernel`` (CUDA lines 88-153) for larger
``N`` is **explicitly deferred** per D-31c — the wrapper asserts
``log_n <= 12`` at the boundary.

Backward (``register_autograd``) is the **self-inverse** identity per D-32:
``d(H @ u) / du = H`` so ``grad_u = H @ grad_out``. The backward routes through
``_torch_ref.hadamard.hadamard_transform_torch`` for deterministic gradients
independent of the forward backend and for fp64 gradcheck precision.

No complex64 support — Phase 6 is real-only per ROADMAP "no complex" and D-31.
No ``view_as_real`` games; ``04-COMPLEX-LAYOUT.md`` does NOT apply.

Kernel-body decision (per D-31a — planner's call): we use the **out_ptr-as-scratch**
shuffle pattern. The first ``tl.store`` seeds ``out_ptr`` from ``u_ptr`` so that
every stage of the unrolled loop reads from ``out_ptr`` uniformly. At each
stage, we load both the "self" element at ``offsets`` and the partner element
at ``offsets ^ stride``, apply ``tl.where(lower_mask, cur + partner,
partner - cur)`` per the CUDA butterfly formula, and write back. Within a
single Triton program the global writes provide the visibility primitive for
the next-stage loads (no cross-block sync needed because the entire row lives
inside one program in a 1-D grid). Correctness is gated by Test 1 (cross-check
vs the ``_torch_ref`` oracle) and Test 4 (self-inverse ``H(H(u)) == N * u``).
"""
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from torch_structured._torch_ref.hadamard import hadamard_transform_torch as _hadamard_transform_torch  # backward oracle (D-32, self-inverse)


@triton.jit
def _hadamard_kernel(
    u_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,  # == N (power-of-2; SC#1 caps N at 4096)
    LOG_N: tl.constexpr,
):
    """Single-pass shared-memory Walsh-Hadamard kernel (D-31, D-31a).

    1-D grid ``(n_batch,)``; each program transforms one row of ``N`` elements.
    Seed ``out_ptr`` from ``u_ptr`` at start, then run ``LOG_N`` butterfly
    stages with ``out_ptr`` as inter-stage scratch.

    At stage ``k`` with ``stride = 1 << k``:
      * ``partner_pos = offsets ^ stride`` — XOR pairs lower (``pos & stride == 0``)
        and upper (``pos & stride != 0``) halves at the current stride.
      * ``lower_mask = (offsets & stride) == 0`` — True on the "lower" element
        of each pair.
      * Lower elements receive ``cur + partner``; upper elements receive
        ``partner - cur`` (so the lower-half value is the addend, the
        upper-half value is the subtrahend). Combined this implements the
        ``(D0 + D1, D0 - D1)`` butterfly per
        ``csrc/hadamard/hadamard_cuda_kernel.cu:50-53``.
    """
    bid = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    row_base = bid * N

    # Seed out_ptr from u_ptr so every stage reads from out_ptr uniformly.
    x = tl.load(u_ptr + row_base + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + row_base + offsets, x, mask=mask)
    # Barrier: ensure the seed write is visible to all threads in this program
    # before the first stage's partner-load.
    tl.debug_barrier()

    for k in tl.static_range(LOG_N):
        stride = 1 << k
        cur = tl.load(out_ptr + row_base + offsets, mask=mask, other=0.0)
        partner_pos = offsets ^ stride
        partner = tl.load(out_ptr + row_base + partner_pos, mask=mask, other=0.0)
        lower_mask = (offsets & stride) == 0
        new_x = tl.where(lower_mask, cur + partner, partner - cur)
        # Barrier BEFORE the store so all threads have read cur/partner from
        # the previous-stage state before any thread overwrites it.
        tl.debug_barrier()
        tl.store(out_ptr + row_base + offsets, new_x, mask=mask)
        # Barrier AFTER the store so the next-stage tl.load sees this stage's
        # writes consistently across all threads.
        tl.debug_barrier()


@triton_op("torch_structured::hadamard_transform", mutates_args={})
def hadamard_transform(u: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """Triton-backed Walsh-Hadamard transform (TRI-02 / D-31).

    Computes ``H_n @ u`` over the last dimension, where ``H_n`` is the
    ``n x n`` Walsh-Hadamard matrix and ``n`` is a power of 2. Single-pass
    shared-memory kernel; ``log_n`` capped at 12 per D-31c (SC#1).

    Parameters:
        u: Tensor of shape ``(*batch, n)`` — must be fp32, contiguous, last
            dim a power of 2 with ``log_n <= 12``.
        normalize: if True, divide the result by ``2 ** (log_n / 2)`` —
            verbatim from ``structured/hadamard.py:58``, NOT rewritten as
            ``math.sqrt(n)`` (preserves numerical parity, D-35a).

    Returns:
        Tensor of shape ``u.shape`` with the same dtype as ``u``.
    """
    # Wrapper-boundary preconditions per CLAUDE.md §"Error Handling" + Pitfall 3 analog.
    assert u.dim() >= 1, f"u must have rank >= 1, got dim={u.dim()}"
    assert u.dtype == torch.float32, (
        f"hadamard kernel is fp32-only (D-31), got {u.dtype}"
    )
    assert u.is_contiguous(), "u must be contiguous (Pitfall 3 analog)"
    n = u.size(-1)
    log_n = int(n.bit_length() - 1)
    assert n == 1 << log_n, f"n must be a power of 2, got {n}"
    # D-31c: single-pass kernel caps at log_n=12 (SC#1 — n <= 4096).
    assert log_n <= 12, (
        f"single-pass kernel caps log_n at 12 per D-31c (SC#1), got log_n={log_n}"
    )

    n_batch = u.numel() // n
    out = torch.empty_like(u)
    # D-31b: fixed num_warps defaults; autotune deferred to Phase 9.
    num_warps = 4 if log_n <= 8 else 8

    grid = (n_batch,)
    wrap_triton(_hadamard_kernel)[grid](
        u,
        out,
        n,
        BLOCK_SIZE=n,
        LOG_N=log_n,
        num_warps=num_warps,
    )

    # D-35 / D-35a: wrapper-side normalization. The kernel is unnormalized;
    # apply the 2 ** (log_n / 2) divisor verbatim from structured/hadamard.py:58.
    # Do NOT rewrite as math.sqrt(n) — preserves numerical parity.
    return out / (2 ** (log_n / 2)) if normalize else out


def _setup_context(ctx, inputs, output):
    """Save the ``normalize`` flag for the self-inverse backward (D-32b).

    No ``save_for_backward(u)`` — the Hadamard transform is self-inverse so
    backward == forward applied to grad; the saved input is unused. Only the
    ``normalize`` flag is needed for chain-rule scaling.
    """
    u, normalize = inputs
    ctx.normalize = normalize


def _backward(ctx, grad_out):
    """Self-inverse backward (D-32, D-32a, D-32b).

    Hadamard is self-inverse: ``d(H @ u) / du = H`` so ``grad_u = H @ grad_out``.
    Route through the ``_torch_ref`` oracle per D-32 — deterministic backward
    and fp64 gradcheck precision, independent of the forward backend. No
    ``.conj()`` (real-only kernel per ROADMAP "no complex"). No broadcast-sum
    (single tensor input).
    """
    grad_u = _hadamard_transform_torch(grad_out, normalize=ctx.normalize)
    return grad_u, None  # 2 returns matching 2 forward inputs (u, normalize)


hadamard_transform.register_autograd(_backward, setup_context=_setup_context)


@hadamard_transform.register_fake
def _hadamard_transform_fake(u, normalize=False):
    """Meta kernel — Phase 4 D-12 mandate (the literal 260419-p27 fix).

    Without ``register_fake``, ``torch.compile``'s dynamo fake-tensor trace
    cannot infer the output shape/dtype/device of ``hadamard_transform`` and
    raises ``"The tensor has a non-zero number of elements, but its data is
    not allocated yet"``.

    Note the ``normalize=False`` default mirrors the wrapper's schema default —
    PyTorch's dispatch elides default-valued scalar args before calling the
    fake impl, so the default here is **load-bearing** (without it, FakeTensorMode
    with the default ``normalize=False`` fails with ``TypeError: missing
    positional argument 'normalize'``).
    """
    return torch.empty_like(u)
