"""Triton kernel + ``@triton_op`` wrapper for butterfly_multiply forward (D-40, TRI-03).

Implements the formula from ``torch_structured/_torch_ref/butterfly.py:23-32``::

    for block in range(nblocks):
        for idx in range(log_n):
            log_stride = idx if cur_increasing_stride else log_n - 1 - idx
            stride = 1 << log_stride
            # in-register 2x2 butterfly multiply, partner = pos ^ stride
        cur_increasing_stride = not cur_increasing_stride

using a **multi-launch 3-stage register-resident tile** structure (D-40, D-40a,
D-40b, D-40c, D-40d). For ``log_n=L``, the wrapper issues ``ceil(L / 3)``
``@triton.jit`` launches per nblock. Each launch handles up to 3 consecutive
butterfly stages on a register-resident tile of width ``TILE_N = 1 << (max(stages_in_group) + 1)``;
no shared memory is needed because all intra-launch state stays in registers
(no inter-stage ``tl.store`` / ``tl.load`` round-trip — contrast with Phase 6
hadamard which used the out_ptr-as-scratch shuffle pattern).

The wrapper drives a Python-side ``nblocks`` loop with ``cur_increasing_stride``
toggling (mirrors the verbatim oracle at ``_torch_ref/butterfly.py:22-32``
verbatim) and ping-pongs between two output buffers to avoid in-place data
dependencies between stage-group launches.

Plan 07-01 ships the kernel signature with ``IS_COMPLEX: tl.constexpr`` already
present and the wrapper ``view_as_real`` machinery in place, but the path is
gated by:

* Kernel-side: a ``tl.static_assert`` on ``IS_COMPLEX`` at function entry
  rejects ``IS_COMPLEX=True`` JIT specializations (D-41a load-bearing
  pre-wiring).
* Wrapper-side: a precondition assert rejecting any dtype other than
  fp32 at the boundary.

Plan 07-02 removes **only these two gates** — zero kernel-signature refactor
between plans. The 4-FMA complex multiply will be added inside the
``if IS_COMPLEX:`` branch of the kernel body per
``.planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/04-COMPLEX-LAYOUT.md``
lines 58-76 verbatim.

Small-N fallback (D-42a): when ``log_n <= 1`` (n in {1, 2}), the wrapper bypasses
the kernel entirely and delegates to ``_torch_ref.butterfly.butterfly_multiply_torch``
inside the ``triton_op`` so the autograd graph stays uniform across the
small-N / large-N split. Triton launch overhead would dominate at n in {1, 2}
and the smallest 3-stage tile (TILE_N=8) is wider than n.

Pad/trim (D-42): the wrapper does ``input = F.pad(input, (0, n - input_size))``
when ``input_size < n`` and ``output[:, :, :output_size]`` on return — mirrors
the oracle's lines 18, 33 verbatim.

Backward (``register_autograd``, D-47): the backward callback computes
``(grad_twiddle, grad_input)`` via
``torch.autograd.grad(_butterfly_multiply_torch(twiddle_d, input_d, ...), [twiddle_d, input_d], grad_out)``
where ``twiddle_d``/``input_d`` are detached + ``requires_grad_(True)`` clones
of the saved tensors. This is the **two-input** variant of Phase 5's Wirtinger
pattern (Phase 5 used direct gradient formulas; Phase 7 delegates the entire
gradient computation to the oracle via ``torch.autograd.grad``). Returns a
4-tuple matching the 4 forward inputs ``(twiddle, input, increasing_stride, output_size)``;
the last two are ``None`` because they are non-tensor (bool, Optional[int]).
"""
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from torch.nn import functional as F
from typing import Optional

from torch_structured._torch_ref.butterfly import butterfly_multiply_torch as _butterfly_multiply_torch  # backward oracle (D-47, two-input via torch.autograd.grad) + small-N fallback (D-42a)


def _pick_num_warps(tile_n: int) -> int:
    """D-40d: fixed num_warps schedule by tile_n. Phase 9 may revisit."""
    if tile_n <= 64:
        return 4
    if tile_n <= 1024:
        return 8
    return 16


@triton.jit
def _butterfly_kernel(
    twiddle_ptr,
    input_ptr,
    output_ptr,
    n,                          # full-N dimension
    nstacks,                    # for twiddle row stride
    block_idx,                  # current nblock index (Python-side loop)
    nblocks,                    # for twiddle stride math
    STAGE_START: tl.constexpr,  # first stage in this group (D-40a)
    STAGE_COUNT: tl.constexpr,  # number of stages this launch (1..3) (D-40a)
    INCREASING_STRIDE: tl.constexpr,  # propagates cur_increasing_stride
    LOG_N: tl.constexpr,        # full log_n (for stride direction calc)
    IS_COMPLEX: tl.constexpr,   # Phase 4 layout flag (D-44 / D-41a) — gated off in 07-01
    TILE_N: tl.constexpr,       # per-launch tile width = 1 << (max_stage + 1) (D-40b)
):
    """3-stage out_ptr-as-scratch butterfly tile (D-40, D-40b, Phase 6 sync pattern).

    Each program handles ``TILE_N`` consecutive elements of one
    ``(batch, nstack)`` row. Loads the input tile once into registers
    (``tl.load``), runs ``STAGE_COUNT`` (1..3) butterfly stages using
    ``output_ptr`` as inter-stage scratch with ``tl.debug_barrier`` sync
    (Phase 6 hadamard pattern), and stores the final result at the end.

    Note: the original plan envisioned a pure register-resident tile with no
    inter-stage stores, but Triton has no in-register XOR-gather primitive
    for arbitrary partner indices — the cleanest correct path is the
    out_ptr-as-scratch shuffle that Phase 6 validated (cf. 06-01-SUMMARY.md
    "Auto-fixed Issues #1"). Same thread-sync rationale: within one Triton
    program, ``tl.store`` + ``tl.load`` to the same offsets is NOT implicitly
    synchronized across threads in the program; ``tl.debug_barrier`` provides
    the visibility ordering.

    Per-stage math (verbatim from the oracle at
    ``_torch_ref/butterfly.py:25-31``):

    * ``log_stride = idx if INCREASING_STRIDE else LOG_N - 1 - idx`` where
      ``idx = STAGE_START + stage_offset``.
    * ``stride = 1 << log_stride``.
    * Partner pair: ``tile_partner = tile_pos ^ stride`` (XOR within tile —
      valid because ``stride < TILE_N`` by construction of
      ``TILE_N = 2 * max(group_stride)``).
    * ``is_lower = (tile_pos & stride) == 0`` — True on the "lower" element.
    * Twiddle pair_flat index: ``pair_flat = col_start/2 + (tile_pos // (2*stride)) * stride + tile_pos % stride``.
      Mirrors the oracle's ``twiddle[:, block, idx].view(nstacks, n // (2*stride), stride, 2, 2)``
      layout after flatten (last 2 dims = ``side_out, side_in``).
    * 2x2 multiply: ``new_lower = t_00 * lower + t_01 * upper``,
      ``new_upper = t_10 * lower + t_11 * upper``. Per-position picks the
      correct row via ``tl.where(is_lower, new_lower, new_upper)``.

    Twiddle layout (last-dim fastest, row-major):
    ``(nstacks, nblocks, log_n, n // 2, 2, 2)``, so the per-position offset is::

        twiddle_off(s, block, idx, pair_flat, side_out, side_in)
            = s * (nblocks * log_n * (n // 2) * 4)
            + block * (log_n * (n // 2) * 4)
            + idx * ((n // 2) * 4)
            + pair_flat * 4
            + side_out * 2
            + side_in

    For Plan 07-01 (fp32 only) the ``IS_COMPLEX=True`` branch is gated by
    ``tl.static_assert``; Plan 07-02 lights it up via the 4-FMA template per
    ``04-COMPLEX-LAYOUT.md:58-76``.
    """
    # D-41a load-bearing gate. Plan 07-02 removes ONLY this single line.
    tl.static_assert(not IS_COMPLEX, "complex64 lands in 07-02 (D-41a pre-wiring)")

    row_id = tl.program_id(axis=0)     # column-tile index within a row
    bn_id = tl.program_id(axis=1)      # (batch, nstack) row id (flattened)

    # Decompose (batch, nstack) row id (consecutive bn_id values share twiddle
    # because nstack_idx varies fastest in this scheme).
    nstack_idx = bn_id % nstacks
    # batch_idx = bn_id // nstacks  (not strictly needed — used only for pointer math below)

    tile_offsets = tl.arange(0, TILE_N)  # in-tile indices [0, TILE_N)
    col_start = row_id * TILE_N
    pos = col_start + tile_offsets       # absolute column positions in the row

    # Row base offset (fp32 path: row stride = n; complex path would double
    # via view_as_real — gated off in 07-01).
    row_base = bn_id * n

    # Twiddle stride math. Twiddle shape (last-fastest, row-major) is
    # (nstacks, nblocks, log_n, n // 2, 2, 2) = numel per-stack
    # = nblocks * log_n * (n // 2) * 4 = nblocks * log_n * 2 * n.
    twiddle_stack_stride = nblocks * LOG_N * 2 * n
    twiddle_block_stride = LOG_N * 2 * n
    twiddle_stage_stride = 2 * n  # (n // 2) * 4
    twiddle_sb_base = nstack_idx * twiddle_stack_stride + block_idx * twiddle_block_stride

    # Seed output buffer with the input tile so the unrolled stage loop reads
    # uniformly from output_ptr (out_ptr-as-scratch per Phase 6 hadamard
    # pattern; cf. 06-01-SUMMARY.md "Auto-fixed Issues #1"). Full-tile load
    # — no mask needed because TILE_N divides n by construction.
    x = tl.load(input_ptr + row_base + pos)
    tl.store(output_ptr + row_base + pos, x)
    tl.debug_barrier()  # seed write visible before first stage's partner-load

    # Unrolled STAGE_COUNT (1..3) butterfly stages. STAGE_COUNT is constexpr
    # so tl.static_range guarantees JIT-time unrolling.
    for stage_offset in tl.static_range(STAGE_COUNT):
        idx = STAGE_START + stage_offset  # absolute stage index in [0, LOG_N)
        if INCREASING_STRIDE:
            log_stride = idx
        else:
            log_stride = LOG_N - 1 - idx
        stride = 1 << log_stride

        # Partner positions (within tile — valid because stride < TILE_N
        # by construction of TILE_N = 1 << (max_stage + 1)).
        tile_partner = tile_offsets ^ stride

        # Load self and partner from the scratch buffer.
        cur = tl.load(output_ptr + row_base + pos)
        partner = tl.load(output_ptr + row_base + (col_start + tile_partner))

        # Twiddle pair_flat index for each tile position. The oracle uses
        #   t = twiddle[:, block, idx].view(nstacks, n//(2*stride), stride, 2, 2)
        #         .permute(0, 1, 3, 4, 2)
        # which keeps twiddle in its native (n//2, 2, 2) flat layout where
        # pair_flat = pair_idx * stride + s_idx and
        # pair_idx = pos // (2*stride), s_idx = pos % stride.
        # Since col_start is divisible by 2*stride (because TILE_N covers the
        # widest stride in this group), col_start % (2*stride) == 0 and the
        # per-position pair_flat reduces to:
        #   pair_flat = col_start/2 + (tile_offsets // (2*stride)) * stride
        #               + tile_offsets % stride
        pair_flat = (col_start >> 1) + (tile_offsets // (2 * stride)) * stride \
            + (tile_offsets % stride)

        # Per-position twiddle base for this (nstack, nblock, stage).
        twiddle_stage_base = twiddle_sb_base + idx * twiddle_stage_stride
        # Each pair_flat has 4 entries (side_out, side_in) in
        # {(0,0), (0,1), (1,0), (1,1)} — last-dim-fastest row-major.
        pf4 = pair_flat * 4
        t00 = tl.load(twiddle_ptr + twiddle_stage_base + pf4 + 0)
        t01 = tl.load(twiddle_ptr + twiddle_stage_base + pf4 + 1)
        t10 = tl.load(twiddle_ptr + twiddle_stage_base + pf4 + 2)
        t11 = tl.load(twiddle_ptr + twiddle_stage_base + pf4 + 3)

        # For lower-side positions (is_lower): new = t00 * cur + t01 * partner.
        # For upper-side positions: new = t10 * partner + t11 * cur
        # (when upper, "lower" input in the 2x2 is the partner, "upper" is cur).
        is_lower = (tile_offsets & stride) == 0
        new_lower = t00 * cur + t01 * partner
        new_upper = t10 * partner + t11 * cur
        new_x = tl.where(is_lower, new_lower, new_upper)

        # Barrier BEFORE the store: ensure all threads finished reading cur/partner
        # from previous-stage state before any thread overwrites it.
        tl.debug_barrier()
        tl.store(output_ptr + row_base + pos, new_x)
        # Barrier AFTER the store: ensure the next-stage tl.load sees this
        # stage's writes consistently across all threads.
        tl.debug_barrier()


@triton_op("torch_structured::butterfly_multiply_triton", mutates_args={})
def butterfly_multiply(
    twiddle: torch.Tensor,
    input: torch.Tensor,
    increasing_stride: bool = True,
    output_size: Optional[int] = None,
) -> torch.Tensor:
    """Triton-backed butterfly_multiply forward (D-40, TRI-03).

    Parameters:
        twiddle: Tensor of shape ``(nstacks, nblocks, log_n, n // 2, 2, 2)`` —
            the butterfly factor stack. For Plan 07-01 must be float32; Plan
            07-02 will also accept complex64.
        input: Tensor of shape ``(batch_size, nstacks, input_size)``. If
            ``input_size < n = 1 << log_n``, the wrapper zero-pads on the
            last dim (mirrors ``_torch_ref/butterfly.py:18``).
        increasing_stride: True iff the butterfly stages within each block
            are traversed in increasing-stride order. Toggles between
            consecutive nblocks (mirrors ``_torch_ref/butterfly.py:32``).
        output_size: If not None, the result is trimmed to
            ``output[:, :, :output_size]`` (mirrors ``_torch_ref/butterfly.py:33``).
            Defaults to ``n``.

    Returns:
        Tensor of shape ``(batch_size, nstacks, output_size_actual)`` with
        the same dtype as ``input``.

    Implementation:
        * **Small-N fallback (D-42a):** for ``log_n <= 1`` the wrapper
          bypasses the kernel and routes through ``_butterfly_multiply_torch``
          for a uniform autograd graph at trivial sizes.
        * **Pad / trim (D-42):** wrapper-side ``F.pad`` and ``[:, :, :output_size]``.
        * **Multi-launch 3-stage tile (D-40):** ``ceil(log_n / 3)`` Triton
          launches per nblock; each launch handles up to 3 consecutive stages
          on a register-resident tile of width ``1 << (max_stage + 1)``.
        * **Ping-pong output buffers:** the wrapper allocates two buffers and
          alternates source/destination per stage-group launch to avoid
          in-place data dependencies.
        * **fp32 gate (Plan 07-01 only — D-41):** Plan 07-02 lifts the fp32
          assert and lights up the IS_COMPLEX path via the pre-wired
          ``view_as_real`` machinery.
    """
    # Wrapper-boundary preconditions (CLAUDE.md "Error Handling": assert for
    # preconditions). Pitfall 3 (contiguity) must hold before any view_as_real.
    assert input.dim() == 3, (
        f"input must be (batch, nstacks, input_size), got dim={input.dim()}"
    )
    assert twiddle.dtype == input.dtype, (
        f"twiddle.dtype ({twiddle.dtype}) must equal input.dtype ({input.dtype})"
    )
    assert input.is_contiguous(), "input must be contiguous (Pitfall 3)"
    assert twiddle.is_contiguous(), "twiddle must be contiguous (Pitfall 3)"
    # Plan 07-01-only gate (D-41). Plan 07-02 removes ONLY this single line.
    assert input.dtype == torch.float32, (
        f"Plan 07-01: fp32-only (complex64 lands in 07-02); got {input.dtype}"
    )

    batch_size, nstacks, input_size = input.shape
    nblocks = twiddle.shape[1]
    log_n = twiddle.shape[2]
    n = 1 << log_n
    assert twiddle.shape == (nstacks, nblocks, log_n, n // 2, 2, 2), (
        f"twiddle shape mismatch: expected (nstacks={nstacks}, nblocks={nblocks}, "
        f"log_n={log_n}, n//2={n // 2}, 2, 2), got {tuple(twiddle.shape)}"
    )
    output_size_actual = n if output_size is None else output_size
    assert output_size_actual <= n, (
        f"output_size ({output_size_actual}) must be <= n ({n})"
    )

    # D-42a small-N fallback. Bypasses kernel for log_n <= 1 where the
    # smallest 3-stage tile (TILE_N=8) is larger than n and Triton launch
    # overhead dominates. Still routes through register_autograd for a
    # uniform autograd graph.
    #
    # NOTE on the .clone(): at log_n=0 the oracle's inner loop is empty and
    # the returned tensor aliases the contiguous input. PyTorch's
    # ``triton_op`` infrastructure rejects ops whose output aliases an
    # input (alias check via _c_check_aliasing_constraint), so we clone to
    # break the alias. At log_n=1 the oracle's loop executes once and
    # produces a fresh tensor, so the clone is a no-op cost; we apply it
    # unconditionally for simplicity (Rule 1 bug fix).
    if log_n <= 1:
        return _butterfly_multiply_torch(twiddle, input, increasing_stride, output_size).clone()

    # D-42 pad/trim wrapping (mirrors _torch_ref/butterfly.py:18 + 33 verbatim).
    input = F.pad(input, (0, n - input_size)) if input_size < n else input[:, :, :n]
    input = input.contiguous()  # F.pad already contiguous; explicit for kernel pointer math.

    # Phase 4 view_as_real wrapper boundary (per 04-COMPLEX-LAYOUT.md:33-50).
    # Gated off in Plan 07-01 because the fp32-only assert above rejects
    # complex inputs; the conditional is included for source-level symmetry
    # with Plan 07-02 (which lights up by removing the fp32 assert).
    is_complex = input.is_complex()
    if is_complex:
        input_work = torch.view_as_real(input).contiguous()
        twiddle_work = torch.view_as_real(twiddle).contiguous()
    else:
        input_work = input
        twiddle_work = twiddle

    # Output buffer allocation. Ping-pong between two buffers across stage-group
    # launches to avoid in-place data dependencies (planner's call per Phase
    # 9 perf gate review). Full-N buffer; the wrapper trims to output_size on
    # return.
    buf_a = torch.empty(batch_size, nstacks, n, dtype=input.dtype, device=input.device)
    buf_b = torch.empty_like(buf_a)
    if is_complex:
        buf_a_work = torch.view_as_real(buf_a)
        buf_b_work = torch.view_as_real(buf_b)
    else:
        buf_a_work = buf_a
        buf_b_work = buf_b

    # Initialize: copy input_work into buf_a_work so the ping-pong loop starts
    # uniformly with buf_a as the source.
    buf_a_work.copy_(input_work)
    src_buf = buf_a_work
    dst_buf = buf_b_work

    # Python-side nblocks loop with cur_increasing_stride toggle (mirrors
    # _torch_ref/butterfly.py:22-32 verbatim per D-40a).
    #
    # Note on indexing semantics (load-bearing for the kernel):
    # The oracle uses a counter ``idx in range(log_n)`` and translates to a
    # log_stride via ``log_stride = idx if cur_increasing_stride else log_n - 1 - idx``.
    # The multi-launch scheme groups counter values in chunks of up to 3 and
    # passes COUNTER_START (= group_start) as the kernel's STAGE_START
    # constexpr. The kernel computes ``idx = STAGE_START + stage_offset`` so
    # idx is the *counter* (0..log_n-1), NOT the absolute stage index. The
    # INCREASING_STRIDE constexpr drives the direction mapping inside the
    # kernel. tile_n is sized to cover the largest log_stride in the group.
    cur_increasing_stride = increasing_stride
    for block in range(nblocks):
        for group_start in range(0, log_n, 3):
            counter_count = min(3, log_n - group_start)  # 1, 2, or 3
            # Largest log_stride in this counter range determines tile_n.
            if cur_increasing_stride:
                # log_strides: group_start, group_start+1, ..., group_start+counter_count-1
                max_log_stride = group_start + counter_count - 1
            else:
                # log_strides: log_n-1-group_start, log_n-1-(group_start+1), ...
                # Max = log_n-1-group_start (counter increases -> log_stride decreases).
                max_log_stride = log_n - 1 - group_start
            tile_n = 1 << (max_log_stride + 1)
            n_row_tiles = n // tile_n
            grid = (n_row_tiles, batch_size * nstacks)
            num_warps = _pick_num_warps(tile_n)
            wrap_triton(_butterfly_kernel)[grid](
                twiddle_work,
                src_buf,
                dst_buf,
                n,
                nstacks,
                block,
                nblocks,
                STAGE_START=group_start,           # counter start (D-40a)
                STAGE_COUNT=counter_count,         # 1..3
                INCREASING_STRIDE=cur_increasing_stride,
                LOG_N=log_n,
                IS_COMPLEX=is_complex,
                TILE_N=tile_n,
                num_warps=num_warps,
            )
            # Ping-pong: dst becomes next src.
            src_buf, dst_buf = dst_buf, src_buf
        cur_increasing_stride = not cur_increasing_stride  # mirrors oracle line 32

    # After the loop, src_buf holds the final output (because the last swap
    # made the just-written dst into the new src).
    final_work = src_buf
    if is_complex:
        final_output_full = torch.view_as_complex(final_work.contiguous())
    else:
        final_output_full = final_work

    # D-42 output_size trim (mirrors _torch_ref/butterfly.py:33).
    return final_output_full[:, :, :output_size_actual]


def _setup_context(ctx, inputs, output):
    """Save twiddle, input, and the two non-tensor flags for the two-input
    register_autograd backward (D-47).

    The backward delegates to ``_butterfly_multiply_torch`` via
    ``torch.autograd.grad`` for the (twiddle, input) gradient pair.
    """
    twiddle, input_, increasing_stride, output_size = inputs
    ctx.save_for_backward(twiddle, input_)
    ctx.increasing_stride = increasing_stride
    ctx.output_size = output_size


def _backward(ctx, grad_out):
    """Two-input register_autograd backward via torch.autograd.grad on the
    _torch_ref oracle (D-47).

    detach + requires_grad_ ensures both twiddle and input are traced through
    the oracle. Returns 4 values matching the 4 forward inputs
    ``(twiddle, input, increasing_stride, output_size)``; the last two are
    ``None`` because they are non-tensor (bool, Optional[int]).

    This is the **two-input** variant of Phase 5's Wirtinger pattern
    (Phase 5 ``diag_mult`` had two tensor inputs and used closed-form
    gradient formulas with ``.conj()`` for the Wirtinger correction).
    Phase 7 ``butterfly_multiply`` delegates the entire gradient
    computation to the oracle via ``torch.autograd.grad``. The pattern is
    NEW in the codebase — no prior Triton op uses ``torch.autograd.grad``
    in the backward callback.
    """
    twiddle, input_ = ctx.saved_tensors
    twiddle_d = twiddle.detach().requires_grad_(True)
    input_d = input_.detach().requires_grad_(True)
    with torch.enable_grad():
        out = _butterfly_multiply_torch(
            twiddle_d, input_d, ctx.increasing_stride, ctx.output_size
        )
    grad_twiddle, grad_input = torch.autograd.grad(
        out, [twiddle_d, input_d], grad_out, retain_graph=False
    )
    return grad_twiddle, grad_input, None, None


butterfly_multiply.register_autograd(_backward, setup_context=_setup_context)


@butterfly_multiply.register_fake
def _butterfly_multiply_fake(twiddle, input, increasing_stride=True, output_size=None):
    """Meta kernel — Phase 4 D-12 mandate (the literal 260419-p27 fix).

    The ``increasing_stride=True`` and ``output_size=None`` defaults mirror
    the wrapper's schema defaults — PyTorch's dispatch elides default-valued
    scalar args before calling the fake impl, so the defaults are
    **load-bearing** (Phase 6 06-01-SUMMARY.md lesson). Without them,
    FakeTensorMode with the default call pattern raises
    ``TypeError: missing positional argument``.
    """
    batch_size, nstacks, _ = input.shape
    log_n = twiddle.shape[2]
    n = 1 << log_n
    output_size_actual = n if output_size is None else output_size
    return torch.empty(
        batch_size, nstacks, output_size_actual,
        dtype=input.dtype, device=input.device,
    )
