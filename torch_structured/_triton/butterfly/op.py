"""Triton kernel + ``@triton_op`` wrapper for butterfly_multiply forward + backward (D-40, D-49/D-50, TRI-03/TRI-04).

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

Plan 07-02 lit up the complex64 path — view_as_real wrapper boundary actively
routes complex64 to the IS_COMPLEX=True kernel branch implementing the 4-FMA
per-pair multiply per
``.planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/04-COMPLEX-LAYOUT.md``
lines 58-76. The kernel signature is unchanged from Plan 07-01 per D-41a (zero
kernel-signature refactor between plans); only the kernel-entry static_assert
and the wrapper fp32-only assert were removed, and the ``if IS_COMPLEX:`` branch
in the per-stage body was filled in with the 4-FMA template adapted for the
butterfly's 2x2 matrix-vector multiply (four pairwise complex multiplies per
pair: ``new = t00*cur + t01*partner`` for the lower side, ``new = t10*partner
+ t11*cur`` for the upper side; each scalar multiply is the canonical
``(a + bi)(c + di) = (ac - bd) + (ad + bc)i`` 4-FMA pattern).

Small-N fallback (D-42a): when ``log_n <= 1`` (n in {1, 2}), the wrapper bypasses
the kernel entirely and delegates to ``_torch_ref.butterfly.butterfly_multiply_torch``
inside the ``triton_op`` so the autograd graph stays uniform across the
small-N / large-N split. Triton launch overhead would dominate at n in {1, 2}
and the smallest 3-stage tile (TILE_N=8) is wider than n.

Pad/trim (D-42): the wrapper does ``input = F.pad(input, (0, n - input_size))``
when ``input_size < n`` and ``output[:, :, :output_size]`` on return — mirrors
the oracle's lines 18, 33 verbatim.

Plan 08-01 backward (D-49/D-50/D-50a/D-57): the ``_backward`` callback body is
REPLACED with a Triton-backed two-input backward that (a) recomputes the
forward into a stage-group-granularity fp32 trail buffer by reusing the
existing forward kernel via the new ``_run_forward_stage_groups(..., trail_out)``
helper (D-49a), then (b) walks the stage groups in REVERSE order via a new
``@triton.jit _butterfly_backward_kernel`` that mirrors the forward kernel's
launch shape + barrier dance but: walks stages in reverse via
``tl.static_range(STAGE_COUNT - 1, -1, -1)``, accumulates ``d_twiddle`` via
per-program ``tl.sum`` reduce + 4 ``tl.atomic_add(..., sem='relaxed')`` per
stage into an fp32 scratch buffer (SC#3), and writes ``d_input`` via the
out_ptr-as-scratch ping-pong pattern Phase 7 uses for the in-flight gradient.
The ``_setup_context``, ``register_autograd`` registration line, and
``register_fake`` are UNCHANGED per D-57.

Plan 08-02 backward complex64 (D-50b/D-50c, RESEARCH correction #3): the
IS_COMPLEX=True kernel branch is now lit up — view_as_real wrapper boundary
actively routes complex64 to the conjugate-4-FMA per-pair multiply for BOTH
d_twiddle (``grad * conj(input)``) AND d_input (``conj(twiddle).T @ grad``)
per CONTEXT.md D-50c and RESEARCH correction #3. The sign-flip taxonomy
(forward 4-FMA has ``out_re = a_re*c_re - a_im*c_im`` MINUS and ``out_im =
a_re*c_im + a_im*c_re`` PLUS; the conjugate path has ``out_re = a_re*c_re +
a_im*c_im`` PLUS and ``out_im = a_im*c_re - a_re*c_im`` MINUS) is the
load-bearing LANDMINE detector for complex correctness — silently passes
fp32 tests (input.imag=0) but fails complex64 gradcheck. The wrapper boundary
gates accept ``{float32, complex64}`` (the fp32-only assert from Plan 08-01
was removed); the kernel-entry ``tl.static_assert(not IS_COMPLEX, ...)`` was
removed; the trail buffer doubles its trailing axis (``trail_n = n * 2``) via
view_as_real flatten for complex64; the d_twiddle scratch gets a trailing
``(2,)`` axis for the re/im split; ``view_as_complex(d_twiddle_scratch.contiguous())``
casts the fp32 scratch back to complex64 at the callback boundary.

Small-N fallback (D-49b): when ``log_n <= 1`` in ``_backward``, the wrapper
routes through ``torch.autograd.grad(_butterfly_multiply_torch(...))`` exactly
as Phase 7 did — Triton launch overhead would dominate at trivial sizes and
the trail/scratch machinery is unnecessary.
"""
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from torch.nn import functional as F
from typing import Optional

from torch_structured._torch_ref.butterfly import butterfly_multiply_torch as _butterfly_multiply_torch  # backward oracle (D-47, two-input via torch.autograd.grad) + small-N fallback (D-42a/D-49b)


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

    For ``IS_COMPLEX=True`` the kernel receives pointers into a ``view_as_real``
    layout (each logical complex element is 2 consecutive floats — real then
    imag). Per-row stride is ``2 * n`` (not ``n``); per-twiddle-entry stride
    is ``2`` (so ``pf8 = pair_flat * 8`` replaces ``pf4 = pair_flat * 4``).
    The per-stage multiply applies the canonical 4-FMA template from
    ``04-COMPLEX-LAYOUT.md:58-76``: ``(a + bi)(c + di) = (ac - bd) + (ad + bc)i``
    — applied per pairwise complex multiply inside the butterfly's
    ``t00*cur + t01*partner`` (lower-side) and ``t10*partner + t11*cur``
    (upper-side) formulas. ``t.l.where(is_lower, new_lower, new_upper)`` is
    applied to ``_re`` and ``_im`` parts independently (the mask is on the
    logical position, not the view_as_real position).
    """
    row_id = tl.program_id(axis=0)     # column-tile index within a row
    bn_id = tl.program_id(axis=1)      # (batch, nstack) row id (flattened)

    # Decompose (batch, nstack) row id (consecutive bn_id values share twiddle
    # because nstack_idx varies fastest in this scheme).
    nstack_idx = bn_id % nstacks
    # batch_idx = bn_id // nstacks  (not strictly needed — used only for pointer math below)

    tile_offsets = tl.arange(0, TILE_N)  # in-tile indices [0, TILE_N)
    col_start = row_id * TILE_N
    pos = col_start + tile_offsets       # absolute column positions in the row

    # Row / twiddle stride math. For the fp32 (real) path the row stride is
    # ``n`` and each twiddle entry occupies 1 float. For the complex64 path the
    # row stride is ``2 * n`` and each twiddle entry occupies 2 floats (re, im),
    # consistent with the wrapper's ``torch.view_as_real`` boundary translation
    # per ``04-COMPLEX-LAYOUT.md`` lines 33-50. We compute both unconditionally
    # — the constexpr branch below selects which to use.
    if IS_COMPLEX:
        # view_as_real layout: each logical element = 2 consecutive floats.
        row_base = bn_id * (2 * n)
        twiddle_stack_stride = nblocks * LOG_N * 2 * n * 2
        twiddle_block_stride = LOG_N * 2 * n * 2
        twiddle_stage_stride = 2 * n * 2  # (n // 2) * 4 * 2
    else:
        row_base = bn_id * n
        twiddle_stack_stride = nblocks * LOG_N * 2 * n
        twiddle_block_stride = LOG_N * 2 * n
        twiddle_stage_stride = 2 * n  # (n // 2) * 4
    twiddle_sb_base = nstack_idx * twiddle_stack_stride + block_idx * twiddle_block_stride

    # Seed output buffer with the input tile so the unrolled stage loop reads
    # uniformly from output_ptr (out_ptr-as-scratch per Phase 6 hadamard
    # pattern; cf. 06-01-SUMMARY.md "Auto-fixed Issues #1"). Full-tile load
    # — no mask needed because TILE_N divides n by construction.
    if IS_COMPLEX:
        # Per-element offsets into the view_as_real layout: real at pos*2,
        # imag at pos*2 + 1. Load and store separately so the in-register
        # values stay as two parallel (re, im) register vectors (Option (b)
        # from the 07-02 plan — cleanest and matches the diag_mult template
        # at _triton/diag_mult/op.py:64-87 verbatim).
        pos2 = pos * 2
        x_re = tl.load(input_ptr + row_base + pos2)
        x_im = tl.load(input_ptr + row_base + pos2 + 1)
        tl.store(output_ptr + row_base + pos2, x_re)
        tl.store(output_ptr + row_base + pos2 + 1, x_im)
    else:
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

        # Per-position twiddle base for this (nstack, nblock, stage). The
        # ``is_lower`` mask is on the logical position (same for re and im
        # in the complex path).
        twiddle_stage_base = twiddle_sb_base + idx * twiddle_stage_stride
        is_lower = (tile_offsets & stride) == 0

        if IS_COMPLEX:
            # Each pair_flat has 4 logical complex entries (side_out, side_in)
            # in {(0,0), (0,1), (1,0), (1,1)}; each occupies 2 floats (re, im)
            # in the view_as_real layout — so per-pair stride is 8 floats and
            # the 8 reals are at offsets 0..7 (t00_re, t00_im, t01_re, t01_im,
            # t10_re, t10_im, t11_re, t11_im) per 04-COMPLEX-LAYOUT.md:58-76.
            pf8 = pair_flat * 8
            t00_re = tl.load(twiddle_ptr + twiddle_stage_base + pf8 + 0)
            t00_im = tl.load(twiddle_ptr + twiddle_stage_base + pf8 + 1)
            t01_re = tl.load(twiddle_ptr + twiddle_stage_base + pf8 + 2)
            t01_im = tl.load(twiddle_ptr + twiddle_stage_base + pf8 + 3)
            t10_re = tl.load(twiddle_ptr + twiddle_stage_base + pf8 + 4)
            t10_im = tl.load(twiddle_ptr + twiddle_stage_base + pf8 + 5)
            t11_re = tl.load(twiddle_ptr + twiddle_stage_base + pf8 + 6)
            t11_im = tl.load(twiddle_ptr + twiddle_stage_base + pf8 + 7)

            # Load self and partner (re, im) from the scratch buffer.
            pos2 = pos * 2
            partner_pos2 = (col_start + tile_partner) * 2
            cur_re = tl.load(output_ptr + row_base + pos2)
            cur_im = tl.load(output_ptr + row_base + pos2 + 1)
            partner_re = tl.load(output_ptr + row_base + partner_pos2)
            partner_im = tl.load(output_ptr + row_base + partner_pos2 + 1)

            # 4-FMA template (verbatim from 04-COMPLEX-LAYOUT.md:65-66) applied
            # per pairwise complex multiply:
            #   (a + bi)(c + di) = (ac - bd) + (ad + bc)i
            # For the lower-side new value:
            #   new_lower = t00 * cur + t01 * partner   (both complex)
            # For the upper-side new value:
            #   new_upper = t10 * partner + t11 * cur   (both complex)
            # Four pairwise complex multiplies + two complex adds per stage.
            t00_cur_re = t00_re * cur_re - t00_im * cur_im
            t00_cur_im = t00_re * cur_im + t00_im * cur_re
            t01_partner_re = t01_re * partner_re - t01_im * partner_im
            t01_partner_im = t01_re * partner_im + t01_im * partner_re
            t10_partner_re = t10_re * partner_re - t10_im * partner_im
            t10_partner_im = t10_re * partner_im + t10_im * partner_re
            t11_cur_re = t11_re * cur_re - t11_im * cur_im
            t11_cur_im = t11_re * cur_im + t11_im * cur_re

            new_lower_re = t00_cur_re + t01_partner_re
            new_lower_im = t00_cur_im + t01_partner_im
            new_upper_re = t10_partner_re + t11_cur_re
            new_upper_im = t10_partner_im + t11_cur_im

            new_x_re = tl.where(is_lower, new_lower_re, new_upper_re)
            new_x_im = tl.where(is_lower, new_lower_im, new_upper_im)

            # Barrier BEFORE the store: ensure all threads finished reading
            # cur/partner from previous-stage state before any thread
            # overwrites it.
            tl.debug_barrier()
            tl.store(output_ptr + row_base + pos2, new_x_re)
            tl.store(output_ptr + row_base + pos2 + 1, new_x_im)
            # Barrier AFTER the store: ensure the next-stage tl.load sees this
            # stage's writes consistently across all threads.
            tl.debug_barrier()
        else:
            # Each pair_flat has 4 entries (side_out, side_in) in
            # {(0,0), (0,1), (1,0), (1,1)} — last-dim-fastest row-major.
            pf4 = pair_flat * 4
            t00 = tl.load(twiddle_ptr + twiddle_stage_base + pf4 + 0)
            t01 = tl.load(twiddle_ptr + twiddle_stage_base + pf4 + 1)
            t10 = tl.load(twiddle_ptr + twiddle_stage_base + pf4 + 2)
            t11 = tl.load(twiddle_ptr + twiddle_stage_base + pf4 + 3)

            # Load self and partner from the scratch buffer.
            cur = tl.load(output_ptr + row_base + pos)
            partner = tl.load(output_ptr + row_base + (col_start + tile_partner))

            # For lower-side positions (is_lower): new = t00 * cur + t01 * partner.
            # For upper-side positions: new = t10 * partner + t11 * cur
            # (when upper, "lower" input in the 2x2 is the partner, "upper" is cur).
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


def _run_forward_stage_groups(
    twiddle_work: torch.Tensor,
    input_work: torch.Tensor,
    increasing_stride: bool,
    log_n: int,
    n: int,
    nstacks: int,
    nblocks: int,
    batch_size: int,
    is_complex: bool,
    *,
    trail_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Phase 7 forward stage-group launch loop, factored out per D-49a.

    Two modes:

    * ``trail_out is None`` (forward call path): byte-equivalent to Phase 7's
      original inlined wrapper launch loop. Ping-pongs between two output
      buffers (``buf_a``/``buf_b``); after the loop the final result lives in
      whichever buffer was the last destination. Returns that buffer as a
      dtype-matched tensor (or its ``view_as_real`` view, depending on
      ``is_complex``); the wrapper takes the view-as-complex round-trip + the
      output_size trim.
    * ``trail_out is not None`` (Plan 08-01 backward recompute path): each
      stage-group launch writes its output to ``trail_out[launch_idx]``
      instead of the ping-pong destination. ``launch_idx`` walks
      ``0..n_launches_per_nblock * nblocks - 1`` in forward order. After each
      launch, ``src_buf`` is repointed at ``trail_out[launch_idx]`` so the
      next stage group reads from the just-written trail slot. The return
      value is meaningless in this mode (the caller has the trail).

    Parameters:
        twiddle_work: the twiddle pointer the kernel should read. For fp32:
            ``twiddle`` itself. For complex64: ``torch.view_as_real(twiddle)``
            (caller is responsible for the view-as-real round-trip + contiguity
            assertion).
        input_work: the per-row input tile the FIRST stage-group launch reads.
            Shape ``(batch_size, nstacks, n)`` for fp32; ``(batch_size, nstacks,
            n, 2)`` for complex64 (view_as_real). Must be contiguous.
        increasing_stride: matches the wrapper's flag, toggles between nblocks.
        log_n, n, nstacks, nblocks, batch_size: redundantly passed (already
            visible to caller) so the helper need not re-derive them.
        is_complex: True iff inputs were complex64 (view_as_real-wrapped).
        trail_out: optional. Shape ``(n_launches_per_nblock * nblocks,
            batch_size, nstacks, n)`` fp32 OR (for complex64, future Plan
            08-02 use)  ``(n_launches_per_nblock * nblocks, batch_size,
            nstacks, n, 2)``. When provided, each launch writes its output
            to ``trail_out[launch_idx]`` instead of the ping-pong buffer.

    Returns:
        ``src_buf`` (the post-loop final result), dtype-matched. In
        ``trail_out`` mode the returned tensor is not the trail — it's
        whichever ping-pong buffer was assigned as src on the very last
        iteration (not a meaningful tensor; caller discards).
    """
    # Allocate ping-pong buffers (fp32 dtype always; for complex64 these hold
    # the view_as_real layout). The dtype matches input_work's dtype because
    # input_work is already view_as_real-ed in the complex path.
    buf_a = torch.empty_like(input_work)
    buf_b = torch.empty_like(input_work)
    # Initialize: copy input_work into buf_a so the ping-pong loop starts
    # uniformly with buf_a as the source.
    buf_a.copy_(input_work)
    src_buf = buf_a
    dst_buf = buf_b

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
    launch_idx = 0
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
            # D-49a: redirect this launch's output_ptr to trail_out[launch_idx]
            # when in recompute-for-backward mode; otherwise use the ping-pong
            # destination as Phase 7 originally did.
            if trail_out is not None:
                dst_for_this_launch = trail_out[launch_idx]
            else:
                dst_for_this_launch = dst_buf
            wrap_triton(_butterfly_kernel)[grid](
                twiddle_work,
                src_buf,
                dst_for_this_launch,
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
            # Ping-pong: dst becomes next src. In trail mode, the just-written
            # trail slot becomes the next launch's source (the backward needs
            # to feed THIS launch's output to the NEXT launch's input — same
            # invariant as Phase 7's ping-pong).
            if trail_out is not None:
                src_buf = trail_out[launch_idx]
            else:
                src_buf, dst_buf = dst_buf, src_buf
            launch_idx += 1
        cur_increasing_stride = not cur_increasing_stride  # mirrors oracle line 32

    # After the loop, src_buf holds the final output (because the last swap
    # made the just-written dst into the new src).
    return src_buf


@triton.jit
def _backward_one_stage(
    twiddle_stage_base_ptr,
    grad_out_ptr,
    d_twiddle_scratch_stage_base_ptr,
    x_in,
    xp_in,
    g,
    row_base,
    col_start,
    tile_offsets,
    pos,
    STRIDE: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """One reverse-walk stage (Plan 08-01 fp32 real path). All shape-determining
    values are constexpr so ``tl.reshape`` accepts the shape literal.

    Parameters:
        twiddle_stage_base_ptr: pointer offset = ``twiddle_ptr +
            twiddle_sb_base + idx * twiddle_stage_stride`` (per-stage base
            address inside the twiddle buffer; pre-computed by the caller).
        d_twiddle_scratch_stage_base_ptr: same offset into the fp32 scratch
            buffer (same shape as twiddle).
        x_in: per-lane activation INPUT to this stage (from the forward
            recompute walk).
        xp_in: per-lane partner activation INPUT (XOR-partner-loaded).
        g: per-lane in-flight gradient BEFORE this stage's update.
        row_base, col_start, tile_offsets, pos: position arithmetic
            from the kernel scope.
        STRIDE: constexpr ``1 << log_stride`` for this stage.
        TILE_N: constexpr tile width.

    Returns:
        ``new_g`` — the per-lane d_input update for the next reverse stage.
    """
    two_stride: tl.constexpr = 2 * STRIDE
    n_pair_blocks: tl.constexpr = TILE_N // two_stride  # outer blocks of 2*stride positions
    n_pairs_in_tile: tl.constexpr = TILE_N // 2          # total pairs in the tile

    tile_partner = tile_offsets ^ STRIDE
    pair_flat = (col_start >> 1) + (tile_offsets // two_stride) * STRIDE \
        + (tile_offsets % STRIDE)
    is_lower = (tile_offsets & STRIDE) == 0
    pf4 = pair_flat * 4
    t00 = tl.load(twiddle_stage_base_ptr + pf4 + 0)
    t01 = tl.load(twiddle_stage_base_ptr + pf4 + 1)
    t10 = tl.load(twiddle_stage_base_ptr + pf4 + 2)
    t11 = tl.load(twiddle_stage_base_ptr + pf4 + 3)

    # Partner-load for g via out_ptr-as-scratch barrier dance (mirrors
    # forward kernel op.py:286-292).
    partner_g = tl.load(grad_out_ptr + row_base + (col_start + tile_partner))

    # d_twiddle accumulation (per-lane masked + per-pair tl.sum reduce + 4
    # tl.atomic_add per stage into fp32 scratch — D-50a / SC#3).
    g_lower_eff = tl.where(is_lower, g, 0.0)
    g_upper_eff = tl.where(is_lower, 0.0, g)
    x_lower_eff = tl.where(is_lower, x_in, 0.0)
    x_upper_eff = tl.where(is_lower, 0.0, x_in)
    xp_lower_eff = tl.where(is_lower, 0.0, xp_in)
    xp_upper_eff = tl.where(is_lower, xp_in, 0.0)

    dt_00_contrib = g_lower_eff * x_lower_eff
    dt_01_contrib = g_lower_eff * xp_upper_eff
    dt_10_contrib = g_upper_eff * xp_lower_eff
    dt_11_contrib = g_upper_eff * x_upper_eff

    # Per-program reduce: pair up the lower-side and upper-side lanes that
    # share the same pair_flat. The tile of TILE_N positions has
    # ``n_pair_blocks`` blocks of 2*stride positions each. Within a block,
    # positions are laid out as
    #   [lower_inner_0, lower_inner_1, ..., lower_inner_{stride-1},
    #    upper_inner_0, upper_inner_1, ..., upper_inner_{stride-1}]
    # where lower/upper refers to the bit set by ``STRIDE``. The pair_flat
    # values within one block are
    #   [pair_0, pair_1, ..., pair_{stride-1}, pair_0, pair_1, ...]
    # i.e. pair_flat depends ONLY on the inner index (and the block).
    # Reshape to ``(n_pair_blocks, 2, STRIDE)`` so axis=1 sums lower+upper
    # for each (block, inner) pair, then flatten to ``n_pairs_in_tile``
    # outputs mapping linearly to pair_flat values (block * STRIDE + inner).
    dt_00_reshaped = tl.reshape(dt_00_contrib, (n_pair_blocks, 2, STRIDE))
    dt_01_reshaped = tl.reshape(dt_01_contrib, (n_pair_blocks, 2, STRIDE))
    dt_10_reshaped = tl.reshape(dt_10_contrib, (n_pair_blocks, 2, STRIDE))
    dt_11_reshaped = tl.reshape(dt_11_contrib, (n_pair_blocks, 2, STRIDE))
    dt_00_per_pair = tl.reshape(tl.sum(dt_00_reshaped, axis=1), (n_pairs_in_tile,))
    dt_01_per_pair = tl.reshape(tl.sum(dt_01_reshaped, axis=1), (n_pairs_in_tile,))
    dt_10_per_pair = tl.reshape(tl.sum(dt_10_reshaped, axis=1), (n_pairs_in_tile,))
    dt_11_per_pair = tl.reshape(tl.sum(dt_11_reshaped, axis=1), (n_pairs_in_tile,))

    # Per-pair atomic-add into the fp32 scratch. The flatten layout above
    # gives pair_flat order (block * STRIDE + inner) which matches the
    # forward kernel's pair_flat formula at the start of the tile.
    pair_flat_in_tile = tl.arange(0, n_pairs_in_tile)
    pair_flat_global = (col_start >> 1) + pair_flat_in_tile
    twiddle_pair_4 = pair_flat_global * 4

    # 4 tl.atomic_add per stage per program. sem='relaxed' per RESEARCH
    # §"tl.atomic_add Semantics on fp32" Gap #7.
    tl.atomic_add(d_twiddle_scratch_stage_base_ptr + twiddle_pair_4 + 0, dt_00_per_pair, sem='relaxed')
    tl.atomic_add(d_twiddle_scratch_stage_base_ptr + twiddle_pair_4 + 1, dt_01_per_pair, sem='relaxed')
    tl.atomic_add(d_twiddle_scratch_stage_base_ptr + twiddle_pair_4 + 2, dt_10_per_pair, sem='relaxed')
    tl.atomic_add(d_twiddle_scratch_stage_base_ptr + twiddle_pair_4 + 3, dt_11_per_pair, sem='relaxed')

    # d_input (= new_g) update — T^T @ g.
    new_g_lower = t00 * g + t10 * partner_g
    new_g_upper = t01 * partner_g + t11 * g
    new_g = tl.where(is_lower, new_g_lower, new_g_upper)

    tl.debug_barrier()
    tl.store(grad_out_ptr + row_base + pos, new_g)
    tl.debug_barrier()
    return new_g


@triton.jit
def _backward_one_stage_complex(
    twiddle_stage_base_ptr,
    grad_out_ptr,
    d_twiddle_scratch_stage_base_ptr,
    x_in_re,
    x_in_im,
    xp_in_re,
    xp_in_im,
    g_re,
    g_im,
    row_base,
    col_start,
    tile_offsets,
    pos,
    STRIDE: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """One reverse-walk stage — Plan 08-02 complex64 conjugate-4-FMA path.

    Implements D-50c (d_twiddle = ``grad * conj(input)``) AND RESEARCH
    correction #3 (d_input = ``conj(twiddle).T @ grad``). Both formulas use
    the SAME sign-flip pattern relative to the forward 4-FMA:

    - Forward (Phase 7 op.py:266-274): ``(a+bi)(c+di) = (ac-bd) + (ad+bc)i``
      (MINUS in real, PLUS in imag).
    - Conjugate ``(a+bi)*conj(c+di) = (a+bi)(c-di) = (ac+bd) + (bc-ad)i``
      (PLUS in real, MINUS in imag) — applied to d_twiddle when c+di is the
      input being conjugated, and to d_input when c+di is the twiddle being
      conjugated (since complex multiply is commutative under conjugation
      swap; see RESEARCH §"Conjugate Sign Flip in the Complex64 Backward
      4-FMA" Gap #8).

    The d_twiddle scratch has a trailing-2 axis (re/im split via view_as_real
    flatten per D-50b), so each pair occupies 8 fp32 slots at offset
    ``pair_flat * 8`` (4 t_ij x 2 re/im). Per-pair atomics: 8 per stage
    instead of 4. The pair_flat reduce semantics are identical to the real
    path — re and im components are summed independently.

    Parameters:
        x_in_re, x_in_im: real/imag of per-lane activation INPUT.
        xp_in_re, xp_in_im: real/imag of partner activation INPUT.
        g_re, g_im: real/imag of per-lane in-flight gradient.
        Other params: as ``_backward_one_stage``.

    Returns:
        ``(new_g_re, new_g_im)`` — re/im of d_input update for next stage.
    """
    two_stride: tl.constexpr = 2 * STRIDE
    n_pair_blocks: tl.constexpr = TILE_N // two_stride
    n_pairs_in_tile: tl.constexpr = TILE_N // 2

    tile_partner = tile_offsets ^ STRIDE
    pair_flat = (col_start >> 1) + (tile_offsets // two_stride) * STRIDE \
        + (tile_offsets % STRIDE)
    is_lower = (tile_offsets & STRIDE) == 0

    # Load 8 floats per twiddle pair (4 t_ij x 2 re/im) at pair_flat * 8 offset.
    pf8 = pair_flat * 8
    t00_re = tl.load(twiddle_stage_base_ptr + pf8 + 0)
    t00_im = tl.load(twiddle_stage_base_ptr + pf8 + 1)
    t01_re = tl.load(twiddle_stage_base_ptr + pf8 + 2)
    t01_im = tl.load(twiddle_stage_base_ptr + pf8 + 3)
    t10_re = tl.load(twiddle_stage_base_ptr + pf8 + 4)
    t10_im = tl.load(twiddle_stage_base_ptr + pf8 + 5)
    t11_re = tl.load(twiddle_stage_base_ptr + pf8 + 6)
    t11_im = tl.load(twiddle_stage_base_ptr + pf8 + 7)

    # Partner-load for g (re and im separately, via stride-2 access).
    partner_pos2 = (col_start + tile_partner) * 2
    partner_g_re = tl.load(grad_out_ptr + row_base + partner_pos2)
    partner_g_im = tl.load(grad_out_ptr + row_base + partner_pos2 + 1)

    # ---------- d_twiddle: grad * conj(input) per D-50c ----------
    # Mask per-lane contributions to each t_ij entry's pair_flat slot.
    g_lower_re_eff = tl.where(is_lower, g_re, 0.0)
    g_lower_im_eff = tl.where(is_lower, g_im, 0.0)
    g_upper_re_eff = tl.where(is_lower, 0.0, g_re)
    g_upper_im_eff = tl.where(is_lower, 0.0, g_im)

    x_lower_re_eff = tl.where(is_lower, x_in_re, 0.0)
    x_lower_im_eff = tl.where(is_lower, x_in_im, 0.0)
    x_upper_re_eff = tl.where(is_lower, 0.0, x_in_re)
    x_upper_im_eff = tl.where(is_lower, 0.0, x_in_im)
    xp_lower_re_eff = tl.where(is_lower, 0.0, xp_in_re)
    xp_lower_im_eff = tl.where(is_lower, 0.0, xp_in_im)
    xp_upper_re_eff = tl.where(is_lower, xp_in_re, 0.0)
    xp_upper_im_eff = tl.where(is_lower, xp_in_im, 0.0)

    # Conjugate 4-FMA: g * conj(x) = (g_re + i g_im)(x_re - i x_im)
    #                              = (g_re*x_re + g_im*x_im) + i(g_im*x_re - g_re*x_im)
    # SIGN-FLIP: real has + (forward MINUS), imag has - (forward PLUS).
    #
    # d_t_00 = g_lower * conj(x_lower) — note "lower" is x_in for is_lower lanes
    # d_t_01 = g_lower * conj(x_upper) — x_upper for is_lower lanes is the PARTNER
    # d_t_10 = g_upper * conj(x_lower) — x_lower for upper lanes is the PARTNER
    # d_t_11 = g_upper * conj(x_upper) — x_upper for upper lanes is x_in
    dt_00_re_contrib = g_lower_re_eff * x_lower_re_eff + g_lower_im_eff * x_lower_im_eff
    dt_00_im_contrib = g_lower_im_eff * x_lower_re_eff - g_lower_re_eff * x_lower_im_eff
    dt_01_re_contrib = g_lower_re_eff * xp_upper_re_eff + g_lower_im_eff * xp_upper_im_eff
    dt_01_im_contrib = g_lower_im_eff * xp_upper_re_eff - g_lower_re_eff * xp_upper_im_eff
    dt_10_re_contrib = g_upper_re_eff * xp_lower_re_eff + g_upper_im_eff * xp_lower_im_eff
    dt_10_im_contrib = g_upper_im_eff * xp_lower_re_eff - g_upper_re_eff * xp_lower_im_eff
    dt_11_re_contrib = g_upper_re_eff * x_upper_re_eff + g_upper_im_eff * x_upper_im_eff
    dt_11_im_contrib = g_upper_im_eff * x_upper_re_eff - g_upper_re_eff * x_upper_im_eff

    # Per-program pair_flat reduce (identical semantics to real path — re and
    # im are reduced independently).
    dt_00_re_per_pair = tl.reshape(tl.sum(
        tl.reshape(dt_00_re_contrib, (n_pair_blocks, 2, STRIDE)), axis=1), (n_pairs_in_tile,))
    dt_00_im_per_pair = tl.reshape(tl.sum(
        tl.reshape(dt_00_im_contrib, (n_pair_blocks, 2, STRIDE)), axis=1), (n_pairs_in_tile,))
    dt_01_re_per_pair = tl.reshape(tl.sum(
        tl.reshape(dt_01_re_contrib, (n_pair_blocks, 2, STRIDE)), axis=1), (n_pairs_in_tile,))
    dt_01_im_per_pair = tl.reshape(tl.sum(
        tl.reshape(dt_01_im_contrib, (n_pair_blocks, 2, STRIDE)), axis=1), (n_pairs_in_tile,))
    dt_10_re_per_pair = tl.reshape(tl.sum(
        tl.reshape(dt_10_re_contrib, (n_pair_blocks, 2, STRIDE)), axis=1), (n_pairs_in_tile,))
    dt_10_im_per_pair = tl.reshape(tl.sum(
        tl.reshape(dt_10_im_contrib, (n_pair_blocks, 2, STRIDE)), axis=1), (n_pairs_in_tile,))
    dt_11_re_per_pair = tl.reshape(tl.sum(
        tl.reshape(dt_11_re_contrib, (n_pair_blocks, 2, STRIDE)), axis=1), (n_pairs_in_tile,))
    dt_11_im_per_pair = tl.reshape(tl.sum(
        tl.reshape(dt_11_im_contrib, (n_pair_blocks, 2, STRIDE)), axis=1), (n_pairs_in_tile,))

    # 8 tl.atomic_add per stage per program (vs 4 for real path) at offset
    # pair_flat * 8 (vs * 4 for real). sem='relaxed' identical.
    pair_flat_in_tile = tl.arange(0, n_pairs_in_tile)
    pair_flat_global = (col_start >> 1) + pair_flat_in_tile
    twiddle_pair_8 = pair_flat_global * 8
    tl.atomic_add(d_twiddle_scratch_stage_base_ptr + twiddle_pair_8 + 0, dt_00_re_per_pair, sem='relaxed')
    tl.atomic_add(d_twiddle_scratch_stage_base_ptr + twiddle_pair_8 + 1, dt_00_im_per_pair, sem='relaxed')
    tl.atomic_add(d_twiddle_scratch_stage_base_ptr + twiddle_pair_8 + 2, dt_01_re_per_pair, sem='relaxed')
    tl.atomic_add(d_twiddle_scratch_stage_base_ptr + twiddle_pair_8 + 3, dt_01_im_per_pair, sem='relaxed')
    tl.atomic_add(d_twiddle_scratch_stage_base_ptr + twiddle_pair_8 + 4, dt_10_re_per_pair, sem='relaxed')
    tl.atomic_add(d_twiddle_scratch_stage_base_ptr + twiddle_pair_8 + 5, dt_10_im_per_pair, sem='relaxed')
    tl.atomic_add(d_twiddle_scratch_stage_base_ptr + twiddle_pair_8 + 6, dt_11_re_per_pair, sem='relaxed')
    tl.atomic_add(d_twiddle_scratch_stage_base_ptr + twiddle_pair_8 + 7, dt_11_im_per_pair, sem='relaxed')

    # ---------- d_input: conj(T).T @ g per RESEARCH correction #3 ----------
    # Per-pair: new_g_lower = conj(t00)*g + conj(t10)*partner_g (for is_lower lanes)
    #          new_g_upper = conj(t01)*partner_g + conj(t11)*g (for upper lanes)
    # Conjugate 4-FMA: conj(t) * g = (t_re - i t_im)(g_re + i g_im)
    #                              = (t_re*g_re + t_im*g_im) + i(t_re*g_im - t_im*g_re)
    # SIGN-FLIP: real has + (forward MINUS), imag has - (forward PLUS).
    # t00_conj_g: re = t00_re * g_re + t00_im * g_im,  im = t00_re * g_im - t00_im * g_re
    t00_g_re = t00_re * g_re + t00_im * g_im
    t00_g_im = t00_re * g_im - t00_im * g_re
    t10_pg_re = t10_re * partner_g_re + t10_im * partner_g_im
    t10_pg_im = t10_re * partner_g_im - t10_im * partner_g_re
    t01_pg_re = t01_re * partner_g_re + t01_im * partner_g_im
    t01_pg_im = t01_re * partner_g_im - t01_im * partner_g_re
    t11_g_re = t11_re * g_re + t11_im * g_im
    t11_g_im = t11_re * g_im - t11_im * g_re

    new_g_lower_re = t00_g_re + t10_pg_re
    new_g_lower_im = t00_g_im + t10_pg_im
    new_g_upper_re = t01_pg_re + t11_g_re
    new_g_upper_im = t01_pg_im + t11_g_im

    new_g_re = tl.where(is_lower, new_g_lower_re, new_g_upper_re)
    new_g_im = tl.where(is_lower, new_g_lower_im, new_g_upper_im)

    # Store both re and im back to the scratch buffer at stride-2 offsets.
    pos2 = pos * 2
    tl.debug_barrier()
    tl.store(grad_out_ptr + row_base + pos2, new_g_re)
    tl.store(grad_out_ptr + row_base + pos2 + 1, new_g_im)
    tl.debug_barrier()
    return new_g_re, new_g_im


@triton.jit
def _butterfly_backward_kernel(
    twiddle_ptr,
    trail_in_ptr,            # the trail slot holding the activation INPUT to this stage group
    grad_in_ptr,             # gradient flowing FROM later stages INTO this group
    grad_out_ptr,            # d_input output buffer for this group (= upstream grad for the next reverse group)
    d_twiddle_scratch_ptr,   # fp32 scratch buffer for atomic-add d_twiddle accumulation
    n,
    nstacks,
    block_idx,
    nblocks,
    STAGE_START: tl.constexpr,    # first stage counter in this group (mirrors forward kernel)
    STAGE_COUNT: tl.constexpr,    # number of stages this launch handles (1..3) (D-50)
    INCREASING_STRIDE: tl.constexpr,
    LOG_N: tl.constexpr,
    IS_COMPLEX: tl.constexpr,     # Plan 08-01 pre-wires per D-51a; gated by static_assert below
    TILE_N: tl.constexpr,
    STRIDE_0: tl.constexpr,       # per-stage 1<<log_stride, host-computed (Triton 3.3.x compat)
    STRIDE_1: tl.constexpr,       # unused slots are clamped to 1 on the host
    STRIDE_2: tl.constexpr,
):
    """Plan 08-01 Triton backward kernel — reverse stage-group walk + per-program tl.sum reduce + tl.atomic_add(sem='relaxed') into fp32 scratch (D-50, D-50a, SC#3).

    Mirrors ``_butterfly_kernel``'s launch shape, constexprs, ``_pick_num_warps``
    schedule, twiddle pointer arithmetic, and the out_ptr-as-scratch barrier
    dance — but: walks stages in REVERSE via ``tl.static_range(STAGE_COUNT - 1,
    -1, -1)`` (D-50), accumulates ``d_twiddle`` via per-program ``tl.sum``
    reduce across the in-tile pair multiplicity + 4 ``tl.atomic_add(...,
    sem='relaxed')`` per stage into the fp32 ``d_twiddle_scratch`` (D-50a /
    SC#3 verbatim — "block-level tl.sum reduce before its single atomicAdd per
    block"), and writes the per-tile ``d_input`` via ``tl.store(grad_out_ptr,
    g)`` after the reverse loop completes.

    Plan 08-02 lit up the IS_COMPLEX=True path: the kernel dispatches on
    ``IS_COMPLEX`` constexpr to either the real (Plan 08-01) or conjugate
    (Plan 08-02) per-stage backward math. The kernel signature is unchanged
    from Plan 08-01 per D-51a (zero kernel-signature refactor).

    Recompute strategy for STAGE_COUNT > 1 stages within a group: the trail
    slot ``trail_in_ptr`` holds the activation BEFORE the first stage of the
    group ran. For the LAST stage of the group's backward (= first stage of
    the reverse walk), we need the activation INPUT to that last forward stage
    — which is the OUTPUT of stage STAGE_COUNT-2 (or trail_in for STAGE_COUNT=1).
    We manually unroll the forward walk for stages 0..STAGE_COUNT-1 in
    registers (using ``grad_out_ptr`` as the partner-exchange scratch
    buffer) to materialize ``x_stages[0..STAGE_COUNT-1]`` where
    ``x_stages[s]`` is the activation at the INPUT of forward stage s. After
    the forward walk we overwrite ``grad_out_ptr`` with the incoming gradient
    ``g`` and begin the reverse walk; the per-stage d_input update uses the
    same ``grad_out_ptr``-as-scratch barrier dance Phase 7 uses for the
    in-flight tile (mirrors op.py:286-292, 315-319).

    Per-stage d_twiddle accumulation (D-50a + SC#3):

    * ``g_lower_eff = where(is_lower, g, 0)``, ``g_upper_eff = where(is_lower,
      0, g)`` — masked contributions per side.
    * ``x_lower_eff`` / ``x_upper_eff`` / ``xp_lower_eff`` / ``xp_upper_eff``
      similarly. Note: ``x`` is the per-lane activation INPUT to this stage
      (= ``x_stages[stage_offset]``); ``x_partner`` is the partner-side
      activation, also from the forward walk's stage-input snapshot (the
      forward walk's stage-input snapshot has the partner at the same lane in
      the partner's tile position).
    * 4 contributions per pair: ``dt_ij_contrib = g_*_eff * x_*_eff``.
    * Per-program reduce: ``tl.sum(tl.reshape(dt_ij_contrib, (n_pairs_in_tile,
      2*stride)), axis=1)`` collapses each pair_flat's ``2*stride``
      contributions into one scalar per pair.
    * 4 ``tl.atomic_add(..., sem='relaxed')`` per stage (D-50a) write the
      per-pair sums into the fp32 scratch at the same pair_flat offsets the
      forward kernel uses (twiddle_ptr arithmetic is reused VERBATIM).

    fp32 atomic-add ordering noise per RESEARCH §"Per-Program tl.sum Reduce
    Semantics": with the per-program reduce factor of ``2*stride``, the
    accumulated noise at batch=4096 is bounded by ``~sqrt(batch * n / TILE_N)
    * 1e-7 ~ 2e-6`` per twiddle slot — well within the D-52
    ``rtol=1e-3, atol=1e-4`` envelope.
    """
    # Plan 08-02 lit up IS_COMPLEX=True — dispatch on constexpr below. No
    # static_assert gate.

    row_id = tl.program_id(axis=0)
    bn_id = tl.program_id(axis=1)
    nstack_idx = bn_id % nstacks

    tile_offsets = tl.arange(0, TILE_N)
    col_start = row_id * TILE_N
    pos = col_start + tile_offsets

    # Per-dtype pointer arithmetic. For fp32 (real): row stride is ``n``;
    # each twiddle entry occupies 1 float. For complex64 (view_as_real
    # flatten — D-50b): row stride is ``2 * n``; each twiddle entry occupies
    # 2 floats (re, im). Mirrors the same dual-stride pattern Phase 7's
    # forward kernel uses at op.py:179-189.
    if IS_COMPLEX:
        row_base = bn_id * (2 * n)
        twiddle_stack_stride = nblocks * LOG_N * 2 * n * 2
        twiddle_block_stride = LOG_N * 2 * n * 2
        twiddle_stage_stride = 2 * n * 2  # (n // 2) * 4 * 2
    else:
        row_base = bn_id * n
        twiddle_stack_stride = nblocks * LOG_N * 2 * n
        twiddle_block_stride = LOG_N * 2 * n
        twiddle_stage_stride = 2 * n  # (n // 2) * 4
    twiddle_sb_base = nstack_idx * twiddle_stack_stride + block_idx * twiddle_block_stride

    # ----------------------------------------------------------------------
    # Forward recompute walk: materialize x_stages[0..STAGE_COUNT-1] in
    # registers so the reverse walk can use the per-stage activation INPUT
    # without re-reading the trail. trail_in_ptr is the START-of-group
    # activation (= input to forward stage STAGE_START). Use grad_out_ptr as
    # the partner-exchange scratch buffer — we'll overwrite it with the
    # incoming gradient g after the forward walk completes.
    # ----------------------------------------------------------------------
    if IS_COMPLEX:
        # Load both re and im components separately via stride-2 access.
        pos2 = pos * 2
        x_loaded_re = tl.load(trail_in_ptr + row_base + pos2)
        x_loaded_im = tl.load(trail_in_ptr + row_base + pos2 + 1)
        tl.store(grad_out_ptr + row_base + pos2, x_loaded_re)
        tl.store(grad_out_ptr + row_base + pos2 + 1, x_loaded_im)
    else:
        x_loaded = tl.load(trail_in_ptr + row_base + pos)
        tl.store(grad_out_ptr + row_base + pos, x_loaded)
    tl.debug_barrier()

    # x_stages will be filled per stage. With STAGE_COUNT ≤ 3 we materialize
    # at most 3 register tiles; the constexpr loop unrolling makes unused
    # tiles dead-code-eliminated.
    if IS_COMPLEX:
        x_stages_0_re = x_loaded_re
        x_stages_0_im = x_loaded_im
        x_stages_1_re = x_loaded_re
        x_stages_1_im = x_loaded_im
        x_stages_2_re = x_loaded_re
        x_stages_2_im = x_loaded_im
        xp_stages_0_re = x_loaded_re
        xp_stages_0_im = x_loaded_im
        xp_stages_1_re = x_loaded_re
        xp_stages_1_im = x_loaded_im
        xp_stages_2_re = x_loaded_re
        xp_stages_2_im = x_loaded_im
        # Provide real-path placeholders to satisfy single-binding rules even
        # in the dead branch (Triton's analysis still walks both branches).
        x_stages_0 = x_loaded_re
        x_stages_1 = x_loaded_re
        x_stages_2 = x_loaded_re
        xp_stages_0 = x_loaded_re
        xp_stages_1 = x_loaded_re
        xp_stages_2 = x_loaded_re
    else:
        x_stages_0 = x_loaded
        x_stages_1 = x_loaded
        x_stages_2 = x_loaded
        xp_stages_0 = x_loaded
        xp_stages_1 = x_loaded
        xp_stages_2 = x_loaded
        # Placeholders for the complex-path branch (dead under IS_COMPLEX=False).
        x_stages_0_re = x_loaded
        x_stages_0_im = x_loaded
        x_stages_1_re = x_loaded
        x_stages_1_im = x_loaded
        x_stages_2_re = x_loaded
        x_stages_2_im = x_loaded
        xp_stages_0_re = x_loaded
        xp_stages_0_im = x_loaded
        xp_stages_1_re = x_loaded
        xp_stages_1_im = x_loaded
        xp_stages_2_re = x_loaded
        xp_stages_2_im = x_loaded

    # Walk forward stages 0..STAGE_COUNT-1 to materialize per-stage activations.
    # The output of stage s is the INPUT to stage s+1 in forward direction —
    # so for backward we need x_stages[s] = INPUT to stage s.
    # We compute the new x after each forward stage and store it back to scratch.
    #
    # Note on annotation: Triton's tl.static_range unrolls at JIT time but
    # variables declared with explicit ``: tl.constexpr`` inside the loop
    # body are treated as single-bindings that cannot be reassigned across
    # iterations. We use plain assignments; Triton infers constexpr-ness from
    # the constexpr inputs (STAGE_START + stage_offset_fw is constexpr because
    # both operands are).
    for stage_offset_fw in tl.static_range(STAGE_COUNT):
        idx_fw = STAGE_START + stage_offset_fw
        if INCREASING_STRIDE:
            log_stride_fw = idx_fw
        else:
            log_stride_fw = LOG_N - 1 - idx_fw
        stride_fw = 1 << log_stride_fw
        tile_partner_fw = tile_offsets ^ stride_fw
        pair_flat_fw = (col_start >> 1) + (tile_offsets // (2 * stride_fw)) * stride_fw \
            + (tile_offsets % stride_fw)
        is_lower_fw = (tile_offsets & stride_fw) == 0
        twiddle_stage_base_fw = twiddle_sb_base + idx_fw * twiddle_stage_stride

        if IS_COMPLEX:
            pf8_fw = pair_flat_fw * 8
            t00_re_fw = tl.load(twiddle_ptr + twiddle_stage_base_fw + pf8_fw + 0)
            t00_im_fw = tl.load(twiddle_ptr + twiddle_stage_base_fw + pf8_fw + 1)
            t01_re_fw = tl.load(twiddle_ptr + twiddle_stage_base_fw + pf8_fw + 2)
            t01_im_fw = tl.load(twiddle_ptr + twiddle_stage_base_fw + pf8_fw + 3)
            t10_re_fw = tl.load(twiddle_ptr + twiddle_stage_base_fw + pf8_fw + 4)
            t10_im_fw = tl.load(twiddle_ptr + twiddle_stage_base_fw + pf8_fw + 5)
            t11_re_fw = tl.load(twiddle_ptr + twiddle_stage_base_fw + pf8_fw + 6)
            t11_im_fw = tl.load(twiddle_ptr + twiddle_stage_base_fw + pf8_fw + 7)

            pos2 = pos * 2
            partner_pos2 = (col_start + tile_partner_fw) * 2
            cur_re_fw = tl.load(grad_out_ptr + row_base + pos2)
            cur_im_fw = tl.load(grad_out_ptr + row_base + pos2 + 1)
            partner_re_fw = tl.load(grad_out_ptr + row_base + partner_pos2)
            partner_im_fw = tl.load(grad_out_ptr + row_base + partner_pos2 + 1)

            # Snapshot the activation INPUT to this stage (re/im).
            if stage_offset_fw == 0:
                x_stages_0_re = cur_re_fw
                x_stages_0_im = cur_im_fw
                xp_stages_0_re = partner_re_fw
                xp_stages_0_im = partner_im_fw
            if stage_offset_fw == 1:
                x_stages_1_re = cur_re_fw
                x_stages_1_im = cur_im_fw
                xp_stages_1_re = partner_re_fw
                xp_stages_1_im = partner_im_fw
            if stage_offset_fw == 2:
                x_stages_2_re = cur_re_fw
                x_stages_2_im = cur_im_fw
                xp_stages_2_re = partner_re_fw
                xp_stages_2_im = partner_im_fw

            # Forward stage math (verbatim from _butterfly_kernel IS_COMPLEX branch):
            # new_lower = t00*cur + t01*partner ; new_upper = t10*partner + t11*cur
            # 4-FMA: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            t00_cur_re = t00_re_fw * cur_re_fw - t00_im_fw * cur_im_fw
            t00_cur_im = t00_re_fw * cur_im_fw + t00_im_fw * cur_re_fw
            t01_partner_re = t01_re_fw * partner_re_fw - t01_im_fw * partner_im_fw
            t01_partner_im = t01_re_fw * partner_im_fw + t01_im_fw * partner_re_fw
            t10_partner_re = t10_re_fw * partner_re_fw - t10_im_fw * partner_im_fw
            t10_partner_im = t10_re_fw * partner_im_fw + t10_im_fw * partner_re_fw
            t11_cur_re = t11_re_fw * cur_re_fw - t11_im_fw * cur_im_fw
            t11_cur_im = t11_re_fw * cur_im_fw + t11_im_fw * cur_re_fw
            new_lower_re = t00_cur_re + t01_partner_re
            new_lower_im = t00_cur_im + t01_partner_im
            new_upper_re = t10_partner_re + t11_cur_re
            new_upper_im = t10_partner_im + t11_cur_im
            new_x_re_fw = tl.where(is_lower_fw, new_lower_re, new_upper_re)
            new_x_im_fw = tl.where(is_lower_fw, new_lower_im, new_upper_im)
            tl.debug_barrier()
            tl.store(grad_out_ptr + row_base + pos2, new_x_re_fw)
            tl.store(grad_out_ptr + row_base + pos2 + 1, new_x_im_fw)
            tl.debug_barrier()
        else:
            pf4_fw = pair_flat_fw * 4
            t00_fw = tl.load(twiddle_ptr + twiddle_stage_base_fw + pf4_fw + 0)
            t01_fw = tl.load(twiddle_ptr + twiddle_stage_base_fw + pf4_fw + 1)
            t10_fw = tl.load(twiddle_ptr + twiddle_stage_base_fw + pf4_fw + 2)
            t11_fw = tl.load(twiddle_ptr + twiddle_stage_base_fw + pf4_fw + 3)

            cur_fw = tl.load(grad_out_ptr + row_base + pos)
            partner_fw = tl.load(grad_out_ptr + row_base + (col_start + tile_partner_fw))

            # Snapshot the activation INPUT to this stage (and its partner).
            if stage_offset_fw == 0:
                x_stages_0 = cur_fw
                xp_stages_0 = partner_fw
            if stage_offset_fw == 1:
                x_stages_1 = cur_fw
                xp_stages_1 = partner_fw
            if stage_offset_fw == 2:
                x_stages_2 = cur_fw
                xp_stages_2 = partner_fw

            # Forward stage math (verbatim from _butterfly_kernel, real path)
            new_lower_fw = t00_fw * cur_fw + t01_fw * partner_fw
            new_upper_fw = t10_fw * partner_fw + t11_fw * cur_fw
            new_x_fw = tl.where(is_lower_fw, new_lower_fw, new_upper_fw)

            tl.debug_barrier()
            tl.store(grad_out_ptr + row_base + pos, new_x_fw)
            tl.debug_barrier()

    # ----------------------------------------------------------------------
    # Overwrite grad_out_ptr scratch with the INCOMING gradient (g flowing
    # from later stages into this stage group). Then begin the reverse walk.
    # ----------------------------------------------------------------------
    if IS_COMPLEX:
        pos2 = pos * 2
        g_re = tl.load(grad_in_ptr + row_base + pos2)
        g_im = tl.load(grad_in_ptr + row_base + pos2 + 1)
        tl.debug_barrier()
        tl.store(grad_out_ptr + row_base + pos2, g_re)
        tl.store(grad_out_ptr + row_base + pos2 + 1, g_im)
        tl.debug_barrier()
        # Provide real-path placeholder.
        g = g_re
    else:
        g = tl.load(grad_in_ptr + row_base + pos)
        tl.debug_barrier()
        tl.store(grad_out_ptr + row_base + pos, g)
        tl.debug_barrier()
        # Placeholders for the complex-path branch (dead under IS_COMPLEX=False).
        g_re = g
        g_im = g

    # ----------------------------------------------------------------------
    # Reverse stage walk: walk stages STAGE_COUNT-1 down to 0 via hand-unrolled
    # calls to _backward_one_stage (real) or _backward_one_stage_complex
    # (complex). Each call uses the appropriate constexpr STRIDE so tl.reshape
    # inside the helper has a literal shape (D-50a).
    # ----------------------------------------------------------------------

    # Per-stage STRIDE (= 1 << log_stride) literals are now computed on the
    # HOST at the backward launch site and passed in as the STRIDE_0/1/2
    # constexpr kernel args. They used to be derived here via
    #   LOG_STRIDE_k = max(..., 0) if STAGE_COUNT ... else 0 ; STRIDE_k = 1 << LOG_STRIDE_k
    # but Triton 3.3.x's front-end cannot evaluate the Python ``max(...)``
    # builtin / ``... if ... else ...`` ternary inside a @triton.jit body and
    # raises ``ValueError('Did you forget to add @triton.jit ?')`` at the
    # ``1 << LOG_STRIDE_0`` line. Every input to those expressions
    # (STAGE_START, STAGE_COUNT, INCREASING_STRIDE, LOG_N) is a host-known
    # constexpr, so the whole computation is hoisted out of the kernel — robust
    # across Triton versions and mirroring how tile_n / num_warps are already
    # host-computed. Unused slots (STAGE_COUNT < 2 / < 3) are clamped to 1 on
    # the host; the surrounding ``if STAGE_COUNT >= 2`` / ``== 3`` guards
    # dead-code-eliminate the corresponding helper calls anyway.
    #
    # IDX_0/1/2 stay in-kernel: plain ``STAGE_START + k`` constexpr addition is
    # front-end-safe (the forward kernel uses the same form at op.py:230).
    IDX_0: tl.constexpr = STAGE_START + 0
    IDX_1: tl.constexpr = STAGE_START + 1
    IDX_2: tl.constexpr = STAGE_START + 2

    if IS_COMPLEX:
        # Reverse walk: highest stage first (complex64 path with conjugate-4-FMA).
        if STAGE_COUNT == 3:
            twiddle_stage_base_2 = twiddle_sb_base + IDX_2 * twiddle_stage_stride
            twiddle_stage_base_ptr_2 = twiddle_ptr + twiddle_stage_base_2
            d_twiddle_stage_base_ptr_2 = d_twiddle_scratch_ptr + twiddle_stage_base_2
            g_re, g_im = _backward_one_stage_complex(
                twiddle_stage_base_ptr_2, grad_out_ptr, d_twiddle_stage_base_ptr_2,
                x_stages_2_re, x_stages_2_im, xp_stages_2_re, xp_stages_2_im,
                g_re, g_im, row_base, col_start, tile_offsets, pos,
                STRIDE=STRIDE_2, TILE_N=TILE_N,
            )
        if STAGE_COUNT >= 2:
            twiddle_stage_base_1 = twiddle_sb_base + IDX_1 * twiddle_stage_stride
            twiddle_stage_base_ptr_1 = twiddle_ptr + twiddle_stage_base_1
            d_twiddle_stage_base_ptr_1 = d_twiddle_scratch_ptr + twiddle_stage_base_1
            g_re, g_im = _backward_one_stage_complex(
                twiddle_stage_base_ptr_1, grad_out_ptr, d_twiddle_stage_base_ptr_1,
                x_stages_1_re, x_stages_1_im, xp_stages_1_re, xp_stages_1_im,
                g_re, g_im, row_base, col_start, tile_offsets, pos,
                STRIDE=STRIDE_1, TILE_N=TILE_N,
            )
        if STAGE_COUNT >= 1:
            twiddle_stage_base_0 = twiddle_sb_base + IDX_0 * twiddle_stage_stride
            twiddle_stage_base_ptr_0 = twiddle_ptr + twiddle_stage_base_0
            d_twiddle_stage_base_ptr_0 = d_twiddle_scratch_ptr + twiddle_stage_base_0
            g_re, g_im = _backward_one_stage_complex(
                twiddle_stage_base_ptr_0, grad_out_ptr, d_twiddle_stage_base_ptr_0,
                x_stages_0_re, x_stages_0_im, xp_stages_0_re, xp_stages_0_im,
                g_re, g_im, row_base, col_start, tile_offsets, pos,
                STRIDE=STRIDE_0, TILE_N=TILE_N,
            )
    else:
        # Reverse walk: highest stage first (fp32 path, Plan 08-01).
        if STAGE_COUNT == 3:
            twiddle_stage_base_2 = twiddle_sb_base + IDX_2 * twiddle_stage_stride
            twiddle_stage_base_ptr_2 = twiddle_ptr + twiddle_stage_base_2
            d_twiddle_stage_base_ptr_2 = d_twiddle_scratch_ptr + twiddle_stage_base_2
            g = _backward_one_stage(
                twiddle_stage_base_ptr_2, grad_out_ptr, d_twiddle_stage_base_ptr_2,
                x_stages_2, xp_stages_2, g, row_base, col_start, tile_offsets, pos,
                STRIDE=STRIDE_2, TILE_N=TILE_N,
            )
        if STAGE_COUNT >= 2:
            twiddle_stage_base_1 = twiddle_sb_base + IDX_1 * twiddle_stage_stride
            twiddle_stage_base_ptr_1 = twiddle_ptr + twiddle_stage_base_1
            d_twiddle_stage_base_ptr_1 = d_twiddle_scratch_ptr + twiddle_stage_base_1
            g = _backward_one_stage(
                twiddle_stage_base_ptr_1, grad_out_ptr, d_twiddle_stage_base_ptr_1,
                x_stages_1, xp_stages_1, g, row_base, col_start, tile_offsets, pos,
                STRIDE=STRIDE_1, TILE_N=TILE_N,
            )
        if STAGE_COUNT >= 1:
            twiddle_stage_base_0 = twiddle_sb_base + IDX_0 * twiddle_stage_stride
            twiddle_stage_base_ptr_0 = twiddle_ptr + twiddle_stage_base_0
            d_twiddle_stage_base_ptr_0 = d_twiddle_scratch_ptr + twiddle_stage_base_0
            g = _backward_one_stage(
                twiddle_stage_base_ptr_0, grad_out_ptr, d_twiddle_stage_base_ptr_0,
                x_stages_0, xp_stages_0, g, row_base, col_start, tile_offsets, pos,
                STRIDE=STRIDE_0, TILE_N=TILE_N,
            )

    # Final g (post-reverse-walk) is the d_input for the stage-group's input;
    # it already lives in grad_out_ptr (last store inside _backward_one_stage*).


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
            the butterfly factor stack. Must be float32 or complex64; mixed
            dtypes (twiddle != input dtype) are rejected.
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
          in-place data dependencies. Plan 08-01 factors the launch loop
          into ``_run_forward_stage_groups(trail_out=None)`` so the backward
          callback can reuse it with ``trail_out`` redirected to a recompute
          trail buffer (D-49a).
        * **Dtype gate (D-41):** Plan 07-02 lit up complex64 — the wrapper now
          accepts ``{float32, complex64}`` and routes complex64 through the
          ``view_as_real`` boundary into the IS_COMPLEX=True kernel branch.
    """
    # Wrapper-boundary preconditions (CLAUDE.md "Error Handling": assert for
    # preconditions). Pitfall 3 (contiguity) must hold before any view_as_real.
    assert input.dim() == 3, (
        f"input must be (batch, nstacks, input_size), got dim={input.dim()}"
    )
    assert twiddle.dtype == input.dtype, (
        f"twiddle.dtype ({twiddle.dtype}) must equal input.dtype ({input.dtype})"
    )
    # Pitfall 3 (contiguity): the kernel + view_as_real boundary require
    # contiguous inputs. Rather than asserting (which crashed on legitimate
    # callers — e.g. Butterfly.forward passes a stride-0 .expand() view when
    # nstacks > 1, i.e. out_size > in_size), COERCE here. A no-op when already
    # contiguous; a contiguous copy of the same data otherwise. This also
    # removes the CPU/CUDA divergence (the torch oracle already tolerated
    # non-contiguous input). _setup_context saves contiguous copies so the
    # autograd backward inherits the same guarantee. (bug torch-structured-7ny)
    input = input.contiguous()
    twiddle = twiddle.contiguous()
    # Phase 7 D-41 dtype gate. Plan 07-02 broadened the gate from fp32-only to
    # {float32, complex64} once the IS_COMPLEX=True kernel branch was lit up.
    assert input.dtype in (torch.float32, torch.complex64), (
        f"butterfly_multiply supports fp32 and complex64 only; got {input.dtype}"
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
    # Active for complex64 inputs; bypassed for fp32. The Pitfall 3 contiguity
    # asserts above are LOAD-BEARING for this path — view_as_real on a
    # non-contiguous complex tensor produces incorrect strides.
    is_complex = input.is_complex()
    if is_complex:
        input_work = torch.view_as_real(input).contiguous()
        twiddle_work = torch.view_as_real(twiddle).contiguous()
    else:
        input_work = input
        twiddle_work = twiddle

    # D-49a: launch loop factored into _run_forward_stage_groups; behavior is
    # byte-equivalent to Phase 7's original inlined wrapper when trail_out is
    # None.
    final_work = _run_forward_stage_groups(
        twiddle_work,
        input_work,
        increasing_stride,
        log_n,
        n,
        nstacks,
        nblocks,
        batch_size,
        is_complex,
        trail_out=None,
    )
    if is_complex:
        final_output_full = torch.view_as_complex(final_work.contiguous())
    else:
        final_output_full = final_work

    # D-42 output_size trim (mirrors _torch_ref/butterfly.py:33).
    return final_output_full[:, :, :output_size_actual]


def _setup_context(ctx, inputs, output):
    """Save twiddle, input, and the two non-tensor flags for the two-input
    register_autograd backward (D-47, D-57 UNCHANGED in Plan 08-01).

    The backward delegates to ``_butterfly_multiply_torch`` for the small-N
    branch (D-49b) and runs the Triton-backed reverse walk for the main path
    (D-49/D-50). _setup_context is UNCHANGED from Phase 7 per D-57 — same
    saved tensors, same scalar attributes.
    """
    twiddle, input_, increasing_stride, output_size = inputs
    # Save contiguous copies so the Triton backward's view_as_real boundary
    # (Pitfall 3 asserts at _backward) inherits the same contiguity guarantee
    # the forward now coerces — setup_context receives the ORIGINAL op args,
    # which may be a non-contiguous .expand() view (bug torch-structured-7ny).
    ctx.save_for_backward(twiddle.contiguous(), input_.contiguous())
    ctx.increasing_stride = increasing_stride
    ctx.output_size = output_size


def _backward(ctx, grad_out):
    """Plan 08-01 (fp32) + Plan 08-02 (complex64) Triton-backed two-input
    backward (D-49/D-50/D-50a/D-50b/D-50c/D-57).

    Replaces Phase 7's oracle delegation with:

      1. small-N fallback (log_n <= 1 — D-49b inheritance via the same
         torch.autograd.grad over _butterfly_multiply_torch that Phase 7 used).
      2. broader dtype gate accepting {float32, complex64} (Plan 08-02 lit up
         the complex64 path — the Plan 08-01 fp32-only assert is removed).
      3. pad grad_out + input from {output_size, input_size} up to n
         (mirror forward pad logic).
      4. allocate fp32 trail buffer at STAGE-GROUP granularity:
         ``(ceil(log_n / 3) * nblocks, batch_size, nstacks, trail_n)`` where
         ``trail_n = n * 2 if is_complex else n`` (view_as_real flatten for
         complex64 per D-50b). At the largest case (log_n=11, nblocks=2,
         batch=4096, nstacks=1, n=2048) the peak trail buffer is ~256 MB
         fp32 for the real path / ~512 MB for the complex64 path (RESEARCH
         correction #2). Acceptable for an 8GB VRAM dev host and is the
         documented memory cost of the recompute-then-walk-back strategy
         (the rejected save-during-forward alternative cost ~720 MB).
      5. recompute forward into trail via _run_forward_stage_groups(trail_out=trail).
      6. allocate fp32 d_twiddle_scratch: shape ``twiddle.shape`` for fp32,
         ``twiddle.shape + (2,)`` for complex64 (the trailing-2 axis is the
         re/im split for view_as_real flatten per D-50b — same shape as
         ``view_as_real(twiddle)``).
      7. walk reverse stage-groups + reverse nblocks, ping-pong between two
         d_input buffers per launch:
           - kernel reads trail[launch_idx], src_grad, twiddle_work
           - kernel writes dst_grad (= upstream grad for next reverse group)
           - kernel atomic_adds into d_twiddle_scratch (4 atomics/stage for
             fp32, 8 atomics/stage for complex64)
      8. cast fp32 d_twiddle_scratch back to twiddle.dtype at the callback
         boundary (D-50a):
           - fp32: ``d_twiddle_scratch.to(twiddle.dtype)`` (no-op for fp32).
           - complex64: ``torch.view_as_complex(d_twiddle_scratch.contiguous())``
             per D-50b view_as_real boundary template.
      9. trim d_input back to input_size.
      10. return (d_twiddle, d_input, None, None) matching the 4 forward inputs
          (twiddle, input, increasing_stride, output_size); last two are
          None because they are non-tensor (bool, Optional[int]).

    Complex64 conjugate-4-FMA (D-50c + RESEARCH correction #3): the kernel's
    IS_COMPLEX=True branch implements
      - d_twiddle = ``grad * conj(input)`` (sign-flip: PLUS in real, MINUS in imag)
      - d_input  = ``conj(twiddle).T @ grad`` (same sign-flip — RESEARCH
        correction #3 is the d_input conjugate-on-twiddle that fp32 tests
        silently pass but complex64 d_input parity catches).
    """
    # Phase 9 D-63a: wrapper-level deterministic gate. When
    # set_deterministic(True) OR torch.use_deterministic_algorithms(True) is
    # active (D-63b additive OR), route through the pure-PyTorch oracle —
    # deterministic by virtue of having no atomicAdd reordering. Identical
    # body shape to the small-N fallback below (lines mirror the D-49b
    # template verbatim). Local-scoped import avoids a module-import cycle:
    # _ops.py imports from _triton for the resolver but the resolver itself
    # does not need the deterministic helper.
    from torch_structured._ops import _is_deterministic_mode_active
    twiddle, input_ = ctx.saved_tensors
    increasing_stride = ctx.increasing_stride
    output_size = ctx.output_size
    if _is_deterministic_mode_active():
        twiddle_d = twiddle.detach().requires_grad_(True)
        input_d = input_.detach().requires_grad_(True)
        with torch.enable_grad():
            out = _butterfly_multiply_torch(twiddle_d, input_d, increasing_stride, output_size)
        gt, gi = torch.autograd.grad(out, [twiddle_d, input_d], grad_out)
        return gt, gi, None, None

    batch_size, nstacks, input_size = input_.shape
    nblocks = twiddle.shape[1]
    log_n = twiddle.shape[2]
    n = 1 << log_n
    is_complex = twiddle.is_complex()

    # D-49b small-N fallback (mirror Phase 7 wrapper's log_n <= 1 branch via
    # torch.autograd.grad over _butterfly_multiply_torch). Triton launch
    # overhead and the trail/scratch machinery dominate at trivial sizes.
    if log_n <= 1:
        twiddle_d = twiddle.detach().requires_grad_(True)
        input_d = input_.detach().requires_grad_(True)
        with torch.enable_grad():
            out = _butterfly_multiply_torch(twiddle_d, input_d, increasing_stride, output_size)
        gt, gi = torch.autograd.grad(out, [twiddle_d, input_d], grad_out)
        return gt, gi, None, None

    # Plan 08-02 broader dtype gate — accepts {float32, complex64} (the
    # fp32-only assert from Plan 08-01 has been removed).
    assert twiddle.dtype in (torch.float32, torch.complex64) and input_.dtype == twiddle.dtype, (
        f"butterfly_multiply backward supports fp32 and complex64; got "
        f"twiddle.dtype={twiddle.dtype}, input.dtype={input_.dtype}"
    )

    # output_size defaults to n at the callback boundary (the wrapper enforces
    # output_size_actual <= n; here we materialize the int form).
    output_size_actual = n if output_size is None else output_size

    # Pad input from input_size up to n (mirror forward op.py:408-410).
    input_padded = (
        F.pad(input_, (0, n - input_size)) if input_size < n else input_[:, :, :n]
    ).contiguous()

    # RESEARCH correction #2: stage-group granularity, NOT per-stage. The
    # forward kernel produces one output per launch (= per stage GROUP), not
    # per individual stage; trail slots match the launch granularity.
    # Memory cost at the largest case (log_n=11, nblocks=2, batch=4096,
    # nstacks=1, n=2048) is ceil(11/3) * 2 * 4096 * 1 * 2048 * 4 bytes ≈
    # 256 MB fp32 — the documented memory cost of the recompute-then-walk-back
    # strategy (the rejected save-during-forward alternative cost ~720 MB).
    # For complex64 the trailing axis doubles (view_as_real flatten — D-50b):
    # trail_n = n * 2, peak ~512 MB at the same largest case.
    trail_n = n * 2 if is_complex else n
    n_launches_per_nblock = (log_n + 2) // 3
    trail = torch.empty(
        n_launches_per_nblock * nblocks, batch_size, nstacks, trail_n,
        dtype=torch.float32, device=input_padded.device,
    )

    # fp32 d_twiddle scratch (SC#3 — atomic_add into fp32 only; final cast
    # at callback boundary, NEVER inside the kernel). For complex64 the
    # scratch has a trailing-2 axis for the re/im split per D-50b — same shape
    # as ``view_as_real(twiddle)`` so we can cast back with view_as_complex.
    scratch_shape = (*twiddle.shape, 2) if is_complex else twiddle.shape
    d_twiddle_scratch = torch.zeros(
        scratch_shape, dtype=torch.float32, device=twiddle.device,
    )

    # Activate view_as_real machinery for complex64 inputs per D-50b.
    # Pitfall 3 (04-COMPLEX-LAYOUT.md:78-95): view_as_real on a non-contiguous
    # complex tensor silently reads garbage; the asserts below are load-bearing.
    if is_complex:
        assert input_padded.is_contiguous(), \
            "input must be contiguous for view_as_real (Pitfall 3)"
        assert twiddle.is_contiguous(), \
            "twiddle must be contiguous for view_as_real (Pitfall 3)"
        twiddle_work = torch.view_as_real(twiddle).contiguous()
        input_work = torch.view_as_real(input_padded).contiguous()
    else:
        twiddle_work = twiddle
        input_work = input_padded

    # Recompute forward into trail (D-49a reuse of Phase 7 launches).
    _run_forward_stage_groups(
        twiddle_work,
        input_work,
        increasing_stride,
        log_n,
        n,
        nstacks,
        nblocks,
        batch_size,
        is_complex,
        trail_out=trail,
    )

    # Pad grad_out from output_size up to n (mirror forward input pad logic).
    grad_full = (
        F.pad(grad_out, (0, n - output_size_actual)) if output_size_actual < n else grad_out
    ).contiguous()

    # Ping-pong d_input buffers (mirror Phase 7 forward ping-pong; the
    # backward walk consumes src_grad and produces dst_grad each launch).
    # For complex64, the kernel operates on view_as_real-flattened buffers;
    # we keep the complex64 buffers for the final return but pass their
    # view_as_real views into the kernel.
    d_input_buf_a = torch.empty(
        batch_size, nstacks, n, dtype=input_.dtype, device=input_.device,
    )
    d_input_buf_b = torch.empty_like(d_input_buf_a)
    if is_complex:
        assert grad_full.is_contiguous(), \
            "grad_out must be contiguous for view_as_real (Pitfall 3)"
        grad_work = torch.view_as_real(grad_full).contiguous()
        d_input_buf_a_work = torch.view_as_real(d_input_buf_a)
        d_input_buf_b_work = torch.view_as_real(d_input_buf_b)
    else:
        grad_work = grad_full
        d_input_buf_a_work = d_input_buf_a
        d_input_buf_b_work = d_input_buf_b
    d_input_buf_a_work.copy_(grad_work)
    src_grad = d_input_buf_a_work
    dst_grad = d_input_buf_b_work

    # Initialize cur_increasing_stride to the value used by the LAST forward
    # block (which is what the last forward launch used — the value we begin
    # the reverse walk with). The forward toggles `cur_increasing_stride =
    # not cur_increasing_stride` at the end of each nblock iteration; after
    # nblocks iterations starting from `increasing_stride`, the value at the
    # START of the LAST nblock iteration is `increasing_stride` XOR (nblocks
    # - 1) parity.
    cur_increasing_stride = increasing_stride
    for _ in range(nblocks - 1):
        cur_increasing_stride = not cur_increasing_stride

    # Walk reverse stage-groups + reverse nblocks (D-50). launch_idx_global
    # walks from the last forward launch backward to 0.
    launch_idx_global = n_launches_per_nblock * nblocks - 1
    for block in range(nblocks - 1, -1, -1):
        group_starts_reversed = list(range(0, log_n, 3))[::-1]
        for group_start in group_starts_reversed:
            counter_count = min(3, log_n - group_start)
            if cur_increasing_stride:
                max_log_stride = group_start + counter_count - 1
            else:
                max_log_stride = log_n - 1 - group_start
            tile_n = 1 << (max_log_stride + 1)
            n_row_tiles = n // tile_n
            grid = (n_row_tiles, batch_size * nstacks)
            num_warps = _pick_num_warps(tile_n)
            # Host-compute the per-stage STRIDE (= 1 << log_stride) constexprs
            # the kernel used to derive internally. Hoisting this out of the
            # @triton.jit body sidesteps the Triton 3.3.x front-end regression
            # on the max()/ternary constexpr pattern. Mirrors the kernel's old
            # formula verbatim; unused slots (counter_count < 2 / < 3) are
            # clamped to log_stride 0 so ``1 << ...`` never sees a negative
            # shift and the resulting STRIDE (=1) is inert (the kernel's
            # STAGE_COUNT guards drop those stages).
            if cur_increasing_stride:
                log_stride_0 = group_start + 0
                log_stride_1 = max(group_start + 1, 0) if counter_count >= 2 else 0
                log_stride_2 = max(group_start + 2, 0) if counter_count == 3 else 0
            else:
                log_stride_0 = log_n - 1 - group_start - 0
                log_stride_1 = max(log_n - 1 - group_start - 1, 0) if counter_count >= 2 else 0
                log_stride_2 = max(log_n - 1 - group_start - 2, 0) if counter_count == 3 else 0
            stride_0 = 1 << log_stride_0
            stride_1 = 1 << log_stride_1
            stride_2 = 1 << log_stride_2
            # IMPORTANT: trail[i] holds the OUTPUT of forward launch i = the
            # INPUT to forward launch i+1. For the backward of forward launch
            # `launch_idx_global`, we need the activation INPUT to that
            # forward launch — which is `trail[launch_idx_global - 1]` for
            # launch_idx_global > 0, or the original `input_work` (the
            # view_as_real form of input_padded for complex64; input_padded
            # itself for fp32) for launch_idx_global == 0 (the very first
            # forward launch). Using ``input_work`` (not the raw complex
            # tensor) is load-bearing for the complex64 path — the kernel
            # expects fp32 pointers per D-50b.
            if launch_idx_global == 0:
                trail_slot = input_work
            else:
                trail_slot = trail[launch_idx_global - 1]
            wrap_triton(_butterfly_backward_kernel)[grid](
                twiddle_work,
                trail_slot,
                src_grad,
                dst_grad,
                d_twiddle_scratch,
                n,
                nstacks,
                block,
                nblocks,
                STAGE_START=group_start,
                STAGE_COUNT=counter_count,
                INCREASING_STRIDE=cur_increasing_stride,
                LOG_N=log_n,
                IS_COMPLEX=is_complex,
                TILE_N=tile_n,
                STRIDE_0=stride_0,
                STRIDE_1=stride_1,
                STRIDE_2=stride_2,
                num_warps=num_warps,
            )
            src_grad, dst_grad = dst_grad, src_grad
            launch_idx_global -= 1
        # End-of-block: toggle direction for the next (earlier) nblock.
        cur_increasing_stride = not cur_increasing_stride

    # After loop, src_grad holds the final d_input view (last swap put dst -> src).
    # For complex64, src_grad is a view_as_real form pointing at one of the
    # two complex64 buffers; we recover the original complex64 buffer to
    # return.
    if is_complex:
        # src_grad is one of (d_input_buf_a_work, d_input_buf_b_work); recover
        # the underlying complex64 buffer.
        if src_grad is d_input_buf_a_work:
            d_input_full = d_input_buf_a
        else:
            d_input_full = d_input_buf_b
        # fp32 scratch -> complex64 cast via view_as_complex (D-50b). The
        # contiguous() guards against any stride mismatch from
        # torch.zeros((..., 2)) — defensive though zeros() is already
        # contiguous; matches RESEARCH §"d_twiddle_scratch Shape + Offsets"
        # reverse-pattern.
        d_twiddle = torch.view_as_complex(d_twiddle_scratch.contiguous())
    else:
        d_input_full = src_grad
        # fp32 scratch -> twiddle.dtype cast at the callback boundary (D-50a).
        # NEVER cast inside the kernel; always at the boundary.
        d_twiddle = d_twiddle_scratch.to(twiddle.dtype)

    # D-42 d_input trim back to input_size (mirror forward output_size trim).
    d_input_out = d_input_full[:, :, :input_size]

    return d_twiddle, d_input_out, None, None


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
