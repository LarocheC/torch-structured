# Phase 8: butterfly_multiply Backward (Triton) - Pattern Map

**Mapped:** 2026-05-28
**Files analyzed:** 3 modified (1 op, 1 test, 1 baseline-data)
**Analogs found:** 3 / 3 (all role-match against in-repo Phase 5/6/7 deliverables; the CUDA legacy kernel provides the algorithmic blueprint for the new backward body)

Phase 8 is the **backward-direction completion** of the Phase 7 butterfly_multiply Triton port. Every file Phase 8 touches **already exists** — this phase modifies bodies, never creates new files. The infrastructure surface (`_ops.py` resolver, `register_autograd` registration line, `_setup_context`, `register_fake`, `_cuda_legacy/butterfly.py`, `tests/conftest.py`, the `_triton/butterfly/__init__.py` marker) is **all unchanged**. Phase 8's authoring surface collapses to **one body replacement + helper factor-out** in `op.py`, **test additions** in `test_butterfly_triton.py`, and **schema extension** in `07-BASELINE.json` (extended in-place, not relocated).

The key algorithmic move is *replacing* the Phase 7 `_backward` body (which delegates to `torch.autograd.grad(_butterfly_multiply_torch(...))`) with a Triton-native backward that (a) factors Phase 7's forward stage-group launch loop into `_run_forward_stage_groups(..., trail_out=None)` per D-49a, (b) re-runs the forward into a trail buffer, (c) walks the stage groups in REVERSE order via a new `_butterfly_backward_kernel`, and (d) atomic-adds `d_twiddle` into an fp32 scratch with per-program `tl.sum` reduce per SC#3.

## File Classification

| Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---------------|------|-----------|----------------|---------------|
| `torch_structured/_triton/butterfly/op.py` | service / Triton kernel + autograd op | compute (GPU kernel; atomic-add reduction) + autograd (two-input gradient via Triton kernel) + meta (unchanged) | **Forward kernel template:** Phase 7's `_butterfly_kernel` at `op.py:77-321` (same launch shape; new kernel walks stages in reverse + adds atomic-add d_twiddle accumulation). **Wrapper template:** Phase 7's `butterfly_multiply` wrapper at `op.py:322-501` (stage-group launch loop is factored out per D-49a). **Backward body template:** Phase 7's `_backward` at `op.py:516-543` (Phase 8 replaces THIS body). **Two-input backward Wirtinger pattern:** Phase 5's `_triton/diag_mult/op.py:173-190` (closed-form Wirtinger with `.conj()`; Phase 8 evolves this to kernel-backed direct gradients per D-50a + D-50c). **CUDA algorithmic blueprint:** `csrc/cuda/butterfly_cuda.cu:421-489` (`b_untied_forward_backward_shared_twiddle` — forward+backward fused with per-step `d_twiddle += grad * conj(input)` + `gpuAtomicAdd`). | role-match — substantive new kernel body; structural skeleton (signature, IS_COMPLEX gating, view_as_real boundary, register_autograd 4-tuple return) transcribes verbatim from Phase 7 |
| `tests/test_butterfly_triton.py` | test | request-response (function call) + parametrized dual-path allclose | Phase 7's existing same-file structure (`tests/test_butterfly_triton.py:36-414`): module-level `RTOL/ATOL` constants, `pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), ...)`, parametrized smoke tier, `@pytest.mark.slow` comprehensive tier, fp64 gradcheck pattern with `pytest.skip` for triton backend. Phase 5's `tests/test_diag_mult.py:1-119` for the dual-path backward allclose comparison pattern (looser tolerance on the gradient). | exact — same file, same parametrize axes, same tiered structure (D-43a inheritance); Phase 8 only ADDS tests, never modifies existing ones |
| `.planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json` | config / data artifact | one-shot write (extended in place) | `07-BASELINE.json` itself (its current 8-row forward-only schema is the analog; Phase 8 EXTENDS with `direction` field + 8 backward rows) | exact — same file, same schema with one new field; Phase 8 keeps the JSON co-located with Phase 7 per the explicit CONTEXT discretion ("extend with backward p50/p95 entries" — NOT a new 08-BASELINE.json) |

## Pattern Assignments

### `torch_structured/_triton/butterfly/op.py` (service — Triton backward kernel + helper extract + body replace)

**Primary analogs (transcribed in combination):**
- **Forward kernel for mirror-with-reversal:** `torch_structured/_triton/butterfly/op.py:77-321` (Phase 7's `_butterfly_kernel`).
- **Wrapper stage-group launch loop for factor-out:** `torch_structured/_triton/butterfly/op.py:322-501` (Phase 7's `butterfly_multiply` wrapper).
- **`_backward` body to replace:** `torch_structured/_triton/butterfly/op.py:516-543` (Phase 7's oracle-delegating backward).
- **Two-input backward callback skeleton:** `torch_structured/_triton/diag_mult/op.py:173-190` (closed-form Wirtinger with `.conj()`; same return-tuple shape).
- **CUDA algorithmic reference:** `csrc/cuda/butterfly_cuda.cu:421-489` and host wrapper at `:535-606` (forward-recompute-then-backward orchestration; per-step `d_twiddle += grad * conj_wrapper(input)` + `gpuAtomicAdd`).

---

#### Imports pattern — UNCHANGED from Phase 7

**Source:** `op.py:58-65`

```python
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from torch.nn import functional as F
from typing import Optional

from torch_structured._torch_ref.butterfly import butterfly_multiply_torch as _butterfly_multiply_torch  # backward oracle (D-47, two-input via torch.autograd.grad) + small-N fallback (D-42a)
```

**Phase 8 change:** None. The `_butterfly_multiply_torch` import is now used ONLY for the small-N fallback (D-49b) and the gradcheck oracle reference (D-52b), not for the main backward path. The comment can be updated but the import line is unchanged.

---

#### Forward kernel pattern — UNCHANGED, REUSED VERBATIM for trail recompute

**Source to read but NOT modify:** `op.py:77-321` (the entire `_butterfly_kernel`).

**Why it matters:** Per D-49a, the recompute-into-trail path **reuses the existing `_butterfly_kernel` verbatim** — no kernel modification needed. The only change is the `output_ptr` argument the wrapper passes (a slice into the trail buffer instead of the ping-pong dst_buf). Phase 8 does NOT modify any line in `:77-321`.

**Critical excerpt for backward-kernel mirror (op.py:201-319 — per-stage walk pattern):**

```python
# Phase 7 forward — Phase 8 backward kernel MIRRORS this in reverse:
for stage_offset in tl.static_range(STAGE_COUNT):
    idx = STAGE_START + stage_offset  # COUNTER, not absolute stage
    if INCREASING_STRIDE:
        log_stride = idx
    else:
        log_stride = LOG_N - 1 - idx
    stride = 1 << log_stride
    tile_partner = tile_offsets ^ stride
    pair_flat = (col_start >> 1) + (tile_offsets // (2 * stride)) * stride \
        + (tile_offsets % stride)
    twiddle_stage_base = twiddle_sb_base + idx * twiddle_stage_stride
    is_lower = (tile_offsets & stride) == 0
    # ... load twiddle/cur/partner, compute 2x2 multiply, tl.store with barriers ...
```

**Backward-kernel adaptation (D-50 — same launch shape, REVERSE static_range, plus atomic-add d_twiddle):**

```python
# Phase 8 NEW backward kernel — walks stages in reverse:
for stage_offset in tl.static_range(STAGE_COUNT - 1, -1, -1):  # REVERSE
    idx = STAGE_START + stage_offset
    if INCREASING_STRIDE:
        log_stride = idx
    else:
        log_stride = LOG_N - 1 - idx
    stride = 1 << log_stride
    # ... identical partner / pair_flat / twiddle_stage_base / is_lower math ...
    # NEW: compute d_input via T^T @ g (or conj for IS_COMPLEX)
    # NEW: compute d_twiddle = g * input (or g * conj(input) for IS_COMPLEX)
    # NEW: per-program tl.sum reduce of d_twiddle across the row tile
    # NEW: single tl.atomic_add into d_twiddle_scratch_ptr per (t_00..t_11)
```

**Twiddle layout / pointer math — VERBATIM REUSE** (op.py:168-179, real path):

```python
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
```

The `d_twiddle_scratch_ptr` uses the **same offset arithmetic** because the scratch has the same shape as `twiddle` (with a trailing `(2,)` for complex64 via `view_as_real` flatten — D-50b). The `pair_flat * 4` (real) / `pair_flat * 8` (complex) per-pair stride from Phase 7 carries through verbatim.

---

#### Forward 4-FMA template (op.py:259-279) — SOURCE FOR THE D-50c CONJUGATE FLIP

**Phase 7 forward 4-FMA** (op.py:267-274):

```python
# (a + bi)(c + di) = (ac - bd) + (ad + bc)i
t00_cur_re = t00_re * cur_re - t00_im * cur_im
t00_cur_im = t00_re * cur_im + t00_im * cur_re
# ... four pairwise multiplies, two sums, tl.where on is_lower
```

**Phase 8 conjugate 4-FMA for d_twiddle (D-50c LANDMINE — sign-flip on BOTH terms):**

```python
# g * conj(input):  (a + bi)(c - di) = (ac + bd) + (bc - ad)i
# SIGN FLIPS from forward:   ----   -> ++++       ++++ -> ----
dt_re = g_re * x_re + g_im * x_im   # PLUS instead of forward's MINUS
dt_im = g_im * x_re - g_re * x_im   # MINUS instead of forward's PLUS
```

**Phase 8 conjugate 4-FMA for d_input (RESEARCH §"Complex path Pitfall 2"):**

```python
# conj(t) * g:  (a - bi)(c + di) = (ac + bd) + (ad - bc)i
dinput_re = t_re * g_re + t_im * g_im
dinput_im = t_re * g_im - t_im * g_re
```

**Same 4 sign flips** — derived from the SAME `(a ± bi)(c ± di)` expansion as the forward, but with conjugation on ONE operand. Plan must spell out BOTH d_twiddle and d_input conjugate formulas; D-50c only states d_twiddle's, and the d_input conjugate is a silent-pass-fp32-fail-complex64 bug surface (RESEARCH Pitfall 2).

---

#### Wrapper stage-group launch loop (op.py:455-489) — FACTORED OUT into `_run_forward_stage_groups`

**Source pattern (D-49a target — Phase 7's wrapper body at op.py:455-489):**

```python
cur_increasing_stride = increasing_stride
for block in range(nblocks):
    for group_start in range(0, log_n, 3):
        counter_count = min(3, log_n - group_start)
        if cur_increasing_stride:
            max_log_stride = group_start + counter_count - 1
        else:
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
            STAGE_START=group_start,
            STAGE_COUNT=counter_count,
            INCREASING_STRIDE=cur_increasing_stride,
            LOG_N=log_n,
            IS_COMPLEX=is_complex,
            TILE_N=tile_n,
            num_warps=num_warps,
        )
        src_buf, dst_buf = dst_buf, src_buf
    cur_increasing_stride = not cur_increasing_stride
```

**Phase 8 refactor (D-49a — extract verbatim into helper with `trail_out=None` hook):**

```python
def _run_forward_stage_groups(
    twiddle_work, input_work, increasing_stride, log_n, n, nstacks,
    nblocks, batch_size, is_complex, *,
    trail_out=None,  # NEW Phase 8 hook (D-49a)
):
    """Phase 7 stage-group launch loop, factored out per D-49a.

    When trail_out is None, behavior is identical to Phase 7's wrapper:
    ping-pong between buf_a/buf_b, return the final src_buf.

    When trail_out is not None (an fp32 tensor of shape
    (n_launches_per_nblock * nblocks, batch, nstacks, n) or doubled-n for
    complex64), each stage-group launch writes its output into
    trail_out[launch_idx] instead of the ping-pong dst. Return value is
    not meaningful in trail mode.
    """
    # ... buf_a/buf_b alloc (verbatim from op.py:428-441) ...
    src_buf = buf_a_work
    dst_buf = buf_b_work
    launch_idx = 0
    cur_increasing_stride = increasing_stride
    for block in range(nblocks):
        for group_start in range(0, log_n, 3):
            counter_count = min(3, log_n - group_start)
            if cur_increasing_stride:
                max_log_stride = group_start + counter_count - 1
            else:
                max_log_stride = log_n - 1 - group_start
            tile_n = 1 << (max_log_stride + 1)
            n_row_tiles = n // tile_n
            grid = (n_row_tiles, batch_size * nstacks)
            num_warps = _pick_num_warps(tile_n)
            dst_for_this_launch = (
                trail_out[launch_idx] if trail_out is not None else dst_buf
            )
            wrap_triton(_butterfly_kernel)[grid](
                twiddle_work, src_buf, dst_for_this_launch,
                n, nstacks, block, nblocks,
                STAGE_START=group_start, STAGE_COUNT=counter_count,
                INCREASING_STRIDE=cur_increasing_stride,
                LOG_N=log_n, IS_COMPLEX=is_complex, TILE_N=tile_n,
                num_warps=num_warps,
            )
            if trail_out is None:
                src_buf, dst_buf = dst_buf, src_buf
            else:
                # In trail mode the next stage reads from trail_out[launch_idx],
                # so we still need src to advance. Simplest: read the next src
                # from the trail slot we just wrote.
                src_buf = trail_out[launch_idx]
            launch_idx += 1
        cur_increasing_stride = not cur_increasing_stride
    return src_buf  # final output in non-trail mode; meaningless in trail mode
```

**Behavioral invariance check** (CRITICAL — Phase 7 forward tests must continue to pass):
- When `trail_out is None`, every line traversed must be byte-equivalent to the inlined Phase 7 wrapper at op.py:455-489.
- The wrapper at op.py:455-489 is REPLACED with a call: `final_work = _run_forward_stage_groups(..., trail_out=None)`.
- Verification: re-run `test_butterfly_eager_fp32` + `test_butterfly_eager_complex64` + the `@pytest.mark.slow` comprehensive tier — must all pass unchanged.

---

#### `_setup_context` — UNCHANGED

**Source:** `op.py:503-513`

```python
def _setup_context(ctx, inputs, output):
    twiddle, input_, increasing_stride, output_size = inputs
    ctx.save_for_backward(twiddle, input_)
    ctx.increasing_stride = increasing_stride
    ctx.output_size = output_size
```

**Phase 8 change:** NONE. Per D-57 explicit guarantee: "Phase 8 leaves `_setup_context` unchanged." The trail buffer is allocated inside `_backward` from `(twiddle, input_)`; no extra context needed.

---

#### `_backward` body REPLACEMENT (D-49 — THE Phase 8 deliverable)

**Source to REPLACE:** `op.py:516-543` (Phase 7's oracle-delegating body):

```python
def _backward(ctx, grad_out):
    """[Phase 7 oracle delegate — to be REPLACED]"""
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
```

**Phase 8 replacement skeleton** (transcribed from RESEARCH §"Pattern 1" + CONTEXT §"Recompute-then-walk-back template"):

```python
def _backward(ctx, grad_out):
    """Triton-backed two-input backward (D-49/D-50/D-50a/D-50b/D-50c).

    Replaces Phase 7's oracle delegation with:
      1. small-N fallback (log_n <= 1 — D-49b inheritance)
      2. allocate fp32 trail buffer (stage-group granularity per RESEARCH Pitfall 3)
      3. recompute forward into trail via _run_forward_stage_groups(trail_out=trail)
      4. allocate fp32 d_twiddle_scratch + d_input ping-pong buffers
      5. walk reverse stage-groups + reverse nblocks, per-launch:
         - kernel reads trail[i], src_grad, twiddle
         - kernel writes dst_grad (= d_input partial) + atomic_add into scratch
      6. cast fp32 scratch -> twiddle.dtype (or view_as_complex for complex64)
      7. trim d_input back to input_size
      8. return (d_twiddle, d_input, None, None)
    """
    twiddle, input_ = ctx.saved_tensors
    increasing_stride = ctx.increasing_stride
    output_size = ctx.output_size

    batch_size, nstacks, input_size = input_.shape
    nblocks = twiddle.shape[1]
    log_n = twiddle.shape[2]
    n = 1 << log_n
    is_complex = twiddle.is_complex()

    # D-49b small-N fallback (mirror Phase 7 wrapper's log_n <= 1 branch)
    if log_n <= 1:
        twiddle_d = twiddle.detach().requires_grad_(True)
        input_d = input_.detach().requires_grad_(True)
        with torch.enable_grad():
            out = _butterfly_multiply_torch(twiddle_d, input_d, increasing_stride, output_size)
        gt, gi = torch.autograd.grad(out, [twiddle_d, input_d], grad_out)
        return gt, gi, None, None

    # Pad/contiguous (mirror op.py:408-410)
    input_padded = (F.pad(input_, (0, n - input_size)) if input_size < n
                    else input_[:, :, :n]).contiguous()

    # Pitfall 3 (RESEARCH §"Recompute-Into-Trail"): stage-group granularity, NOT per-stage.
    n_launches_per_nblock = (log_n + 2) // 3  # = ceil(log_n / 3)
    trail_n = n * (2 if is_complex else 1)    # view_as_real flatten width for complex64
    trail = torch.empty(
        n_launches_per_nblock * nblocks, batch_size, nstacks, trail_n,
        dtype=torch.float32, device=input_padded.device,
    )

    # D-50b: scratch shape twiddle.shape + (2,) for complex (view_as_real flatten)
    scratch_shape = twiddle.shape + ((2,) if is_complex else ())
    d_twiddle_scratch = torch.zeros(scratch_shape, dtype=torch.float32, device=twiddle.device)

    # Recompute forward into trail (D-49a)
    if is_complex:
        twiddle_work = torch.view_as_real(twiddle).contiguous()
        input_work = torch.view_as_real(input_padded).contiguous()
    else:
        twiddle_work = twiddle
        input_work = input_padded
    _run_forward_stage_groups(
        twiddle_work, input_work, increasing_stride, log_n, n, nstacks,
        nblocks, batch_size, is_complex, trail_out=trail,
    )

    # Pad grad_out from output_size up to n (mirror forward op.py:408)
    grad_full = (F.pad(grad_out, (0, n - output_size)) if output_size < n
                 else grad_out).contiguous()

    # Ping-pong d_input buffers
    d_input_buf_a = torch.empty(batch_size, nstacks, n, dtype=input_.dtype, device=input_.device)
    d_input_buf_b = torch.empty_like(d_input_buf_a)
    if is_complex:
        d_input_buf_a_work = torch.view_as_real(d_input_buf_a)
        d_input_buf_b_work = torch.view_as_real(d_input_buf_b)
        grad_work = torch.view_as_real(grad_full).contiguous()
    else:
        d_input_buf_a_work = d_input_buf_a
        d_input_buf_b_work = d_input_buf_b
        grad_work = grad_full
    d_input_buf_a_work.copy_(grad_work)
    src_grad = d_input_buf_a_work
    dst_grad = d_input_buf_b_work

    # Walk reverse stage-groups + reverse nblocks (D-50; Pitfall 6 — ping-pong direction)
    # cur_increasing_stride initial value mirrors the forward loop's final state:
    # after Phase 7's `for block in range(nblocks): ... cur_increasing_stride = not cur_increasing_stride`,
    # the final value is `increasing_stride XOR (nblocks % 2 == 1)`. The LAST nblock's
    # in-loop value is the value at the START of its iteration; that's what backward needs.
    cur_increasing_stride = increasing_stride
    for _ in range(nblocks - 1):
        cur_increasing_stride = not cur_increasing_stride

    launch_idx_global = n_launches_per_nblock * nblocks - 1
    for block in range(nblocks - 1, -1, -1):
        # Reverse group_start order: 0, 3, 6, ... in forward; reverse here.
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
            trail_slot = trail[launch_idx_global]   # post-group activation
            wrap_triton(_butterfly_backward_kernel)[grid](
                twiddle_work,
                trail_slot,
                src_grad,
                dst_grad,
                d_twiddle_scratch,  # IS_COMPLEX: view_as_real-shaped scratch already
                n, nstacks, block, nblocks,
                STAGE_START=group_start, STAGE_COUNT=counter_count,
                INCREASING_STRIDE=cur_increasing_stride,
                LOG_N=log_n, IS_COMPLEX=is_complex, TILE_N=tile_n,
                num_warps=num_warps,
            )
            src_grad, dst_grad = dst_grad, src_grad
            launch_idx_global -= 1
        cur_increasing_stride = not cur_increasing_stride

    # After loop, src_grad holds the final d_input (last swap put dst -> src)
    d_input_full_work = src_grad
    if is_complex:
        d_input_full = torch.view_as_complex(d_input_full_work.contiguous())
        d_twiddle = torch.view_as_complex(d_twiddle_scratch.contiguous())
    else:
        d_input_full = d_input_full_work
        d_twiddle = d_twiddle_scratch.to(twiddle.dtype)
    d_input_out = d_input_full[:, :, :input_size]
    return d_twiddle, d_input_out, None, None
```

**Note the LANDMINEs** (RESEARCH §"Common Pitfalls"):
- Trail allocated at **stage-group granularity** (`ceil(log_n/3) * nblocks` slots), NOT per-stage. CONTEXT.md D-49 mentions "log_n*nblocks" but RESEARCH Pitfall 3 corrects this to launch-count granularity.
- `cur_increasing_stride` starting value for backward must mirror the forward loop's final block — apply `nblocks-1` toggles before entering the backward loop.
- `view_as_real` requires `.contiguous()` (Phase 4 Pitfall 3) — every cross-boundary view must be preceded by a contiguity check or `.contiguous()` call.

---

#### Per-program `tl.sum` reduce + single `tl.atomic_add` per program (SC#3, D-50a)

**Source pattern (RESEARCH §"Per-program tl.sum reduce semantics"):**

```python
# Inside the new _butterfly_backward_kernel, for each reverse stage:
# Per-position d_twiddle contributions (TILE_N-wide vectors, masked per is_lower)
g_lower_eff = tl.where(is_lower, g, 0.0)
g_upper_eff = tl.where(is_lower, 0.0, g)
x_lower_eff = tl.where(is_lower, x, 0.0)
x_upper_eff = tl.where(is_lower, 0.0, x)
xp_lower_eff = tl.where(is_lower, 0.0, x_partner)
xp_upper_eff = tl.where(is_lower, x_partner, 0.0)

dt_00_contrib = g_lower_eff * x_lower_eff   # real path; complex uses 4-FMA conjugate
dt_01_contrib = g_lower_eff * xp_upper_eff
dt_10_contrib = g_upper_eff * xp_lower_eff
dt_11_contrib = g_upper_eff * x_upper_eff

# Reduce within program across the (2*stride)-wide axis of each pair
n_pairs_in_tile = TILE_N // (2 * stride)
dt_00_per_pair = tl.sum(tl.reshape(dt_00_contrib, (n_pairs_in_tile, 2 * stride)), axis=1)
# ... same for dt_01, dt_10, dt_11

# Single tl.atomic_add per t_ij entry, vector-form (per-pair scratch offsets)
pair_flat_in_tile = tl.arange(0, n_pairs_in_tile)
pair_flat = (col_start >> 1) + pair_flat_in_tile
twiddle_pair_base = twiddle_stage_base + pair_flat * 4   # *8 for complex64
tl.atomic_add(d_twiddle_scratch_ptr + twiddle_pair_base + 0, dt_00_per_pair, sem="relaxed")
tl.atomic_add(d_twiddle_scratch_ptr + twiddle_pair_base + 1, dt_01_per_pair, sem="relaxed")
tl.atomic_add(d_twiddle_scratch_ptr + twiddle_pair_base + 2, dt_10_per_pair, sem="relaxed")
tl.atomic_add(d_twiddle_scratch_ptr + twiddle_pair_base + 3, dt_11_per_pair, sem="relaxed")
```

**Why it matters:** SC#3 verbatim mandate — "single atomic_add per program per t_ij entry." The `tl.sum` reduce amortizes atomic traffic by a factor of `TILE_N / (2*stride)` (e.g., 2048× at the largest tile, smallest stride), keeping fp32 noise below the D-52 envelope (rtol=1e-3, atol=1e-4 at batch=4096).

**Fallback per RESEARCH §A2/A7:** If Triton 3.6's `tl.reshape` has constexpr-shape constraints, substitute an explicit `tl.static_range`-loop over pairs with scalar `tl.sum` per pair (slower but always works). Plan must document the fallback explicitly.

---

#### `register_autograd` registration line — UNCHANGED

**Source:** `op.py:546`

```python
butterfly_multiply.register_autograd(_backward, setup_context=_setup_context)
```

**Phase 8 change:** NONE. The new `_backward` is bound by the same line; only its body changes.

---

#### `register_fake` — UNCHANGED

**Source:** `op.py:549-567`

```python
@butterfly_multiply.register_fake
def _butterfly_multiply_fake(twiddle, input, increasing_stride=True, output_size=None):
    batch_size, nstacks, _ = input.shape
    log_n = twiddle.shape[2]
    n = 1 << log_n
    output_size_actual = n if output_size is None else output_size
    return torch.empty(
        batch_size, nstacks, output_size_actual,
        dtype=input.dtype, device=input.device,
    )
```

**Phase 8 change:** NONE. `register_fake` describes the FORWARD signature only. Backward is handled by autograd's tracing of the composed ops inside `_backward` (`torch.empty`, `torch.zeros`, `view_as_real`, `wrap_triton(...)` launches). Per RESEARCH §"register_autograd callback signature + saved-tensor invariants", PyTorch 2.6+'s `register_autograd` supports this composition without a separate `register_fake` for the backward kernel.

---

### `tests/test_butterfly_triton.py` (test — extend with backward tests)

**Analog:** the file itself (`tests/test_butterfly_triton.py:36-414`) — existing Phase 7 structure transcribes directly.

**Module-level constants (UNCHANGED):**

```python
# tests/test_butterfly_triton.py:54-55
RTOL = 1e-3
ATOL = 1e-3
```

Phase 8 adds **per-test tolerance overrides** for the tighter d_input envelope and the looser d_twiddle envelope per D-52a.

---

#### Plan 08-01 NEW test: `test_butterfly_backward_gradcheck_fp64`

**Analog:** `tests/test_butterfly_triton.py:167-197` (`test_butterfly_gradcheck_fp64` — same parametrize, same skip pattern).

**Pattern to copy** (skip-triton + torch-backend gradcheck — note D-47 was the Phase 7 pattern; Phase 8 needs the SAME skip but for the NEW reason explained below):

```python
def test_butterfly_backward_gradcheck_fp64(backend):
    """fp64 gradcheck — SC#1 layer (a) acceptance.

    Triton backend SKIPPED because the kernel is fp32/complex64 only.

    LANDMINE vs Phase 7: in Phase 7, the Triton backward delegated to _torch_ref,
    so the torch-backend gradcheck WAS testing the same plumbing. In Phase 8,
    the Triton backward is its own kernel — gradcheck on the torch backend no
    longer exercises the Triton backward. Layer (b)/(c) allclose tests (below)
    cover the Triton path.
    """
    if backend == "triton":
        pytest.skip(
            "Triton kernel is fp32/complex64 only; fp64 gradcheck on torch backend "
            "covers the autograd plumbing — Triton backward correctness verified by "
            "layer (b)/(c) allclose tests at fp32 noise floor"
        )
    log_n, nstacks, nblocks, batch_size = 2, 1, 1, 1  # SC#1 case
    n = 1 << log_n
    twiddle = torch.randn(nstacks, nblocks, log_n, n // 2, 2, 2,
                          dtype=torch.float64, device="cuda", requires_grad=True)
    input_ = torch.randn(batch_size, nstacks, n,
                         dtype=torch.float64, device="cuda", requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda t, x: torch_structured._ops.butterfly_multiply(t, x, True, n),
        (twiddle, input_), eps=1e-6, atol=1e-5,
    )
```

---

#### Plan 08-01 NEW test: `test_butterfly_dinput_allclose_fp32` (layer (b))

**Analog skeleton:** `tests/test_butterfly_triton.py:59-70` (smoke parametrize + dual-path comparison) + `tests/test_diag_mult.py:50-95` (the dual-clone allclose pattern with `requires_grad=True`).

**Pattern to copy** (RESEARCH §"Test Surface Mechanics"):

```python
def test_butterfly_dinput_allclose_fp32(backend):
    """SC#1 layer (b): d_input fp32 allclose vs autograd-of-oracle.

    Tighter tolerance than d_twiddle because d_input is a sum-of-products,
    not an atomicAdd-noisy reduction. rtol=1e-5, atol=1e-6 (RESEARCH §"Layer (b)").
    """
    if backend != "triton":
        pytest.skip("d_input parity check is meaningful only for triton backend")
    log_n, nstacks, nblocks, batch_size = 8, 1, 1, 8
    n = 1 << log_n
    twiddle = torch.randn(nstacks, nblocks, log_n, n // 2, 2, 2,
                          device="cuda", dtype=torch.float32, requires_grad=True)
    input_ = torch.randn(batch_size, nstacks, n,
                         device="cuda", dtype=torch.float32, requires_grad=True)
    grad_out = torch.randn(batch_size, nstacks, n, device="cuda", dtype=torch.float32)

    # Triton path
    twiddle_t = twiddle.detach().clone().requires_grad_()
    input_t = input_.detach().clone().requires_grad_()
    out_t = torch_structured._ops.butterfly_multiply(twiddle_t, input_t, True, n)
    out_t.backward(grad_out)

    # Oracle path (autograd-of-oracle per D-52b)
    twiddle_o = twiddle.detach().clone().requires_grad_()
    input_o = input_.detach().clone().requires_grad_()
    out_o = butterfly_ref(twiddle_o, input_o, True, n)
    out_o.backward(grad_out)

    assert torch.allclose(input_t.grad, input_o.grad, rtol=1e-5, atol=1e-6)
```

---

#### Plan 08-01 NEW test: `test_butterfly_dtwiddle_allclose_fp32` (layer (c))

Same dual-path pattern as layer (b) but at `log_n=9, batch=4096` with the **looser D-52 envelope**:

```python
def test_butterfly_dtwiddle_allclose_fp32(backend):
    """SC#1 layer (c): d_twiddle fp32 allclose at batch=4096 — atomicAdd noise envelope (D-52).

    rtol=1e-3, atol=1e-4 — calibrated for sqrt(batch) * machine_eps_fp32 ~ 1e-4
    atomicAdd reorder noise at batch=4096. Looser than layer (b) because
    d_twiddle goes through the atomic_add scratch path (SC#3).
    """
    if backend != "triton":
        pytest.skip("d_twiddle parity check is meaningful only for triton backend")
    # ... same dual-clone pattern, log_n=9, batch_size=4096 ...
    assert torch.allclose(twiddle_t.grad, twiddle_o.grad, rtol=1e-3, atol=1e-4)
```

---

#### Plan 08-01 NEW test: `test_butterfly_backward_no_cpp_symbol` (SC#4)

**Source pattern:** RESEARCH §"SC#4 Verification Mechanism" — Option 2 + Option 3 combined (dispatch-binding `is` check + monkey-patch shim that raises on legacy call).

**CRITICAL — DO NOT use CONTEXT.md's `'_butterfly' not in sys.modules` form.** Per RESEARCH §"SC#4 Verification Mechanism" + Pitfall 4: `sys.modules['_butterfly']` is **always** False because `torch.ops.load_library` uses dlopen at the C level, never enters `sys.modules`. The CONTEXT.md text says "OR" — RESEARCH corrects to the monkey-patch + dispatch-binding approach.

**Pattern to copy verbatim from RESEARCH §"Pattern 3":**

```python
def test_butterfly_backward_no_cpp_symbol():
    """SC#4: BACKEND=triton must not invoke any symbol from csrc/butterfly.cpp."""
    import torch_structured
    import torch_structured.butterfly.multiply as legacy_mod
    from torch_structured._triton.butterfly.op import butterfly_multiply as triton_op
    torch_structured.set_backend('triton')

    # Part 1: dispatch-binding assertion (cheap, deterministic)
    assert torch_structured._ops.butterfly_multiply is triton_op, \
        "SC#4: _ops.butterfly_multiply must be bound to the Triton kernel"

    # Part 2: runtime invocation tracking via monkey-patch shim
    raised_calls = []
    original_fw = legacy_mod.butterfly_multiply_fw
    original_bw = legacy_mod.butterfly_multiply_bw
    def _fail_fw(*a, **kw):
        raised_calls.append('fw'); raise AssertionError("SC#4: legacy_fw invoked")
    def _fail_bw(*a, **kw):
        raised_calls.append('bw'); raise AssertionError("SC#4: legacy_bw invoked")
    legacy_mod.butterfly_multiply_fw = _fail_fw
    legacy_mod.butterfly_multiply_bw = _fail_bw
    try:
        n = 256; log_n = 8
        twiddle = torch.randn(1, 1, log_n, n // 2, 2, 2, device='cuda',
                              dtype=torch.float32, requires_grad=True)
        x = torch.randn(4, 1, n, device='cuda', dtype=torch.float32, requires_grad=True)
        loss = torch_structured._ops.butterfly_multiply(twiddle, x, True, n).sum()
        loss.backward()
        assert raised_calls == [], f"SC#4 violated: {raised_calls}"
        assert twiddle.grad is not None
        assert x.grad is not None
    finally:
        legacy_mod.butterfly_multiply_fw = original_fw
        legacy_mod.butterfly_multiply_bw = original_bw
```

---

#### Plan 08-02 NEW test: `test_butterfly_backward_complex64`

**Analog:** `tests/test_butterfly_triton.py:262-281` (`test_butterfly_eager_complex64`) for the parametrize/dtype skeleton + the dual-clone allclose pattern from layer (c).

**Pattern to copy:**

```python
def test_butterfly_backward_complex64(backend):
    """SC#2: complex64 d_input + d_twiddle allclose vs autograd-of-oracle.

    Same envelope as SC#3 fp32 d_twiddle: rtol=1e-3, atol=1e-4 at batch=4096.
    The load-bearing detector for D-50c sign errors (forward 4-FMA cargo-culted
    into d_twiddle = g*x instead of g*conj(x)).
    """
    if backend != "triton":
        pytest.skip("complex64 backward parity check is meaningful only for triton")
    log_n, nstacks, nblocks, batch_size = 9, 1, 1, 4096
    n = 1 << log_n
    twiddle = torch.randn(nstacks, nblocks, log_n, n // 2, 2, 2,
                          device="cuda", dtype=torch.complex64, requires_grad=True)
    input_ = torch.randn(batch_size, nstacks, n,
                         device="cuda", dtype=torch.complex64, requires_grad=True)
    grad_out = torch.randn(batch_size, nstacks, n, device="cuda", dtype=torch.complex64)

    # ... same dual-clone pattern as layer (c) ...
    assert torch.allclose(input_t.grad, input_o.grad, rtol=1e-3, atol=1e-4)
    assert torch.allclose(twiddle_t.grad, twiddle_o.grad, rtol=1e-3, atol=1e-4)
```

---

#### Plan 08-02 NEW test: extend comprehensive tier (`test_butterfly_backward_comprehensive`)

**Analog:** `tests/test_butterfly_triton.py:200-240` (`test_butterfly_comprehensive` — `@pytest.mark.slow` with `itertools.product` over the full grid).

**Pattern to copy** (extend the comprehensive grid to include backward):

```python
@pytest.mark.slow
@pytest.mark.parametrize(
    "log_n,nstacks,nblocks,increasing_stride,output_size_kind,dtype",
    list(itertools.product(
        range(2, 12),       # log_n in {2..11}
        [1, 2, 3],          # nstacks
        [1, 2],             # nblocks
        [True, False],      # increasing_stride
        ["n", "half", "n-1"],
        [torch.float32, torch.complex64],
    )),
)
def test_butterfly_backward_comprehensive(
    backend, log_n, nstacks, nblocks, increasing_stride, output_size_kind, dtype
):
    """D-43a comprehensive tier extended to backward.

    Mirrors test_butterfly_comprehensive but verifies backward gradients via
    the layer (b)/(c) dual-clone pattern.
    """
    # ... full grid coverage with d_input + d_twiddle allclose at RTOL/ATOL ...
```

---

### `.planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json` (config — schema extension)

**Analog:** the file itself — `07-BASELINE.json:1-101`.

**Current schema (8 rows, forward only):**

```json
{
  "rows": [
    {
      "kernel": "butterfly_multiply",
      "dtype": "fp32",
      "log_n": 8,
      "nstacks": 1,
      "nblocks": 1,
      "wall_ms_p50": 0.197632,
      "wall_ms_p95": 0.444416,
      "reference_torch_ref_p50": 0.37376,
      "measured_at": "2026-05-27T20:04:10.271752Z",
      "gpu": "NVIDIA RTX 2000 Ada Generation Laptop GPU"
    },
    ...
  ]
}
```

**Phase 8 extension (RESEARCH §"Performance Baseline Strategy" + CONTEXT D-51):**

1. **Add `direction` field** to ALL existing rows (set to `"forward"`).
2. **Append 8 new rows** with `direction: "backward"` covering the same `log_n × dtype` grid (`log_n ∈ {8,9,10,11} × {fp32, complex64}`).
3. **Measurement:** triggered via `out.sum().backward()` inside the do_bench loop with `retain_graph=True` (option (a) from RESEARCH — measures FULL backward callback overhead including trail recompute).
4. **Location:** extend the existing `07-BASELINE.json` in place (per CONTEXT.md explicit phrasing "extend `07-BASELINE.json` schema with backward p50/p95 entries" + per D-51's "extend Phase 7's `07-BASELINE.json`").

**Pattern for new rows:**

```json
{
  "kernel": "butterfly_multiply",
  "direction": "backward",
  "dtype": "fp32",
  "log_n": 8,
  "nstacks": 1,
  "nblocks": 1,
  "wall_ms_p50": <float>,
  "wall_ms_p95": <float>,
  "reference_torch_ref_p50": <float>,
  "measured_at": "<ISO8601>",
  "gpu": "NVIDIA RTX 2000 Ada Generation Laptop GPU"
}
```

**No-analog note:** The existing 07-BASELINE.json is the analog for itself. Phase 9 (TEST-04 perf gate) will consume both `direction` values; Phase 8 only writes them.

---

## Shared Patterns

### Pattern: Two-input `register_autograd` callback signature

**Source:** `torch_structured/_triton/butterfly/op.py:503-546` (Phase 7's full registration).
**Apply to:** `_backward` body (Phase 8's deliverable) — the **signature** (4-tuple return, `(twiddle, input_)` saved tensors, `(increasing_stride, output_size)` scalar attrs) stays IDENTICAL to Phase 7.

```python
# Save context (UNCHANGED from Phase 7 op.py:503-513)
def _setup_context(ctx, inputs, output):
    twiddle, input_, increasing_stride, output_size = inputs
    ctx.save_for_backward(twiddle, input_)
    ctx.increasing_stride = increasing_stride
    ctx.output_size = output_size

# Backward signature (UNCHANGED — only body replaced)
def _backward(ctx, grad_out):
    twiddle, input_ = ctx.saved_tensors
    # ... NEW BODY: trail recompute + reverse stage-group walk ...
    return d_twiddle, d_input, None, None  # 4-tuple matching forward inputs

# Registration line (UNCHANGED from op.py:546)
butterfly_multiply.register_autograd(_backward, setup_context=_setup_context)
```

**Cross-reference:** Phase 5's `_triton/diag_mult/op.py:165-193` is the canonical two-input `register_autograd` shape (different math, same 4-tuple-return contract).

---

### Pattern: `view_as_real` / `view_as_complex` wrapper boundary (D-50b)

**Source:** Phase 7's wrapper at `op.py:412-422` + `op.py:494-497`.
**Apply to:** Phase 8's `_backward` body wherever a complex64 tensor crosses into a Triton kernel call (input, twiddle, d_twiddle_scratch, grad_full, d_input ping-pong buffers).

```python
# Boundary INTO kernel (view_as_real, asserting contiguity per Pitfall 3)
if is_complex:
    twiddle_work = torch.view_as_real(twiddle).contiguous()
    input_work = torch.view_as_real(input_padded).contiguous()
    grad_work = torch.view_as_real(grad_full).contiguous()
    # d_twiddle_scratch already allocated with trailing (2,) — no view needed

# Boundary OUT of kernel (view_as_complex on a contiguous fp32-trailing-2 buffer)
if is_complex:
    d_input_full = torch.view_as_complex(d_input_full_work.contiguous())
    d_twiddle = torch.view_as_complex(d_twiddle_scratch.contiguous())
else:
    d_twiddle = d_twiddle_scratch.to(twiddle.dtype)
```

**Pitfall 3 check** (`op.py:372-373`): every `view_as_real` MUST be preceded by an `is_contiguous()` assert or a `.contiguous()` call. Non-contiguous complex tensors produce wrong strides.

---

### Pattern: `tl.atomic_add` on fp32 with `sem="relaxed"`

**Source pattern:** RESEARCH §"`tl.atomic_add` semantics on fp32" + the Triton docs.
**Apply to:** Every `d_twiddle_scratch` accumulator atomic in the new backward kernel (4 atomics per program per stage — one per `t_ij` in `{t_00, t_01, t_10, t_11}`).

```python
tl.atomic_add(d_twiddle_scratch_ptr + offset, value, sem="relaxed")
```

**Why `sem="relaxed"`:** RESEARCH §"`tl.atomic_add` semantics" — the only ordering constraint we need is among atomic-adds to the SAME address; relaxed is faster than the default `acq_rel` and correct for ULP-noise-tolerant reductions.

**Why fp32 (NOT bf16/fp16):** SC#3 verbatim — bf16 is unsupported (Triton issue #2834); fp16 atomic_add is brittle (Triton issue #891). The fp32-scratch + boundary-cast pattern is the only viable path.

---

### Pattern: Small-N fallback via `_butterfly_multiply_torch`

**Source:** `op.py:405-406` (Phase 7's forward small-N fallback).
**Apply to:** Phase 8's `_backward` body — D-49b inheritance (when `log_n <= 1`, route through `torch.autograd.grad(_butterfly_multiply_torch(...))`).

```python
# Phase 7 forward (op.py:405-406):
if log_n <= 1:
    return _butterfly_multiply_torch(twiddle, input, increasing_stride, output_size).clone()

# Phase 8 backward (D-49b — same threshold, autograd graph for both gradients):
if log_n <= 1:
    twiddle_d = twiddle.detach().requires_grad_(True)
    input_d = input_.detach().requires_grad_(True)
    with torch.enable_grad():
        out = _butterfly_multiply_torch(twiddle_d, input_d, increasing_stride, output_size)
    gt, gi = torch.autograd.grad(out, [twiddle_d, input_d], grad_out)
    return gt, gi, None, None
```

This is verbatim the Phase 7 `_backward` body (the thing Phase 8 replaces for `log_n > 1`).

---

### Pattern: Pad/trim via `F.pad` and slice

**Source:** `op.py:408-410` (input pad) + `op.py:499-500` (output trim).
**Apply to:** Phase 8's `_backward` body — mirror the forward's pad/trim for `grad_out` (pad to `n`) and `d_input` (trim back to `input_size`).

```python
# Forward (Phase 7 op.py:408-410):
input = F.pad(input, (0, n - input_size)) if input_size < n else input[:, :, :n]
input = input.contiguous()

# Backward (Phase 8 — grad_out pad mirrors forward input pad):
grad_full = (F.pad(grad_out, (0, n - output_size)) if output_size < n
             else grad_out).contiguous()

# Backward (Phase 8 — d_input trim mirrors forward output trim):
d_input_out = d_input_full[:, :, :input_size]
```

---

### Pattern: `assert` for preconditions (CLAUDE.md)

**Source:** `op.py:366-391` (Phase 7's wrapper-boundary asserts).
**Apply to:** Phase 8's `_backward` body if it adds preconditions (e.g., `assert input.is_contiguous()` before `view_as_real(input_padded)`).

```python
# Style from op.py:366-378:
assert input.dim() == 3, (
    f"input must be (batch, nstacks, input_size), got dim={input.dim()}"
)
assert input.is_contiguous(), "input must be contiguous (Pitfall 3)"
```

Per `/home/claroche/torch-structured/CLAUDE.md` "Error Handling": `assert` for preconditions, no try/except in core library code. One documented exception: `_cuda_legacy/*.py` try-imports (Phase 5 D-21 honest-probe).

---

### Pattern: Tiered test surface (D-43a)

**Source:** `tests/test_butterfly_triton.py:58-70` (dense smoke) + `:200-240` (`@pytest.mark.slow` comprehensive Cartesian).
**Apply to:** All Phase 8 backward tests — same tier structure:
- **Dense smoke:** `log_n ∈ {2, 4, 8, 10}`, default axes, every-CI runs.
- **Comprehensive:** full `itertools.product(range(2, 12), [1,2,3], [1,2], [True,False], ["n","half","n-1"], [fp32, complex64])` marked `@pytest.mark.slow`, opt-in via `pytest -m slow`.

```python
# Dense smoke parametrize (op.py-test line 58):
@pytest.mark.parametrize("log_n", [2, 4, 8, 10])
def test_butterfly_backward_smoke(backend, log_n):
    ...

# Comprehensive @pytest.mark.slow (op.py-test line 200):
@pytest.mark.slow
@pytest.mark.parametrize(
    "log_n,nstacks,nblocks,increasing_stride,output_size_kind,dtype",
    list(itertools.product(range(2, 12), [1, 2, 3], [1, 2], [True, False],
                           ["n", "half", "n-1"], [torch.float32, torch.complex64])),
)
def test_butterfly_backward_comprehensive(backend, ...):
    ...
```

---

### Pattern: `backend` fixture skip-gate (D-58 — UNCHANGED)

**Source:** `tests/conftest.py:22-30` + `torch_structured/_ops.py:139` (`_has_any_triton_kernel` iterates butterfly_multiply).
**Apply to:** All Phase 8 tests — they automatically pick up the `backend` fixture; no conftest changes needed.

**Verification:** RESEARCH §A3 confirms Phase 6 D-39 already widened `_has_any_triton_kernel()` to iterate butterfly_multiply; Phase 7 inherited; Phase 8 inherits.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| (none) | — | — | All Phase 8 files have in-repo analogs. The new `_butterfly_backward_kernel` mirrors Phase 7's `_butterfly_kernel` (same launch shape, reversed walk, plus atomic-add reduction); the CUDA legacy at `csrc/cuda/butterfly_cuda.cu:421-489` provides the algorithmic blueprint for the per-step `d_twiddle += grad * conj(input)` + `gpuAtomicAdd` pattern. The Phase 8 `_backward` body composition is novel in the codebase (no prior op replaces a Phase 7-style "oracle delegate" body with a kernel-backed reverse walk), but every constituent piece — trail buffer allocation (`torch.empty`), recompute via `wrap_triton(...)` launches, reverse static_range, `tl.atomic_add` on fp32, view_as_real boundary — has a direct in-repo precedent. |

---

## Metadata

**Analog search scope:**
- `torch_structured/_triton/butterfly/op.py` (Phase 7 deliverable — primary source for forward kernel + wrapper + register_autograd skeleton)
- `torch_structured/_triton/diag_mult/op.py` (Phase 5 — two-input backward Wirtinger pattern)
- `torch_structured/_triton/hadamard_transform/op.py` (Phase 6 — Triton op skeleton precedent)
- `torch_structured/_torch_ref/butterfly.py` (the verbatim oracle — autograd reference per D-52b)
- `csrc/cuda/butterfly_cuda.cu:419-606` (CUDA forward-backward fused kernel + host wrapper — algorithmic blueprint)
- `csrc/butterfly.cpp:127-131` (TORCH_LIBRARY registration — symbols SC#4 verifies are not invoked)
- `tests/test_butterfly_triton.py` (Phase 7 test file — extension target)
- `tests/test_diag_mult.py` (Phase 5 test skeleton — dual-clone backward allclose pattern)
- `tests/conftest.py` (backend fixture skip-gate — UNCHANGED)
- `.planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json` (perf baseline — schema extension target)
- `.planning/phases/07-butterfly-multiply-forward-triton/07-01-PLAN.md` / `07-02-PLAN.md` (plan templates — task structure to mirror in 08-01 / 08-02)
- `.planning/phases/07-butterfly-multiply-forward-triton/07-PATTERNS.md` (Phase 7's pattern map — structural template for this document)

**Files scanned:** 12 source files + 4 planning artifacts.

**Pattern extraction date:** 2026-05-28

---

## PATTERN MAPPING COMPLETE
