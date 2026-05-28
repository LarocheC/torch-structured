# Phase 8: butterfly_multiply Backward (Triton) - Research

**Researched:** 2026-05-28
**Domain:** Triton backward kernel for butterfly multiply — recompute-into-trail + reverse stage-group walk + fp32 atomic scratch reduction
**Confidence:** HIGH (claims grounded in repo source + Phase 7 deliverable + CUDA legacy kernel + verified Triton 3.6/PyTorch 2.11 behavior on the dev host)

## Summary

Phase 8 replaces the `_backward` body at `torch_structured/_triton/butterfly/op.py:516-543` (currently `torch.autograd.grad(_butterfly_multiply_torch(...))`) with a Triton-backed backward that (a) recomputes the forward into a `(log_n*nblocks, batch, nstacks, n)` fp32 trail buffer by reusing Phase 7's stage-group launches, then (b) walks the stage groups in REVERSE order, using a register-resident 3-stage tile mirroring Phase 7's forward, accumulating `d_twiddle` via per-program `tl.sum` + single `tl.atomic_add` into a `fp32` scratch buffer, and writing `d_input` via ping-ponged buffers. Two plans split by dtype (08-01 fp32, 08-02 complex64). Every locked decision in CONTEXT.md (D-49..D-59) carries through.

The gaps the planner needs filled are: the per-stage backward algebra (forward + reverse 2x2 transpose-conjugate), the reverse register-walk pattern within a tile, the precise `tl.sum` / `tl.atomic_add` shape semantics, the d_twiddle_scratch offset arithmetic (especially the view_as_real flatten), the trail-buffer launch strategy (recompute path), `triton_op.register_autograd` invariants under FakeTensorMode (Phase 8 allocates buffers INSIDE `_backward` — does that compose?), and the SC#4 mechanism. Each is resolved below with file:line citations.

**Primary recommendation:** Write `_butterfly_backward_kernel` as a near-mirror of Phase 7's `_butterfly_kernel` (`op.py:77-321`), but with stages walked in reverse-of-counter order inside a 3-stage group, a register-resident `d_input` tile + per-stage local `d_twiddle` registers, a `tl.sum` reduce of the `d_twiddle` registers across the row tile (axis=0 of the in-register vector), and a single `tl.atomic_add` per program with `mask=` on the 4 reduced lanes per pair × (2 for complex). Allocate the trail buffer + scratch + d_input in the wrapper (`_backward` body); factor Phase 7's launch loop into `_run_forward_stage_groups(..., trail_out=None)` per D-49a.

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-49** Recompute forward into a `(log_n * nblocks, batch, nstacks, n)` fp32 trail buffer (or doubled trailing axis for complex64). NOT save-during-forward, NOT fused single-launch.
- **D-49a** Factor Phase 7's stage-group launch loop at `_triton/butterfly/op.py:322-501` into `_run_forward_stage_groups(..., trail_out=None)`. `trail_out=None` preserves Phase 7 forward behavior verbatim (ping-pong).
- **D-49b** `log_n <= 1` → small-N fallback through `torch.autograd.grad(_butterfly_multiply_torch(...))`.
- **D-50** Backward = `ceil(log_n / 3)` launches per nblock in REVERSE stage-group order. Same TILE_N schedule + same 2-D grid + same num_warps schedule as Phase 7 D-40d.
- **D-50a** Per-program `tl.sum` reduce + single `tl.atomic_add` into fp32 scratch. fp32 → twiddle.dtype cast at callback boundary, NEVER inside the kernel.
- **D-50b/c** complex64 d_twiddle via `view_as_real` flatten + conjugate 4-FMA: `(a+bi) * conj(c+di) = (ac+bd) + (bc-ad)i`.
- **D-51 / D-51a** 2 plans by dtype (08-01 fp32, 08-02 complex64). IS_COMPLEX pre-wire pattern. Zero kernel-signature refactor between plans.
- **D-52** d_twiddle envelope rtol=1e-3, atol=1e-4 at batch=4096.
- **D-52a** Three-layer gradcheck pattern (fp64 small-case + d_input allclose + d_twiddle allclose).
- **D-52b** Oracle is `torch.autograd.grad(butterfly_multiply_torch, ...)`, not CUDA legacy.
- **D-53** SC#4 verified at runtime via `sys.modules` / `torch.ops` registry probe. NOT build-time.
- **D-57** Replace ONLY the body of `_backward` (op.py:516-543). `_setup_context`, `register_autograd` registration, `register_fake` stay as-is.
- **D-58** No conftest changes.
- **D-59** Pad-on-input + trim-on-output preserved; small-N fallback via torch oracle's autograd graph.

### Claude's Discretion

Per CONTEXT.md `Claude's Discretion`:
- SC#4 probe mechanism (see §SC#4 Verification Mechanism below — research surfaces a correction)
- Per-call vs. saved trail buffer (recommend per-call; matches CUDA)
- Single `STAGE_COUNT: tl.constexpr ∈ {1,2,3}` kernel vs. separate kernels (recommend single)
- Order of recompute-then-walk-back (recommend all recompute first then all backward)
- `d_twiddle_scratch` allocation form (recommend `torch.zeros_like(twiddle, dtype=torch.float32)`)
- `range(nblocks-1, -1, -1)` over `reversed(range(...))` (recommend explicit)
- `d_input` ping-pong between two buffers (recommend ping-pong; mirrors forward)
- `trail` allocated with `torch.empty` (not `zeros`) — every slot written before read

### Deferred Ideas (OUT OF SCOPE)

- 5-stage tile backward (Phase 9)
- Save-during-forward (~720MB at log_n=11/nblocks=2/batch=4096 — rejected)
- Fused forward-backward single-launch kernel (Phase 9 perf candidate)
- `TORCH_STRUCTURED_BACKEND_BW=triton` opt-in (rejected)
- bf16 / fp16 backward (TRI-FUT-01)
- `log_n > 11` test surface
- `@triton.autotune` (Phase 9)
- Touching csrc/butterfly.cpp / butterfly_cuda.cu (Phase 10)
- Editing `_setup_context` (stays as Phase 7 wrote it)
- Build-time guarantee for SC#4 (runtime assertion only)

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TRI-04 | `butterfly_multiply` backward runs on Triton with fp32 scratch accumulator for atomic adds (no direct bf16/fp16 atomicAdd) | §"Backward algebra", §"tl.atomic_add semantics", §"d_twiddle_scratch shape + offsets" — full algorithmic recipe + fp32 cast at boundary per D-50a |

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| `_backward` body (allocate trail + scratch + d_input; orchestrate launches) | Python wrapper inside `_triton/butterfly/op.py` | — | Mirrors CUDA's `butterfly_multiply_bw_cuda` host-side wrapper at `csrc/cuda/butterfly_cuda.cu:535-606`; pure orchestration, no math |
| Recompute forward into trail | Python wrapper helper `_run_forward_stage_groups(..., trail_out)` | Triton `_butterfly_kernel` (unchanged) | D-49a: reuse Phase 7's launches via a refactor that adds an optional output-redirect; ZERO new Triton source |
| Reverse stage-group walk + d_twiddle reduce + d_input write | New Triton `_butterfly_backward_kernel` | — | Math lives where the math runs; per-program `tl.sum` then single `tl.atomic_add` per D-50a |
| fp32 → twiddle.dtype cast on d_twiddle | Python wrapper (callback boundary) | — | D-50a explicit: cast OUTSIDE the kernel |
| Wirtinger gradient correction for complex64 | Triton kernel via 4-FMA conjugate at d_twiddle stage; `view_as_complex` at callback boundary | — | Identical to Phase 7's wrapper-boundary pattern, but with conjugate 4-FMA per D-50c |
| Small-N fallback (log_n ≤ 1) | Python wrapper inside `_backward` | torch oracle `_butterfly_multiply_torch` + `torch.autograd.grad` | D-49b inheritance; identical mechanic to Phase 7 wrapper `op.py:405-406` but invoked in backward |
| SC#4 verification | New test in `tests/test_butterfly_triton.py` | — | Probes `torch.ops` registry call activity around a `loss.backward()` ; see §"SC#4 Verification Mechanism" for the correction |

## Project Constraints (from CLAUDE.md)

- **`assert` for preconditions; no try/except in core lib.** Verified at `/home/claroche/torch-structured/CLAUDE.md` ("Error Handling" section — also visible in Phase 7 `op.py:366-391`). Plan 08-01 wrapper preconditions must be `assert`s. One documented exception: `_cuda_legacy/*.py` try-imports (honest-probe pattern from Phase 5 D-21).
- **Beads (bd) for task tracking, NOT TaskCreate / TodoWrite.** Plan executors use `bd ready` / `bd update --claim` / `bd close` per `/home/claroche/CLAUDE.md`.
- **GSD workflow enforcement** — edits via GSD command surface only.
- **No emojis in files unless requested.**

## Phase 7 Deliverable Reuse Map

Phase 8 ships INSIDE the same `op.py` file Phase 7 wrote — concrete reuse:

| Phase 7 asset | Phase 8 action |
|---|---|
| `_butterfly_kernel` (op.py:77-321) | Reuse VERBATIM for trail recompute |
| Forward stage-group launch loop (op.py:322-501) | Factor into `_run_forward_stage_groups(..., trail_out=None)` per D-49a; identical behavior when `trail_out is None` |
| `_setup_context` (op.py:503-513) | Unchanged — saves `(twiddle, input_)`, `ctx.increasing_stride`, `ctx.output_size` |
| `_backward` body (op.py:516-543) | REPLACE — this is the deliverable |
| `register_autograd` registration line (op.py:546) | Unchanged |
| `register_fake` (op.py:549-567) | Unchanged — meta kernel returns output shape only, doesn't touch backward |
| `_pick_num_warps` (op.py:68-74) | Reuse VERBATIM |
| `IS_COMPLEX: tl.constexpr` flag pattern | Mirror in new backward kernel signature |
| Pad/trim wrapper logic (op.py:408-410) | Mirror for `grad_out` (pad up to n) + final `d_input` trim |
| `view_as_real`/`view_as_complex` boundary | Mirror with conjugate 4-FMA branch inside backward kernel + `d_twiddle_scratch` allocation with trailing-2 axis |

## Backward Algebra (Gap #1)

For one butterfly stage with stride `stride = 1 << log_stride`:

The forward, position-wise (verbatim from `_torch_ref/butterfly.py:25-31`):
```
log_stride = idx if INCREASING_STRIDE else LOG_N - 1 - idx
stride = 1 << log_stride
t = twiddle[:, block, idx].view(nstacks, n//(2*stride), stride, 2, 2).permute(0, 1, 3, 4, 2)
out = (t * input_reshape).sum(dim=4)
```

Concretely, for each pair `(lower, upper)` with `upper = lower ^ stride`:
```
new_lower = t_00 * lower + t_01 * upper
new_upper = t_10 * lower + t_11 * upper
```
i.e., `[new_lower; new_upper] = T_2x2 @ [lower; upper]`.

### Real-path backward

For a single 2x2 stage `y = T @ x`, with upstream gradient `g_y = (g_new_lower, g_new_upper)`:

**Pulling back to input:**
```
g_x = T^T @ g_y
  g_lower_in = t_00 * g_new_lower + t_10 * g_new_upper
  g_upper_in = t_01 * g_new_lower + t_11 * g_new_upper
```
The "T_transpose" pattern is the key. Cross-reference: CUDA legacy at `csrc/cuda/butterfly_cuda.cu:462-463` shows the analog using shfl_xor — `grad_val[mult][item] = twiddle_val[0] * grad_val[mult][item] + __shfl_xor_sync(... twiddle_val[1] * grad_val[mult][item] ...)`. Note: that line uses `conj_wrapper(twiddle_val[*])` because the CUDA path handles complex with the same wrapper for real and complex (conj is a no-op on real). For the Triton real path we drop the conj.

**Pulling back to weights:**
```
g_T = g_y * x^T   (outer product)
  d_t_00 += g_new_lower * lower    # lower as input to "new lower"
  d_t_01 += g_new_lower * upper
  d_t_10 += g_new_upper * lower
  d_t_11 += g_new_upper * upper
```
i.e., for each of the 4 twiddle entries, the gradient is `grad_out_at_output_side * input_at_input_side`. CUDA at `:456-460` does `d_twiddle_val[0] += grad_val * conj(input_val); d_twiddle_val[1] += grad_val * conj(input_other)` — same pattern (the [0]/[1] indexing is on the "side_in" axis: [0] is the same-side input, [1] is the partner-side input).

### Complex path: Wirtinger / conjugate (D-50c verification)

For complex-holomorphic gradients with a real-valued downstream loss, PyTorch's autograd uses **conjugate Wirtinger derivatives** ([PyTorch Autograd Mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html#autograd-for-complex-numbers)). This means the gradients propagated through autograd already incorporate the conjugate convention: `g_W = ∂L/∂(W*)` in Wirtinger notation, where `W*` is the conjugate.

For `y = T @ x` with complex T and x:
```
g_x = conj(T)^T @ g_y     i.e., g_x = conj(T^T) @ g_y
g_T = g_y @ conj(x)^T     i.e., d_t_ij += g_y_i * conj(x_j)
```

**Note on D-50c's formula:** `(a+bi) * conj(c+di) = (a+bi)(c-di) = ac + bd + i(bc - ad)`. This matches what's needed for `g_y * conj(x)`:
```
out_re = a_re * c_re + a_im * c_im    (the +bd term — note the SIGN FLIP from forward 4-FMA's -)
out_im = a_im * c_re - a_re * c_im    (the bc - ad term — note the SIGN FLIP from forward 4-FMA's +)
```

**This matches D-50c verbatim and is the LANDMINE called out:** the forward 4-FMA (`out_im = a_re * c_im + a_im * c_re`) has a PLUS in `out_im`; the conjugate path has a MINUS. Getting this sign backwards (a) silently passes fp32 tests (input.imag = 0 → conj does nothing), (b) fails the unitary `U U^* = I` analog AND the complex64 gradcheck. Test layer (c) at `n=512, batch=4096, complex64` is the load-bearing detector.

**For the d_input formula's conjugate:** the CUDA kernel at `:462-463` applies `conj_wrapper(twiddle_val[0]) * grad_val + shfl_xor(conj_wrapper(twiddle_val[1]) * grad_val)`. So d_input ALSO needs the conjugate on twiddle when IS_COMPLEX. The formula:
```
g_lower_in = conj(t_00) * g_new_lower + conj(t_10) * g_new_upper
g_upper_in = conj(t_01) * g_new_lower + conj(t_11) * g_new_upper
```
applied as 4 complex multiplies with conjugate. In the 4-FMA form (using `(a+bi) * conj(c+di) = ac+bd + i(bc-ad)` where a+bi is the conj'd thing — twiddle — and c+di is the grad — but actually we apply conj on the LEFT operand of the multiply, not the right; equivalent: swap conjugation conventions). Concretely:

`conj(t) * g` where `t = t_re + i*t_im`, `g = g_re + i*g_im`:
```
out_re = t_re * g_re + t_im * g_im     (conj on left flips sign of t_im on cross-term)
out_im = t_re * g_im - t_im * g_re
```

This is the SAME form as `g_y * conj(x)`, just with t and g swapped. The kernel can use one helper `complex_mul_conj_left(a_conj, b)` reused for both d_twiddle (with `a=x`, `b=g_y` → returns `conj(x)*g_y`, equivalent to `g_y*conj(x)` since both are complex multiplies) and d_input (with `a=t`, `b=g`).

**Confidence:** HIGH — formula verified against CUDA legacy `:456-485` AND consistent with [PyTorch Autograd Mechanics — Complex Numbers](https://docs.pytorch.org/docs/stable/notes/autograd.html#autograd-for-complex-numbers).

## In-Kernel Reverse Stage Walk Within a 3-Stage Tile (Gap #2)

Phase 7's forward kernel (`op.py:201-319`) walks stages forward:
```python
for stage_offset in tl.static_range(STAGE_COUNT):
    idx = STAGE_START + stage_offset   # counter increments 0..STAGE_COUNT-1
    # ... compute new_lower, new_upper, store ...
```

For Phase 8 backward, **reverse the loop direction**:
```python
for stage_offset in tl.static_range(STAGE_COUNT - 1, -1, -1):   # decrements
    idx = STAGE_START + stage_offset   # idx walks high-to-low *within the group*
    # ... compute d_t_local + d_input_new, store d_input_new + atomic_add d_t ...
```

**Critical insight (Phase 7's lesson, verified at op.py:201-202 + summary p41-42):** `STAGE_START` is a COUNTER (range 0..log_n-1), not an absolute stage index. The `INCREASING_STRIDE: tl.constexpr` does the counter→log_stride direction mapping inside the kernel (`log_stride = idx if INCREASING_STRIDE else LOG_N - 1 - idx`). Phase 8 inherits this verbatim — the reverse-walk is over counter-values, the direction flag is unchanged.

**Partner indexing in reverse walk** is identical to forward — `partner = pos ^ stride` is symmetric. The XOR pairing is the same in both directions; the only thing that changes is the per-stage transformation: forward does `new = T @ x`, backward does `d_x = T^T @ d_y` and `d_T += d_y * x^T` (or conjugate variants).

**Reading the trail in the backward kernel:** The "input" to a backward stage `idx` (the activation at the start of that forward stage) is `trail[stage_global_idx - 1]` where `stage_global_idx = block * log_n + idx`. The backward starts from `grad_out` (the output of the LAST stage = stage `log_n*nblocks - 1`) and walks back. After processing reverse stage-group covering stages `(s, s+1, s+2)`, the new `d_input_current` becomes the upstream grad for the next reverse group `(s-3, s-2, s-1)`.

**Sketch (3-stage backward group, real path):**
```python
@triton.jit
def _butterfly_backward_kernel(
    twiddle_ptr, trail_ptrs (3 stages worth or one with stage offset),
    grad_in_ptr, grad_out_ptr,
    d_twiddle_scratch_ptr,
    n, nstacks, block_idx, nblocks,
    STAGE_START: tl.constexpr, STAGE_COUNT: tl.constexpr,
    INCREASING_STRIDE: tl.constexpr, LOG_N: tl.constexpr,
    IS_COMPLEX: tl.constexpr, TILE_N: tl.constexpr,
):
    row_id = tl.program_id(0)
    bn_id = tl.program_id(1)
    nstack_idx = bn_id % nstacks
    tile_offsets = tl.arange(0, TILE_N)
    col_start = row_id * TILE_N
    pos = col_start + tile_offsets
    # ... pointer base computation identical to forward ...

    # Load grad_in (the gradient flowing FROM later stages INTO this group) into registers
    g = tl.load(grad_in_ptr + row_base + pos)

    # Walk stages in REVERSE counter order
    for stage_offset in tl.static_range(STAGE_COUNT - 1, -1, -1):
        idx = STAGE_START + stage_offset
        log_stride = idx if INCREASING_STRIDE else LOG_N - 1 - idx
        stride = 1 << log_stride
        tile_partner = tile_offsets ^ stride
        is_lower = (tile_offsets & stride) == 0

        # Load the input to THIS stage from the trail (i.e., the activation BEFORE this stage ran).
        # Trail index for stage idx in this block = (block * log_n) + idx  →
        # but in the recompute scheme, trail[i] = output of stage i. So input to stage idx = trail[idx-1] for idx>0
        # OR the very original input for idx=0 block=0. Concretely the wrapper computes the right base ptr to pass.
        x = tl.load(trail_in_ptr + row_base + pos)
        x_partner = tl.load(trail_in_ptr + row_base + (col_start + tile_partner))

        # Load twiddle for this stage (same indexing as forward)
        pair_flat = (col_start >> 1) + (tile_offsets // (2*stride)) * stride + (tile_offsets % stride)
        twiddle_stage_base = twiddle_sb_base + idx * twiddle_stage_stride
        pf4 = pair_flat * 4
        t_00 = tl.load(twiddle_ptr + twiddle_stage_base + pf4 + 0)
        t_01 = tl.load(twiddle_ptr + twiddle_stage_base + pf4 + 1)
        t_10 = tl.load(twiddle_ptr + twiddle_stage_base + pf4 + 2)
        t_11 = tl.load(twiddle_ptr + twiddle_stage_base + pf4 + 3)

        # ---- d_twiddle accumulation (per-program reduce + atomic) ----
        # For each in-tile position, compute the 4 d_t contributions:
        # If is_lower: this position contributes to t_00 (with self=x) and t_01 (with partner=x_partner)
        # If is_upper: this position contributes to t_10 (with partner=x_partner=lower of pair) and t_11 (with self=x=upper)
        # Equivalently:
        #   d_t_00 += g[lower] * x[lower]
        #   d_t_01 += g[lower] * x[upper]
        #   d_t_10 += g[upper] * x[lower]
        #   d_t_11 += g[upper] * x[upper]
        # The cleanest single-program reduce: each in-tile lane computes its
        # OWN contribution conditionally based on is_lower (mask off other lanes
        # via tl.where), then the reduce across the tile aggregates pair-by-pair.

        # Per-lane contributions (masked so each pair_flat slot reduces correctly)
        g_lower_eff = tl.where(is_lower, g, 0.0)             # g if lower-side, 0 otherwise
        g_upper_eff = tl.where(is_lower, 0.0, g)             # g if upper-side, 0 otherwise
        x_lower_eff = tl.where(is_lower, x, 0.0)
        x_upper_eff = tl.where(is_lower, 0.0, x)             # NB: when is_lower=False, x IS the upper
        # for the partner side, we want x_partner (the "other" side)
        xp_lower_eff = tl.where(is_lower, 0.0, x_partner)    # if upper, x_partner is the lower
        xp_upper_eff = tl.where(is_lower, x_partner, 0.0)    # if lower, x_partner is the upper

        # Contributions (still TILE_N-wide vectors, each lane is 0 for "wrong side")
        dt_00_contrib = g_lower_eff * x_lower_eff
        dt_01_contrib = g_lower_eff * xp_upper_eff
        dt_10_contrib = g_upper_eff * xp_lower_eff
        dt_11_contrib = g_upper_eff * x_upper_eff

        # Within this tile, each PAIR (groups of 2*stride positions sharing one
        # twiddle pair_flat entry) needs its own reduction. Two tiles at different
        # row_id never share pair_flat, but within one tile multiple pair_flats
        # can coexist. Use reshape + tl.sum(axis=...) — see §"Per-program reduce
        # semantics" below for the canonical form.

        # ---- d_input update ----
        # d_lower_new = t_00 * g_lower + t_10 * g_upper
        # d_upper_new = t_01 * g_lower + t_11 * g_upper
        # For each lane: if is_lower compute d_lower_new, else compute d_upper_new
        partner_g = tl.load(some_register_holding_partner_grad_after_a_barrier)
        # ... barrier dance similar to forward ...
        new_g = tl.where(is_lower, t_00 * g + t_10 * partner_g,
                                    t_01 * partner_g + t_11 * g)
        # Note: when is_lower=True, "g" IS g_lower, "partner_g" IS g_upper.
        # When is_lower=False, "g" IS g_upper, "partner_g" IS g_lower.
        # Hence is_lower branch gets (t_00 * g_lower + t_10 * g_upper), upper branch gets (t_01 * g_lower + t_11 * g_upper).
        # ... store new_g back to output_ptr via the same barrier dance Phase 7 uses ...

    # After the 3 stages, write the final d_input tile to grad_out_ptr (= the input to the next reverse group)
    tl.store(grad_out_ptr + row_base + pos, g_after_loop)
```

The barrier dance for partner-load between stages is IDENTICAL to Phase 7's `op.py:199, 287, 292, 315, 319` (`tl.debug_barrier()` before+after each store). The kernel uses `output_ptr`-as-scratch for the d_input within the program — the same lesson Phase 7 learned at SUMMARY p41 ("Out_ptr-as-scratch shuffle with tl.debug_barrier — Triton has no in-register XOR-gather primitive").

**Subtle gotcha:** The barrier pattern needs the PARTNER's gradient and PARTNER's activation. Both of these come from XOR-pair lanes. The same `tl.store` → `tl.debug_barrier` → `tl.load(partner_offset)` pattern Phase 7 uses for activations applies — separately for `g` (the in-flight gradient) and for `x` (the trail-loaded activation). Allocate one scratch buffer (the `output_ptr`-as-scratch role) for `g`; load `x` and `x_partner` directly from `trail_in_ptr` (these are constants for the stage — no inter-stage dependency, so no barrier needed for activations within a stage).

**Confidence:** MEDIUM-HIGH on the algorithmic skeleton; the exact ordering of barriers within the reverse 3-stage tile will require careful execution-time validation (Phase 7 SUMMARY documents the FIRST attempt at the forward kernel had a STAGE_START semantic bug that produced abs errors >1e30 — backward has similar bug surface).

## Per-Program `tl.sum` Reduce Semantics for `d_twiddle` Accumulation (Gap #3)

**Reinterpret SC#3 carefully.** Phase 7's grid is `(n_row_tiles, batch_size * nstacks)` = `(n // TILE_N, batch * nstacks)`. So:
- `program_id(0)` = column-tile index (which set of TILE_N consecutive positions in a row)
- `program_id(1)` = `(batch, nstack)` row id (flattened)

**A single program handles one (column-tile, batch-row) pair.** Within that program's `TILE_N` positions, multiple butterfly-pairs coexist (e.g., at stride=1, TILE_N=8 holds 4 pairs (0,1), (2,3), (4,5), (6,7)). For `d_twiddle`, each pair_flat index in the tile contributes to 4 twiddle entries `(t_00, t_01, t_10, t_11)` at that pair_flat.

**The reduce is across:**
1. **The in-tile pair multiplicity** (e.g., 4 pairs at stride=1 in TILE_N=8) — within ONE program. `tl.sum` on a reshape `(n_pairs_in_tile, 2*stride) → axis=1` collapses each pair_flat's 2*stride contributions to one value per pair. Equivalently: at each pair_flat, contributions come from `2*stride` positions in the tile (the `stride` "lower" positions + `stride` "upper" positions). Each contributes a different t_ij. So the reduce is along the within-pair axis.
2. **The batch × nstacks axis** (which is `program_id(1)`) — across programs. This is the `tl.atomic_add` axis.

**Concrete reduce form within one program (real path):**
```python
# Per-position contribs computed above (each TILE_N-wide, with is_lower/is_upper masking)
# Reshape to (n_pairs_in_tile, 2*stride) where each row is one pair_flat's positions
n_pairs_in_tile = TILE_N // (2 * stride)
dt_00_per_pair = tl.sum(tl.reshape(dt_00_contrib, (n_pairs_in_tile, 2*stride)), axis=1)
dt_01_per_pair = tl.sum(tl.reshape(dt_01_contrib, (n_pairs_in_tile, 2*stride)), axis=1)
dt_10_per_pair = tl.sum(tl.reshape(dt_10_contrib, (n_pairs_in_tile, 2*stride)), axis=1)
dt_11_per_pair = tl.sum(tl.reshape(dt_11_contrib, (n_pairs_in_tile, 2*stride)), axis=1)

# Now atomic-add each pair's 4 d_t entries
# Per-pair offsets in the d_twiddle_scratch buffer
pair_flat_in_tile = tl.arange(0, n_pairs_in_tile)
pair_flat = (col_start >> 1) + pair_flat_in_tile  # global pair_flat (per the forward kernel's indexing)
twiddle_pair_base = twiddle_stage_base + pair_flat * 4
tl.atomic_add(d_twiddle_scratch_ptr + twiddle_pair_base + 0, dt_00_per_pair)
tl.atomic_add(d_twiddle_scratch_ptr + twiddle_pair_base + 1, dt_01_per_pair)
tl.atomic_add(d_twiddle_scratch_ptr + twiddle_pair_base + 2, dt_10_per_pair)
tl.atomic_add(d_twiddle_scratch_ptr + twiddle_pair_base + 3, dt_11_per_pair)
```

This is **4 atomic_add calls per program per stage**, each writing a `n_pairs_in_tile`-wide vector. Per [Triton tutorial example "atomic kernel"](https://triton-lang.org/main/python-api/generated/triton.language.atomic_add.html) and [Triton GitHub Issue #7125](https://github.com/triton-lang/triton/issues/7125), `tl.atomic_add` accepts a Block (vector) of values and a corresponding Block of pointers; it performs per-lane atomic adds. Verified on the dev host with `triton 3.6.0 / PyTorch 2.11`.

**At log_n=11, TILE_N=4096 (the last stage group for INCREASING_STRIDE=True):** the largest stage in the group has `log_stride=11`, so `stride=2048`, `n_pairs_in_tile = 4096 / (2 * 2048) = 1`. So `dt_*_per_pair` is a single scalar and the atomic_add reduces to one atomic per twiddle entry. At smaller stages within the same group (e.g., log_stride=9, stride=512), `n_pairs_in_tile = 4096 / 1024 = 4` — 4 atomics per twiddle entry, 16 total per program. Atomic traffic is bounded.

**Total atomic traffic across all programs:** per stage, `(n_row_tiles * batch * nstacks)` programs each do `4 * n_pairs_in_tile = 4 * TILE_N / (2*stride) = 2 * TILE_N / stride` atomic_add lane-writes. Summing across the full stage: `(n / TILE_N) * batch * nstacks * 2 * TILE_N / stride = 2 * n * batch * nstacks / stride` atomic writes per stage. At stride=1, log_n=11, batch=4096, nstacks=1, that's `2 * 2048 * 4096 * 1 = 16.7M` atomic adds. **HIGH but expected** (CUDA does the same volume at the warp level). Pre-`tl.sum` reduce inside each program multiplies amortization by `2*stride`, so at stride=2048 traffic is only ~16k atomics.

**Confidence:** HIGH on the formula; HIGH-MEDIUM on the in-Triton reshape+sum form — `tl.reshape` is documented and works on register-resident tensors with constexpr shapes, but if Triton 3.6's `tl.reshape` has a constexpr-shape constraint the executor may need to substitute `tl.sum(contrib, axis=...)` over a different layout. Worst case: skip the reshape, loop over pairs at JIT time via `tl.static_range`.

## d_twiddle_scratch Shape + Offset Arithmetic (Gap #4)

**Twiddle base layout** (verified at `_torch_ref/butterfly.py:17` + Phase 7 `op.py:384-386`):
```
twiddle.shape == (nstacks, nblocks, log_n, n//2, 2, 2)
```

**Real-path scratch:**
```python
d_twiddle_scratch = torch.zeros_like(twiddle, dtype=torch.float32)  # SAME shape
```
Total bytes: `nstacks * nblocks * log_n * n//2 * 4 * 4`. At log_n=11, nstacks=1, nblocks=2: `2 * 11 * 1024 * 4 * 4 = 360 KB` — negligible.

**Offset arithmetic — identical to Phase 7 forward kernel `op.py:174-181`:**
```
twiddle_stack_stride = nblocks * LOG_N * 2 * n          (= nblocks * log_n * (n//2) * 4)
twiddle_block_stride = LOG_N * 2 * n
twiddle_stage_stride = 2 * n                            (= (n//2) * 4)
twiddle_sb_base = nstack_idx * twiddle_stack_stride + block_idx * twiddle_block_stride
twiddle_stage_base = twiddle_sb_base + idx * twiddle_stage_stride
# Within stage: pair_flat * 4 + (0,1,2,3) for (t_00, t_01, t_10, t_11)
```
Phase 8 reuses these formulas VERBATIM for `d_twiddle_scratch` because the scratch has the same shape as twiddle.

**Complex64 scratch (D-50b):**
```python
d_twiddle_scratch_shape = twiddle.shape + (2,)   # appended re/im axis via view_as_real flatten
# concretely (nstacks, nblocks, log_n, n//2, 2, 2, 2)
d_twiddle_scratch = torch.zeros(d_twiddle_scratch_shape, dtype=torch.float32, device=twiddle.device)
```

Offsets:
```
# Each "logical" twiddle slot occupies 2 fp32 floats (re, im)
twiddle_stack_stride_c = nblocks * LOG_N * 2 * n * 2
twiddle_block_stride_c = LOG_N * 2 * n * 2
twiddle_stage_stride_c = 2 * n * 2                       (= (n//2) * 4 * 2)
# Within stage: pair_flat * 8 + (0,1,...,7) for (t_00_re, t_00_im, t_01_re, t_01_im, t_10_re, t_10_im, t_11_re, t_11_im)
```
Matches Phase 7 forward complex64 path at `op.py:170-178` — `pf8 = pair_flat * 8` and 8 consecutive offsets. Phase 8 reuses this pointer math for the scratch.

**Final cast at callback boundary (D-50a):**
```python
if is_complex:
    # d_twiddle_scratch is fp32 with trailing (2,) axis
    d_twiddle = torch.view_as_complex(d_twiddle_scratch.contiguous())
    # This produces a complex64 tensor of shape (nstacks, nblocks, log_n, n//2, 2, 2)
else:
    d_twiddle = d_twiddle_scratch.to(twiddle.dtype)
```

`torch.view_as_complex` requires the input to be contiguous and have its last dim == 2 with stride 1. Both hold for a freshly allocated `torch.zeros(shape, dtype=torch.float32)`. Cross-reference Phase 7 `op.py:494-497`.

**Confidence:** HIGH — direct transcription of Phase 7's verified forward pointer math.

## Recompute-Into-Trail Launch Shape (Gap #5)

**Option A: kernel writes directly to `trail_out[stage_global_idx]`.** Phase 7's kernel takes `output_ptr` as an arg (`op.py:81`) — the wrapper passes the ping-pong buffer. To redirect to trail, the kernel could just take a different output pointer per launch. The wrapper computes the trail-slot base pointer for stage `stage_global_idx` and passes it as `output_ptr` (offset into a pre-allocated big trail buffer of shape `(log_n * nblocks, batch, nstacks, n)`).

**Option B: wrapper copies/views ping-pong results into trail post-launch.** Wrapper preserves Phase 7's ping-pong, then `trail[i].copy_(dst_buf)` after each stage.

**Recommend Option A.** Reasoning:
- ZERO extra memory traffic vs. the ping-pong + copy form.
- Phase 7's wrapper already passes `dst_buf` as the kernel's `output_ptr`. Substituting a slice into `trail` is one pointer-arithmetic change.
- The kernel doesn't know anything about ping-pong; it writes to `output_ptr + row_base + pos` at the end of its stage group. Pointing `output_ptr` at `trail[stage_global_idx]` is transparent.
- **Important wrinkle:** Phase 7's kernel does NOT write per-stage to its output buffer — it ping-pongs ACROSS stage-group launches, but WITHIN a launch the 3 stages all write to the same `output_ptr`-as-scratch buffer. So `trail[stage_global_idx]` only captures the **end of the stage GROUP**, not each intermediate stage. The CONTEXT.md "trail" wording implies "one slot per stage" but really we want "one slot per stage GROUP" (`ceil(log_n / 3) * nblocks` slots per nblock-row).

**Re-read D-49's trail shape:** `(log_n * nblocks, batch, nstacks, n)`. CONTEXT.md says "trail[stage_idx]" — but with the 3-stage-group launch shape, we have only `ceil(log_n / 3) * nblocks` launch outputs, not `log_n * nblocks` per-stage activations.

**Resolution (clarification for planner):** Either:
- **(a) Allocate trail with shape `(ceil(log_n / 3) * nblocks, batch, nstacks, n)`** — one slot per launch. The backward then walks groups in reverse, with each backward group's kernel input being the corresponding trail slot. This is the natural fit for 3-stage group launches.
- **(b) Allocate trail with shape `(log_n * nblocks, batch, nstacks, n)`** — one slot per stage. To populate this, the forward recompute path would need a kernel modification to write intermediate per-stage activations (extra TILE_N stores per stage, ~3× memory traffic). Defeats the recompute efficiency.

**Recommend (a).** The math is unchanged — each backward group consumes the trail slot from the forward group of the same index. **Adjust CONTEXT.md's trail shape note in the plan** to use `(ceil(log_n / 3) * nblocks, batch, nstacks, n)` (or equivalently, `(num_launches_per_nblock * nblocks, batch, nstacks, n)`). Peak memory at log_n=11, nblocks=2, batch=4096, nstacks=1: `ceil(11/3) * 2 * 4096 * 1 * 2048 * 4 = 4 * 2 * 4096 * 2048 * 4 bytes ≈ 256 MB` for real, `512 MB` for complex64 (view_as_real flatten doubles n). **THIS IS LARGER THAN CONTEXT.md's "~88MB" estimate** because CONTEXT.md computed per-stage, but for stage-group-granularity the per-slot stride is the same (full `n`-wide buffer).

Actually re-checking: CONTEXT.md said `log_n * nblocks * batch * nstacks * n * 4 = 11 * 2 * 4096 * 1 * 2048 * 4 ≈ 740MB`. Wait, the original quoted number "~88MB" doesn't match either calculation. Let me redo: `log_n * nblocks * batch * nstacks * n * 4 bytes`. At log_n=11, nblocks=2, batch=4096, nstacks=1, n=2048: `11 * 2 * 4096 * 2048 * 4 = 738 MB`. CONTEXT.md's "~88MB" estimate appears to be incorrect — it's actually closer to 740MB at log_n=11/batch=4096. Even at the group-granularity (Option a), it's still 256MB.

**LANDMINE for plan:** Peak GPU memory is `~256-740MB` for the trail buffer alone at log_n=11/nblocks=2/batch=4096 — non-trivial. At lower batch or log_n it's fine. Plan should NOT promise "manageable" without verification on the dev host (RTX 2000 Ada has 8GB VRAM).

**Mitigation if memory is tight:** Two options the planner can defer:
- Allocate trail as fp16/bf16 (save 50%) — but trail is the FORWARD activation, not a gradient. Quantizing forward activations introduces accumulating error in the backward. Not recommended for Phase 8.
- Use Option (a) (stage-group granularity, ~256MB at worst).

**Confidence:** HIGH on the launch-mechanic option recommendation; the trail-buffer memory footprint requires the executor to verify on the dev host.

## `register_autograd` Callback Signature + Saved-Tensor Invariants Under `triton_op` (Gap #6)

**Phase 7's pattern (op.py:503-546):**
- `_setup_context(ctx, inputs, output)` saves via `ctx.save_for_backward(twiddle, input_)` and stashes scalars on `ctx`.
- `_backward(ctx, grad_out)` retrieves via `ctx.saved_tensors` and returns a 4-tuple `(grad_twiddle, grad_input, None, None)`.
- `butterfly_multiply.register_autograd(_backward, setup_context=_setup_context)` (op.py:546).

**Verified PyTorch behavior (PyTorch 2.6+ — verified against [PyTorch 2.9 torch.library docs](https://docs.pytorch.org/docs/2.9/library.html) and [User-Defined Triton Kernels tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)):**

> "The backward must be a composition of PyTorch-understood operators. If you want the backward to call Triton kernels, then those must be wrapped in triton_op as well."

This is the **load-bearing constraint** for Phase 8. **However**, the backward CAN call:
1. `triton_op`-wrapped Triton kernels (the new `_butterfly_backward_kernel` must be wrapped via `wrap_triton` or be a `@triton_op` itself).
2. Standard PyTorch ops (`torch.zeros_like`, `torch.empty_like`, `torch.view_as_real`, `torch.view_as_complex`, `F.pad`, `.contiguous()`).
3. Other already-wrapped `triton_op`s (e.g., `butterfly_multiply` itself for the recompute).

**Concrete Phase 8 backward composition:**
- Allocate trail (`torch.empty`), d_twiddle_scratch (`torch.zeros`), d_input (`torch.empty_like`) — all standard ops.
- Call `_run_forward_stage_groups(..., trail_out=trail)` which issues `wrap_triton(_butterfly_kernel)[grid](...)` launches (compose-friendly).
- Issue backward launches via `wrap_triton(_butterfly_backward_kernel)[grid](...)`.
- Cast scratch via `.to(dtype)` or `view_as_complex` (standard ops).

**`register_fake` does NOT change.** `register_fake` is only invoked under FakeTensorMode (which sees the FORWARD signature), and its role is to return a fake output tensor of the right shape/dtype/device. Phase 7's `register_fake` (`op.py:549-567`) does exactly this. The backward is invoked via the autograd graph, NOT through `register_fake` — its allocations (`torch.zeros_like`, etc.) inside `_backward` create REAL tensors at backward-time (or fake tensors at fake-mode trace time, depending on the dispatch state). PyTorch's autograd handles the fake-vs-real distinction via the dispatcher.

**Allocations inside backward under FakeTensorMode:** Standard PyTorch ops (`torch.zeros_like`, `torch.empty_like`, `torch.view_as_real`) are all fake-tensor-aware — they produce FakeTensors under FakeTensorMode with the right metadata. The `wrap_triton(...)` kernel launches under FakeTensorMode become opaque ops; the user must register `register_fake` for each `triton_op`. Since Phase 8's new backward kernel is registered as a `triton_op` (or wrapped via `wrap_triton`) with `register_fake` returning shapes equal to its outputs (d_input + nothing, since d_twiddle is via atomic mutation), the trace should work.

**LANDMINE — `mutates_args` for the backward kernel:** Phase 7's forward `@triton_op` declares `mutates_args={}` (`op.py:322`) — pure op. Phase 8's backward kernel MUTATES `d_twiddle_scratch` (atomic_add) AND `d_input` (regular stores). If wrapped as a `triton_op`, its declaration must be `mutates_args={"d_twiddle_scratch", "d_input"}` (or equivalent). **However**, the backward kernel can also simply be wrapped via `wrap_triton(...)` inside `_backward` without being a top-level `triton_op` — `wrap_triton` is documented for "make existing Triton kernels compose with triton_op", and PyTorch 2.10's `torch/_library/triton.py` source confirms this is the lighter-weight wrapping for compose-friendly launches. Recommend `wrap_triton` over a second `@triton_op` to avoid the schema-registration overhead.

**LANDMINE — trail buffer allocation inside `_backward`:** A separate concern: AOTAutograd (the layer that fakeifies backward) sometimes hoists tensor allocations to before backward. If `torch.empty(log_n * nblocks, batch, nstacks, n, ...)` runs at fake-tensor time, the tensor exists only as fake. The `wrap_triton(_butterfly_kernel)[grid](...)` recompute launch then operates on fake tensors — opaque to the kernel. Outside of `torch.compile`, this isn't an issue (eager autograd runs the backward eagerly with real tensors). Under `torch.compile`, the backward gets retraced and the trail allocation lands in the captured graph. Recommend executor verifies this end-to-end under `torch.compile(model)` as part of Plan 08-01 verification (not specified in CONTEXT.md but worth surfacing in the plan).

**Confidence:** HIGH for non-compile path (Phase 7 verified this works); MEDIUM for `torch.compile` interaction — may need empirical test.

## `tl.atomic_add` Semantics on fp32 (Gap #7)

**Verified via [Triton documentation](https://triton-lang.org/main/python-api/generated/triton.language.atomic_add.html) and dev-host probe (triton 3.6.0):**

`tl.atomic_add(pointer, val, mask=None, sem=None, scope=None)`:
- `pointer`: Block of pointers (i.e., a vector of pointers; can be ONE pointer broadcast to a tile or a tile of distinct pointers).
- `val`: Block of values with `dtype == pointer.dtype.element_ty`. Can be a 1-D vector or higher-rank tensor — atomic add is performed lane-by-lane.
- `mask`: optional Block of bools (same shape as val); lanes with mask=False are skipped.
- `sem`: defaults to `"acq_rel"`. For Phase 8's noise-tolerant `d_twiddle` reduction, **`sem="relaxed"`** is correct and faster — the only ordering constraint we need is among atomic-adds to the same address, and atomic-add is associative-ish enough (within fp32 ULP noise) that we don't need release/acquire fences across lanes.
- `scope`: defaults to `"gpu"` — fine for our case.

**Supported dtypes:** fp32 is fully supported. bf16 is unsupported (per [Triton issue #2834](https://github.com/openai/triton/issues/2834)); fp16 is brittle (per [issue #891](https://github.com/openai/triton/issues/891)). **Phase 8 sticks to fp32 scratch per SC#3 → no concern.**

**Vector vs scalar:** `tl.atomic_add` accepts both. The "vector" form is N lanes, each atomically adding `val[lane]` to `pointer[lane]`. The atomics are INDEPENDENT per lane (each lane's address might or might not collide with another program's lane). This is exactly what we need: per-program, each pair_flat in the tile contributes to a different scratch slot, and the 4 atomic_add calls (one per t_ij index in the per-pair quadruple) handle all 4 entries in parallel.

**Return value:** `tl.atomic_add` returns the value AT the pointer location BEFORE the atomic operation. We don't need this — Phase 8 just discards the return.

**Recommend explicit `sem="relaxed"` annotation** in the kernel for clarity (default is acq_rel which is unnecessarily strict).

**Confidence:** HIGH — verified against Triton 3.6 inline docstring (see `python -c "import triton.language as tl; help(tl.atomic_add)"` output captured above) AND Triton docs.

## Conjugate Sign Flip in the Complex64 Backward 4-FMA (Gap #8)

**Already derived in §"Backward algebra" — re-stating with the sign-flip taxonomy:**

| Operation | Formula | Real-component output | Imag-component output |
|-----------|---------|------------------------|------------------------|
| Forward `T*x` (Phase 7 op.py:266-274) | `(a+bi)(c+di) = (ac-bd) + (ad+bc)i` | `out_re = a_re*c_re - a_im*c_im` | `out_im = a_re*c_im + a_im*c_re` |
| Backward `g * conj(x)` for d_twiddle (D-50c) | `(a+bi)(c-di) = (ac+bd) + (bc-ad)i` | `out_re = a_re*c_re + a_im*c_im` (SIGN FLIP on im*im term: − → +) | `out_im = a_im*c_re - a_re*c_im` (SIGN FLIP on re*im term: + → −) |
| Backward `conj(t) * g` for d_input | `(a-bi)(c+di) = (ac+bd) + (ad-bc)i` | Same as above with operand-rename: `out_re = a_re*c_re + a_im*c_im` | `out_im = a_re*c_im - a_im*c_re` (SIGN FLIP only on the OUTPUT formula relative to forward) |

**Verified against:**
- The CUDA legacy backward at `csrc/cuda/butterfly_cuda.cu:456-463` which uses `conj_wrapper(input_val)` and `conj_wrapper(twiddle_val[*])` — the CUDA code adds conjugates explicitly via the `conj_wrapper` macro. For real inputs `conj_wrapper` is a no-op; for complex it flips the imag sign.
- [PyTorch Autograd for Complex Numbers](https://pytorch.org/blog/overview-of-pytorch-autograd-engine/) — confirms backward uses conjugate Wirtinger convention.

**The d_input formula DOES need conjugation on the twiddle.** Specifically: `g_lower_in = conj(t_00) * g_new_lower + conj(t_10) * g_new_upper` (and similarly for upper). This is NOT documented in CONTEXT.md's D-50c which only spells out the d_twiddle formula. **The planner MUST surface this** — it's the second half of the complex backward and is also a silent-pass-real-fail-complex bug surface.

**LANDMINE matrix (sign errors that pass fp32, fail complex64):**

| Error | Symptom |
|-------|---------|
| Forward 4-FMA `out_im = a_re*c_im - a_im*c_re` (wrong sign on second term) | Phase 7 complex64 test failures; caught by unitary test |
| d_twiddle uses forward 4-FMA instead of conjugate | Passes fp32 (input.imag=0); fails complex64 gradcheck and `d_twiddle` allclose at batch=4096 |
| d_input uses forward 4-FMA (no twiddle conj) | Passes fp32; fails complex64 gradcheck — d_input direction broken |
| d_twiddle uses conjugate on grad instead of input | `(a+bi) * conj(c+di)` ≠ `conj(a+bi) * (c+di)` — different sign convention on out_im; the math comes out wrong |

**Confidence:** HIGH — formula derivation is straightforward arithmetic; verified against CUDA legacy.

## SC#4 Verification Mechanism (Gap #9)

**CONTEXT.md's recommendation (`sys.modules['_butterfly']` absence check) is incorrect for this codebase.**

**Empirical probe on the dev host (triton 3.6, PyTorch 2.11):**
```
import sys, torch
import torch_structured

# After full torch_structured import:
'_butterfly' in sys.modules        → False
'torch_structured._butterfly' in sys.modules → False
[m for m in sys.modules if '_butterfly' in m.lower()] → []
```

**Why:** `_butterfly.so` is loaded via `torch.ops.load_library(matches[0])` at `torch_structured/butterfly/__init__.py:28, 39`. `torch.ops.load_library` calls `torch._C._dl_open` (dlopen) which loads the .so as a native shared library, NOT as a Python module. The library registers its symbols with the torch.ops dispatcher via `TORCH_LIBRARY(torch_structured, m)` (`csrc/butterfly.cpp:127-131`), but no Python `sys.modules` entry is created.

**Result:** `sys.modules['_butterfly']` is ALWAYS `False`, even when `_butterfly.so` is loaded AND `torch.ops.torch_structured.butterfly_multiply_fw` IS callable. A test that asserts `'_butterfly' not in sys.modules` is a TAUTOLOGY — passes trivially without verifying anything.

**However, the symbols ARE in `torch.ops`:**
```
sorted([a for a in dir(torch.ops.torch_structured) if not a.startswith('_')])
→ ['butterfly_multiply', 'butterfly_multiply_bw', 'butterfly_multiply_fw',
   'butterfly_multiply_triton', 'cuda_version', 'diag_mult', 'hadamard_transform', 'name']
torch.ops.torch_structured.butterfly_multiply_fw  # exists, callable
torch.ops.torch_structured.butterfly_multiply_bw  # exists, callable
```

So the absence-check on `torch.ops` registry ALSO fails — the C++ ops are always registered as long as `_butterfly.so` is loaded (which it always is, per `torch_structured/butterfly/__init__.py:39`).

**Recommended SC#4 mechanism: invocation tracking, not presence tracking.**

**Option 1 (recommended): monkey-patch the C++ op `__call__` for the duration of the test.**
```python
def test_butterfly_backward_no_cpp_symbol():
    import torch_structured
    from torch_structured.butterfly import Butterfly
    torch_structured.set_backend('triton')

    counter = {'fw': 0, 'bw': 0}
    original_fw = torch.ops.torch_structured.butterfly_multiply_fw
    original_bw = torch.ops.torch_structured.butterfly_multiply_bw
    original_mul = torch.ops.torch_structured.butterfly_multiply

    # OpOverloadPacket is a C++ object — direct monkey-patch is fragile.
    # The cleanest approach: wrap via a Python-level shim and rebind torch.ops.torch_structured
    # ... but OpOverloadPacket is read-only via getattr.
    # Alternative: rebind the legacy module's references.
    import torch_structured.butterfly.multiply as legacy_mod
    legacy_fw = legacy_mod.butterfly_multiply_fw
    # ... rebind to a counted shim ...
```

**Option 2 (simpler, recommended over Option 1): assert that the test path doesn't import `torch_structured.butterfly.multiply` (the @torch.jit.script wrapper of the C++ ops).** Since `_ops.butterfly_multiply` under `BACKEND=triton` resolves to `_triton.butterfly.op.butterfly_multiply` (verified at `_ops.py:220-224`), there's NO codepath from `_ops` into the legacy C++ ops. The test calls `torch_structured._ops.butterfly_multiply(...)` directly and asserts the output matches the oracle.

The cleanest concrete probe:
```python
def test_butterfly_backward_no_cpp_symbol():
    import torch_structured
    from torch_structured._triton.butterfly.op import butterfly_multiply as triton_op
    torch_structured.set_backend('triton')

    # SC#4 verification: _ops.butterfly_multiply IS the Triton op (not the C++ op)
    assert torch_structured._ops.butterfly_multiply is triton_op, \
        "SC#4: _ops.butterfly_multiply must be bound to the Triton kernel under BACKEND=triton"

    # Sanity: full forward+backward step works
    n = 256
    twiddle = torch.randn(1, 1, 8, 128, 2, 2, device='cuda', dtype=torch.float32, requires_grad=True)
    x = torch.randn(4, 1, n, device='cuda', dtype=torch.float32, requires_grad=True)
    out = torch_structured._ops.butterfly_multiply(twiddle, x, True, n)
    loss = out.sum()
    loss.backward()

    # Gradients exist (forward + backward both ran)
    assert twiddle.grad is not None
    assert x.grad is not None
```

**Why this works:** The `is` check verifies the dispatch binding. As long as `_ops.butterfly_multiply` resolves to the Triton `triton_op` callable, any `loss.backward()` invocation on its output flows through `register_autograd`'s `_backward` callback — which is the NEW Phase 8 body that NEVER calls `torch.ops.torch_structured.butterfly_multiply_fw/bw`. The C++ ops are unreachable from this dispatch path.

**Option 3 (most-rigorous): monkey-patch `torch.ops.torch_structured.butterfly_multiply_fw/_bw` to raise.**
```python
def test_butterfly_backward_no_cpp_symbol():
    torch_structured.set_backend('triton')

    # Replace the C++ op with a raising shim
    sentinel = object()
    raised = []
    original_fw = torch.ops.torch_structured.butterfly_multiply_fw

    # OpOverloadPacket.__call__ is implemented in C++; can't easily replace
    # But torch.library.Library has fallback registration we could attempt
    # ... see PyTorch torch.library override pattern...

    # Simpler: instrument legacy_mod which is what consumers go through
    import torch_structured.butterfly.multiply as legacy_mod
    def _fail_fw(*args, **kwargs):
        raised.append('fw')
        raise AssertionError("SC#4 violation: legacy butterfly_multiply_fw was invoked")
    def _fail_bw(*args, **kwargs):
        raised.append('bw')
        raise AssertionError("SC#4 violation: legacy butterfly_multiply_bw was invoked")
    original_legacy_fw = legacy_mod.butterfly_multiply_fw
    original_legacy_bw = legacy_mod.butterfly_multiply_bw
    legacy_mod.butterfly_multiply_fw = _fail_fw
    legacy_mod.butterfly_multiply_bw = _fail_bw

    try:
        n = 256
        twiddle = torch.randn(1, 1, 8, n//2, 2, 2, device='cuda', dtype=torch.float32, requires_grad=True)
        x = torch.randn(4, 1, n, device='cuda', dtype=torch.float32, requires_grad=True)
        out = torch_structured._ops.butterfly_multiply(twiddle, x, True, n)
        loss = out.sum()
        loss.backward()
        assert raised == [], f"SC#4 violated: legacy invocations recorded: {raised}"
    finally:
        legacy_mod.butterfly_multiply_fw = original_legacy_fw
        legacy_mod.butterfly_multiply_bw = original_legacy_bw
```

This is the SHARPEST detector — if Phase 7 or Phase 8 accidentally invoked the legacy ops, the test fires loudly.

**Recommend Option 2 + Option 3 combined** in `test_butterfly_backward_no_cpp_symbol` — Option 2 verifies the dispatch binding (cheap, deterministic); Option 3 verifies the runtime behavior (catches indirect routes through the legacy module).

**LANDMINE for planner:** Do NOT use `'_butterfly' not in sys.modules` — it's a tautology. The CONTEXT.md text says "OR" — pick a real mechanism, not the placeholder.

**Confidence:** HIGH — empirically verified on the dev host that `sys.modules['_butterfly']` is `False` in both backends, making the absence check meaningless.

## Test Surface Mechanics for `d_twiddle` Gradcheck (Gap #11)

**Layer (a) — fp64 gradcheck on n=4, batch=1, log_n=2:**
```python
def test_butterfly_backward_gradcheck_fp64(backend):
    if backend == "triton":
        pytest.skip("Triton kernel is fp32; gradcheck demands fp64 — Triton backward is exercised by allclose layer (b)/(c)")
    log_n, nstacks, nblocks, batch_size = 2, 1, 1, 1
    n = 1 << log_n  # n=4
    twiddle = torch.randn(nstacks, nblocks, log_n, n//2, 2, 2,
                          dtype=torch.float64, device='cuda', requires_grad=True)
    input_ = torch.randn(batch_size, nstacks, n,
                         dtype=torch.float64, device='cuda', requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda t, x: torch_structured._ops.butterfly_multiply(t, x, True, n),
        (twiddle, input_),
        eps=1e-6, atol=1e-5,
    )
```

**Key — gradcheck SKIPS the triton backend.** Reason: the Triton kernel is fp32 only; gradcheck demands fp64 numerical gradients. **However**, per D-52b the gradcheck on the TORCH backend exercises the SAME `register_autograd` plumbing both backends rely on (because the Triton `_backward` callback... wait — Phase 8's _backward DOES NOT delegate to `_torch_ref` anymore. It runs the Triton backward kernel.)

**LANDMINE for D-52a / D-52b interaction:** In Phase 7, the Triton backward callback delegated to `_torch_ref` via `torch.autograd.grad` — so the torch-backend gradcheck WAS testing the same plumbing both backends rely on. **In Phase 8, this is no longer true.** The Triton backward now runs its own kernel; the torch-backend gradcheck no longer exercises it.

**Solution:** Phase 8's `test_butterfly_backward_gradcheck_fp64` does two things:
1. **(legacy from Phase 7) Skip Triton backend with the same skip reason** (kernel fp32-only); torch-backend gradcheck still exercises the oracle's autograd, which is the FALLBACK path (D-49b small-N + the `BACKEND=torch` case).
2. **Add a NEW test `test_butterfly_backward_gradcheck_fp64_smallN`** that exercises the Triton backward's small-N fallback path explicitly via `log_n=1, batch=1` — Triton hits the `if log_n <= 1` branch and routes through the torch oracle, which IS gradcheck-compatible (fp64 input). This validates that the small-N fallback works correctly under autograd at fp64.

**For the large-N Triton backward, layer (b) and (c) substitute for gradcheck:**

Layer (b) — d_input allclose at n=256, batch=8, fp32:
```python
def test_butterfly_dinput_allclose_fp32(backend):
    if backend != "triton":
        pytest.skip("d_input comparison vs oracle is meaningful only for triton")
    log_n, nstacks, nblocks, batch_size = 8, 1, 1, 8
    n = 1 << log_n
    twiddle = torch.randn(nstacks, nblocks, log_n, n//2, 2, 2, device='cuda',
                          dtype=torch.float32, requires_grad=True)
    input_ = torch.randn(batch_size, nstacks, n, device='cuda',
                         dtype=torch.float32, requires_grad=True)
    grad_out = torch.randn(batch_size, nstacks, n, device='cuda', dtype=torch.float32)

    # Triton path
    twiddle_t = twiddle.detach().clone().requires_grad_()
    input_t = input_.detach().clone().requires_grad_()
    out_t = torch_structured._ops.butterfly_multiply(twiddle_t, input_t, True, n)
    out_t.backward(grad_out)

    # Oracle path
    twiddle_o = twiddle.detach().clone().requires_grad_()
    input_o = input_.detach().clone().requires_grad_()
    out_o = butterfly_ref(twiddle_o, input_o, True, n)
    out_o.backward(grad_out)

    assert torch.allclose(input_t.grad, input_o.grad, rtol=1e-5, atol=1e-6)
```

Layer (c) — d_twiddle allclose at n=512, batch=4096, fp32:
```python
def test_butterfly_dtwiddle_allclose_fp32(backend):
    if backend != "triton":
        pytest.skip("d_twiddle parity check is meaningful only for triton")
    log_n, nstacks, nblocks, batch_size = 9, 1, 1, 4096
    n = 1 << log_n  # n=512
    # Same dual-path pattern as layer (b)
    # Tolerance: rtol=1e-3, atol=1e-4 per D-52 — the atomicAdd noise floor at batch=4096
    assert torch.allclose(twiddle_t.grad, twiddle_o.grad, rtol=1e-3, atol=1e-4)
```

**Known issue (Pitfall 8 from PITFALLS.md):** "gradcheck Fails Because the Reference Backward and the Triton Backward Were Both Hand-Derived." Phase 8's reference IS `torch.autograd.grad(butterfly_multiply_torch, ...)` — autograd-of-oracle, NOT hand-derived. This is the canonical safe pattern.

**gradcheck and atomic-add noise:** `torch.autograd.gradcheck` runs the function MANY times (one per input element, per direction). Each invocation re-runs the backward with potentially different atomic-add ordering, giving slightly different `d_twiddle` values. **At fp64 the gradcheck DOES tolerate this** — the analytical-vs-numerical comparison tolerance is per-element and the atomic-add reordering noise at small batch (gradcheck small case is `n=4, batch=1`) is below the gradcheck tolerance. Verified: Phase 7 ran fp64 gradcheck through the oracle's autograd path without issue.

**For Phase 8 large-N gradcheck:** The fp64 gradcheck at `log_n=2, batch=1` (the SC#1 layer (a) case) goes through the Triton backward kernel only on triton backend. Since the kernel is fp32-only, this SKIPS triton backend. The torch-backend gradcheck DOES exercise the small-N fallback path (`log_n <= 1`) on triton (the `log_n=2` case is JUST above the fallback threshold — fix the spec to `log_n=1` if you want the gradcheck to exercise the small-N path on triton too). Actually re-reading SC#1: the gradcheck is at `n=4, batch=1, log_n=2` — and `log_n=2 > 1`, so it routes through the FULL Triton backward kernel on triton backend. This **WILL** fire the wrapper's dtype assert (Phase 7 op.py:376) because gradcheck uses fp64. So the triton skip in the gradcheck test is correct: gradcheck cannot exercise the Triton kernel directly.

**For Triton-backward correctness at small log_n where fp32 noise is minimal:** Add a `test_butterfly_backward_kernel_smallcase_fp32` that runs `n=4, batch=4096, log_n=2, fp32` and compares to autograd-of-oracle within fp32 tolerance (`rtol=1e-4, atol=1e-5`). This is the "fp32 gradcheck-equivalent" that exercises the kernel.

**Confidence:** HIGH — pattern is verified against Phase 7's test structure + PITFALLS Pitfall 8.

## Performance Baseline Strategy for 08-02 (Gap #12)

**Phase 7 `07-BASELINE.json` schema (already exists):**
```json
{
  "rows": [
    {
      "kernel": "butterfly_multiply",
      "dtype": "fp32" | "complex64",
      "log_n": 8 | 9 | 10 | 11,
      "nstacks": 1, "nblocks": 1,
      "wall_ms_p50": <float>, "wall_ms_p95": <float>,
      "reference_torch_ref_p50": <float>,
      "measured_at": "<ISO8601>", "gpu": "<name>"
    }
  ]
}
```
8 rows (4 log_n × 2 dtypes). Verified at `.planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json`.

**Recommend Phase 8 extension schema (extend the same file):**
- Add a `direction` field: `"forward"` (existing rows get this) or `"backward"`.
- Phase 8 Plan 08-02 adds 8 NEW rows with `direction: "backward"` covering the same `log_n × dtype` grid.
- TEST-04 (Phase 9) reads both `direction` values for the parity gate.

**Measurement harness — mirror Phase 7's:**
- `triton.testing.do_bench` for the Triton path (TEST-04 mandate).
- `torch.cuda.Event` warmup + measurement loops for the reference.
- 10 warmup iterations, 100 measurement iterations.
- batch_size=64 (matches Phase 7's choice — much smaller than backward's worst-case batch=4096, but Phase 7's choice was for forward-path consistency).
- Same `log_n × dtype` grid.

**For backward:** The harness must trigger a backward pass. Two options:
- **(a)** Time only the backward kernel by calling `out.sum().backward()` inside the measurement loop with `retain_graph=True` (the trail allocation + recompute happens each iteration). This measures the FULL backward callback overhead.
- **(b)** Pre-allocate trail + grad_out once, time only the backward kernel launches (skipping the recompute). This isolates the kernel cost.

**Recommend (a)** — the user-visible cost is the full backward callback; (b) hides the recompute overhead that Phase 8 deliberately accepts per D-49.

**LANDMINE for the baseline harness:** Phase 7's harness was written as a "baseline-mark" pytest test (per Plan 07-02 mention). Phase 8's executor can either:
- Extend the existing harness to call backward in addition to forward, OR
- Write a separate `tests/_baseline_butterfly_backward.py` script that runs as a standalone CLI.

Phase 7's actual implementation (per 07-02-PLAN.md line 167) suggested either approach was acceptable. Phase 8 inherits this discretion.

**Confidence:** HIGH on schema extension; recommendation HIGH on (a) for measurement form.

## Runtime State Inventory

This is a code-only phase (no rename/refactor/migration), so the runtime state categories are all empty:

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — Phase 8 doesn't touch any persistent store | none |
| Live service config | None — no external services | none |
| OS-registered state | None — no OS-level registration | none |
| Secrets/env vars | `TORCH_STRUCTURED_BACKEND` (env var) — already documented; D-53 inherits the existing behavior | none — value unchanged |
| Build artifacts | `_butterfly.so` (existing, copied into worktree per Phase 7 precedent) — Phase 8 does NOT modify | none — phase explicitly does not touch csrc/ per D-53a |

**Verified:** Greppd for all "rename" / "rebrand" / "migration" candidates — none apply.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | All | ✓ | 3.13 | — |
| PyTorch (≥2.6) | triton_op, register_autograd | ✓ | 2.11.0+cu130 | — |
| Triton (≥3.0) | @triton.jit, wrap_triton, tl.atomic_add | ✓ | 3.6.0 | — |
| CUDA | GPU execution | ✓ | 13.0 (PyTorch-side) / 0.0 (legacy .so — version mismatch known) | — |
| NVIDIA GPU | Kernel execution | ✓ | RTX 2000 Ada Generation Laptop GPU | — |
| `_butterfly.so` (legacy C++) | SC#4 verification (the thing we verify is NOT invoked) | ✓ (already built/copied) | CUDA stamp 0.0 (still loads symbols) | n/a — Phase 8 deliberately avoids this path |
| `pytest` | Test execution | ✓ | (project requirement, verified working in Phase 7) | — |
| `_butterfly_kernel` from Phase 7 (op.py:77-321) | Recompute path for trail | ✓ | committed in Phase 7 | — |

All dependencies satisfied. No blockers.

## Validation Architecture

(Skipped — `workflow.nyquist_validation` is `false` in `.planning/config.json`. Per phase instructions, no Validation Architecture section emitted.)

## Security Domain

(Skipped — `security_enforcement` is not set in `.planning/config.json` and the phase is internal kernel work with no auth / session / external-input surface.)

## Don't Hand-Roll (Library-Level)

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Complex multiply in Triton | Custom `tl.complex64`-style shim | `view_as_real` + IS_COMPLEX constexpr + explicit 4-FMA per `04-COMPLEX-LAYOUT.md:58-76` | Triton has no complex dtype (PITFALLS §1); the codebase already settled on the 4-FMA layout in Phase 4 |
| Atomic accumulator for d_twiddle | Hand-rolled CAS loop | `tl.atomic_add` on fp32 scratch | Triton primitive is correct, fast, and deterministic-within-ULP-of-fp32 |
| Backward derivation | Hand-derive `d_input` and `d_twiddle` formulas independently | Use `torch.autograd.grad(_butterfly_multiply_torch, ...)` as the oracle for verification (D-52b) | PITFALLS Pitfall 8: hand-derived references introduce sign/transpose bugs that survive testing |
| FakeTensorMode meta kernel for backward | Custom register_fake for `_butterfly_backward_kernel` | Compose existing PyTorch ops (`zeros_like`, `empty_like`, `view_as_real`) inside `_backward` callback | `register_autograd`'s callback is automatically fake-tracable when composed from PyTorch ops + `wrap_triton`-wrapped kernels |
| Trail buffer storage | Custom dtype variants (fp16/bf16) | fp32 — even when twiddle is fp32 | Storage consistency + future-compatibility with bf16/fp16 inputs (TRI-FUT-01); per CONTEXT.md "specifics" §3 |
| SC#4 verification probe | `sys.modules` absence check | Direct dispatch-binding `is` check + monkey-patch shim raising on legacy call | Empirically verified: `sys.modules['_butterfly']` is ALWAYS False (Gap #9) |

## Common Pitfalls

### Pitfall 1: Conjugate sign error in complex64 d_twiddle 4-FMA
- **What goes wrong:** d_twiddle formula `g * conj(x)` written as `g * x` (no conjugation) or as `conj(g) * x` (wrong-side conjugation).
- **Why it happens:** Phase 7's forward 4-FMA at `op.py:266-274` is `(a+bi)(c+di) = (ac-bd) + (ad+bc)i`. The visible-difference is just two minus signs flipping to plus. Engineer cargo-cults the forward formula without re-deriving for conjugate.
- **How to avoid:** Derive `g * conj(x) = (a+bi)(c-di) = (ac+bd) + (bc-ad)i` from scratch in a code comment. Write test layer (c) at complex64, batch=4096 — fires loudly on this bug.
- **Warning signs:** fp32 tests pass; complex64 d_twiddle differs from oracle by sign on imag axis; unitary `U U^* = I` analog test fails after a backward pass updates the twiddle.

### Pitfall 2: d_input formula missing conjugate on twiddle
- **What goes wrong:** d_input formula `T^T @ g` (no conjugate) used instead of `conj(T)^T @ g` for the complex path. CONTEXT.md D-50c only specifies the d_twiddle conjugate; d_input is implicit.
- **Why it happens:** D-50c spotlight on d_twiddle obscures the symmetric requirement on d_input.
- **How to avoid:** Plan must explicitly state d_input formula for complex64: `conj(t_00) * g_lower + conj(t_10) * g_upper` for the lower-side new gradient (and symmetric for upper). Cross-reference CUDA legacy `:462-463` which conj_wrappers BOTH the twiddle entries for the grad update.
- **Warning signs:** Complex64 d_input matches in magnitude but differs in phase by a complex sign on cross-terms.

### Pitfall 3: Trail buffer dimension mismatch (one slot per stage vs. one slot per stage-group)
- **What goes wrong:** CONTEXT.md D-49 says trail shape `(log_n * nblocks, batch, nstacks, n)` (one slot per stage), but the recompute via 3-stage-group launches naturally produces one output per launch = `ceil(log_n/3) * nblocks` slots.
- **Why it happens:** D-49 says "one per stage" because the BACKWARD will consume per-stage inputs; the FORWARD recompute via Phase 7's launches produces per-stage-group outputs.
- **How to avoid:** Plan defines `n_launches_per_nblock = ceil(log_n / 3)` and allocates `trail = torch.empty(n_launches_per_nblock * nblocks, batch, nstacks, n)`. Or: trail is shape `(log_n * nblocks, ...)` and the forward recompute writes ALL intermediate stage activations (modify kernel to store per-stage — defeats reuse). Recommend the former.
- **Warning signs:** Backward kernel reads wrong trail slot; gradients diverge by the activation contribution of skipped intermediate stages.

### Pitfall 4: SC#4 sys.modules tautology
- **What goes wrong:** Test asserts `'_butterfly' not in sys.modules` — passes trivially.
- **Why it happens:** Phase author assumed `_butterfly.so` would register as a Python module, but `torch.ops.load_library` uses dlopen at the C level — never enters sys.modules.
- **How to avoid:** Use Option 2 + Option 3 from §"SC#4 Verification Mechanism" — dispatch-binding `is` check + monkey-patch shim on `torch_structured.butterfly.multiply.butterfly_multiply_fw/_bw`.
- **Warning signs:** The test "always passes" even when you intentionally inject a `butterfly_multiply_fw` call into the path.

### Pitfall 5: Atomic-add ordering noise exceeds D-52 envelope
- **What goes wrong:** `d_twiddle` allclose fails at batch=4096 even though formula is correct, because the cross-program atomic-add reordering creates fp32 noise exceeding `rtol=1e-3, atol=1e-4`.
- **Why it happens:** PITFALLS §2 — naive atomic_add to bf16/fp16 has 3 ulp noise. fp32 scratch reduces this to ~1 ulp per add, but at `batch * n/2 = 4096 * 256 ~ 1M` atomics per twiddle slot, noise accumulates to ~`sqrt(1M) * 1e-7 ~ 1e-4`. The D-52 envelope (`atol=1e-4`) is calibrated for exactly this.
- **How to avoid:** Per-program `tl.sum` reduce (per D-50a) cuts atomic counts by `TILE_N / (2*stride)` — at TILE_N=4096, stride=1, factor=2048. Total atomics per twiddle slot drops to ~500, fp32 noise ~`sqrt(500) * 1e-7 ~ 2e-6` — well within envelope.
- **Warning signs:** Test passes at batch=128 but fails at batch=4096; absolute error scales with `sqrt(batch)`.

### Pitfall 6: Backward kernel ping-pong vs. forward direction
- **What goes wrong:** Phase 7's forward ping-pongs `buf_a` ↔ `buf_b` across stage-group launches; Phase 8's backward must mirror but in reverse. Forgetting to reverse the ping-pong direction means later backward groups read stale gradients.
- **Why it happens:** Backward is "the same kernel run backwards" — easy mental shortcut, but the buffer rotation direction matters.
- **How to avoid:** Wrapper code carefully tracks `src_grad_buf` vs. `dst_grad_buf` and swaps after each backward launch (mirror Phase 7 op.py:488). Initial `src_grad_buf` is `grad_full` (padded grad_out). Final `dst_grad_buf` (after last reverse-stage-group, last reverse-block) holds the final d_input full-n tensor.
- **Warning signs:** d_input is wrong at sub-pages of n; error is concentrated at specific block boundaries.

### Pitfall 7: log_n=2 fp64 gradcheck routes around the Triton backward
- **What goes wrong:** SC#1 layer (a) gradcheck at `n=4, batch=1, log_n=2` skips the Triton backend (kernel is fp32-only). The torch backend exercises the oracle's autograd graph but NOT the Triton backward kernel.
- **Why it happens:** gradcheck demands fp64 inputs; Triton kernel rejects via the dtype assert.
- **How to avoid:** Add a fp32 small-case allclose test (`n=4, batch=4096, log_n=2`) comparing Triton backward to oracle-autograd at fp32 tolerance. Cover the Triton-kernel-specific path that gradcheck cannot.
- **Warning signs:** All gradcheck tests pass but real training runs produce wrong gradients on small-batch / small-n cases.

### Pitfall 8: Trail allocation Peak GPU memory exceeds available
- **What goes wrong:** At log_n=11, nblocks=2, batch=4096, n=2048, the trail buffer at stage-granularity is ~740MB; at stage-group-granularity ~256MB. Plus complex64 doubles. On the dev host (8GB RTX 2000 Ada), this is fine but on smaller cards or larger batches it can OOM.
- **Why it happens:** CONTEXT.md's "~88MB" estimate is incorrect; actual is 256MB+ (real) or 512MB+ (complex64) at the stage-group granularity at log_n=11/nblocks=2/batch=4096.
- **How to avoid:** Plan acknowledges the actual memory footprint in the wrapper docstring. The executor verifies on the dev host before declaring the implementation done. For very-large cases (batch≥16384), Phase 9 may need to revisit the trail strategy.
- **Warning signs:** OOM at the verify step for log_n=11, batch=4096; intermittent OOM in nightly comprehensive tests.

## Code Examples

### Pattern 1: Trail-allocation + recompute + atomicAdd-into-fp32-scratch (real, illustrative)

```python
# Inside _backward(ctx, grad_out):
def _backward(ctx, grad_out):
    twiddle, input_ = ctx.saved_tensors
    increasing_stride = ctx.increasing_stride
    output_size = ctx.output_size

    batch_size, nstacks, input_size = input_.shape
    nblocks = twiddle.shape[1]
    log_n = twiddle.shape[2]
    n = 1 << log_n
    is_complex = twiddle.is_complex()

    # D-49b small-N fallback
    if log_n <= 1:
        twiddle_d = twiddle.detach().requires_grad_(True)
        input_d = input_.detach().requires_grad_(True)
        with torch.enable_grad():
            out = _butterfly_multiply_torch(twiddle_d, input_d, increasing_stride, output_size)
        gt, gi = torch.autograd.grad(out, [twiddle_d, input_d], grad_out)
        return gt, gi, None, None

    # Pad input to n + ensure contiguous (mirror Phase 7 op.py:408-410)
    input_padded = (F.pad(input_, (0, n - input_size)) if input_size < n else input_[:, :, :n]).contiguous()

    # Allocate trail (stage-group granularity per §"Recompute-Into-Trail Launch Shape")
    n_launches_per_nblock = (log_n + 2) // 3   # = ceil(log_n / 3)
    trail_n = n * (2 if is_complex else 1)     # view_as_real flatten width
    trail = torch.empty(n_launches_per_nblock * nblocks, batch_size, nstacks, trail_n,
                        dtype=torch.float32, device=input_padded.device)

    # Allocate scratch + d_input + ping-pong grad buffers
    scratch_shape = twiddle.shape + ((2,) if is_complex else ())
    d_twiddle_scratch = torch.zeros(scratch_shape, dtype=torch.float32, device=twiddle.device)
    d_input_buf_a = torch.empty(batch_size, nstacks, n, dtype=input_.dtype, device=input_.device)
    d_input_buf_b = torch.empty_like(d_input_buf_a)

    # Recompute forward into trail (D-49a)
    _run_forward_stage_groups(
        twiddle, input_padded, increasing_stride, output_size,
        trail_out=trail, is_complex=is_complex,
    )

    # Pad grad_out from output_size up to n (mirror forward pad/trim pattern)
    grad_full = F.pad(grad_out, (0, n - output_size)) if output_size < n else grad_out
    # Initial source grad buffer = padded grad_out
    d_input_buf_a.copy_(grad_full)
    src_grad = d_input_buf_a
    dst_grad = d_input_buf_b

    # Walk reverse stage groups + reverse nblocks
    cur_increasing_stride = increasing_stride
    # First flip cur_increasing_stride (nblocks - 1) times to reach the LAST block's
    # direction (mirror oracle line 32 forward toggle).
    if (nblocks - 1) % 2 == 1:
        cur_increasing_stride = not cur_increasing_stride

    launch_idx_global = n_launches_per_nblock * nblocks - 1
    for block in range(nblocks - 1, -1, -1):
        for group_start in range(log_n - (log_n % 3 or 3), -1, -3):
            counter_count = min(3, log_n - group_start)
            # Largest log_stride in this group (mirrors forward op.py:460-466)
            if cur_increasing_stride:
                max_log_stride = group_start + counter_count - 1
            else:
                max_log_stride = log_n - 1 - group_start
            tile_n = 1 << (max_log_stride + 1)
            n_row_tiles = n // tile_n
            grid = (n_row_tiles, batch_size * nstacks)
            num_warps = _pick_num_warps(tile_n)

            trail_slot = trail[launch_idx_global]   # the activation BEFORE this group ran
            wrap_triton(_butterfly_backward_kernel)[grid](
                twiddle_work,
                trail_slot,            # input for this backward group
                src_grad,              # incoming gradient
                dst_grad,              # outgoing gradient (= upstream for next reverse group)
                d_twiddle_scratch_work,
                n, nstacks, block, nblocks,
                STAGE_START=group_start, STAGE_COUNT=counter_count,
                INCREASING_STRIDE=cur_increasing_stride,
                LOG_N=log_n, IS_COMPLEX=is_complex, TILE_N=tile_n,
                num_warps=num_warps,
            )
            src_grad, dst_grad = dst_grad, src_grad
            launch_idx_global -= 1
        cur_increasing_stride = not cur_increasing_stride

    # After loop, src_grad holds the final d_input (just-written dst became src)
    d_input_full = src_grad

    # Cast scratch + trim d_input
    if is_complex:
        d_twiddle = torch.view_as_complex(d_twiddle_scratch.contiguous())
    else:
        d_twiddle = d_twiddle_scratch.to(twiddle.dtype)
    d_input_out = d_input_full[:, :, :input_size]

    return d_twiddle, d_input_out, None, None
```

### Pattern 2: `_run_forward_stage_groups` helper (refactor of Phase 7 op.py:322-501)

```python
def _run_forward_stage_groups(
    twiddle: torch.Tensor,
    input_padded: torch.Tensor,
    increasing_stride: bool,
    output_size,
    *,
    trail_out: Optional[torch.Tensor] = None,    # NEW Phase 8 hook
    is_complex: bool = False,
) -> torch.Tensor:
    """Phase 7 stage-group launch loop, factored out per D-49a.

    When trail_out is None, behavior is identical to Phase 7's wrapper
    (ping-pong between buf_a/buf_b, return final result).

    When trail_out is not None, each stage-group launch writes its output to
    trail_out[launch_idx] instead of ping-ponging. The function does NOT
    return a meaningful tensor in this mode (caller already has trail).
    """
    # ... preserve Phase 7's setup verbatim (buf_a / buf_b alloc, etc.) ...
    # ... only the per-launch `wrap_triton(...)` call site is conditionally
    #     redirected to write into trail_out[launch_idx] when not None ...
```

### Pattern 3: SC#4 test (recommended Option 2 + 3 combined)

```python
def test_butterfly_backward_no_cpp_symbol():
    """SC#4: BACKEND=triton must not invoke any symbol from csrc/butterfly.cpp at runtime."""
    import torch_structured
    import torch_structured.butterfly.multiply as legacy_mod
    from torch_structured._triton.butterfly.op import butterfly_multiply as triton_op
    torch_structured.set_backend('triton')

    # Part 1: dispatch-binding assertion (cheap, deterministic)
    assert torch_structured._ops.butterfly_multiply is triton_op, \
        "SC#4: _ops.butterfly_multiply must be bound to the Triton kernel"

    # Part 2: runtime invocation tracking
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
        twiddle = torch.randn(1, 1, log_n, n//2, 2, 2, device='cuda',
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

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Phase 7 backward via `torch.autograd.grad(_torch_ref.butterfly_multiply_torch(...))` | Phase 8: full Triton backward kernel with atomic-add fp32 scratch | Phase 8 (D-49) | TRI-04 complete; backward runs ~oracle-speed → Triton-speed (~5-10× faster expected at large batch) |
| Save-during-forward strategy (~720MB at log_n=11/batch=4096) | Recompute-into-trail strategy (~256MB at log_n=11/batch=4096) | Phase 8 (D-49 rejected save-during-forward) | Lower peak memory; one extra forward-recompute pass per backward |
| Forward kernel ping-pong only | Forward kernel `trail_out`-redirection via Phase 7 helper refactor | Phase 8 (D-49a) | Recompute path reuses the same kernel; zero new kernel source for forward |
| CUDA legacy backward `butterfly_multiply_untied_forward_backward_max5_fast_cuda_kernel` (csrc/cuda/butterfly_cuda.cu:497, 5-stage tile) | Triton 3-stage tile backward | Phase 8 (D-50) | Phase 8 stays at 3-stage; 5-stage parity deferred to Phase 9 perf gate |

**Deprecated/outdated:**
- Hand-derived backward references (PITFALLS Pitfall 8) — replaced by autograd-of-oracle (D-52b).
- bf16/fp16 atomicAdd attempts (PITFALLS Pitfall 2) — replaced by fp32 scratch + boundary cast (SC#3, D-50a).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `tl.atomic_add` in Triton 3.6 accepts a Block (vector) of values + Block of pointers; per-lane independent atomic | "`tl.atomic_add` semantics on fp32" | LOW — verified via Triton docs + dev-host probe |
| A2 | `tl.reshape` accepts constexpr shapes for in-register reshape-then-sum reduce pattern | "Per-program `tl.sum` reduce semantics" | MEDIUM — Triton 3.x may have constexpr-shape constraints; executor verifies and substitutes loop+sum if reshape unavailable |
| A3 | `register_autograd` callback inside `triton_op` can allocate intermediate `torch.empty`/`torch.zeros` tensors AND issue `wrap_triton(...)` launches without breaking FakeTensorMode | "`register_autograd` callback signature + saved-tensor invariants" | MEDIUM — verified via PyTorch 2.9 docs + Phase 7 success with `torch.autograd.grad(...)` inside backward (which also allocates intermediate tensors via the oracle); `torch.compile` interaction not specifically tested for Phase 8's new pattern |
| A4 | Phase 7's existing trail-buffer memory estimate "~88MB" in CONTEXT.md is incorrect; actual is ~256MB (real) / ~512MB (complex64) at log_n=11/nblocks=2/batch=4096 with stage-group granularity | "Recompute-Into-Trail Launch Shape" | LOW — math is straightforward; planner adjusts the docstring + plan to reflect the actual footprint |
| A5 | `torch.ops.torch_structured.butterfly_multiply_fw` and `_bw` are loaded via `torch.ops.load_library` (dlopen) and DO NOT register in `sys.modules` | "SC#4 Verification Mechanism" | LOW — empirically verified on dev host |
| A6 | The Triton 3.6 `_butterfly_kernel`'s reverse-direction static_range walk (`tl.static_range(STAGE_COUNT - 1, -1, -1)`) compiles correctly with unrolling | "In-Kernel Reverse Stage Walk" | LOW-MEDIUM — `tl.static_range` accepts (start, stop, step) per Triton docs; executor verifies on first compile |
| A7 | d_twiddle computed across pair contributions reduces correctly via `tl.sum(tl.reshape(contrib, (n_pairs_in_tile, 2*stride)), axis=1)` to a per-pair scalar | "Per-program `tl.sum` reduce semantics" | MEDIUM — depends on A2; fallback is explicit loop |
| A8 | The trail buffer's stage-group granularity (one slot per launch) is sufficient for the backward — i.e., each backward stage-group launch only needs the activation at the START of the forward group, not per-stage intermediate activations | "Recompute-Into-Trail Launch Shape" | HIGH — load-bearing for the recompute strategy. If the backward kernel needs MIDDLE-of-group activations, the trail must be per-stage (and the forward recompute must be modified to emit per-stage activations). The CUDA legacy at `csrc/cuda/butterfly_cuda.cu:418-443` shows per-stage activations stored in shared memory (`input_val[step][mult][item]`) — the equivalent in Triton is per-stage registers. Within a 3-stage backward group, the kernel can RECOMPUTE intermediate activations from the start-of-group activation by walking the FORWARD direction with the same 3 stages. The trail then stores only start-of-group activations. **Executor must verify the kernel pattern supports this — if not, fall back to per-stage trail.** |

## Open Questions

1. **Does the backward kernel need start-of-group activations (1 per group) OR per-stage activations (3 per group)?**
   - What we know: CUDA stores per-stage in shared memory and walks both directions in one kernel. Phase 8 splits forward (trail) and backward (Triton kernel) into separate launches.
   - What's unclear: Whether the 3-stage backward Triton kernel can re-run the forward walk for stages 0-1 internally given only the start-of-group activation, or whether it needs middle activations from the trail.
   - Recommendation: The kernel CAN re-walk forward 0→1→2 using the start-of-group activation + twiddles. This means trail at stage-group granularity is sufficient. Plan 08-01 documents this explicitly; executor verifies.

2. **Does `_run_forward_stage_groups(..., trail_out=trail)` work correctly under FakeTensorMode and torch.compile?**
   - What we know: Phase 7's forward works under FakeTensorMode via `register_fake`. Phase 8 backward is invoked via `register_autograd`, which is well-defined under autograd but less-tested under `torch.compile`.
   - What's unclear: Whether the FakeTensorMode trace of the backward (which allocates trail + scratch + d_input AND launches kernels) produces a correct compiled graph.
   - Recommendation: Plan 08-01 includes a smoke test `test_butterfly_compile_backward` running `torch.compile(model).backward()` to verify end-to-end compose. If fails, document the issue and ship a workaround (e.g., `@torch._dynamo.allow_in_graph` or refactor allocations outside the backward).

3. **Does `tl.sum(tl.reshape(contrib, (n_pairs_in_tile, 2*stride)), axis=1)` compile in Triton 3.6 with constexpr shapes?**
   - What we know: Triton 3.x supports `tl.reshape` and `tl.sum`; both accept constexpr-derived shapes.
   - What's unclear: Whether the executor encounters a Triton compile error on this specific pattern (some Triton versions limit reshape with non-power-of-2 dimensions, but 2*stride is always power-of-2 here).
   - Recommendation: Try the reshape+sum form first; fallback is explicit `tl.static_range` over pairs with scalar `tl.sum` per lane (slower but always works).

## Sources

### Primary (HIGH confidence)
- `/home/claroche/torch-structured/torch_structured/_triton/butterfly/op.py:77-321` — Phase 7's forward kernel; the algorithmic template Phase 8 mirrors.
- `/home/claroche/torch-structured/torch_structured/_triton/butterfly/op.py:322-501` — Phase 7's wrapper with ping-pong + stage-group launch loop; the helper-extraction target per D-49a.
- `/home/claroche/torch-structured/torch_structured/_triton/butterfly/op.py:503-543` — Phase 7's `_setup_context` + `_backward`; Phase 8 replaces lines 516-543 only.
- `/home/claroche/torch-structured/torch_structured/_torch_ref/butterfly.py:12-33` — the autograd-oracle; the verifiable reference per D-52b.
- `/home/claroche/torch-structured/csrc/cuda/butterfly_cuda.cu:419-489` — CUDA forward-backward shared-twiddle kernel; the algorithmic blueprint for backward (`d_twiddle_val[*] += grad_val * conj(input_val); gpuAtomicAdd(...)`).
- `/home/claroche/torch-structured/csrc/cuda/butterfly_cuda.cu:497-606` — CUDA backward host wrapper; the orchestration pattern Phase 8 mirrors at the Python level (forward-recompute then backward walk).
- `/home/claroche/torch-structured/csrc/butterfly.cpp:127-131` — `TORCH_LIBRARY(torch_structured, m)` registration of `butterfly_multiply`, `_fw`, `_bw`; the symbols SC#4 verifies are not invoked.
- `/home/claroche/torch-structured/torch_structured/butterfly/__init__.py:28, 39` — `torch.ops.load_library` mechanism for `_butterfly.so`; explains why `sys.modules['_butterfly']` is always False.
- `/home/claroche/torch-structured/.planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/04-COMPLEX-LAYOUT.md:58-76` — 4-FMA template + view_as_real wrapper boundary.
- `/home/claroche/torch-structured/.planning/phases/07-butterfly-multiply-forward-triton/07-01-SUMMARY.md` — counter-based STAGE_START fix lessons + clone fallback + tolerance scale-awareness.
- `/home/claroche/torch-structured/.planning/research/PITFALLS.md` §1, §2, §8, §11 — complex64 layout, bf16/fp16 atomic_add, hand-derived backward dangers, view_as_real strides.
- `triton 3.6.0 / torch 2.11.0+cu130` on dev host (RTX 2000 Ada Generation Laptop GPU, CUDA 13.0) — verified `tl.atomic_add` signature, `sys.modules` behavior, `torch.ops.torch_structured.*` op registration.

### Secondary (MEDIUM confidence)
- [triton.language.atomic_add — Triton documentation](https://triton-lang.org/main/python-api/generated/triton.language.atomic_add.html) — mask, sem, scope semantics; vector vs. scalar.
- [Scalar vs. Tensor atomic, Synchronization & Broadcasting Behavior · Issue #7125 · triton-lang/triton](https://github.com/triton-lang/triton/issues/7125) — known inconsistencies in atomic semantics (relevant for sem= selection).
- [Triton issue #2834 — Bf16 with tl.dot and tl.atomic_add](https://github.com/openai/triton/issues/2834) — bf16 atomic_add unsupported (validates fp32 scratch design).
- [Triton issue #891 — atomic_add for fp16 non-deterministic compile segfaults](https://github.com/openai/triton/issues/891) — fp16 atomic_add brittleness (validates fp32 scratch design).
- [Using User-Defined Triton Kernels with torch.compile — PyTorch Tutorials](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html) — `triton_op` + `register_autograd` + `register_fake` patterns.
- [torch.library — PyTorch 2.9 documentation](https://docs.pytorch.org/docs/2.9/library.html) — `register_autograd` invariants.
- [PyTorch Autograd Mechanics — Complex Numbers](https://docs.pytorch.org/docs/stable/notes/autograd.html#autograd-for-complex-numbers) — conjugate Wirtinger convention.
- [PyTorch Autograd Engine Overview](https://pytorch.org/blog/overview-of-pytorch-autograd-engine/) — autograd's conjugate-Wirtinger-as-default behavior for complex inputs.

### Tertiary (LOW confidence — verified anyway via direct dev-host probe)
- Dev-host empirical probe: `sys.modules['_butterfly']` is False; `torch.ops.torch_structured.butterfly_multiply_fw/_bw` are registered via `torch.ops.load_library` invocation in `torch_structured/butterfly/__init__.py:28, 39`.

## Metadata

**Confidence breakdown:**
- Backward algebra: HIGH — formula derivation matches CUDA legacy + autograd documentation
- In-kernel reverse walk pattern: MEDIUM-HIGH — algorithm clear; barrier orchestration needs careful implementation
- d_twiddle scratch + offsets: HIGH — direct transcription of Phase 7's verified pointer math
- Recompute-into-trail launch shape: MEDIUM — trail shape recommendation (stage-group vs. stage granularity) corrects CONTEXT.md's "~88MB" estimate
- `register_autograd` invariants under triton_op: MEDIUM — Phase 7 verified eager; torch.compile not exhaustively tested
- `tl.atomic_add` semantics: HIGH — verified via Triton docs + dev-host probe
- Complex conjugate sign flip: HIGH — formula derivation + CUDA cross-reference + Wirtinger convention
- SC#4 verification mechanism: HIGH — empirically corrected CONTEXT.md's sys.modules approach (which was tautology)
- Gradcheck test mechanics: HIGH — pattern verified in Phase 7
- Perf baseline strategy: HIGH — extension of Phase 7's existing schema

**Research date:** 2026-05-28
**Valid until:** 2026-06-27 (30 days for stable kernel work; Triton 3.6 / PyTorch 2.11 are recent stable releases)

## RESEARCH COMPLETE
