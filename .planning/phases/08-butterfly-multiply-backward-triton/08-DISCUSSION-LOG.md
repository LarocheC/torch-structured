# Phase 8: butterfly_multiply Backward (Triton) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-28
**Phase:** 8-butterfly_multiply Backward (Triton)
**Areas discussed:** Activation strategy, Plan split, Backend rollout, SC#4 verification, Recompute granularity, Atomic-add tolerance, Backward launch shape

---

## Activation Strategy

**Question:** How should the backward kernel access pre-stage activations? (Backward needs the input at each butterfly stage to compute `d_twiddle`.)

| Option | Description | Selected |
|--------|-------------|----------|
| Recompute forward inside backward | Backward callback re-runs the Triton forward to materialize intermediates, then computes gradients walking back. Zero memory overhead; Phase 7's `_setup_context` stays as-is. ~2× backward compute. Matches CUDA legacy's forward-backward-fused pattern at `csrc/cuda/butterfly_cuda.cu:497`. | ✓ |
| Save intermediates during forward | Modify `_setup_context` to save `log_n*nblocks` activation tensors (one per stage). Zero recompute. ~720MB at log_n=11/nblocks=2/batch=4096 fp32. Heavies the forward setup. | |
| Fused forward-backward kernel (single launch) | Mirror CUDA's `butterfly_multiply_untied_forward_backward_max5_fast_cuda_kernel`: one launch that walks forward into shared/scratch then walks backward. Tightest memory + compute. Significant kernel-body complexity; no Phase 6/7 pattern to crib from. | |

**User's choice:** Recompute forward inside backward (Recommended)
**Notes:** Aligns with CUDA-legacy pattern; trades ~2× compute for zero memory overhead. Captured as D-49.

---

## Plan Split

**Question:** How should Phase 8 be split into plans?

| Option | Description | Selected |
|--------|-------------|----------|
| Mirror Phase 7 — 2 plans by dtype | 08-01 fp32 backward + three-layer gradcheck + d_input/d_twiddle correctness + SC#4 stack-trace verification. 08-02 complex64 backward + perf baseline. Pre-wire IS_COMPLEX per D-41a pattern. | ✓ |
| Single plan — fp32 + complex64 together | One PLAN.md ships both dtypes. Simpler tracking; larger blast radius; breaks the small-plans/atomic-commits rhythm. | |
| Split by gradient — d_input first, d_twiddle second | 08-01 d_input kernel only (no atomicAdd). 08-02 d_twiddle kernel + complex64. Isolates the load-bearing atomicAdd path. Backward incomplete after 08-01 (gradcheck can't really pass). | |

**User's choice:** Mirror Phase 7 — 2 plans by dtype (Recommended)
**Notes:** Direct transcription of Phase 7 07-01/07-02 shape. Captured as D-51 + D-51a.

---

## Backend Rollout

**Question:** When does the Triton backward become live for the user?

| Option | Description | Selected |
|--------|-------------|----------|
| Hard switch on `BACKEND=triton` | As soon as 08-01 ships, `BACKEND=triton` routes both forward AND backward through Triton. Phase 7's `torch.autograd.grad` fallback is deleted. Clean break, matches SC#4. | ✓ |
| Gated opt-in via `TORCH_STRUCTURED_BACKEND_BW=triton` | `BACKEND=triton` keeps Phase 7's oracle backward by default; users opt in to Triton backward with a separate env var until Phase 9 perf gate validates it. Conservative; adds a second knob. | |
| Inherit Phase 7 small-N fallback for backward too | `BACKEND=triton` flips backward on for `log_n >= 2`, but `log_n <= 1` keeps the torch-oracle backward path (matching D-42a). Composable with option 1. | (composed) |

**User's choice:** Hard switch on `BACKEND=triton` (Recommended)
**Notes:** Single backend selector; matches SC#4 ('no C++ symbol invoked'). Small-N fallback inherits from Phase 7 D-42a via D-49b — this is composable, not an alternative. Captured under the In-scope block of `<domain>`.

---

## SC#4 Verification

**Question:** How is SC#4 verified — 'with `BACKEND=triton`, `loss.backward()` invokes no symbol from `csrc/butterfly.cpp`'?

| Option | Description | Selected |
|--------|-------------|----------|
| Test that `_butterfly.so` is not loaded after backward | Pytest assertion checks `sys.modules` / `torch.ops` registry after `loss.backward()` to assert no symbol from `_butterfly.so` is registered. Cheap, deterministic, every CI. | ✓ |
| Stack-trace inspection via `sys.settrace` / py-spy | Wrap `loss.backward()` with a tracer that records every C-extension call. More authoritative; heavier to wire. | |
| Build-time guarantee — don't build `_butterfly.so` in Phase 8+ CI | Skip the `setup.py` extension build entirely. Strongest signal but couples SC#4 to CI config rather than to a runtime assertion. | |

**User's choice:** Test that `_butterfly.so` is not loaded after backward (Recommended)
**Notes:** Captured as D-53; build-time guarantee (D-53a) explicitly rejected to keep `_butterfly.so` building for opt-in CUDA users (DEPR-03).

---

## Recompute Granularity

**Question:** Given recompute-forward-inside-backward, how is recompute structured?

| Option | Description | Selected |
|--------|-------------|----------|
| Materialize full per-stage trail once, then walk back | Allocate `(log_n*nblocks, batch, nstacks, n)` fp32 trail, run forward once writing each post-stage output into `trail[i]`, then issue backward launches in reverse stage-group order reading `trail[i-1]`. ~88MB at log_n=11/nblocks=2/batch=4096. Peak memory ~22× forward. | ✓ |
| Recompute per stage-group inside each backward launch | Each backward stage-group launch first recomputes its 3-stage forward in-register from a saved per-block boundary. Zero trail-buffer memory. ~O(nblocks²) compute when going back across blocks. | |
| Save only inter-block boundaries (nblocks outputs) | Save just `nblocks` intermediate outputs (one per block boundary). Each block's backward recomputes its `log_n` stages in-register from the saved boundary. ~8MB trail at nblocks=2. Kernel chains forward + backward in one launch sequence — more complex. | |

**User's choice:** Materialize full per-stage trail once, then walk back (Recommended)
**Notes:** Simpler and well within memory budget. Captured as D-49 + D-49a.

---

## Atomic-Add Tolerance

**Question:** What rtol/atol envelope locks the 'atomicAdd noise' assertion for d_twiddle at batch=4096?

| Option | Description | Selected |
|--------|-------------|----------|
| rtol=1e-3, atol=1e-4 | Matches SC#2's complex64 d_twiddle tolerance verbatim. Reuses the existing number so fp32 and complex64 share one envelope. ~sqrt(4096) × machine_eps_fp32 ≈ 1e-4. | ✓ |
| rtol=1e-4, atol=1e-5 (tighter) | Stricter envelope. Risk of flaky CI when atomicAdd reorder noise exceeds the bound. | |
| Computed envelope: sqrt(batch) × 2 × machine_eps | Compute rtol/atol from batch at test time. Scales automatically. Extra test-side math; opaque assertion failures. | |

**User's choice:** rtol=1e-3, atol=1e-4 (Recommended)
**Notes:** Captured as D-52. The fp64 gradcheck (layer a of SC#1) and the d_input allclose (layer b) use the standard `rtol=1e-5, atol=1e-6` and `rtol=1e-5, atol=1e-6` respectively; only d_twiddle layer (c) uses the looser envelope per SC#3.

---

## Backward Launch Shape

**Question:** Backward kernel launch shape — mirror Phase 7's 3-stage-tile pattern in reverse?

| Option | Description | Selected |
|--------|-------------|----------|
| Mirror Phase 7 — reverse stage-group order | `ceil(log_n/3)` backward launches per nblock, traversed in REVERSE order: `(9,10,11)` first, then `(6,7,8)`, then `(3,4,5)`, then `(0,1,2)`. Same `TILE_N` schedule, same 2-D grid, same `num_warps` schedule from D-40d. Zero structural divergence from Phase 7. | ✓ |
| Per-stage launches (1 stage per kernel) | Issue `log_n*nblocks` separate launches. Simpler kernel body; 3× more launches; breaks symmetry with forward's 3-stage tile. | |
| Single fused backward kernel for the whole nblock | One launch per nblock walks all `log_n` backward stages internally. Minimal launch overhead; large `TILE_N` ties register pressure to the full n. | |

**User's choice:** Mirror Phase 7 — reverse stage-group order (Recommended)
**Notes:** Captured as D-50. Per-program d_twiddle reduce + single atomicAdd per program is D-50a (SC#3 verbatim). Complex64 conjugate path is D-50c.

---

## Claude's Discretion

The user selected all four "Recommended" answers in the first round and all three "Recommended" answers in the second round — these recommended values are the locked decisions captured under `<decisions>`. The following items remain planner-flexible per the CONTEXT.md `### Claude's Discretion` block:

- Exact SC#4 probe mechanism (D-53 — `sys.modules` check vs `torch.ops` registry call-count delta vs `sys.settrace`)
- Trail buffer reuse vs per-call allocation
- Per-stage-group `_backward_kernel` signature (`STAGE_COUNT: tl.constexpr ∈ {1,2,3}` uniform vs separate kernels)
- Exact ordering of recompute-then-walk-back inside `_backward` (all recompute first vs interleaved)
- `d_twiddle_scratch` allocation form (`zeros_like(dtype=fp32)` vs explicit shape)
- Nblock iteration form (`range(nblocks-1, -1, -1)` vs `reversed(range(nblocks))`)
- `d_input` allocation form (`empty_like` recommended over `zeros_like`)
- d_input ping-pong between buffers across reverse stage-groups (recommended yes)
- Trail buffer `empty` vs `zeros` (recommended `empty`)

## Deferred Ideas

- 5-stage tile backward (Phase 9 perf gate)
- Save-during-forward activation strategy (rejected — may revisit in Phase 9 if recompute overhead dominates)
- Fused forward-backward single-launch kernel (CUDA `:497` pattern — deferred to Phase 9)
- Gated opt-in `TORCH_STRUCTURED_BACKEND_BW=triton` (rejected — single backend selector)
- bf16 / fp16 backward (TRI-FUT-01)
- log_n > 11
- `_butterfly.so` build verification in CI (Phase 9/10)
- `@triton.autotune` over `num_warps` / `TILE_N` (Phase 9)
- CUDA backend in `backend` conftest fixture (Phase 9 / TEST-03)
- Build-time guarantee for SC#4 (deliberately rejected — runtime assertion is primary)
- Trail buffer reuse across calls (Phase 9 perf gate)
