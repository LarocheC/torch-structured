# Phase 7: butterfly_multiply Forward (Triton) - Context

**Gathered:** 2026-05-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Port `butterfly_multiply` **forward** to Triton with fp32 + complex64 support, exposed via `torch_structured._ops.butterfly_multiply(twiddle, input, increasing_stride=True, output_size=None)`. Backward **temporarily routes through `_torch_ref.butterfly.butterfly_multiply_torch` via `register_autograd`** so Phase 7 ships before the heavy Phase 8 atomic-add backward kernel lands. The kernel uses a **multi-launch 3-stage tile** structure: `ceil(log_n / 3)` Triton launches per nblock, each launch handles 3 consecutive butterfly stages on register-resident tiles. Two plans split by dtype: 07-01 fp32 forward + tests; 07-02 complex64 + view_as_real + unitary butterfly test.

**In scope:**
- New `torch_structured/_triton/butterfly/__init__.py` + `op.py` — `@triton.jit` 3-stage-tile kernel + `@triton_op` wrapper + `register_autograd` backward (delegates to `_torch_ref` two-input gradient via `torch.autograd.grad`) + `register_fake` meta kernel. Plan 1 pre-wires `IS_COMPLEX: tl.constexpr` flag with `tl.static_assert(not IS_COMPLEX, ...)` gate; Plan 2 lights up the else branch via Phase 4 `04-COMPLEX-LAYOUT.md` 4-FMA template.
- `_ops.py` resolver Step 2 — the `butterfly_multiply` block already exists at `_ops.py:207-218` (pre-wired by Phase 4). Phase 7 verifies the `_has_triton_kernel("butterfly_multiply")` branch now lights up correctly and the per-op `log.info` line at the bottom of `_resolve()` already reports `butterfly_multiply=<actual>` (Phase 5 D-36c).
- Plan 07-01 (fp32):
  - Triton kernel real-only path; wrapper does NOT call `view_as_real` (gated)
  - Python-side `nblocks` loop + `cur_increasing_stride` toggling (mirrors `_torch_ref/butterfly.py:23-37` verbatim)
  - Python-side small-N fallback: `if log_n <= 1: return butterfly_multiply_torch(twiddle, input, ...)` — bypasses kernel for n=1, n=2 (trivial cases, kernel launch overhead dominates)
  - Python-side `F.pad(input, (0, n - input_size))` before kernel launch + `output[:, :, :output_size]` slice after (mirrors `_torch_ref/butterfly.py:18, 37`)
  - `register_autograd` backward callback computes `(d_twiddle, d_input)` via `torch.autograd.grad(_hadamard_transform_torch(twiddle.detach().requires_grad_(), input.detach().requires_grad_(), ...), [twiddle, input], grad_out)` (Phase 5 D-26 pattern, two-input variant)
  - `register_fake` returns `torch.empty(batch_size, nstacks, output_size, dtype=input.dtype, device=input.device)`
  - New `tests/test_butterfly_triton.py`: dense smoke tier (log_n in {2,4,8,10}, nstacks=1, nblocks=1, increasing_stride=True, output_size=n, fp32 only); sparse comprehensive tier marked `@pytest.mark.slow` (Cartesian over log_n ∈ {2..11}, nstacks ∈ {1,2,3}, nblocks ∈ {1,2}, increasing_stride ∈ {True,False}, output_size ∈ {n, n//2, n-1})
- Plan 07-02 (complex64):
  - Remove the `tl.static_assert(not IS_COMPLEX, ...)` gate; implement the IS_COMPLEX=True branch via Phase 4 `04-COMPLEX-LAYOUT.md` 4-FMA template (`(a+bi)(c+di) = (ac - bd) + (ad + bc)i`)
  - Wrapper: gate `view_as_real(input)` and `view_as_real(twiddle)` with `assert input.is_contiguous()` (Phase 4 Pitfall 3); `view_as_complex(out.contiguous())` on return
  - Extend `tests/test_butterfly_triton.py`: complex64 smoke + comprehensive tiers; `test_butterfly_unitary` (the `U U^* = I` gate from `PITFALLS.md §1` — the load-bearing complex-correctness detector at `test_butterfly.py:234` analog)
  - Perf baseline: at end of Plan 07-02 execution, write `.planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json` with schema `{ kernel, dtype, log_n, nstacks, nblocks, wall_ms_p50, wall_ms_p95, reference_torch_ref_p50, measured_at, gpu }` for log_n ∈ {8,9,10,11} × {fp32, complex64}. Phase 9 parity gate reads this verbatim.

**Out of scope:**
- **Backward Triton kernel** (TRI-04 / Phase 8). Phase 7's backward is the temporary torch-reference oracle. The heavy `tl.atomic_add` into `d_twiddle` reduction lands in Phase 8.
- **5-stage tile** (deferred). ROADMAP explicitly defers; the 5-stage variant is the perf-optimized version that the CUDA kernel at `csrc/cuda/butterfly_cuda.cu:288` (`butterfly_multiply_untied_forward_max5_fast_cuda_kernel`) implements. Phase 9 (TEST-04 perf gate) decides whether to land 5-stage based on Phase 7's baseline.
- **log_n > 11** — ROADMAP SC#1 caps at log_n ∈ {2..11} for the parameter grid; the multi-launch tiled kernel works for any log_n in principle, but the test surface and perf baseline only exercise up to log_n=11.
- **bf16/fp16** — TRI-FUT-01, deferred. Phase 7 is fp32 + complex64.
- **CUDA legacy backend resurrection** — the existing `_cuda_legacy/butterfly.py` try-import + `_has_cuda_legacy()` probe already exists from Phase 5; Phase 7 doesn't touch it. When `_butterfly.so` is missing AND `BACKEND=cuda` is requested, `_ops.py:217-218` falls back to `_torch_ref` with the same `log.warning` pattern (Phase 5 D-22).
- **Touching `csrc/butterfly.cpp` / `butterfly_cuda.cu`** — Phase 10 deletion candidates per DEPR-03/04. Phase 7 leaves them alone.
- **Editing existing `tests/test_butterfly.py`** — the existing nn.Module-surface tests use `unittest.TestCase` style and import from the legacy `torch_structured.butterfly` package. They continue to work because the `Butterfly` nn.Module already calls `torch_structured._ops.butterfly_multiply` via the resolver block at `_ops.py:207-218`. Phase 7 verifies they still pass; no source edits.
- **Editing `tests/conftest.py`** — Phase 6 D-39 widened the skip-gate to `_has_any_triton_kernel()` which already iterates `butterfly_multiply`. Phase 7 doesn't touch conftest.

</domain>

<decisions>
## Implementation Decisions

### Kernel structure — multi-launch 3-stage tile (Claude's discretion, locked)

- **D-40:** The Triton kernel is **multi-launch 3-stage tiled**: each `@triton.jit` launch handles exactly 3 consecutive butterfly stages within a tile. For `log_n=L`, the wrapper issues `ceil(L / 3)` launches per nblock. Tile size doubles per stage group: stages (0,1,2) use `2^3=8`-wide tile; stages (3,4,5) use `2^6=64`-wide tile; (6,7,8) use `2^9=512`-wide; (9,10,11) use `2^12=4096`-wide. Mirrors the CUDA `fwtBatch1Kernel` shape at `csrc/cuda/butterfly_cuda.cu:288` but caps at 3 stages instead of 5. Phase 9 can swap in the 5-stage variant for perf parity.
- **D-40a:** Python-side `nblocks` loop and `cur_increasing_stride` toggling. The wrapper iterates `for block in range(nblocks)` and flips `cur_increasing_stride` between blocks (mirrors `_torch_ref/butterfly.py:23-37` verbatim — including the `cur_increasing_stride = not cur_increasing_stride` after each block). Within each block, the wrapper computes the stage groups and the per-group `STAGE_START`, `STAGE_END`, `INCREASING_STRIDE` constexpr values. Each group is one `@triton.jit` launch with these flags so the kernel specializes per group at JIT time.
- **D-40b:** Register-resident tile body. Each program loads `tile_n = 2^(STAGE_START + 3)` elements per `(batch, nstack)` row into registers via `tl.load`. Runs 3 unrolled butterfly stages on the in-register tile using `tl.where` to swap partners at each stride. Single `tl.store` at the end. No shared memory needed at the 3-stage cap (max tile_n = 4096 for stages 9-11 still fits in registers for fp32; tight for complex64 at the largest stage group but still under register-spill threshold). For complex64, the kernel uses the `IS_COMPLEX: tl.constexpr` 4-FMA path per `04-COMPLEX-LAYOUT.md` template. Twiddles for the 3 stages are batch-loaded once at tile start (size = 3 × (tile_n / 2) × 4 fp32 × {1 if not IS_COMPLEX else 2}).
- **D-40c:** 2-D grid `(n_row_tiles, batch_size * nstacks)` per stage-group launch, where `n_row_tiles = n // tile_n`. Program_id(1) selects the `(batch, nstack)` row; program_id(0) selects the column tile within that row. Twiddle is shared across program_id(1) axis and benefits from L1 cache. Phase 9 perf gate may revisit the grid shape (e.g., flatten to 1-D for some launchers); locked to 2-D for Phase 7.
- **D-40d:** Plain `@triton.jit` only — no `@triton.heuristics`, no `@triton.autotune` (rejected by `wrap_triton` in PyTorch 2.6+, per Phase 4 `_ops.py:264-268`). Fixed `num_warps` schedule by tile_n: 4 for tile_n ≤ 64, 8 for tile_n in {128..1024}, 16 for tile_n ≥ 2048. Tunable in Phase 9.

### Plan split — 2 plans by dtype (Claude's discretion, locked)

- **D-41:** Two plans split by dtype:
  - **07-01: fp32 forward + tests + register_autograd + register_fake.** Kernel includes `IS_COMPLEX: tl.constexpr` flag but the `IS_COMPLEX=True` branch contains only `tl.static_assert(not IS_COMPLEX, 'complex64 lands in 07-02')`. Wrapper asserts `input.dtype == torch.float32 and twiddle.dtype == torch.float32` (NOT complex64). All real-only tests (eager fp32, gradcheck fp64 against `_torch_ref`, increasing_stride grid, output_size grid, nstacks/nblocks grid, edge cases n=1/n=2 fallback path).
  - **07-02: complex64 forward + unitary test + perf baseline.** Removes the `tl.static_assert` gate; implements the `IS_COMPLEX=True` 4-FMA branch verbatim per `04-COMPLEX-LAYOUT.md`. Wrapper gates `view_as_real(input)` + `view_as_real(twiddle)` with `assert input.is_contiguous()` + `assert twiddle.is_contiguous()` (Pitfall 3). Wrapper restores via `view_as_complex(out.contiguous())`. Tests: complex64 smoke + comprehensive tiers + `test_butterfly_unitary` (U U^* = I gate) + perf baseline JSON dump.
- **D-41a:** Pre-wire IS_COMPLEX in 07-01. Plan 1 writes the kernel signature with `IS_COMPLEX: tl.constexpr` already present and the `view_as_real` machinery in the wrapper already in place but gated by `assert not is_complex` until 07-02 removes the gate. **Zero kernel-signature refactor between plans — Plan 2 only adds code, never changes signatures.** Eliminates the integration risk of "the kernel that worked yesterday looks different today".

### Edge case handling (Claude's discretion, locked)

- **D-42:** `output_size != n` (input padded to next power-of-2, output trimmed) handled in the Python wrapper. The wrapper does `input = F.pad(input, (0, n - input_size)) if input_size < n else input[:, :, :n]` (mirrors `_torch_ref/butterfly.py:18` verbatim) before the kernel launches, and `output = output_full[:, :, :output_size]` after the final stage group (mirrors line 37). The kernel itself sees only the full-N tensor. Two extra allocations (pad creates a new tensor; final slice is a view but the kernel writes to a full-N out buffer). Simpler kernel — no masking logic, no `output_size: tl.constexpr`. Matches the torch_ref oracle exactly. Phase 9 perf gate can revisit kernel-side masking if pad cost dominates at small output_size, but Phase 7 prioritizes correctness symmetry with the oracle.
- **D-42a:** Small-N fallback (`log_n <= 1`). The wrapper checks `if log_n <= 1: return butterfly_multiply_torch(twiddle, input, increasing_stride, output_size)` and bypasses the kernel entirely. Rationale: at `n=1` there are zero butterfly stages (output is `input * twiddle` scalar); at `n=2` there's one stage (a single 2x2 multiply). Triton launch overhead dominates the computation, and tile_n=8 (the smallest 3-stage tile) is larger than `n` so the kernel would need degenerate masking. The fallback still routes through `register_autograd` (the `register_autograd` callback runs the oracle backward regardless of whether forward used the kernel or the fallback — the autograd graph is identical). The fallback IS exercised by tests at log_n=1 in the smoke tier so the path stays warm.

### Test surface + perf baseline (Claude's discretion, locked)

- **D-43:** New `tests/test_butterfly_triton.py` (top-level, mirrors `tests/test_diag_mult.py` from Phase 5 and `tests/test_dispatch.py` from Phase 4). Parametrized via the existing conftest `backend` fixture (Phase 6 D-39 widened the skip-gate to cover butterfly). Test functions:
  - **Plan 07-01:** `test_butterfly_eager_fp32`, `test_butterfly_gradcheck_fp64`, `test_butterfly_output_size_grid`, `test_butterfly_increasing_stride`, `test_butterfly_nstacks_nblocks_grid`, `test_butterfly_smallN_fallback` (verifies log_n ≤ 1 routes through `_torch_ref`).
  - **Plan 07-02:** `test_butterfly_eager_complex64`, `test_butterfly_gradcheck_complex64` (real-imag parts, Wirtinger pattern from Phase 5), `test_butterfly_unitary` (the `U U^* = I` correctness gate per PITFALLS §1).
- **D-43a:** Tiered parametrization. **Dense smoke** (runs on every CI, every test): `log_n ∈ {2, 4, 8, 10}`, `nstacks=1`, `nblocks=1`, `increasing_stride=True`, `output_size=n`, plus the unitary test at one log_n. ~5-10 cases per dtype. **Sparse comprehensive** marked `@pytest.mark.slow`: full Cartesian over `log_n ∈ {2..11} × nstacks ∈ {1,2,3} × nblocks ∈ {1,2} × increasing_stride ∈ {True,False} × output_size ∈ {n, n//2, n-1}`. Hundreds of cases, opt-in via `pytest -m slow`. Satisfies SC#1 "full parameter grid" literally but doesn't slow every CI.
- **D-43b:** Perf baseline JSON. At end of Plan 07-02 execution, run a parametrized baseline harness (e.g., `pytest tests/test_butterfly_triton.py -m baseline --baseline-out 07-BASELINE.json` or an equivalent measurement script) that writes `.planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json` with the schema documented above. The harness measures p50/p95 wall time for both backends at log_n ∈ {8,9,10,11} × {fp32, complex64}, batch_size=64, nstacks=1, nblocks=1, increasing_stride=True, output_size=n. Phase 9's TEST-04 parity gate reads this JSON to assert Triton is within 1.5× of the CUDA legacy backend (when the `.so` is built) or within 5× of `_torch_ref` (CPU fallback comparison).

### Inherited from prior phases (NOT re-discussed — locked upstream)

- **D-44 (inherits Phase 4 D-01..D-03):** Complex64 layout via `torch.view_as_real()` at the `_ops.py` wrapper boundary (zero-copy on contiguous input). Twiddle layout `(nstacks, nblocks, log_n, n/2, 2, 2)` reinterpretable as `(nstacks, nblocks, log_n, n/2, 2, 2, 2)` real via the same view. Kernel uses `IS_COMPLEX: tl.constexpr` flag — same `@triton.jit` source specializes per dtype at JIT time. 4-FMA pattern: `out_re = a_re * c_re - a_im * c_im; out_im = a_re * c_im + a_im * c_re`. No `tl.complex*` (Triton has no complex dtype; PITFALLS §1).
- **D-45 (inherits Phase 5 D-21, D-22):** `_cuda_legacy/butterfly.py` already exists with the try-import + sentinel pattern. `_has_cuda_legacy()` probe at `_ops.py` already exists. When `BACKEND=cuda` AND `_butterfly.so` is missing, `_ops.py:217-218` falls back to `_torch_ref` with the existing `log.warning` (D-22 asymmetric fallback pattern). Phase 7 does NOT touch `_cuda_legacy/butterfly.py` or the cuda fallback wiring.
- **D-46 (inherits Phase 5 D-25, D-26):** Consumer call sites use D-05 attribute access. The `Butterfly`, `ButterflyBmm`, `ButterflyUnitary`, `ButterflyBase4` nn.Modules already call `torch_structured._ops.butterfly_multiply` via the dispatch surface (set up in Phase 4). Phase 7 does NOT refactor consumer code — the existing call sites are already attribute-access-correct, and switching to Triton happens transparently via the resolver `set_backend()`.
- **D-47 (inherits Phase 5/6 D-32):** `register_autograd` + `register_fake` + `triton_op` skeleton — five-component pattern: `@triton.jit` kernel + `@triton_op` wrapper + `_setup_context` + `_backward` callback + `@register_fake` meta. Backward callback for butterfly is two-input: returns `(grad_twiddle, grad_input)` via `torch.autograd.grad(_torch_ref.butterfly_multiply_torch(twiddle, input, increasing_stride, output_size).sum(), [twiddle, input], grad_outputs=[grad_out])`. **Note:** twiddle and input must have `requires_grad=True` set inside the callback before calling `_torch_ref` so autograd traces both inputs; the wrapper passes detached copies and re-enables grad inside the callback.
- **D-48 (inherits Phase 6 D-39):** `tests/conftest.py` `backend` fixture skip-gate already uses `_has_any_triton_kernel()` which iterates `butterfly_multiply`. No conftest changes needed.

### Claude's Discretion

All gray areas resolved as Claude's discretion at user's selection. Remaining planner-flexible items:
- Exact stage-group boundary handling at non-divisible log_n (e.g., log_n=10 → groups (0,1,2), (3,4,5), (6,7,8), (9,) where the last group has only 1 stage). Planner picks: either run a degenerate 1-stage launch (cleaner, uniform kernel) or fuse the last stage into the prior group via constexpr stage count (saves one launch). Recommend the degenerate-launch approach for symmetry.
- Exact `num_warps` constants per tile_n band (D-40d gives recommended values; planner verifies against Triton's runtime warnings).
- Whether `tests/test_butterfly_triton.py` uses `@pytest.fixture(params=...)` parametrize or `@pytest.mark.parametrize` decorator — either works; pick for consistency with `tests/test_diag_mult.py`.
- Exact threshold for the perf-baseline assertion in Phase 9 (1.5× vs CUDA, 5× vs torch_ref — recorded here as a Phase 9 input, not enforced in Phase 7).
- Whether the small-N fallback (D-42a) requires its own `_torch_ref` route through `register_autograd` or just calls `butterfly_multiply_torch` directly outside the `triton_op`. Recommend the former — keeps autograd graph uniform across the small-N / large-N split.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 7 charter
- `.planning/ROADMAP.md` §"Phase 7" — phase goal, depends on Phase 6, 4 success criteria, 2 plan slots
- `.planning/REQUIREMENTS.md` §"v1.2 Requirements" → TRI-03 (the sole REQ this phase covers — `butterfly_multiply` forward fp32 + complex64)
- `.planning/REQUIREMENTS.md` §"Traceability" — confirms TRI-03 mapped to Phase 7

### Phase 4 hand-off (LOCKED — load-bearing for Phase 7)
- `.planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/04-COMPLEX-LAYOUT.md` — **CRITICAL.** D-01 `view_as_real` at wrapper boundary; D-02 wrapper template (copy verbatim for Plan 07-02); D-03 twiddle layout invariant. Kernel-side `IS_COMPLEX: tl.constexpr` 4-FMA template. Contiguity Gotcha (Pitfall 3) — Plan 07-02 wrapper MUST assert `input.is_contiguous()` and `twiddle.is_contiguous()` before `view_as_real`.
- `.planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/04-CONTEXT.md` — D-04..D-08 (dispatch + set_backend); D-09..D-10 (`_torch_ref/` layout); D-11..D-12 (torch>=2.6, triton_op pattern); D-13 (register_autograd + register_fake); D-15 (deprecation plan); D-16 (CI cache)
- `.planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/04-DEPRECATION-PLAN.md` — Phase 10 implements; Phase 7's cuda-legacy fallback is `log.warning`, not `DeprecationWarning`

### Phase 5 hand-off (LOCKED — pattern source for two-input gradient)
- `.planning/phases/05-diag-mult-triton-port/05-CONTEXT.md` — D-21 (try-import + sentinel), D-22 (per-op asymmetric fallback), **D-26 (backward via `_torch_ref` oracle — Phase 7 extends to two-input variant via `torch.autograd.grad(..., [twiddle, input], ...)`)**, D-27/D-28 (test surface pattern)
- `.planning/phases/05-diag-mult-triton-port/05-01-PLAN.md` — 7-task template; Phase 7's two plans each adopt a subset (kernel + autograd + tests, then complex extension + baseline)
- `.planning/phases/05-diag-mult-triton-port/05-01-SUMMARY.md` — concrete delta lines and code excerpts; the `_triton/diag_mult/op.py` skeleton is the literal template for `_triton/butterfly/op.py` (transcribe with butterfly kernel body + two-input backward)

### Phase 6 hand-off (LOCKED — most recent op port; closest analog except for kernel body)
- `.planning/phases/06-hadamard-triton-port/06-CONTEXT.md` — D-31..D-39: single-pass shared-memory pattern (Phase 7 diverges to multi-launch tiled per D-40); D-32 self-inverse backward (Phase 7 diverges to two-input `torch.autograd.grad` per D-47); D-37 test surface pattern (Phase 7 follows)
- `.planning/phases/06-hadamard-triton-port/06-01-PLAN.md` — 7-task plan template; Phase 7's plan structure adapts to 2 plans (per ROADMAP)
- `.planning/phases/06-hadamard-triton-port/06-01-SUMMARY.md` — `tl.debug_barrier()` lesson learned (Phase 7's register-resident tiles avoid this entirely — no inter-stage stores within a launch); the `normalize=False` default in `register_fake` lesson applies (Phase 7's fake must include defaults for `increasing_stride=True` and `output_size=None`)
- `.planning/phases/06-hadamard-triton-port/06-02-PLAN.md` — gap-closure pattern (rank-N widening); Phase 7's `_torch_ref/butterfly.py` already handles arbitrary `(batch_size, nstacks, input_size)` rank — no rank widening needed
- `.planning/phases/06-hadamard-triton-port/06-VERIFICATION.md` — verification report style; the `passed`-after-gap-closure pattern is what Phase 7 aims for

### Research outputs (milestone-wide — load-bearing for Phase 7)
- `.planning/research/PITFALLS.md` §1 — **CRITICAL.** Complex64 in Triton: no `tl.complex64` dtype; `view_as_real` + 4-FMA is the only viable path. The unitary butterfly `U U^* = I` test (`test_butterfly.py:234` analog) is the cheapest correctness gate. Plan 07-02 makes this test a quality gate.
- `.planning/research/PITFALLS.md` §3 — Phase 7 hadamard challenge was "two-pass mixed-radix in Triton" (resolved in Phase 6 via single-pass); Phase 7 butterfly challenge is "multi-stage tile with twiddle loading"
- `.planning/research/STACK.md` — `@triton.jit` + `wrap_triton` + `register_autograd` + `register_fake` API contract
- `.planning/research/ARCHITECTURE.md` — `_triton/<op>/op.py` layout pattern; `_torch_ref` + `_cuda_legacy` analog file pattern

### Project-level constraints
- `.planning/PROJECT.md` §"Current Milestone: v1.2" — `butterfly_multiply_torch` preserved as oracle + runtime fallback (TRI-07 already locked)
- `./CLAUDE.md` (project root) — `assert` for preconditions, no try/except in core lib (one exception: `_cuda_legacy/*.py` try-imports — documented honest-probe pattern from Phase 5 D-21)
- `/home/claroche/CLAUDE.md` (user-level) — `bd` for task tracking, NOT TaskCreate/TodoWrite

### Code-level references (read before editing)
- `torch_structured/_torch_ref/butterfly.py:1-46` — the pure-PyTorch oracle. Plan 07-01 wrapper mirrors lines 13-37 verbatim for nblocks loop + cur_increasing_stride toggling. Backward callback delegates to this function.
- `torch_structured/_cuda_legacy/butterfly.py` — already exists. Phase 7 does NOT touch.
- `torch_structured/_ops.py:204-228` — existing `butterfly_multiply` resolver block. The `if _has_triton_kernel("butterfly_multiply")` branch is pre-wired; once Plan 07-01 lands `_triton/butterfly/op.py`, the branch lights up automatically on `set_backend('triton')`.
- `torch_structured/_ops.py:139` — `_has_any_triton_kernel()` already iterates `butterfly_multiply` (Phase 6 D-39 widened scope; no further widening needed for Phase 7).
- `torch_structured/_triton/diag_mult/op.py:1-206` — Phase 5 deliverable; the five-component skeleton (`@triton.jit` + `@triton_op` + `_setup_context` + `_backward` + `@register_fake`). Plan 07-01 transcribes this structure with butterfly kernel body + two-input backward.
- `torch_structured/_triton/hadamard_transform/op.py:1-190` — Phase 6 deliverable; the most recent op skeleton with `IS_COMPLEX` absent (Phase 6 is real-only). Plan 07-01 follows the same shape but adds `IS_COMPLEX: tl.constexpr` and a 4-FMA branch.
- `csrc/cuda/butterfly_cuda.cu:288` — `butterfly_multiply_untied_forward_max5_fast_cuda_kernel` (the 5-stage CUDA kernel). Phase 7's 3-stage tile is the analog at 3 stages; the math is identical (per-stage butterfly multiply with twiddle), only the tile size and stage count differ.
- `tests/test_butterfly.py:234` (or equivalent line) — `Butterfly(complex=True)` unitary test. Plan 07-02 verifies this exact test passes on the Triton backend.
- `tests/test_diag_mult.py:1-119` — Phase 5 test skeleton. Plan 07-01 transcribes the structure with butterfly-specific parametrize axes.
- `tests/test_dispatch.py` — Phase 4 dispatch tests; Phase 7 should not break any of these.
- `tests/conftest.py` — `backend` fixture (Phase 6 D-39 widened). Phase 7 does NOT touch.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`butterfly_multiply_torch` at `_torch_ref/butterfly.py:13-46`** — the verbatim oracle. The Phase 7 Python wrapper mirrors lines 13-37 (nblocks loop, cur_increasing_stride toggle, F.pad input, output slice) and delegates the per-block stage computation to the Triton kernel. Backward callback delegates the entire two-input gradient to `torch.autograd.grad(_torch_ref.butterfly_multiply_torch(...), [twiddle, input], grad_out)`.
- **`_cuda_legacy/butterfly.py`** — already exists. Try-import + sentinel pattern from Phase 5 D-21. Phase 7 does NOT touch.
- **`_ops.py:204-228` butterfly_multiply resolver block** — already pre-wired by Phase 4. Once `_triton/butterfly/op.py` ships, the `if _has_triton_kernel("butterfly_multiply")` branch lights up automatically. Phase 7 verifies the binding works; no resolver edits needed.
- **`_ops.py:139` `_has_any_triton_kernel()`** — already iterates `butterfly_multiply`. No widening needed (Phase 6 D-39 covered all three current ops + butterfly).
- **`_triton/diag_mult/op.py` + `_triton/hadamard_transform/op.py`** — two prior templates for the five-component skeleton. Plan 07-01 picks whichever is closer in shape (hadamard_transform has no IS_COMPLEX, but is more recent; diag_mult has the full IS_COMPLEX + 4-FMA pattern). **Recommend transcribing from diag_mult** for the IS_COMPLEX scaffolding (even though Plan 07-01 gates it off via `tl.static_assert`).

### Established Patterns
- **Phase 5's per-op three-branch resolver block** (`_ops.py:204-228` for butterfly) — already in place. No new resolver block needed.
- **Phase 5's try-import + sentinel idiom** (`_cuda_legacy/butterfly.py`) — already in place.
- **D-05 attribute access** — `Butterfly`, `ButterflyBmm`, etc. nn.Modules already call `torch_structured._ops.butterfly_multiply`. Phase 7 does NOT refactor consumer code.
- **`assert` preconditions** — wrapper-boundary asserts in Plan 07-01: `assert input.dim() == 3` (batch, nstacks, input_size), `assert twiddle.shape == (nstacks, nblocks, log_n, n//2, 2, 2)`, `assert input.dtype == torch.float32 and twiddle.dtype == torch.float32` (Plan 07-01 ONLY; relaxed to allow complex64 in Plan 07-02), `assert output_size <= n`. Power-of-2 check on `n` is inherited via `log_n = twiddle.shape[2]` (well-formed twiddle implies n is a power of 2 by construction).

### Integration Points
- **`torch_structured.butterfly` legacy package** — the `Butterfly` nn.Module (and variants) live here. They already route through `torch_structured._ops.butterfly_multiply` via the Phase 4 dispatch surface. The existing `tests/test_butterfly.py` exercises these nn.Modules and will automatically run through the Triton path once `set_backend('triton')` is active. Phase 7 verifies the existing test still passes; no edits.
- **Existing `tests/test_butterfly.py`** — unittest-style, imports from `torch_structured.butterfly.complex_utils`. Mixed test style with modern pytest parametrize would be jarring. Phase 7 creates a NEW `tests/test_butterfly_triton.py` for the kernel-level surface and leaves the existing file alone.
- **`_ops.py` per-op `log.info` line** — the format string already includes `butterfly_multiply=%s` (Phase 5 D-36c). Phase 7 doesn't change the format; the log line will report `butterfly_multiply=triton` once the kernel is loaded.

</code_context>

<specifics>
## Specific Ideas

- **The Phase 6 `_triton/hadamard_transform/op.py` is the recency template; Phase 5 `_triton/diag_mult/op.py` is the complex-aware template.** Plan 07-01 should transcribe from diag_mult (which has the `IS_COMPLEX` scaffolding and `view_as_real` machinery in the wrapper) and adapt the kernel body to butterfly's multi-launch tiled pattern. Hadamard's single-pass + tl.debug_barrier pattern does NOT apply (register-resident tiles per D-40b avoid the barrier).
- **Stage-group computation in the wrapper.** For log_n=L and a per-block stride direction (cur_increasing_stride), the wrapper computes:
  ```python
  stage_order = list(range(log_n)) if cur_increasing_stride else list(reversed(range(log_n)))
  for group_start in range(0, log_n, 3):
      group_stages = stage_order[group_start:group_start + 3]
      tile_n = 1 << (max(group_stages) + 1)  # tile must cover the widest stride in the group
      n_row_tiles = n // tile_n
      grid = (n_row_tiles, batch_size * nstacks)
      _butterfly_kernel[grid](
          twiddle_ptr, input_ptr, output_ptr, ...,
          STAGE_START=group_stages[0], STAGE_COUNT=len(group_stages),
          INCREASING_STRIDE=cur_increasing_stride,
          IS_COMPLEX=is_complex,
          TILE_N=tile_n,
          num_warps=_pick_num_warps(tile_n),
      )
  ```
  The exact form is Plan 07-01's call; this is illustrative.
- **Two-input backward formula via `torch.autograd.grad`** — at the backward callback site:
  ```python
  def _backward(ctx, grad_out):
      twiddle, input = ctx.saved_tensors
      twiddle_d = twiddle.detach().requires_grad_(True)
      input_d = input.detach().requires_grad_(True)
      with torch.enable_grad():
          out = _torch_ref.butterfly_multiply_torch(twiddle_d, input_d, ctx.increasing_stride, ctx.output_size)
      grad_twiddle, grad_input = torch.autograd.grad(out, [twiddle_d, input_d], grad_out, retain_graph=False)
      return grad_twiddle, grad_input, None, None  # (twiddle, input, increasing_stride, output_size) — 4 returns
  ```
  Number of returns must match the 4 forward inputs. Last two are None (non-tensor int/bool args).
- **Unitary butterfly U U^* = I test** — at `test_butterfly_unitary`, build a `Butterfly(in_size=n, out_size=n, bias=False, complex=True, increasing_stride=True, init='ortho')` module, compute `U = b.matrix()` (or equivalent — verify the existing API), assert `torch.allclose(U @ U.conj().T, torch.eye(n, dtype=torch.complex64), atol=1e-4)`. Per PITFALLS §1 this is the cheapest detector of a wrong complex code path.
- **`fastfood_multiply` consumer integration sanity** — like Phase 6 SC#3, Phase 7 should verify that `Butterfly` nn.Module chains in `structured/layers.py` (`linear.py`, `circulant.py`, etc.) produce correct outputs on the Triton backend. The existing `tests/test_butterfly.py` and `tests/test_special.py` already exercise these; Phase 7 verifies they pass via `set_backend('triton')` rather than adding new tests.

</specifics>

<deferred>
## Deferred Ideas

- **5-stage tile** (D-40 explicit) — perf-optimized 5-stage tile per `csrc/cuda/butterfly_cuda.cu:288`. Phase 9 (TEST-04 perf gate) decides whether to land. Phase 7 records baseline; if Triton 3-stage is more than 1.5× slower than CUDA 5-stage, Phase 9 implements 5-stage. Otherwise stays at 3-stage.
- **Backward Triton kernel** (TRI-04 / Phase 8). The heavy `tl.atomic_add` into `d_twiddle` reduction is Phase 8's scope per ROADMAP. Phase 7's backward via `_torch_ref` is the temporary stand-in.
- **bf16 / fp16 forward** (TRI-FUT-01). Phase 7 is fp32 + complex64.
- **`log_n > 11`** — kernel works in principle for any log_n; test surface and perf baseline only exercise up to 11 per ROADMAP SC#1.
- **`_butterfly.so` build verification** — existing `setup.py` conditional build. Phase 9 may add CI matrix entry; Phase 7 stays out.
- **CUDA backend in `backend` conftest fixture** — Phase 5 D-30 / Phase 6 deferred the `"cuda"` param to Phase 9 per TEST-03. Phase 7 inherits this deferral.
- **`@triton.autotune` over (`num_warps`, `tile_n`)** — fixed values for Phase 7 per D-40d; Phase 9 may revisit if perf parity gate flags an issue.

### Reviewed Todos (not folded)
None — no pending todos surfaced for Phase 7.

</deferred>

---

*Phase: 7-butterfly_multiply Forward (Triton)*
*Context gathered: 2026-05-27*
