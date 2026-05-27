---
phase: 07-butterfly-multiply-forward-triton
plan: 02
subsystem: triton-port
tags: [triton, butterfly, complex64, view-as-real, 4-fma, unitary, perf-baseline, wirtinger-gradcheck]

requires:
  - phase: 07-01
    provides: "Multi-launch 3-stage out_ptr-as-scratch @triton.jit kernel with IS_COMPLEX: tl.constexpr pre-wired and gated; @triton_op wrapper with view_as_real / view_as_complex machinery in place (inert under fp32-only assert); 7-test smoke + 720-case slow Cartesian fp32 regression suite; _ops.py _TRITON_PACKAGE_NAMES dict; tests/conftest.py slow-marker registration"
  - phase: 04
    provides: "04-COMPLEX-LAYOUT.md:58-76 canonical 4-FMA template (a + bi)(c + di) = (ac - bd) + (ad + bc)i; Pitfall 3 contiguity-before-view_as_real gate"
  - phase: 05
    provides: "_triton/diag_mult/op.py IS_COMPLEX=True 4-FMA reference implementation (lines 64-87)"
provides:
  - "torch_structured._ops.butterfly_multiply: complete fp32 + complex64 forward dispatch-bound callable (the kernel-signature is unchanged from Plan 07-01 per D-41a)"
  - "torch_structured._triton.butterfly.op: IS_COMPLEX=True kernel branch lit up via the 4-FMA per 04-COMPLEX-LAYOUT.md:58-76; wrapper accepts {float32, complex64} (the fp32-only assert is gone, the static_assert at kernel entry is gone)"
  - "tests/test_butterfly_triton.py: 4 new tests covering complex64 forward smoke + slow comprehensive + complex128 gradcheck + the load-bearing U U^H=I unitary detector per PITFALLS §1"
  - "tests/_baseline_butterfly.py: standalone perf measurement harness (D-43b schema; NOT a pytest test — leading-underscore filename prevents auto-collection)"
  - ".planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json: 8-row perf baseline; Phase 9 TEST-04 parity gate reads this verbatim"
affects: [phase-08-butterfly-backward, phase-09-integration-hardening, phase-10-deprecation]

tech-stack:
  added: []  # No new dependencies; everything routes through torch.library.triton_op + torch.view_as_real
  patterns:
    - "Plan 07-02 IS_COMPLEX kernel-branch pattern: separate _re and _im register vectors loaded from the view_as_real layout (Option (b) from the plan), 4-FMA applied per pairwise complex multiply, `tl.where(is_lower, ..., ...)` applied independently on _re and _im (the mask is on the logical position; same mask for both halves). Mirrors _triton/diag_mult/op.py:64-87 verbatim, extended to the butterfly's 2x2 matrix-vector multiply (four pairwise complex multiplies per pair: t00*cur, t01*partner, t10*partner, t11*cur)"
    - "Twiddle pointer arithmetic in view_as_real layout: each logical complex twiddle entry occupies 2 floats; per-pair stride becomes 8 floats (4 entries x 2 floats); `pf8 = pair_flat * 8` replaces `pf4 = pair_flat * 4`. Per-stack/per-block/per-stage strides all double (`twiddle_stack_stride = nblocks * LOG_N * 2 * n * 2`, etc.)"
    - "Direct-call unitary acceptance gate: the unitary test (PITFALLS §1) cannot go through `Butterfly.forward()` because per D-46 the nn.Module routes through the legacy C++ op; calling `torch_structured._ops.butterfly_multiply(b.twiddle, identity_batch, True, n)` directly with the module's twiddle parameter exercises the Triton kernel under `set_backend('triton')`"
    - "torch.cuda.Event(enable_timing=True) GPU-time measurement: warmup 10 iter + measure 100 iter, sort, take p50 = times[len/2] and p95 = times[int(len*0.95)]. The canonical Triton/PyTorch GPU-time pattern — not time.perf_counter (which includes Python overhead)"

key-files:
  created:
    - "tests/_baseline_butterfly.py (+169 lines): standalone perf measurement harness with cuda.Event timing; 8-row baseline producer matching D-43b schema; skips cleanly on CPU-only hosts"
    - ".planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json (+100 lines): 8-row JSON baseline for Phase 9 TEST-04 parity gate; NVIDIA RTX 2000 Ada Generation Laptop GPU; Triton 1.15x-2.25x speedup vs torch_ref oracle across log_n in {8,9,10,11} x dtype in {fp32, complex64}"
    - ".planning/phases/07-butterfly-multiply-forward-triton/07-02-SUMMARY.md (this file)"
  modified:
    - "torch_structured/_triton/butterfly/op.py (+84 net lines: 153 insertions / 69 deletions): removed `tl.static_assert(not IS_COMPLEX, ...)` at kernel entry; removed `assert input.dtype == torch.float32` in wrapper preamble (replaced with broader `{float32, complex64}` gate); added IS_COMPLEX-branched seed-load and per-stage body with 4-FMA per-pair complex multiply; module docstring + wrapper docstring updates"
    - "tests/test_butterfly_triton.py (+179 net lines: 182 insertions / 3 deletions): added `test_butterfly_eager_complex64` (smoke), `test_butterfly_eager_complex64_grid` (@pytest.mark.slow comprehensive 720-case), `test_butterfly_gradcheck_complex64` (D-47 Wirtinger acceptance — complex128 on torch backend), `test_butterfly_unitary` (the load-bearing PITFALLS §1 U U^H=I detector — direct-call into _ops.butterfly_multiply)"

key-decisions:
  - "IS_COMPLEX kernel implementation: Option (b) per plan — separate `_re` and `_im` register vectors loaded from the view_as_real layout. Cleaner than Option (a) (interleaved single vector with `tl.where(pair_mask_re, ..., ...) + tl.where(pair_mask_im, ..., ...)`) because the per-stage body becomes textually parallel to the fp32 path but with 2x the loads / stores. Mirrors `_triton/diag_mult/op.py:64-87` verbatim and produces correct results across all parametrized cases"
  - "Twiddle stride doubling for view_as_real: the kernel computes `twiddle_stack_stride = nblocks * LOG_N * 2 * n * 2` and `twiddle_stage_stride = 2 * n * 2` (and the in-tile per-pair offset `pf8 = pair_flat * 8`) when IS_COMPLEX=True. This is a constexpr branch — selected at JIT time — so there is no runtime branch overhead"
  - "Per-position tile_partner offsets: `partner_pos2 = (col_start + tile_partner) * 2` reuses the existing tile_partner XOR machinery; only the offset multiplication doubles. The XOR/stride math at the logical-position level is unchanged from Plan 07-01"
  - "tl.where on _re and _im independently: `is_lower = (tile_offsets & stride) == 0` is a logical-position mask of shape (TILE_N,); applied to (new_lower_re, new_upper_re) and (new_lower_im, new_upper_im) independently because the re and im of each logical position are governed by the same is_lower predicate"
  - "Unitary test uses direct `_ops.butterfly_multiply` call (not Butterfly.forward): per D-46 the legacy `Butterfly.forward` routes through `torch.ops.torch_structured.butterfly_multiply` (the C++ op via `csrc/butterfly.cpp`), which is independent of Triton dispatch. To exercise the Triton kernel via the unitary gate, the test must call `torch_structured._ops.butterfly_multiply(b.twiddle, identity_batch, True, n)` directly with the Haar-unitary twiddle from `init='ortho'`. This is the load-bearing acceptance gate — if it passes, the 4-FMA is correct"
  - "Complex64 gradcheck skipped on triton backend (D-47): the kernel is fp32/complex64 at the register-arithmetic level; gradcheck demands fp64/complex128 precision. The torch-backend gradcheck IS the acceptance gate per D-47 because the Triton backward delegates to `_torch_ref.butterfly_multiply_torch` via `torch.autograd.grad(...)` — testing the autograd plumbing on torch backend is sufficient. No manual `.conj()` correction needed (Phase 5 diag_mult had to add `.conj()` because it used a hand-rolled Wirtinger formula; Phase 7 lets `torch.autograd.grad` handle Wirtinger natively inside the oracle's natural execution)"

patterns-established:
  - "Plan 07-02 establishes the 'lift the gate, fill in the branch' pattern for partial-fp32 kernels with complex64 pre-wiring: Plan N ships the kernel signature + IS_COMPLEX constexpr + view_as_real wrapper machinery + a tl.static_assert / fp32-only wrapper gate; Plan N+1 removes the two gates and implements the IS_COMPLEX=True branch. Zero kernel-signature refactor between plans (D-41a load-bearing). Plans 07-01 / 07-02 are the first instance of this pattern in the codebase"
  - "Butterfly 4-FMA structure: each butterfly pair has 2x2 complex twiddle = 8 reals; per-stage requires 4 pairwise complex multiplies (t00*cur, t01*partner, t10*partner, t11*cur) = 16 FMAs and 6 adds. Cleanest implementation keeps _re / _im as separate register vectors throughout the stage; mirrors diag_mult's per-pair 4-FMA but with 4 pairs per butterfly stage instead of 1"
  - "Direct-call unitary acceptance gate for complex-correctness detection: when the nn.Module wrapper routes through a different backend (D-46), the load-bearing PITFALLS §1 test calls the dispatch-bound `_ops` function directly with the module's parameters. This pattern generalizes to any future port where the new kernel's wrapping nn.Module is not yet refactored"

requirements-completed: [TRI-03]

duration: ~40min
completed: 2026-05-27
---

# Phase 7 Plan 2: butterfly_multiply Triton forward (complex64 + perf baseline) Summary

**Complex64 path lit up by removing the two Plan-07-01 gates (kernel-entry `tl.static_assert(not IS_COMPLEX, ...)` and wrapper `assert input.dtype == torch.float32`) and implementing the IS_COMPLEX=True branch with the verbatim 4-FMA template adapted for butterfly's four pairwise complex multiplies per stage; kernel signature UNCHANGED (D-41a load-bearing); four new tests including the load-bearing PITFALLS §1 U U^H=I unitary detector; 8-row perf baseline JSON produced for Phase 9 TEST-04 parity gate.**

## Performance

- **Duration:** ~40 min (longer than Plan 07-01's 30 min due to the 4-FMA-per-pair butterfly extension + perf baseline runtime)
- **Started:** ~2026-05-27 (commit efcd111)
- **Completed:** 2026-05-27 (commit 8aba42a)
- **Tasks:** 3 (Task 1 auto-tdd, Task 2 auto-tdd, Task 3 auto)
- **Files created:** 3 (`tests/_baseline_butterfly.py`, `07-BASELINE.json`, this `07-02-SUMMARY.md`)
- **Files modified:** 2 (`_triton/butterfly/op.py`, `tests/test_butterfly_triton.py`)
- **No files deleted**
- **No STATE.md / ROADMAP.md modifications (parallel executor — orchestrator owns those writes)**

## Accomplishments

- **D-41 complex64 path lit up:** the kernel-entry `tl.static_assert(not IS_COMPLEX, ...)` line is gone (verified by `grep -c` returning 0); the wrapper's `assert input.dtype == torch.float32` is replaced by `assert input.dtype in (torch.float32, torch.complex64), ...`. The `view_as_real` / `view_as_complex` machinery from Plan 07-01 is now ACTIVE for complex64 inputs; fp16/fp64/int are rejected with `AssertionError` containing the canonical message "supports fp32 and complex64 only".
- **D-44 inheritance from Phase 4:** the wrapper boundary applies `view_as_real(input)` + `view_as_real(twiddle)` before the kernel launch (with the Plan 07-01 Pitfall 3 contiguity asserts still firing); the kernel sees the trailing-2 real layout; the result is reassembled via `view_as_complex(out.contiguous())` on return. No native `tl.complex*` usage (Triton has no complex dtype).
- **Kernel signature UNCHANGED between Plan 07-01 and Plan 07-02 (D-41a load-bearing):** `git diff 4f00b85..HEAD -- torch_structured/_triton/butterfly/op.py` shows the `_butterfly_kernel` parameter list (`twiddle_ptr, input_ptr, output_ptr, n, nstacks, block_idx, nblocks, STAGE_START, STAGE_COUNT, INCREASING_STRIDE, LOG_N, IS_COMPLEX, TILE_N`) is identical; only the kernel body and the wrapper's dtype assert changed.
- **4-FMA verifiable from source:** the kernel contains four `..._re * ..._re - ..._im * ..._im` patterns (one per pairwise complex multiply: `t00*cur`, `t01*partner`, `t10*partner`, `t11*cur`) and four matching `..._re * ..._im + ..._im * ..._re` patterns. The sign pattern is verbatim per `04-COMPLEX-LAYOUT.md:65-66`.
- **SC#1 complex64 cross-backend forward correctness:** all 7 smoke test parametrizations on `log_n in {2, 4, 8, 10}` x `backend in {torch, triton}` pass within `rtol=1e-4`. The slow-tier comprehensive Cartesian (720 cases per backend = 1440 total) also passes within `RTOL=ATOL=1e-3` (fp32 noise floor at log_n=11).
- **SC#2 unitary acceptance (the load-bearing PITFALLS §1 detector):** `test_butterfly_unitary` passes on BOTH backends. At log_n=4 (n=16) on the Triton backend, `max |U U^H - I| = 2.69e-07` — well below the atol=1e-4 threshold. Identical on torch backend. **This is the gate that fails loudly on any 4-FMA sign error; passing it confirms the IS_COMPLEX=True branch is sign-correct.**
- **SC#4 perf baseline recorded:** `.planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json` exists with 8 rows covering `log_n in {8, 9, 10, 11}` x `dtype in {fp32, complex64}` per the locked D-43b schema. NVIDIA RTX 2000 Ada Generation Laptop GPU; Triton 1.15x-2.25x speedup over the pure-PyTorch oracle.
- **Wirtinger gradcheck D-47 acceptance:** `test_butterfly_gradcheck_complex64` passes on torch backend with complex128 inputs at log_n=3; triton intentionally skipped per fp32/complex64 kernel-level precision (the backward delegates to `_torch_ref` exactly per D-47, so the torch-backend gradcheck IS the acceptance gate for both backends).
- **No regressions:** Plan 07-01's 29 fp32 smoke tests + 720 fp32 comprehensive tests still PASS. Phase 5 + Phase 6 + legacy butterfly nn.Module regression: 72 PASS + 2 SKIP (identical to baseline). Legacy `tests/test_butterfly.py`: 6 PASS + 5 pre-existing CUDA-version-mismatch FAILS (unchanged from Plan 07-01 baseline).

## Task Commits

Each task committed atomically on `worktree-agent-a541f02c10c324559`:

1. **Task 1: light up complex64 IS_COMPLEX branch in op.py** — `efcd111` (feat)
2. **Task 2: complex64 + Wirtinger gradcheck + unitary tests** — `c254252` (test)
3. **Task 3: baseline harness + 07-BASELINE.json** — `8aba42a` (perf)

## Files Created/Modified

### Created

- `tests/_baseline_butterfly.py` (169 lines): standalone perf measurement harness; NOT a pytest test (leading underscore filename prevents auto-collection). Uses `torch.cuda.Event(enable_timing=True)` per-iteration timing with 10 warmup + 100 measure iterations; per-iteration sync; sorted p50/p95. Measures `log_n in {8, 9, 10, 11}` x `dtype in {fp32, complex64}` at `batch_size=64, nstacks=1, nblocks=1, increasing_stride=True, output_size=n`. Skips cleanly on CPU-only hosts. Documents `PYTHONPATH=. python tests/_baseline_butterfly.py` invocation requirement when the worktree's `torch_structured` is shadowed by a pip-editable install of the main repo.

- `.planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json` (100 lines): 8-row JSON baseline matching the locked D-43b schema. Each row: `{kernel: "butterfly_multiply", dtype, log_n, nstacks, nblocks, wall_ms_p50, wall_ms_p95, reference_torch_ref_p50, measured_at, gpu}`. NVIDIA RTX 2000 Ada Generation Laptop GPU; Triton kernel shows 1.15x-2.25x speedup vs pure-PyTorch oracle. Phase 9 TEST-04 parity gate reads this file verbatim.

### Modified

- `torch_structured/_triton/butterfly/op.py` (483 → 567 lines; +84 net via 153 insertions / 69 deletions):
  - **Removed** the kernel-entry line `tl.static_assert(not IS_COMPLEX, "complex64 lands in 07-02 (D-41a pre-wiring)")` (Plan 07-01's D-41a load-bearing gate).
  - **Removed** the wrapper's `assert input.dtype == torch.float32, ...` preamble assert.
  - **Added** the broader gate `assert input.dtype in (torch.float32, torch.complex64), "butterfly_multiply supports fp32 and complex64 only; got {input.dtype}"`.
  - **Added** an `if IS_COMPLEX:` branch in the kernel-level seed-load + per-stage body. The seed-load now does `pos2 = pos * 2; x_re = tl.load(input_ptr + row_base + pos2); x_im = tl.load(input_ptr + row_base + pos2 + 1); tl.store(...)` (view_as_real layout). The per-stage body loads 8 twiddle reals at `pf8 = pair_flat * 8` offsets 0..7 (t00_re, t00_im, t01_re, t01_im, t10_re, t10_im, t11_re, t11_im), loads cur/partner re/im pairs, applies the canonical 4-FMA `(a + bi)(c + di) = (ac - bd) + (ad + bc)i` to each of the four pairwise complex multiplies (t00*cur, t01*partner, t10*partner, t11*cur), computes `new_lower = t00*cur + t01*partner` and `new_upper = t10*partner + t11*cur` complex (re and im separately), and applies `tl.where(is_lower, new_lower_*, new_upper_*)` to _re and _im independently. The fp32 `else:` branch is unchanged from Plan 07-01.
  - **Updated** stride math: when IS_COMPLEX is True, `row_base = bn_id * (2 * n)` and twiddle strides all double (`twiddle_stack_stride = nblocks * LOG_N * 2 * n * 2`, etc.).
  - **Updated** module docstring + wrapper docstring to reflect the lit-up complex64 path.
  - **Kernel signature UNCHANGED** — D-41a load-bearing.

- `tests/test_butterfly_triton.py` (234 → 413 lines; +179 net via 182 insertions / 3 deletions):
  - **Added** `from torch_structured.butterfly import Butterfly` (legacy nn.Module surface for the unitary test).
  - **Added** 4 new test functions appended after `test_butterfly_comprehensive`:
    - `test_butterfly_eager_complex64(backend, log_n)` — smoke tier (D-43a), 4 log_n params x 2 backends = 8 cases.
    - `test_butterfly_eager_complex64_grid(backend, log_n, nstacks, nblocks, increasing_stride, output_size_kind)` — comprehensive Cartesian @pytest.mark.slow, 720 cases per backend = 1440 cases (matches the locked SC#1 "full parameter grid").
    - `test_butterfly_gradcheck_complex64(backend)` — D-47 Wirtinger acceptance, complex128 inputs at log_n=3 on torch backend; triton intentionally skipped.
    - `test_butterfly_unitary(backend)` — the load-bearing PITFALLS §1 U U^H=I detector, log_n=4 (n=16) on both backends, atol=1e-4. Uses direct `_ops.butterfly_multiply` call (per D-46 the legacy `Butterfly.forward` routes through C++ — direct call exercises Triton).
  - **Updated** module docstring to mention the Plan 07-02 additions.

## Acceptance Gate Results

| Gate | Result |
|------|--------|
| SC#1 complex64 smoke (`test_butterfly_eager_complex64` × backend × log_n in {2,4,8,10} = 8 cases) | 8 PASS |
| SC#1 complex64 comprehensive (`test_butterfly_eager_complex64_grid` @pytest.mark.slow × backend = 1440 cases) | 1440 PASS |
| SC#2 unitary U U^H = I (`test_butterfly_unitary` × backend = 2 cases at log_n=4) | 2 PASS (triton `max_err = 2.69e-07`, torch `max_err = 2.66e-07`) |
| SC#4 perf baseline JSON exists with 8 rows + D-43b schema | PASS |
| D-47 Wirtinger gradcheck (`test_butterfly_gradcheck_complex64` × torch backend) | 1 PASS + 1 SKIP (intentional triton skip per D-41/D-47) |
| `grep -c 'tl.static_assert(not IS_COMPLEX'` returns 0 | PASS |
| `grep -c 'assert input.dtype == torch.float32'` returns 0 | PASS |
| `grep -c 'IS_COMPLEX: tl.constexpr'` returns 1 (kernel signature) | PASS |
| `grep -c 'view_as_real'` returns 15 (multiple wrapper + kernel docstring uses) | PASS |
| `grep -c 'view_as_complex'` returns 1 (wrapper output side) | PASS |
| 4-FMA sign pattern verifiable from source | PASS (4 `..._re * ..._re - ..._im * ..._im` matches + 4 `..._re * ..._im + ..._im * ..._re` matches) |
| Plan 07-01 fp32 smoke tier (`-m 'not slow'` × backend = 40 cases including 11 new) | 40 PASS + 2 SKIP (intentional triton gradcheck skips) |
| Plan 07-01 fp32 comprehensive tier (720 cases × backend) | 1440 PASS opt-in via `pytest -m slow` (combined with Plan 07-02 complex64 grid) |
| Phase 5 + Phase 6 regression (`test_diag_mult.py` + `test_dispatch.py` + `test_hadamard_triton.py` + `test_hadamard.py` + `test_imports.py`) | 72 PASS + 2 SKIP (identical to Plan 07-01 baseline) |
| Legacy `tests/test_butterfly.py` (D-46) | 6 PASS + 5 pre-existing CUDA-version-mismatch FAILS (identical to Plan 07-01 baseline) |
| Unsupported dtype reject (fp16) | PASS — `AssertionError: butterfly_multiply supports fp32 and complex64 only; got torch.float16` |
| Mixed dtype reject (fp32 twiddle, complex64 input) | PASS — `AssertionError: twiddle.dtype (torch.float32) must equal input.dtype (torch.complex64)` |

## Perf Baseline (NVIDIA RTX 2000 Ada Generation Laptop GPU)

| log_n | dtype     | wall_ms_p50 | wall_ms_p95 | ref_p50 | speedup |
|-------|-----------|-------------|-------------|---------|---------|
| 8     | fp32      | 0.1976      | 0.4444      | 0.3738  | 1.89x   |
| 8     | complex64 | 0.1996      | 0.2580      | 0.3492  | 1.75x   |
| 9     | fp32      | 0.1855      | 0.2427      | 0.4168  | 2.25x   |
| 9     | complex64 | 0.2191      | 0.5612      | 0.4035  | 1.84x   |
| 10    | fp32      | 0.2026      | 0.2365      | 0.4342  | 2.14x   |
| 10    | complex64 | 0.2775      | 0.4239      | 0.4403  | 1.59x   |
| 11    | fp32      | 0.2630      | 0.3328      | 0.4794  | 1.82x   |
| 11    | complex64 | 0.4188      | 0.5181      | 0.4803  | 1.15x   |

Triton speedup over the pure-PyTorch oracle ranges 1.15x (complex64, log_n=11) to 2.25x (fp32, log_n=9). The complex64 path at log_n=11 is roughly 2x more expensive than fp32 at log_n=11 (0.42 ms vs 0.26 ms) — consistent with 2x the load/store traffic per element (re + im) and 4x the FMAs per pairwise multiply. Phase 9 TEST-04 reads these rows verbatim.

## Decisions Made

- **Option (b) IS_COMPLEX kernel pattern** (planner-recommended): two parallel register vectors `_re` and `_im` rather than a single interleaved vector. The kernel body becomes textually parallel to the fp32 path (4 pairwise multiplies → 4-FMA each → `tl.where` on _re and _im independently) but with 2x the load/store volume. Mirrors `_triton/diag_mult/op.py:64-87` verbatim.

- **Stride doubling for view_as_real layout** (D-44 inheritance): all twiddle strides double (`twiddle_stack_stride = nblocks * LOG_N * 2 * n * 2`); `pf8 = pair_flat * 8` replaces `pf4 = pair_flat * 4`; `row_base = bn_id * (2 * n)`. The constexpr branch ensures zero runtime overhead.

- **Direct-call unitary acceptance gate** (D-46 inheritance): the legacy `Butterfly.forward` routes through `torch.ops.torch_structured.butterfly_multiply` (the C++ op), independent of Triton dispatch. To exercise the Triton kernel via PITFALLS §1, `test_butterfly_unitary` calls `torch_structured._ops.butterfly_multiply(b.twiddle, identity_batch, True, n)` directly with the Haar-unitary twiddle from `init='ortho'`.

- **Complex64 smoke tolerance `rtol=1e-4, atol=1e-4`** (ROADMAP SC#1 literal): the smoke tier at log_n ≤ 10 stays within the SC#1 tolerance literally. The comprehensive grid uses the module-level `RTOL=ATOL=1e-3` to accommodate fp32 noise floor at log_n=11 (consistent with Plan 07-01's tolerance discussion in the file docstring).

- **No `.conj()` correction needed in butterfly backward** (D-47 inheritance): Phase 5 `diag_mult` used a hand-rolled Wirtinger formula and had to add `.conj()` manually; Phase 7 `butterfly_multiply` delegates the entire gradient to `torch.autograd.grad(_torch_ref(...), [twiddle, input], grad_out)` — `torch.autograd.grad` handles Wirtinger gradients natively inside the oracle's natural execution. Verified by `test_butterfly_gradcheck_complex64` passing without any `.conj()` in the kernel or wrapper.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocker] PYTHONPATH=. invocation needed for `tests/_baseline_butterfly.py`**

- **Found during:** Task 3 first run.
- **Issue:** `python tests/_baseline_butterfly.py` (no `PYTHONPATH`) imported the main-repo's `torch_structured` (because `torch_structured` is pip-editable-installed against `/home/claroche/torch-structured`, not the worktree). The main repo still has the Plan-07-01 `op.py` with the fp32-only assert, so the harness hit `AssertionError: Plan 07-01: fp32-only (complex64 lands in 07-02)`. The earlier pytest runs worked without `PYTHONPATH` because pytest auto-injects the cwd into `sys.path`; the standalone script does not get that benefit.
- **Fix:** Documented the requirement in the script docstring (`Invocation note: ... PYTHONPATH=. python tests/_baseline_butterfly.py`); produced the JSON with `PYTHONPATH=. python tests/_baseline_butterfly.py`.
- **Files modified:** `tests/_baseline_butterfly.py` (docstring).
- **Verification:** With `PYTHONPATH=.`, the harness runs cleanly and produces the 8-row JSON; `python -c "...load and validate..."` confirms schema compliance.
- **Committed in:** `8aba42a` (Task 3 commit).

### Auto-added critical functionality

**2. [Rule 2 - Hygiene] Schema validation in measurement script**

- **Found during:** Task 3 design.
- **Why added:** D-43b schema is consumed verbatim by Phase 9 TEST-04. Any schema drift (missing key, wrong type) would silently break Phase 9. The harness intentionally produces JSON with all keys present and uses `round(_, 6)` for diff-friendliness.
- **Files modified:** `tests/_baseline_butterfly.py` (the schema is enforced by construction in the row dict).

---

**Total deviations:** 1 auto-fixed (1 Rule 3 blocker) + 1 auto-added (1 Rule 2 hygiene).

**Impact on plan:** Deviation #1 is workflow-only (the JSON is produced; the script docs the requirement). Deviation #2 is hygiene (the JSON schema is enforced by construction). No deviations affect the kernel or test correctness; all SC#1/SC#2/SC#4 acceptance gates pass.

## Issues Encountered

- **`torch_structured` pip-editable shadows the worktree.** Like Plan 07-01 (which copied .so files), the worktree shares the pip-editable install of the main repo. For Python imports that go through `import torch_structured`, pytest auto-injects cwd but standalone scripts do not — hence the `PYTHONPATH=.` requirement for `tests/_baseline_butterfly.py`. Documented inline.

- **No new test failures introduced.** Verified — Phase 5 + Phase 6 + legacy regression all pass identically to Plan 07-01 baseline. Plan 07-01's 29 smoke tests + 720 comprehensive tests + the 4 new Plan 07-02 tests + 720 new complex64 comprehensive tests all pass.

- **The kernel-signature contract held (D-41a).** Zero kernel-signature refactor between Plan 07-01 and Plan 07-02 — the `_butterfly_kernel` parameter list is identical pre- and post-modification; only the kernel body and one wrapper assert changed. Verified by `git diff 4f00b85..efcd111 -- torch_structured/_triton/butterfly/op.py` showing changes are isolated to the body (lines 144-238 → 144-298 in old/new file).

## User Setup Required

None — no external service configuration. The phase is internal kernel work.

## Next Phase Readiness

### Plan 07-03 (if any)

Plan 07-02 satisfies all Phase 7 forward goals (SC#1 complex64 + SC#2 unitary + SC#4 perf baseline + the Plan-07-02-specific invariants). The phase is complete for the forward direction. There is no Plan 07-03 in the current phase roadmap.

### Phase 8 (butterfly_multiply backward Triton port)

- **The forward kernel is feature-complete for fp32 + complex64.** The backward via `_torch_ref` is a temporary stand-in (`_backward` in `op.py:432-459`); Phase 8 replaces the `_backward` callback with the heavy `tl.atomic_add`-into-`d_twiddle` reduction kernel.

- **The two-input register_autograd pattern is in place** (D-47): `torch.autograd.grad(_butterfly_multiply_torch(twiddle_d, input_d, ...), [twiddle_d, input_d], grad_out)`. Phase 8 replaces this oracle call with the native Triton backward kernel; the autograd plumbing (save_for_backward, requires_grad_(True) on detached clones, returning a 4-tuple matching forward inputs) is reusable.

- **The perf baseline at `07-BASELINE.json` is the parity reference for Phase 9 TEST-04 (Triton backward perf gate).** Phase 9 reads the file verbatim; Phase 8's backward kernel results should be measured against this baseline.

### Phase 9 (integration hardening)

- **TEST-04 parity gate is ready.** The 8-row baseline JSON exists; Phase 9 reads it verbatim and asserts Triton ≥ 60% of CUDA throughput at log_n in {8, 9, 10, 11}.

- **No blockers** for Phase 8 or Phase 9.

## Threat Flags

No new threat surface introduced beyond what Plan 07-01 documented and Plan 07-02 mitigated:

- **T-07-08 (Tampering: non-canonical strides through view_as_real)** — mitigated by the Plan 07-01 contiguity asserts which now FIRE on the complex64 path (load-bearing per Pitfall 3).
- **T-07-09 (Tampering: unsupported dtypes)** — mitigated by the new `assert input.dtype in (torch.float32, torch.complex64), ...` gate; fp16, fp64, int, bf16 all reject with the canonical message.
- **T-07-10 (Tampering: 4-FMA sign error)** — mitigated by `test_butterfly_unitary` (the load-bearing PITFALLS §1 detector); it passes on both backends with `max |U U^H - I| ≈ 2.69e-07` at log_n=4, confirming the 4-FMA is sign-correct.
- **T-07-11 (Information Disclosure: GPU model in baseline)** — accepted; the baseline JSON is a development artifact committed to `.planning/` and not user-facing.

All threats remain LOW severity (internal kernel code with no external attacker surface). No new threat surface from Plan 07-02 changes.

---

## Self-Check: PASSED

Files verified to exist (absolute paths):

- FOUND: /home/claroche/torch-structured/.claude/worktrees/agent-a541f02c10c324559/torch_structured/_triton/butterfly/op.py (modified — 567 lines; +84 net vs Plan 07-01)
- FOUND: /home/claroche/torch-structured/.claude/worktrees/agent-a541f02c10c324559/tests/test_butterfly_triton.py (modified — 413 lines; +179 net vs Plan 07-01)
- FOUND: /home/claroche/torch-structured/.claude/worktrees/agent-a541f02c10c324559/tests/_baseline_butterfly.py (new — 169 lines)
- FOUND: /home/claroche/torch-structured/.claude/worktrees/agent-a541f02c10c324559/.planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json (new — 100 lines, 8 rows)
- FOUND: /home/claroche/torch-structured/.claude/worktrees/agent-a541f02c10c324559/.planning/phases/07-butterfly-multiply-forward-triton/07-02-SUMMARY.md (this file)

Commits verified to exist on `worktree-agent-a541f02c10c324559`:

- FOUND: efcd111 (Task 1 — feat IS_COMPLEX branch lit up in op.py)
- FOUND: c254252 (Task 2 — test complex64 + Wirtinger + unitary tests in test_butterfly_triton.py)
- FOUND: 8aba42a (Task 3 — perf _baseline_butterfly.py + 07-BASELINE.json)

Test acceptance gates verified on this CUDA host (NVIDIA RTX 2000 Ada Generation Laptop GPU, CUDA 13.0):

- FOUND: tests/test_butterfly_triton.py smoke tier — 40 PASS + 2 SKIP (intentional triton gradcheck skips for fp64 and complex128)
- FOUND: tests/test_butterfly_triton.py comprehensive slow tier — 1440 PASS (`pytest -m slow`)
- FOUND: Phase 5 + Phase 6 regression — 72 PASS + 2 SKIP (no change from baseline)
- FOUND: Legacy tests/test_butterfly.py — 6 PASS + 5 pre-existing CUDA-version-mismatch FAILS (no change from baseline; failures predate Phase 7)
- FOUND: D-41 gate removal verified by grep (static_assert IS_COMPLEX count=0; fp32-only assert count=0; complex64 reference count≥1)
- FOUND: D-41a kernel-signature contract verified by `git diff 4f00b85..HEAD -- _triton/butterfly/op.py` showing zero parameter-list changes
- FOUND: SC#2 unitary U U^H = I PASS on both backends with `max_err ≈ 2.69e-07`
- FOUND: SC#4 perf baseline JSON exists with 8 rows + D-43b schema; Triton 1.15x-2.25x speedup over torch_ref oracle

---
*Phase: 07-butterfly-multiply-forward-triton*
*Plan: 02*
*Completed: 2026-05-27*
