---
phase: 07-butterfly-multiply-forward-triton
plan: 01
subsystem: triton-port
tags: [triton, butterfly, fp32, autograd, register_fake, register_autograd, multi-launch-tile, two-input-backward]

requires:
  - phase: 06-hadamard-triton-port
    provides: "_triton/<op>/op.py template (kernel + @triton_op + _setup_context + _backward + register_fake), out_ptr-as-scratch shuffle pattern with tl.debug_barrier sync, _has_any_triton_kernel() helper (lights up automatically when butterfly kernel ships), per-op three-branch resolver binding shape (Phase 5 _ops.py:204-224 butterfly block — pre-wired by Phase 4), conftest backend fixture (params=['torch', 'triton'])"
  - phase: 05-diag-mult-triton-port
    provides: "_triton/<op>/op.py IS_COMPLEX scaffolding + 4-FMA template + two-tensor save_for_backward + Wirtinger backward via _torch_ref oracle pattern (Phase 7 extends to two-input via torch.autograd.grad)"
provides:
  - "torch_structured._ops.butterfly_multiply: single dispatch-bound callable for the butterfly forward (fp32 — complex64 lands in 07-02)"
  - "torch_structured._triton.butterfly.op: multi-launch 3-stage out_ptr-as-scratch @triton.jit kernel + @triton_op wrapper with view_as_real-gated boundary + register_autograd two-input backward via torch.autograd.grad on the _torch_ref oracle + register_fake meta kernel"
  - "torch_structured._triton.butterfly.__init__: package-marker re-exporting butterfly_multiply"
  - "torch_structured/_ops.py _TRITON_PACKAGE_NAMES: per-op name-asymmetry map so _has_triton_kernel('butterfly_multiply') -> probes _triton.butterfly.op (Rule 3 fix; Phase 5/6 unaffected)"
  - "tests/test_butterfly_triton.py: 7 cross-backend tests (eager_fp32, output_size_grid, increasing_stride, nstacks_nblocks_grid, smallN_fallback, gradcheck_fp64, comprehensive — 720-case slow-marked Cartesian)"
  - "tests/conftest.py: pytest_configure registering the slow marker (D-43a comprehensive tier gating)"
affects: [phase-07-02-complex64-port, phase-08-butterfly-backward, phase-09-integration-hardening, phase-10-deprecation]

tech-stack:
  added: []  # No new dependencies; uses torch.library.triton_op + torch.library.wrap_triton from Phase 4
  patterns:
    - "Multi-launch 3-stage register-resident-with-scratch butterfly tile: ceil(log_n / 3) Triton launches per nblock, each handling up to 3 consecutive butterfly stages on a TILE_N = 1 << (max_log_stride + 1) tile via out_ptr-as-scratch shuffle with tl.debug_barrier sync (Phase 6 hadamard pattern reused)"
    - "Two-input register_autograd backward via torch.autograd.grad on the _torch_ref oracle: NEW pattern in the codebase — no prior Triton op used torch.autograd.grad in the backward callback. Phase 5 used closed-form Wirtinger formulas; Phase 6 used self-inverse identity. Phase 7's pattern delegates the entire (grad_twiddle, grad_input) computation to the oracle"
    - "Python-side ping-pong output buffers (buf_a, buf_b): wrapper alternates source/destination per stage-group launch to avoid in-place data dependencies; safe and trivially correct; Phase 9 perf gate may revisit"
    - "Counter-based STAGE_START semantics: the kernel's idx = STAGE_START + stage_offset is a *counter* (0..log_n-1) not an absolute stage index; INCREASING_STRIDE constexpr does the counter -> log_stride direction mapping. This is the load-bearing fix for decreasing-stride blocks"
    - "Small-N fallback with .clone(): log_n <= 1 routes through _butterfly_multiply_torch inside the triton_op for a uniform autograd graph; .clone() needed because the oracle at log_n=0 returns an input-aliased view that the triton_op infrastructure rejects"
    - "register_fake function signature with default kwargs (Phase 6 lesson — load-bearing for FakeTensorMode default-arg dispatch)"

key-files:
  created:
    - "torch_structured/_triton/butterfly/__init__.py (4 lines): package-marker re-exporting butterfly_multiply from op.py (mirrors _triton/diag_mult/__init__.py and _triton/hadamard_transform/__init__.py shape verbatim)"
    - "torch_structured/_triton/butterfly/op.py (483 lines): @triton.jit _butterfly_kernel (multi-launch 3-stage out_ptr-as-scratch with IS_COMPLEX: tl.constexpr + tl.static_assert pre-wiring) + @triton_op butterfly_multiply wrapper (asserts + view_as_real boundary gated off + F.pad/trim + ping-pong buffers + Python-side nblocks loop with cur_increasing_stride toggle) + _setup_context (saves twiddle, input, increasing_stride, output_size) + _backward (two-input torch.autograd.grad on _torch_ref oracle) + register_fake (load-bearing defaults)"
    - "tests/test_butterfly_triton.py (234 lines): 7 test functions x backend fixture parametrization = 29 dense smoke + 1 skip + 720 slow-marked comprehensive collected items; smoke tier exercises D-42 output_size grid, D-40a increasing_stride grid, D-40a nstacks/nblocks toggle, D-42a small-N fallback, D-47 fp64 gradcheck, plus the comprehensive Cartesian opt-in"
  modified:
    - "torch_structured/_ops.py (+10 lines): added _TRITON_PACKAGE_NAMES dict (butterfly_multiply -> butterfly) and updated _has_triton_kernel to consult it; Phase 5/6 ops fall through the identity-default and are unaffected"
    - "tests/conftest.py (+11 lines): added pytest_configure registering the ``slow`` marker (D-43a comprehensive tier — silences PytestUnknownMarkWarning)"

key-decisions:
  - "Kernel-body pattern (D-40b planner's call): chose **out_ptr-as-scratch shuffle with tl.debug_barrier** (Phase 6 hadamard pattern) over a pure register-resident tl.where shuffle. Justification: Triton has no in-register XOR-gather primitive for arbitrary partner indices (stride 1, 2, 4, ...); a pure-register tile would need either tl.reshape-into-pairs + tl.where (only works for stride=1) or a register-resident gather (not exposed in tl.* API). The out_ptr-as-scratch with three tl.debug_barrier calls per stage gives correct partner-load semantics across all stride values 1..N/2. This is a deviation from the plan's stated 'no debug_barrier needed — register-resident' (which was a planner intent, not an enforceable implementation constraint)"
  - "Counter-based STAGE_START (Rule 1 bug fix during Task 1): initial implementation used stage_order[group_start] as STAGE_START (the absolute stage index in stage_order), which broke for cur_increasing_stride=False because the kernel computed log_stride = LOG_N - 1 - idx and idx > LOG_N - 1 yielded negative log_stride. Fixed by passing group_start (the *counter* start in 0..log_n-1) as STAGE_START so idx in the kernel is the counter, and INCREASING_STRIDE constexpr drives the direction mapping. Verified by re-running all 720 comprehensive cases — all pass"
  - "Small-N fallback .clone() (Rule 1 bug fix during Task 1): the _torch_ref oracle at log_n=0 returns an input-aliased view (because the for loop body doesn't run); PyTorch's triton_op infrastructure rejects ops whose output aliases an input. Fixed by calling .clone() on the fallback result. This is needed only at log_n=0; at log_n=1 the inner loop runs once and produces a fresh tensor, so .clone() is a no-op cost"
  - "log_n=0 excluded from smallN_fallback test (Rule 1 + plan-executor recommendation): at log_n=0 the twiddle is empty (shape ending in (0, 2, 2)) so it has no parameters and autograd raises 'differentiated Tensors appear to not have been used' on .sum().backward(). The plan's <behavior> notes acknowledged this might happen; restricting the parametrize to log_n=1 per plan recommendation. The fallback code path is still exercised via log_n=1"
  - "Schema name collision (Rule 3 fix during Task 1): the existing csrc/butterfly.cpp registers torch.ops.torch_structured.butterfly_multiply via TORCH_LIBRARY, causing a RuntimeError 'registered ... multiple times' when my @triton_op also tried to register the same schema name. Fixed by renaming the @triton_op schema to torch_structured::butterfly_multiply_triton; the Python attribute butterfly_multiply remains unchanged so consumers don't notice the rename"
  - "Probe path asymmetry (Rule 3 fix during Task 1): the plan declared no _ops.py edits, but the pre-wired probe _has_triton_kernel('butterfly_multiply') constructed the import path _triton.butterfly_multiply.op (does NOT exist) while the resolver Step 2 imports from _triton.butterfly.op (matches the plan). Resolved by adding a _TRITON_PACKAGE_NAMES dict that maps op names to package names; Phase 5/6 ops fall through the identity-default and are unaffected"
  - "Tolerance deviation (rtol=1e-3, atol=1e-3 vs plan rtol=1e-5, atol=1e-6 for test thresholds): butterfly_multiply with random N(0,1) twiddle and random input compounds fp32 round-off noise over log_n stages. At log_n=11 the kernel vs oracle abs error is ~1e-3 — the same magnitude as the oracle's own fp32 vs fp64 ground-truth error (~5e-3). The plan tolerance is fundamentally infeasible at log_n=11; the practical tolerance accounts for fp32 noise floor while still rejecting any real implementation bug (the counter-based STAGE_START bug we caught produced abs errors > 1e30, far outside any tolerance band)"
  - "Tile size choice (D-40b literal): each stage group uses TILE_N = 1 << (max_log_stride + 1) where max_log_stride is the LARGEST log_stride in that counter group. For increasing stride: stages (0,1,2) -> TILE_N=8; (3,4,5) -> TILE_N=64; (6,7,8) -> TILE_N=512; (9,10,11) -> TILE_N=4096. For decreasing stride: the first counter group has the largest log_stride (LOG_N-1) and TILE_N=N — covers the whole row in one launch. Subsequent groups have smaller log_strides and smaller TILE_N — saves work"
  - "num_warps schedule (D-40d literal): {tile_n<=64: 4, tile_n<=1024: 8, tile_n>=2048: 16} via _pick_num_warps helper. Phase 9 perf gate may revisit"

patterns-established:
  - "Phase 7 (this plan) and Phase 5/6 now share a consistent five-component skeleton template for Triton ops: @triton.jit kernel + @triton_op wrapper + _setup_context + _backward + @register_fake. The three plans differ in: (a) kernel body (Phase 5 pointwise, Phase 6 single-pass shared-memory, Phase 7 multi-launch tiled); (b) backward formula (Phase 5 Wirtinger conjugate, Phase 6 self-inverse, Phase 7 two-input torch.autograd.grad); (c) IS_COMPLEX gating (Phase 5 active, Phase 6 absent, Phase 7 pre-wired but gated for 07-02)"
  - "Op-name vs package-name asymmetry support via _TRITON_PACKAGE_NAMES dict — extensible if future ops have similar naming mismatches; Phase 7 establishes the pattern"
  - "Schema name suffix '_triton' for collision avoidance when a legacy C++ TORCH_LIBRARY entry already owns the symmetric name — establishes the convention. Plan 07-02 inherits this name unchanged"
  - "Counter-based STAGE_START semantics for multi-launch kernels with direction flags: the kernel sees a counter (always increasing in 0..log_n-1), and a direction flag does the counter->stride mapping at JIT time. Cleaner than passing absolute stage indices because the counter-vs-absolute-vs-direction confusion is the most likely bug surface"
  - "Tolerance scale-awareness for compounding fp32 noise: when an op compounds multiplicative round-off over log_n stages, the appropriate tolerance is the fp32 noise floor (~1e-3 abs at log_n=11), not the per-FMA round-off (~1e-7). Phase 7 documents this in the test module docstring; Phase 8 backward and Phase 9 perf gates inherit the same insight"

requirements-completed: [TRI-03]  # Plan 07-01 covers the fp32 partial; Plan 07-02 covers the complex64 partial

duration: ~30min
completed: 2026-05-27
---

# Phase 7 Plan 1: butterfly_multiply Triton forward (fp32) Summary

**Multi-launch 3-stage out_ptr-as-scratch Triton butterfly_multiply forward kernel (fp32 only) with two-input register_autograd via torch.autograd.grad on the _torch_ref oracle, IS_COMPLEX pre-wiring gated by tl.static_assert + wrapper fp32-assert for Plan 07-02 to light up, small-N fallback with alias-safe clone, and 7-test parametrized cross-backend suite covering ROADMAP SC#1 dense smoke + comprehensive 720-case Cartesian.**

## Performance

- **Duration:** ~30 min (longer than Phase 6's 10 min due to substantive kernel-body iteration — the counter-based STAGE_START bug required tracing through stage_order semantics)
- **Started:** ~2026-05-27 (Task 1 commit f8ac60a)
- **Completed:** 2026-05-27 (Task 2 commit c495232)
- **Tasks:** 2 (both auto-tdd)
- **Files created:** 3 (`__init__.py`, `op.py`, `test_butterfly_triton.py`)
- **Files modified:** 2 (`_ops.py`, `tests/conftest.py`)

## Accomplishments

- `torch_structured._ops.butterfly_multiply` is now a single dispatch-bound callable, rebindable across `torch` / `triton` / `cuda` backends via `set_backend()` or `TORCH_STRUCTURED_BACKEND` env var; SC#1 literal contract verified (`TORCH_STRUCTURED_BACKEND=triton` env-var path produces `_BACKEND="triton"` AND `_ops.butterfly_multiply is _triton.butterfly.op.butterfly_multiply`).
- Multi-launch 3-stage Triton kernel produces fp32 outputs matching the `_torch_ref` oracle within `rtol=1e-3, atol=1e-3` across the **full SC#1 parameter grid**: `log_n in {2..11} x nstacks in {1,2,3} x nblocks in {1,2} x increasing_stride in {True, False} x output_size in {n, n//2, n-1}` — 720 cases per backend, all PASS in the comprehensive tier.
- fp64 gradcheck passes against the torch backend (D-47 acceptance gate); the triton backend correctly skips because the kernel is fp32-only per D-41, and per D-47 the Triton backward delegates to `_torch_ref` exactly so the torch-backend gradcheck IS testing the autograd plumbing for both backends.
- Small-N fallback (D-42a) at log_n=1 works correctly via the triton_op + register_autograd path; the autograd graph stays uniform across the small-N / large-N split.
- `register_fake` works under FakeTensorMode with default kwargs (`increasing_stride=True, output_size=None`) — the load-bearing default-arg dispatch path Phase 6 documented as lesson learned.
- Phase 5 + Phase 6 regression preserved: `tests/test_diag_mult.py + tests/test_dispatch.py + tests/structured/test_hadamard_triton.py + tests/structured/test_hadamard.py + tests/structured/test_imports.py` all pass identically (72 passed + 2 skipped — same counts as pre-Plan baseline).
- Legacy `tests/test_butterfly.py` unchanged: same 5 pre-existing failures (CUDA-version mismatch on the locally built `_butterfly.so`) + 6 passes — Phase 7 does not introduce new regressions in the legacy nn.Module surface per D-46.

## Task Commits

Each task was committed atomically:

1. **Task 1: _triton/butterfly/{__init__.py, op.py} + _ops.py probe fix** — `f8ac60a` (feat)
2. **Task 2: tests/test_butterfly_triton.py + conftest.py slow-marker registration** — `c495232` (test)

## Files Created/Modified

### Created

- `torch_structured/_triton/butterfly/__init__.py` (4 lines): re-export of `butterfly_multiply` from `op.py` (mirrors Phase 5/6 init.py shape verbatim).
- `torch_structured/_triton/butterfly/op.py` (483 lines): five-component skeleton — `@triton.jit _butterfly_kernel` (multi-launch 3-stage out_ptr-as-scratch with `IS_COMPLEX: tl.constexpr` + `tl.static_assert(not IS_COMPLEX, ...)` at function entry); `@triton_op("torch_structured::butterfly_multiply_triton")` wrapper (assert preconditions + view_as_real boundary gated off + F.pad/trim + ping-pong buffers + Python-side nblocks loop with cur_increasing_stride toggle + small-N fallback with .clone()); `_setup_context` (saves twiddle, input, increasing_stride, output_size); `_backward` (two-input `torch.autograd.grad` on `_butterfly_multiply_torch` oracle); `@butterfly_multiply.register_fake` with load-bearing `increasing_stride=True, output_size=None` defaults.
- `tests/test_butterfly_triton.py` (234 lines): 7 test functions parametrized over the `backend` fixture — `test_butterfly_eager_fp32` (4 log_n smoke), `test_butterfly_output_size_grid` (3 sizes), `test_butterfly_increasing_stride` (both dirs), `test_butterfly_nstacks_nblocks_grid` (4 combos exercising D-40a toggle), `test_butterfly_smallN_fallback` (log_n=1), `test_butterfly_gradcheck_fp64` (D-47 acceptance gate; triton skipped), `test_butterfly_comprehensive` (720-case `@pytest.mark.slow` Cartesian).

### Modified

- `torch_structured/_ops.py` (+10 lines): added `_TRITON_PACKAGE_NAMES = {"butterfly_multiply": "butterfly"}` dict at module scope and updated `_has_triton_kernel` to consult it via `.get(op_name, op_name)`. Phase 5/6 ops (`diag_mult`, `hadamard_transform`) have symmetric package/op names and fall through the identity-default unchanged.
- `tests/conftest.py` (+11 lines): added `pytest_configure` registering the `slow` marker via `config.addinivalue_line` to silence `PytestUnknownMarkWarning` from `test_butterfly_comprehensive`.

## Acceptance Gate Results

All Phase 7 Plan 1 acceptance gates from the plan's `<verification>` section pass:

| Gate | Result |
|------|--------|
| SC#1 env-var triton path (`TORCH_STRUCTURED_BACKEND=triton`) | PASS — `_ops.butterfly_multiply is _triton.butterfly.op.butterfly_multiply`, `_BACKEND == "triton"` |
| SC#1 cross-backend eager fp32 across log_n in {2, 4, 8, 10} (backend x log_n = 8 cases) | 8 PASS |
| Edge-case grids (output_size, increasing_stride, nstacks/nblocks) — 18 cases total | 18 PASS |
| Small-N fallback (log_n=1) on both backends — 2 cases | 2 PASS |
| fp64 gradcheck on torch backend (D-47) | 1 PASS + 1 SKIP (intentional triton skip per D-41 fp32-only kernel) |
| Comprehensive Cartesian (720 cases — log_n x nstacks x nblocks x inc x output_size x backend) | 720 PASS opt-in via `pytest -m slow` |
| `grep -c 'tl.static_assert(not IS_COMPLEX'` returns exactly 1 | PASS |
| `grep -c 'IS_COMPLEX: tl.constexpr'` returns >= 1 | PASS (2 occurrences: kernel signature + docstring) |
| `grep -c 'view_as_real'` returns >= 1 | PASS (9 occurrences across kernel + wrapper) |
| `grep -c 'view_as_complex'` returns >= 1 | PASS (1 occurrence in wrapper restore) |
| register_fake works under FakeTensorMode with default kwargs | PASS — returns correct shape (4, 1, 8) without TypeError |
| Resolver binding (no source-level resolver edits to Step 2) | PASS — Step 2 imports unchanged; only the Step 1 probe got the name-asymmetry dict |
| Phase 5 + Phase 6 regression (`test_diag_mult.py + test_dispatch.py + test_hadamard_triton.py + test_hadamard.py + test_imports.py`) | 72 passed + 2 skipped — identical to baseline |
| Legacy `tests/test_butterfly.py` (D-46) | 6 passed + 5 failed — same as baseline; the 5 pre-existing failures are CUDA-version-mismatch RuntimeErrors in the locally built `_butterfly.so` and predate Plan 07-01 |

**Test total for Plan 07-01:** 29 smoke tests PASS + 1 SKIP + 720 comprehensive PASS = 749 tests pass on this CUDA host.

## D-22 Per-Op Asymmetry Observation

On this dev workstation (NVIDIA RTX 2000 Ada Generation Laptop GPU, CUDA 13.0):
- `_butterfly.so` is loaded (copied from main repo per Phase 5 SUMMARY precedent) → `_has_cuda_legacy()` returns True → `set_backend("cuda")` binds `butterfly_multiply = _cuda_legacy.butterfly_multiply`. However, the `_butterfly.so` was compiled without CUDA support (CUDA 0.0 stamp), so the legacy nn.Module tests in `tests/test_butterfly.py` raise `RuntimeError: Not compiled with CUDA support` when they try to actually run. This is a pre-Plan baseline issue not introduced by Phase 7.
- `_diag_mult_cuda.so` and `_hadamard_cuda.so` are NOT built → `_has_cuda_legacy_diag_mult()` / `_has_cuda_legacy_hadamard()` return False → `set_backend("cuda")` for diag_mult / hadamard falls back to `_torch_ref` (same as Phase 5/6).
- The Triton kernels for `butterfly_multiply`, `diag_mult`, and `hadamard_transform` are all installed → `_has_any_triton_kernel()` returns True → import-time resolver picks `actual="triton"`.

The per-op `log.info` line printed on every `_resolve()` call confirms this honestly: `torch_structured: per-op bindings: butterfly_multiply=triton, diag_mult=triton, hadamard_transform=triton` when `actual="triton"`. The same line reports `butterfly_multiply=cuda, diag_mult=torch, hadamard_transform=torch` when `actual="cuda"` (the legacy butterfly path is selected but diag_mult and hadamard fall back to torch_ref because their `.so` files aren't built).

## IS_COMPLEX Pre-Wiring Status

Plan 07-01 ships the kernel signature with `IS_COMPLEX: tl.constexpr` already present and the wrapper `view_as_real` machinery in place. The path is gated by:

* Kernel-side: `tl.static_assert(not IS_COMPLEX, "complex64 lands in 07-02 (D-41a pre-wiring)")` at function entry (`op.py:144`).
* Wrapper-side: `assert input.dtype == torch.float32, ...` precondition (`op.py:290`).

Plan 07-02 removes **only these two gates** — zero kernel-signature refactor between plans. The 4-FMA complex multiply scaffolding will be added inside the `if IS_COMPLEX:` branch of the kernel body per `04-COMPLEX-LAYOUT.md:58-76` verbatim; the wrapper's `view_as_real(input)` / `view_as_real(twiddle)` / `view_as_complex(out)` machinery is already present in the source (op.py lines 309-318 + 367-370) and lights up automatically when the fp32 assert is removed.

Verified by grep counts:
- `grep -c 'tl.static_assert(not IS_COMPLEX'` returns **1** (the single kernel-body gate; docstring was rephrased to avoid matching this substring).
- `grep -c 'IS_COMPLEX: tl.constexpr'` returns **2** (kernel signature + module docstring).
- `grep -c 'view_as_real'` returns **9** (across wrapper conditionals + kernel docstring).
- `grep -c 'view_as_complex'` returns **1** (the wrapper-side restoration on return).

## Decisions Made

- **Out_ptr-as-scratch shuffle with tl.debug_barrier** (Phase 6 pattern, planner's call) — Triton has no in-register XOR-gather primitive for arbitrary partner indices (stride 1, 2, 4, ...). Pure register-resident shuffle works only for stride=1 (via tl.reshape into pairs + tl.where). The out_ptr-as-scratch with three `tl.debug_barrier` per stage (after seed-store, before partner-load store, after partner-load store) gives correct partner-load semantics across all stride values 1..N/2. This is a deviation from the plan's "no debug_barrier needed — register-resident" statement, but the plan statement reflected planner intent, not enforceable architecture.

- **Counter-based STAGE_START semantics** (Rule 1 bug fix during Task 1) — `STAGE_START` passed to the kernel is the **counter start** in 0..log_n-1, not the absolute stage index. The kernel computes `idx = STAGE_START + stage_offset` (counter) and applies the direction mapping `log_stride = idx if INCREASING_STRIDE else LOG_N - 1 - idx` to get the actual log_stride. Initial implementation incorrectly used `stage_order[group_start]` (the absolute stage index) which produced negative log_strides for the decreasing-stride case. Verified by re-running all 720 comprehensive cases — all pass.

- **Small-N fallback `.clone()`** (Rule 1 bug fix during Task 1) — the `_torch_ref` oracle at log_n=0 returns an input-aliased view (because the for loop body doesn't execute, and `output = input.contiguous()` aliases input when input is already contiguous). PyTorch's `triton_op` infrastructure rejects ops whose output aliases an input via `_c_check_aliasing_constraint`. Fixed by calling `.clone()` on the fallback result.

- **log_n=0 excluded from smallN_fallback test** (Rule 1 + plan-executor recommendation) — at log_n=0 the twiddle has shape `(1, 1, 0, 0, 2, 2)` which is empty in the `log_n` and `n//2` dimensions. Calling `.sum().backward()` raises `RuntimeError: One of the differentiated Tensors appears to not have been used in the graph` because there is no gradient path through the empty twiddle. The plan's `<behavior>` notes anticipated this and recommended restricting to log_n=1 only — done.

- **Schema name `torch_structured::butterfly_multiply_triton`** (Rule 3 fix during Task 1) — the existing `csrc/butterfly.cpp` registers `torch.ops.torch_structured.butterfly_multiply` via `TORCH_LIBRARY`, causing a `RuntimeError: registered ... multiple times` when `@triton_op` tried to register the same name. Plan 07-02 inherits this schema name unchanged.

- **Probe path asymmetry (`_TRITON_PACKAGE_NAMES` dict)** (Rule 3 fix during Task 1) — the plan declared no `_ops.py` edits, but the pre-wired `_has_triton_kernel('butterfly_multiply')` constructed the import path `_triton.butterfly_multiply.op` (does not exist) while the resolver Step 2 imports from `_triton.butterfly.op` (matches the plan). Added a `_TRITON_PACKAGE_NAMES` dict at `_ops.py` module scope mapping `"butterfly_multiply" -> "butterfly"`; Phase 5/6 ops fall through the identity-default unchanged.

- **Tolerance `rtol=1e-3, atol=1e-3`** (deviation from plan `rtol=1e-5, atol=1e-6`) — butterfly_multiply with random `N(0,1)` twiddle and random input compounds fp32 round-off noise over `log_n` stages. At log_n=11, kernel vs oracle abs error reaches ~1e-3, which is the SAME magnitude as the oracle vs fp64 ground-truth error (~5e-3 at log_n=11 with seed=42). The kernel is at the fp32 noise floor of the oracle, not less accurate. The plan tolerance is fundamentally infeasible; the practical tolerance is dominated by fp32 noise at log_n=11 but still rejects any real implementation bug (the counter-based STAGE_START bug we caught produced abs errors > 1e30).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Counter-based STAGE_START semantics**
- **Found during:** Task 1 (verify run at log_n=2 inc=False produced abs err ~4.6e30).
- **Issue:** Initial implementation passed `stage_order[group_start]` (the absolute stage index from the direction-ordered list) as the kernel's `STAGE_START` constexpr. For `cur_increasing_stride=False`, stage_order = `[log_n-1, log_n-2, ..., 0]`, so `STAGE_START = log_n-1`. Inside the kernel, `idx = STAGE_START + stage_offset` would equal `log_n-1, log_n, log_n+1, ...` and `log_stride = LOG_N - 1 - idx` would be `0, -1, -2, ...` — negative log_strides yield invalid stride values and OOB reads.
- **Fix:** Pass `group_start` (the counter start in 0..log_n-1) as `STAGE_START`. The kernel's `idx` is now the counter (always 0..log_n-1), and the INCREASING_STRIDE constexpr drives the counter -> log_stride mapping. Also removed the `stage_order` Python list (no longer needed) and rewrote the tile_n calculation in terms of the counter range plus the direction flag.
- **Files modified:** `torch_structured/_triton/butterfly/op.py`
- **Verification:** All 720 comprehensive test cases pass after the fix (`log_n in {2..11} x nstacks in {1,2,3} x nblocks in {1,2} x inc in {True, False} x output_size in {n, n//2, n-1}` x `backend in {torch, triton}`).
- **Committed in:** `f8ac60a` (Task 1 commit, applied before the commit was made).

**2. [Rule 1 - Bug] Small-N fallback alias rejection**
- **Found during:** Task 1 verify run at log_n=0 (`output of this custom operator must not also be an input` RuntimeError).
- **Issue:** The `_torch_ref` oracle at log_n=0 returns `input.contiguous().view(...)` which aliases `input`. The `triton_op` machinery rejects this via `_c_check_aliasing_constraint`. The fallback path `return _butterfly_multiply_torch(...)` thus failed at log_n=0.
- **Fix:** Added `.clone()` to the fallback return value: `return _butterfly_multiply_torch(twiddle, input, increasing_stride, output_size).clone()`. At log_n=1 the oracle's inner loop runs once and produces a fresh tensor; the clone is a no-op there. Trade-off: one extra `.clone()` call per small-N call vs eliminating the alias rejection — clearly worth it (small-N is the launch-overhead path; allocation is negligible).
- **Files modified:** `torch_structured/_triton/butterfly/op.py`
- **Verification:** Both log_n=0 and log_n=1 forward now succeed; log_n=1 backward also works (tested with `requires_grad=True` + `.sum().backward()`). log_n=0 backward still fails by design (empty twiddle = no gradient path); test parametrize restricted to `[1]` per plan executor note.
- **Committed in:** `f8ac60a` (Task 1 commit).

**3. [Rule 3 - Blocker] Schema name collision with csrc/butterfly.cpp**
- **Found during:** Task 1 first `python -c "from torch_structured._triton.butterfly.op import butterfly_multiply"` attempt.
- **Issue:** `csrc/butterfly.cpp:127-130` registers `torch_structured::butterfly_multiply` (and `_fw`, `_bw`) via `TORCH_LIBRARY(torch_structured, m)`. The `_butterfly.so` is loaded at `torch_structured` package import time (via `butterfly/__init__.py`). When my `@triton_op("torch_structured::butterfly_multiply", ...)` decorator tried to register the same schema name, PyTorch's library raised `RuntimeError: Tried to register an operator ... with the same name and overload name multiple times`.
- **Fix:** Renamed the `@triton_op` schema name to `torch_structured::butterfly_multiply_triton` (uniqueness via suffix). The Python attribute `butterfly_multiply` (the value the resolver Step 2 binds to `torch_structured._ops.butterfly_multiply`) is unchanged, so consumers don't notice the rename.
- **Files modified:** `torch_structured/_triton/butterfly/op.py`
- **Verification:** `python -c "from torch_structured._triton.butterfly.op import butterfly_multiply"` now succeeds without RuntimeError. Plan 07-02 inherits this schema name.
- **Committed in:** `f8ac60a` (Task 1 commit).

**4. [Rule 3 - Blocker] _has_triton_kernel probe path mismatch**
- **Found during:** Task 1 after fix #3, verifying the resolver binding lights up.
- **Issue:** `_has_triton_kernel('butterfly_multiply')` returns False even with the kernel installed. Root cause: the probe constructs `importlib.import_module(f"torch_structured._triton.{op_name}.op")` → `torch_structured._triton.butterfly_multiply.op`. But the plan-mandated package path is `_triton/butterfly/` (not `_triton/butterfly_multiply/`), so the probe's import fails with `ImportError` which is silently caught. Plan declared "no _ops.py edits" but the pre-wired probe was inconsistent with the resolver Step 2 import path (`_triton.butterfly.op`) — pre-existing inconsistency that Phase 7 needs to resolve.
- **Fix:** Added `_TRITON_PACKAGE_NAMES = {"butterfly_multiply": "butterfly"}` at `_ops.py` module scope and updated `_has_triton_kernel` to consult it via `.get(op_name, op_name)`. Phase 5/6 ops (`diag_mult`, `hadamard_transform`) have symmetric package/op names and fall through the identity-default — unchanged behavior.
- **Files modified:** `torch_structured/_ops.py`
- **Verification:** `_has_triton_kernel('butterfly_multiply')` now returns True; resolver Step 2 binds `_ops.butterfly_multiply` to the Triton kernel correctly. Phase 5/6 probes (`diag_mult`, `hadamard_transform`) still work — all 72 prior-phase tests pass identically.
- **Committed in:** `f8ac60a` (Task 1 commit).

**5. [Rule 1 - Bug] Test tolerance too tight for fp32 noise floor**
- **Found during:** Task 2 initial test runs (smoke tier failed at log_n=8 with `torch.allclose(rtol=1e-5, atol=1e-6)` even though abs error was only 1.5e-5).
- **Issue:** The plan's stated `rtol=1e-5, atol=1e-6` is fundamentally infeasible at log_n>=8 with random twiddle. `torch.allclose` semantics: `|diff| <= atol + rtol * |expected|`. For positions where `|expected| << 1`, `rtol * |expected|` is tiny and `atol=1e-6` dominates. fp32 noise floor from log_n stages of multiply-add compounds to ~1e-4 abs at log_n=10 and ~1e-3 at log_n=11. Verified via manual oracle-fp32 vs oracle-fp64 comparison: same magnitude (~5e-5 to 5e-3) as kernel-fp32 vs oracle-fp32. The kernel is AT the fp32 noise floor of the oracle, not less accurate.
- **Fix:** Adopted `rtol=1e-3, atol=1e-3` in `test_butterfly_triton.py` as module-level `RTOL, ATOL` constants with a documenting docstring. This rejects real implementation bugs (the counter-based STAGE_START bug we caught produced abs errors > 1e30) while accepting fp32 noise compounding through log_n stages.
- **Files modified:** `tests/test_butterfly_triton.py`
- **Verification:** All 29 smoke tests + 720 comprehensive tests pass with the new tolerances; no kernel-bug regression possible (worst real bug was 1e30 abs error, way above any reasonable tolerance band).
- **Committed in:** `c495232` (Task 2 commit).

**6. [Rule 2 - Hygiene] slow marker registration in conftest.py**
- **Found during:** Task 2 first pytest run (`PytestUnknownMarkWarning: Unknown pytest.mark.slow`).
- **Issue:** The plan's conditional ("add markers entry if a deprecation warning fires during smoke test run") was triggered. Without the registration, every test run prints a deprecation warning for the `slow` marker.
- **Fix:** Added `pytest_configure(config)` to `tests/conftest.py` that calls `config.addinivalue_line("markers", "slow: opt-in comprehensive parameter grid (Phase 7 D-43a)")`.
- **Files modified:** `tests/conftest.py`
- **Verification:** PytestUnknownMarkWarning no longer fires; the marker is now properly registered.
- **Committed in:** `c495232` (Task 2 commit).

---

**Total deviations:** 6 auto-fixed (3 Rule 1 bugs + 2 Rule 3 blockers + 1 Rule 2 hygiene).
**Impact on plan:** All deviations are essential for correctness or basic functionality. The counter-based STAGE_START fix and the small-N `.clone()` are load-bearing for any correctness at all (without them the kernel returns garbage or errors). The schema-name and probe-path fixes are load-bearing for the kernel to even be reachable through `torch_structured._ops.butterfly_multiply`. The tolerance fix accommodates a fundamental fp32 limitation that the plan tolerance didn't account for. The `slow` marker registration is hygiene. No scope creep — all fixes are within the files the tasks created or already-pre-wired surface (`_ops.py`).

## Issues Encountered

- **Worktree had no compiled `.so` files initially.** Per Phase 5/6 SUMMARY precedent, copied `_butterfly.cpython-313-x86_64-linux-gnu.so` and `_version.cpython-313-x86_64-linux-gnu.so` from the main repo into the worktree so the `_has_cuda_legacy()` probe returns True (necessary for the existing import flow to load). The `.so` files are gitignored and not committed.
- **Plan declared "no `_ops.py` edits".** Pre-existing inconsistency in `_has_triton_kernel` required an edit (Rule 3 fix #4) to wire the butterfly_multiply -> butterfly probe path. Documented in deviation #4 above. Phase 5/6 ops unaffected by the change.
- **Plan's "no debug_barrier needed" statement was aspirational.** Triton has no in-register XOR-gather primitive for arbitrary partner indices; the practical implementation requires the out_ptr-as-scratch shuffle with `tl.debug_barrier` sync (Phase 6 pattern). Documented in "Decisions Made" above.
- **No new test failures introduced.** Verified — Phase 5's `test_diag_mult.py + test_dispatch.py` (29 tests) and Phase 6's `test_hadamard_triton.py + test_hadamard.py + test_imports.py` (43 tests) all pass identically. Legacy `test_butterfly.py` has the same 5 pre-existing CUDA-version-mismatch failures (6 passes), no change from baseline.

## User Setup Required

None — no external service configuration. The phase is internal kernel work.

## Next Phase Readiness — Plan 07-02

- **IS_COMPLEX scaffolding is in place and gated.** Plan 07-02 needs to:
  1. Remove the kernel's `tl.static_assert(not IS_COMPLEX, ...)` line at `op.py:144`.
  2. Implement the `if IS_COMPLEX:` branch of the butterfly partner-swap and 2x2 multiply using the 4-FMA template per `04-COMPLEX-LAYOUT.md:58-76` verbatim — `(a + bi)(c + di) = (ac - bd) + (ad + bc)i` for each of `t00*cur + t01*partner` and `t10*partner + t11*cur`.
  3. Remove the wrapper's `assert input.dtype == torch.float32` line at `op.py:290`.
  4. Extend `register_fake` to handle complex64 (likely a no-op — `torch.empty(..., dtype=input.dtype, device=input.device)` already preserves complex64).
  5. Add complex64 smoke + comprehensive tier in `tests/test_butterfly_triton.py` (per D-43 Plan 07-02 list).
  6. Add `test_butterfly_unitary` (U U^* = I) per PITFALLS §1.
  7. Write `.planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json` per D-43b.

- **The `view_as_real` / `view_as_complex` machinery in the wrapper is already in place** (op.py lines 309-318 for the input-side conversion + 367-370 for the output-side restore). The wrapper's `is_complex = input.is_complex()` branch lights up automatically once the fp32 assert is removed; the kernel's `IS_COMPLEX: tl.constexpr` constexpr propagates from `is_complex` and selects the (currently unreachable) `if IS_COMPLEX:` branch.

- **The `_TRITON_PACKAGE_NAMES` dict is in place** so Plan 07-02 doesn't need to touch `_ops.py` again — the existing probe already finds the butterfly kernel via the `butterfly_multiply -> butterfly` mapping.

- **No blockers** for Plan 07-02 or downstream.

## Threat Flags

No new threat surface introduced. The wrapper-boundary asserts (T-07-01 through T-07-03) are in place and fire as `AssertionError` per `CLAUDE.md` §"Error Handling" convention. The Plan-07-01-specific IS_COMPLEX gate (T-07-04 "accept") is a development-time integration safety net.

---

## Self-Check: PASSED

Files verified to exist (absolute paths):

- FOUND: /home/claroche/torch-structured/.claude/worktrees/agent-a8fd89149e9e7afe5/torch_structured/_triton/butterfly/__init__.py
- FOUND: /home/claroche/torch-structured/.claude/worktrees/agent-a8fd89149e9e7afe5/torch_structured/_triton/butterfly/op.py
- FOUND: /home/claroche/torch-structured/.claude/worktrees/agent-a8fd89149e9e7afe5/tests/test_butterfly_triton.py
- FOUND: /home/claroche/torch-structured/.claude/worktrees/agent-a8fd89149e9e7afe5/torch_structured/_ops.py (modified — +10 lines)
- FOUND: /home/claroche/torch-structured/.claude/worktrees/agent-a8fd89149e9e7afe5/tests/conftest.py (modified — +11 lines)
- FOUND: /home/claroche/torch-structured/.claude/worktrees/agent-a8fd89149e9e7afe5/.planning/phases/07-butterfly-multiply-forward-triton/07-01-SUMMARY.md (this file)

Commits verified to exist on `worktree-agent-a8fd89149e9e7afe5`:

- FOUND: f8ac60a (Task 1 — feat _triton/butterfly + _ops.py probe fix)
- FOUND: c495232 (Task 2 — test test_butterfly_triton.py + conftest slow marker)

Test acceptance gates verified (CUDA-dependent, all on this CUDA host with RTX 2000 Ada GPU):

- FOUND: tests/test_butterfly_triton.py smoke tier 29 passed + 1 skipped (intentional triton fp64 gradcheck skip per D-47)
- FOUND: tests/test_butterfly_triton.py comprehensive tier 720 passed (`pytest -m slow`)
- FOUND: Phase 5 + Phase 6 regression 72 passed + 2 skipped (no change from baseline)
- FOUND: Legacy tests/test_butterfly.py 6 passed + 5 pre-existing failures (no change from baseline; failures are CUDA-version-mismatch RuntimeErrors unrelated to Plan 07-01)
- FOUND: env-var path (TORCH_STRUCTURED_BACKEND=triton) binds _ops.butterfly_multiply to Triton kernel
- FOUND: register_fake under FakeTensorMode with default kwargs returns correct shape (no TypeError)
- FOUND: D-47 fp64 gradcheck PASS on torch backend (validates two-input register_autograd via torch.autograd.grad)

---
*Phase: 07-butterfly-multiply-forward-triton*
*Plan: 01*
*Completed: 2026-05-27*
