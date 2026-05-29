---
phase: quick-260529-bdr
plan: 01
status: complete
subsystem: testing
tags: [deprecation-warning, cuda-legacy, backend-resolver, pytest, fp32-tolerance, gradcheck-skip]

# Dependency graph
requires:
  - phase: 09-backend-routing
    provides: "_ops._resolve routing closure + _has_cuda_legacy probes + 3-axis backend fixture"
  - phase: 10-deprecation
    provides: "_cuda_legacy import-time DeprecationWarning (D-74/D-74b)"
provides:
  - "DEPR-02 fix: CUDA DeprecationWarning decoupled from module-import timing; fires only on explicit set_backend('cuda')"
  - "TEST-01/02/03 fix: 3-axis backend gate honestly green on matched-CUDA hardware"
affects: [v1.3-cuda-default-disable, milestone-audit-defect-closure]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Explicit idempotent warn_cuda_deprecation() emitter (replaces import-time warning side effect)"
    - "Scale-aware sqrt(n) fp32 atol envelope for FWHT cross-backend assertions"
    - "Per-axis pytest.skip extended from triton-only to (triton, cuda) for fp32-only kernels"

key-files:
  created:
    - .planning/quick/260529-bdr-fix-two-v1-2-audit-defects-depr-02-warni/260529-bdr-SUMMARY.md
  modified:
    - torch_structured/_cuda_legacy/__init__.py
    - torch_structured/_ops.py
    - tests/test_diag_mult.py
    - tests/structured/test_hadamard_triton.py
    - tests/test_butterfly_triton.py

key-decisions:
  - "Decouple the warning into warn_cuda_deprecation() rather than only wrapping the leak site — the _WARNED trap (a suppressed first import would set _WARNED=True and silence the later explicit path) made suppression-alone insufficient"
  - "Legacy CUDA diag_mult is real-fp32-only (no complex kernel) — skip complex64 forward on the cuda axis rather than mask with tolerance"
  - "FWHT cuda-axis fp32 noise floor grows ~sqrt(n); the flat phase9 log_n>=8 -> 1e-5 rule both under-shoots (log_n>=10) and over-shoots (log_n 4-7), so use a seeded sqrt(n)-scaled envelope"
  - "Seed the two unseeded forward comprehensive butterfly tiers (every backward comprehensive test in the file already seeds) — they flaked at log_n=11 on BOTH triton and cuda axes; seeding does not weaken tolerance"

patterns-established:
  - "warn_cuda_deprecation(): module import is side-effect-free w.r.t. the warning; explicit-selection call sites emit, _WARNED gate keeps at-most-once-per-process"
  - "Leak-site import wrapped in catch_warnings()/simplefilter('ignore', DeprecationWarning) mirroring the per-op probe pattern at _ops.py:133-139"

requirements-completed: [DEPR-02, TEST-01, TEST-02, TEST-03]

# Metrics
duration: ~35min
completed: 2026-05-29
---

# Phase quick-260529-bdr Plan 01: Fix two v1.2 audit defects (DEPR-02 + cuda-axis test gaps) Summary

**DeprecationWarning now fires 0 times on a bare default-backend import (even with a CUDA build present) and exactly once on explicit set_backend('cuda'); the full 3-axis {torch,triton,cuda} backend gate is honestly green (3042 passed, 0 failures) across three consecutive runs on matched-CUDA hardware.**

## Performance

- **Duration:** ~35 min
- **Tasks:** 3/3 (Task 3 is verification-only)
- **Files modified:** 5 (exactly the `files_modified` set; version strings untouched)

## Accomplishments

### Task 1 — DEPR-02 warning leak (commit f290411)
- Added an idempotent `warn_cuda_deprecation()` emitter in `_cuda_legacy/__init__.py`; the module body no longer emits the warning at import time, so importing `_cuda_legacy` (leak-site import or per-op probe) is side-effect-free w.r.t. the warning. Exported in `__all__`.
- Wrapped the routing-fallback leak-site import (`_ops.py` ~329-336) in `warnings.catch_warnings() + simplefilter('ignore', DeprecationWarning)`, mirroring the per-op probe pattern at `_ops.py:133-139`.
- Called `warn_cuda_deprecation()` on the three explicit `actual=='cuda'` binding paths (butterfly ~380, diag_mult ~396, hadamard ~415). The `_WARNED` gate collapses all three in a single `_resolve('cuda')` to at-most-one warning.
- Resolves the `_WARNED` trap documented in the plan's `<interfaces>`: because the warning is now an explicit call rather than an import-time side effect, the suppressed leak-site import does not burn the gate against a later explicit-cuda selection.

### Task 2 — cuda-axis test gaps (commit 449e385)
- **diag_mult:** `n>=128` fp32 atol -> 1e-5 (`test_diag_mult_eager_fp32`); cuda skip on `test_diag_mult_eager_complex64` (legacy `_diag_mult_cuda` is real-fp32 only — raises "expected Float but found ComplexFloat"); cuda skip on `test_diag_mult_gradcheck_fp64_real` and `test_diag_mult_gradcheck_fp64_complex`; cuda skip guarding only the fp64 backward block of `test_diag_mult_shift_grid` (its fp32 forward, N=16 < 128, still runs on the cuda axis at atol=1e-6).
- **hadamard:** seeded + `sqrt(n)`-scaled fp32 atol on `test_hadamard_eager_fp32`; seeded `test_hadamard_normalize`; cuda skip on `test_hadamard_gradcheck_fp64`.
- **butterfly:** extended the four existing `triton` fp64/complex128 gradcheck skips to also cover `cuda` (`test_butterfly_gradcheck_fp64`, `test_butterfly_gradcheck_complex64`, `test_butterfly_backward_gradcheck_fp64`, `test_butterfly_backward_complex64_gradcheck_fp64`); seeded the two forward comprehensive tiers (fp32 + complex64).

### Task 3 — verification (no source edits)
- Default import under `-W error::DeprecationWarning` exits 0.
- Explicit `set_backend('cuda')` in a fresh process records exactly 1 DeprecationWarning.
- The routed `butterfly_multiply::11::complex64::forward` cell (marked `route_to_cuda: True` in `_routing.json`) executes on the triton backend via the routing closure (`_ops.butterfly_multiply.__name__ == '_routed_butterfly_multiply'`, `_should_route_to_cuda(...) == True`), producing a finite output of shape (1,1,2048) — no routing regression.
- Combined 3-axis suite: 3042 passed, 1484 skipped, 0 failures (3 consecutive runs).
- Version strings unchanged (`1.2.1` in `pyproject.toml` and `torch_structured/__init__.py`); nothing published; no dist/build/.so artifacts committed.

## Verification Output (actual)

- **(a) default-import-no-warning:** `python -W error::DeprecationWarning -c "import torch_structured"` -> exit 0, `OK default import, no DeprecationWarning`.
- **(b) explicit-cuda-warns-once (fresh process):** `CUDA-WARN-COUNT 1`.
- **(c) routed-cell-uses-CUDA:** `_should_route_to_cuda(log_n=11, complex64, forward) = True`; bound fn `_routed_butterfly_multiply`; `OK routed cell executes, backend= triton shape= (1, 1, 2048)`.
- **(d) 3-axis suite:** `3042 passed, 1484 skipped, 1 warning` (the 1 warning is the expected single cuda-axis DeprecationWarning) — reproduced 3x.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] FWHT fp32 atol: flat log_n>=8 -> 1e-5 rule was both under- and over-tight; replaced with a seeded sqrt(n)-scaled envelope**
- **Found during:** Task 2 (the plan's `<interfaces>` cited the phase9 flat two-tier rule).
- **Issue:** The per-op `test_hadamard_eager_fp32` is unseeded, so the FWHT cuda-kernel noise floor was seed-dependent and flaky. With seed(0) the measured drift is ~1.2e-6 at log_n=4 rising monotonically to ~3e-5 at log_n=12 — the flat `1e-6 if log_n<8 else 1e-5` rule FAILS at log_n 4-7 (just above 1e-6) and at log_n 10-12 (above 1e-5). `test_phase9_integration.py`'s flat rule only happened to land OK because it seeds AND tests one size.
- **Fix:** Seed `torch.manual_seed(0)` (matching the phase9 integration test and the FWHT non-associativity model) and use `atol = max(1e-6, 1.5e-6 * sqrt(n))`, which tracks the documented noise floor at every size with ~3x headroom while still rejecting real (>1e30-class) bugs.
- **Files modified:** `tests/structured/test_hadamard_triton.py`
- **Commit:** 449e385

**2. [Rule 2 - Missing coverage] diag_mult complex64 forward on the cuda axis was not anticipated by the plan**
- **Found during:** Task 2 (`test_diag_mult_eager_complex64[cuda]` failed).
- **Issue:** The plan's `<interfaces>` covered only the fp32-atol and fp64-gradcheck cuda gaps for diag_mult, but the legacy `_diag_mult_cuda` kernel has NO complex support at all (raises "expected Float but found ComplexFloat"). This is the same fp32-only-kernel class as the fp64 skips.
- **Fix:** Added a cuda-axis skip to `test_diag_mult_eager_complex64` citing TEST-03-cuda-axis (complex64 diag_mult is covered on torch + triton).
- **Files modified:** `tests/test_diag_mult.py`
- **Commit:** 449e385

**3. [Rule 1 - Bug] Two unseeded forward comprehensive butterfly tiers flaked at log_n=11 on BOTH triton and cuda axes**
- **Found during:** Task 2 (first full-suite run failed `test_butterfly_comprehensive[triton-11-...]` and `[cuda-11-...]`; standalone reruns passed — order/seed-dependent).
- **Issue:** `test_butterfly_comprehensive` (fp32) and `test_butterfly_eager_complex64_grid` (complex64) were the only comprehensive tiers in the file NOT calling `manual_seed(0)` (all backward comprehensive tests already seed). With random RNG state the log_n=11 fp32 noise floor occasionally produced a small-magnitude output element whose abs error exceeded the fixed atol=1e-3.
- **Fix:** Added `torch.manual_seed(0)` to both forward comprehensive tiers, matching the file's established convention. Does not weaken any tolerance — only makes the existing envelope deterministic.
- **Files modified:** `tests/test_butterfly_triton.py`
- **Commit:** 449e385

The complex64 *gradcheck* butterfly failures (`test_butterfly_gradcheck_complex64`, `test_butterfly_backward_complex64_gradcheck_fp64`) were the anticipated "fp64-equivalent gradcheck on cuda" case (gradcheck promotes to ComplexDouble, which the fp32/complex64-only kernel does not implement) — handled by the planned cuda-skip extension, not a deviation.

## Authentication Gates

None.

## Threat Flags

None — no new network endpoints, auth paths, file access, or schema changes introduced. The two threat-register `mitigate`/`accept` dispositions (T-bdr-02 scoped `catch_warnings()` suppression; T-bdr-03 bounded atol widening) were honored: the suppression is a single scoped `with` block around one import (no global `warnings.filters` mutation), and atol widening tracks the documented fp32 noise floor only (real bugs still rejected).

## Known Stubs

None.

## Self-Check: PASSED

- All 5 modified source files exist.
- SUMMARY.md created.
- Commits f290411 (Task 1) and 449e385 (Task 2) present in git log.
- `warn_cuda_deprecation` emitter present in `_cuda_legacy/__init__.py`.
