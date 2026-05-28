---
phase: quick-260528-tv9
plan: 01
subsystem: tests/perf-gate
tags: [TEST-04, perf-gate, triton, do_bench, D-65b]
requires: [".planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json"]
provides: ["TEST-04 perf gate reading do_bench_p50_ms"]
affects: ["tests/test_perf_grid.py"]
tech-stack:
  added: []
  patterns: ["JSON-driven perf gate", "triton.testing.do_bench as canonical measurement"]
key-files:
  created: []
  modified: ["tests/test_perf_grid.py"]
decisions:
  - "TEST-04 gate computes the Triton-vs-CUDA ratio from do_bench_p50_ms (triton.testing.do_bench), not the inflated custom-harness wall_ms_p50. Amends Phase 9 D-65b."
  - "A cell is gateable only when it has BOTH non-null reference_cuda_p50 AND non-null do_bench_p50_ms; null do_bench is skipped, not crashed on."
metrics:
  duration: "~6 min"
  completed: "2026-05-28"
  tasks: 1
  files: 1
requirements: [TEST-04]
---

# Quick 260528-tv9: TEST-04 do_bench Gate Fix Summary

Switched the TEST-04 perf gate `test_perf_gate_triton_at_60pct_cuda` to compute the
Triton-vs-CUDA ratio from the reliable `do_bench_p50_ms` (`triton.testing.do_bench`)
column instead of the inflated custom-harness `wall_ms_p50`, making the gate accurate:
real shortfalls only, plausible ratios.

## What Changed

`tests/test_perf_grid.py` (single file):

1. **Gateable-cell filter** (`cells_with_cuda`): now requires BOTH `reference_cuda_p50 is not None`
   AND `do_bench_p50_ms is not None`. A row missing either is skipped (no crash). The existing
   empty-list soft-skip path is preserved.
2. **Ratio numerator**: changed from `row["wall_ms_p50"]` to `row["do_bench_p50_ms"]`. Denominator
   unchanged (`row["reference_cuda_p50"]`). `THRESHOLD = 1.0 / 0.60` (1.67) unchanged. Routing
   escape-valve logic (`route_to_cuda`) unchanged.
3. **Failure-message f-string**: now reports `triton(do_bench)=<do_bench_p50_ms> ms` as the primary
   Triton figure, `cuda=<reference_cuda_p50> ms`, and `diag wall_ms_p50=<wall_ms_p50> ms` as a
   clearly-labeled diagnostic aside so the over-reporting stays visible.
4. **Gate docstring**: states the gate uses `do_bench_p50_ms` (canonical per TEST-04 wording) and
   that `wall_ms_p50` from `measure_p50_p95` is retained in the baseline only as a diagnostic —
   it over-reports at small kernel sizes because per-iteration `cuda.Event` + `synchronize()`
   overhead dominates 30-250 µs kernels (up to ~11× high). Cites "amends Phase 9 D-65b".
5. **Module docstring**: updated the `wall_ms_p50 / reference_cuda_p50` phrasing to
   `do_bench_p50_ms / reference_cuda_p50` and notes the diagnostic-only role of `wall_ms_p50`.

## Verification

Ran `python -m pytest tests/test_perf_grid.py -v`. (The worktree itself has no compiled
`.so`; the editable install points at the main repo `/home/claroche/torch-structured`, which has
the built extension. The test was therefore executed from the main-repo root against the worktree's
edited test file — the gate reads the committed JSON only, no GPU work happens in the gate.)

Result: **1 failed, 4 passed, 4 skipped** — exactly as designed.

- The 4 selector unit tests that can run (`test_should_route_to_cuda_unknown_cell_returns_false`,
  `test_should_route_to_cuda_disabled_returns_false`, `test_set_routing_enabled_round_trip`,
  and one more) PASS unchanged. The remaining selector tests SKIP because no cell in the dev-host
  bake is marked `route_to_cuda` / no GPU — same skip behavior as before this fix.
- The gate test `test_perf_gate_triton_at_60pct_cuda` still FAILs. **This is EXPECTED and CORRECT.**
  The failure list is now exactly the 6 genuine-shortfall cells with plausible ratios:

  | Cell | do_bench ratio | (old inflated wall ratio) |
  |------|----------------|---------------------------|
  | `butterfly_multiply::10::complex64::forward`  | 1.77× | 3.77× |
  | `butterfly_multiply::11::complex64::forward`  | 3.19× | 6.12× |
  | `butterfly_multiply::9::complex64::backward`  | 3.90× | 4.90× |
  | `butterfly_multiply::10::fp32::backward`      | 3.28× | 4.74× |
  | `butterfly_multiply::11::fp32::backward`      | 3.71× | 6.43× |
  | `butterfly_multiply::11::complex64::backward` | 4.62× | 9.70× |

- **CRITICAL ASSERTION confirmed**: `butterfly_multiply::11::fp32::forward` is NOT in the failure
  list — it flips from the inflated 11.10× (wall) to 1.63× (do_bench), which PASSES. No ratio in
  the output exceeds ~5×; all of the previously-inflated 7-11× ratios are gone.

**Framing (so the gate-still-red state is not mistaken for a failed fix):** the gate is now
ACCURATE — it reports real Triton-slower-than-CUDA cells with plausible ratios rather than artifacts
of the custom-harness measurement overhead. Making the gate fully GREEN requires regenerating
`torch_structured/_routing.json` so the 6 genuine cells route to CUDA via the D-61 escape valve —
that is a SEPARATE, out-of-scope follow-up. The 6 cells staying RED is the correct, intended state.

## Grep / containment assertions

- `ratio = row["do_bench_p50_ms"] / row["reference_cuda_p50"]` present (line 93).
- `row["wall_ms_p50"] / row["reference_cuda_p50"]` no longer appears anywhere (the wall figure
  survives only in the diagnostic aside of the failure message).
- `THRESHOLD = 1.0 / 0.60` (1.67) unchanged.
- `git status --short` shows only `tests/test_perf_grid.py` modified — `measure_p50_p95`
  (in `tests/_baseline_butterfly*.py`), `torch_structured/_routing.json`, and `07-BASELINE.json`
  are untouched.

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- Gate ratio numerator is `do_bench_p50_ms`; `wall_ms_p50` no longer in the ratio computation. ✓
- `cells_with_cuda` requires both `reference_cuda_p50` and `do_bench_p50_ms` non-null. ✓
- THRESHOLD unchanged (1.67). ✓
- `measure_p50_p95`, `_routing.json`, `07-BASELINE.json` untouched (git status shows only the test file). ✓
- Gate failure list ⊆ the 6 genuine-shortfall cells; `11::fp32::forward` NOT in it; no ratio > ~5×. ✓
- Selector unit tests still pass (4 passed, 4 skipped — same as pre-fix skip behavior). ✓

Commit: `9f0fdbe` fix(260528-tv9): TEST-04 gate uses do_bench_p50_ms not inflated wall_ms_p50
