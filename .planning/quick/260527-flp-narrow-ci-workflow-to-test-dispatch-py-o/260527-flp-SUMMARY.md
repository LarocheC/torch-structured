---
phase: 260527-flp
plan: 01
type: execute
wave: 1
status: completed
completed_date: 2026-05-27
duration_seconds: ~30
tasks_completed: 1
files_modified: [.github/workflows/test.yml]
commits: [1108185]
requirements: [TEST-05-followup]
key-decisions:
  - "CI workflow scoped to tests/test_dispatch.py — the suite the Phase 4 workflow was designed to gate; legacy tests excluded as deferred items"
  - "Switched -x → -v so the dispatch suite's per-test output is legible on PR check pages; -x is moot for a single-file run"
---

# Phase 260527-flp Plan 01: Narrow CI Workflow to test_dispatch.py Summary

One-line edit to `.github/workflows/test.yml:48` changing `pytest tests/ -x` → `pytest tests/test_dispatch.py -v`, scoping CI to the Phase 4 dispatch-infrastructure test suite and bypassing pre-existing deferred-item failures in the legacy `tests/` tree.

## What Changed

| File                       | Change                                                                                  |
| -------------------------- | --------------------------------------------------------------------------------------- |
| `.github/workflows/test.yml` | Line 48: `      - run: pytest tests/ -x` → `      - run: pytest tests/test_dispatch.py -v` |

Single-line diff, no collateral edits. Caching block (lines 37–44), install step (line 46), and all other workflow lines remain untouched.

## Why

The CI workflow shipped in Phase 4 Plan 04-02 (TEST-05) was scoped to gate the dispatch infrastructure and the `_demo_identity_op` triton_op skeleton — i.e. `tests/test_dispatch.py`. Running the full `tests/` tree under that workflow surfaces two pre-existing failure modes unrelated to v1.2 (both documented in `.planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/deferred-items.md`):

1. **Legacy C++ extension tests** (`test_butterfly.py`, `test_multiply.py`, `test_permutation.py`) — the worktree is built with `FORCE_CPU=1` while the bundled torch CUDA mismatches the host CUDA; the `.so` registers ops that raise at runtime on CUDA tensors. The right fix is environment-side (CUDA-aware build), out of scope here.
2. **`tests/test_special.py`** — fails at collection time because `pywt` (PyWavelets) is not declared in the `test` extra. The right fix is packaging-side (add `pywt` or `importorskip`), out of scope here.

Running these under CI today produces red builds that mask real regressions in the dispatch layer the workflow is supposed to gate. Narrowing to `tests/test_dispatch.py` restores CI as a meaningful signal. The `-x` → `-v` swap improves PR-page readability for the dispatch suite; `-x` (stop on first failure) is moot for a single-file run.

## Verification

| Check                                                            | Expected exit | Actual exit |
| ---------------------------------------------------------------- | ------------- | ----------- |
| `grep "pytest tests/test_dispatch.py" .github/workflows/test.yml` | 0             | 0           |
| `grep "pytest tests/ -x" .github/workflows/test.yml`              | 1             | 1           |
| `git diff` shows one changed line on line 48                      | 1 hunk        | 1 hunk      |

Diff:
```
-      - run: pytest tests/ -x
+      - run: pytest tests/test_dispatch.py -v
```

## Deviations from Plan

None — plan executed exactly as written. Single-task plan, single-line edit, all verification criteria pass.

## Self-Check: PASSED

- File `.github/workflows/test.yml` exists and contains `pytest tests/test_dispatch.py -v` on line 48 (verified by grep)
- Commit `1108185` exists in `git log --oneline` on branch `worktree-agent-ae16b33eff8120292`
- No unexpected file deletions in the commit (`git diff --diff-filter=D HEAD~1 HEAD` returned empty)
- Working tree clean after commit (`git status --short` returned empty)
