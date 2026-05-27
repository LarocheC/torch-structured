---
phase: 260527-flp
plan: 01
type: execute
wave: 1
depends_on: []
files_modified: [.github/workflows/test.yml]
autonomous: true
requirements: [TEST-05-followup]
must_haves:
  truths:
    - "CI runs only tests/test_dispatch.py, not the full tests/ tree"
    - "CI uses -v (verbose) so the dispatch suite output is readable on PR pages"
    - "The legacy CUDA-stub failures (test_butterfly/test_multiply/test_permutation) and the pywt import gap (test_special) no longer trip CI"
  artifacts:
    - path: ".github/workflows/test.yml"
      provides: "CI test step narrowed to tests/test_dispatch.py -v"
      contains: "pytest tests/test_dispatch.py -v"
  key_links:
    - from: ".github/workflows/test.yml:48"
      to: "tests/test_dispatch.py"
      via: "pytest invocation"
      pattern: "pytest tests/test_dispatch.py"
---

<objective>
Narrow the CI workflow's pytest invocation to only run `tests/test_dispatch.py` (the Phase 4 dispatch-infrastructure test suite), so CI validates the new dispatch layer and demonstrator op without being blocked by pre-existing environment-driven failures in the legacy test files.

Purpose: The Phase 4 CI workflow added in Plan 04-02 (TEST-05) exists to gate the dispatch infrastructure and the `_demo_identity_op` triton_op skeleton. The remaining tests in `tests/` have pre-existing failures unrelated to v1.2 (documented in `.planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/deferred-items.md`):

  1. Legacy C++ extension tests (`test_butterfly.py`, `test_multiply.py`, `test_permutation.py`) fail because the worktree is built with `FORCE_CPU=1` while the bundled torch CUDA mismatches the host CUDA — the `.so` registers ops that raise at runtime on CUDA tensors.
  2. `tests/test_special.py` fails at collection time because `pywt` (PyWavelets) is not declared in the `test` extra.

Both are deferred items — the right fix is environment-side (CUDA-aware build) and packaging-side (add `pywt` or `importorskip`), neither of which is in scope here. Running them under CI today produces red builds that mask real regressions in the dispatch layer we actually want gated.

Output: A one-line edit to `.github/workflows/test.yml:48` changing `pytest tests/ -x` → `pytest tests/test_dispatch.py -v`.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
</execution_context>

<context>
@.planning/STATE.md
@.github/workflows/test.yml
@.planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/deferred-items.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Narrow CI pytest invocation to tests/test_dispatch.py</name>
  <files>.github/workflows/test.yml</files>
  <action>Edit `.github/workflows/test.yml` line 48. Replace the existing pytest step:

    - run: pytest tests/ -x

with:

    - run: pytest tests/test_dispatch.py -v

Use the Edit tool on the exact line (the only occurrence in the file). Do NOT modify any other line, do NOT touch the caching block above (lines 37–44), do NOT change the install step (line 46), do NOT add comments above the pytest line. Preserve YAML indentation exactly (the existing `- run:` is indented six spaces under `steps:`).

Rationale (do not embed in the file, but for executor awareness): the new CI workflow shipped in Phase 4 Plan 04-02 (TEST-05) was scoped to validate the dispatch infrastructure and the `_demo_identity_op` triton_op skeleton — i.e. `tests/test_dispatch.py`. The legacy tests in `tests/` have pre-existing CUDA-stub `.so` failures and a missing `pywt` import (both deferred per `.planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/deferred-items.md`). Switching `-x` → `-v` makes the dispatch suite's per-test output legible in PR check logs; `-x` is moot for a single-file run.</action>
  <verify>
    <automated>grep -q "pytest tests/test_dispatch.py -v" .github/workflows/test.yml && ! grep -q "pytest tests/ -x" .github/workflows/test.yml</automated>
  </verify>
  <done>`.github/workflows/test.yml` line 48 reads `      - run: pytest tests/test_dispatch.py -v` (six-space indentation, verbose flag). The substring `pytest tests/ -x` no longer appears anywhere in the file. No other line in the file is modified.</done>
</task>

</tasks>

<verification>
- `grep "pytest tests/test_dispatch.py" .github/workflows/test.yml` exits 0 (new invocation present).
- `grep "pytest tests/ -x" .github/workflows/test.yml` exits 1 (old invocation gone).
- `git diff .github/workflows/test.yml` shows exactly one changed line (line 48), `-` removing `pytest tests/ -x`, `+` adding `pytest tests/test_dispatch.py -v`. No other hunks.
- YAML structure remains valid (no syntax errors): the file still parses as a GitHub Actions workflow. A `python -c "import yaml; yaml.safe_load(open('.github/workflows/test.yml'))"` sanity check passes if executed.
</verification>

<success_criteria>
- CI workflow runs `pytest tests/test_dispatch.py -v` on push/PR instead of the full `tests/` tree.
- The legacy CUDA-stub failures and the `pywt` import gap (both deferred items from Phase 4) no longer surface in CI runs.
- Triton JIT cache step and `pip install -e .[test]` step are unchanged.
- Single-line diff; no collateral edits.
</success_criteria>

<output>
After completion, create `.planning/quick/260527-flp-narrow-ci-workflow-to-test-dispatch-py-o/260527-flp-SUMMARY.md` capturing:
- What changed (one-line edit, line 48)
- Why (Phase 4 follow-up: scope CI to the dispatch suite the workflow was designed to gate; deferred-items.md references for the excluded failures)
- Verification command results (the two grep checks above)
</output>
