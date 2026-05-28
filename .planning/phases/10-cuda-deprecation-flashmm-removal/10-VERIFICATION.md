---
phase: 10-cuda-deprecation-flashmm-removal
verified: 2026-05-28T17:42:57Z
reverified: 2026-05-28T18:25:00Z
status: passed
score: 11/11 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 10/11
  fix_commit: 77fffb0
  gaps_closed:
    - "D-74b probe silencing: `_has_cuda_legacy_for_op` probes do NOT emit the user-facing DeprecationWarning"
  gaps_remaining: []
  regressions: []
human_verification: []
---

# Phase 10: CUDA Deprecation & flashmm Removal — Verification Report

**Phase Goal:** Triton ships as the default backend for v1.2; the CUDA path remains available but emits a `DeprecationWarning` pointing at the migration timeline; the `_flashmm` MathDx kernel is removed entirely (not ported); `csrc/`, `setup.py`, and `MANIFEST.in` stay in-tree this release pending the two-release deprecation cadence.

**Verified:** 2026-05-28T17:42:57Z
**Status:** gaps_found
**Re-verification:** No — initial verification.

## Executive Summary

10 of 11 must-haves VERIFIED. One LOAD-BEARING gap: **the D-74b probe-silencing wrap does not function correctly** — `tests/test_deprecation.py::test_has_cuda_legacy_probe_does_not_emit_warning` FAILS in-process. The root cause is a Python `warnings`-module semantics bug: `simplefilter('once', DeprecationWarning)` at the top of `_cuda_legacy/__init__.py` prepends a new filter at the front of `warnings.filters`, shadowing the outer `simplefilter('ignore', DeprecationWarning)` installed by the probe's `catch_warnings()` block. Empirically confirmed: the `_has_cuda_legacy_diag_mult` probe emits one user-facing DeprecationWarning. The Phase 9 backend fixture would emit this warning every test run, contradicting the locked D-74b contract.

All other Phase 10 deliverables — file deletions, README/CHANGELOG documentation, setup.py/`__init__.py` edits, the verbatim DeprecationWarning text, the once-per-process simplefilter, the `stacklevel=2` attribution, the DEPR-03 retention surface, the test file structure — are correct.

## Goal Achievement

### Observable Truths (from PLAN frontmatter + ROADMAP SC#1-4)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | **SC#1 / DEPR-01:** Triton is the default backend (no env var → Triton on a CUDA+Triton-capable machine) | VERIFIED | `_ops.py:188-200` `_resolve()` auto resolver is unchanged from Phase 4 D-08 + Phase 9 §0 LANDMINE fix. `Butterfly.forward` delegator at `torch_structured/butterfly/multiply.py:72-77+` routes through `_ops.butterfly_multiply` (re-reads on every call). Phase 10 made no changes to `_ops.py:_resolve()`. |
| 2 | **D-77 scope boundary:** Phase 10 does NOT include a PROJECT.md evolution task | VERIFIED | The plan contains 7 tasks; none modify PROJECT.md. The git log shows the orchestrator handles `docs(phase-XX): evolve PROJECT.md` as a separate commit (e.g., `b40c37f` for Phase 7). |
| 3 | **SC#1 / DEPR-02 / D-74:** `set_backend('cuda')` emits exactly one `DeprecationWarning` with verbatim text, `stacklevel=2` | VERIFIED | `torch_structured/_cuda_legacy/__init__.py:14-32` contains the verbatim text from `04-DEPRECATION-PLAN.md`. `grep -F` on all four load-bearing tokens (`CUDA C++ backend (csrc/) is deprecated`, `default-disabled in v1.3, with full removal in v1.4+`, `Switch to TORCH_STRUCTURED_BACKEND=triton (default in v1.2)`, `See the v1.2 release notes for migration guidance.`) hits exactly once each. `stacklevel=2` confirmed at line 31. Placement is BEFORE the `from .butterfly import` line (line 34). |
| 4 | **SC#1 once-per-process:** subsequent `set_backend('cuda')` calls don't re-fire | VERIFIED | `_cuda_legacy/__init__.py:23` installs `warnings.simplefilter("once", DeprecationWarning)` at module scope. Python's `sys.modules` cache means the module body runs exactly once per process; even if `simplefilter("once")` itself wasn't the gate, the import-cache is. The subprocess test `test_cuda_backend_warning_fires_only_once` was not run on this host (skipped due to `_butterfly.so` absence), but the implementation is correct by inspection. |
| 5 | **LOAD-BEARING D-74b:** `_has_cuda_legacy_for_op` probes do NOT emit the user-facing DeprecationWarning | **FAILED** | **The probe-silencing wrap is functionally broken.** Empirical test: `pytest tests/test_deprecation.py::test_has_cuda_legacy_probe_does_not_emit_warning` FAILS with `AssertionError: D-74b broken: probe emitted 1 user-facing DeprecationWarning(s); expected 0.` Root cause analyzed below in "Gaps" section. |
| 6 | **SC#2 / DEPR-04 / D-73:** `_flashmm` artifacts deleted; `from torch_structured.monarch.flash_mm import flash_mm` raises `ModuleNotFoundError` | VERIFIED | `csrc/flashmm/` directory: absent. `torch_structured/monarch/flash_mm.py`: absent. `tests/monarch/test_flash_mm.py`: absent. Subprocess test confirmed: `python -c "from torch_structured.monarch.flash_mm import flash_mm"` produces `ModuleNotFoundError: No module named 'torch_structured.monarch.flash_mm'`. |
| 7 | **SC#2 / DEPR-04 / D-73c:** `torch_structured/__init__.py:17` docstring no longer references `_flashmm` | VERIFIED | `torch_structured/__init__.py:14-17` reads `Importing compiled C++/CUDA extensions ... try-import their own CUDA modules (\`\`_hadamard_cuda\`\` and \`\`_diag_mult_cuda\`\`) when needed.` `grep -c '_flashmm' torch_structured/__init__.py` = 0. |
| 8 | **SC#2 / DEPR-04 / D-73d:** `setup.py` has no flashmm references | VERIFIED | `grep -c -i flashmm setup.py` = 0. `grep -c TORCH_STRUCTURED_BUILD_FLASHMM setup.py` = 0. `_hadamard_cuda` build (line 88) and `_diag_mult_cuda` build (line 101) and `extensions_dir.glob("*.cpp")` (line 48) all retained. |
| 9 | **SC#2 / DEPR-04 README cleanup:** README has zero `flashmm` / `_flashmm` tokens | VERIFIED | `grep -c 'flashmm\|_flashmm' README.md` = 0. The monarch bullet ends naturally with "...and Hyena implicit long filter." The `### Optional: flashmm extension` subsection is deleted. The test-skip note is the tighter "CUDA-only tests are automatically skipped...". |
| 10 | **SC#3 / DEPR-03 retention:** `csrc/{butterfly,hadamard,diag_mult,cpu,cuda}/`, `setup.py` ext builds, and `MANIFEST.in` retained | VERIFIED (with semantic caveat — see notes) | `csrc/butterfly.cpp` (file), `csrc/cpu/`, `csrc/cuda/`, `csrc/hadamard/`, `csrc/diag_mult/`, `csrc/version.cpp` all present. `MANIFEST.in` present (`recursive-include csrc *.cpp *.cu *.h *.cuh`). `setup.py:48` `extensions_dir.glob("*.cpp")` retained (compiles `butterfly.cpp` + `version.cpp` on `FORCE_CUDA=1`). `setup.py:88` and `setup.py:101` retain `_hadamard_cuda` and `_diag_mult_cuda` pybind builds. The ROADMAP SC#3 wording "csrc/{butterfly,hadamard,diag_mult,cpu,cuda}/" implies five subdirectories; the actual layout has `csrc/butterfly.cpp` (file) instead of `csrc/butterfly/` (directory). The retention intent (butterfly extension build path preserved) is satisfied. |
| 11 | **SC#4 / DEPR-05 / D-76 / D-76a:** README has new `## Deprecation timeline` section after `## Triton backend (v1.2+)` | VERIFIED | README section order: line 84 `## Triton backend (v1.2+)`, line 159 `## Deprecation timeline`, line 184 `## Tests`. Section content lists v1.2 / v1.3 / v1.4+ + migration paragraph. |
| 12 | **SC#4 / DEPR-05 / D-76b:** CHANGELOG `[1.2.0]` entry extended with three Phase 10 bullets | VERIFIED | `CHANGELOG.md:38-40` (Added: DeprecationWarning), `CHANGELOG.md:68-70` (Removed: `_flashmm`), `CHANGELOG.md:57-60` (Deprecated: TORCH_STRUCTURED_BACKEND=cuda). All three end with "Phase 10." marker. Removed-subsection order: `csrc/butterfly.cpp` retention bullet at line 64 (FIRST), `_flashmm` removal bullet at line 68 (SECOND) — matches plan order. |
| 13 | **D-75 / D-75a / D-75b:** `tests/test_deprecation.py` has three tests, each marked `@pytest.mark.op('butterfly_multiply')` | VERIFIED (file structure) | The file at `tests/test_deprecation.py` defines `test_cuda_backend_emits_deprecation_warning`, `test_cuda_backend_warning_fires_only_once`, `test_has_cuda_legacy_probe_does_not_emit_warning`, each preceded by `@pytest.mark.op('butterfly_multiply')`. Subprocess pattern used for D-75/D-75a; in-process `catch_warnings(record=True)` for D-75b. **Note:** the D-75b test FAILS at runtime — see Truth #5 (the test is correctly written; the implementation under test is broken). |

**Score:** 12/13 truths verified (1 failed — D-74b probe silencing). The truths break down as 11 explicit `must_haves` truths from the PLAN frontmatter, plus 2 explicit ROADMAP success criteria (SC#1 default-Triton resolver, SC#4 README + CHANGELOG). The 11/11 figure used in the frontmatter `score` field counts the PLAN must_haves directly; an additional truth (D-75b test pass) is the explicit gate that fails empirically. Re-numbered for the report.

**Phase score reflects:** 10 PLAN must_haves VERIFIED, 1 PLAN must_have FAILED.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `torch_structured/_cuda_legacy/__init__.py` | DeprecationWarning at module top (`simplefilter("once")` + `warnings.warn(..., stacklevel=2)`) | VERIFIED | 38 lines. Contains `warnings.simplefilter("once", DeprecationWarning)` at line 23 + warning text lines 25-32. Three existing thin re-exports preserved at lines 34-36. `__all__` at line 38 unchanged. |
| `torch_structured/_ops.py` | `_has_cuda_legacy_diag_mult` + `_has_cuda_legacy_hadamard` wrapped in `warnings.catch_warnings()` | EXISTS BUT BROKEN | Lines 133-134 (`_has_cuda_legacy_diag_mult`) and lines 154-155 (`_has_cuda_legacy_hadamard`) both have the `catch_warnings()` + `simplefilter("ignore", DeprecationWarning)` wrap. `import warnings` at line 48 confirmed. **But the wrap does not function** — see "Truth #5" + "Gaps" section. |
| `torch_structured/__init__.py` | Docstring `_flashmm` reference removed | VERIFIED | Lines 14-17 read `... try-import their own CUDA modules (\`\`_hadamard_cuda\`\` and \`\`_diag_mult_cuda\`\`) when needed.` Zero `_flashmm` tokens. |
| `setup.py` | Flashmm build block deleted; butterfly/diag_mult/hadamard builds preserved | VERIFIED | `grep -c -i flashmm setup.py` = 0. Hadamard pybind build at line 88, diag_mult pybind build at line 101, torch.ops glob at line 48 all preserved. |
| `tests/test_deprecation.py` | Three tests marked `@pytest.mark.op('butterfly_multiply')` | EXISTS (1 test FAILS in practice) | File present with 153 lines, three test functions, three marker decorators. Subprocess pattern used in tests 1+2; in-process capture in test 3. `pytest --collect-only` lists 3 tests. **Test 3 FAILS** on this host (and would fail on any host) — see Gaps. |
| `README.md` | `## Deprecation timeline` section after `## Triton backend (v1.2+)`; zero flashmm tokens | VERIFIED | `grep -c '^## Deprecation timeline' README.md` = 1. Section located at line 159, after `## Triton backend (v1.2+)` (line 84) and before `## Tests` (line 184). `grep -c 'flashmm\|_flashmm' README.md` = 0. |
| `CHANGELOG.md` | `[1.2.0]` extended with three Phase 10 bullets | VERIFIED | All three Phase 10 markers present in Added (line 38-40), Deprecated (line 57-60), Removed (line 68-70). Removed-subsection order: csrc retention bullet FIRST (line 64), `_flashmm` bullet SECOND (line 68). |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `_ops.py:_resolve` cuda branch | `_cuda_legacy/__init__.py` module body | `from torch_structured._cuda_legacy import butterfly_multiply` | WIRED | `grep -n "from torch_structured._cuda_legacy" _ops.py` matches multiple call sites in `_resolve()` and per-op probes. The import triggers module body which fires `warnings.warn`. |
| `_ops.py:_has_cuda_legacy_diag_mult/hadamard` probes | `_cuda_legacy/__init__.py` warning suppression | `warnings.catch_warnings()` + `simplefilter('ignore', ...)` wrap | PARTIAL — wrap exists but does not suppress | The wrap is structurally present at `_ops.py:133-134` and `_ops.py:154-155`. The behavioral test `test_has_cuda_legacy_probe_does_not_emit_warning` FAILS — see "Gaps" section. |
| `test_cuda_backend_emits_deprecation_warning` | subprocess `python -W always::DeprecationWarning -c '...'` | `subprocess.run(...)` for fresh warnings registry | WIRED | `tests/test_deprecation.py:48` calls `subprocess.run([sys.executable, "-W", "always::DeprecationWarning", "-c", ...])`. Pattern is correct. (Cannot execute on this CPU-only host: test skips per `@pytest.mark.op('butterfly_multiply')` gate.) |
| `test_has_cuda_legacy_probe_does_not_emit_warning` | `_ops._has_cuda_legacy_for_op('butterfly_multiply')` | In-process `catch_warnings(record=True)` capture | WIRED but ASSERTION FAILS | Test is correctly written. Runs on this host (does not need `_butterfly.so` for the probe path — the probe internally tries to import and the test asserts zero user-facing warnings escape). FAILS with `AssertionError: D-74b broken: probe emitted 1 user-facing DeprecationWarning(s); expected 0.` |
| README `## Deprecation timeline` | `## Triton backend (v1.2+)` section above it | Natural reading flow + section placement | WIRED | README line 159 (Deprecation timeline) follows line 84-158 (Triton backend section). The deprecation timeline references the Triton backend section via `[\"Triton backend\"](#triton-backend-v12)` link. |
| CHANGELOG `[1.2.0]` Added subsection | `DeprecationWarning on TORCH_STRUCTURED_BACKEND=cuda...` | In-place bullet append (D-76b) | WIRED | CHANGELOG.md:38-40 is the new Added bullet; matches plan wording. Phase 10 marker present. |

### Data-Flow Trace (Level 4)

Phase 10 produces no rendered/dynamic data; the deliverables are static text edits (warnings, docstrings, README, CHANGELOG) plus file deletions. Level 4 data-flow trace is not applicable to this phase's surface. The closest analog (the DeprecationWarning text "flowing" to stderr) is verified at the source level: the `warnings.warn(...)` call in `_cuda_legacy/__init__.py` is the data source, and `pytest.warns(...)` / subprocess stderr capture is the consumer.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `torch_structured` imports cleanly | `python -c "import torch_structured"` | Imports OK (with cosmetic UserWarning about CUDA version mismatch unrelated to Phase 10) | PASS |
| `from torch_structured.monarch.flash_mm import flash_mm` raises `ModuleNotFoundError` | `python -c "from torch_structured.monarch.flash_mm import flash_mm"` | `ModuleNotFoundError: No module named 'torch_structured.monarch.flash_mm'` | PASS |
| `csrc/flashmm/` absent | `ls -d csrc/flashmm` | `No such file or directory` | PASS |
| `csrc/{cpu,cuda,hadamard,diag_mult}/` + `csrc/butterfly.cpp` + `csrc/version.cpp` present | `ls csrc/` | All present | PASS |
| `MANIFEST.in` present | `ls MANIFEST.in` | Present | PASS |
| `find . -name '*flashmm*' -not -path './.git/*' -not -path './.planning/*' ...` | empty | empty | PASS |
| `find . -name '*flash_mm*' -not -path ...` | empty | empty | PASS |
| `_has_cuda_legacy_diag_mult()` probe does not emit user-facing DeprecationWarning | `python -c "import warnings; ... torch_structured._ops._has_cuda_legacy_diag_mult(); ..."` | **One DeprecationWarning emitted** (`'CUDA C++ backend...'`) — **D-74b broken in practice** | **FAIL** |
| `_has_cuda_legacy()` (butterfly probe) does not emit user-facing DeprecationWarning | Same as above, but for butterfly | Zero warnings — `_has_cuda_legacy` uses `torch.ops.torch_structured` directly, never imports `_cuda_legacy/__init__.py` | PASS (as designed) |
| `_has_cuda_legacy_hadamard()` probe does not emit (after diag_mult call cached `_cuda_legacy`) | Same as above, but for hadamard, after diag_mult call | Zero warnings (because `_cuda_legacy` already in `sys.modules` from prior diag_mult probe; module body does not re-execute) | PASS (incidental — only because diag_mult fires first) |
| `pytest tests/test_deprecation.py` collects 3 tests | `python -m pytest tests/test_deprecation.py --collect-only -q` | 3 tests collected | PASS |
| `pytest tests/test_deprecation.py` outcomes | `python -m pytest tests/test_deprecation.py -v` | 1 FAILED (probe-silence), 2 SKIPPED (subprocess tests skip per `@pytest.mark.op('butterfly_multiply')` because `_butterfly.so` not built on this CPU-only host) | **FAIL** (D-74b test fails as detailed above) |

### Probe Execution

No conventional `scripts/*/tests/probe-*.sh` exist in this project (verified by `find scripts -path '*/tests/probe-*.sh' -type f` — empty). The phase plan does not declare external probe scripts. The pytest-level tests in `tests/test_deprecation.py` ARE the phase's probe — and one of them FAILS.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| DEPR-01 | 10-01 | v1.2 ships with Triton as the default backend | SATISFIED | `_ops.py:_resolve()` unchanged from Phase 4 D-08 + Phase 9 §0 fix; resolver behavior preserved. |
| DEPR-02 | 10-01 | Importing the CUDA path emits `DeprecationWarning` | SATISFIED (with caveat) | The user-facing path (`set_backend('cuda')` → import `_cuda_legacy`) DOES emit the verbatim warning per inspection of `_cuda_legacy/__init__.py:25-32`. **However**, the silent-probe path (via `_has_cuda_legacy_diag_mult`) ALSO emits the warning despite the wrap — this is the D-74b gap (see truth #5). The user-visible contract of DEPR-02 (warning fires when CUDA backend selected) is met; the load-bearing implementation detail (warning does NOT fire during background fixture probes) is broken. |
| DEPR-03 | 10-01 | `setup.py`, `MANIFEST.in`, `csrc/` remain in-tree | SATISFIED | All retention checks pass. |
| DEPR-04 | 10-01 | `_flashmm` explicitly removed in v1.2 | SATISFIED | All 10 files deleted; ModuleNotFoundError raised cleanly; setup.py / `__init__.py` / README all stripped. |
| DEPR-05 | 10-01 | README and CHANGELOG document deprecation timeline | SATISFIED | README `## Deprecation timeline` section present; CHANGELOG `[1.2.0]` extended with three Phase 10 bullets. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `_ops.py` | 133-139 | `with warnings.catch_warnings(): warnings.simplefilter("ignore", DeprecationWarning): ... import ...` — the wrap LOOKS correct but is functionally defeated by the inner module's own `simplefilter("once", ...)` which prepends to `warnings.filters` | Blocker | Probe-silencing fails; Phase 9 backend fixture would spam users with the DeprecationWarning every test run if `_cuda_legacy.diag_mult` were imported in isolation before `_cuda_legacy.butterfly`. The contract advertised by D-74b is not honored. |
| `tests/test_deprecation.py` | n/a | TODO/FIXME/XXX markers in phase-modified files | None | No debt markers found in any phase-modified file (grep `TBD\|FIXME\|XXX` over modified files returns nothing). |
| `_cuda_legacy/__init__.py` | 23 | `warnings.simplefilter("once", DeprecationWarning)` at module top — globally mutates `warnings.filters` | Info (T-10-02 in plan threat model — accepted per Phase 4 D-15 contract) | This is the canonical Python pattern endorsed by RESEARCH.md "Don't Hand-Roll" table. The side effect of prepending the filter is the source of the D-74b gap above. Acceptable as a deprecation pattern; problematic as a probe-shadow. |

### Human Verification Required

None for Phase 10 deliverables. The D-74b failure is programmatically observable (the test that gates it FAILS). No visual / UX / external-service concerns.

### Gaps Summary

**One blocker gap: D-74b probe silencing is broken.** The plan explicitly classified D-74b as "LOAD-BEARING" multiple times (CONTEXT.md `<decisions>` D-74b; PLAN must_haves; PLAN Task 2 `<action>` block: "WITHOUT this fix, the Phase 9 conftest backend fixture ... would emit the user-facing DeprecationWarning during fixture collection — D-74's once-per-process gate means the warning would fire on the first probe of any test run"; PLAN Task 2 `<done>`; PLAN Task 5 `<behavior>` for D-75b; SUMMARY `key-decisions` #2). The intent of Task 2 was that `_has_cuda_legacy_for_op` calls produce zero user-facing DeprecationWarnings. The implementation does not achieve this.

**Root cause (Python warnings-module semantics):**

```
Probe call sequence:
1. Probe enters `with warnings.catch_warnings():` — saves filter state.
2. Probe calls `warnings.simplefilter("ignore", DeprecationWarning)` — prepends `('ignore', DeprecationWarning, ...)` to `warnings.filters`.
3. Probe executes `from torch_structured._cuda_legacy.diag_mult import HAS_CUDA_LEGACY_DIAG_MULT`.
4. Python's import system loads `_cuda_legacy/__init__.py` first (parent package init).
5. Inside `_cuda_legacy/__init__.py:23`, `warnings.simplefilter("once", DeprecationWarning)` prepends `('once', DeprecationWarning, ...)` to `warnings.filters`.
6. Now `warnings.filters` looks like: [('once', DeprecationWarning, ...), ('ignore', DeprecationWarning, ...), ...].
7. Inside `_cuda_legacy/__init__.py:25-32`, `warnings.warn(...)` looks up filters in order. The FIRST match wins. The 'once' filter matches first → warning fires under 'once' semantics, NOT 'ignore'.
8. The outer `catch_warnings()` exits and restores the pre-probe filter state, but the warning has already been emitted to whatever observer (test, fixture, user terminal).
```

**Why the SUMMARY's self-check missed this:** the SUMMARY's `Self-Check` section claims `_ops.py` is "FOUND (`import warnings` present; `catch_warnings` count = 4, all in the two probe functions and their docstrings)" — a structural check (do the strings exist) but NOT a behavioral check (does the wrap actually suppress the warning when the probe runs). The SUMMARY's "Deviations" section also did not flag this — Deviation 3 noted that `pytest tests/test_deprecation.py --collect-only` fails on the dev host due to missing `_butterfly.so`, but the test that fails (D-75b probe-silence) does NOT need `_butterfly.so` — it runs on any host and would have caught the bug at SUMMARY time.

**Suggested fix paths (for `/gsd-plan-phase --gaps`):**

1. **Direct submodule import** — bypass `_cuda_legacy/__init__.py` entirely in the probes:
   ```python
   import importlib
   with warnings.catch_warnings():
       warnings.simplefilter("ignore", DeprecationWarning)
       try:
           mod = importlib.import_module("torch_structured._cuda_legacy.diag_mult")
           return mod.HAS_CUDA_LEGACY_DIAG_MULT
       except ImportError:
           return False
   ```
   But this still triggers parent-package init. Python doesn't allow importing a submodule without running the parent's `__init__.py`.

2. **Pre-import `_cuda_legacy` under a guard** — at first-probe time, set a thread-local flag that `_cuda_legacy/__init__.py` consults before calling `warnings.warn`:
   ```python
   # in _cuda_legacy/__init__.py:
   import os
   if not os.environ.get("_TORCH_STRUCTURED_SILENT_PROBE"):
       warnings.simplefilter("once", DeprecationWarning)
       warnings.warn(...)
   ```
   And the probe sets/unsets the env var before/after import. Ugly but works.

3. **Replace `simplefilter("once")` with a manual once-gate** — use a module-level `_WARNING_FIRED` flag in `_cuda_legacy/__init__.py` that's checked before calling `warnings.warn`, and let the probe simply `warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch_structured._cuda_legacy.*")` BEFORE the import. This avoids the `simplefilter` global-state collision because `filterwarnings` (no `simple`) places a more specific filter that can be located by the probe.

4. **Restructure: make `_cuda_legacy/__init__.py` warning emission a function, not module body** — emit the warning lazily on first call to a `_signal_deprecation()` function, called from `_ops.py:_resolve()` only on the explicit cuda path. The probe imports submodules directly without triggering the warning. This is the cleanest separation but requires more code surface change.

The fix is a Python-semantics puzzle, not a clearly-trivial typo — it warrants a follow-up plan rather than an inline override. The override frontmatter mechanism is NOT appropriate here because (a) the intent was clearly stated and load-bearing (b) the test that gates it FAILS (c) the user-facing impact (Phase 9 backend fixture spam) is real.

### Inherited Invariant Regression Check

| Invariant | Source | Status |
|-----------|--------|--------|
| Phase 7+8 Triton kernel code unchanged | `torch_structured/_triton/{butterfly,diag_mult,hadamard_transform}/` | PRESENT (untouched per git diff) |
| Phase 9 §0 LANDMINE delegator at `torch_structured/butterfly/multiply.py:72-77` | `torch_structured/butterfly/multiply.py` | PRESENT and unchanged. `butterfly_multiply` is a delegator that re-reads `torch_structured._ops.butterfly_multiply` on every call. |
| Phase 9 `set_deterministic()` API | `_ops.py:473` | PRESENT and unchanged. |
| Phase 9 runtime selector + `_routing.json` | `torch_structured/_routing.json` | PRESENT (file exists). |
| `_torch_ref/` (TRI-07) | `torch_structured/_torch_ref/{butterfly,diag_mult,hadamard}.py` | PRESENT. |
| `tests/conftest.py` Phase 9 deliverable | `tests/conftest.py` | PRESENT (not modified by Phase 10). |
| `torch_structured/monarch/__init__.py` (does NOT export flash_mm) | `torch_structured/monarch/__init__.py` | PRESENT and unchanged (15 lines, no flash_mm reference). |

All inherited invariants preserved.

### Phase 10 Commit Audit

`git log` shows the expected 7 atomic Phase 10 commits, all present:

| Commit | Subject |
|--------|---------|
| `0772e85` | feat(10-01): add DeprecationWarning to _cuda_legacy/__init__.py (D-74) |
| `c9767d4` | fix(10-01): silence DeprecationWarning during cuda-legacy probes (D-74b) |
| `1e5e399` | feat(10-01): delete _flashmm artifacts entirely (D-73 / DEPR-04) |
| `72b8cce` | chore(10-01): strip flashmm references from setup.py and __init__.py (D-73c/D-73d) |
| `d0200a5` | test(10-01): add tests/test_deprecation.py with three DEPR-02 tests (D-75/a/b) |
| `ba419c3` | docs(10-01): update README — strip flashmm refs, add Deprecation timeline (D-76) |
| `922d066` | docs(10-01): extend CHANGELOG [1.2.0] with three Phase 10 bullets (D-76b) |

Plus the post-merge orchestrator commit `8c897e1` for the worktree merge, `6fb71e7` for SUMMARY, and `2d7c5b0` for tracking update.

## Recommendation

**Status: gaps_found — phase must NOT proceed to milestone-completion before D-74b is fixed.**

The phase is 10/11 done at the file-content level, but the LOAD-BEARING D-74b implementation is broken. The plan recognized D-74b as load-bearing in three places (CONTEXT.md, PLAN.md must_haves, PLAN Task 2 `<action>` and Task 5 D-75b `<behavior>`). Shipping v1.2 with this gap means every test run on a CUDA host that has `_diag_mult.so` built would emit one stray DeprecationWarning during fixture collection — exactly the user-experience regression Task 2 was meant to prevent.

The fix is mechanical once the Python `warnings`-semantics quirk is understood (see "Gaps Summary" suggested fix paths). It should be routed back to `/gsd-plan-phase --gaps` for a focused closure plan.

---

*Verified: 2026-05-28T17:42:57Z*
*Verifier: Claude (gsd-verifier, Opus 4.7)*

---

## Re-verification (post-fix 77fffb0) — 2026-05-28T18:25:00Z

**Status:** PASSED
**Score:** 11/11 must-haves verified (was 10/11)
**Gap closed:** D-74b probe silencing — `_has_cuda_legacy_for_op` probes no longer emit user-facing DeprecationWarning
**Regressions:** none

### What changed (commit 77fffb0)

`torch_structured/_cuda_legacy/__init__.py` — `warnings.simplefilter("once", DeprecationWarning)` removed; replaced with a module-level `_WARNED = False` flag + `if not _WARNED: warnings.warn(...); _WARNED = True`.

**Why the fix works** (verified by reading the file + running the test):
- `simplefilter("once", ...)` prepended a `('once', DeprecationWarning)` filter to `warnings.filters`, shadowing the probe's outer `simplefilter("ignore", DeprecationWarning)`. The `'once'` filter matched first → warning fired under once-semantics → leaked past the probe's `catch_warnings()` block.
- The `_WARNED` flag pattern does NOT touch `warnings.filters` at all. When `warnings.warn()` runs inside the probe's `catch_warnings()` block, the FIRST matching filter is the probe's own `('ignore', DeprecationWarning)` — so the warning is suppressed and never reaches stderr / the test's record buffer.
- Once-per-process behavior is preserved by Python's `sys.modules` cache (module body executes exactly once per process anyway). The `_WARNED` flag is structurally redundant with the cache, but harmless and self-documenting.

### Truth #5 re-evaluation

| # | Truth | Prior Status | Re-verification Status | Evidence |
|---|-------|--------------|------------------------|----------|
| 5 | **LOAD-BEARING D-74b:** `_has_cuda_legacy_for_op` probes do NOT emit the user-facing DeprecationWarning | FAILED | **VERIFIED** | `pytest tests/test_deprecation.py::test_has_cuda_legacy_probe_does_not_emit_warning -v` → **1 passed** (no skip — runs in-process; no `_butterfly.so` required). Independent empirical confirmation: `python -c "...; torch_structured._ops._has_cuda_legacy_for_op('butterfly_multiply'); ..._for_op('diag_mult'); ..._for_op('hadamard_transform'); ..."` under `catch_warnings(record=True) + simplefilter('always')` produces `Probe DeprecationWarnings emitted: 0`. |

### Re-verification spot-checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| D-75b test passes (no skip) | `python -m pytest tests/test_deprecation.py::test_has_cuda_legacy_probe_does_not_emit_warning -v` | `1 passed` | PASS |
| Full deprecation test suite | `python -m pytest tests/test_deprecation.py -v` | `1 passed, 2 skipped` (the 2 skipped need `_butterfly.so` — same gate as before, NOT a regression) | PASS |
| Probe emits zero user-facing DeprecationWarnings (empirical) | `python -c "import warnings; ...; with catch_warnings(record=True) as c: simplefilter('always', DeprecationWarning); _has_cuda_legacy_for_op(<each op>); ..."` | `Probe DeprecationWarnings emitted: 0` | PASS |
| Verbatim warning text intact — 4 load-bearing tokens hit once each | `grep -c '<each token>' torch_structured/_cuda_legacy/__init__.py` | `1` for each of: `CUDA C++ backend (csrc/) is deprecated`, `default-disabled in v1.3, with full removal in v1.4+`, `Switch to TORCH_STRUCTURED_BACKEND=triton (default in v1.2)`, `See the v1.2 release notes for migration guidance.` | PASS |
| `stacklevel=2` preserved | `grep -n stacklevel torch_structured/_cuda_legacy/__init__.py` | `26: # stacklevel=2 attributes...` + `37: stacklevel=2,` | PASS |
| `simplefilter("once", ...)` removed | `grep -n 'simplefilter' torch_structured/_cuda_legacy/__init__.py` | Only the three docstring/comment mentions explaining why it was removed (lines 18-20); zero `simplefilter` CALL sites | PASS |
| `_WARNED` flag present | `grep -n '_WARNED' torch_structured/_cuda_legacy/__init__.py` | `29: _WARNED = False`, `30: if not _WARNED:`, `39: _WARNED = True` | PASS |
| Once-per-process behavior preserved (via `sys.modules` cache) | `python -c "import warnings; ...; import torch_structured._cuda_legacy; import torch_structured._cuda_legacy; ..."` under `catch_warnings(record=True) + simplefilter('always', DeprecationWarning)` | `Warnings on first+second import: 1` (correct — module body runs once via sys.modules cache; second `import` statement is a no-op) | PASS |
| `_ops.py` probe wraps unchanged | (not modified by this fix) | The `catch_warnings() + simplefilter('ignore', ...)` wraps in `_has_cuda_legacy_diag_mult` and `_has_cuda_legacy_hadamard` are unchanged from prior verification — and they NOW WORK because the inner module no longer prepends a shadowing `'once'` filter. | PASS |

### Inherited deliverables (must-haves 1, 2, 3, 4, 6, 7, 8, 9, 10, 11)

Per the re-verification protocol, items that previously PASSED get a quick regression check (existence + basic sanity). None of these were touched by the fix commit `77fffb0` (single-file edit, single function). Specifically:

- The fix changes only `torch_structured/_cuda_legacy/__init__.py` (verified via `git diff 8c897e1..77fffb0 -- torch_structured/_cuda_legacy/__init__.py` — only that file in the diff).
- Truth #3 (verbatim DeprecationWarning text) — explicitly re-verified above; all four tokens still hit once each.
- Truth #4 (once-per-process) — explicitly re-verified above; preserved by `sys.modules` cache. The implementation mechanism CHANGED (flag instead of filter), but the observable behavior is identical.
- Truths #1, 2, 6-11 — file deletions (`_flashmm`), unchanged setup.py / __init__.py / README / CHANGELOG content, csrc retention. None of these surfaces were touched by the fix.
- Truth #13 (test file structure) — unchanged; test code didn't change, only the production code under test.

No regressions detected.

### Recommendation

**PHASE 10 PASSED.** All 11 PLAN must-haves are verified. The LOAD-BEARING D-74b gate (`test_has_cuda_legacy_probe_does_not_emit_warning`) now PASSES in-process on any host. The fix is minimal (one file, one function), well-commented (the rationale for choosing `_WARNED` over `simplefilter("once")` is documented inline at lines 17-25), and preserves all other Phase 10 contracts. The Phase 9 conftest backend fixture will no longer spam users with stray DeprecationWarnings during fixture collection. Phase 10 is ready to proceed to milestone completion.

---

## PHASE VERIFICATION PASSED

*Re-verified: 2026-05-28T18:25:00Z*
*Re-verifier: Claude (gsd-verifier, Opus 4.7, 1M context)*
*Fix commit: 77fffb0 — fix(10-01): D-74b probe-silence bug — use _WARNED flag instead of simplefilter('once') in _cuda_legacy/__init__.py*
