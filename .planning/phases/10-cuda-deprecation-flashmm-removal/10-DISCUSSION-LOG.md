# Phase 10: CUDA Deprecation & flashmm Removal - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-28
**Phase:** 10-CUDA Deprecation & flashmm Removal
**Areas discussed:** flash_mm.py disposition, CUDA CI gate, PROJECT.md evolution, README section structure

---

## flash_mm.py Disposition

**Question:** How should `torch_structured/monarch/flash_mm.py` (the Python wrapper for the removed `_flashmm` extension) be handled?

| Option | Description | Selected |
|--------|-------------|----------|
| Delete entirely | Remove the file. Python's default `ModuleNotFoundError: No module named 'torch_structured.monarch.flash_mm'` is clear enough. Mirrors DEPR-04 wording. | ✓ |
| Replace with a raising stub | Keep a placeholder file containing `raise ModuleNotFoundError(...)` with a custom message. More actionable error; placeholder file remains. | |
| Keep the file, make `_require_flashmm` raise immediately | Minimal diff. Leaves dead code paths. | |

**User's choice:** Delete entirely (Recommended)
**Notes:** Captured as D-73. No tombstone per D-73a. `monarch/__init__.py` doesn't export flash_mm so no further cleanup needed there.

---

## CUDA CI Gate

**Question:** Should Phase 10 add a CI matrix entry that explicitly validates the DeprecationWarning fires under `BACKEND=cuda`?

| Option | Description | Selected |
|--------|-------------|----------|
| Add a pytest-level test only | `pytest.warns(DeprecationWarning, ...)` in existing test job. Skip-gated via Phase 9 `_has_cuda_legacy_for_op`. Cheap, deterministic. | ✓ |
| Add a separate CI matrix entry + the pytest test | Both pytest test AND `TORCH_STRUCTURED_BACKEND=cuda pytest tests/` matrix job. End-to-end coverage; requires `_butterfly.so` build in CI. | |
| Pytest test only, gated `@pytest.mark.cuda_backend` | Manual opt-in only. Pragmatic if CI lacks `_butterfly.so` build. | |

**User's choice:** Add a pytest-level test only (Recommended)
**Notes:** Captured as D-75. Three tests: (D-75) warning fires; (D-75a) only once per process; (D-75b) probe-doesn't-fire-warning composition test.

---

## PROJECT.md Evolution

**Question:** Should Phase 10 also evolve PROJECT.md to mark v1.2 complete and outline v1.3 milestone?

| Option | Description | Selected |
|--------|-------------|----------|
| No — PROJECT.md evolution is the post-phase-evolve commit pattern | Keep Phase 10 scope tight. PROJECT.md update happens via the existing post-phase-evolve commit pattern after Phase 10 verification passes. | ✓ |
| Yes — include a PROJECT.md evolution task in Plan 10-01 | One less manual step. Couples v1.2-ship and v1.3-bootstrap concerns. | |

**User's choice:** No — post-phase-evolve pattern (Recommended)
**Notes:** Captured as D-77. v1.3 milestone setup is user-initiated via `/gsd-new-milestone` after Phase 10 ships.

---

## README Section Structure

**Question:** Where does the README deprecation-timeline content live — extend the existing Phase 9 "Triton backend (v1.2+)" section, or add a separate "Deprecation timeline" section?

| Option | Description | Selected |
|--------|-------------|----------|
| Add a separate `## Deprecation timeline` section | Clean separation of "current state" vs "timeline". Section grows. | ✓ |
| Extend the existing Phase 9 section | Tighter prose. Mixes "use this now" with "this will be removed". | |

**User's choice:** Add a separate section (Recommended)
**Notes:** Captured as D-76. Section placed AFTER the existing Phase 9 section per D-76a. CHANGELOG.md `[1.2.0]` entry extended with three new bullets per D-76b.

---

## Claude's Discretion

The user selected all four "Recommended" answers. The following remain planner-flexible per the CONTEXT.md `### Claude's Discretion` block:

- Exact `pytest.warns(match=...)` regex
- `tests/test_deprecation.py` placement (top-level recommended)
- `warnings.catch_warnings()` placement in `_has_cuda_legacy_for_op` (D-74b)
- Which op(s) to exercise in the warning-fires test (butterfly_multiply recommended)
- Order of task execution (deletes first → setup.py → docstring → warning → test → README → CHANGELOG)
- README section format (bulleted list recommended)

## Deferred Ideas

- `csrc/{butterfly,hadamard,diag_mult,cpu,cuda}/` deletion (v1.4+)
- `setup.py` CUDA extension code deletion (v1.4+)
- `_cuda_legacy/` deletion (v1.4+)
- Default-disabled CUDA build (v1.3 milestone)
- PROJECT.md v1.3 milestone setup (post-phase-evolve + user-initiated)
- CI matrix entry for `BACKEND=cuda` (may revisit in v1.3)
- Tombstone stub for flash_mm.py (rejected — deletion cleaner)
- `hyena_utils.HyenaFilter` removal (out of scope; stays as v1.2 surface)
- `monarch/` subpackage deprecation (out of scope; only flash_mm.py goes)
- bf16/fp16 support (TRI-FUT-01)
- ROCm / AMD / Intel XPU validation (PLAT-01, PLAT-02)
