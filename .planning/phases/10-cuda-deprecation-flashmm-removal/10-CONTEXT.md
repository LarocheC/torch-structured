# Phase 10: CUDA Deprecation & flashmm Removal - Context

**Gathered:** 2026-05-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Ship v1.2: Triton becomes the default backend (already wired by Phase 4 D-08 + Phase 9 09-01 §0 LANDMINE fix); `TORCH_STRUCTURED_BACKEND=cuda` emits a one-time `DeprecationWarning` at import time per Phase 4 D-15 (verbatim text + `stacklevel=2` + `simplefilter("once", DeprecationWarning)` locked in `04-DEPRECATION-PLAN.md`); `_flashmm` (the opt-in MathDx kernel) is **completely removed** — `csrc/flashmm/`, `torch_structured/monarch/flash_mm.py`, and `tests/monarch/test_flash_mm.py` are deleted (D-73). `setup.py`, `MANIFEST.in`, and `csrc/{butterfly,hadamard,diag_mult,cpu,cuda}/` STAY in-tree per DEPR-03 (2-release deprecation cadence — they're removed in v1.4+ per the timeline). README gains a separate `## Deprecation timeline` section (D-76) documenting the v1.2/v1.3/v1.4+ cadence; CHANGELOG.md (Phase 9 09-03 created) gains a v1.2 release-notes entry for the deprecation deltas.

**One plan — D-77** (per ROADMAP "Plans: 1 plan"). Phase 10 is a small focused phase: ~7 tasks (DeprecationWarning + 3 deletes + setup.py edit + docstring edit + test + README section + CHANGELOG update). No new kernel work, no new Triton primitives.

**In scope:**
- **DeprecationWarning (DEPR-02, D-74):** Add the verbatim warning text from `04-DEPRECATION-PLAN.md` to `torch_structured/_cuda_legacy/__init__.py` at module top-level:
  ```python
  import warnings
  warnings.simplefilter("once", DeprecationWarning)
  warnings.warn(
      "torch_structured: the CUDA C++ backend (csrc/) is deprecated and will be "
      "default-disabled in v1.3, with full removal in v1.4+. "
      "Switch to TORCH_STRUCTURED_BACKEND=triton (default in v1.2). "
      "See the v1.2 release notes for migration guidance.",
      DeprecationWarning,
      stacklevel=2,
  )
  ```
  Placed AT THE TOP of `_cuda_legacy/__init__.py` (before the `from .butterfly import butterfly_multiply` lines) so the warning fires on first import of the module — which only happens when `_ops.py` does `from torch_structured._cuda_legacy import ...` under `BACKEND=cuda` (per Phase 5 D-22 asymmetric fallback). Fires ONCE per process (the `simplefilter("once", DeprecationWarning)` install at module import is the once-gate).

- **`_flashmm` removal (DEPR-04, D-73):**
  - **Delete:** `csrc/flashmm/` directory (all 8 files: `README.md`, `fetch_kernel_sources.py`, `flash_mm.cpp`, `hyena_filter_cuda.cu`, `lut_code_gen.py`, `map.h`, `static_switch.h`, `twiddle.cuh`).
  - **Delete:** `torch_structured/monarch/flash_mm.py` (the Python wrapper — after deletion, `from torch_structured.monarch.flash_mm import ...` raises Python's default `ModuleNotFoundError: No module named 'torch_structured.monarch.flash_mm'` — clear enough per D-73a).
  - **Delete:** `tests/monarch/test_flash_mm.py` (no other tests reference flashmm — grep verified).
  - **Edit `setup.py`:** Remove the flashmm conditional build block (line 77 docstring mention + lines 111-129 the actual build logic). Keep all other `csrc/` extension builds (butterfly, diag_mult, hadamard).
  - **Edit `torch_structured/__init__.py:17`:** Remove `_flashmm` from the docstring list `(_diag_mult_cuda, _flashmm)` → `(_diag_mult_cuda)`. The rest of monarch/ subpackage (`blockdiag_butterfly_*.py`, `blockdiag_multiply.py`, `butterfly_factor.py`, `hyena_utils.py`, `low_rank.py`, `structured_linear.py`, `_optim_module.py`) STAYS — these don't import flash_mm.

- **`csrc/` + `setup.py` + `MANIFEST.in` retention (DEPR-03):** Verbatim. `csrc/{butterfly,hadamard,diag_mult,cpu,cuda}/` stays. `setup.py` keeps the butterfly/diag_mult/hadamard extension builds (only removes the flashmm block). `MANIFEST.in` stays as-is. `uv pip install .` with `FORCE_CUDA=1` still compiles the remaining `.so` files. SC#3 verified by a test that imports `torch_structured` after `pip install . --no-deps` (or equivalent CI smoke).

- **CI/test gate for DeprecationWarning (D-75):** New `tests/test_deprecation.py::test_cuda_backend_emits_deprecation_warning`:
  ```python
  def test_cuda_backend_emits_deprecation_warning():
      """DEPR-02: BACKEND=cuda fires a single DeprecationWarning at import time."""
      # Reset the simplefilter cache so the once-per-process gate doesn't block subsequent tests
      with pytest.warns(DeprecationWarning, match="CUDA C\\+\\+ backend.*deprecated.*default-disabled in v1\\.3"):
          torch_structured._ops.set_backend('cuda')  # triggers from torch_structured._cuda_legacy import ...
  ```
  Marked with `@pytest.mark.op('butterfly_multiply')` so the Phase 9 per-op cuda skip-gate fires when `_butterfly.so` is missing. Runs in the existing CI test job — no new matrix entry per D-75 (option 1 selected).

- **README `## Deprecation timeline` section (DEPR-05, D-76):** Add a NEW section (NOT extend the existing Phase 9 "Triton backend (v1.2+)" section per D-76 — clean separation). Document:
  - **v1.2 (current):** Triton is the default backend. `TORCH_STRUCTURED_BACKEND=cuda` still works but emits `DeprecationWarning`. `_flashmm` removed.
  - **v1.3 (next milestone, ~6 months out):** CUDA build default-disabled. `csrc/` extensions still buildable via `FORCE_CUDA=1` env var, but the wheel released to PyPI does NOT include them.
  - **v1.4+ (post-milestone, TRI-FUT-04):** `csrc/` tree, `setup.py` CUDA extension code, and `_cuda_legacy/` deleted.

- **CHANGELOG.md update (DEPR-05, D-76):** Phase 9 09-03 created CHANGELOG.md in Keep a Changelog v1.1 format with a `[1.2.0]` entry. Phase 10 EXTENDS that `[1.2.0]` entry with:
  - **Added:** DeprecationWarning on `BACKEND=cuda` import path
  - **Removed:** `_flashmm` MathDx kernel (`csrc/flashmm/`, `torch_structured/monarch/flash_mm.py`, `tests/monarch/test_flash_mm.py`)
  - **Deprecated:** `TORCH_STRUCTURED_BACKEND=cuda` (will be default-disabled in v1.3; removed in v1.4+; see README "Deprecation timeline")

**Out of scope:**
- **`csrc/{butterfly,hadamard,diag_mult,cpu,cuda}/` deletion** — DEPR-03 explicit; deferred to v1.4+ per TRI-FUT-04.
- **`setup.py` C++ extension code deletion** — same. Phase 10 ONLY removes the flashmm block; the butterfly/diag_mult/hadamard extension builds stay.
- **`MANIFEST.in` cleanup** — verbatim; stays as-is (per DEPR-03).
- **Default-disabled CUDA build in v1.2** — explicitly DEPR-03: `csrc/` is still built when `FORCE_CUDA=1` is set. v1.3 changes the default.
- **`_cuda_legacy/` deletion** — DEPR-03 implicit; deferred to v1.4+ when the underlying `csrc/` goes.
- **`monarch/` subpackage deletion** — only flash_mm.py goes. The blockdiag-butterfly primitives (blockdiag_butterfly_multiply.py, structured_linear.py, butterfly_factor.py, low_rank.py, hyena_utils.py, blockdiag_multiply.py, blockdiag_butterfly_einsum.py, blockdiag_butterfly_projection.py, blockdiag_linear.py, _optim_module.py) STAY. They are v1.2 surface and continue to work with the (now Triton-backed) butterfly_multiply.
- **PROJECT.md evolution** — per D-78 (option 1), happens via the post-phase-evolve commit pattern AFTER Phase 10 verification passes. NOT included as a task in Plan 10-01.
- **v1.3 milestone setup** — out of v1.2 scope. User initiates via `/gsd-new-milestone` after Phase 10 ships.
- **CI matrix entry for `BACKEND=cuda`** — D-75 (option 1 selected). Pytest-level test only; no new matrix entry.
- **Touching `_triton/` kernels** — Phases 5-8 deliverables stay untouched.
- **Touching `_torch_ref/`** — TRI-07 preserved.
- **`set_backend('triton')` semantics change** — already correct after Phase 9 09-01 §0 LANDMINE fix; nothing to change.
- **`set_deterministic()` API changes** — Phase 9 09-02 deliverable; stays.
- **`_routing.json` or runtime selector changes** — Phase 9 09-03 deliverable; stays.
- **`hyena_utils.HyenaFilter`** — does NOT import `flash_mm.py`; keeps working.

</domain>

<decisions>
## Implementation Decisions

### `_flashmm` removal — delete entirely (User choice, locked)

- **D-73:** **Delete `torch_structured/monarch/flash_mm.py`, `csrc/flashmm/`, and `tests/monarch/test_flash_mm.py` entirely.** Do NOT replace with a raising stub. After deletion, `from torch_structured.monarch.flash_mm import ...` raises Python's default `ModuleNotFoundError: No module named 'torch_structured.monarch.flash_mm'` — clear enough per the SC#2 wording "any references to it raise ModuleNotFoundError with a clear message". The default Python error message identifies the missing module by name, which directly tells the user what's gone. Matches DEPR-04 wording "explicitly removed in v1.2 — not ported, not maintained".
- **D-73a:** **No tombstone file.** Rationale: a tombstone stub adds Python noise + leaves a dead file in the package; the deletion is the cleanest signal. Users encountering the `ModuleNotFoundError` look at the README "Deprecation timeline" section (D-76) for migration guidance.
- **D-73b:** **`torch_structured/monarch/__init__.py` UNCHANGED.** It doesn't export `flash_mm` or any of its symbols (verified via grep). The rest of the `monarch/` subpackage (blockdiag_butterfly_multiply, structured_linear, butterfly_factor, etc.) is independent and stays.
- **D-73c:** **`torch_structured/__init__.py:17` docstring edit.** Remove `_flashmm` from `(_diag_mult_cuda, _flashmm)` → `(_diag_mult_cuda)`. Single-line edit.
- **D-73d:** **`setup.py` flashmm build block removal.** Delete lines 77 (docstring mention) and lines 111-129 (the conditional `flashmm_dir = Path(...)` block). Keep all other csrc/ extension builds (butterfly, diag_mult, hadamard). The grep-test for SC#3: `grep -c 'flashmm' setup.py` returns 0 after the edit.

### DeprecationWarning — Phase 4 D-15 verbatim (Inherited, locked upstream)

- **D-74 (inherits Phase 4 D-15 verbatim):** Implement the exact incantation from `.planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/04-DEPRECATION-PLAN.md`:
  - **Location:** `torch_structured/_cuda_legacy/__init__.py` at module top-level (BEFORE the `from .butterfly import butterfly_multiply` lines).
  - **Mechanism:** `warnings.simplefilter("once", DeprecationWarning)` + `warnings.warn(..., DeprecationWarning, stacklevel=2)`.
  - **Text (verbatim):** "torch_structured: the CUDA C++ backend (csrc/) is deprecated and will be default-disabled in v1.3, with full removal in v1.4+. Switch to TORCH_STRUCTURED_BACKEND=triton (default in v1.2). See the v1.2 release notes for migration guidance."
  - **stacklevel=2:** attribution shifts from `_cuda_legacy/__init__.py` (Python guts the user doesn't control) to `_ops.py`'s `from torch_structured._cuda_legacy import ...` line (the legitimate observation point on the dispatch path).
  - **simplefilter("once"):** fires once per process via Python's `warnings` registry. Subsequent imports of `_cuda_legacy` in the same process do NOT re-fire.
- **D-74a:** **The warning ONLY fires under explicit `BACKEND=cuda`.** Under `BACKEND=triton` (default) or `BACKEND=torch`, `_cuda_legacy` is NOT imported (per Phase 5 D-22 asymmetric fallback), so the warning stays silent. SC#1 verified by setting `BACKEND=triton` (no env var or explicit set) and asserting no warning.
- **D-74b:** **Composition with Phase 9 `_has_cuda_legacy_for_op`:** Phase 9's per-op probe imports `_cuda_legacy.butterfly` (or _cuda_legacy.diag_mult, _cuda_legacy.hadamard) directly to check if the .so loaded successfully. This is a probe — does it count as "importing the CUDA path" and trigger the warning? Per Phase 4 D-15: yes, ANY import of `_cuda_legacy/__init__.py` triggers. The probe currently does `from torch_structured._cuda_legacy.butterfly import ...` which DOES execute `_cuda_legacy/__init__.py`'s module body. **Resolution:** the probe is run silently — wrap the probe in `warnings.catch_warnings()` + `warnings.simplefilter("ignore", DeprecationWarning)` so the probe doesn't fire the user-facing warning. The warning fires ONLY when `set_backend('cuda')` actually selects the cuda path (not during the silent probe).

### CI gate for DeprecationWarning — pytest-level test only (User choice, locked)

- **D-75:** **Add `tests/test_deprecation.py::test_cuda_backend_emits_deprecation_warning`** using `pytest.warns(DeprecationWarning, match=...)`. Runs in the existing CI test job; no new matrix entry. Skip-gates on `_has_cuda_legacy_for_op('butterfly_multiply')` via Phase 9 D-62 marker pattern.
- **D-75a:** **Once-per-process gate test.** Add a SECOND test `test_cuda_backend_warning_fires_only_once` that asserts the warning fires exactly ONCE across multiple `set_backend('cuda')` calls in the same process. Use `warnings.catch_warnings(record=True)` to capture and count.
- **D-75b:** **Probe-doesn't-fire-warning test (D-74b coverage).** Add a THIRD test `test_has_cuda_legacy_probe_does_not_emit_warning` that asserts the Phase 9 `_has_cuda_legacy_for_op` probe does NOT fire the user-facing warning (because it's wrapped in `warnings.catch_warnings()` + `simplefilter("ignore", DeprecationWarning)` per D-74b). This is the load-bearing test for the composition with Phase 9's probe surface — without this gate, the warning would spam users every time the backend fixture probes cuda availability.
- **D-75c:** **Test placement.** New file `tests/test_deprecation.py` (NOT extending an existing test file — clean separation; the deprecation concern is orthogonal to the existing per-op test surfaces). Marked `@pytest.mark.op('butterfly_multiply')` per Phase 9 D-62 so it skips when `_butterfly.so` is missing.

### PROJECT.md evolution — post-phase pattern (User choice, locked)

- **D-77:** **Phase 10 does NOT include a PROJECT.md evolution task.** PROJECT.md update to mark v1.2 milestone Complete + add v1.3 placeholder happens AFTER Phase 10 verification passes via the existing `docs(phase-XX): evolve PROJECT.md after phase completion` commit pattern (visible in prior phases — see git log for commit `b40c37f`). This keeps Phase 10's plan tight (deprecation deliverables only) and separates the v1.2-ship concern from the v1.3-bootstrap concern (which the user initiates via `/gsd-new-milestone`).

### README — separate `## Deprecation timeline` section (User choice, locked)

- **D-76:** **Add a NEW `## Deprecation timeline` section to README.md**, separate from the existing Phase 9 "## Triton backend (v1.2+)" section. Document three timeline phases:
  - **v1.2 (current):** Triton is the default backend. `TORCH_STRUCTURED_BACKEND=cuda` still works but emits a one-time `DeprecationWarning` at import time. `_flashmm` removed.
  - **v1.3 (next milestone, ~6 months out):** CUDA build default-disabled. `csrc/` extensions still buildable via `FORCE_CUDA=1`, but the wheel released to PyPI does NOT include them.
  - **v1.4+ (post-milestone, TRI-FUT-04):** `csrc/` tree, `setup.py` CUDA extension code, and `_cuda_legacy/` deleted.
- **D-76a:** **Placement:** AFTER the existing `## Triton backend (v1.2+)` section (which Phase 9 09-03 added at README.md:93-160). Mirrors the natural reading flow "what's current → what's coming". No reordering of existing sections.
- **D-76b:** **CHANGELOG.md extension:** Phase 9 09-03 created CHANGELOG.md in Keep a Changelog v1.1 format with a `[1.2.0]` entry. Phase 10 EXTENDS that `[1.2.0]` entry with three new bullets:
  - **Added:** "DeprecationWarning on TORCH_STRUCTURED_BACKEND=cuda import path (fires once per process)"
  - **Removed:** "`_flashmm` MathDx kernel (csrc/flashmm/, torch_structured/monarch/flash_mm.py, tests/monarch/test_flash_mm.py) — see README \"Deprecation timeline\""
  - **Deprecated:** "TORCH_STRUCTURED_BACKEND=cuda (default-disabled in v1.3; removed in v1.4+)"

### Inherited from prior phases (NOT re-discussed — locked upstream)

- **D-78 (inherits Phase 4 D-08):** `auto` resolves to Triton when both Triton and CUDA are available. Already wired in `torch_structured/_ops.py:188-200`. Phase 10 does NOT touch this. SC#1 verified by `import torch_structured` (no env var set) on a CUDA+Triton-capable machine → `_BACKEND == 'triton'`.
- **D-79 (inherits Phase 5 D-22):** Per-op asymmetric fallback. `BACKEND=cuda` with `_butterfly.so` missing falls back to `_torch_ref` (NOT Triton). Phase 10 does NOT change this. The DeprecationWarning still fires when `_cuda_legacy/__init__.py` is imported, even if the underlying `.so` is missing (the warning is in `__init__.py`, fired before `from .butterfly import butterfly_multiply` is attempted — if that fails, the warning still fired).
- **D-80 (inherits Phase 9 §0 LANDMINE fix):** `Butterfly.forward` routes through `_ops.butterfly_multiply` via the D-05 attribute-access delegator at `torch_structured/butterfly/multiply.py:27-30`. Phase 10 does NOT touch the delegator. The `BACKEND=cuda` path under the delegator still hits the CUDA `.so` (per Phase 5 D-22), which triggers the DeprecationWarning per D-74.
- **D-81 (inherits Phase 9 D-62 / 09-01):** Per-op cuda skip-gate via `_has_cuda_legacy_for_op(op_name)`. Phase 10's `tests/test_deprecation.py` uses `@pytest.mark.op('butterfly_multiply')` so it skips when `_butterfly.so` is missing.
- **D-82 (inherits Phase 9 D-63):** `set_deterministic()` API. Orthogonal to deprecation. Phase 10 does NOT touch.
- **D-83 (inherits Phase 9 D-66):** Runtime selector + `_routing.json`. Orthogonal. Phase 10 does NOT touch.
- **D-84 (inherits Phase 9 09-03):** CHANGELOG.md in Keep a Changelog v1.1 format. Phase 10 EXTENDS the existing `[1.2.0]` entry per D-76b — does NOT create a new release entry (v1.2.0 is the milestone-completing version; Phase 10 is the final phase of that milestone).

### Claude's Discretion

Areas where Claude (planner / executor) has flexibility:
- Exact form of the `pytest.warns(...)` regex in `test_cuda_backend_emits_deprecation_warning`. Recommend `match=r"CUDA C\+\+ backend.*deprecated.*default-disabled in v1\.3"` — anchors on key tokens from the verbatim text without depending on exact whitespace.
- Whether `tests/test_deprecation.py` lives at the top of `tests/` (sibling to `test_butterfly_triton.py`) or under a new subdirectory. Recommend top-level for consistency with Phase 9's test files.
- Exact placement of the `warnings.catch_warnings()` wrapping in `_ops._has_cuda_legacy_for_op` (D-74b). Recommend: inside the function body, around the `from torch_structured._cuda_legacy.<op> import ...` line.
- Whether to test ALL THREE ops (butterfly_multiply, diag_mult, hadamard_transform) in `test_cuda_backend_emits_deprecation_warning` or just one. Recommend just butterfly (the load-bearing op; the warning is module-level so it fires once regardless of which op triggers the import).
- Order of edits: deletes first (flashmm) → setup.py edit → docstring edit → DeprecationWarning add → test → README → CHANGELOG. Recommend this order; deletes early reduce blast radius if a later step fails.
- Whether the README "Deprecation timeline" section uses a Markdown table or bulleted list. Recommend bulleted list (matches the v1.2/v1.3/v1.4 prose style from 04-DEPRECATION-PLAN.md).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 10 charter
- `.planning/ROADMAP.md` §"Phase 10" — phase goal, depends on Phase 9, 4 success criteria, 1 plan slot
- `.planning/REQUIREMENTS.md` §"v1.2 Requirements" → DEPR-01, DEPR-02, DEPR-03, DEPR-04, DEPR-05 (all 5 REQs this phase covers)
- `.planning/REQUIREMENTS.md` §"Traceability" — confirms DEPR-01..05 mapped to Phase 10
- `.planning/PROJECT.md` §"Current Milestone: v1.2 Triton Migration" — Phase 10 is the milestone-completing phase

### Phase 4 hand-off (LOCKED — DeprecationWarning text + mechanism)
- `.planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/04-DEPRECATION-PLAN.md` — **CRITICAL.** The exact warning text (verbatim, locked), location (`_cuda_legacy/__init__.py` top-level), mechanism (`simplefilter("once", DeprecationWarning)` + `warn(..., stacklevel=2)`), and timeline (v1.2 warn, v1.3 default-off, v1.4+ remove). Phase 10 implements this verbatim.
- `.planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/04-CONTEXT.md` — D-08 (auto → Triton when both available); D-15 (deprecation plan companion doc); D-16 (CI cache)

### Phase 5 hand-off (LOCKED — per-op asymmetric fallback)
- `.planning/phases/05-diag-mult-triton-port/05-CONTEXT.md` — D-21 (try-import + sentinel; sanctioned try/except site); D-22 (per-op asymmetric fallback — BACKEND=cuda with missing .so falls back to torch oracle, NOT Triton)

### Phase 9 hand-off (LOCKED — §0 LANDMINE fix + per-op cuda skip-gate + CHANGELOG)
- `.planning/phases/09-integration-hardening-correctness-gates/09-CONTEXT.md` — D-62 (per-op cuda skip-gate via `_has_cuda_legacy_for_op` + `@pytest.mark.op` markers). Phase 10 uses this verbatim for the test gate (D-75).
- `.planning/phases/09-integration-hardening-correctness-gates/09-01-SUMMARY.md` — §0 LANDMINE fix landed. `Butterfly.forward` routes through `_ops.butterfly_multiply` per D-05 delegator. The cuda path still works (per D-22 asymmetric fallback) → DeprecationWarning fires via D-74.
- `.planning/phases/09-integration-hardening-correctness-gates/09-03-SUMMARY.md` — CHANGELOG.md created in Keep a Changelog v1.1 format with `[1.2.0]` entry. Phase 10 EXTENDS the entry per D-76b. README "Triton backend (v1.2+)" section at README.md:93-160 — Phase 10 adds a separate `## Deprecation timeline` section AFTER it per D-76a.

### Project-level constraints
- `./CLAUDE.md` (project root) — `assert` preconditions, no try/except in core lib (one exception: `_cuda_legacy/*.py` try-imports — sanctioned per Phase 5 D-21). Phase 10's `warnings.catch_warnings()` wrapping in `_has_cuda_legacy_for_op` (D-74b) is a context-manager pattern, not a try/except — composes with CLAUDE.md.
- `/home/claroche/CLAUDE.md` (user-level) — `bd` for runtime task tracking.

### Code-level references (read before editing)
- `torch_structured/_cuda_legacy/__init__.py` — current state (3 lines + docstring). Phase 10 D-74 inserts the `warnings.simplefilter` + `warnings.warn` block AT THE TOP, BEFORE the `from .butterfly import` lines.
- `torch_structured/_ops.py:188-200` — current `auto` resolver. Phase 10 does NOT touch.
- `torch_structured/_ops.py` `_has_cuda_legacy_for_op` (Phase 9 09-01 deliverable) — Phase 10 D-74b wraps the probe import in `warnings.catch_warnings()` + `simplefilter("ignore", DeprecationWarning)` so the probe stays silent.
- `torch_structured/__init__.py:17` — docstring line `(_diag_mult_cuda, _flashmm)`. Phase 10 D-73c edits to drop `_flashmm`.
- `torch_structured/monarch/__init__.py` — does NOT export `flash_mm` (verified). Phase 10 does NOT touch.
- `torch_structured/monarch/flash_mm.py` — Phase 10 D-73 deletes.
- `csrc/flashmm/` — 8 files. Phase 10 D-73 deletes the entire directory.
- `tests/monarch/test_flash_mm.py` — Phase 10 D-73 deletes.
- `setup.py:77` (docstring mention) and `setup.py:111-129` (conditional flashmm build block) — Phase 10 D-73d removes.
- `tests/conftest.py` (Phase 9 09-01 deliverable) — already has `@pytest.mark.op` marker. Phase 10 D-75 uses verbatim.
- `README.md:93-160` — Phase 9 09-03 "Triton backend (v1.2+)" section. Phase 10 D-76a adds the NEW `## Deprecation timeline` section AFTER this section.
- `CHANGELOG.md` (Phase 9 09-03 created) — `[1.2.0]` entry. Phase 10 D-76b extends with three bullets (Added/Removed/Deprecated).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`torch_structured/_cuda_legacy/__init__.py`** — current 14-line file. Top-level docstring + 3 import lines + `__all__`. The DeprecationWarning insertion site (D-74) is BEFORE the first `from .butterfly import` line.
- **`_ops.py` per-op probe pattern** — Phase 9 D-62 / 09-01's `_has_cuda_legacy_for_op(op_name)`. Phase 10 D-74b wraps the probe import in `warnings.catch_warnings()` to keep the silent path silent.
- **`tests/conftest.py` `@pytest.mark.op` marker** — Phase 9 09-01 registered. Phase 10 D-75 uses verbatim for the deprecation test.
- **CHANGELOG.md `[1.2.0]` entry** — Phase 9 09-03 created. Phase 10 D-76b extends in-place with three new bullets.

### Established Patterns
- **`assert` preconditions / no try/except in core lib** (CLAUDE.md). Exception: `_cuda_legacy/*.py` try-imports (D-21 sanctioned). Phase 10's `warnings.catch_warnings()` is a context-manager pattern, not a try/except (composes with CLAUDE.md).
- **Verbatim implementation of locked Phase 4 deliverables** — Phase 10 D-74 transcribes the exact warning text + mechanism from `04-DEPRECATION-PLAN.md` without modification. Same pattern as Phase 7 D-44 (Phase 4 D-01..D-03 view_as_real layout) + Phase 8 D-54 (same).
- **Test file naming `tests/test_<concern>.py`** — `test_deprecation.py` for the deprecation gate. Consistent with `test_butterfly_triton.py`, `test_distributed_triton.py`, etc.
- **CHANGELOG extension in-place** — Phase 9 09-03's pattern (write new release entry as a section; Phase 10 just adds bullets under the existing `[1.2.0]` heading).

### Integration Points
- **`set_backend('cuda')` → `_ops.py:204-228` butterfly resolver → `from torch_structured._cuda_legacy import butterfly_multiply` → triggers `_cuda_legacy/__init__.py` module body → DeprecationWarning fires (D-74).** This is the load-bearing path.
- **`_has_cuda_legacy_for_op` probe → `from torch_structured._cuda_legacy.<op> import ...` → also triggers `_cuda_legacy/__init__.py` module body → DeprecationWarning would fire WITHOUT D-74b's catch_warnings wrap.** This is why D-74b is mandatory — without it, the warning spams users every time the backend fixture probes cuda availability.
- **`Butterfly.forward` → `_ops.butterfly_multiply` (Phase 9 §0 fix) → resolver dispatch → cuda path (when BACKEND=cuda) → DeprecationWarning fires once.** No changes needed at the consumer surface.
- **Phase 7+8+9 Triton-path tests** — continue passing under BACKEND=triton (no DeprecationWarning fires; `_cuda_legacy` is not imported in the Triton path).

</code_context>

<specifics>
## Specific Ideas

- **The Phase 10 implementation IS small.** Estimated ~7 atomic tasks for the single plan 10-01:
  1. Add DeprecationWarning to `_cuda_legacy/__init__.py` per D-74 verbatim
  2. Wrap probe in `warnings.catch_warnings()` per D-74b
  3. Delete `csrc/flashmm/`, `torch_structured/monarch/flash_mm.py`, `tests/monarch/test_flash_mm.py` per D-73
  4. Remove flashmm build block from `setup.py` per D-73d
  5. Edit `torch_structured/__init__.py:17` docstring per D-73c
  6. Add `tests/test_deprecation.py` with 3 tests per D-75 / D-75a / D-75b
  7. Add README `## Deprecation timeline` section per D-76 + extend CHANGELOG `[1.2.0]` entry per D-76b

- **D-74b probe-silencing illustrative pattern:**
  ```python
  # torch_structured/_ops.py
  def _has_cuda_legacy_for_op(op_name: str) -> bool:
      """Honest probe — returns True iff _<op_name>.so loaded successfully.
      Wrapped in warnings.catch_warnings() so the probe import does NOT trigger
      the user-facing DeprecationWarning from _cuda_legacy/__init__.py (D-74b).
      The warning is reserved for explicit set_backend('cuda') invocations."""
      with warnings.catch_warnings():
          warnings.simplefilter("ignore", DeprecationWarning)
          try:
              if op_name == "butterfly_multiply":
                  from torch_structured._cuda_legacy.butterfly import butterfly_multiply  # noqa: F401
              elif op_name == "diag_mult":
                  from torch_structured._cuda_legacy.diag_mult import diag_mult  # noqa: F401
              elif op_name == "hadamard_transform":
                  from torch_structured._cuda_legacy.hadamard import hadamard_transform  # noqa: F401
              else:
                  return False
              return True
          except (ImportError, AttributeError, RuntimeError):
              return False
  ```

- **D-75 test surface (3 tests in `tests/test_deprecation.py`):**
  ```python
  import pytest
  import warnings
  import torch_structured

  @pytest.mark.op('butterfly_multiply')
  def test_cuda_backend_emits_deprecation_warning(backend):
      """DEPR-02: BACKEND=cuda fires a single DeprecationWarning."""
      if backend != 'cuda':
          pytest.skip("Test only runs under BACKEND=cuda")
      # The backend fixture already called set_backend('cuda') which triggered
      # the import of _cuda_legacy. Check the warning fired in this process.
      with warnings.catch_warnings(record=True) as w:
          warnings.simplefilter("always", DeprecationWarning)
          # Force a re-import to re-trigger if simplefilter("once") suppressed.
          # Actually — once-per-process means we can't easily re-trigger. Test
          # by capturing during a sub-process invocation.
          import subprocess
          import sys
          result = subprocess.run(
              [sys.executable, "-W", "always::DeprecationWarning", "-c",
               "import warnings; warnings.simplefilter('always'); import torch_structured; torch_structured._ops.set_backend('cuda')"],
              capture_output=True, text=True
          )
          assert "CUDA C++ backend" in result.stderr
          assert "default-disabled in v1.3" in result.stderr
          # Count occurrences: must be exactly 1
          assert result.stderr.count("CUDA C++ backend") == 1, (
              f"Warning fired {result.stderr.count('CUDA C++ backend')} times; expected exactly 1"
          )

  @pytest.mark.op('butterfly_multiply')
  def test_cuda_backend_warning_fires_only_once():
      """DEPR-02 once-per-process: subsequent set_backend('cuda') calls don't re-fire."""
      # Verified via subprocess that does set_backend('cuda') twice and counts.
      ...

  @pytest.mark.op('butterfly_multiply')
  def test_has_cuda_legacy_probe_does_not_emit_warning():
      """D-74b: the probe is silenced via warnings.catch_warnings()."""
      with warnings.catch_warnings(record=True) as w:
          warnings.simplefilter("always", DeprecationWarning)
          torch_structured._ops._has_cuda_legacy_for_op("butterfly_multiply")
          deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)
                                  and "CUDA C++ backend" in str(x.message)]
          assert len(deprecation_warnings) == 0, (
              f"Probe emitted {len(deprecation_warnings)} DeprecationWarning(s); D-74b broken"
          )
  ```
  The `subprocess` pattern is needed because Python's `simplefilter("once", DeprecationWarning)` registry persists across `warnings.catch_warnings()` blocks within the same process. Each subprocess invocation gets a fresh registry. The planner picks the exact mechanism — subprocess is robust; a `warnings.resetwarnings()` + `__warningregistry__.clear()` pattern could also work but is more fragile.

- **D-76 README section template (illustrative):**
  ```markdown
  ## Deprecation timeline

  torch_structured ships Triton as the default backend in v1.2. The legacy CUDA
  C++ backend (`csrc/`) is being retired over a two-release deprecation cadence:

  - **v1.2 (current):** Triton is the default. `TORCH_STRUCTURED_BACKEND=cuda`
    still works for users who built `_butterfly.so` / `_diag_mult.so` /
    `_hadamard.so` locally, but emits a one-time `DeprecationWarning` at import
    time pointing here. `_flashmm` (the MathDx kernel) is removed entirely.
  - **v1.3 (next minor release, ~6 months out):** CUDA build is default-disabled.
    `csrc/` extensions stay in the source tree and can still be compiled via
    `FORCE_CUDA=1`, but the PyPI wheel does NOT include them. The
    `DeprecationWarning` still fires when a locally-built CUDA path is used.
  - **v1.4+ (post-milestone):** `csrc/` tree, `setup.py` CUDA extension code,
    and `_cuda_legacy/` are deleted. The standard 2-release deprecation cadence
    gives users two minor releases to migrate.

  Migration: most users should set nothing and let the Triton default take over.
  If you have a workload that needs the CUDA backend (e.g., Volta sm_70 / Turing
  sm_75 hardware that Triton doesn't fully support), see the
  ["Triton backend"](#triton-backend-v12) section above for hardware
  requirements; otherwise pin to v1.1.
  ```

- **D-76b CHANGELOG extension illustrative:**
  ```markdown
  ## [1.2.0] - 2026-05-28
  ### Added
  - Triton backend as the default for all kernels (butterfly_multiply, diag_mult, hadamard_transform). Phase 4-8.
  - `torch_structured.set_deterministic(value: bool) -> bool` opt-in API for reproducible d_twiddle accumulation. Phase 9.
  - Runtime selector routing below-60% perf cells to CUDA. Phase 9.
  - DeprecationWarning on `TORCH_STRUCTURED_BACKEND=cuda` import path (fires once per process). Phase 10.

  ### Changed
  - PyTorch minimum bumped from `>=2.0` to `>=2.6`.
  - `torch_structured.butterfly.multiply.butterfly_multiply` now delegates to `torch_structured._ops.butterfly_multiply` (the dispatch surface) — `Butterfly` nn.Modules respect `set_backend()`. Phase 9 09-01 §0 LANDMINE fix.

  ### Removed
  - `_flashmm` MathDx kernel (`csrc/flashmm/`, `torch_structured/monarch/flash_mm.py`, `tests/monarch/test_flash_mm.py`) — see README "Deprecation timeline". Phase 10.

  ### Deprecated
  - `TORCH_STRUCTURED_BACKEND=cuda` (default-disabled in v1.3; removed in v1.4+). Phase 10.
  ```

- **Plan structure transcription from Phase 8 / Phase 9:** Plan 10-01 transcribes the SUMMARY.md format from those phases. Single plan, ~7 tasks, ~5 files modified (multiply.py, _cuda_legacy/__init__.py, _ops.py, __init__.py, setup.py) + ~3 files deleted (flash_mm.py, csrc/flashmm/, test_flash_mm.py) + ~2 files extended (README.md, CHANGELOG.md) + ~1 file created (test_deprecation.py).

</specifics>

<deferred>
## Deferred Ideas

- **`csrc/{butterfly,hadamard,diag_mult,cpu,cuda}/` deletion** — DEPR-03 explicit; v1.4+ per TRI-FUT-04. Phase 10 keeps in-tree.
- **`setup.py` CUDA extension code deletion** — same. v1.4+.
- **`_cuda_legacy/` deletion** — implicit; v1.4+ when csrc/ goes.
- **Default-disabled CUDA build** — v1.3 milestone (default-disabled wheel; `FORCE_CUDA=1` still works).
- **PROJECT.md v1.3 milestone setup** — happens after Phase 10 ships (out-of-band, user-initiated via `/gsd-new-milestone`).
- **CI matrix entry for `BACKEND=cuda`** — D-75 (option 1 — pytest-level only). May revisit in v1.3 if the cuda path needs more end-to-end coverage.
- **Tombstone stub for `monarch/flash_mm.py`** — D-73a explicit; deletion is cleaner than a placeholder.
- **`hyena_utils.HyenaFilter` removal** — out of scope. HyenaFilter doesn't import flash_mm.py; it stays as part of v1.2 monarch/ surface.
- **`monarch/` subpackage deprecation** — out of scope. Only flash_mm.py goes. The blockdiag-butterfly primitives are v1.2 surface.
- **bf16/fp16 support** — TRI-FUT-01, deferred.
- **ROCm / AMD / Intel XPU validation** — PLAT-01, PLAT-02, deferred.

### Reviewed Todos (not folded)
None — no pending todos surfaced for Phase 10.

</deferred>

---

*Phase: 10-CUDA Deprecation & flashmm Removal*
*Context gathered: 2026-05-28*
