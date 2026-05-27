# Phase 4: CUDA Backend Deprecation Plan

**Written:** 2026-05-27
**Consumers:** Phase 10 (DEPR-02 implementation) — reads this verbatim
**Locks:** D-15 — and DEPR-01..05 groundwork from REQUIREMENTS.md

## Decision (D-15)

In Phase 10, the `DeprecationWarning` fires only on **explicit**
`TORCH_STRUCTURED_BACKEND=cuda` selection — NOT on the auto-resolved cuda
path. The auto path uses the heads-up `logging.info(...)` per D-08, which is
already wired in `torch_structured/_ops.py._resolve()` (dormant in Phase 4
because no Triton kernel binding can occur yet). The warning fires once per
process via `warnings.simplefilter("once", DeprecationWarning)` installed at
the `_cuda_legacy` module top-level, with `stacklevel=2` so the attribution
points at the importer (`_ops.py` doing `from torch_structured._cuda_legacy
import ...`) rather than at the warning line inside `_cuda_legacy`.

## Exact Incantation

Phase 10 implements this verbatim in
`torch_structured/_cuda_legacy/__init__.py` (additive to the Phase 4 thin
wrapper — the existing `from .butterfly import butterfly_multiply` line stays):

```python
# torch_structured/_cuda_legacy/__init__.py (Phase 10 ADDITION)
import warnings

# Module-level filter setup: "once" suppresses repeats based on
# (message, category) IGNORING module and line number. Even if multiple
# call sites import this module, the warning fires exactly once per process.
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

## Why `stacklevel=2`

With `stacklevel=1` (the default), the warning is attributed to the line
*inside* `_cuda_legacy/__init__.py` containing `warnings.warn(...)` — Python
guts the user does not control. With `stacklevel=2`, attribution shifts to the
*importer* — `_ops.py`'s `from torch_structured._cuda_legacy import
butterfly_multiply` line — which is part of `torch_structured`'s code, but is
the legitimate observation point the user would see when tracing the dispatch
path. This makes the warning actionable: the user can grep for the import in
their stack and understand the path that triggered it.

For module-top-level code (not inside a function), Python attributes
`stacklevel=2` to "whoever imported this module" — for once-per-process
imports, that is exactly the call site we want to surface.

## Why `simplefilter("once", DeprecationWarning)`

Per Python's `warnings` documentation: "A warning is considered a repeat if
the `(message, category)` are the same, ignoring the module and line number."
This is *per process*, not per session. Calling `simplefilter` at module
top-level installs the filter for the rest of the process. Phase 10 prefers
this over `warnings.filterwarnings("once", ...)` because `simplefilter`
applies globally rather than to a category-specific subset.

The filter does NOT need to be torn down — `DeprecationWarning` is the
correct category for an end-user-actionable deprecation, and applications
that want to silence it can use `python -W ignore::DeprecationWarning` or
`warnings.filterwarnings("ignore", category=DeprecationWarning,
module="torch_structured.*")` themselves.

## Why NOT a Custom `_warned` Flag

The RESEARCH.md "Don't Hand-Roll" table is explicit: custom flags miss
`warn_only=True` semantics, do not compose with pytest's warning capture
(`pytest.warns(DeprecationWarning)`), and do not integrate with the `python
-W` CLI flag mechanism. Use the stdlib pattern.

```python
# Anti-pattern — DO NOT DO THIS
_warned = False
def _maybe_warn():
    global _warned
    if not _warned:
        # ...
        _warned = True
```

## Routing in `_ops.py`

Phase 10 adds a one-line check in `torch_structured/_ops.py._resolve()`: when
`name == "cuda"` AND the user explicitly requested it (not via auto), the
import of `_cuda_legacy` triggers the module-top-level `warn`. The
`auto -> cuda` path bypasses this distinction because import order is the
same — Python's module cache means the warn fires on the FIRST import of
`_cuda_legacy` regardless of whether the request was `cuda` or `auto`.

**Open detail for Phase 10:** the once-filter still fires once for the
process, so an `auto -> cuda` import would emit the warning even though we
want it suppressed for auto. Phase 10 must either (a) gate the `warn` inside
a helper function called only on explicit `cuda` paths, or (b) accept that
auto resolutions to cuda also see the warning (acceptable per D-08's
heads-up framing — both are "the user is on cuda; tell them about it"). The
Phase 4 plan documents this as the one open implementation detail; the rest
of the incantation is locked verbatim.

Recommended Phase 10 implementation: extract the warn into
`_cuda_legacy/_warn_on_explicit.py:warn_once()` and call it from
`_ops.py._resolve()` only when the requested `name == "cuda"`. Keep the
module-top-level `simplefilter` registration in `_cuda_legacy/__init__.py`
so the dedup semantics survive.

## Timeline References

- **v1.2** (this milestone): default backend is Triton (when CUDA available
  AND Triton importable). Explicit `TORCH_STRUCTURED_BACKEND=cuda` still
  works; emits the `DeprecationWarning` once per process.
- **v1.3** (next milestone, ~6 months out): CUDA build is default-disabled.
  The csrc/ extensions are still buildable via `FORCE_CUDA=1` env var, but
  the wheel released to PyPI does NOT include them. CUDA kernels still load
  if the user rebuilt locally, so the deprecation warning still applies.
- **v1.4+** (post-milestone, deferred to TRI-FUT-04): csrc/ tree, setup.py
  CUDA extension code, and `_cuda_legacy/` are deleted. The standard 2-release
  deprecation cadence (warn in v1.2, default-off in v1.3, remove in v1.4)
  gives users two minor releases to migrate.

Cross-reference: REQUIREMENTS.md `DEPR-01` (warning fires on explicit cuda),
`DEPR-02` (the exact warning text, locked in this doc), `DEPR-03` (once-per-
process via simplefilter), `DEPR-04` (stacklevel=2), `DEPR-05` (timeline text
references v1.3 and v1.4+). TRI-FUT-04 covers the eventual csrc/ deletion.

## Phase 10 Acceptance Reference

Phase 10's success criteria from ROADMAP.md require:
1. `TORCH_STRUCTURED_BACKEND=cuda python -c "import torch_structured"` emits
   exactly one `DeprecationWarning` with the verbatim text from the
   incantation block above.
2. The warning's attribution (visible via `python -W error::DeprecationWarning`
   on import) points at the `_ops.py` importer line, not the inside of
   `_cuda_legacy/__init__.py`.
3. Repeated imports / repeated explicit cuda selections in the same process
   do NOT emit further warnings (verified by capturing stderr and asserting
   the warning text appears exactly once).
4. `pytest.warns(DeprecationWarning, match="CUDA C\\+\\+ backend")` works
   for tests that need to assert the warning fires.
