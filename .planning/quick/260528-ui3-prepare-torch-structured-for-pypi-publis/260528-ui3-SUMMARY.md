---
phase: quick-260528-ui3
plan: 01
subsystem: packaging
tags: [packaging, pypi, build-system, cuda, triton]
requires: []
provides:
  - "pure-Python py3-none-any wheel (default build compiles nothing)"
  - "graceful extension load (import works without any .so)"
  - "FORCE_CUDA=1 opt-in compiled build path"
  - "version 1.2.0 + LarocheC fork metadata"
affects: [setup.py, pyproject.toml, torch_structured/__init__.py, torch_structured/butterfly/__init__.py, CHANGELOG.md]
tech-stack:
  added: []
  patterns:
    - "default build = pure-Python; native CUDA C++ extensions are opt-in via FORCE_CUDA=1"
    - "extension load is non-fatal: warn + Triton/torch fallback when .so absent"
key-files:
  created:
    - .planning/quick/260528-ui3-prepare-torch-structured-for-pypi-publis/260528-ui3-SUMMARY.md
  modified:
    - setup.py
    - pyproject.toml
    - torch_structured/__init__.py
    - torch_structured/butterfly/__init__.py
    - CHANGELOG.md
decisions:
  - "Default build emits no ext_modules and never imports torch, so uv build runs with normal build isolation"
  - "FORCE_CUDA=1 retains the full CUDA/CPP build path but now requires torch+ninja preinstalled (--no-build-isolation)"
  - "Kept license = {text = 'Apache-2.0'} as a TOML table despite a future-dated setuptools deprecation warning (plan constraint; non-blocking, deprecation is 2027)"
metrics:
  duration: "~3 min"
  completed: "2026-05-28"
  tasks: 3
  files: 5
---

# Quick 260528-ui3: Prepare torch-structured for PyPI Publishing Summary

Made `torch_structured` cleanly publishable as a pure-Python `py3-none-any` wheel: the default `uv build` compiles nothing (Triton is JIT, plus a pure-PyTorch fallback), the legacy CUDA C++ extensions are opt-in via `FORCE_CUDA=1`, import no longer crashes without a `.so`, and metadata/version were updated for the LarocheC fork. Build-verified locally — NOT published.

## What Was Done

### Task 1 — Gate CUDA build behind FORCE_CUDA=1 + graceful import (commit 52d58df)
- **setup.py:** `get_extensions()` now returns `[]` unless `FORCE_CUDA=1` (the existing `BUILD_DOCS` early-return is preserved). `import torch` and the `torch.utils.cpp_extension` imports were moved into the FORCE_CUDA opt-in path (inside `_with_cuda`, `get_torch_ops_extensions`, `get_pybind_extensions`, and a new `_build_kwargs()` helper), so the default build does not require torch to be importable and never touches nvcc/the C++ compiler. `FORCE_CPU=1` still routes through `_with_cuda()` so `FORCE_CUDA=1 FORCE_CPU=1` yields the CppExtension (CPU-compiled) variant. `BuildExtension` cmdclass is attached only when there are extensions to build.
- **torch_structured/butterfly/__init__.py:** `_load_extension` now returns `True`/`False` instead of raising `ImportError` when no `.so` is found. A single low-noise `warnings.warn(..., UserWarning)` is emitted when `_version`/`_butterfly` are absent, explaining the Triton/pure-PyTorch fallback. `check_cuda_version()` is now guarded behind `_version_loaded` (it calls `torch.ops.torch_structured.cuda_version()`, which only exists when `_version` loaded). `_ops.py` was NOT modified — its `_has_cuda_legacy()` probe already returns False when the op is unregistered.

### Task 2 — Version 1.2.0, fork metadata, relaxed build reqs (commit e174361)
- Version `0.4.0 -> 1.2.0` in both `pyproject.toml` (`[project].version`) and `torch_structured/__init__.py` (`__version__`).
- `authors`/`maintainers` set to `Clement Laroche <clement.laroche@gmail.com>`, with a comment crediting the HazyResearch/Tri Dao upstream and pointing at the NOTICE file (NOTICE itself untouched).
- `[project.urls]` repointed to `LarocheC/torch-structured` (Homepage / Repository / Changelog); the old `hazyresearch/learning-circuits` Homepage is gone.
- `[build-system].requires` reduced to `["setuptools>=64", "wheel"]` (dropped `torch` and `ninja`), with a documented tradeoff comment: a `FORCE_CUDA=1` build now needs torch+ninja preinstalled and `--no-build-isolation`. Runtime `torch>=2.6` dependency in `[project].dependencies` retained. `license = {text = "Apache-2.0"}` left unchanged.

### Task 3 — CHANGELOG links + clean build verification (commit 857bd21)
- CHANGELOG.md: the two bottom link-reference URLs (`[Unreleased]`, `[1.2.0]`) repointed from `HazyResearch/torch-structured` to `LarocheC/torch-structured`. The `[1.2.0]` body was left as-is (already complete).
- Ran `uv build` (sdist, then wheel built from the sdist — the authoritative check for what a PyPI consumer receives, avoiding stray local `.so` files in the tree).

## Verification Results (all PASSED)

| Check | Result |
|-------|--------|
| Task 1 AST gate: `get_extensions` references FORCE_CUDA | PASS |
| Task 1: `_load_extension` warns (no raise) | PASS |
| Default `get_extensions()` returns `[]` and does not import torch | PASS |
| Task 2: version 1.2.0 in both files | PASS |
| Task 2: torch + ninja dropped from build reqs | PASS |
| Task 2: LarocheC URLs, no hazyresearch, author email correct | PASS |
| `uv build` exit 0 | PASS |
| Exactly 1 sdist + 1 wheel | PASS |
| Wheel tag is `py3-none-any` | PASS |
| Wheel contains NO `.so`/`.pyd`/`.dylib` | PASS |
| Build log contains no `nvcc` | PASS |
| Wheel METADATA: Version 1.2.0, LarocheC Project-URLs | PASS |

**Final artifacts (in `dist/`, gitignored, NOT committed):**
- `torch_structured-1.2.0.tar.gz` (sdist)
- `torch_structured-1.2.0-py3-none-any.whl` (pure-Python wheel, 58 members, 0 compiled)

## Publishing status

**Build-verified but NOT published.** Per the plan's PREP-ONLY constraint, no `uv publish` and no `twine upload` was run. The user will run `uv publish` (or equivalent) themselves.

## Deviations from Plan

None — plan executed exactly as written.

The only notable observation: `uv build` emits a `SetuptoolsDeprecationWarning` because `project.license` is a TOML table rather than an SPDX string. The plan explicitly directs keeping `license = {text = "Apache-2.0"}` unchanged, and the warning is informational with a 2027-Feb deadline — it does not affect the produced artifacts. Out of scope for this prep task; flagged here for future awareness.

## Known Stubs

None.

## Self-Check: PASSED

All 5 modified source files plus the SUMMARY exist on disk; all 3 task commits (52d58df, e174361, 857bd21) present in git history.
