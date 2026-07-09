---
phase: quick-260709-ik1
plan: 01
subsystem: torch_structured/monarch + factory
status: complete
tags: [monarch, rank-fix, factory-naming, breaking-change, variance-init]
requires: [blockdiag_butterfly_multiply primitive, BlockdiagLinear, MonarchLinear]
provides:
  - "Rank-fixed two-factor MonarchLinear (q = r = in_blksz)"
  - "Inverted factory naming: 'monarch' = two-factor, 'blockdiag' = single-factor"
  - "Corrected variance-matched init for the new fan-in"
  - "Full-rank regression guard"
affects:
  - torch_structured/monarch/monarch_linear.py
  - torch_structured/factory.py
  - experiments/recurrent_poc/layers.py
  - tests/monarch/test_monarch_linear.py
  - CHANGELOG.md
tech-stack:
  added: []
  patterns: ["variance-matched two-factor init", "explicit rtol rank test"]
key-files:
  created: []
  modified:
    - torch_structured/monarch/monarch_linear.py
    - torch_structured/factory.py
    - experiments/recurrent_poc/layers.py
    - tests/monarch/test_monarch_linear.py
    - CHANGELOG.md
decisions:
  - "Corrected set_weights_from_dense_init (Rule 1): the rank fix changed the second contraction fan-in from nblocks to in_blksz, so the composed fan-in is in_blksz**2 not nblocks*in_blksz — the plan's 'no change' assumption was wrong."
  - "Rank test uses explicit rtol=1e-6 to treat genuinely-nonzero singular values (smallest ~4.55e-5 relative at seed 0) as full-rank while still catching a rank-16 structural bottleneck."
metrics:
  duration: ~15min
  completed: 2026-07-09
---

# Quick Task 260709-ik1: Fix Monarch Formulation (Rank Bottleneck + Naming Inversion) Summary

Fixed the two-factor `MonarchLinear` rank bottleneck by setting the intermediate block dims `q = r = in_blksz` (was `nblocks`), corrected the variance-matched init for the resulting wider fan-in, and fully inverted the factory naming so `"monarch"` now builds the genuine two-factor Monarch and a new `"blockdiag"` kind builds the single block-diagonal factor.

## What Changed

### Task 1 — Rank bottleneck fix (`monarch_linear.py`, commit c9d7964)
- `w1` now `(b, in_blksz, in_blksz)` (was `(b, b, in_blksz)`); `w2` now `(b, out_blksz, in_blksz)` (was `(b, out_blksz, b)`).
- Intermediate width `k*q == nblocks*in_blksz == in_features_extended`, so the composed dense-equivalent can reach full rank `min(in,out)` instead of collapsing to `nblocks**2` (fixed 16 for `nblocks=4`).
- Rewrote the `__init__` docstring to describe `q = r = in_blksz` and the rank rationale.
- Verified: `MonarchLinear(400,1200,nblocks=4)` → `w1 (4,100,100)`, `w2 (4,300,100)`, composed rank 400.

### Task 2 — Factory naming inversion (`factory.py`, commit fa210f4, BREAKING)
- `_MonarchLinear` now wraps the two-factor `MonarchLinear`; new `_BlockdiagLinear` wraps `BlockdiagLinear`.
- `_SUPPORTED == ("dense","butterfly","monarch","blockdiag","circulant")`.
- `make_linear("monarch")` → two-factor; `make_linear("blockdiag")` → single-factor; `make_linear("monarch2")` → `ValueError`.
- Updated module + `make_linear` docstrings.

### Task 3 — Follow-on edits (`layers.py`, `CHANGELOG.md`, commit cc61e19)
- Experiments shim re-exports `_BlockdiagLinear` for completeness; `_MonarchLinear` name preserved (name-only re-export, imports cleanly).
- `tests/test_lru.py` left unedited — passes (5 passed, 6 subtests); `"monarch"` in its loop now exercises the real two-factor Monarch.
- CHANGELOG `[Unreleased]`: reconciled the `monarch2` Added entry, added `### Changed` (BREAKING naming inversion), `### Fixed` (rank bottleneck, with the 400→1200 numbers: rank 16→400, params 6400→160000, saving 0.013→0.333), and `### Removed` (`monarch2` kind).

### Task 4 — Full-rank regression guard (`test_monarch_linear.py`, commit 74b68b0)
- Added `test_composed_weight_reaches_full_rank`: asserts the composed dense-equivalent of `MonarchLinear(400,1200,nblocks=4)` reaches rank `min(in,out)==400`, with a message calling out `nblocks**2==16` as the regression signal.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected variance-matched init for the new fan-in**
- **Found during:** Task 4 (the existing `test_variance_matched_init_regression_guard` failed, ratio 25.0 ≈ `in_blksz/nblocks = 100/4`).
- **Issue:** The plan stated `set_weights_from_dense_init` needs no change because `k*q == nblocks*in_blksz` stays exact. That is true, but the init's scaling depends on the two *contraction fan-ins* (`p` for `x@w1^T`, `r` for `out1@w2^T`), not on `k*q`. The rank fix changed `r` from `nblocks` to `in_blksz`, so the composed fan-in went from `p*r = in_blksz*nblocks = in_features_extended` to `p*r = in_blksz**2`. The old formula `Var(out)=nblocks*in_blksz*Var(w1)*Var(w2)*Var(x)` therefore overshot the composed variance by `in_blksz/nblocks` (25× for 400→1200), breaking the regression guard.
- **Fix:** Scale each factor's per-element variance by `sqrt((in_features_extended / in_blksz**2) * v_target)` so `Var(out) = in_blksz**2 * Var(w1)*Var(w2)*Var(x)` matches the dense target `in_features_extended * v_target * Var(x)` again. Updated the docstring with the corrected two-contraction derivation. (This reduces to `sqrt(v_target)` under the old `r=nblocks` shapes, confirming consistency.)
- **Files modified:** `torch_structured/monarch/monarch_linear.py`
- **Commit:** b972813
- **Verification:** `test_variance_matched_init_regression_guard` now passes (ratio within 0.8–1.2 of dense-equivalent).

**2. [Rule 1 - Robustness] Rank test tolerance**
- **Found during:** Task 4.
- **Issue:** With `torch.manual_seed(0)` the composed 1200×400 matrix is genuinely full-rank but its smallest singular value (≈4.55e-5 relative) falls below `matrix_rank`'s default relative tolerance, yielding 399 (seeds 1–5 give 400). A strict `== 400` under the default tolerance was seed-fragile.
- **Fix:** `torch.linalg.matrix_rank(W, rtol=1e-6)` — treats genuinely-nonzero singular values as full-rank while a real rank bottleneck (384 singular values ≈1e-16 relative → rank 16) is still caught for any sane tolerance. Assertion remains `== min(in,out)`.
- **Files modified:** `tests/monarch/test_monarch_linear.py`
- **Commit:** 74b68b0

## Test Results (actual output)

`.venv/bin/python -m pytest tests/monarch/test_monarch_linear.py tests/test_lru.py -q`:
```
29 passed, 3 warnings, 6 subtests passed in 0.42s
```
Broader run `tests/monarch/ tests/test_lru.py`:
```
36 passed, 3 warnings, 6 subtests passed in 0.56s
```
`make_linear` smoke for every `_SUPPORTED` kind (dense, butterfly, monarch, blockdiag, circulant) built and ran a forward pass (shapes `(3,96)` / `(3,64)` for circulant). `make_linear("monarch2")` raises `ValueError`.

Note: the expected `UserWarning` about the compiled CUDA extension not being found (Triton/pure-PyTorch fallback) and `torch.cuda.amp` `FutureWarning`s appear but are not failures. pytest was not pre-installed in `.venv`; installed via `uv pip install pytest` (test-only dependency, per `pyproject.toml [project.optional-dependencies].test`).

## Known Stubs

None.

## Success Criteria

- [x] `w1`/`w2` shapes `(b,in_blksz,in_blksz)` / `(b,out_blksz,in_blksz)`; composed reaches full rank `min(in,out)` for non-square shapes.
- [x] `_SUPPORTED == ("dense","butterfly","monarch","blockdiag","circulant")`; `"monarch"` two-factor, `"blockdiag"` single-factor, `"monarch2"` → `ValueError`.
- [x] Experiments shim imports cleanly; `tests/test_lru.py` passes unedited.
- [x] CHANGELOG documents both the rank fix and the breaking naming inversion.
- [x] New rank-regression test passes; existing monarch tests pass (variance guard fixed).

## Self-Check: PASSED

All modified files present; all task commits (c9d7964, fa210f4, cc61e19, b972813, 74b68b0) exist in git history.
