---
phase: 260419-v4b
plan: 01
subsystem: packaging
tags: [lru, factory, public-api, version-bump, shim]
dependency-graph:
  requires: []
  provides:
    - torch_structured.LRU
    - torch_structured.make_linear
    - torch_structured.recurrent (subpackage)
    - torch_structured.factory (module)
  affects:
    - torch_structured.__init__
    - pyproject.toml
    - experiments/recurrent_poc/layers.py
    - experiments/recurrent_poc/lru.py
tech-stack:
  added: []
  patterns:
    - "Backward-compat shims re-export from canonical package modules"
    - "Private test-only module flag (`_FORCE_NAIVE`) for deterministic scan-vs-naive equivalence tests"
key-files:
  created:
    - torch_structured/factory.py
    - torch_structured/recurrent/__init__.py
    - torch_structured/recurrent/lru.py
    - tests/test_lru.py
  modified:
    - torch_structured/__init__.py
    - pyproject.toml
    - experiments/recurrent_poc/layers.py
    - experiments/recurrent_poc/lru.py
decisions:
  - "Bumped torch-structured 0.3.0 -> 0.4.0 to reflect new public API (LRU + make_linear)"
  - "Keep `make_linear` bias keyword-only (behavioral change from the original layers.py positional form); all existing callers already pass bias= by keyword"
  - "Use BUILD_DOCS=1 during reinstall to skip CUDA rebuild — .so files already compiled in-place and editable mode picks them up from the source tree"
  - "Shims in experiments/recurrent_poc/ kept to pure re-exports (no warnings.warn) because the bench scripts are hot paths"
  - "_FORCE_NAIVE test hook documented as private; do not rely on in production code"
metrics:
  duration_seconds: 401
  tasks_completed: 6
  files_touched: 8
  completed_date: 2026-04-19
---

# Phase 260419-v4b Plan 01: Promote LRU into torch_structured Package Summary

Promote the production-ready `LRU` layer and `make_linear` factory out of `experiments/recurrent_poc/` and into the installed `torch_structured` package so downstream repos can do `from torch_structured import LRU, make_linear` after a single `pip install`. Bench scripts keep working via thin re-export shims. Version bumped to 0.4.0.

## Files Created

- `torch_structured/factory.py` (136 LOC) — `make_linear` + `_ButterflyLinear` / `_MonarchLinear` / `_CirculantLinear` wrappers, `_is_pow2`, `_SUPPORTED`, `_NOT_WIRED` constants. `make_linear` signature hardened with keyword-only `bias`.
- `torch_structured/recurrent/__init__.py` (3 LOC) — exposes `LRU` on `torch_structured.recurrent.LRU`.
- `torch_structured/recurrent/lru.py` (293 LOC) — full `LRU` class + `_LRULayer` + `_lru_scan` dispatcher (parallel scan with naive fallback). Moved verbatim from `experiments/recurrent_poc/lru.py`; import switched to `torch_structured.factory`; full type hints added; `_FORCE_NAIVE` private test hook added.
- `tests/test_lru.py` (91 LOC) — 5 unittest.TestCase methods following `tests/test_butterfly.py` style.

## Files Modified

- `torch_structured/__init__.py` — added `from .factory import make_linear`, `from .recurrent import LRU`; extended `__all__`; bumped `__version__` to `'0.4.0'`; added recurrent subpackage bullet to module docstring.
- `pyproject.toml` — `version = "0.3.0"` → `"0.4.0"`.
- `experiments/recurrent_poc/layers.py` — now a 16-LOC re-export shim from `torch_structured.factory`.
- `experiments/recurrent_poc/lru.py` — now a 10-LOC re-export shim from `torch_structured.recurrent.lru`.

## Version Bump

0.3.0 → 0.4.0 in both `torch_structured/__init__.py` and `pyproject.toml`.

## Test Outcome (Sanity Check c)

```
tests/test_lru.py::LRUTest::test_lru_bidirectional_concat PASSED
tests/test_lru.py::LRUTest::test_lru_forward_backward PASSED
tests/test_lru.py::LRUTest::test_lru_matches_nn_gru_interface PASSED
tests/test_lru.py::LRUTest::test_lru_scan_vs_naive PASSED
tests/test_lru.py::LRUTest::test_lru_structured_kind PASSED

=== 5 passed, 6 subtests passed in 0.69s ===
```

Broader suite: 49 passed, 1 skipped, 2 pre-existing failures in `test_permutation` / `test_special::test_dst` (documented in STATE.md as known before this plan — unrelated to LRU, factory, or make_linear).

## Bench Outcome (Sanity Check d)

- `bench_recurrent.py --quick`: ok (cudnn / gru / lru / mamba all report fwd_ms + fwd_bwd_ms rows)
- `bench_lin_rnn.py --quick`: ok (dense + butterfly both report naive_ms / scan_ms / speedup)

## CUDA Public API Smoke (Sanity Check b)

```
python -c "
import torch_structured
from torch_structured import LRU, make_linear
import torch
m = LRU(64, 128, num_layers=2, bidirectional=True, batch_first=True,
        kind='butterfly').cuda()
out, hn = m(torch.randn(4, 100, 64).cuda())
print(out.shape, hn.shape, torch_structured.__version__)
"
# -> torch.Size([4, 100, 256]) torch.Size([4, 4, 128]) 0.4.0
```

Exact match against the spec.

## Downstream Usage Snippet

```python
from torch_structured import LRU, make_linear

# Drop-in nn.GRU peer
rnn = LRU(input_size=64, hidden_size=128, num_layers=2,
          batch_first=True, bidirectional=True, kind='butterfly')
out, h_n = rnn(x)  # x: (B, T, 64) -> out: (B, T, 256), h_n: (4, B, 128)

# Factory for structured-linear layers
layer = make_linear('butterfly', 128, 128, bias=True)
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Reinstall rebuild failure**
- **Found during:** Task 4 / Task 6 sanity check a
- **Issue:** The plan's reinstall command (`uv pip install --no-build-isolation -e . --no-deps`) triggered a full CUDA extension rebuild which fails on this environment (ninja: `csrc/cuda/butterfly_cuda.o` compile error). The environment notes explicitly stated: "C++ extensions are already built. The reinstall should NOT rebuild them."
- **Fix:** Prefixed the reinstall command with `BUILD_DOCS=1`, which `setup.py:get_extensions()` interprets as "skip ext_modules". The already-compiled `.so` files in `torch_structured/*.so` remain in place and are loaded via the editable source path.
- **Files modified:** None — command-invocation-level fix.
- **Commit:** Reflected in Task 4 verification (install succeeded with `0.3.0 → 0.4.0`).

## Self-Check: PASSED

Files verified present:
- `/home/clement/torch-structured/torch_structured/factory.py` FOUND
- `/home/clement/torch-structured/torch_structured/recurrent/__init__.py` FOUND
- `/home/clement/torch-structured/torch_structured/recurrent/lru.py` FOUND
- `/home/clement/torch-structured/tests/test_lru.py` FOUND

Commits verified in `git log master..HEAD`:
- `03aa799` feat(260419-v4b-01): add torch_structured.factory with make_linear — FOUND
- `f2302ef` feat(260419-v4b-01): add torch_structured.recurrent subpackage with LRU — FOUND
- `3e6ed79` test(260419-v4b-01): add unittest suite for LRU public API — FOUND
- `e58366a` feat(260419-v4b-01): wire LRU + make_linear into public API, bump to 0.4.0 — FOUND
- `c7929f3` refactor(260419-v4b-01): replace layers.py and lru.py with re-export shims — FOUND

All 4 user sanity checks (a, b, c, d) passed. All `must_haves.truths` satisfied.
