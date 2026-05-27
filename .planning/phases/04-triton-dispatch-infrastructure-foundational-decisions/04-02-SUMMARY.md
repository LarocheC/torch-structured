---
phase: 04-triton-dispatch-infrastructure-foundational-decisions
plan: 02
subsystem: dispatch
tags: [triton-op, register-fake, torch-compile, dynamo, gradcheck, ci-cache, pytest-fixtures]

# Dependency graph
requires:
  - "Plan 04-01 (torch_structured._ops dispatch module, _torch_ref/ + _cuda_legacy/ packages, torch>=2.6 floor)"
provides:
  - "torch_structured._ops._demo_identity_op — canonical triton_op + wrap_triton + register_autograd + register_fake skeleton; deleted at the start of Phase 5 per D-13"
  - "torch_structured._triton/ empty placeholder package (HAS_TRITON sentinel) — Phase 5+ kernel home"
  - "tests/conftest.py with `backend` fixture (params=['torch']) — Phase 5+ extends params"
  - "tests/test_dispatch.py — five-test acceptance suite; THE 260419-p27 acceptance gate"
  - ".github/workflows/test.yml — actions/cache@v4 for ~/.triton/cache; first CI workflow in the repo"
affects:
  - "torch_structured/_ops.py (extended below the Plan 04-01 resolver with the demonstrator op section)"

# Tech tracking
tech-stack:
  added:
    - "triton, triton.language imports at torch_structured/_ops.py top level (Phase 4 demonstrator only; goes away when demonstrator is deleted in Phase 5)"
    - "torch.library.triton_op, wrap_triton (consumed in _ops.py for the demonstrator)"
    - "actions/cache@v4 CI infrastructure (first time in the repo)"
  patterns:
    - "Canonical @torch.library.triton_op + @triton.jit kernel + register_autograd + register_fake skeleton (template for Phase 5+ kernels)"
    - "Wrapper-boundary view_as_real/view_as_complex with contiguity assertion for complex64 routing (planned-in feature; not contingency code)"
    - "pytest backend fixture with yield-based teardown that restores _BACKEND (the Phase 5+ test parametrization pattern)"
    - "actions/cache@v4 keyed on (os, python, torch, hashFiles('_triton/**/*.py')) — explicitly NOT keyed on github.sha (Pitfall 6 avoidance)"
    - "BLOCK_SIZE passed as constexpr call-site argument (no @triton.heuristics — wrap_triton rejects Heuristics-wrapped kernels; auto-fixed during execution)"

key-files:
  created:
    - "torch_structured/_triton/__init__.py"
    - "tests/conftest.py"
    - "tests/test_dispatch.py"
    - ".github/workflows/test.yml"
    - ".planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/deferred-items.md"
  modified:
    - "torch_structured/_ops.py"

key-decisions:
  - "D-12 / D-13 / D-14 implemented verbatim: _demo_identity_op uses triton_op + wrap_triton + register_autograd + register_fake; lives in _ops.py; tested via the 5-test suite (D-14a fullgraph compile, D-14b gradcheck, D-14c FakeTensorMode trace)"
  - "D-16 implemented: actions/cache@v4 with cache key (os, python-version, torch-version, hashFiles('torch_structured/_triton/**/*.py')) — NOT keyed on github.sha"
  - "@triton.heuristics dropped from the kernel decorator stack — wrap_triton in PyTorch >=2.6 only accepts plain @triton.jit or @triton.autotune (Rule 1 auto-fix during execution); BLOCK_SIZE is passed as a constexpr call-site argument instead. The RESEARCH.md skeleton's use of @triton.heuristics would crash at first call."
  - "Complex64 path is unconditional in the wrapper body (not gated behind a runtime feature flag): it exercises view_as_real/view_as_complex on every call where input is complex, with an explicit contiguity assertion (Pitfall 3). Phase 7 inherits a working reference."
  - "_triton/__init__.py defines HAS_TRITON via try-import. The acceptance criterion's strict 'torch_structured._triton not in sys.modules after import torch_structured' is unreachable in Phase 4 because (a) _ops.py imports triton unconditionally at top-level for the demonstrator decorators, and (b) the import-time _has_triton_kernel('butterfly_multiply') resolver probe loads the _triton parent package as a side effect. Both costs disappear when Phase 5 deletes the demonstrator and replaces the import-time probe with the first real kernel."
  - "Empty kernel directory means hashFiles('torch_structured/_triton/**/*.py') returns a stable empty-glob hash in Phase 4; cache key effectively reduces to (os, python, torch) until Phase 5 lands the first real kernel"
  - "All 5 demonstrator tests passed on CUDA 13.0 / torch 2.11 / Triton 3.6.0 — including the literal 260419-p27 acceptance gate (test_demo_identity_compile_fake_tensor_trace) which exercises explicit FakeTensorMode tracing"

requirements-completed: [TRI-05, TEST-05]

# Metrics
duration: 1h30m
completed: 2026-05-27
---

# Phase 04 Plan 02: Triton Dispatch Demonstrator + CI Cache Summary

**Demonstrates the canonical @torch.library.triton_op + wrap_triton + register_autograd + register_fake pipeline via a no-op identity op (`_demo_identity_op`) in `torch_structured/_ops.py`, ships the five-test acceptance suite that gates the 260419-p27 dynamo bug fix (test_demo_identity_compile_fake_tensor_trace), introduces the pytest `backend` fixture pattern for Phase 5+, and lands the first CI workflow with `actions/cache@v4` for `~/.triton/cache` (Pitfall 6 avoided).**

## Performance

- **Duration:** ~1h30m
- **Started:** 2026-05-27
- **Completed:** 2026-05-27
- **Tasks:** 3

## Tasks Completed

### Task 1: Append `_demo_identity_op` to `_ops.py` + create `_triton/` placeholder package
**Commit:** `7922971`

- Added three imports at the top of `torch_structured/_ops.py`: `import triton`, `import triton.language as tl`, `from torch.library import triton_op, wrap_triton`. Plan 04-01's existing resolver code is untouched.
- Appended the demonstrator section under the import-time `_resolve(_initial)` call, clearly delimited by a section comment header that calls out D-13's "deleted at start of Phase 5" lifecycle.
- The kernel `_demo_identity_kernel` is plain `@triton.jit` (no `@triton.heuristics` — see Deviations below). `BLOCK_SIZE` is a `tl.constexpr` passed at the call site (fixed at 1024).
- The wrapper `_demo_identity_op` is decorated `@triton_op("torch_structured::_demo_identity", mutates_args={})` and its body **unconditionally** routes complex64 inputs through `view_as_real → kernel → view_as_complex`. An `assert x.is_contiguous()` precondition guards the Pitfall 3 footgun. The real path is the no-branch case.
- `_setup_context` and `_backward` are minimal — identity backward is `return grad` — and wired via `_demo_identity_op.register_autograd(_backward, setup_context=_setup_context)`.
- `@_demo_identity_op.register_fake` decorates a meta kernel that returns `torch.empty_like(x)`. **This is the literal 260419-p27 fix.**
- Created `torch_structured/_triton/__init__.py` as a tiny placeholder package with a docstring explaining the Phase 5+ role and a `HAS_TRITON: bool` constant resolved via try-import (5 lines of executable code, ~14 lines total with comments).

### Task 2: tests/conftest.py + tests/test_dispatch.py (5-test acceptance suite)
**Commit:** `2d75d1a`

- `tests/conftest.py` defines the `backend` fixture parametrized on `params=["torch"]` only. The fixture captures `_BACKEND`, calls `set_backend(request.param)`, yields the chosen name, and restores the original `_BACKEND` on teardown (yield-based cleanup) so tests are order-independent. The module-import-side-effect comment style (`import torch_structured  # noqa: F401`) matches `tests/test_lru.py:6`.
- `tests/test_dispatch.py` has the verbatim 5-test acceptance suite from RESEARCH.md lines 661-714:

| Test | D-XX gate | What it verifies |
|------|-----------|------------------|
| `test_demo_identity_eager_fp32` | (sanity) | `wrap_triton` + `@triton.jit` plumbing on fp32 |
| `test_demo_identity_eager_complex64` | D-14 (+ B2 fix + RESOLVED Q1) | `view_as_real`/`view_as_complex` wrapper boundary works on complex64; dtype preserved |
| `test_demo_identity_gradcheck` | D-14b | `register_autograd` wires backward correctly (gradcheck passes) |
| `test_demo_identity_compile_no_graph_break` | D-14a | `register_fake` lets `torch.compile(fullgraph=True)` trace through without graph-breaking |
| `test_demo_identity_compile_fake_tensor_trace` | D-14c — **THE 260419-p27 acceptance gate** | Explicit `FakeTensorMode()` tracing does NOT raise "The tensor has a non-zero number of elements, but its data is not allocated yet" |

- `pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="demo op is GPU-only")` so the suite skips cleanly on CPU-only hosts.
- **All 5 tests pass on CUDA 13.0 / torch 2.11 / Triton 3.6.0.**

### Task 3: .github/workflows/test.yml (first CI workflow in the repo)
**Commit:** `dcbf93b`

- Workflow runs on `[push, pull_request]`, uses `actions/checkout@v4` + `actions/setup-python@v5` (Python 3.11).
- A "Resolve torch version" step pre-installs torch and exports `TORCH_VERSION` to `$GITHUB_ENV` so the cache key can reference it.
- The cache step uses `actions/cache@v4` (literally — pinned per D-16, not v3 or v5) with:
  - `path: ~/.triton/cache`
  - `key: triton-${{ runner.os }}-py${{ env.PYTHON_VERSION }}-torch${{ env.TORCH_VERSION }}-${{ hashFiles('torch_structured/_triton/**/*.py') }}` — **explicitly NOT `${{ github.sha }}` (Pitfall 6 avoided)**
  - Two `restore-keys` fallback prefixes for graceful partial-match cache hits
- Install + test: `pip install -e .[test]` then `pytest tests/ -x`.

## Architectural Outcomes

### The canonical Triton-op skeleton (template for Phase 5+)

```python
import triton, triton.language as tl
from torch.library import triton_op, wrap_triton

@triton.jit
def _kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)

@triton_op("torch_structured::<name>", mutates_args={})
def <name>(x):
    # Complex64 wrapper boundary if applicable:
    if x.is_complex():
        assert x.is_contiguous(), "Pitfall 3"
        x = torch.view_as_real(x)
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    wrap_triton(_kernel)[grid](x, out, x.numel(), 1024)
    return torch.view_as_complex(out.contiguous()) if was_complex else out

<name>.register_autograd(_backward, setup_context=_setup_context)

@<name>.register_fake
def _(x):
    return torch.empty_like(x)
```

Phase 5 will copy this skeleton verbatim into `torch_structured/_triton/diag_mult/op.py` for the first real kernel.

### Backend-fixture extension path

```python
# Phase 4 (this plan)
@pytest.fixture(params=["torch"])

# Phase 5 (when first real Triton kernel lands AND CUDA available)
@pytest.fixture(params=["torch", "triton"])

# Phase 7+ (when both Triton + legacy CUDA paths are exercised by tests)
@pytest.fixture(params=["torch", "triton", "cuda"])
```

### CI cache key composition

```
triton-${runner.os}-py${PYTHON_VERSION}-torch${TORCH_VERSION}-${hashFiles('torch_structured/_triton/**/*.py')}
```

Phase 4 reality check: the empty `_triton/` directory (just `__init__.py`) means `hashFiles` returns a stable empty-glob hash, so the cache key effectively reduces to `(os, python, torch)` until Phase 5 adds kernel files. The `restore-keys` fallback prefixes give graceful degradation on partial matches.

### Phase 5 transition plan (per D-13)

When Phase 5 begins:

1. **Delete** the demonstrator section from `_ops.py` (everything below the `# ─── Phase 4 demonstrator op` header).
2. **Add** `torch_structured/_triton/diag_mult/op.py` containing the first real Triton kernel using the demonstrator's skeleton as a template (D-12 contract: `triton_op + wrap_triton + register_autograd + register_fake`).
3. **Extend** `tests/conftest.py` `backend` fixture: `params=["torch", "triton"]` (Phase 5+ kernels exist now).
4. **The CI cache infrastructure works unchanged** — Phase 5 inherits a working cache the moment the first kernel JITs.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] RESEARCH.md skeleton's `@triton.heuristics` is incompatible with `wrap_triton` in PyTorch ≥2.6**

- **Found during:** Task 1 smoke test (eager fp32 call)
- **Issue:** `wrap_triton` only accepts callables that are bare `@triton.jit`-decorated kernels OR `@triton.autotune`-wrapped. `@triton.heuristics` produces a `Heuristics` object (not a `JITFunction`), and `wrap_triton` raises `RuntimeError: wrap_triton only works on functions annotated with triton.jit or triton.autotune` at the first call.
- **Fix:** Dropped `@triton.heuristics` from the decorator stack. `BLOCK_SIZE` is now passed as a `tl.constexpr` argument at the call site (fixed at 1024). For a no-op identity demonstrator this is fine; autotune isn't needed. Phase 5+ kernels that need autotuning can use `@triton.autotune` (which IS accepted by `wrap_triton`).
- **Files modified:** `torch_structured/_ops.py` (`_demo_identity_kernel` decorator stack + `_demo_identity_op` call site).
- **Commit:** `7922971` (folded into Task 1 — discovered immediately after the initial write).
- **Note:** The RESEARCH.md and the plan's `<action>` block both prescribed `@triton.heuristics` per the official PyTorch tutorial. Per Pitfall 1 reasoning, the official tutorial misses things — and this is another one. The fix preserves all five acceptance criteria.

**2. [Rule 1 - Bug] `_triton/__init__.py` cannot fully prevent `torch_structured._triton` from appearing in `sys.modules` after `import torch_structured`**

- **Found during:** Task 1 verification
- **Issue:** The acceptance criterion text said `python -c "import sys; import torch_structured; assert 'torch_structured._triton' not in sys.modules"` should succeed. But Plan 04-01's `_resolve(_initial)` runs at import time and calls `_has_triton_kernel("butterfly_multiply")`, which uses `importlib.import_module("torch_structured._triton.butterfly_multiply.op")`. Python's import system loads the parent package `torch_structured._triton` as a side effect, even though the submodule import fails. So `_triton` ends up in `sys.modules` regardless of how `_triton/__init__.py` is written.
- **Spirit of the criterion:** The intent is "don't eagerly load LLVM/Triton at `import torch_structured` time." But the plan ALSO requires `import triton` at the top of `_ops.py` for the demonstrator decorators — so LLVM is loaded at import time unavoidably as long as the demonstrator lives.
- **Fix:** Wrote `_triton/__init__.py` exactly as the plan specifies (try-import for HAS_TRITON). Documented the LLVM-cost transient nature in the file's docstring and in this Summary's "key-decisions" section. The cost goes away in Phase 5 (when the demonstrator is deleted and `_ops.py`'s top-level `import triton` goes away).
- **Files modified:** `torch_structured/_triton/__init__.py` (the plan's prescription).
- **Commit:** `7922971` (folded into Task 1).

### Out-of-scope discoveries (deferred, not fixed)

Logged in `.planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/deferred-items.md`:

1. **Pre-existing CUDA-extension test failures** (`test_butterfly.py`, `test_multiply.py`, `test_permutation.py` — 8 failures total). Same root cause as Plan 04-01's "Deviations" section: the worktree has a CPU-only stub `.so` (because host CUDA 12.6 doesn't match torch CUDA 13.0). Plan 04-02 doesn't touch the C++ path; these failures reproduce identically on master before this plan.
2. **`tests/test_special.py` collection error** — module imports `pywt` which isn't in `pyproject.toml`'s `test` extra. Predates Plan 04-02.

## Authentication Gates

None — Phase 4 is purely local Python + CI YAML work.

## Verification Results

| Check | Result |
|-------|--------|
| `from torch_structured._ops import _demo_identity_op` exits 0 | PASS |
| `_ops.py` (filtered) contains `@triton_op(`, `wrap_triton`, `register_autograd`, `@_demo_identity_op.register_fake` | PASS |
| Op registered with namespace `torch_structured::_demo_identity` (`torch.ops.torch_structured._demo_identity` exists) | PASS |
| Plan 04-01 invariants preserved (`set_backend('torch')` → `_BACKEND == 'torch'`) | PASS |
| `torch_structured/_triton/__init__.py` exists with `HAS_TRITON: bool` module-level constant | PASS |
| `_ops.py` contains both `view_as_real` and `view_as_complex` literal calls (complex64 wrapper boundary) | PASS |
| `tests/conftest.py` `backend` fixture parametrized on `["torch"]` only | PASS |
| `tests/test_dispatch.py` contains all 5 required test functions | PASS |
| `tests/test_dispatch.py` uses `pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), ...)` | PASS |
| `pytest tests/test_dispatch.py --collect-only` collects 5 tests | PASS |
| **`pytest tests/test_dispatch.py -v` — 5/5 PASS on CUDA** | **PASS** |
| **`test_demo_identity_compile_fake_tensor_trace` (THE 260419-p27 acceptance gate)** | **PASS — no "data is not allocated yet" error** |
| `.github/workflows/test.yml` uses `actions/cache@v4` literally (not v3 or v5) | PASS |
| Cache `path:` is `~/.triton/cache` literally | PASS |
| Cache key includes `runner.os`, python version, torch version, `hashFiles('torch_structured/_triton/**/*.py')` | PASS |
| Cache key does NOT contain `github.sha` (Pitfall 6 avoided) | PASS |
| `restore-keys` has two fallback prefixes | PASS |
| `python -c "import yaml; yaml.safe_load(open('.github/workflows/test.yml'))"` exits 0 (valid YAML) | PASS |

### THE acceptance gate — direct evidence

```
$ python -m pytest tests/test_dispatch.py::test_demo_identity_compile_fake_tensor_trace -v
tests/test_dispatch.py::test_demo_identity_compile_fake_tensor_trace PASSED [100%]
======================== 1 passed, 4 warnings in 0.42s =========================
```

The `register_fake` decorator on `_demo_identity_op` is what makes this pass. Without it, this test would reproduce the literal 260419-p27 error string ("The tensor has a non-zero number of elements, but its data is not allocated yet").

## Known Stubs

None — `_demo_identity_op` is a deliberate no-op demonstrator (per D-13). It is documented in source comments and in this Summary that it will be deleted at the start of Phase 5. This is not a stub in the "data that should be wired but isn't" sense; it's a planned-in proof-of-concept with a planned deletion date.

The `_triton/` package is empty by design (Phase 4 placeholder per RESEARCH §"Recommended Project Structure"). Phase 5+ kernel ports populate it.

## Self-Check: PASSED

Verified the following claims:

```
$ [ -f torch_structured/_triton/__init__.py ] && echo FOUND
FOUND
$ [ -f tests/conftest.py ] && echo FOUND
FOUND
$ [ -f tests/test_dispatch.py ] && echo FOUND
FOUND
$ [ -f .github/workflows/test.yml ] && echo FOUND
FOUND
$ [ -f .planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/deferred-items.md ] && echo FOUND
FOUND
$ git log --all --oneline | grep -q 7922971 && echo FOUND
FOUND
$ git log --all --oneline | grep -q 2d75d1a && echo FOUND
FOUND
$ git log --all --oneline | grep -q dcbf93b && echo FOUND
FOUND
```
