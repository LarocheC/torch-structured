---
phase: 04-triton-dispatch-infrastructure-foundational-decisions
plan: 01
subsystem: dispatch
tags: [dispatch, infrastructure, torch-library, complex64, deprecation, python-packaging]

# Dependency graph
requires: []
provides:
  - "torch_structured._ops module-level dispatch (DISP-01..05) — read at import, switchable via set_backend()"
  - "torch_structured._torch_ref/ peer package holding butterfly_multiply_torch (Phase 5/6 will add diag_mult.py and hadamard.py)"
  - "torch_structured._cuda_legacy/ minimal wrapper around torch.ops.torch_structured.butterfly_multiply (no @torch.jit.script)"
  - "torch_structured.set_backend top-level re-export"
  - "torch>=2.6 floor in pyproject.toml (the triton_op API floor)"
  - "04-COMPLEX-LAYOUT.md — Phase 7 implementation reference for TRI-06"
  - "04-DEPRECATION-PLAN.md — Phase 10 implementation reference for DEPR-02"
  - "Call-site contract D-05 (attribute-access discipline) documented verbatim in _ops.py module docstring"
affects:
  - "torch_structured/butterfly/multiply.py — butterfly_multiply_torch is now a re-export from _torch_ref/"
  - "torch_structured/__init__.py — re-exports set_backend; triggers _ops import-time _resolve()"

# Tech tracking
tech-stack:
  added:
    - "torch.library.triton_op (declared via torch>=2.6 floor; consumer kernels land Phase 5+)"
  patterns:
    - "Module-level rebindable callables (_ops.butterfly_multiply = ...) for backend dispatch with set_backend()"
    - "Honest per-op probe (_has_triton_kernel(op_name)) distinct from _has_triton() to prevent silent backend mismatch"
    - "Wrapper-boundary view_as_real / view_as_complex pattern for complex64 routing (documented for Phase 7)"
    - "warnings.simplefilter('once', DeprecationWarning) + stacklevel=2 for per-process deprecation (documented for Phase 10)"

key-files:
  created:
    - "torch_structured/_torch_ref/__init__.py"
    - "torch_structured/_torch_ref/butterfly.py"
    - "torch_structured/_cuda_legacy/__init__.py"
    - "torch_structured/_cuda_legacy/butterfly.py"
    - "torch_structured/_ops.py"
    - ".planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/04-COMPLEX-LAYOUT.md"
    - ".planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/04-DEPRECATION-PLAN.md"
  modified:
    - "torch_structured/butterfly/multiply.py"
    - "torch_structured/__init__.py"
    - "pyproject.toml"

key-decisions:
  - "D-04 _ops.py module-level callables rebound by _resolve(); D-05 attribute-access call-site contract documented in module docstring"
  - "B3 honest resolver: _has_triton_kernel(op_name) per-op probe ensures auto/triton never silently 'binds to triton' when no kernel exists (Phase 4 _triton/ empty so probe always returns False)"
  - "W4 warning: explicit set_backend('triton') in Phase 4 falls back AND emits log.warning recording the fallback path"
  - "D-08 INFO heads-up gated on actual=='triton' AND _has_cuda_legacy() — dormant in Phase 4, exercised first in Phase 5"
  - "D-09 / D-10 butterfly_multiply_torch moved (not deleted) into _torch_ref/, shim preserves all existing test imports"
  - "D-11 torch>=2.6 floor non-negotiable (triton_op API requirement)"
  - "D-15 deprecation incantation captured verbatim in 04-DEPRECATION-PLAN.md so Phase 10 doesn't re-derive"
  - "T-04-01 mitigation: _resolve() validates name against {triton,cuda,torch,auto} and raises ValueError otherwise (no dynamic import of env-var value)"

patterns-established:
  - "Wave 2 (Plan 04-02) will add the demonstrator op into _ops.py using the same module — _demo_identity_op with triton_op + wrap_triton + register_autograd + register_fake"
  - "Phase 5+ consumer migrations follow the D-05 call-site contract: torch_structured._ops.X(...) NEVER from torch_structured._ops import X"

requirements-completed: [DISP-01, DISP-02, DISP-03, DISP-04, DISP-05, COMPAT-05, TRI-06, TRI-07]

# Metrics
duration: 1h00m
completed: 2026-05-27
---

# Phase 04 Plan 01: Triton Dispatch Infrastructure & Foundational Decisions Summary

**Lays down the dispatch infrastructure foundation (DISP-01..05) every subsequent kernel port phase inherits: peer package layout (`_torch_ref/`, `_cuda_legacy/`), single dispatch point (`_ops.py`) with honest backend resolution, top-level `set_backend` re-export, PyTorch floor bump to >=2.6, and two phase-companion docs that Phase 7 (complex64 routing) and Phase 10 (CUDA deprecation) will implement verbatim.**

## Performance

- **Duration:** ~1 hour
- **Started:** 2026-05-27 (after build environment troubleshooting — CPU-only editable install was needed due to CUDA 12.6 host vs. torch CUDA 13.0 mismatch; this is a worktree env issue, not a plan issue)
- **Completed:** 2026-05-27
- **Tasks:** 3

## Tasks Completed

### Task 1: Extract `_torch_ref/` + `_cuda_legacy/` packages and back-compat shim
**Commit:** 4f37991

- Created `torch_structured/_torch_ref/__init__.py` and `torch_structured/_torch_ref/butterfly.py`. `butterfly_multiply_torch` moved verbatim (zero logic change) from `torch_structured/butterfly/multiply.py:28-49`. Drops the unused `math` / `Tuple` / `Optional` imports because they were for the `_fw` / `_bw` JIT wrappers that stayed in the old location.
- Created `torch_structured/_cuda_legacy/__init__.py` and `torch_structured/_cuda_legacy/butterfly.py`. The wrapper is a plain Python function with `@torch.jit.script` deliberately omitted (TorchScript composes poorly with Inductor; the dispatch wrapper in `_ops.py` may itself be inside a `torch.compile` graph).
- Replaced the original `butterfly_multiply_torch` definition in `torch_structured/butterfly/multiply.py` with the shim `from torch_structured._torch_ref.butterfly import butterfly_multiply_torch  # noqa: F401`. The three `@torch.jit.script` wrappers (`butterfly_multiply_fw`, `butterfly_multiply_bw`, `butterfly_multiply`) stay UNCHANGED.
- All three pre-Phase-4 import paths (`torch_structured._torch_ref.butterfly`, `torch_structured.butterfly.multiply`, `torch_structured.butterfly.butterfly`) resolve to the SAME function object via the `is` identity check.

### Task 2: Create `torch_structured/_ops.py` + top-level `set_backend` re-export + `pyproject.toml` torch>=2.6 bump
**Commit:** 4bb68ba

- Created `torch_structured/_ops.py` per CHECKER B3 honest-resolver design:
  - `_has_triton_kernel(op_name)` per-op probe distinct from `_has_triton()` — in Phase 4 the `_triton/` package is empty (Plan 04-02 will create the placeholder) so the probe returns False for every op. `auto` therefore resolves to `cuda` (when `.so` present) or `torch`, NEVER to `triton`.
  - Explicit `set_backend('triton')` requests fall back to `cuda` / `torch` AND emit `log.warning("set_backend('triton') requested but no Triton kernel installed; falling back to %s", actual)` (W4 gate per CHECKER).
  - `_BACKEND` global reflects the ACTUAL binding, never the requested name — observers cannot be deceived.
  - D-08 INFO heads-up gated on `actual == "triton" AND _has_cuda_legacy()` — dormant in Phase 4, exercised when Phase 5 ships the first real Triton kernel.
  - T-04-01 mitigation: `_resolve(name)` validates against `{triton, cuda, torch, auto}` and raises `ValueError` otherwise. NO `importlib.import_module(name)` and NO `eval/exec` on the env-var value.
- Module docstring documents the D-05 call-site contract verbatim with WRONG/CORRECT examples for Phase 5+ consumer migrations.
- Added `from ._ops import set_backend` to `torch_structured/__init__.py` and extended `__all__`. No circular import — verified via `python -c "import torch_structured; torch_structured.set_backend('torch')"` (RESEARCH Open Q2).
- Bumped `pyproject.toml`: `torch>=2.0` → `torch>=2.6` in BOTH `[build-system].requires` (line 2) AND `[project].dependencies` (line 25). `grep -c 'torch>=2.6' pyproject.toml` returns 2; `torch>=2.0` no longer appears.

### Task 3: Write `04-COMPLEX-LAYOUT.md` and `04-DEPRECATION-PLAN.md` companion docs
**Commit:** 8ae69a5

- `04-COMPLEX-LAYOUT.md` (124 lines, D-01/D-02/D-03/TRI-06): locks the Phase 7 implementation. Covers the wrapper-boundary `view_as_real → kernel → view_as_complex` template, kernel-side `IS_COMPLEX: tl.constexpr` flag with explicit 4-FMA complex multiply, twiddle layout invariance (`(nstacks, nblocks, log_n, n/2, 2, 2)` — NOT touched, COMPAT-02), the Pitfall 3 contiguity assertion that the wrapper MUST enforce before `view_as_real`, autograd preservation through PyTorch's `derivatives.yaml` view backward rules, and the rejection of `tl.complex64` (Triton has no native complex type).
- `04-DEPRECATION-PLAN.md` (147 lines, D-15 / DEPR-01..05): locks the Phase 10 implementation. Includes the verbatim `warnings.warn(...)` text referencing v1.3 default-disabled and v1.4+ removal, the `warnings.simplefilter("once", DeprecationWarning)` once-per-process pattern, `stacklevel=2` rationale, the routing detail (explicit `cuda` vs. auto path), and Phase 10 acceptance criteria.

## Architectural Outcomes

### New package layout

```
torch_structured/
├── __init__.py                  # +set_backend re-export, +__all__ extension
├── _ops.py                      # NEW — dispatch entry point (DISP-01..05)
├── _torch_ref/                  # NEW
│   ├── __init__.py
│   └── butterfly.py             # butterfly_multiply_torch (moved from butterfly/multiply.py:28)
├── _cuda_legacy/                # NEW
│   ├── __init__.py
│   └── butterfly.py             # pass-through to torch.ops.torch_structured.butterfly_multiply (no @torch.jit.script)
└── butterfly/
    ├── multiply.py              # MODIFIED — butterfly_multiply_torch is now a re-export shim
    └── ...                      # everything else UNCHANGED
```

### Call-site contract for Phase 5+

```python
# CORRECT — sees set_backend() rebinds
import torch_structured
torch_structured._ops.butterfly_multiply(twiddle, x, ...)

# WRONG — captures CURRENT object at import time
from torch_structured._ops import butterfly_multiply
butterfly_multiply(twiddle, x, ...)   # set_backend() invisible
```

Phase 5 onward enforces this in consumer plans (every `nn.Module.forward` that calls a dispatched op).

### Companion docs for downstream phases

- **Phase 7** reads `04-COMPLEX-LAYOUT.md` verbatim when implementing TRI-03 butterfly forward kernel. The doc contains the canonical wrapper pattern, the kernel-side `IS_COMPLEX` template with 4-FMA complex multiply, and the Pitfall 3 contiguity assertion.
- **Phase 10** reads `04-DEPRECATION-PLAN.md` verbatim when implementing DEPR-02. The doc contains the exact `warnings.warn(...)` text, the `warnings.simplefilter("once", DeprecationWarning)` setup, and the `stacklevel=2` rationale.

### Demonstrator op (Plan 04-02 Wave 2)

The demonstrator op (`_demo_identity_op` per D-13) will land in `_ops.py` in Plan 04-02. It will use `@triton_op + register_autograd + register_fake` to prove the wrapper pattern survives `torch.compile` and `gradcheck` and to reproduce-then-fix the 260419-p27 dynamo bug. Plan 04-01's `_ops.py` only contains the resolver + `set_backend`; the demonstrator is purely additive in Wave 2.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking issue] Build environment had no `.so` extensions present in worktree**

- **Found during:** Task 1 verify (post-implementation)
- **Issue:** The worktree had no compiled `_butterfly.so` / `_version.so`. The default `pip install -e .` failed with CUDA 12.6 vs. torch CUDA 13.0 version mismatch (a host-environment issue, not a plan issue).
- **Fix:** Built with `FORCE_CPU=1 pip install --no-build-isolation -e .`. This compiled CPU-only extensions sufficient to satisfy `hasattr(torch.ops.torch_structured, 'butterfly_multiply')` and run the test suite.
- **Files modified:** None (no source change — just a one-off build step). The pre-existing test failures (`tests/test_multiply.py::test_input_padding_output_slicing` and `tests/test_multiply.py::test_multiply` fail with "Not compiled with CUDA support") are environment-driven, not plan-driven — they would have failed identically before any Phase 4 work touched the repo. The `test_multiply_base4.py` test (which uses the moved pure-PyTorch `butterfly_multiply_torch`) passes, confirming the move and shim work correctly.
- **Commit:** None — this was an environment setup step, not a code change. Recorded here for transparency.

No other deviations. Plan executed as written.

## Authentication Gates

None — Phase 4 is purely local Python / packaging work.

## Verification Results

| Check | Result |
|-------|--------|
| `torch_structured._torch_ref.butterfly` and `torch_structured.butterfly.multiply` re-exports of `butterfly_multiply_torch` are `is` identical | PASS |
| `torch_structured.butterfly.butterfly.butterfly_multiply_torch` (verify-only) resolves through shim | PASS |
| `_cuda_legacy/butterfly.py` has no `@torch.jit.script` decorator | PASS |
| Original `def butterfly_multiply_torch` deleted from `butterfly/multiply.py` (`grep -c '^def butterfly_multiply_torch'` returns 0) | PASS |
| `_ops.py` module docstring contains D-05 call-site contract (both WRONG and CORRECT forms verbatim) | PASS |
| `_ops._resolve('arbitrary_module_path')` raises `ValueError` with message containing `triton\|cuda\|torch\|auto` | PASS |
| `_ops._BACKEND in ('cuda', 'torch')` in Phase 4 (B3 gate: never `'triton'`) | PASS |
| `set_backend('triton')` returns `'cuda'` or `'torch'` AND emits the W4 warning | PASS |
| `set_backend('torch')` makes `_ops.butterfly_multiply is butterfly_multiply_torch` | PASS |
| `_BACKEND` reflects actual binding after `set_backend('triton')` (B3 gate: never `'triton'` in Phase 4) | PASS |
| Top-level `torch_structured.set_backend` callable; `'set_backend' in torch_structured.__all__` | PASS |
| `python -c "import torch_structured; torch_structured.set_backend('torch')"` exits 0 (no circular import) | PASS |
| `TORCH_STRUCTURED_BACKEND=torch` → `_BACKEND == 'torch'` | PASS |
| `TORCH_STRUCTURED_BACKEND=arbitrary_module` → non-zero exit with ValueError | PASS |
| `grep -c 'torch>=2.6' pyproject.toml` returns 2 | PASS |
| `grep -c 'torch>=2.0' pyproject.toml` returns 0 | PASS |
| `04-COMPLEX-LAYOUT.md` ≥40 lines + view_as_real ≥2 + view_as_complex ≥2 + IS_COMPLEX ≥1 + is_contiguous ≥1 + twiddle layout literal present | PASS (124 lines) |
| `04-DEPRECATION-PLAN.md` ≥30 lines + simplefilter ≥1 + stacklevel=2 ≥1 + DeprecationWarning ≥2 + v1.3 ≥1 + v1.4 ≥1 | PASS (147 lines) |
| `pytest tests/test_multiply.py tests/test_multiply_base4.py --collect-only -q` collects without ImportError | PASS (3 tests collected) |

## Known Stubs

None — no UI-facing placeholder data. The `_ops.py` resolver has `hadamard_transform = None` and `diag_mult = None` at module level as expected (Phase 5/6 populate); these are documented as such in the source comments and acceptance criteria. They are not stubs in the sense of "data that should be wired but isn't" — they are intentional placeholders documented in the plan.

## Self-Check: PASSED

Verified the following claims:

```
[ -f torch_structured/_torch_ref/__init__.py ] && echo FOUND
FOUND
[ -f torch_structured/_torch_ref/butterfly.py ] && echo FOUND
FOUND
[ -f torch_structured/_cuda_legacy/__init__.py ] && echo FOUND
FOUND
[ -f torch_structured/_cuda_legacy/butterfly.py ] && echo FOUND
FOUND
[ -f torch_structured/_ops.py ] && echo FOUND
FOUND
[ -f .planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/04-COMPLEX-LAYOUT.md ] && echo FOUND
FOUND
[ -f .planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/04-DEPRECATION-PLAN.md ] && echo FOUND
FOUND
git log --all --oneline | grep -q 4f37991 && echo FOUND
FOUND
git log --all --oneline | grep -q 4bb68ba && echo FOUND
FOUND
git log --all --oneline | grep -q 8ae69a5 && echo FOUND
FOUND
```
