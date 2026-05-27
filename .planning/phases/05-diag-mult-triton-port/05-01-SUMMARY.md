---
phase: 05-diag-mult-triton-port
plan: 01
subsystem: triton-port
tags: [triton, autograd, complex64, gradcheck, dispatch, wirtinger, register_fake]

requires:
  - phase: 04-triton-dispatch-infrastructure-foundational-decisions
    provides: "Phase 4 demonstrator skeleton (kernel + @triton_op + _setup_context + _backward + register_fake), _ops.py resolver with per-op honest probes, 04-COMPLEX-LAYOUT.md (view_as_real boundary + 4-FMA IS_COMPLEX template), tests/conftest.py backend fixture (param=['torch'])"
provides:
  - "torch_structured._ops.diag_mult: single dispatch-bound callable for cycle_mult primitive (subdiag, v, shift_subdiag, shift_v)"
  - "torch_structured._torch_ref.diag_mult: fp64-capable pure-PyTorch oracle used as register_autograd backward callback and as D-22 fallback"
  - "torch_structured._triton.diag_mult.op: @triton.jit cycle_mult kernel (2-D grid) + @triton_op wrapper + Wirtinger-correct register_autograd backward + register_fake meta kernel"
  - "torch_structured._cuda_legacy.diag_mult: try-import passthrough with HAS_CUDA_LEGACY_DIAG_MULT sentinel"
  - "torch_structured._ops._has_cuda_legacy_diag_mult(): symmetric D-22 honest probe"
  - "torch_structured._ops._has_any_triton_kernel(): BLOCKER-1 fix widening _resolve Step 1"
  - "tests/test_diag_mult.py: 5 test functions × backend fixture = 26 tests covering eager fp32, eager complex64, gradcheck fp64 real + complex, shift grid"
  - "tests/test_dispatch.py: 3 cross-cutting smoke tests (round-trip, ValueError, B3 probe regression)"
affects: [phase-06-hadamard-triton-port, phase-07-butterfly-triton-port, phase-08-butterfly-backward, phase-09-integration-hardening, phase-10-deprecation]

tech-stack:
  added: []  # No new dependencies; uses torch.library.triton_op + torch.library.wrap_triton from Phase 4
  patterns:
    - "_triton/<op>/op.py layout: kernel + @triton_op wrapper + _setup_context + _backward + @register_fake (single source, IS_COMPLEX constexpr branch)"
    - "_torch_ref/<op>.py: pure-PyTorch oracle for gradcheck + D-22 fallback (assert preconditions, no try/except)"
    - "_cuda_legacy/<op>.py: top-of-module try-import + HAS_<NAME> sentinel + defensive RuntimeError (documented exception to no-try/except rule)"
    - "Resolver Step 2 per-op three-branch binding: actual=triton + per-op honest probe → triton; actual=cuda + _has_cuda_legacy_<op>() → cuda; else → _torch_ref (+ log.warning when actual=cuda fallback)"
    - "Wirtinger-correct backward formula: grad_a = ref(grad_out, b.conj(), ...); grad_b = ref(a.conj(), grad_out, ...) — .conj() is no-op for real so single path handles both"
    - "Broadcast-sum in _backward: explicit grad.sum(dim=tuple(range(ndims_to_sum))) for 1-D operand broadcast over batch (Pitfall 6)"

key-files:
  created:
    - "torch_structured/_torch_ref/diag_mult.py (55 lines): pure-PyTorch oracle, torch.roll(s, -ss) * torch.roll(v, -sv)"
    - "torch_structured/_triton/diag_mult/__init__.py (4 lines): package re-export"
    - "torch_structured/_triton/diag_mult/op.py (205 lines): @triton.jit _cycle_mult_kernel + @triton_op diag_mult + register_autograd + register_fake"
    - "torch_structured/_cuda_legacy/diag_mult.py (45 lines): try-import + HAS_CUDA_LEGACY_DIAG_MULT sentinel + thin pass-through"
    - "tests/test_diag_mult.py (118 lines): 5 test functions × backend fixture × shift grid = 26 tests"
  modified:
    - "torch_structured/_torch_ref/__init__.py: re-export diag_mult"
    - "torch_structured/_cuda_legacy/__init__.py: re-export diag_mult"
    - "torch_structured/_ops.py: +_has_cuda_legacy_diag_mult, +_has_any_triton_kernel (BLOCKER-1 fix), per-op diag_mult binding in _resolve Step 2 with D-22 fallback, widened butterfly_multiply triton branch fallback, per-op log.info line; deleted Phase 4 demonstrator (-102 / +76 net)"
    - "torch_structured/structured/krylov.py: -CycleDownMultCuda class, -cycle_down_mult alias, -try-import of _diag_mult_cuda, +import torch_structured, subdiag_linear_map_cuda rewritten to call torch_structured._ops.diag_mult (D-05 attribute access)"
    - "tests/conftest.py: backend fixture params=['torch', 'triton'] with _has_triton_kernel('diag_mult') skip-gate"
    - "tests/test_dispatch.py: deleted 5 demonstrator tests; added 3 smoke tests (round-trip, ValueError, B3 probe regression)"

key-decisions:
  - "BLOCKER-1 fix applied: _has_any_triton_kernel() iterates ('butterfly_multiply', 'diag_mult', 'hadamard_transform') and is used in both _resolve Step 1 auto and triton branches (replaces hardcoded _has_triton_kernel('butterfly_multiply')). Without this, SC#1 env-var triton contract was unreachable in Phase 5 since butterfly Triton kernel doesn't land until Phase 7."
  - "D-22a coarse-global _BACKEND retained per RESEARCH recommendation A; per-op visibility provided via log.info line printing actual per-op bindings on every _resolve() call"
  - "Resolver Step 2 butterfly_multiply triton branch widened with fallback: if _has_triton_kernel('butterfly_multiply') is False (the Phase 5 reality) the resolver now falls back to cuda or torch instead of attempting the (failing) import. Necessary auto-fix to avoid breaking the existing butterfly tests once actual=='triton' becomes reachable (which the BLOCKER-1 fix enables)."
  - "Wirtinger .conj() on the OTHER operand in _backward — load-bearing for complex64 gradcheck; verified numerically PASS in tests/test_diag_mult.py::test_diag_mult_gradcheck_fp64_complex[torch+triton]"
  - "register_fake meta kernel kept verbatim from Phase 4 demonstrator pattern (returns torch.empty_like(v)) — the literal 260419-p27 fix"

patterns-established:
  - "Phase 6 (hadamard) and Phase 7 (butterfly) should transcribe _triton/<op>/op.py from _triton/diag_mult/op.py: imports, kernel skeleton, wrapper boundary, _setup_context/_backward, register_fake"
  - "Per-op Triton kernels light up _has_any_triton_kernel() progressively; no further Step 1 changes needed in Phases 6/7"
  - "Per-op resolver branch in _resolve Step 2 follows the same 3-arm shape (triton if per-op probe / cuda if cuda probe / else torch with optional warning)"
  - "tests/test_<op>.py uses backend fixture from conftest + attribute access through torch_structured._ops.<op>; pytestmark = pytest.mark.skipif(not torch.cuda.is_available())"

requirements-completed: [TRI-01]

duration: 14min
completed: 2026-05-27
---

# Phase 5 Plan 1: diag_mult Triton Port Summary

**Triton-backed cycle_mult primitive (subdiag, v, shift_subdiag, shift_v) with Wirtinger-correct complex64 backward, replacing the legacy pybind11 _diag_mult_cuda extension and proving the Phase 4 dispatch + register_autograd plumbing end-to-end.**

## Performance

- **Duration:** ~14 min
- **Started:** 2026-05-27T13:57:39Z
- **Completed:** 2026-05-27T14:12:29Z
- **Tasks:** 7
- **Files created:** 5
- **Files modified:** 6

## Accomplishments

- `torch_structured._ops.diag_mult` is now a single dispatch-bound callable, rebindable across `torch` / `triton` / `cuda` backends via `set_backend()` or `TORCH_STRUCTURED_BACKEND` env var; SC#1 literal contract verified
- fp64 `gradcheck` passes for **both** real and complex128 inputs (D-26 Wirtinger acceptance gate — the load-bearing test that fails with errors ~2.0 if `.conj()` is missing from `_backward`)
- `structured/krylov.py` no longer depends on the legacy `_diag_mult_cuda` pybind11 extension; the custom `torch.autograd.Function` is deleted in favor of `register_autograd` flowing through `_ops.diag_mult`
- Phase 4 demonstrator (`_demo_identity_*` block, ~90 lines) fully removed from `_ops.py` per D-13/D-27 — diag_mult now exercises the same skeleton on a real kernel
- All 26 `test_diag_mult.py` tests pass (5 tests × 2 backends, plus 18 shift-grid combinations) plus the 3 cross-cutting `test_dispatch.py` smoke tests
- **BLOCKER-1 iteration-2 fix applied**: `_has_any_triton_kernel()` widens `_resolve()` Step 1 so the `triton`/`auto` branches reach `actual="triton"` as soon as any per-op Triton kernel ships, not gated on butterfly (Phase 7). Without this fix SC#1 was unreachable in Phase 5.

## Task Commits

Each task was committed atomically:

1. **Task 1: _torch_ref/diag_mult.py pure-PyTorch oracle** — `4ea14f1` (feat)
2. **Task 2: _triton/diag_mult Triton kernel + autograd op** — `f13350c` (feat)
3. **Task 3: _cuda_legacy/diag_mult.py try-import passthrough** — `668ad96` (feat)
4. **Task 4: _ops.py resolver extension + demonstrator deletion** — `4283371` (refactor)
5. **Task 5: krylov.py consumes _ops.diag_mult; delete legacy autograd path** — `5eb36ac` (refactor)
6. **Task 6: tests/conftest.py backend fixture widening** — `011a94e` (test)
7. **Task 7: tests/test_diag_mult.py + trim tests/test_dispatch.py** — `33a2900` (test)

## Files Created/Modified

### Created
- `torch_structured/_torch_ref/diag_mult.py` (55 lines): pure-PyTorch oracle, formula `torch.roll(subdiag, -shift_subdiag) * torch.roll(v, -shift_v)`. Used as gradcheck oracle, register_autograd backward callback, and D-22 fallback.
- `torch_structured/_triton/diag_mult/__init__.py` (4 lines): package re-export of `diag_mult` symbol.
- `torch_structured/_triton/diag_mult/op.py` (205 lines): `@triton.jit _cycle_mult_kernel` (2-D grid `(n_batch, cdiv(N, BLOCK_SIZE))` with `IS_COMPLEX: tl.constexpr` 4-FMA branch); `@triton_op("torch_structured::diag_mult")` wrapper with view_as_real boundary; `register_autograd` Wirtinger-correct `_backward` with broadcast-sum fix for 1-D subdiag; `register_fake` meta kernel returning `torch.empty_like(v)`.
- `torch_structured/_cuda_legacy/diag_mult.py` (45 lines): top-of-module try-import of `_diag_mult_cuda`; `HAS_CUDA_LEGACY_DIAG_MULT: bool` sentinel; thin pass-through to `_diag_mult_cuda_module.cycle_mult` with defensive RuntimeError when `.so` absent.
- `tests/test_diag_mult.py` (118 lines): 5 test functions parametrized over `backend` fixture (26 tests total covering eager fp32, eager complex64, gradcheck fp64 real, gradcheck fp64 complex, and `{-1,0,1}^2` shift grid × 2 backends).

### Modified
- `torch_structured/_torch_ref/__init__.py`: re-export `diag_mult`.
- `torch_structured/_cuda_legacy/__init__.py`: re-export `diag_mult`.
- `torch_structured/_ops.py` (-102 / +76 lines net): added `_has_cuda_legacy_diag_mult()` probe + `_has_any_triton_kernel()` helper; widened `_resolve()` Step 1 to use `_has_any_triton_kernel()` in both auto and triton branches (BLOCKER-1 fix); widened the butterfly_multiply Step 2 triton branch to fall back to cuda/torch when its per-op probe is False (necessary now that actual='triton' is reachable in Phase 5); added per-op three-branch diag_mult binding with D-22 fallback warning; added per-op log.info line; **deleted Phase 4 demonstrator** (`_demo_identity_kernel`, `_demo_identity_op`, `_setup_context`, `_backward`, `register_fake` shim — lines 216-304).
- `torch_structured/structured/krylov.py`: deleted top-of-file try-import of `_diag_mult_cuda`; deleted `CycleDownMultCuda(torch.autograd.Function)` class + `cycle_down_mult = CycleDownMultCuda.apply` alias; added `import torch_structured` for D-05 attribute access; rewrote `subdiag_linear_map_cuda` lambda to call `torch_structured._ops.diag_mult(subdiag_extended, v, 0, -1)`.
- `tests/conftest.py`: `params=["torch", "triton"]` with `pytest.skip` on triton when `_has_triton_kernel("diag_mult")` is False.
- `tests/test_dispatch.py`: deleted 5 demonstrator-specific tests; added 3 cross-cutting smoke tests (`test_set_backend_round_trip`, `test_unknown_backend_raises_value_error`, `test_has_triton_kernel_probe_returns_bool`); removed GPU-only `pytestmark` (smoke tests do not need CUDA).

## Acceptance Gate Results

All Phase 5 acceptance gates from the plan's `<verification>` section pass:

| Gate | Result |
|------|--------|
| SC#1 env-var triton path (`TORCH_STRUCTURED_BACKEND=triton`) | PASS — `_ops.diag_mult is _triton.diag_mult.op.diag_mult`, `_BACKEND == "triton"` |
| SC#1 eager fp32 + complex64 (backend × dtype = 4 tests) | 4 PASS |
| SC#2 fp64 gradcheck real + complex (backend × dtype = 4 tests) | 4 PASS (Wirtinger gate confirmed) |
| SC#3a `grep -c "torch_structured._ops.diag_mult" krylov.py` | 2 (>=1 required) |
| SC#3b krylov no legacy refs (`CycleDownMultCuda`/`cycle_down_mult`/`_diag_mult_cuda` try-import) | 0 |
| Per-op resolver smoke tests (`test_dispatch.py`) | 3 PASS |
| Demonstrator deletion (`grep -v '^#' _ops.py \| grep -c _demo_identity`) | 0 |
| `_demo_identity_op` import raises | PASS |
| Shift grid `{-1,0,1}^2` × backend (18 tests) | 18 PASS |

**Total Phase 5 tests:** 29 (26 diag_mult + 3 dispatch), 0 failures, 0 skips on this CUDA host.

## D-22 Per-Op Asymmetry Observation

On this dev workstation:
- `_butterfly.so` is loaded → `_has_cuda_legacy()` returns True → `set_backend("cuda")` binds `butterfly_multiply = _cuda_legacy.butterfly_multiply`.
- `_diag_mult_cuda.so` is NOT built → `_has_cuda_legacy_diag_mult()` returns False → `set_backend("cuda")` for diag_mult falls back to `_torch_ref.diag_mult` and emits `log.warning("set_backend('cuda') requested but _diag_mult_cuda not built; falling back to torch_ref for diag_mult (D-22)")`.

The per-op `log.info` line printed on every `_resolve()` call confirms this asymmetry honestly: `torch_structured: per-op bindings: butterfly_multiply=cuda, diag_mult=torch` when `actual="cuda"`. Verified working in `set_backend('cuda')` test path of Task 4.

## Per-Op log.info Behavior

The `log.info("torch_structured: per-op bindings: butterfly_multiply=%s, diag_mult=%s", actual, _diag_mult_backend)` line fires on:
- Module import (via `_resolve(_initial)` at line 277)
- Every `set_backend()` call (via `_resolve(name)`)

Verified observable when running pytest with `-s --log-cli-level=INFO`.

## Decisions Made

- **BLOCKER-1 fix applied verbatim from iteration-2 plan revision.** `_has_any_triton_kernel()` is the load-bearing widening that makes SC#1 reachable in Phase 5 (without it, the Step 1 predicate's hardcoded `_has_triton_kernel("butterfly_multiply")` is False until Phase 7, so `actual == "triton"` is unreachable). The plan's verify gates (Task 4 gates 3 and the env-var subprocess test) caught this exactly as intended.
- **Resolver Step 2 butterfly_multiply triton branch widened** (deviation, see below) — necessary correctness fix once `actual=='triton'` becomes reachable.
- **Wirtinger `.conj()` placement** verified empirically: fp64 complex128 gradcheck PASSES on both `torch` and `triton` backends. The `.conj()` is no-op for real so a single code path handles both gradcheck modes (D-26 unified design works as planned).
- **Coarse `_BACKEND` global retained** (D-22a recommendation A from RESEARCH lines 542-574): added per-op `log.info` line instead of restructuring to a `_BACKENDS: dict` per-op map. Lower-maintenance, equivalent observability for the user; Phase 7 may revisit when butterfly + hadamard + diag_mult asymmetry becomes more common in practice.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Widened butterfly_multiply Step 2 triton branch with cuda/torch fallback**
- **Found during:** Task 4 (resolver refactor)
- **Issue:** Once BLOCKER-1 fix made `actual == "triton"` reachable in Phase 5 (when `_has_any_triton_kernel()` is True via diag_mult), the existing `if actual == "triton":` block at original `_ops.py:161-168` would unconditionally try `from torch_structured._triton.butterfly.op import butterfly_multiply` — which does NOT exist until Phase 7. This would raise `ImportError` and crash `set_backend("triton")` / `TORCH_STRUCTURED_BACKEND=triton` on every CUDA host running Phase 5.
- **Fix:** Widened the Step 2 triton branch to a 3-arm conditional matching the diag_mult pattern: `if _has_triton_kernel("butterfly_multiply"): from _triton... elif _has_cuda_legacy(): from _cuda_legacy... else: from _torch_ref...`. Now butterfly_multiply gracefully falls back to its own legacy/torch_ref path when the per-op Triton kernel isn't installed yet.
- **Files modified:** `torch_structured/_ops.py` (Step 2 butterfly branch, ~10 line addition)
- **Verification:** `TORCH_STRUCTURED_BACKEND=triton python -c "import torch_structured; ..."` succeeds and `torch_structured._ops.butterfly_multiply` is bound to the cuda legacy implementation (since `_butterfly.so` is loaded on this dev host) while `_ops.diag_mult` is bound to the Triton kernel — the correct per-op honest binding.
- **Committed in:** `4283371` (Task 4)

**Total deviations:** 1 auto-fixed (Rule 1 bug — necessary correctness fix exposed by the BLOCKER-1 widening).
**Impact on plan:** Auto-fix essential for correctness; without it the BLOCKER-1 fix would have shipped a worse bug (silent ImportError crash on `set_backend("triton")`). No scope creep — the fix preserves all existing butterfly_multiply behaviors (Phase 4 acceptance gates still pass because butterfly continues to bind to the legacy .so via the `elif _has_cuda_legacy()` arm).

## Issues Encountered

- **Pre-existing test failures (not introduced by Phase 5):** `tests/test_butterfly.py` (5 tests), `tests/test_multiply.py` (2 tests), `tests/test_permutation.py` (1 test) fail with `RuntimeError: a view of a leaf Variable that requires grad is being used in an in-place operation`. Verified these failures exist on the plan base commit `231a8f4` before any Phase 5 changes — they are a long-standing PyTorch compatibility issue with the legacy butterfly tests, unrelated to this phase. `tests/test_special.py` also has a pre-existing `ModuleNotFoundError: pywt` collection error.
- **Worktree had no compiled `.so` files initially.** Copied `_butterfly.cpython-313-x86_64-linux-gnu.so` and `_version.cpython-313-x86_64-linux-gnu.so` from the main repo into the worktree so the `_has_cuda_legacy()` probe returns True (necessary for the SC#3 cross-backend tests). The `.so` files are gitignored and not committed.

## User Setup Required

None — no external service configuration. The phase is internal kernel work.

## Next Phase Readiness

- **Phase 6 (hadamard Triton port)** can transcribe verbatim from `_triton/diag_mult/op.py`: the file layout (`_triton/<op>/op.py`), the five-component skeleton (kernel + `@triton_op` wrapper + `_setup_context` + `_backward` + `@register_fake`), the `IS_COMPLEX: tl.constexpr` complex branch with 4-FMA, the Wirtinger `.conj()` backward pattern, and the `_torch_ref` oracle + `_cuda_legacy` try-import pattern are all battle-tested on a real kernel.
- **Phase 7 (butterfly Triton port)** inherits the same template plus the `_has_any_triton_kernel()` helper (no further Step 1 changes needed — `_has_triton_kernel("butterfly_multiply")` simply turns True and the existing Step 2 butterfly branch lights up).
- **Per-op resolver pattern** is now established: each new op adds (a) its `_has_triton_kernel(<name>)` automatically via the existing probe, (b) optionally `_has_cuda_legacy_<op>()` if it has a legacy `.so`, (c) a per-op three-branch binding block in `_resolve()` Step 2, (d) an entry in the `_has_any_triton_kernel()` iteration tuple.
- **No blockers** for Phase 6 or downstream.

---

## Self-Check: PASSED

Files verified to exist:
- FOUND: torch_structured/_torch_ref/diag_mult.py
- FOUND: torch_structured/_triton/diag_mult/__init__.py
- FOUND: torch_structured/_triton/diag_mult/op.py
- FOUND: torch_structured/_cuda_legacy/diag_mult.py
- FOUND: tests/test_diag_mult.py
- FOUND: .planning/phases/05-diag-mult-triton-port/05-01-SUMMARY.md (this file)

Commits verified to exist on `worktree-agent-ab4e0fe4aa356e478`:
- FOUND: 4ea14f1 (Task 1)
- FOUND: f13350c (Task 2)
- FOUND: 668ad96 (Task 3)
- FOUND: 4283371 (Task 4)
- FOUND: 5eb36ac (Task 5)
- FOUND: 011a94e (Task 6)
- FOUND: 33a2900 (Task 7)

---
*Phase: 05-diag-mult-triton-port*
*Completed: 2026-05-27*
