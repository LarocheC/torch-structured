---
phase: quick-260529-bdr
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - torch_structured/_ops.py
  - torch_structured/_cuda_legacy/__init__.py
  - tests/test_diag_mult.py
  - tests/structured/test_hadamard_triton.py
  - tests/test_butterfly_triton.py
autonomous: true
requirements: [DEPR-02, TEST-01, TEST-02, TEST-03]

must_haves:
  truths:
    - "Bare `import torch_structured` on the default (triton) backend emits 0 DeprecationWarning, even when a CUDA build is present and _has_cuda_legacy() is True"
    - "Explicit set_backend('cuda') in a fresh process emits exactly 1 DeprecationWarning (D-74b intent preserved)"
    - "A routed shape (butterfly fwd, log_n=11, complex64) on the triton backend still transparently uses the CUDA path — no routing regression"
    - "The full 3-axis backend fixture {torch,triton,cuda} passes clean (0 failures) on this matched-CUDA host across test_diag_mult.py, test_hadamard_triton.py, test_butterfly_triton.py"
    - "fp32 cross-backend allclose assertions tolerate the CUDA accumulation-order noise floor (atol widened to 1e-5 for log_n>=8 in the log_n-parametrized hadamard/butterfly suites, and for n>=128 in the fixed-size diag_mult suite)"
    - "fp64 gradcheck tests skip the cuda param (legacy CUDA kernels are fp32-only) just as they already skip the triton param"
  artifacts:
    - path: "torch_structured/_cuda_legacy/__init__.py"
      provides: "Explicit warn_cuda_deprecation() emitter decoupled from module-import timing, _WARNED-gated"
      contains: "def warn_cuda_deprecation"
    - path: "torch_structured/_ops.py"
      provides: "Suppressed leak-site import (line ~329-332) + explicit warn call on cuda-selection paths"
      contains: "catch_warnings"
    - path: "tests/test_diag_mult.py"
      provides: "cuda-param fp64 skip-gates on gradcheck tests; widened fp32 atol (n>=128 -> 1e-5)"
    - path: "tests/structured/test_hadamard_triton.py"
      provides: "cuda-param fp64 skip alongside triton skip; widened fp32 atol for log_n>=8 (incl. test_hadamard_normalize log_n=10)"
    - path: "tests/test_butterfly_triton.py"
      provides: "cuda-param fp64 skip alongside triton skip"
  key_links:
    - from: "torch_structured/_ops.py:_resolve (actual=='cuda' branches)"
      to: "torch_structured/_cuda_legacy.warn_cuda_deprecation"
      via: "explicit function call on explicit-cuda-selection paths"
      pattern: "warn_cuda_deprecation\\(\\)"
    - from: "torch_structured/_ops.py:329-332 routing closure import"
      to: "warnings.catch_warnings()"
      via: "DeprecationWarning suppression wrap mirroring the per-op probes at _ops.py:127-160"
      pattern: "with warnings.catch_warnings"
    - from: "tests fp64 gradcheck sites"
      to: "pytest.skip on cuda param"
      via: "skip-gate matching the existing triton skip"
      pattern: "backend ==.*cuda"
---

<objective>
Fix the two non-blocking defects from the v1.2 milestone audit (.planning/v1.2-MILESTONE-AUDIT.md `gaps.defects`):

1. DEPR-02 warning leak — the routing-fallback closure in `_ops.py:329-332` imports `_cuda_legacy` without a `warnings.catch_warnings()` wrap, so a bare `import torch_structured` on the DEFAULT triton backend fires the CUDA DeprecationWarning whenever a CUDA build is present. This contradicts D-74b (warning reserved for explicit `set_backend('cuda')`).
2. TEST-01/02/03 cuda-axis gaps — 23 cuda-axis tests fail on matched-CUDA hardware due to (a) `atol=1e-6` below the fp32 accumulation noise floor and (b) fp64 gradcheck tests that skip the `triton` param but not the `cuda` param (legacy CUDA kernels are also fp32-only).

Purpose: Close the milestone's only outstanding defects so the 3-axis agreement gate is honestly green on matched-CUDA hardware and the deprecation warning fires exactly when D-74b intends.
Output: One production fix (DEPR-02), two-file warning-emitter refactor, and test-tolerance/skip fixes across three suites. NO version bump, NO publish.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/v1.2-MILESTONE-AUDIT.md
@.planning/STATE.md
@./CLAUDE.md

<interfaces>
<!-- Key code the executor needs. Extracted from the codebase — no exploration required. -->

CURRENT — torch_structured/_cuda_legacy/__init__.py (the module-top warning, lines 29-39):
The warning is emitted at MODULE IMPORT, gated by a module-level `_WARNED` flag (NOT
`simplefilter("once")`, deliberately — see the module comment lines 16-28). Once any
import of `_cuda_legacy` runs the module body, `_WARNED` becomes True and the warning
never fires again for the process. `from .butterfly/.diag_mult/.hadamard import ...`
follow at lines 41-43.

THE TRAP: if the leak-site import (below) runs FIRST under a `catch_warnings()/ignore`
wrap, the module body executes while suppressed and sets `_WARNED=True`. A LATER explicit
`set_backend('cuda')` re-imports `_cuda_legacy`, but the module is already in `sys.modules`,
so the body does NOT re-run — and no warning fires. Suppressing the leak site alone would
SILENCE the explicit-cuda path too. This is the nuance the fix must resolve.

CURRENT — torch_structured/_ops.py the leak site (lines 329-332, inside the
`actual == "triton"` + `_has_triton_kernel("butterfly_multiply")` + `_has_cuda_legacy()`
branch — the DEFAULT path on this host):
```
            if _has_cuda_legacy():
                from torch_structured._cuda_legacy import (
                    butterfly_multiply as _cuda_bm_for_route,
                )
                def _routed_butterfly_multiply(twiddle, input_, *args, **kwargs):
                    ...
```

CURRENT — torch_structured/_ops.py the explicit-cuda selection paths (these SHOULD warn):
- line 380-382: `elif actual == "cuda": from torch_structured._cuda_legacy import butterfly_multiply as _cuda_bm`
- line 396-399: `elif actual == "cuda" and _has_cuda_legacy_diag_mult(): from ...diag_mult import diag_mult`
- line 415-417: `elif actual == "cuda" and _has_cuda_legacy_hadamard(): from ...hadamard import hadamard_transform`
`actual` becomes "cuda" only via `set_backend('cuda')` (line 292-294) or `auto` with no triton (line 271-272).

EXISTING per-op probe suppression pattern to MIRROR (torch_structured/_ops.py:133-139):
```
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            from torch_structured._cuda_legacy.diag_mult import HAS_CUDA_LEGACY_DIAG_MULT
            return HAS_CUDA_LEGACY_DIAG_MULT
        except ImportError:
            return False
```

EXISTING atol-widening pattern to MIRROR (tests/test_phase9_integration.py:357):
```
    atol = _FP32_ATOL if log_n < 8 else 1e-5
    assert torch.allclose(out, expected, rtol=_FP32_RTOL, atol=atol), ...
```

EXISTING triton fp64 skip pattern to MIRROR (tests/test_butterfly_triton.py:178-182,
tests/structured/test_hadamard_triton.py:69-73):
```
    if backend == "triton":
        pytest.skip("Triton kernel is fp32-only per D-41; fp64 gradcheck covered on ...")
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: DEPR-02 — suppress the leak-site import and make explicit cuda-selection warn exactly once</name>
  <files>torch_structured/_cuda_legacy/__init__.py, torch_structured/_ops.py</files>
  <action>
    Fix the DeprecationWarning leak so it fires ONLY on explicit cuda selection, never on the default-backend import. The naive "wrap line 329-332 in catch_warnings" is INSUFFICIENT on its own because of the `_WARNED` trap documented in `<interfaces>`: a suppressed first import sets `_WARNED=True`, silencing the later explicit path. Decouple the user-facing warning from module-import timing.

    In `torch_structured/_cuda_legacy/__init__.py`: Refactor the module-top warning (current lines 29-39) into an explicit, idempotent emitter function `warn_cuda_deprecation()` that performs the `_WARNED`-gated `warnings.warn(...)` (keep the exact same message string, `DeprecationWarning` category, and `stacklevel=2`; preserve the `_WARNED` once-per-process semantics). Do NOT emit the warning at module import time anymore — importing `_cuda_legacy` (e.g., the suppressed leak-site import, or a per-op probe) must be side-effect-free w.r.t. the warning. Export `warn_cuda_deprecation` in `__all__`. Keep the explanatory comment block updated to reflect the new emit-on-explicit-selection design (replace the now-stale "once the user-facing path emits the warning" import-time rationale with the function-based rationale). This is the sanctioned `_cuda_legacy` warning site per CLAUDE.md — keep `warnings` usage confined here and at the existing `_ops.py` probe sites.

    In `torch_structured/_ops.py`: (a) Wrap the leak-site import at lines 329-332 (`from torch_structured._cuda_legacy import butterfly_multiply as _cuda_bm_for_route`) in `with warnings.catch_warnings(): warnings.simplefilter("ignore", DeprecationWarning)`, mirroring the per-op probe pattern at lines 133-139. (b) On EACH explicit-cuda-selection binding path, call `torch_structured._cuda_legacy.warn_cuda_deprecation()` so the warning fires once when the user actually selects cuda: the `elif actual == "cuda"` butterfly branch (~line 380), the `elif actual == "cuda" and _has_cuda_legacy_diag_mult()` diag_mult branch (~line 396), and the `elif actual == "cuda" and _has_cuda_legacy_hadamard()` hadamard branch (~line 415). The `_WARNED` gate inside `warn_cuda_deprecation()` guarantees at-most-once per process even though all three may fire in a single `_resolve('cuda')` call. Add a short comment at each call site referencing D-74b and this defect (DEPR-02-leak). Do NOT alter routing/closure behavior — `_routed_butterfly_multiply` must keep delegating to the CUDA path on routed cells exactly as before.
  </action>
  <verify>
    <automated>cd "$(git rev-parse --show-toplevel)" && python -c "import subprocess,sys; r=subprocess.run([sys.executable,'-W','error::DeprecationWarning','-c','import torch_structured'],capture_output=True,text=True); print('DEFAULT-IMPORT-RC',r.returncode); print(r.stderr[-800:]); sys.exit(r.returncode)"</automated>
    <automated>cd "$(git rev-parse --show-toplevel)" && python -c "import subprocess,sys,textwrap; child=textwrap.dedent('''
import warnings, torch_structured
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter(\"always\")
    torch_structured.set_backend(\"cuda\")
dep=[x for x in w if issubclass(x.category, DeprecationWarning)]
print(\"CUDA-WARN-COUNT\", len(dep))
assert len(dep)==1, f\"expected exactly 1 DeprecationWarning, got {len(dep)}\"
'''); r=subprocess.run([sys.executable,'-c',child],capture_output=True,text=True); print(r.stdout); print(r.stderr[-800:]); sys.exit(r.returncode)"</automated>
  </verify>
  <done>
    - Subprocess `python -W error::DeprecationWarning -c "import torch_structured"` exits 0 (no DeprecationWarning raised on default backend with CUDA present).
    - A FRESH subprocess (separate process, mirroring the first verify) that imports torch_structured then calls `set_backend('cuda')` under `warnings.catch_warnings(record=True)` records exactly 1 DeprecationWarning. (Must be process-isolated: an in-process check can false-pass if import ever routes to cuda and sets `_WARNED` before the count window.)
    - `warn_cuda_deprecation` exists in `_cuda_legacy/__init__.py`, is `_WARNED`-gated, and is exported in `__all__`.
    - The leak-site import (_ops.py ~329-332) is wrapped in `warnings.catch_warnings()`.
    - Routing closure behavior unchanged (verified by Task 3 regression check).
  </done>
</task>

<task type="auto">
  <name>Task 2: TEST-01/02/03 — widen fp32 atol and add cuda-param fp64 skip-gates in the three per-op suites</name>
  <files>tests/test_diag_mult.py, tests/structured/test_hadamard_triton.py, tests/test_butterfly_triton.py</files>
  <action>
    Make the 3-axis backend fixture {torch,triton,cuda} pass clean on matched-CUDA hardware. Two test-side fixes (the CUDA kernels are numerically correct — do NOT touch kernel code). Locate the EXACT failing assertions/skip-sites in each suite and fix them; prefer the consistent inline pattern already established over scattershot edits.

    FIX (a) — fp32 atol widening. NOTE: two DIFFERENT threshold rules apply, because the suites differ in how they parametrize size. Do not conflate them:

      - `tests/test_diag_mult.py` (FIXED-size suite — no `log_n` parametrize): apply the UNAMBIGUOUS size-based rule `n >= 128 -> atol = 1e-5`, else keep `atol = 1e-6`. This is diag_mult's OWN threshold (the audit measured ~2.3e-5 abs error at N=128 due to its summation depth) — it is NOT the `log_n >= 8` rule and must NOT be framed as "mirrors log_n>=8". Concretely: the fp32 forward assert in `test_diag_mult_eager_fp32` (line 34, N=128) gets `atol=1e-5`; the `test_diag_mult_shift_grid` fp32 forward assert (line 93, N=16) keeps `atol=1e-6` because N=16 < 128. If any other diag_mult fp32 cross-backend allclose runs at n>=128, apply the same `n >= 128 -> 1e-5` rule there too.

      - `tests/structured/test_hadamard_triton.py` (log_n-PARAMETRIZED suite): apply the phase9 rule `atol = 1e-6 if log_n < 8 else 1e-5`. Apply to `test_hadamard_eager_fp32` (line 41, log_n in {2..12}) and `test_hadamard_normalize` (line 47, runs at log_n=10 on the backend-parametrized fixture with `atol=1e-6` — a likely cuda-axis failure; widen it under the same log_n>=8 rule). Also review the rank-3 fp32 sites — `test_hadamard_module_consumer` (line 113), `test_hadamard_eager_fp32_rank3` (line 142), `test_hadamard_backward_rank3` (line 172), and `test_hadamard_self_inverse_rank3` (line 200) — and widen only the ones whose size reaches log_n>=8 and that fail on the cuda axis.

      - `tests/test_butterfly_triton.py`: already uses module-level `RTOL=ATOL=1e-3` (line 55-56) which is above the noise floor — likely no atol change needed here; confirm via the run and only adjust if a specific cuda-axis fp32 assertion still fails.

    Use the exact size-dependent form (a local `atol` variable computed per-parametrize), not a blanket loosening — the goal is to tolerate accumulation-order drift while still rejecting real bugs (>1e30-class errors). For hadamard/butterfly use the `log_n`-keyed form; for diag_mult use the `n`-keyed form.

    FIX (b) — cuda-param fp64 skip-gate, mirroring the existing triton skip at `test_butterfly_triton.py:178-182` and `test_hadamard_triton.py:69-73`. Legacy CUDA kernels are fp32-only (same as Triton), so fp64 gradcheck on the `cuda` axis raises `"..._cuda" not implemented for 'Double'`. Add a matching `cuda` skip wherever the `triton` fp64 skip exists, and add one where the triton skip is ABSENT but a cuda fp64 failure occurs:
      - `tests/test_butterfly_triton.py:test_butterfly_gradcheck_fp64` (~line 168): extend the existing `if backend == "triton": pytest.skip(...)` to also skip `cuda` (e.g., `if backend in ("triton", "cuda"): pytest.skip("...fp32-only kernels; fp64 gradcheck covered on torch backend")`).
      - `tests/structured/test_hadamard_triton.py:test_hadamard_gradcheck_fp64` (~line 60): same — extend the triton skip to cover `cuda`.
      - `tests/test_diag_mult.py`: the fp64 gradcheck tests `test_diag_mult_gradcheck_fp64_real` (~line 52), `test_diag_mult_gradcheck_fp64_complex` (~line 65), and the fp64 backward block inside `test_diag_mult_shift_grid` (~lines 99-115) have NO triton skip currently (triton passes because `_ops.diag_mult` routes fp64 through the torch_ref oracle). On the `cuda` axis these bind to the fp32-only `_diag_mult_cuda` kernel and fail on Double. Add a `cuda`-axis fp64 skip to the two dedicated gradcheck tests, and guard the fp64 backward section of `test_diag_mult_shift_grid` so the `cuda` axis skips the fp64 backward portion (keep its fp32 forward portion running on cuda). Match the existing skip-reason phrasing style (cite that legacy CUDA is fp32-only).
    Add a brief comment at each new skip referencing the audit defect (TEST-03-cuda-axis) and that legacy CUDA kernels are fp32-only. Do not change the torch/triton behavior of any test.
  </action>
  <verify>
    <automated>cd "$(git rev-parse --show-toplevel)" && python -m pytest tests/test_diag_mult.py tests/structured/test_hadamard_triton.py tests/test_butterfly_triton.py -q 2>&1 | tail -25</automated>
  </verify>
  <done>
    - `pytest tests/test_diag_mult.py tests/structured/test_hadamard_triton.py tests/test_butterfly_triton.py` reports 0 failures with the cuda axis live (skips for genuinely-unsupported fp64-cuda and multigpu are acceptable).
    - diag_mult fp32 cross-backend assertions at n>=128 (e.g. test_diag_mult_eager_fp32, N=128) now pass with atol=1e-5; n<128 sites (e.g. test_diag_mult_shift_grid fp32 forward at N=16) retain atol=1e-6.
    - hadamard fp32 cross-backend assertions that previously failed at log_n>=8 now pass with atol=1e-5, INCLUDING `test_hadamard_normalize` (line 47, log_n=10); log_n<8 sites retain atol=1e-6.
    - fp64 gradcheck tests skip the `cuda` param (and still skip `triton` where they already did); torch-axis fp64 gradcheck still runs and passes.
  </done>
</task>

<task type="auto">
  <name>Task 3: Verification — no routing regression and full 3-axis gate green</name>
  <files>(verification only — no source edits expected)</files>
  <action>
    Confirm both fixes hold together with no regression. This task runs the must_haves end-to-end. If any check fails, return to Task 1 or Task 2 to correct (do NOT relax a must_have to make a check pass — the warning semantics and the kernel-correctness atol floor are load-bearing). Specifically prove: (1) the routed butterfly cell still transparently uses CUDA on the triton backend (a routed shape — butterfly fwd log_n=11 complex64 — still executes via the CUDA path, no exception, output finite), and (2) the combined suite is green. If the log_n=11 complex64 cell is not actually marked route_to_cuda in `torch_structured/_routing.json`, pick any cell the routing table DOES mark `route_to_cuda` for `butterfly_multiply ... forward` and exercise that shape instead — the point is that a routed cell still reaches the CUDA path after the leak-site suppression. Do NOT add a version bump or any publish step.
  </action>
  <verify>
    <automated>cd "$(git rev-parse --show-toplevel)" && python -W error::DeprecationWarning -c "import torch_structured; print('OK default import, no DeprecationWarning')"</automated>
    <automated>cd "$(git rev-parse --show-toplevel)" && python -c "
import torch, torch_structured
from torch_structured import _ops
_ops.set_backend('triton')
log_n=11; n=1<<log_n
tw=torch.randn(1,1,log_n,n//2,2,2,device='cuda',dtype=torch.complex64)
x=torch.randn(1,1,n,device='cuda',dtype=torch.complex64)
out=_ops.butterfly_multiply(tw,x,True,n)
assert torch.isfinite(out.abs()).all(), 'non-finite routed output'
print('OK routed cell executes, backend=', _ops._BACKEND, 'shape=', tuple(out.shape))
"</automated>
    <automated>cd "$(git rev-parse --show-toplevel)" && python -m pytest tests/test_diag_mult.py tests/structured/test_hadamard_triton.py tests/test_butterfly_triton.py -q 2>&1 | tail -5</automated>
  </verify>
  <done>
    - Default import under `-W error::DeprecationWarning` exits 0.
    - A routed butterfly cell on the triton backend executes via the CUDA path producing a finite output of the expected shape (no regression from the leak-site suppression).
    - Combined 3-axis suite is green (0 failures); acceptable skips only for fp64-cuda and multigpu.
    - pyproject/__init__ version unchanged (still 1.2.1); no publish performed.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| env var → backend resolver | `TORCH_STRUCTURED_BACKEND` / `set_backend(name)` is validated against the fixed set {triton,cuda,torch,auto} in `_resolve` (existing T-04-01 mitigation); this change adds no new untrusted input |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-bdr-01 | Tampering | `_cuda_legacy.warn_cuda_deprecation()` refactor | accept | Pure-local emitter, no input, no I/O; `_WARNED` gate prevents warning-spam DoS. No attack surface change vs. the prior module-import warning. |
| T-bdr-02 | Information disclosure | leak-site `catch_warnings()` suppression | mitigate | Suppression is scoped to a `with` block around a single import (mirrors the audited per-op probe pattern); it does not globally mutate `warnings.filters`, so it cannot silence unrelated warnings elsewhere in the process. |
| T-bdr-03 | Denial of service | test-tolerance widening | accept | atol widened only to the documented fp32 noise floor (1e-5 at log_n>=8 / n>=128); real-bug errors (>1e30-class) still rejected — the gate stays meaningful. |
</threat_model>

<verification>
- DEPR-02 (a): subprocess default import under `-W error::DeprecationWarning` → exit 0.
- DEPR-02 (b): fresh-process (subprocess-isolated) explicit `set_backend('cuda')` → exactly 1 recorded DeprecationWarning.
- Routing regression: routed butterfly cell on triton backend still reaches the CUDA path, output finite.
- TEST-01/02/03: `pytest tests/test_diag_mult.py tests/structured/test_hadamard_triton.py tests/test_butterfly_triton.py` → 0 failures on the matched-CUDA host (cuda axis live).
- Scope guard: pyproject.toml and torch_structured/__init__.py version strings unchanged (1.2.1); no publish.
</verification>

<success_criteria>
- Both audit defects (DEPR-02-leak, TEST-03-cuda-axis) closed.
- Zero DeprecationWarning on default-backend import; exactly one on explicit cuda selection.
- 3-axis backend agreement gate honestly green on matched-CUDA hardware.
- No version bump, no publish; changes confined to the 5 files in `files_modified`.
- Commit trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
</success_criteria>

<output>
After completion, create `.planning/quick/260529-bdr-fix-two-v1-2-audit-defects-depr-02-warni/260529-bdr-SUMMARY.md` (include a `status:` frontmatter field — the audit flagged prior quick SUMMARYs for omitting it).
</output>
