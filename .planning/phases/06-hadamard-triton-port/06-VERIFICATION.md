---
phase: 06-hadamard-triton-port
verified: 2026-05-27T16:45:00Z
status: gaps_found
score: 12/13 must-haves verified
overrides_applied: 0
gaps:
  - truth: "SC#2 (D-31a — self-inverse): Composing hadamard_transform ∘ hadamard_transform on any (batch, 2**log_n) fp32 input yields N*u (unnormalized) or u (normalized) within fp32 noise floor; this consistency check fires loudly on any kernel sign error."
    status: partial
    reason: "Self-inverse composition is bit-equivalent in code for the (batch, n) input rank actually exercised by the test suite. The ROADMAP SC#2 wording 'on any input shape' is, however, NOT met: rank-3+ inputs to torch_structured._ops.hadamard_transform on the torch backend fail at the forward step (oracle hardcodes batch_size, n = u.shape — confirmed by execution: ValueError: too many values to unpack). On the triton backend forward works for rank-3+ (wrapper uses n_batch = u.numel() // n) but backward then fails inside register_autograd → _hadamard_transform_torch with the same ValueError. The shape contract claimed by the wrapper docstring at _triton/hadamard_transform/op.py:104 (`(*batch, n)`) is NOT honored end-to-end. This regression is silent: no test exercises rank > 2. Note this is a verbatim relocation of the pre-Phase-6 bug at structured/hadamard.py:25 — so it is not introduced by Phase 6, but neither is it fixed, and Phase 6 newly elevates it to the resolver-bound public surface."
    artifacts:
      - path: "torch_structured/_torch_ref/hadamard.py"
        issue: "Line 41 (`batch_size, n = u.shape`) hardcodes rank-2 inputs. Docstring on lines 36-39 still says `(batch_size, n)` (matches behavior); but the Triton wrapper at _triton/hadamard_transform/op.py:104 advertises `(*batch, n)` and consumes this function for backward, creating a contract mismatch."
    missing:
      - "Replace `batch_size, n = u.shape` with `n = u.shape[-1]` in torch_structured/_torch_ref/hadamard.py:41 (the rest of the body already uses `...` indexing, so the fix is a single line)."
      - "Add a rank-3 case to tests/structured/test_hadamard_triton.py::test_hadamard_eager_fp32 (or a new test_hadamard_higher_rank) and a rank-3 case to test_hadamard_self_inverse to lock the SC#2 'any input shape' contract."
      - "Optionally update the docstring on _torch_ref/hadamard.py:36-39 from `(batch_size, n)` to `(..., n)` to match the actual contract."
deferred:
  - truth: "_cuda_legacy/hadamard.py signature (u) is incompatible with the (u, normalize=False) contract used by the triton/torch bindings (CR-02 from 06-REVIEW.md)."
    addressed_in: "Phase 7+ / Phase 10"
    evidence: "Phase 10 success criteria reference CUDA backend behavior and deprecation; Phase 9 success criteria include backend-parametrized testing across {triton, cuda, torch}. The cuda branch is dormant on this host (_hadamard_cuda.so is not built); it cannot be exercised without that .so. SC#1 explicitly says `TORCH_STRUCTURED_BACKEND=triton` (not cuda), so Phase 6 SCs do not require cuda-parity. Recommend tracking via a Phase 9 follow-up to either reapply Python-side normalization at the cuda binding site or extend _cuda_legacy/hadamard.py to accept `normalize`."
---

# Phase 6: hadamard Triton Port Verification Report

**Phase Goal:** `hadamard` runs on Triton as a forward-only self-inverse kernel, proving the two-pass mixed-radix shared-memory pattern in Triton without atomics or complex
**Verified:** 2026-05-27T16:45:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (Plan must_haves)

| #   | Truth                                                                                                                                                                                                       | Status     | Evidence                                                                                                                                                                                                                                                                                                                                                              |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | SC#1: With `TORCH_STRUCTURED_BACKEND=triton`, `_ops.hadamard_transform(u, normalize=False)` matches `_torch_ref.hadamard.hadamard_transform_torch` within rtol=1e-5, atol=1e-6 across log_n ∈ {2..12}        | VERIFIED   | `test_hadamard_eager_fp32[triton-{2..12}]` 11 cases all PASS; child-process env-var binding confirmed: `_BACKEND == "triton"` AND `_ops.hadamard_transform is _triton.hadamard_transform.op.hadamard_transform`. normalize=True path also verified: `test_hadamard_normalize[triton]` PASS.                                                                              |
| 2   | SC#2: Composing `hadamard_transform ∘ hadamard_transform` yields N·u (unnormalized) or u (normalized) within fp32 noise on any (batch, 2**log_n) fp32 input — fires on any kernel sign error                | PARTIAL    | `test_hadamard_self_inverse[{torch,triton}-{8,10}]` 4 PASS (rank-2 only). Rank-3+ inputs FAIL backward (CR-01): forward works on triton (`n_batch = u.numel() // n`) but backward `_torch_ref.hadamard_transform_torch(grad_out)` raises `ValueError: too many values to unpack (expected 2)`. Confirmed by direct execution. Tests do not cover rank-3+. See Gaps. |
| 3   | SC#3: `structured/fastfood.fastfood_multiply` routes through `_ops.hadamard_transform` via D-05 attribute access and produces identical outputs on Triton vs torch within rtol=1e-5                          | VERIFIED   | `test_hadamard_module_consumer[{torch,triton}]` 2 PASS; `pytest tests/structured/` on TORCH_STRUCTURED_BACKEND=triton: 37 PASS, 2 SKIP (intentional). Grep checks: `torch_structured._ops.hadamard_transform` appears 2× in fastfood.py; `from .hadamard import hadamard_transform` removed.                                                                            |
| 4   | D-32 self-inverse register_autograd: backward routes grad_out through `_torch_ref.hadamard.hadamard_transform_torch`; propagates the normalize flag via setup_context; passes torch.autograd.gradcheck at fp64 | VERIFIED   | `test_hadamard_gradcheck_fp64[torch]` PASS; `test_hadamard_gradcheck_fp64[triton]` SKIP (kernel is fp32-only by D-31; the backward callback delegates to the same `_torch_ref` so the torch-backend gradcheck does double-duty per plan). _setup_context captures normalize at op.py:155-156; _backward uses ctx.normalize at op.py:168.                              |
| 5   | D-32c register_fake: `_hadamard_transform_fake(u, normalize=False)` returns torch.empty_like(u); FakeTensorMode traces without "data is not allocated yet" error                                                | VERIFIED   | op.py:175-190 contains `@hadamard_transform.register_fake` with default `normalize=False` (load-bearing per Phase 4 D-12). Tested implicitly through register_autograd plumbing in test_hadamard_gradcheck_fp64.                                                                                                                                                       |
| 6   | D-33/a/b/c: structured/hadamard.py no longer contains `class HadamardTransformCuda`, `hadamard_transform_cuda`, module-level binding, `use_hadamard_transform_cuda` try-import, scipy.linalg, or device      | VERIFIED   | grep counts: 0 hits for any of these names; file is now 35 lines (was 62). `grep -c "class HadamardTransformCuda\|hadamard_transform_cuda\|use_hadamard_transform_cuda" → 0`.                                                                                                                                                                                          |
| 7   | D-33d back-compat shim: structured/hadamard.py re-exports `hadamard_transform_torch` AND a callable `hadamard_transform` (via D-05 attribute access on `torch_structured._ops`)                              | VERIFIED   | `test_imports.py::test_import_structured` PASS — both are imported and `callable()`. Shim re-reads `torch_structured._ops.hadamard_transform` on every call (rebind-safe); verified by code inspection at structured/hadamard.py:25-34.                                                                                                                                |
| 8   | D-34 consumer refactor (D-05): structured/fastfood.py line 1 = `import torch_structured`; lines 8 + 10 = `torch_structured._ops.hadamard_transform(...)`                                                       | VERIFIED   | Verified by Read: line 1 is `import torch_structured`; lines 8 and 10 both call `torch_structured._ops.hadamard_transform(...)`. grep `from .hadamard import hadamard_transform` returns 0.                                                                                                                                                                            |
| 9   | D-36/a/c per-op resolver wiring: `_has_cuda_legacy_hadamard()` exists; three-branch hadamard binding in `_resolve()` Step 2; per-op log.info extended; stale comment removed                                  | VERIFIED   | _ops.py:97-107 defines `_has_cuda_legacy_hadamard()`; lines 250-266 implement the three-branch binding; line 269 includes `hadamard_transform=%s` and line 270 passes `_hadamard_transform_backend`. INFO line at runtime: `per-op bindings: butterfly_multiply=triton, diag_mult=triton, hadamard_transform=triton`. No stale Phase 6 comment found.                  |
| 10  | D-39 conftest widening: backend skip-gate uses `_has_any_triton_kernel()` (not `_has_triton_kernel('diag_mult')`)                                                                                              | VERIFIED   | tests/conftest.py:25 contains `_has_any_triton_kernel()`; `_has_triton_kernel("diag_mult")` appears 0 times.                                                                                                                                                                                                                                                          |
| 11  | D-37 test surface: tests/structured/test_hadamard_triton.py with 5 tests parametrized via the backend fixture using D-05 attribute access                                                                     | VERIFIED   | File exists; 32 collected items (eager 22 + normalize 2 + gradcheck 2 + self_inverse 4 + module_consumer 2 = 32); 31 PASS + 1 intentional SKIP. All test bodies use `torch_structured._ops.hadamard_transform(...)` (verified by grep — 5+ attribute-access call sites).                                                                                                |
| 12  | D-38 no edits to existing test_hadamard.py: the three existing tests pass unchanged via the D-33d shim                                                                                                          | VERIFIED   | `tests/structured/test_hadamard.py`: 4 PASS + 1 SKIP (the `_hadamard_cuda.so` probe correctly skips since .so absent); tests/structured/test_imports.py: 2 PASS. File contents unchanged (uses `from torch_structured.structured.hadamard import hadamard_transform_torch` which the shim provides).                                                                    |
| 13  | (Plan SC#2 truth restated above) Self-inverse on any input shape                                                                                                                                              | PARTIAL    | Same evidence as Truth #2 — listed separately because ROADMAP SC#2 says "on any input shape". Strict reading not honored; lenient (rank-2-only) reading honored.                                                                                                                                                                                                       |

**Score:** 12/13 truths verified (one PARTIAL — Truth #2 / SC#2 strict "any input shape" interpretation)

### Deferred Items

Items not yet met but explicitly addressed in later milestone phases.

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | `_cuda_legacy/hadamard.py` signature mismatch with `(u, normalize=False)` contract (CR-02 from 06-REVIEW.md) | Phase 7+/9/10 | Phase 9 SC#1: "the `backend` fixture parametrizes every shared correctness test over `{triton, cuda, torch}`"; Phase 10 SC#1: cuda backend deprecation. The cuda branch is dormant on this host (.so not built); SC#1 of Phase 6 explicitly says `TORCH_STRUCTURED_BACKEND=triton` (not cuda). |

### Required Artifacts

| Artifact                                                       | Expected                                                                                                                                                                                       | Status   | Details                                                                                                                                                                                                            |
| -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `torch_structured/_torch_ref/hadamard.py`                      | Pure-PyTorch oracle `hadamard_transform_torch(u, normalize=False)`; gradcheck reference; D-32 backward oracle                                                                                  | EXISTS, STUB-RISK | Exists (48 lines). Function present and callable. **But rank-2-only at line 41** (`batch_size, n = u.shape`). Used as backward callback by the Triton kernel which advertises `(*batch, n)` — contract mismatch. |
| `torch_structured/_triton/hadamard_transform/__init__.py`      | Package marker re-exporting hadamard_transform from op.py                                                                                                                                      | VERIFIED | 4 lines; re-exports correctly                                                                                                                                                                                      |
| `torch_structured/_triton/hadamard_transform/op.py`            | @triton.jit single-pass shared-memory Walsh-Hadamard kernel + @triton_op wrapper + register_autograd backward (self-inverse) + register_fake                                                   | VERIFIED | 191 lines; all 5 components present. `@triton.jit _hadamard_kernel`, `@triton_op("torch_structured::hadamard_transform")`, `_setup_context`, `_backward`, `@hadamard_transform.register_fake`. Uses `tl.static_range`. |
| `torch_structured/_cuda_legacy/hadamard.py`                    | Try-import passthrough to `_hadamard_cuda.hadamard_transform`; HAS_CUDA_LEGACY_HADAMARD sentinel                                                                                               | VERIFIED, with contract concern | 46 lines; `HAS_CUDA_LEGACY_HADAMARD` sentinel + try-import + RuntimeError on call. Signature is `(u)` only — does NOT accept normalize. See deferred CR-02.                                                       |
| `torch_structured/_ops.py`                                     | `_has_cuda_legacy_hadamard()` honest probe; three-branch hadamard_transform binding; per-op log.info extended                                                                                  | VERIFIED | Lines 97-107 (probe), 250-266 (binding), 269-270 (log.info). Stale Phase 6 placeholder comment removed.                                                                                                              |
| `torch_structured/structured/hadamard.py`                      | Back-compat shim re-exporting `hadamard_transform_torch` + callable `hadamard_transform` (D-05 attribute access)                                                                               | VERIFIED | 35 lines; no legacy autograd Function, no cuda wrapper, no module-level binding, no scipy.linalg, no device.                                                                                                          |
| `torch_structured/structured/fastfood.py`                      | `fastfood_multiply` rewritten per D-05 to use `torch_structured._ops.hadamard_transform(...)` at both call sites                                                                                | VERIFIED | 12 lines; both call sites rewritten; line 1 = `import torch_structured`.                                                                                                                                            |
| `tests/conftest.py`                                            | Backend fixture skip-gate widened to `_has_any_triton_kernel()` (D-39)                                                                                                                          | VERIFIED | 31 lines; line 25 uses `_has_any_triton_kernel()`.                                                                                                                                                                  |
| `tests/structured/test_hadamard_triton.py`                     | 5 tests covering SC#1/SC#2/SC#3 cross-backend + gradcheck + self-inverse + consumer surface; parametrized via `backend` fixture                                                                | VERIFIED | 140 lines; 5 functions × parametrizations = 32 collected. 31 PASS + 1 SKIP. **But no rank-3+ case** — test surface does not exercise the SC#2 "any input shape" semantic. See gap on Truth #2.                       |

### Key Link Verification

| From                                                                                  | To                                                                                                | Via                                                                          | Status   | Details                                                                                                                |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------- |
| `torch_structured/structured/fastfood.py:fastfood_multiply`                           | `torch_structured._ops.hadamard_transform`                                                        | attribute access per D-05 / D-34                                             | WIRED    | grep `torch_structured\._ops\.hadamard_transform` returns 2 in fastfood.py — both call sites                          |
| `torch_structured/_triton/hadamard_transform/op.py:_backward`                         | `torch_structured._torch_ref.hadamard.hadamard_transform_torch`                                   | imported as `_hadamard_transform_torch`; called in register_autograd backward | WIRED, but brittle | Import at op.py:39 verified; call at op.py:168 verified. **However**, the callee is rank-2-only — see Truth #2 gap.    |
| `torch_structured/_ops.py:_resolve Step 2 (hadamard block)`                            | `_triton.../op.hadamard_transform / _cuda_legacy.hadamard.hadamard_transform / _torch_ref.hadamard_transform_torch` | three-branch per-op binding                                                  | WIRED    | _ops.py:250-266 (three branches) + 269-270 log.info include hadamard_transform=%s                                       |
| `torch_structured/structured/hadamard.py:back-compat shim`                            | `torch_structured._torch_ref.hadamard.hadamard_transform_torch + torch_structured._ops.hadamard_transform` | re-export shim per D-33d                                                     | WIRED    | structured/hadamard.py:22 + 34. Both names callable from this module per test_imports.py.                              |
| `tests/structured/test_hadamard_triton.py`                                            | `torch_structured._ops.hadamard_transform`                                                        | attribute access through the parametrized backend fixture                    | WIRED    | 5+ attribute-access call sites in the test file; backend fixture parametrizes torch/triton with skip-gate.            |
| `tests/conftest.py:backend fixture`                                                   | `torch_structured._ops._has_any_triton_kernel`                                                    | widened skip-gate per D-39                                                   | WIRED    | tests/conftest.py:25                                                                                                  |

### Data-Flow Trace (Level 4)

| Artifact                                          | Data Variable | Source                                                                          | Produces Real Data | Status   |
| ------------------------------------------------- | ------------- | ------------------------------------------------------------------------------- | ------------------ | -------- |
| `_ops.hadamard_transform` (bound to Triton)       | `out`         | `wrap_triton(_hadamard_kernel)[grid](u, out, ...)` writes to a real CUDA buffer | Yes                | FLOWING  |
| `_triton/.../op._hadamard_kernel`                  | `out_ptr`     | tl.load/tl.store on u_ptr/out_ptr — real device memory operations               | Yes                | FLOWING  |
| `_ops.hadamard_transform` (bound to torch_ref)    | `x`           | `torch.cat((x[..., ::2, :] + ...))` — actual tensor math                        | Yes                | FLOWING  |
| `structured/fastfood.py:fastfood_multiply`        | `HBx`, `HGPHBx` | `torch_structured._ops.hadamard_transform(...)` — re-reads binding each call    | Yes                | FLOWING  |

### Behavioral Spot-Checks

| Behavior                                                              | Command                                                                                                                                                                                                                              | Result                                                                                | Status |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- | ------ |
| Env-var triton path resolves correctly                                 | `TORCH_STRUCTURED_BACKEND=triton python -c "import torch_structured; from torch_structured._triton.hadamard_transform.op import hadamard_transform as tri; assert torch_structured._ops._BACKEND == 'triton'; assert torch_structured._ops.hadamard_transform is tri"` | `_BACKEND= triton`, `hadamard_transform is triton kernel? True`                       | PASS   |
| Per-op log.info reports all three op bindings                          | `logging.basicConfig(INFO); set_backend('triton')` → captured INFO log                                                                                                                                                              | `per-op bindings: butterfly_multiply=triton, diag_mult=triton, hadamard_transform=triton` | PASS   |
| D-22 fallback warning for cuda hadamard when .so absent                | `set_backend('cuda')` with no `_hadamard_cuda.so`                                                                                                                                                                                    | WARNING: `set_backend('cuda') requested but _hadamard_cuda not built; falling back to torch_ref for hadamard_transform (D-22)` | PASS |
| `_has_cuda_legacy_hadamard()` returns clean bool                        | `_has_cuda_legacy_hadamard()`                                                                                                                                                                                                       | `False` (bool) on this workstation (no `_hadamard_cuda.so`)                            | PASS   |
| `_has_triton_kernel('hadamard_transform')` returns True                | `_has_triton_kernel('hadamard_transform')`                                                                                                                                                                                          | `True`                                                                                | PASS   |
| Rank-3 backward succeeds                                              | `torch_structured._ops.hadamard_transform(torch.randn(2,3,8,device='cuda',requires_grad=True)).sum().backward()`                                                                                                                  | `ValueError: too many values to unpack (expected 2)`                                  | FAIL   |
| Rank-3 forward on torch backend                                        | `torch_structured._ops.set_backend('torch'); torch_structured._ops.hadamard_transform(torch.randn(2,3,8))`                                                                                                                          | `ValueError: too many values to unpack (expected 2)`                                  | FAIL   |

### Probe Execution

| Probe | Command                                                                                                                                                                                                                                                                                                                                                                                                              | Result                          | Status |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- | ------ |
| `pytest tests/structured/test_hadamard_triton.py -v` | Phase 6 dedicated test suite                                                                                                                                                                                                                                                                                                                                                                                          | 31 PASS, 1 SKIP                  | PASS   |
| `pytest tests/structured/test_hadamard.py tests/structured/test_imports.py -v` | Back-compat shim regression                                                                                                                                                                                                                                                                                                                                                                                          | 6 PASS, 1 SKIP                   | PASS   |
| `TORCH_STRUCTURED_BACKEND=triton pytest tests/structured/ -v` | SC#3 literal contract (`pass pytest tests/structured/ on the Triton backend`)                                                                                                                                                                                                                                                                                                                                                                | 37 PASS, 2 SKIP, 0 FAIL          | PASS   |
| `pytest tests/test_dispatch.py tests/test_diag_mult.py -v` | Phase 5 regression                                                                                                                                                                                                                                                                                                                                                                                                  | 29 PASS                          | PASS   |

### Requirements Coverage

| Requirement | Source Plan | Description                                                                                  | Status                      | Evidence                                                                                                                                                                                                                       |
| ----------- | ----------- | -------------------------------------------------------------------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| TRI-02      | 06-01-PLAN  | `hadamard` runs on Triton (self-inverse, forward kernel only, fp32)                          | PARTIALLY SATISFIED          | Triton kernel exists, runs correctly fp32 across log_n {2..12}, self-inverse holds for rank-2 inputs. **But** `(*batch, n)` contract not honored — rank-3+ backward crashes. The wording of TRI-02 in REQUIREMENTS.md does not specify rank, but the wrapper docstring claims rank-arbitrary. |

REQUIREMENTS.md maps TRI-02 to Phase 6. No orphaned requirements (Phase 6 only owns TRI-02).

### Anti-Patterns Found

| File                                          | Line | Pattern                                                          | Severity | Impact                                                                                                          |
| --------------------------------------------- | ---- | ---------------------------------------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------- |
| `torch_structured/_torch_ref/hadamard.py`     | 41   | `batch_size, n = u.shape` — narrow contract; rank-2-only         | WARNING  | Wired into the Triton register_autograd backward callback; causes silent rank-2-only behavior at the resolver surface |
| `torch_structured/_cuda_legacy/hadamard.py`   | 33   | Signature `(u)` lacks `normalize` — incompatible with siblings   | WARNING  | Dormant (.so absent). Would raise TypeError on cuda hosts with `_hadamard_cuda.so` + normalize=True keyword.    |

No TODO/FIXME/HACK/XXX markers in Phase 6-modified files.
No empty implementations / placeholders.
No hardcoded empty data flowing to rendering.

### Code Review Findings Cross-Reference

The 06-REVIEW.md report identified 2 CRITICAL findings. Cross-referenced:

- **CR-01 (`_torch_ref/hadamard.py:41` rank-2-only)** → SURFACES AS BLOCKER GAP on Truth #2 (SC#2 strict reading). The pre-existing bug is verbatim-relocated; it becomes load-bearing because the Triton kernel's wrapper docstring advertises `(*batch, n)` and the register_autograd backward feeds rank-N grads into the rank-2-only oracle. Confirmed by direct execution.
- **CR-02 (`_ops.py:254-257` cuda branch signature mismatch)** → DEFERRED. The branch is unreachable on this host (no `.so` built). Phase 6 success criteria explicitly mention `TORCH_STRUCTURED_BACKEND=triton`, not cuda. Phase 9 will exercise the cuda axis when the `backend` fixture is extended; tracked there.

The 4 WARNING and 3 INFO findings from 06-REVIEW.md are not blocking gaps; they are noted here for completeness:
- WR-01 — Wrapper docstring promises `(*batch, n)` — same root cause as CR-01.
- WR-02 — No log.warning for triton-requested-but-kernel-missing — minor observability nit.
- WR-03 — No log_n=1 or log_n=0 edge cases — outside SC#1 ({2..12}).
- WR-04 — `_torch_ref/diag_mult` naming inconsistency — Phase 5 artifact; not Phase 6.
- IN-01..03 — minor style / docstring issues — not blocking.

### Human Verification Required

None — all SC contract assessments are programmatically verifiable. The SC#2 wording ambiguity ("any input shape") is a policy call best resolved by the developer (see Gaps Summary below); no functional or visual review needed.

### Gaps Summary

**One real gap** (BLOCKER under strict reading; WARNING under lenient reading):

The Phase 6 plan's Truth #2 and ROADMAP SC#2 both speak of self-inverse holding "on any input shape" / "on any (batch, 2**log_n) fp32 input". The codebase passes for rank-2 inputs but FAILS for rank-3+ inputs:

- Triton-backend forward succeeds on `(2, 3, 8)` (the wrapper uses `n_batch = u.numel() // n`).
- Triton-backend backward FAILS on `(2, 3, 8)` because the register_autograd backward callback feeds `grad_out` (rank-3) into `_torch_ref.hadamard.hadamard_transform_torch` which does `batch_size, n = u.shape`.
- Torch-backend forward FAILS on `(2, 3, 8)` for the same reason.

This is a single-line fix at `torch_structured/_torch_ref/hadamard.py:41` (replace `batch_size, n = u.shape` with `n = u.shape[-1]`), plus a rank-3 regression test in `tests/structured/test_hadamard_triton.py`.

**Decision policy:**

- If the SC#2 phrase "any input shape" means "any (batch, n)" only (matching the pre-Phase-6 oracle's rank-2 implementation), then the codebase satisfies SC#2 and this is a documentation drift in op.py:104 (which advertises `(*batch, n)`). Recommend: tighten the wrapper docstring and the `_torch_ref/hadamard.py` docstring to `(batch_size, n)` to make the contract explicit.
- If the SC#2 phrase "any input shape" means "any rank ending in n", then Phase 6 has NOT met SC#2 and the one-line fix is required.

**This looks like a real BLOCKER under the strict reading.** The wrapper docstring saying `(*batch, n)` is load-bearing — Phase 7+ consumers reading that docstring would write code that crashes. Recommend fixing in Phase 6 closure rather than deferring.

**An override can accept this gap** if the developer is content with the rank-2-only interpretation. Add to VERIFICATION.md frontmatter:

```yaml
overrides:
  - must_have: "SC#2 self-inverse on any input shape"
    reason: "Pre-Phase-6 contract was rank-2-only (matches structured/hadamard.py:15-30); Phase 6 relocates verbatim. The (*batch, n) wording in the Triton wrapper docstring will be corrected to (batch, n). Higher-rank support is a Phase 7+ concern."
    accepted_by: "{name}"
    accepted_at: "{ISO timestamp}"
```

---

_Verified: 2026-05-27T16:45:00Z_
_Verifier: Claude (gsd-verifier)_
