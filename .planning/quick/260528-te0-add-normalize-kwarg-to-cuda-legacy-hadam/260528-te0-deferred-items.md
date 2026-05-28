# Deferred Items — Quick Task 260528-te0

## Out of scope (discovered during execution, NOT fixed here)

### CUDA hadamard fp32 accumulation drift exceeds `atol=1e-6` at n=256

**Type:** bug (pre-existing, latent — exposed by matching the CUDA build)

**Symptom:** `tests/test_phase9_integration.py::test_backend_agreement_hadamard_transform_fp32[cuda-8]` fails:
```
AssertionError: hadamard_transform fp32 mismatch (backend=cuda, log_n=8): max_err=7.62939453125e-06
  torch.allclose(..., rtol=1e-5, atol=1e-6)
```

**Path:** `normalize=False`, `log_n=8` (n=256). NOT the `normalize=True` path this quick task touched — the fix here is orthogonal and verified correct (parity max_err ≤ 4.8e-07 at normalize=True across n=4..256).

**Root cause:** The CUDA FWHT kernel's tree-reduction order differs from `butterfly_multiply_torch` / `hadamard_transform_torch` (the torch_ref oracle) at n=256. This is inherent fp32 non-associativity — `7.63e-06` absolute on values of magnitude ~20 is ~3.8e-07 relative, within `~sqrt(256) * machine_eps_fp32`. The **triton backend passes** at the same size (matches torch_ref within 1e-6), so the drift is specific to the CUDA kernel's accumulation order, not a shared issue.

**Why it surfaced now:** `_has_cuda_legacy()` returned False on the dev host throughout Phase 6 and Phase 9 (PyTorch CUDA 13.0 vs prebuilt `.so` CUDA 12.6 mismatch), so the `[cuda-*]` test parametrizations always skipped. The user rebuilt the extensions against CUDA 13.0 (`TORCH_CUDA_ARCH_LIST=8.9` for the RTX 2000 Ada), making `_has_cuda_legacy()` True and the cuda-axis tests runnable for the first time.

**Recommended fix (separate task):**
- **(a) — recommended:** Widen the cuda-axis fp32 `atol` for `log_n >= 8` to `~1e-5` with a documented rationale (larger Hadamard sizes need a looser envelope; the drift is benign fp32 non-associativity, not a correctness bug). Mirrors the size-dependent tolerance pattern already used elsewhere in the suite.
- **(b):** Investigate whether the CUDA kernel can be made to match torch_ref's accumulation order. Higher effort; likely not worth it for a deprecated backend (CUDA path is `DeprecationWarning`'d as of Phase 10, slated for v1.4+ removal).

**References:**
- `tests/test_phase9_integration.py:350` (the assertion)
- `torch_structured/_cuda_legacy/hadamard.py` (the wrapper — fix here is unrelated to the drift)

**Tracking note:** Project CLAUDE.md mandates `bd` (beads) for issue tracking, but no beads database is initialized (`bd create` → "no beads database found"). File this via `bd create` after running `bd init`, or carry it as a deferred item here.
