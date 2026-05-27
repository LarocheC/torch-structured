# Deferred Items — Phase 04

Out-of-scope discoveries logged during plan execution. NOT fixed here; tracked
for future plans.

## From Plan 04-02 execution

### 1. Pre-existing test failures: CUDA C++ extension stubs

**Discovered during:** Plan 04-02 final verification (`pytest tests/`)

**Tests affected:**
- `tests/test_butterfly.py` — 5 failures (test_butterfly, test_butterfly_bmm, test_butterfly_to_base4, test_butterfly_unitary, test_transpose_conjugate_multiply)
- `tests/test_multiply.py` — 2 failures (test_input_padding_output_slicing, test_multiply)
- `tests/test_permutation.py` — 1 failure (test_matrix_to_butterfly_factor)

**Cause:** The worktree was built with `FORCE_CPU=1` because the host CUDA
(12.6) does not match the bundled PyTorch CUDA (13.0). The resulting `.so`
file registers the op (so `_has_cuda_legacy()` returns True) but the actual
butterfly kernel raises `RuntimeError: Not compiled with CUDA support` when
called on CUDA tensors. These are environment-driven failures that would
reproduce identically before Plan 04-02 touched the repo (and were already
documented in 04-01-SUMMARY.md under "Deviations").

**Scope:** Out of scope for Plan 04-02. Plan 04-02 does not touch the legacy
C++ extension path; these tests exercise `torch.ops.torch_structured.butterfly_multiply`
(the C++ op), not the new dispatch layer or the demonstrator op.

**Verification that Plan 04-02 is not the cause:** `tests/test_dispatch.py`
(the Plan 04-02 deliverable) passes 5/5 on the same worktree.

**Disposition:** Defer. Resolution requires a CUDA-aware build environment
(matching host CUDA + torch CUDA), which is a CI/host environment concern.
Phase 5 (real Triton kernel for diag_mult) will not depend on the legacy
C++ extension and will run cleanly on this worktree.

### 2. Pre-existing collection error: `pywt` missing

**Discovered during:** Plan 04-02 final verification

**Test affected:** `tests/test_special.py` (entire module fails to import)

**Cause:** `tests/test_special.py:13` imports `pywt` (PyWavelets) for wavelet
testing. `pywt` is not declared in `pyproject.toml`'s `test` extra.

**Scope:** Out of scope for Plan 04-02. The `pywt` dependency predates this
work and Plan 04-02 does not modify `pyproject.toml` or `tests/test_special.py`.

**Disposition:** Defer. Either add `pywt` to the `dev` / `test` extras in a
future packaging plan, or convert the import to a soft import with
`pytest.importorskip("pywt")` inside the test file.
