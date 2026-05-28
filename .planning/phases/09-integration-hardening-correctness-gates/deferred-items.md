# Phase 9 Deferred Items

Items discovered during Plan 09-01 execution that are out-of-scope per the
GSD executor scope boundary (only auto-fix issues DIRECTLY caused by current
task changes). Each item is independent of Plan 09-01's deliverables; the
plan's success criteria are met without addressing these.

## Pre-existing test failures (master baseline — pre-Phase 9)

Verified by running `git checkout master -- torch_structured/butterfly/multiply.py`
and re-running the suite. These failures exist on master BEFORE any Phase 9
work; Plan 09-01 did not introduce them.

### 1. `tests/test_butterfly.py::ButterflyTest::test_butterfly` (and variants)

- **Files:** `tests/test_butterfly.py` (4 tests: `test_butterfly`,
  `test_butterfly_bmm`, `test_butterfly_to_base4`, `test_butterfly_unitary`).
- **Pre-fix error (master):** `RuntimeError: ...` (when calling C++ op with
  CUDA non-contiguous tensors — manifests via `torch.ops.torch_structured`
  dispatcher).
- **Post-fix error (Plan 09-01):** `AssertionError: input must be contiguous
  (Pitfall 3)` — surfaces at `torch_structured/_triton/butterfly/op.py:1224`.
- **Root cause:** Tests build CUDA tensors that aren't contiguous and pass
  them to `Butterfly(...).forward`. The Triton wrapper has a hard
  contiguity precondition (Pitfall 3, Phase 7). Neither master nor Plan 09-01
  changes the test surface; Plan 09-01 just makes the error message more
  honest by surfacing the precondition assertion instead of crashing in
  C++.
- **Fix-out-of-scope rationale:** Touching `tests/test_butterfly.py` to
  ensure inputs are contiguous would (a) modify test sources outside this
  plan's scope, (b) potentially mask other latent bugs. The cleaner fix is
  to either (1) make the Triton wrapper call `.contiguous()` on input
  internally (which would be a `_triton/butterfly/op.py` change — Phase 8
  D-57 invariants are explicitly off-limits), or (2) update the test
  source. Both are deferred.
- **Recommended next plan:** A small task in Plan 09-03 (or a separate
  cleanup ticket) to `input = input.contiguous()` at the test source level.

### 2. `tests/test_permutation.py::ButterflyPermutationTest::test_matrix_to_butterfly_factor`

- **File:** `tests/test_permutation.py:28`.
- **Error:** `RuntimeError: a view of a leaf Variable that requires grad is
  being used in an in-place operation` — `b.twiddle[0, 0, log_k - 1].copy_(factor)`.
- **Root cause:** The test does an in-place `.copy_()` into a slice of a
  leaf Parameter, which PyTorch's autograd rejects. Pre-existing on master.
- **Fix-out-of-scope rationale:** This is a test-source bug (should wrap
  in `torch.no_grad():` block), not a library bug.

## Dev-host environment limitations (NOT regressions)

### 3. `_butterfly.so` CPU-only build on the executor's dev host

- **Symptom:** `torch.ops.torch_structured.butterfly_multiply(cuda_tensor, ...)`
  raises `RuntimeError: Not compiled with CUDA support`.
- **Root cause:** PyTorch's compiled CUDA version (13.0) differs from the
  CUDA toolkit that compiled `_butterfly.so` (12.6). PyTorch's loader
  registers the schema but the CUDA dispatch is unavailable. Pre-existing
  environment issue (the `check_cuda_version()` warning at
  `butterfly/__init__.py:66` flags this).
- **Resolution in Plan 09-01:** Made the `_has_cuda_legacy()` probe more
  honest (Rule 2 — D-21 / CHECKER B3 honest-probe pattern). It now
  performs a one-shot CUDA dispatch sanity check at first invocation and
  caches the result. On this dev host the probe returns False, which makes
  the conftest cuda axis skip cleanly. On properly-built hosts (matched
  PyTorch and toolkit CUDA versions) the probe returns True and the cuda
  axis runs as designed.
