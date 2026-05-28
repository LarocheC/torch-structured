---
quick_id: 260528-te0
description: Add normalize kwarg to _cuda_legacy/hadamard.py wrapper to match torch_ref and Triton backend signatures
date: 2026-05-28
status: complete
commit: 342bdaa
---

# Quick Task 260528-te0 — Summary

> Note: this SUMMARY.md was reconstructed by the orchestrator after the
> executor's in-worktree copy was lost during worktree teardown (the rescue-cp
> step was skipped before `git worktree remove --force`). Content is faithful
> to the executor's completion report plus the orchestrator's post-merge
> verification — the code fix itself (commit 342bdaa) was committed in the
> worktree and merged intact.

## Objective

Add a `normalize: bool = False` parameter to
`torch_structured/_cuda_legacy/hadamard.py::hadamard_transform` so the CUDA
legacy backend has the same signature as `_torch_ref` and `_triton`. Before
this fix, the CUDA backend raised
`TypeError: hadamard_transform() got an unexpected keyword argument 'normalize'`
whenever a caller (or the backend-agreement tests) passed `normalize=...`.

## Root cause

The `_ops.py` resolver binds all three backends under the same name
`hadamard_transform`, but the three signatures had drifted:

| Backend | Signature |
|---------|-----------|
| `_torch_ref/hadamard.py` | `hadamard_transform_torch(u, normalize=False)` |
| `_triton/hadamard_transform/op.py` | `hadamard_transform(u, normalize=False)` |
| `_cuda_legacy/hadamard.py` (before) | `hadamard_transform(u)` — **no `normalize`** |

The raw pybind op `_hadamard_cuda.hadamard_transform(u)` returns the
**unnormalized** FWHT (verified: `H @ [1,0,0,0] = [1,1,1,1]`), so normalization
must be applied in the Python wrapper. Never caught during Phase 6/9 because
`_has_cuda_legacy()` was False on the dev host (CUDA 13.0 vs prebuilt-`.so`
12.6 mismatch); the user's CUDA-13.0 rebuild made the `[cuda-*]` test
parametrizations runnable and exposed the gap.

## Change

`torch_structured/_cuda_legacy/hadamard.py` (commit `342bdaa`):
- Added `normalize: bool = False` to the signature.
- After the existing `RuntimeError`-when-`.so`-missing guard (D-22 honest
  fallback, preserved as the first statement) and after calling the raw op:
  when `normalize=True`, divide by `2 ** (m / 2)` where `m = n.bit_length() - 1`
  (exact integer log2 for power-of-2 `n`, `n = u.shape[-1]`), guarded by
  `assert n == 1 << m, 'n must be a power of 2'`. Matches `_torch_ref` exactly.
- Docstring updated to document the `normalize` parameter (same wording as
  `_torch_ref`).

Single file, single function, +14/-2 lines.

## Verification (post-merge, on the matched CUDA-13.0 build)

- **Numerical parity (`normalize=True`)** vs `_torch_ref.hadamard_transform_torch`:
  n=4 → 2.38e-07, n=16 → 2.38e-07, n=64 → 3.58e-07, n=256 → 4.77e-07. All OK.
- **`inspect.signature` confirms** `normalize` is now a parameter.
- **`pytest tests/test_phase9_integration.py -k hadamard`:** 6 passed, 1 failed.
  - `[cuda-4]` now **PASSES** (was erroring with TypeError before the fix).
  - torch/triton axes all pass.
  - `[cuda-8]` fails on a **pre-existing, out-of-scope** fp32 accumulation gap
    (`normalize=False` path, n=256, max_err=7.63e-06 vs `atol=1e-6`) — NOT a
    TypeError, NOT caused by this fix. See `260528-te0-deferred-items.md`.

## Self-Check: PASSED

- `normalize` param present in signature ✓
- `2 ** (m / 2)` division applied only when `normalize=True` ✓
- `RuntimeError` guard remains the first statement ✓
- cuda-axis hadamard tests no longer raise TypeError ✓ (`[cuda-4]` passes)
- Numerical parity with `_torch_ref` at `normalize=True` holds (≤4.8e-07) ✓

## Deferred

One out-of-scope finding recorded in `260528-te0-deferred-items.md`: the
`[cuda-8]` fp32 tolerance gap on the `normalize=False` path. Recommended
disposition: widen the cuda-axis `atol` for `log_n >= 8`, or accept as benign
fp32 non-associativity on a deprecated backend. Beads tracking not filed —
no beads DB initialized (`bd init` not run).
