---
quick_id: 260709-ew0
title: Fix Triton 3.3.1 incompatibility — host-compute STRIDE in butterfly backward kernel
status: complete
date: 2026-07-09
commit: e173410
---

# Quick Task 260709-ew0 — Summary

## What changed

`torch_structured/_triton/butterfly/op.py` — hoisted the per-stage `STRIDE`
constexpr computation out of the `_butterfly_backward_kernel` `@triton.jit` body
to the host launch site.

- **Kernel signature:** added `STRIDE_0/STRIDE_1/STRIDE_2: tl.constexpr` params.
- **Removed from kernel:** the `if INCREASING_STRIDE: LOG_STRIDE_* = max(...) if
  STAGE_COUNT ... else 0` block and `STRIDE_* = 1 << LOG_STRIDE_*`. `IDX_0/1/2`
  (plain `STAGE_START + k`) stay in-kernel — front-end-safe, same form the
  forward kernel uses.
- **Launch site (`_backward`):** compute `stride_0/1/2` in plain Python
  (mirroring the old formula, with `max(...,0)` clamps for unused slots) and
  pass them as `STRIDE_0=/1=/2=`.

## Why

Triton **3.3.x**'s front-end can't evaluate the Python `max(...)` builtin / `... if
... else ...` ternary in a constexpr context inside a `@triton.jit` body — it
raises `ValueError('Did you forget to add @triton.jit ?')` at the
`1 << LOG_STRIDE_0` line. All inputs to those expressions are host-known
constexprs, so moving the computation to the host sidesteps the front-end
entirely, on any Triton version. Mirrors how `tile_n` / `num_warps` are already
host-computed. The forward kernel was never affected (it uses no `max()`).

## Verification

- **Environment:** torch 2.12.0+cu130, **triton 3.7.0**, CUDA available (1 device).
  Note the local Triton is 3.7.0, where the old code already compiled — the
  crash is 3.3.x-specific and could not be reproduced directly here. The fix is
  version-robust by construction (no `max()`/ternary in any jit body).
- **Parity sweep** (`scratchpad/sweep.py`): forward + `d_twiddle` + `d_input`
  vs `_torch_ref.butterfly.butterfly_multiply_torch`, `rtol=1e-3, atol=1e-4`,
  over log_n∈{2,3,4,5,7} × nblocks∈{1,2} × increasing∈{T,F} × dtype∈{fp32,
  complex64} = **40/40 OK**, identical to the pre-change baseline.
- `grep` confirms no `max(` / `LOG_STRIDE` remains inside any `@triton.jit`
  body (remaining hits are host-side Python in `_backward` or comments).
- `import torch_structured` clean (pure-Triton backend; CUDA C++ ext not built,
  expected).

## Notes

- pytest is not installed in `.venv`, so the standalone parity sweep is the
  authoritative functional check.
- Pre-existing Pylance "not accessed" diagnostics on `_butterfly_multiply_fake`
  / `_setup_context` args are unrelated to this change (standard `torch.library`
  registration signatures).
- Committed on branch `fix/triton-3-3-1-backward-constexpr-stride` (code commit
  `e173410`); `master` left untouched for the maintainer to merge.
