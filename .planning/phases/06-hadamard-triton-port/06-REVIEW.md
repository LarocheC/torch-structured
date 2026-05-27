---
phase: 06-hadamard-triton-port
reviewed: 2026-05-27T16:38:36Z
depth: standard
files_reviewed: 11
files_reviewed_list:
  - tests/conftest.py
  - tests/structured/test_hadamard_triton.py
  - torch_structured/_cuda_legacy/__init__.py
  - torch_structured/_cuda_legacy/hadamard.py
  - torch_structured/_ops.py
  - torch_structured/_torch_ref/__init__.py
  - torch_structured/_torch_ref/hadamard.py
  - torch_structured/_triton/hadamard_transform/__init__.py
  - torch_structured/_triton/hadamard_transform/op.py
  - torch_structured/structured/fastfood.py
  - torch_structured/structured/hadamard.py
findings:
  critical: 2
  warning: 4
  info: 3
  total: 9
status: issues_found
---

# Phase 6: Code Review Report

**Reviewed:** 2026-05-27T16:38:36Z
**Depth:** standard
**Files Reviewed:** 11
**Status:** issues_found

## Summary

The Phase 6 Triton port implements the Walsh-Hadamard transform with a single-pass shared-memory `@triton.jit` kernel, `register_autograd` (self-inverse) backward, `register_fake` meta kernel, resolver wiring, and a back-compat shim. The Triton kernel butterfly logic is correct (verified against `csrc/hadamard/hadamard_cuda_kernel.cu:50-53`), the `tl.debug_barrier()` placement is appropriate for the out_ptr-as-scratch shuffle pattern, the resolver's three-branch hadamard binding follows the D-22 honest-probe shape, and the back-compat shim correctly preserves both names as callables.

Two **CRITICAL** issues found:

1. The `_torch_ref/hadamard.py` oracle is hardcoded to rank-2 inputs (`batch_size, n = u.shape`), but it is used as the **backward delegate** for the Triton kernel, which itself accepts arbitrary leading dimensions. Backward on rank != 2 inputs crashes with `ValueError: too many values to unpack`. This makes the Triton wrapper's documented `(*batch, n)` contract un-trainable for any non-2D input. **Verified by direct execution.**
2. The `cuda` branch of the hadamard resolver binds `_cuda_legacy/hadamard.py:hadamard_transform`, which has a signature `(u)` with **no `normalize` parameter** — incompatible with the Triton/torch backends' `(u, normalize=False)` signature. Any consumer calling `_ops.hadamard_transform(u, normalize=True)` while the resolver is on the cuda backend will raise `TypeError`, and a positional call would silently return an unnormalized result (silent wrong answer if normalize=True is the contract).

Warnings cover the schema/dispatch contract mismatch in the wrapper docstring vs. backward path, the lack of cross-rank/cross-batch tests for the Triton forward, a missing per-op `cuda` test path through the resolver, and an inconsistent naming pattern in `_torch_ref/__init__.py`.

## Critical Issues

### CR-01: `_torch_ref/hadamard.py` oracle crashes on non-rank-2 inputs, breaking Triton backward for `(*batch, n)` shapes

**File:** `torch_structured/_torch_ref/hadamard.py:41`
**Issue:** The pure-PyTorch oracle hardcodes a rank-2 unpacking — `batch_size, n = u.shape` — but it is used in two places that promise to support arbitrary leading dimensions:

1. The **Triton backward callback** in `torch_structured/_triton/hadamard_transform/op.py:168` calls `_hadamard_transform_torch(grad_out, normalize=ctx.normalize)`. The Triton **forward** explicitly handles `(*batch, n)` via `n_batch = u.numel() // n` (op.py:127) and its docstring on op.py:104 says "u: Tensor of shape `(*batch, n)`". So a forward succeeds on rank-3 inputs but the backward crashes — verified by execution:
   ```text
   >>> u = torch.randn(2, 3, 8, device='cuda', requires_grad=True)
   >>> torch_structured._ops.hadamard_transform(u).sum().backward()
   ValueError: too many values to unpack (expected 2)
   ```
2. The **`torch` backend** binding (`_ops.py:259-260`) is also `hadamard_transform_torch`, so `set_backend("torch")` makes the same forward crash on any rank != 2:
   ```text
   >>> torch_structured._ops.set_backend('torch')
   >>> u = torch.randn(2, 3, 8, device='cuda')
   >>> torch_structured._ops.hadamard_transform(u)
   ValueError: too many values to unpack (expected 2)
   ```

This makes the Triton backend silently rank-2-only for any training (forward works, backward crashes), and the torch backend rank-2-only across the board. The legacy CUDA C++ extension at `csrc/hadamard/hadamard_cuda.cpp:11` correctly does `batchSize = x.numel() / (1 << log2N)`, so the existing API surface promised arbitrary leading dims; the new Python oracle has narrowed it.

No test exercises non-rank-2 (`test_hadamard_eager_fp32` uses `(4, n)`, gradcheck uses `(2, n)`, `fastfood_multiply` uses `(B, n)`), so this regression is not caught.

**Fix:** Generalize the oracle to handle arbitrary leading dimensions. Replace the rank-2-only path in `_torch_ref/hadamard.py`:

```python
def hadamard_transform_torch(u, normalize=False):
    """Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
    n must be a power of 2.
    Parameters:
        u: Tensor of shape (..., n)
        normalize: if True, divide the result by 2^{m/2} where m = log_2(n).
    Returns:
        product: Tensor of shape (..., n)
    """
    n = u.shape[-1]
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    x = u[..., np.newaxis]
    for d in range(m)[::-1]:
        x = torch.cat((x[..., ::2, :] + x[..., 1::2, :],
                       x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
    return x.squeeze(-2) / 2**(m / 2) if normalize else x.squeeze(-2)
```

(Only the `batch_size, n = u.shape` line needs to change to `n = u.shape[-1]`; the rest of the body already uses `...` indexing.) Add a rank-3 case to `test_hadamard_eager_fp32` and a rank-3 case to `test_hadamard_gradcheck_fp64` to prevent regression.

### CR-02: `_ops.py` cuda binding has incompatible signature (no `normalize`), breaking the contract on hosts that select the cuda backend

**File:** `torch_structured/_ops.py:254-257`
**Issue:** When the resolver lands `actual == "cuda"` AND `_has_cuda_legacy_hadamard()` is True, it binds:

```python
elif actual == "cuda" and _has_cuda_legacy_hadamard():
    from torch_structured._cuda_legacy.hadamard import hadamard_transform as _cuda_ht
    hadamard_transform = _cuda_ht
```

But `_cuda_legacy/hadamard.py:33` defines `hadamard_transform(u: torch.Tensor) -> torch.Tensor` — **no `normalize` argument**. The Triton (`op.py:96`) and torch (`_torch_ref/hadamard.py:32`) bindings both have `(u, normalize=False)`. Three concrete failure modes for any consumer that calls through the resolver:

1. `_ops.hadamard_transform(u, normalize=True)` → `TypeError: hadamard_transform() got an unexpected keyword argument 'normalize'`
2. `_ops.hadamard_transform(u, True)` (positional) → `TypeError: hadamard_transform() takes 1 positional argument but 2 were given`
3. `_ops.hadamard_transform(u)` returns an **unnormalized** result silently, even though the consumer may rely on `normalize=False` semantics being identical across backends.

The legacy C++ extension itself does NOT apply normalization — `csrc/hadamard/hadamard_cuda.cpp:5-13` just runs `fwtBatchGPU`. The old `structured/hadamard.py` (pre-Phase 6) wrapped this in `hadamard_transform_cuda(u, normalize=False)` which divided by `2**(m/2)` Python-side; that wrapping was deleted in Task 4 along with the `HadamardTransformCuda` autograd class. The new resolver should reapply that wrapping at the cuda binding site (mirroring the Triton wrapper, which also applies normalization Python-side at op.py:145).

This branch is not exercised by any test (`test_hadamard_triton.py` parametrizes only `["torch", "triton"]`; the future `cuda` axis is deferred to Phase 7 per `conftest.py:10`), so the bug is dormant until either (a) a user sets `TORCH_STRUCTURED_BACKEND=cuda` on a host with `_hadamard_cuda.so` built, or (b) Phase 7+ adds the cuda axis to the fixture.

**Fix:** Either (preferred) wrap the cuda binding to honor the contract:

```python
elif actual == "cuda" and _has_cuda_legacy_hadamard():
    from torch_structured._cuda_legacy.hadamard import hadamard_transform as _cuda_ht_raw

    def _cuda_ht(u: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        out = _cuda_ht_raw(u)
        if normalize:
            n = u.size(-1)
            log_n = int(n.bit_length() - 1)
            return out / (2 ** (log_n / 2))
        return out

    hadamard_transform = _cuda_ht
    _hadamard_transform_backend = "cuda"
```

…or fix it at the source in `_cuda_legacy/hadamard.py:33` by adding the `normalize` parameter and Python-side division there. Either approach makes the cuda binding contract-compatible with Triton/torch. Add a test that exercises `set_backend("cuda")` with `normalize=True` (gated on `_has_cuda_legacy_hadamard()` to avoid CI breakage) to lock the contract.

## Warnings

### WR-01: Wrapper docstring promises `(*batch, n)` support that backward cannot fulfill

**File:** `torch_structured/_triton/hadamard_transform/op.py:104, 169`
**Issue:** The wrapper docstring on op.py:104 says `u: Tensor of shape (*batch, n)` and the forward kernel is shape-generic via `n_batch = u.numel() // n` (op.py:127). But the `_backward` callback (op.py:168) routes through `_hadamard_transform_torch` which is rank-2-only (see CR-01). The documented contract is not honored by the backward path.

This is the same root cause as CR-01 but specifically the **wrapper-level documentation** lies about the supported input rank. Even after CR-01 is fixed, a defensive assertion at the wrapper boundary would catch a future regression in the oracle. The wrapper already has `assert u.dim() >= 1` at op.py:114; consider adding either (a) explicit rank-2-or-greater enforcement, or (b) leave it open and rely on CR-01's fix.

**Fix:** Resolve CR-01 first, then add a rank-3 cross-backend test to lock the contract:

```python
@pytest.mark.parametrize("shape", [(4, 16), (2, 3, 16), (2, 3, 5, 16)])
def test_hadamard_eager_higher_rank(backend, shape):
    u = torch.randn(*shape, device="cuda", dtype=torch.float32)
    out = torch_structured._ops.hadamard_transform(u, normalize=False)
    expected = hadamard_ref(u)
    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-6)
```

### WR-02: Resolver Step 2 hadamard binding does not log per-op fallback when `actual == "triton"` but kernel missing

**File:** `torch_structured/_ops.py:249-266`
**Issue:** The diag_mult binding at `_ops.py:243-247` emits a `log.warning` when `actual == "cuda"` but the `.so` is missing (fallback to torch). The hadamard binding at `_ops.py:262-266` emits a parallel warning for the same case. **But** neither emits a warning for the case `actual == "triton"` AND the per-op kernel is missing — the resolver silently falls through to the `else: torch` branch with no log.

The most common path where this fires: a user sets `TORCH_STRUCTURED_BACKEND=triton` on a host where only one of {diag_mult, hadamard_transform, butterfly_multiply} has a Triton kernel installed. `_has_any_triton_kernel()` returns True in Step 1, `actual="triton"`, but Step 2 binds the missing op silently to torch.

This is technically the documented D-22a "coarse-actual-with-per-op-truth" pattern (the per-op truth is logged via `log.info` at op.py:268-271), but only at INFO level — easy to miss compared to the per-op WARNING for the cuda branch. Inconsistent log severity for symmetric fallback cases.

**Fix:** Add a parallel warning for the triton-requested-but-kernel-missing case:

```python
else:
    from torch_structured._torch_ref.hadamard import hadamard_transform_torch as _torch_ht
    hadamard_transform = _torch_ht
    _hadamard_transform_backend = "torch"
    if actual == "cuda":
        log.warning(
            "set_backend('cuda') requested but _hadamard_cuda not built; "
            "falling back to torch_ref for hadamard_transform (D-22)"
        )
    elif actual == "triton":
        log.warning(
            "set_backend('triton') requested but no Triton hadamard_transform "
            "kernel installed; falling back to torch_ref for hadamard_transform "
            "(D-22a per-op asymmetric fallback)"
        )
```

Apply the same change to the diag_mult binding at op.py:239-247 for symmetry.

### WR-03: Test coverage gap — no `log_n=1` (n=2) or `log_n=0` (n=1) edge case

**File:** `tests/structured/test_hadamard_triton.py:34`
**Issue:** `test_hadamard_eager_fp32` parametrizes over `log_n in range(2, 13)`, missing `log_n=1` (n=2, smallest non-trivial transform) and `log_n=0` (n=1, identity). The kernel's `tl.static_range(LOG_N)` with `LOG_N=0` would run zero iterations — output should equal input. The wrapper's `assert log_n <= 12` allows `log_n=0` (`1 << 0 == 1`), but `tl.arange(0, BLOCK_SIZE)` with `BLOCK_SIZE=1` is a corner case that's not tested.

Similarly, the self-inverse test only covers `log_n in [8, 10]`, skipping the small-n behavior. The CUDA C++ legacy (`csrc/hadamard/hadamard_cuda_kernel.cu`) supports `log_n >= 1` so the Triton port should at least cover that range.

**Fix:** Extend `test_hadamard_eager_fp32` parametrize to `list(range(1, 13))` (or `list(range(0, 13))` if `log_n=0` is in-scope) and add `log_n in [1, 2]` to `test_hadamard_self_inverse`. If `log_n=0` is intentionally out of scope, add `assert log_n >= 1` to the wrapper at op.py:121 and document why.

### WR-04: `_torch_ref/__init__.py` re-export naming is inconsistent

**File:** `torch_structured/_torch_ref/__init__.py:3-4, 6`
**Issue:** The module re-exports:
- `butterfly_multiply_torch` (with `_torch` suffix)
- `diag_mult` (no suffix)
- `hadamard_transform_torch` (with `_torch` suffix)

`diag_mult` is the odd one out. Per CLAUDE.md "Pure-Python reference implementations use `_torch` suffix: `butterfly_multiply_torch`, `butterfly_multiply_base4_torch`". The `diag_mult` name violates that convention. This was introduced in Phase 5, not Phase 6, but Phase 6's addition of `hadamard_transform_torch` to the same `__init__.py` highlights the inconsistency.

This isn't load-bearing (the resolver at `_ops.py:240` already aliases `diag_mult as _torch_dm`), but external consumers reading `_torch_ref/__init__.py` cannot infer the naming rule from the three exports.

**Fix:** Either rename `_torch_ref/diag_mult.py:32` to `diag_mult_torch` and re-export accordingly, or document the deviation explicitly in `_torch_ref/__init__.py`. The rename is the cleaner option — only one resolver site (`_ops.py:240`) needs the alias updated.

## Info

### IN-01: `_setup_context` parameter `output` is unused

**File:** `torch_structured/_triton/hadamard_transform/op.py:148`
**Issue:** `_setup_context(ctx, inputs, output)` accepts `output` but does not use it. The body only unpacks `inputs` to extract `normalize`. This matches the documented "no `save_for_backward(u)` — self-inverse" pattern (op.py:151-153), but the unused parameter could be silenced by `_ = output` or commented. Style only — `register_autograd`'s API requires the signature.

**Fix:** Add an explicit `del output` or `_ = output  # unused — self-inverse backward doesn't need forward output` for clarity. Optional.

### IN-02: `n_batch = u.numel() // n` integer division silently truncates on malformed input

**File:** `torch_structured/_triton/hadamard_transform/op.py:127`
**Issue:** If `u.numel()` is not a multiple of `n` (e.g., due to non-contiguous strides interacting with `numel()`), `n_batch` would silently round down and the kernel would process only `n_batch * n` elements, leaving the rest untouched in the output. The wrapper already asserts `u.is_contiguous()` at op.py:118, which makes `numel() == prod(shape)` so `numel() % n == 0` whenever `u.size(-1) == n` (always true). So the bug is unreachable in practice, but adding an assertion would make the invariant explicit and defend against future contiguity-assertion drift.

**Fix:** Optional belt-and-suspenders:

```python
n_batch = u.numel() // n
assert n_batch * n == u.numel(), f"numel mismatch: {u.numel()} not divisible by {n}"
```

### IN-03: `_cuda_legacy/__init__.py` docstring drift

**File:** `torch_structured/_cuda_legacy/__init__.py:7-12`
**Issue:** The docstring says "Phase 10 may absorb the loader into `_cuda_legacy/`" but Phase 6 has now added both `diag_mult` and `hadamard_transform` to this module via the same pattern (try-import + sentinel). The docstring still talks only about `butterfly_multiply`. The actual exports at line 14-18 now include all three. Update the docstring to reflect the current scope.

**Fix:** Update lines 7-12 of the docstring to acknowledge that `_cuda_legacy/` already houses diag_mult and hadamard_transform (Phase 5/6), and that Phase 10 will fold in butterfly's loader.

---

_Reviewed: 2026-05-27T16:38:36Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
