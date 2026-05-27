---
phase: 05-diag-mult-triton-port
reviewed: 2026-05-27T14:21:05Z
depth: standard
files_reviewed: 11
files_reviewed_list:
  - torch_structured/_torch_ref/diag_mult.py
  - torch_structured/_torch_ref/__init__.py
  - torch_structured/_triton/diag_mult/__init__.py
  - torch_structured/_triton/diag_mult/op.py
  - torch_structured/_cuda_legacy/diag_mult.py
  - torch_structured/_cuda_legacy/__init__.py
  - torch_structured/_ops.py
  - torch_structured/structured/krylov.py
  - tests/conftest.py
  - tests/test_diag_mult.py
  - tests/test_dispatch.py
findings:
  critical: 1
  warning: 6
  info: 3
  total: 10
status: issues_found
---

# Phase 5: Code Review Report

**Reviewed:** 2026-05-27T14:21:05Z
**Depth:** standard
**Files Reviewed:** 11
**Status:** issues_found

## Summary

Phase 5 ports `cycle_mult` from C++/CUDA to a Triton kernel exposed as
`_ops.diag_mult(...)`. The implementation is broadly correct: the kernel index
arithmetic matches the C++ reference, the Wirtinger backward formula is sound,
the `view_as_real` contiguity guard is in place, and the `_resolve()` step-1
widening (`_has_any_triton_kernel()`) successfully fixes BLOCKER-1 from the
iteration-2 plan. 29/29 tests pass.

However, several real defects exist in shape/broadcast handling that the
test suite does not exercise:

1. **CR-01 (Critical):** The backward callback only handles pure 1-D
   broadcast and full-numel-match. Any *partial* broadcast (e.g.,
   `subdiag.shape=(1, N)` with `v.shape=(B, N)`, or `subdiag.shape=(B, 1, N)`
   with `v.shape=(B, K, N)`) silently produces incorrect or shape-mismatched
   gradients — and in some configurations the forward kernel ALSO miscomputes
   for the same input shapes. This is undefined behaviour with no precondition
   guard.

2. **WR-01..04:** Test coverage gaps (CUDA-only skip masks torch-ref backend
   on CPU runners), missing `is_conj()` precondition guard, and an
   asymmetric-fallback warning that fires only for `cuda` and not for `triton`.

3. **WR-05..06 / IN-01..03:** Minor robustness and docstring issues.

Findings are grouped by severity below.

## Critical Issues

### CR-01: Partial-broadcast subdiag silently miscomputed in both forward and backward

**File:** `torch_structured/_triton/diag_mult/op.py:127-139,185-189`

**Issue:** The forward kernel uses a binary classification:

```python
is_batched_subdiag = (subdiag.numel() == v.numel())   # line 129
...
subdiag_batch_stride = 2 * N if is_batched_subdiag else 0   # line 135 / 139
```

This collapses everything that isn't a full-numel match into the
"stride 0 — broadcast over all batch rows" case. For shapes like
`subdiag=(B, 1, N)` and `v=(B, K, N)` (a legitimate broadcast pattern PyTorch
supports natively), the kernel sets `subdiag_batch_stride = 0` and reads
`subdiag[0, 0, :]` for every batch row — silently producing wrong results.
For `subdiag=(1, N)` with `v=(B, N)` and `B > 1`, the forward happens to be
correct (storage layout is identical to 1-D `(N,)`), but the backward then
fails:

```python
if subdiag.shape != grad_subdiag.shape:
    ndims_to_sum = grad_subdiag.dim() - subdiag.dim()     # line 187
    if ndims_to_sum > 0:                                   # line 188
        grad_subdiag = grad_subdiag.sum(dim=tuple(range(ndims_to_sum)))
```

With `subdiag.shape=(1, N)` (dim=2) and `grad_subdiag.shape=(B, N)` (dim=2),
`ndims_to_sum = 0`, no reduction occurs, and autograd raises a gradient-shape
mismatch (`(B, N) != (1, N)`).

The C++ kernel had the same binary classification, so this isn't a
regression — but the new code path advertises NumPy-style broadcast in its
docstring (`broadcasts over batch`) without enforcing the precondition that
makes broadcast safe (1-D subdiag only).

**Fix:** Either (a) explicitly forbid partial broadcast with an assertion, or
(b) properly handle the general broadcast case.

Option (a) — simpler, matches C++ legacy semantics. Add to `diag_mult()`
(after line 125):

```python
# Only two subdiag layouts are supported: pure 1-D broadcast or full-numel match.
# Partial broadcasts (mismatched leading dims that aren't all-1 vs full) would be
# silently miscomputed by the kernel (stride 0 means all batch rows share row 0).
assert subdiag.dim() == 1 or subdiag.shape == v.shape, (
    f"subdiag must be 1-D ({(v.size(-1),)}) for broadcast OR exactly match "
    f"v.shape ({tuple(v.shape)}); got subdiag.shape={tuple(subdiag.shape)}"
)
```

Option (b) — general broadcast — requires reducing over all dims where
`subdiag.shape[i] == 1` and `v.shape[i] > 1` in the backward, and computing
per-batch-row strides for subdiag in the forward kernel. This is strictly
more complex and probably out of scope.

The minimal correctness fix is option (a) plus updating the docstring on
line 102-104 to remove the "broadcasts over batch" implication of arbitrary
broadcast.

## Warnings

### WR-01: `pytestmark = skipif(not cuda)` masks the torch_ref backend on CPU runners

**File:** `tests/test_diag_mult.py:22-24`

**Issue:** The module-level skip:

```python
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="diag_mult tests require CUDA"
)
```

skips ALL diag_mult tests on CPU-only runners, including the
`backend="torch"` parametrization which doesn't actually need CUDA. The
`backend` fixture already does its own per-parametrization skip for the
triton branch (conftest.py:18-19), so the module-level skip is over-broad.

Effect: the torch_ref code path
(`torch_structured/_torch_ref/diag_mult.py`) is never exercised in CI on
CPU-only runners — a regression in torch_ref (e.g., wrong `torch.roll` sign,
broken assertion) can land green if the runner doesn't have CUDA.

**Fix:** Move the CUDA-only skip into individual tests (or onto the device
parametrization). Force the torch fixture branch to use CPU when CUDA isn't
available:

```python
# In conftest.py
@pytest.fixture(params=["torch", "triton"])
def backend(request):
    if request.param == "triton" and not torch_structured._ops._has_triton_kernel("diag_mult"):
        pytest.skip("Triton kernel for diag_mult not installed")
    ...

# In test_diag_mult.py — remove pytestmark, instead select device based on backend:
def _device_for(backend):
    return "cuda" if backend == "triton" else ("cuda" if torch.cuda.is_available() else "cpu")
```

Or simpler: drop `pytestmark`, and let individual tests use `cuda` only when
the backend requires it.

### WR-02: `is_conj()`-flagged complex tensor passes contiguity assertion but fails inside `view_as_real`

**File:** `torch_structured/_triton/diag_mult/op.py:119-121`

**Issue:** A complex tensor with the conjugate bit set (e.g., the result of
`t.conj()`) reports `is_contiguous() == True` but `torch.view_as_real` raises:

> `RuntimeError: view_as_real doesn't work on unresolved conjugated tensors.`

The current assertions only check `is_contiguous()`, so a caller passing a
conj-flagged tensor gets a cryptic downstream error instead of a clear
precondition violation. The backward callback in this same file calls
`_diag_mult_torch(... v.conj() ...)` — which is safe because the *torch_ref*
path uses `torch.roll` and doesn't call `view_as_real` — but a user calling
`_ops.diag_mult(s.conj(), v, ...)` directly with the Triton backend bound
will hit this rough edge.

**Fix:** Tighten the precondition assertions for complex inputs:

```python
if v.is_complex():
    assert not v.is_conj(), "v must not be conj-flagged before view_as_real; call .resolve_conj()"
    assert not subdiag.is_conj(), "subdiag must not be conj-flagged before view_as_real; call .resolve_conj()"
```

Or pre-resolve internally: `if v.is_conj(): v = v.resolve_conj()` (loses
zero-copy but works).

### WR-03: D-22 fallback warning is silently asymmetric — fires for `cuda` fallback but not `triton` fallback

**File:** `torch_structured/_ops.py:226-234`

**Issue:** When `actual == "triton"` but `_has_triton_kernel("diag_mult")` is
False (e.g., Phase 6 lights up hadamard's triton kernel, so
`_has_any_triton_kernel()` is True and `actual="triton"`, but diag_mult's
triton kernel is somehow missing — partial install, version skew), the
resolver falls through to the `else` branch and silently binds torch_ref
**without emitting a warning**. The `cuda` fallback in the same `else` block
DOES emit a warning (lines 230-234).

This is the exact pattern D-22 was supposed to address: per-op honest
fallback. In Phase 5 the scenario is impossible (only diag_mult lights up
`_has_any_triton_kernel`), but it becomes reachable in Phase 6+.

**Fix:** Add the symmetric warning for the triton-coarse / no-triton-op case:

```python
else:
    from torch_structured._torch_ref.diag_mult import diag_mult as _torch_dm
    diag_mult = _torch_dm
    _diag_mult_backend = "torch"
    if actual == "cuda":
        log.warning(
            "set_backend('cuda') requested but _diag_mult_cuda not built; "
            "falling back to torch_ref for diag_mult (D-22)"
        )
    elif actual == "triton":
        log.warning(
            "Triton backend selected but no Triton diag_mult kernel installed; "
            "falling back to torch_ref for diag_mult (D-22)"
        )
```

### WR-04: `_cuda_legacy/diag_mult.py` pass-through accepts complex/wrong-dtype tensors and corrupts memory

**File:** `torch_structured/_cuda_legacy/diag_mult.py:32-45`

**Issue:** Unlike the Triton wrapper (op.py:115-117) and the torch_ref
(diag_mult.py:48-50), the legacy CUDA pass-through does NOT validate dtype.
The legacy C++ kernel signature is `float*` (csrc/diag_mult/diag_mult_cuda_kernel.cu:1)
— passing a `complex64`, `float64`, or `int` tensor will reinterpret memory
and produce silent corruption (or crash). This is a pre-existing legacy
defect inherited by Phase 5, but Phase 5 promotes this path to first-class
status via the `_ops.diag_mult` dispatch.

**Fix:** Add the same precondition assertions as the other backends:

```python
def diag_mult(subdiag, v, shift_subdiag, shift_v):
    if _diag_mult_cuda_module is None:
        raise RuntimeError(...)
    assert v.dtype == torch.float32 and subdiag.dtype == torch.float32, (
        f"_cuda_legacy.diag_mult requires float32; got subdiag={subdiag.dtype}, v={v.dtype}"
    )
    assert v.is_cuda and subdiag.is_cuda, "_cuda_legacy.diag_mult requires CUDA tensors"
    return _diag_mult_cuda_module.cycle_mult(subdiag, v, shift_subdiag, shift_v)
```

### WR-05: Missing assertion that `shift_subdiag` / `shift_v` are within bounds for the Triton kernel

**File:** `torch_structured/_triton/diag_mult/op.py:127-158`

**Issue:** The kernel computes `(pos + shift_subdiag + N) % N`. This is
correct iff `shift_subdiag >= -N` (so the addend stays non-negative).
Outside that range, Triton's integer `%` semantics on negative numerator
follow C-style truncated division — i.e., the result can be negative,
producing out-of-bounds loads (silent bad reads).

For the documented use case (`shift in {-1, 0, 1}`) this is safe, but the
op's public signature accepts any int. A caller passing
`shift_v=-2*N` (a valid double-roll) would silently corrupt outputs.

**Fix:** Either (a) clamp/normalize shifts with `shift_v = shift_v % N` in
the wrapper before passing to the kernel (idempotent, matches `torch.roll`
semantics — `torch.roll` accepts any integer), or (b) assert bounds:

```python
N = v.size(-1)
assert -N <= shift_subdiag < N, f"shift_subdiag must be in [-N, N); got {shift_subdiag} for N={N}"
assert -N <= shift_v < N, f"shift_v must be in [-N, N); got {shift_v} for N={N}"
```

Option (a) is more user-friendly because it matches `torch.roll`
(unbounded), which is what the torch_ref backend already accepts via
`torch.roll(..., -shift, ...)`. Diverging backend behaviour (torch_ref
accepts any int, Triton silently corrupts) is the kind of cross-backend
inconsistency D-26 was meant to prevent.

### WR-06: `conftest.py` backend fixture re-resolves and rebinds globals during teardown — leaks across module boundaries

**File:** `tests/conftest.py:20-23`

**Issue:** The fixture restores `_BACKEND` by calling
`torch_structured._ops.set_backend(original)`. `set_backend` invokes
`_resolve()` which rebinds ALL module-level callables (`butterfly_multiply`,
`hadamard_transform`, `diag_mult`) by re-importing them. If the original
backend was `"triton"` and a Triton kernel module raised on import inside the
test (e.g., the test mutated `sys.modules` or registered conflicting ops),
`_resolve()` would re-trigger the import side effects during teardown and
potentially raise from the fixture — failing the test with a confusing
"teardown failure" instead of the real test failure.

More subtly: the fixture captures `_BACKEND` (a string), not a snapshot of
the bound callable objects. If a test directly assigns
`torch_structured._ops.diag_mult = my_mock`, the teardown only restores the
backend NAME — the mock survives if the resolver picks the same backend.
Tests in `test_diag_mult.py` don't do this, but the surface for
order-dependent test pollution exists.

**Fix:** Capture and restore the actual callable objects, not just the
backend name:

```python
@pytest.fixture(params=["torch", "triton"])
def backend(request):
    if request.param == "triton" and not torch_structured._ops._has_triton_kernel("diag_mult"):
        pytest.skip("Triton kernel for diag_mult not installed")
    saved = {
        "_BACKEND": torch_structured._ops._BACKEND,
        "butterfly_multiply": torch_structured._ops.butterfly_multiply,
        "diag_mult": torch_structured._ops.diag_mult,
        "hadamard_transform": torch_structured._ops.hadamard_transform,
    }
    chosen = torch_structured._ops.set_backend(request.param)
    yield chosen
    for k, v in saved.items():
        setattr(torch_structured._ops, k, v)
```

This is more robust but slower (no re-resolution). Acceptable for tests.

## Info

### IN-01: Docstring claim "wrap_triton in PyTorch 2.6+ accepts only plain @triton.jit" is unverifiable from the comment

**File:** `torch_structured/_triton/diag_mult/op.py:143-145`

**Issue:** The inline comment:

```python
# wrap_triton in PyTorch 2.6+ accepts only plain @triton.jit (no
# @triton.heuristics); BLOCK_SIZE is a fixed power-of-2 — the kernel is
# pointwise and not block-size sensitive.
```

is a justification for not using `@triton.heuristics` to autotune
`BLOCK_SIZE`. The constraint mentioned (heuristics incompatibility with
`wrap_triton`) is asserted without a citation to a PyTorch issue, release
note, or test. If this constraint relaxes in a future PyTorch release, the
comment becomes misleading and the hardcoded `BLOCK_SIZE = 1024` (line 142)
becomes a missed optimization opportunity.

**Fix:** Link to the PyTorch issue or commit that established the
constraint, e.g.:

```python
# wrap_triton (torch.library.triton_op) requires plain @triton.jit kernels —
# @triton.heuristics / @triton.autotune are not yet supported as of PyTorch
# 2.6 (see pytorch/pytorch#XXXXX). BLOCK_SIZE=1024 is a fixed power-of-2 — the
# kernel is purely pointwise and not block-size sensitive in practice.
```

### IN-02: `_has_any_triton_kernel()` hardcodes the op list — diverges from `set_backend` rebinding semantics

**File:** `torch_structured/_ops.py:117-129`

**Issue:** The function iterates over a hardcoded tuple
`("butterfly_multiply", "diag_mult", "hadamard_transform")`. When Phase 8+
adds a new op (e.g., `monarch_multiply`), this list must be updated in lockstep
with the `_resolve()` step-2 bindings. Easy to forget. The list also doesn't
match the order in which ops are bound in `_resolve()` (butterfly first, then
diag_mult, then hadamard) — minor inconsistency.

**Fix:** Centralize the op list as a module-level constant:

```python
_TRITON_OPS = ("butterfly_multiply", "diag_mult", "hadamard_transform")

def _has_any_triton_kernel() -> bool:
    return any(_has_triton_kernel(op) for op in _TRITON_OPS)
```

Then references in `_resolve()` and any new op additions only touch
`_TRITON_OPS`.

### IN-03: `_diag_mult_backend` is bound but only used in `log.info` — dead-codey

**File:** `torch_structured/_ops.py:221,225,229,236-239`

**Issue:** The local `_diag_mult_backend` variable is set in each of the
three diag_mult binding branches, then used only in the single
`log.info(...)` call at lines 236-239. The variable adds visual noise — the
log statement could read `_BACKEND` derivation directly via a small helper,
or the log message could be inlined into each branch.

This is purely a readability nit, but it's reinforced by the comment "the
only consumer is the log.info line below; the module-level ``_BACKEND``
global stays coarse" — which acknowledges the smell.

**Fix:** Optionally fold the binding and logging into a small helper, or
just accept the current shape (it's clear enough). Lowest-priority finding.

---

_Reviewed: 2026-05-27T14:21:05Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
