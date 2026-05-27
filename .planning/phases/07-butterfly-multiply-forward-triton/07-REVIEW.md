---
phase: 07-butterfly-multiply-forward-triton
reviewed: 2026-05-27T22:30:00Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - tests/_baseline_butterfly.py
  - tests/conftest.py
  - tests/test_butterfly_triton.py
  - torch_structured/_ops.py
  - torch_structured/_triton/butterfly/__init__.py
  - torch_structured/_triton/butterfly/op.py
findings:
  critical: 0
  warning: 3
  info: 6
  total: 9
status: issues_found
---

# Phase 7: Code Review Report

**Reviewed:** 2026-05-27T22:30:00Z
**Depth:** standard
**Files Reviewed:** 6
**Status:** issues_found

## Summary

The phase delivers a multi-launch 3-stage register-resident butterfly kernel with
fp32 + complex64 forward, two-input `register_autograd` backward (via
`torch.autograd.grad` on the `_torch_ref` oracle), `register_fake` meta with
load-bearing kwarg defaults, contiguity-guarded `view_as_real` wrapper boundary,
small-N fallback, schema-name collision fix in `_ops.py`, and a perf baseline
harness with the locked JSON schema.

Verified to be correct:
- `tl.static_assert(not IS_COMPLEX, ...)` correctly REMOVED (`grep -c` returns 0).
- IS_COMPLEX=True branch implements the 4-FMA per-pair multiply with the right
  sign convention (`(a+bi)(c+di) = (ac-bd) + (ad+bc)i`) on each of the four
  scalar complex multiplies `t00*cur, t01*partner, t10*partner, t11*cur`, then
  sums into the 2-vector outputs.
- Wrapper-boundary `assert input.is_contiguous()` AND `assert twiddle.is_contiguous()`
  precede every `view_as_real` (Pitfall 3 honored). Return path uses
  `view_as_complex(final_work.contiguous())` — `.contiguous()` is load-bearing.
- Small-N fallback at `log_n <= 1` routes through `_torch_ref` with `.clone()`
  to break the alias for log_n=0 (alias-check rejection); autograd graph stays
  uniform via the always-registered `_backward` callback.
- `register_fake` defaults `increasing_stride=True, output_size=None` mirror
  the wrapper schema defaults (load-bearing per the Phase 6 lesson).
- `_ops.py` `_TRITON_PACKAGE_NAMES` correctly maps the `butterfly_multiply` →
  `butterfly` package-name asymmetry; schema is registered as
  `torch_structured::butterfly_multiply_triton` to avoid collision with
  `csrc/butterfly.cpp`'s `TORCH_LIBRARY(torch_structured, m) { m.def("butterfly_multiply", ...) }`.
- Backward returns a 4-tuple matching the 4 forward inputs `(twiddle, input,
  increasing_stride, output_size)` with non-tensor `None`s for the last two.
- `tests/_baseline_butterfly.py` leading underscore prevents pytest collection;
  `conftest.py` registers the `slow` marker so the comprehensive tier doesn't
  emit `PytestUnknownMarkWarning`.
- pair_flat math is correct (col_start divisibility holds because
  TILE_N = 2 * 2^max_log_stride and stride ≤ 2^max_log_stride).
- Twiddle stride math agrees for both layouts: fp32 `twiddle_stage_stride = 2n`,
  complex64 view_as_real `twiddle_stage_stride = 4n` (verified by counting
  flat strides through the 6-D / 7-D shapes).
- pad/trim mirrors `_torch_ref/butterfly.py:18, 33` verbatim, including the
  `input.contiguous()` after the slice path.

Below are the issues I did find. None are correctness blockers, but three are
real warnings about robustness / test-coverage gaps, and the rest are info /
documentation drift.

## Critical Issues

None.

## Warnings

### WR-01: Triton-backend backward callback `_backward` is never exercised by any test

**File:** `tests/test_butterfly_triton.py:167-197, 327-360`
**Issue:**
- `test_butterfly_gradcheck_fp64` and `test_butterfly_gradcheck_complex64` both
  `pytest.skip("Triton kernel is fp32-only ...")` on the triton backend, and
  the docstrings argue that the torch-backend gradcheck "is testing the autograd
  plumbing for both backends" because `_backward` delegates to `_torch_ref`.
- That logic is correct ONLY about the gradient VALUES — it does NOT test the
  `register_autograd` wiring itself (return-tuple order, count, None placement,
  detach + requires_grad_ pattern, save_for_backward unpacking).
- `test_butterfly_smallN_fallback` does call `.backward()` on the triton
  backend, but it only asserts `twiddle.grad is not None` and
  `input_.grad is not None` — it never compares gradient VALUES against the
  oracle.
- Practical consequence: a bug like swapping the return order in
  `_backward` (returning `(grad_input, grad_twiddle, None, None)`) would only
  be caught accidentally by the smallN test because `twiddle.shape !=
  input.shape` — the shape-mismatch assignment to `.grad` would error out. A
  bug like dropping `requires_grad_(True)` from `input_d` would silently
  produce zero `input_.grad` and the smallN test would PASS (`grad is not
  None` is True for a zeros tensor).

**Fix:** Add a triton-backend gradient-value test that compares
gradients against the oracle directly:

```python
def test_butterfly_backward_values_triton(backend):
    """Compare backward gradients against the oracle directly on the Triton backend."""
    if backend != "triton":
        pytest.skip("torch-backend backward IS the oracle — nothing to compare against")
    log_n, nstacks, nblocks, batch_size = 4, 1, 1, 4
    n = 1 << log_n
    twiddle = torch.randn(nstacks, nblocks, log_n, n // 2, 2, 2,
                          device="cuda", dtype=torch.float32, requires_grad=True)
    input_ = torch.randn(batch_size, nstacks, n, device="cuda",
                        dtype=torch.float32, requires_grad=True)
    grad_out = torch.randn(batch_size, nstacks, n, device="cuda", dtype=torch.float32)

    out = torch_structured._ops.butterfly_multiply(twiddle, input_, True, n)
    grad_t, grad_i = torch.autograd.grad(out, [twiddle, input_], grad_out)

    # Compare against oracle path
    twiddle_d = twiddle.detach().requires_grad_(True)
    input_d = input_.detach().requires_grad_(True)
    out_ref = butterfly_ref(twiddle_d, input_d, True, n)
    grad_t_ref, grad_i_ref = torch.autograd.grad(out_ref, [twiddle_d, input_d], grad_out)

    assert torch.allclose(grad_t, grad_t_ref, rtol=RTOL, atol=ATOL)
    assert torch.allclose(grad_i, grad_i_ref, rtol=RTOL, atol=ATOL)
```

### WR-02: `_resolve()` log line misreports `butterfly_multiply` backend when triton butterfly is unavailable

**File:** `torch_structured/_ops.py:280-283`
**Issue:** The log line emitted at the end of resolution reads:
```python
log.info(
    "torch_structured: per-op bindings: butterfly_multiply=%s, diag_mult=%s, hadamard_transform=%s",
    actual, _diag_mult_backend, _hadamard_transform_backend,
)
```
This passes `actual` (coarse) for butterfly_multiply, but `_diag_mult_backend`
/ `_hadamard_transform_backend` (per-op) for the others.

When the user requests `triton` on a host that has diag_mult / hadamard
Triton kernels but no butterfly Triton kernel, `actual` becomes `"triton"`
(because `_has_any_triton_kernel()` is true), and the Step-2 binding for
butterfly_multiply correctly falls back to `_cuda_legacy` or `torch_ref`
(lines 220-230) — but the log line claims `butterfly_multiply=triton`. This
silently lies to observers and contradicts CHECKER B3 ("the resolver must be
honest about backend availability").

**Fix:** Track a per-op `_butterfly_multiply_backend` variable symmetric to
`_diag_mult_backend` and pass it to the log:

```python
if actual == "triton":
    if _has_triton_kernel("butterfly_multiply"):
        from torch_structured._triton.butterfly.op import butterfly_multiply as _triton_bm
        butterfly_multiply = _triton_bm
        _butterfly_multiply_backend = "triton"
    elif _has_cuda_legacy():
        from torch_structured._cuda_legacy import butterfly_multiply as _cuda_bm
        butterfly_multiply = _cuda_bm
        _butterfly_multiply_backend = "cuda"
    else:
        from torch_structured._torch_ref.butterfly import butterfly_multiply_torch
        butterfly_multiply = butterfly_multiply_torch
        _butterfly_multiply_backend = "torch"
# ... (similar for cuda / torch top-level branches)

log.info(
    "torch_structured: per-op bindings: butterfly_multiply=%s, diag_mult=%s, hadamard_transform=%s",
    _butterfly_multiply_backend, _diag_mult_backend, _hadamard_transform_backend,
)
```

### WR-03: `_baseline_butterfly.py` silently writes misleading JSON when Triton butterfly kernel is unavailable

**File:** `tests/_baseline_butterfly.py:121-127`
**Issue:** The script gates on `torch.cuda.is_available()` (line 91) but doesn't
verify that `set_backend("triton")` actually resolved to `"triton"`. If Triton
the package is unimportable or the kernel module fails to load,
`set_backend("triton")` silently falls back to `"cuda"` or `"torch"` (with
only a `log.warning`), then:
- `triton_call()` measures the legacy CUDA / torch_ref backend
- The JSON row records `"wall_ms_p50"` under the `triton` column anyway
- Phase 9 TEST-04 parity gate consumes this verbatim and compares the
  measured "triton" perf against a future commit — but the BASELINE was never
  actually Triton

**Fix:** Assert the actual resolved backend equals `"triton"` and fail loudly:

```python
torch_structured._ops.set_backend("triton")
if torch_structured._ops._BACKEND != "triton":
    print(f"ERROR: set_backend('triton') resolved to {torch_structured._ops._BACKEND!r}; "
          "baseline JSON would be misleading. Aborting.")
    return 1
```

This makes the script fail-closed when Triton isn't actually selected (e.g.,
on CI hosts where Triton is installed but the kernel module fails to import
due to a regression).

## Info

### IN-01: Misleading comment about `bn_id` decomposition

**File:** `torch_structured/_triton/butterfly/op.py:153-156`
**Issue:** The comment reads:
```python
# Decompose (batch, nstack) row id (consecutive bn_id values share twiddle
# because nstack_idx varies fastest in this scheme).
nstack_idx = bn_id % nstacks
```
This is backwards. With the input layout `(batch, nstacks, n)` row-major and
`bn_id = batch_idx * nstacks + nstack_idx`, consecutive `bn_id` values have
DIFFERENT `nstack_idx` (nstack varies fast → nstack_idx changes per bn_id).
So consecutive `bn_id` values DO NOT share twiddle — they use DIFFERENT
twiddle slices (per nstack). The code is correct; only the comment is wrong.
**Fix:** Rewrite as:
```python
# Decompose flattened (batch, nstack) row id; input layout is
# (batch, nstacks, n) row-major so bn_id = batch_idx * nstacks + nstack_idx,
# i.e., nstack_idx varies fastest along consecutive bn_id values. Each bn_id
# uses the nstack-specific twiddle slice (twiddle.shape[0] == nstacks).
nstack_idx = bn_id % nstacks
```

### IN-02: Stale "gated off in 07-01" comment on the IS_COMPLEX constexpr

**File:** `torch_structured/_triton/butterfly/op.py:90`
**Issue:**
```python
IS_COMPLEX: tl.constexpr,   # Phase 4 layout flag (D-44 / D-41a) — gated off in 07-01
```
The "gated off in 07-01" is stale — Plan 07-02 lit up the IS_COMPLEX=True
branch (the entire purpose of Plan 07-02). The static_assert that previously
gated the branch off is confirmed REMOVED.
**Fix:** `IS_COMPLEX: tl.constexpr,   # Phase 4 layout flag (D-44 / D-41a) — lit up in 07-02`

### IN-03: Kernel docstring contradicts implementation

**File:** `torch_structured/_triton/butterfly/op.py:95-98`
**Issue:** Docstring says:
```
Each program handles ``TILE_N`` consecutive elements of one
``(batch, nstack)`` row. Loads the input tile once into registers
(``tl.load``), runs ``STAGE_COUNT`` (1..3) butterfly stages using
``output_ptr`` as inter-stage scratch ...
```
The implementation does NOT keep the tile in registers — it seeds output_ptr
with the input tile, then reads back from output_ptr at every stage. The
following paragraph (lines 100-108) correctly explains this, but line 96
("Loads the input tile once into registers") contradicts it.
**Fix:** Replace "Loads the input tile once into registers" with "Seeds the
``output_ptr`` scratch buffer with the input tile".

### IN-04: Bind logic hardcodes `_triton.butterfly.op` path instead of using `_TRITON_PACKAGE_NAMES`

**File:** `torch_structured/_ops.py:221-224`
**Issue:** The `_has_triton_kernel("butterfly_multiply")` probe correctly
resolves the package name via `_TRITON_PACKAGE_NAMES.get("butterfly_multiply",
"butterfly_multiply")` → `"butterfly"`. The actual import that follows
hardcodes the path:
```python
from torch_structured._triton.butterfly.op import (
    butterfly_multiply as _triton_bm,
)
```
If `_TRITON_PACKAGE_NAMES` is updated (e.g., to rename the package) and the
hardcoded import is missed, the probe and bind disagree silently. Low risk
in practice since there's only one entry today, but a code smell.
**Fix:** Either inline a comment "keep in sync with `_TRITON_PACKAGE_NAMES`"
near the import, or use `importlib.import_module` symmetrically.

### IN-05: `_has_triton_kernel` swallows non-ImportError exceptions to `False`

**File:** `torch_structured/_ops.py:135-139`
**Issue:**
```python
try:
    mod = importlib.import_module(f"torch_structured._triton.{package_name}.op")
except (ImportError, AttributeError):
    return False
```
Triton kernel modules execute `@triton_op(...)` and other registration calls
at import time. A failure inside these calls might raise non-ImportError
exceptions (e.g., `RuntimeError` from `torch.library` if the schema is
malformed, `AssertionError` from `triton`'s checks, etc.). Currently those
exceptions propagate, which gives a loud signal on a real bug — that is
arguably desirable. But if the catch list is later widened (a tempting
"defensive" change), a real kernel-registration bug would be masked as
"backend unavailable". Worth a comment.
**Fix:** Add a comment near the except clause:
```python
except (ImportError, AttributeError):
    # Intentionally narrow: do NOT swallow registration errors (e.g., schema
    # conflicts from @triton_op). Those should fail loudly so the bug surfaces
    # at import time, not as a silent "backend unavailable".
    return False
```

### IN-06: `test_butterfly_smallN_fallback` asserts `.grad is not None` without checking values

**File:** `tests/test_butterfly_triton.py:163-164`
**Issue:**
```python
assert twiddle.grad is not None, "small-N fallback broke twiddle autograd"
assert input_.grad is not None, "small-N fallback broke input autograd"
```
This catches the worst-case "backward was never called" failure, but does
not detect (a) incorrect gradient values or (b) gradients that are silently
all-zero (e.g., if `requires_grad_(True)` were dropped from one of the
detached clones in `_backward`). Pair this with WR-01 to close the gap.
**Fix:** Augment with value comparison against the oracle:
```python
# Oracle gradient comparison (small-N fallback IS the oracle, so equality
# is expected modulo the .clone() in the wrapper).
twiddle_ref = twiddle.detach().requires_grad_(True)
input_ref = input_.detach().requires_grad_(True)
out_ref = butterfly_ref(twiddle_ref, input_ref, True, n)
out_ref.sum().backward()
assert torch.allclose(twiddle.grad, twiddle_ref.grad, rtol=RTOL, atol=ATOL)
assert torch.allclose(input_.grad, input_ref.grad, rtol=RTOL, atol=ATOL)
```

---

_Reviewed: 2026-05-27T22:30:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
