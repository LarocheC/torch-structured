# Phase 5: diag_mult Triton Port - Pattern Map

**Mapped:** 2026-05-27
**Files analyzed:** 12 (5 new + 7 modified)
**Analogs found:** 12 / 12 (every file has at least one in-repo analog — Phase 4 paved the way)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `torch_structured/_triton/diag_mult/__init__.py` (new) | package marker | n/a | `torch_structured/_torch_ref/__init__.py:1-4` | exact (minimal init style) |
| `torch_structured/_triton/diag_mult/op.py` (new) | Triton kernel + autograd op | pointwise GPU kernel + request-response wrapper | `torch_structured/_ops.py:225-304` (Phase 4 demonstrator) | exact (literal template; only the kernel body and signature change) |
| `torch_structured/_torch_ref/diag_mult.py` (new) | reference impl | request-response (tensor in/out) | `torch_structured/_torch_ref/butterfly.py:1-33` (file shape) + `csrc/diag_mult/diag_mult_cuda_kernel.cu:1-10` (formula) | role-match (the file shape mirrors `_torch_ref/butterfly.py`; the implementation body is freshly authored as 2 lines of `torch.roll`) |
| `torch_structured/_cuda_legacy/diag_mult.py` (new) | C++ extension passthrough | request-response | `torch_structured/_cuda_legacy/butterfly.py:1-20` (file shape) + `torch_structured/structured/krylov.py:21-24` (try-import idiom) | exact (combine the two patterns) |
| `tests/test_diag_mult.py` (new) | test module | test infra | `tests/test_dispatch.py:1-92` (pytest module style, GPU skipif, gradcheck) | role-match (analogous structure; tests target a real kernel instead of the demonstrator) |
| `torch_structured/_ops.py` (modified) | dispatch module | request-response + lazy import | self at `_ops.py:102-193` (`_resolve()`) + `_ops.py:72-99` (probes) | exact (extend in place; delete demonstrator block at 216-304) |
| `torch_structured/_torch_ref/__init__.py` (modified) | package init | n/a | self at `_torch_ref/__init__.py:1-4` | exact (extend `__all__`) |
| `torch_structured/_cuda_legacy/__init__.py` (modified) | package init | n/a | self at `_cuda_legacy/__init__.py:1-16` | exact (add conditional re-export) |
| `torch_structured/_triton/__init__.py` (verify only) | package init | n/a | self at `_triton/__init__.py:1-21` | exact (no edit required — HAS_TRITON sentinel stays) |
| `torch_structured/structured/krylov.py` (modified) | consumer | request-response | `torch_structured/_ops.py:11-39` (D-05 attribute-access contract); pattern in same file for inline call | exact (delete `CycleDownMultCuda`; rewrite call site) |
| `tests/conftest.py` (modified) | pytest fixture | test infra | self at `tests/conftest.py:1-21` | exact (widen `params` list + add skip gate) |
| `tests/test_dispatch.py` (modified or deleted) | test module | test infra | self at `tests/test_dispatch.py:1-92` (5 demonstrator tests to remove) | exact (trim or delete; planner picks per Open Question 1) |

---

## Pattern Assignments

### `torch_structured/_triton/diag_mult/__init__.py` (new, package marker)

**Analog:** `torch_structured/_torch_ref/__init__.py:1-4` (minimal package init with single re-export and `__all__`).

**Pattern to copy** (`torch_structured/_torch_ref/__init__.py:1-4`):
```python
"""Pure-PyTorch reference implementations used by the dispatch fallback path."""
from .butterfly import butterfly_multiply_torch  # noqa: F401

__all__ = ["butterfly_multiply_torch"]
```

**Apply as** (Phase 5 — note: `_has_triton_kernel("diag_mult")` only needs `op.py:diag_mult` to resolve; this `__init__.py` is for ergonomic top-level access):
```python
"""Triton kernel package for diag_mult — Phase 5 (TRI-01)."""
from .op import diag_mult  # noqa: F401

__all__ = ["diag_mult"]
```

**Notes:** The probe at `torch_structured/_ops.py:96` does `importlib.import_module(f"torch_structured._triton.{op_name}.op")` and then `hasattr(mod, op_name)`. That probe does NOT require the `__init__.py` to re-export — but mirroring `_torch_ref/__init__.py`'s convention is harmless and gives Phase 5+ consumers a stable `from torch_structured._triton.diag_mult import diag_mult` path.

---

### `torch_structured/_triton/diag_mult/op.py` (new, Triton kernel + autograd op)

**Analog:** `torch_structured/_ops.py:225-304` — the Phase 4 demonstrator op. The five-component shape (kernel + `@triton_op` wrapper + `_setup_context` + `_backward` + `@register_fake`) transcribes verbatim; only the kernel body, wrapper signature, and backward formula change.

**Imports pattern** (mirror `_ops.py:42-49`, drop unused module-level state, add `_torch_ref` import for backward oracle):
```python
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from torch_structured._torch_ref.diag_mult import diag_mult as _diag_mult_torch  # backward oracle (D-26)
```

**Kernel pattern** (`_ops.py:225-233` — Phase 4 demonstrator kernel skeleton; replace body with cycle_mult formula + `IS_COMPLEX` branch per `04-COMPLEX-LAYOUT.md:58-76`):
```python
# Phase 4 demonstrator template (lines 225-233):
@triton.jit
def _demo_identity_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)
```

**Cycle_mult kernel body** (from `csrc/diag_mult/diag_mult_cuda_kernel.cu:1-10` — verbatim formula, ported to Triton with `IS_COMPLEX` constexpr from `04-COMPLEX-LAYOUT.md:58-76`):
```c
// csrc/diag_mult/diag_mult_cuda_kernel.cu:1-10 — the formula being ported:
__global__ void subdiagMult(float *d_Subdiag, float *d_Data, float *d_Output,
                            int shiftSubdiag, int shiftV, int N, int subdiagOffset) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    float *d_Src = d_Data  + blockIdx.y * N;
    float *d_Dst = d_Output + blockIdx.y * N;
    float *d_Sub = d_Subdiag + blockIdx.y * subdiagOffset;
    if (pos < N) {
        d_Dst[pos] = d_Sub[(pos + shiftSubdiag + N) % N] * d_Src[(pos + shiftV + N) % N];
    }
}
```

Apply as a 2-D grid `(n_batch, cdiv(N, BLOCK_SIZE))` Triton kernel, gated on `IS_COMPLEX: tl.constexpr`. The complex branch follows the 4-FMA template from `04-COMPLEX-LAYOUT.md:64-71`:
```python
# 04-COMPLEX-LAYOUT.md:64-71 — 4-FMA complex multiply:
if IS_COMPLEX:
    a_re, a_im = tl.load(in_ptr + off_re), tl.load(in_ptr + off_im)
    c_re, c_im = tl.load(twiddle_ptr + t_re), tl.load(twiddle_ptr + t_im)
    out_re = a_re * c_re - a_im * c_im
    out_im = a_re * c_im + a_im * c_re
    tl.store(out_ptr + off_re, out_re)
    tl.store(out_ptr + off_im, out_im)
```

**Wrapper pattern** (`_ops.py:236-278` — `@triton_op` wrapper with `view_as_real` boundary; same assert sequence per Pitfall 3):
```python
# _ops.py:236-278 (demonstrator):
@triton_op("torch_structured::_demo_identity", mutates_args={})
def _demo_identity_op(x: torch.Tensor) -> torch.Tensor:
    is_complex = x.is_complex()
    if is_complex:
        assert x.is_contiguous(), (
            "complex input must be contiguous before view_as_real "
            "(Pitfall 3 / 04-COMPLEX-LAYOUT.md)"
        )
        x_work = torch.view_as_real(x)
    else:
        x_work = x
    out_work = torch.empty_like(x_work)
    n_elements = x_work.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    wrap_triton(_demo_identity_kernel)[grid](x_work, out_work, n_elements, BLOCK_SIZE)
    if is_complex:
        return torch.view_as_complex(out_work.contiguous())
    return out_work
```

Phase 5 wrapper changes:
- Signature `diag_mult(subdiag, v, shift_subdiag: int, shift_v: int) -> Tensor` (4 args, last two ints).
- Asserts add `assert subdiag.dtype == v.dtype` (D-20) and `assert subdiag.size(-1) == v.size(-1)` (D-19b).
- `is_batched_subdiag = (subdiag.numel() == v.numel())` (mirrors `csrc/diag_mult/diag_mult_cuda.cpp:13` `batchedSubdiag` flag).
- Grid is 2-D: `lambda meta: (n_batch, triton.cdiv(N, meta["BLOCK_SIZE"]))`.
- `subdiag_batch_stride` argument passed to kernel (`N` if batched / `2*N` if batched-complex; `0` if broadcast).
- `BLOCK_SIZE = 1024` (matches demonstrator; pointwise — not block-size sensitive per research line 86).

**setup_context + backward pattern** (`_ops.py:281-289` — Phase 4 demonstrator; Phase 5 adapts to 4-arg signature with 2 int returns of `None`):
```python
# _ops.py:281-289 (demonstrator — identity, no state saved, gradient passthrough):
def _setup_context(ctx, inputs, output):
    pass

def _backward(ctx, grad):
    return grad

_demo_identity_op.register_autograd(_backward, setup_context=_setup_context)
```

Phase 5 adaptation (RESEARCH lines 711-728 — verified `.conj()` formula for Wirtinger correctness):
```python
def _setup_context(ctx, inputs, output):
    subdiag, v, shift_subdiag, shift_v = inputs
    ctx.save_for_backward(subdiag, v)
    ctx.shift_subdiag = shift_subdiag
    ctx.shift_v = shift_v

def _backward(ctx, grad_out):
    subdiag, v = ctx.saved_tensors
    s_sub, s_v = ctx.shift_subdiag, ctx.shift_v
    # Wirtinger: .conj() on the OTHER operand (no-op for real)
    grad_subdiag = _diag_mult_torch(grad_out,        v.conj(),  -s_sub,    s_v - s_sub)
    grad_v       = _diag_mult_torch(subdiag.conj(), grad_out,   s_sub - s_v, -s_v)
    # 1-D subdiag broadcast → sum over leading dims (Pitfall 6)
    if subdiag.shape != grad_subdiag.shape:
        ndims_to_sum = grad_subdiag.dim() - subdiag.dim()
        if ndims_to_sum > 0:
            grad_subdiag = grad_subdiag.sum(dim=tuple(range(ndims_to_sum)))
    return grad_subdiag, grad_v, None, None  # 4 returns matching 4 forward inputs

diag_mult.register_autograd(_backward, setup_context=_setup_context)
```

**register_fake pattern** (`_ops.py:295-304` — the 260419-p27 fix; mandatory per D-12):
```python
# _ops.py:295-304 (demonstrator):
@_demo_identity_op.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)
```

Phase 5 adaptation:
```python
@diag_mult.register_fake
def _diag_mult_fake(subdiag, v, shift_subdiag, shift_v):
    """Meta kernel — required by Phase 4 D-12 (260419-p27 fix)."""
    return torch.empty_like(v)
```

---

### `torch_structured/_torch_ref/diag_mult.py` (new, reference impl)

**Analog (file shape):** `torch_structured/_torch_ref/butterfly.py:1-33` — minimal module-docstring + `import torch` + single function definition.

**Analog (formula source):** `csrc/diag_mult/diag_mult_cuda_kernel.cu:8` — the C++ pointwise formula `d_Sub[(pos + shiftSubdiag + N) % N] * d_Src[(pos + shiftV + N) % N]`. RESEARCH §"torch.roll Direction Sanity Check" (lines 518-540) verifies that `torch.roll(t, -shift, dims=-1)` exactly matches this convention.

**Imports + docstring pattern** (`_torch_ref/butterfly.py:1-9`):
```python
"""Pure-PyTorch reference implementation of butterfly_multiply.

Moved verbatim from torch_structured/butterfly/multiply.py:28-49 (D-09, TRI-07).
This is the reference impl consumed by the dispatch fallback path
(torch_structured/_ops.py) when the Triton kernel is unavailable and the
compiled CUDA backend is not requested.
"""
import torch
from torch.nn import functional as F
```

**Apply as:**
```python
"""Pure-PyTorch reference implementation of cycle_mult / diag_mult.

Implements ``out[pos] = subdiag[(pos + shift_subdiag + N) % N] * v[(pos + shift_v + N) % N]``
via ``torch.roll(t, -shift, dims=-1)``. Reference contract: csrc/diag_mult/diag_mult_cuda_kernel.cu:8.
Accepts real or complex inputs; broadcasts a 1-D ``subdiag`` over ``v``'s leading dims.
"""
import torch


def diag_mult(subdiag: torch.Tensor, v: torch.Tensor,
              shift_subdiag: int, shift_v: int) -> torch.Tensor:
    assert subdiag.dtype == v.dtype, \
        f"subdiag dtype {subdiag.dtype} != v dtype {v.dtype}"
    assert subdiag.size(-1) == v.size(-1), \
        f"trailing dim mismatch: subdiag {subdiag.size(-1)} vs v {v.size(-1)}"
    return torch.roll(subdiag, -shift_subdiag, dims=-1) * torch.roll(v, -shift_v, dims=-1)
```

**Asserts pattern** (mirrors `_torch_ref/butterfly.py:17,20`, which uses `assert` for preconditions per CLAUDE.md §"Error Handling"):
```python
# _torch_ref/butterfly.py:17 — established assert style:
assert twiddle.shape == (nstacks, nblocks, log_n, n // 2, 2, 2)
```

**Notes:** The function is a 1-line implementation per RESEARCH line 357. The two-line `assert` block matches the existing `butterfly_multiply_torch` precondition style. No `_diag_mult_torch` alias is needed — the function is named `diag_mult` directly (consumer in `op.py` imports as `from torch_structured._torch_ref.diag_mult import diag_mult as _diag_mult_torch`).

---

### `torch_structured/_cuda_legacy/diag_mult.py` (new, C++ extension passthrough)

**Primary analog (file shape):** `torch_structured/_cuda_legacy/butterfly.py:1-20` — module docstring + `Optional` typing + thin pass-through to `torch.ops.torch_structured.<op>`.

**Secondary analog (try-import idiom):** `torch_structured/structured/krylov.py:21-24` — try-import an `_xxx_cuda` pybind module, set to `None` on failure. This is the pattern Phase 5 is *moving* (deletion per D-25) into `_cuda_legacy/`.

**Try-import pattern** (`structured/krylov.py:21-24`):
```python
try:
    from torch_structured import _diag_mult_cuda as diag_mult_cuda
except (ImportError, RuntimeError) as e:
    diag_mult_cuda = None
```

**Pass-through pattern** (`_cuda_legacy/butterfly.py:1-20`):
```python
"""Pass-through wrapper for the compiled C++ butterfly_multiply op.

The .so is already loaded by butterfly/__init__.py at package import time. This
wrapper exposes the registered op as a plain Python callable (no
``@torch.jit.script``) so it composes cleanly with ``torch.compile`` / Inductor.
TorchScript is deprecated as of PyTorch 2.10 and composes poorly with the
post-2.6 compile path; the dispatch wrapper in ``torch_structured/_ops.py`` may
invoke this callable from inside a compiled graph. Phase 10 may absorb this
into the deprecation-warning module per ``04-DEPRECATION-PLAN.md``.
"""
from typing import Optional

import torch


def butterfly_multiply(twiddle: torch.Tensor, input: torch.Tensor, increasing_stride: bool,
                       output_size: Optional[int] = None) -> torch.Tensor:
    """Pass-through to the compiled C++ op (already loaded by butterfly/__init__.py)."""
    return torch.ops.torch_structured.butterfly_multiply(twiddle, input, increasing_stride,
                                                          output_size)
```

**Apply as** (combine both patterns + add `HAS_CUDA_LEGACY_DIAG_MULT` sentinel per RESEARCH lines 390-394):
```python
"""Pass-through wrapper for the compiled _diag_mult_cuda pybind module.

Unlike butterfly's `.so` (loaded eagerly into ``torch.ops.torch_structured`` via
``torch.ops.load_library``), the diag_mult `.so` is a pybind11 extension that
must be imported by name. The try-import here decouples module load failure
from package import. The ``_has_cuda_legacy_diag_mult()`` probe in _ops.py
checks ``HAS_CUDA_LEGACY_DIAG_MULT`` and falls back to ``_torch_ref.diag_mult``
when False (D-22).
"""
try:
    from torch_structured import _diag_mult_cuda as _diag_mult_cuda_module
except (ImportError, RuntimeError):
    _diag_mult_cuda_module = None


HAS_CUDA_LEGACY_DIAG_MULT: bool = _diag_mult_cuda_module is not None


def diag_mult(subdiag, v, shift_subdiag, shift_v):
    """Pass-through to the compiled C++ cycle_mult.

    Raises RuntimeError when the `.so` is absent — the resolver should
    probe ``HAS_CUDA_LEGACY_DIAG_MULT`` via ``_has_cuda_legacy_diag_mult()``
    before binding to this callable.
    """
    if _diag_mult_cuda_module is None:
        raise RuntimeError(
            "_diag_mult_cuda not built — caller should use "
            "_has_cuda_legacy_diag_mult() probe (D-22)"
        )
    return _diag_mult_cuda_module.cycle_mult(subdiag, v, shift_subdiag, shift_v)
```

**Notes:** Unlike `_cuda_legacy/butterfly.py` (which goes through `torch.ops.torch_structured.butterfly_multiply`), diag_mult uses pybind11 (`csrc/diag_mult/diag_mult_cuda.cpp:33-36`'s `PYBIND11_MODULE`), so the underlying call is `_diag_mult_cuda_module.cycle_mult(...)`. The `(ImportError, RuntimeError)` exception tuple matches the existing `krylov.py:21-24` convention (RuntimeError can fire for CUDA-version mismatches).

---

### `tests/test_diag_mult.py` (new, test module)

**Analog (file shape + pytest style):** `tests/test_dispatch.py:1-92` — module docstring, `pytest.mark.skipif` for GPU gating, individual `def test_*` functions, mix of eager + complex64 + gradcheck.

**Module-level skip pattern** (`tests/test_dispatch.py:22`):
```python
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="demo op is GPU-only")
```

**Apply as:**
```python
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="diag_mult tests require CUDA")
```

**Imports pattern** (`tests/test_dispatch.py:16-19` — pytest-style; mirror this convention for new files):
```python
import pytest
import torch

from torch_structured._ops import _demo_identity_op
```

**Apply as** (add `itertools` for shift grid; import reference oracle per RESEARCH lines 781-786):
```python
import itertools

import pytest
import torch

import torch_structured  # noqa: F401 — triggers extension load + _ops.py resolver
from torch_structured._torch_ref.diag_mult import diag_mult as diag_mult_ref
```

**Test structure pattern** (`tests/test_dispatch.py:25-92` — five tests, each ~10 lines):
```python
# tests/test_dispatch.py:25-29 (eager fp32 — base sanity):
def test_demo_identity_eager_fp32():
    x = torch.randn(128, device="cuda", requires_grad=True)
    y = _demo_identity_op(x)
    assert torch.equal(y, x)
```

```python
# tests/test_dispatch.py:32-43 (eager complex64 — exercises view_as_real):
def test_demo_identity_eager_complex64():
    x = torch.randn(128, dtype=torch.complex64, device="cuda", requires_grad=True)
    y = _demo_identity_op(x)
    assert torch.equal(y, x)
    assert y.dtype == torch.complex64
```

```python
# tests/test_dispatch.py:46-54 (gradcheck — D-14b, fp64):
def test_demo_identity_gradcheck():
    x = torch.randn(16, dtype=torch.float64, device="cuda", requires_grad=True)
    assert torch.autograd.gradcheck(_demo_identity_op, (x,), eps=1e-6, atol=1e-5)
```

**Apply as** (verbatim from RESEARCH Example G, lines 776-832 — 5 tests covering D-29):
- `test_diag_mult_eager_fp32(backend)` — uses the parametrized `backend` fixture from conftest.py
- `test_diag_mult_eager_complex64(backend)` — exercises view_as_real/view_as_complex
- `test_diag_mult_gradcheck_fp64_real(backend)` — D-26 gradcheck (fp64 real)
- `test_diag_mult_gradcheck_fp64_complex(backend)` — D-26 gradcheck (complex128) — the Wirtinger validator
- `test_diag_mult_shift_grid(backend, s_sub, s_v)` — `@pytest.mark.parametrize` over `{-1,0,1}^2` shift grid

**Notes:** This file replaces (and supersedes) the 5 demonstrator-specific tests in `tests/test_dispatch.py`. Each test calls `torch_structured._ops.diag_mult(...)` (attribute access per D-05) so `set_backend()` rebindings from the `backend` fixture take effect. The `backend` fixture from `conftest.py` provides the `["torch", "triton"]` parametrization.

---

### `torch_structured/_ops.py` (modified, dispatch module)

**Self-analog (resolver shape):** `torch_structured/_ops.py:102-193` — `_resolve()` already has the three-branch structure (`triton` / `cuda` / `torch`) for `butterfly_multiply`. D-22 extends each branch to also bind `diag_mult` with its own per-op probe.

**Probe pattern** (`_ops.py:72-99` — existing `_has_cuda_legacy()` and `_has_triton_kernel(op_name)`):
```python
# _ops.py:72-79 — existing probe for butterfly's .so:
def _has_cuda_legacy() -> bool:
    """Return True iff the compiled C++ butterfly op is registered.

    The .so is loaded as a side effect of importing
    ``torch_structured.butterfly`` (see butterfly/__init__.py:22-39). This
    probe simply checks whether the registration succeeded.
    """
    return hasattr(torch.ops.torch_structured, "butterfly_multiply")
```

**Apply as** (new probe — symmetric to `_has_cuda_legacy`, per RESEARCH lines 566-574; checks the `HAS_CUDA_LEGACY_DIAG_MULT` sentinel set in `_cuda_legacy/diag_mult.py`):
```python
def _has_cuda_legacy_diag_mult() -> bool:
    """True iff _cuda_legacy/diag_mult.py imported the .so successfully (D-22).

    Unlike butterfly's `.so` (loaded eagerly into torch.ops via load_library),
    diag_mult uses a pybind11 module imported by name. This probe checks the
    sentinel set at module import time.
    """
    try:
        from torch_structured._cuda_legacy.diag_mult import HAS_CUDA_LEGACY_DIAG_MULT
        return HAS_CUDA_LEGACY_DIAG_MULT
    except ImportError:
        return False
```

**Resolver per-op binding pattern** (`_ops.py:161-174`):
```python
# _ops.py:161-174 — existing three-branch bind for butterfly_multiply:
if actual == "triton":
    from torch_structured._triton.butterfly.op import (  # type: ignore[import-not-found]
        butterfly_multiply as _triton_bm,
    )
    butterfly_multiply = _triton_bm
elif actual == "cuda":
    from torch_structured._cuda_legacy import butterfly_multiply as _cuda_bm
    butterfly_multiply = _cuda_bm
else:  # actual == "torch"
    from torch_structured._torch_ref.butterfly import butterfly_multiply_torch
    butterfly_multiply = butterfly_multiply_torch

# hadamard_transform / diag_mult: Phase 6 / Phase 5 populate; stay None for now.
```

**Apply as** (add a parallel three-branch bind for `diag_mult` after the existing butterfly block — per D-22 use a per-op probe so asymmetric "cuda butterfly + torch diag_mult" works transparently). The decision per Open Question / D-22a recommendation: keep `_BACKEND` coarse global. Emit a single `log.info` line after binding listing the actual per-op bindings:
```python
# Phase 5: diag_mult per-op resolution. The `actual` variable is the coarse
# global backend choice; per-op falls back when its specific probe is False.
if actual == "triton" and _has_triton_kernel("diag_mult"):
    from torch_structured._triton.diag_mult.op import diag_mult as _triton_dm
    diag_mult = _triton_dm
    _diag_mult_backend = "triton"
elif actual == "cuda" and _has_cuda_legacy_diag_mult():
    from torch_structured._cuda_legacy.diag_mult import diag_mult as _cuda_dm
    diag_mult = _cuda_dm
    _diag_mult_backend = "cuda"
else:
    # Fallback path — includes the D-22 asymmetric case:
    # `actual == "cuda" but _has_cuda_legacy_diag_mult() is False`
    from torch_structured._torch_ref.diag_mult import diag_mult as _torch_dm
    diag_mult = _torch_dm
    _diag_mult_backend = "torch"
    if actual == "cuda":
        log.warning(
            "set_backend('cuda') requested but _diag_mult_cuda not built; "
            "falling back to torch_ref for diag_mult (D-22)"
        )

log.info(
    "torch_structured: per-op bindings: butterfly_multiply=%s, diag_mult=%s",
    actual, _diag_mult_backend,
)
```

**Demonstrator deletion (D-27):** Delete lines 216-304 of `_ops.py` (the `_demo_identity_kernel`, `_demo_identity_op`, `_setup_context`, `_backward`, `register_autograd` line, and `register_fake` block). Keep the module-level `import triton`, `import triton.language as tl`, and `from torch.library import triton_op, wrap_triton` (lines 47-49) — Phase 5+ kernel imports need them, even though they're not directly used in `_ops.py` after the delete. (Optional: planner may move those imports into the new `_triton/diag_mult/op.py` and remove them from `_ops.py` since `_ops.py` no longer instantiates Triton ops directly — recommend keeping in `_ops.py` for now to keep the diff small.)

**D-08 heads-up log (already in place):** Lines 185-190 of `_ops.py` already implement the D-08 INFO log; in Phase 5 this fires for the first time when both `_has_triton_kernel("diag_mult")` and `_has_cuda_legacy()` are True (and a Triton kernel is wired). No edit required to that block.

**Notes:** This is the most touched file in Phase 5 (extensions to existing patterns + a 89-line deletion). The planner MUST verify that the `auto` path also resolves correctly — currently `_ops.py:124` checks `_has_triton_kernel("butterfly_multiply")` for the auto decision; Phase 5 does NOT change the auto-precedence logic (the coarse-vs-per-op question stays per D-22a recommendation).

---

### `torch_structured/_torch_ref/__init__.py` (modified, package init)

**Self-analog:** `torch_structured/_torch_ref/__init__.py:1-4` — the current state.

**Current state:**
```python
"""Pure-PyTorch reference implementations used by the dispatch fallback path."""
from .butterfly import butterfly_multiply_torch  # noqa: F401

__all__ = ["butterfly_multiply_torch"]
```

**Apply as** (extend with diag_mult re-export):
```python
"""Pure-PyTorch reference implementations used by the dispatch fallback path."""
from .butterfly import butterfly_multiply_torch  # noqa: F401
from .diag_mult import diag_mult  # noqa: F401

__all__ = ["butterfly_multiply_torch", "diag_mult"]
```

**Notes:** No naming conflict (the function is named `diag_mult`, not `diag_mult_torch` — consumers disambiguate at the import site with `as` aliases per `op.py`'s pattern: `from torch_structured._torch_ref.diag_mult import diag_mult as _diag_mult_torch`).

---

### `torch_structured/_cuda_legacy/__init__.py` (modified, package init)

**Self-analog:** `torch_structured/_cuda_legacy/__init__.py:1-16` — current state imports butterfly unconditionally.

**Current state:**
```python
"""Wrapper around the already-loaded torch.ops.torch_structured.* C++ ops.
[...]
"""
from .butterfly import butterfly_multiply  # noqa: F401

__all__ = ["butterfly_multiply"]
```

**Apply as** (add conditional diag_mult re-export — the `.diag_mult` submodule itself handles the try-import, so importing the module is always safe; only the bound name may resolve to a callable that raises RuntimeError at call time):
```python
"""Wrapper around the already-loaded torch.ops.torch_structured.* C++ ops.
[... existing docstring unchanged ...]
"""
from .butterfly import butterfly_multiply  # noqa: F401
from .diag_mult import diag_mult  # noqa: F401  — may raise RuntimeError if .so absent

__all__ = ["butterfly_multiply", "diag_mult"]
```

**Notes:** Unlike `_cuda_legacy/butterfly.py` (whose underlying `.so` is loaded by `butterfly/__init__.py` side-effects), `_cuda_legacy/diag_mult.py` performs its own try-import. Importing the module is always safe; the resolver in `_ops.py` checks the `HAS_CUDA_LEGACY_DIAG_MULT` sentinel before binding (per D-21 honest-probe pattern).

---

### `torch_structured/_triton/__init__.py` (verify only — no edit)

**Self-analog:** `torch_structured/_triton/__init__.py:1-21` — current state imports triton and sets `HAS_TRITON` sentinel.

**Notes:** No edit required. The probe at `_ops.py:96` does `importlib.import_module(f"torch_structured._triton.{op_name}.op")` — it loads `_triton/diag_mult/op.py` directly, not through this `__init__.py`. The existing `HAS_TRITON` sentinel stays as-is. (Optional ergonomic improvement: add `from .diag_mult.op import diag_mult` here too, but it's not required and risks an early triton import for non-Triton users. Recommend leaving untouched.)

---

### `torch_structured/structured/krylov.py` (modified, consumer)

**Self-analog (try-import to remove):** `torch_structured/structured/krylov.py:21-24`:
```python
try:
    from torch_structured import _diag_mult_cuda as diag_mult_cuda
except (ImportError, RuntimeError) as e:
    diag_mult_cuda = None
```

**Self-analog (autograd Function to delete):** `torch_structured/structured/krylov.py:325-339`:
```python
class CycleDownMultCuda(torch.autograd.Function):
    '''Cycle v down and do pointwise multiplication with subdiag.
    '''
    @staticmethod
    def forward(ctx, subdiag, v):
        ctx.save_for_backward(subdiag, v)
        return diag_mult_cuda.cycle_mult(subdiag, v, 0, -1)

    @staticmethod
    def backward(ctx, grad):
        subdiag, v = ctx.saved_tensors
        return diag_mult_cuda.cycle_mult(grad, v, 0, -1).sum(dim=0), diag_mult_cuda.cycle_mult(subdiag, grad, 1, 1)


cycle_down_mult = CycleDownMultCuda.apply
```

**Self-analog (call site to rewrite):** `torch_structured/structured/krylov.py:342-344`:
```python
def subdiag_linear_map_cuda(subdiag, upper_right_corner=0.0):
    subdiag_extended = torch.cat((torch.tensor([upper_right_corner], dtype=subdiag.dtype, device=subdiag.device), subdiag))
    return lambda v: cycle_down_mult(subdiag_extended, v)
```

**Apply as** (delete lines 21-24, delete lines 325-339, rewrite line 344):

1. Delete the try-import block (lines 21-24) entirely. No replacement needed at top-of-file (the new call path goes through `torch_structured._ops`).

2. Delete the `CycleDownMultCuda` class and `cycle_down_mult = ...` assignment (lines 325-339).

3. Add `import torch_structured` (or `from torch_structured import _ops`) somewhere near the top imports if not already present. The existing imports show `from torch.nn import functional as F`, `from ._compat import krylov_construct` — but no top-level `torch_structured` import yet. Add `import torch_structured` after the `import torch` line.

4. Rewrite the `subdiag_linear_map_cuda` lambda to use `_ops.diag_mult` per attribute-access D-05 contract (`torch_structured/_ops.py:11-39` is the load-bearing module docstring):

```python
def subdiag_linear_map_cuda(subdiag, upper_right_corner=0.0):
    subdiag_extended = torch.cat((torch.tensor([upper_right_corner], dtype=subdiag.dtype, device=subdiag.device), subdiag))
    return lambda v: torch_structured._ops.diag_mult(subdiag_extended, v, 0, -1)
```

**D-05 attribute-access pattern** (`torch_structured/_ops.py:11-39` — module docstring):
```python
# torch_structured/_ops.py:18-28 (CORRECT pattern, copied from the docstring):
#
# CORRECT — attribute access (re-reads binding on each call)::
#
#     import torch_structured
#
#     def some_function(twiddle, x, ...):
#         return torch_structured._ops.butterfly_multiply(twiddle, x, ...)
```

**Notes:** The lambda capture of `subdiag_extended` is unchanged (RESEARCH Open Question 5 — closure semantics are standard Python). The `subdiag_extended` tensor from `torch.cat` is contiguous (RESEARCH Pitfall 3 — verified), so the `assert subdiag.is_contiguous()` in `op.py` will not fire. The downstream consumer `subdiag_mult_cuda` at lines 347-350 does not change. Phase 5 SC#3 (LDRSubdiagonalC forward+backward unchanged) is satisfied by the autograd rerouting through `_ops.diag_mult.register_autograd`.

---

### `tests/conftest.py` (modified, pytest fixture)

**Self-analog:** `tests/conftest.py:1-21` — current state with `params=["torch"]`.

**Current state:**
```python
"""Phase 4: backend fixture parametrized over ["torch"] only.
[... existing docstring unchanged ...]
"""
import pytest

import torch_structured  # noqa: F401 — triggers extension load + _ops.py resolver


@pytest.fixture(params=["torch"])
def backend(request):
    """Switch backend for the duration of a test, restore after."""
    original = torch_structured._ops._BACKEND
    chosen = torch_structured._ops.set_backend(request.param)
    yield chosen
    torch_structured._ops.set_backend(original)
```

**Apply as** (widen params per D-30; add skip-gate for Triton-without-CUDA per RESEARCH Example F, lines 766-774):
```python
"""Phase 5: backend fixture widened to ["torch", "triton"] per D-30.

Phase 7+ will extend to ``["torch", "triton", "cuda"]`` once the CUDA backend
axis is added per the milestone-wide TEST-03 (integration hardening).
The Triton parametrization is skipped on hosts without a registered Triton
diag_mult kernel (CPU-only runners, no-Triton envs).
"""
import pytest

import torch_structured  # noqa: F401 — triggers extension load + _ops.py resolver


@pytest.fixture(params=["torch", "triton"])
def backend(request):
    """Switch backend for the duration of a test, restore after."""
    if request.param == "triton" and not torch_structured._ops._has_triton_kernel("diag_mult"):
        pytest.skip("Triton kernel for diag_mult not installed (no CUDA or CPU-only runner)")
    original = torch_structured._ops._BACKEND
    chosen = torch_structured._ops.set_backend(request.param)
    yield chosen
    torch_structured._ops.set_backend(original)
```

**Notes:** The skip-gate uses the Phase 4 honest probe `_has_triton_kernel("diag_mult")` (defined at `_ops.py:82-99`). When `_triton/diag_mult/op.py` ships in Phase 5, the probe returns True on CUDA hosts. The existing `_BACKEND` snapshot + restore pattern is preserved for test order-independence.

---

### `tests/test_dispatch.py` (modified or deleted, test module)

**Self-analog:** `tests/test_dispatch.py:1-92` — the 5 demonstrator tests to delete per D-28.

**Current state:**
- `test_demo_identity_eager_fp32` (lines 25-29)
- `test_demo_identity_eager_complex64` (lines 32-43)
- `test_demo_identity_gradcheck` (lines 46-54)
- `test_demo_identity_compile_no_graph_break` (lines 57-71)
- `test_demo_identity_compile_fake_tensor_trace` (lines 74-92)

All five import `_demo_identity_op` from `torch_structured._ops` (line 19) — that symbol disappears after D-27 deletion. Test file CANNOT survive unchanged.

**Apply as (planner choice per Open Question 1):**

**Option A — Delete file entirely:**
- Remove `tests/test_dispatch.py`. The replacement `tests/test_diag_mult.py` carries the kernel correctness load.

**Option B — Keep as thin set_backend smoke tests (RESEARCH recommendation):**
Replace with 2-3 cross-cutting dispatch tests that DO NOT require GPU:
```python
"""Cross-cutting dispatch tests (set_backend round-trip, env-var override,
ValueError on unknown backend, B3 honest-probe). The demonstrator-specific
tests were removed per D-28 — kernel correctness moved to test_diag_mult.py.
"""
import pytest

import torch_structured


def test_set_backend_round_trip():
    """set_backend('torch') always succeeds; the call is idempotent."""
    chosen = torch_structured._ops.set_backend("torch")
    assert chosen == "torch"
    # Restore (the conftest backend fixture handles this for parametrized tests)


def test_unknown_backend_raises_value_error():
    """Resolver rejects arbitrary env-var values (T-04-01 mitigation)."""
    with pytest.raises(ValueError, match="Unknown backend"):
        torch_structured._ops.set_backend("nonsense")


def test_has_triton_kernel_probe_returns_bool():
    """B3 honest probe — never raises on a missing op; returns a clean bool."""
    assert isinstance(torch_structured._ops._has_triton_kernel("diag_mult"), bool)
    assert isinstance(torch_structured._ops._has_triton_kernel("does_not_exist"), bool)
```

**Recommend Option B** (per RESEARCH Open Question 1 line 926) — keeps regression coverage for the dispatch contract that Phase 6/7 will edit again. Drop the `pytestmark = pytest.mark.skipif(...)` line since these no longer require GPU.

**Notes:** Either option satisfies D-28. Planner picks Option A only if the 3 cross-cutting tests duplicate existing coverage. The `B3 honest-probe` line is the cheapest insurance against `_has_triton_kernel` ever raising (which it doesn't today, but a future refactor might break).

---

## Shared Patterns

### Try-import + module-level sentinel (compiled-extension presence detection)

**Source:** `torch_structured/structured/krylov.py:21-24` AND `torch_structured/structured/hadamard.py:1-8`
**Apply to:** `_cuda_legacy/diag_mult.py` (Phase 5 new file); `_ops.py:_has_cuda_legacy_diag_mult()` probe

```python
# krylov.py:21-24 — the pattern being relocated:
try:
    from torch_structured import _diag_mult_cuda as diag_mult_cuda
except (ImportError, RuntimeError) as e:
    diag_mult_cuda = None
```

The `(ImportError, RuntimeError)` exception tuple is non-negotiable — RuntimeError can fire for CUDA-version mismatches.

---

### Demonstrator op template (the full triton_op + autograd + fake pipeline)

**Source:** `torch_structured/_ops.py:225-304` (Phase 4 demonstrator op)
**Apply to:** `torch_structured/_triton/diag_mult/op.py` (Phase 5 — and Phase 6 hadamard, Phase 7 butterfly)

The five components in fixed order:
1. `@triton.jit` kernel
2. `@triton_op("torch_structured::<name>", mutates_args={})` wrapper
3. `_setup_context(ctx, inputs, output)` + `_backward(ctx, grad)` + `op.register_autograd(_backward, setup_context=_setup_context)`
4. `@op.register_fake` meta kernel
5. `view_as_real`/`view_as_complex` wrapper boundary per `04-COMPLEX-LAYOUT.md`

This is enforced by Phase 4 D-12. Diverging from this shape is the textbook regression PITFALLS §3 calls out.

---

### `assert` for preconditions (CLAUDE.md §"Error Handling")

**Source:** `torch_structured/_torch_ref/butterfly.py:17,20` and `torch_structured/_ops.py:255-258`
**Apply to:** All wrapper/reference functions in Phase 5 (`op.py`, `_torch_ref/diag_mult.py`)

```python
# _ops.py:255-258 (demonstrator wrapper assert):
assert x.is_contiguous(), (
    "complex input must be contiguous before view_as_real "
    "(Pitfall 3 / 04-COMPLEX-LAYOUT.md)"
)
```

Phase 5 wrapper assertions per D-19b / D-20:
- `assert subdiag.dtype == v.dtype` — D-20 mixed-dtype rejection
- `assert v.is_contiguous()` — Pitfall 3 (before `view_as_real`)
- `assert subdiag.is_contiguous()` — Pitfall 3
- `assert subdiag.size(-1) == v.size(-1)` — D-19b

NO `if x: raise ValueError(...)` patterns — those conflict with established convention.

---

### `# noqa: F401` re-export comment style

**Source:** `torch_structured/_torch_ref/__init__.py:2`, `torch_structured/_cuda_legacy/__init__.py:14`, `torch_structured/_triton/__init__.py:16`
**Apply to:** All package init re-exports in Phase 5

Both bare `# noqa: F401` and annotated `# noqa: F401 (re-exported)` forms are accepted; prefer annotated form for new shims so intent is obvious.

---

### Module-import-as-side-effect comment

**Source:** `tests/conftest.py:12` (`import torch_structured  # noqa: F401 — triggers extension load + _ops.py resolver`)
**Apply to:** `tests/test_diag_mult.py` (any test that needs the `_ops.py` resolver to have run before backend probing)

```python
import torch_structured  # noqa: F401 — triggers extension load + _ops.py resolver
```

---

### Attribute-access call contract (D-05)

**Source:** `torch_structured/_ops.py:11-39` (module docstring) — the load-bearing call-site contract
**Apply to:** `structured/krylov.py` (Phase 5 rewrite) and `tests/test_diag_mult.py` (every kernel call site)

CORRECT pattern:
```python
import torch_structured
# ... inside a function or lambda:
return torch_structured._ops.diag_mult(subdiag, v, 0, -1)
```

WRONG pattern (would freeze the binding at import time, missing `set_backend()` rebindings):
```python
from torch_structured._ops import diag_mult  # ← DO NOT use
def f(s, v): return diag_mult(s, v, 0, -1)
```

This is the single most consumer-facing contract Phase 5 introduces.

---

## No Analog Found

No file in Phase 5 lacks an in-repo analog. Phase 4 created the dispatch + `_torch_ref` + `_cuda_legacy` + Triton-package shape; Phase 5 transcribes those shapes to new ops. Every new file maps to a Phase 4 file:

| Phase 5 File | Phase 4 Analog | Mapping Type |
|--------------|----------------|--------------|
| `_triton/diag_mult/__init__.py` | `_torch_ref/__init__.py:1-4` | minimal init |
| `_triton/diag_mult/op.py` | `_ops.py:225-304` (demonstrator) | literal template |
| `_torch_ref/diag_mult.py` | `_torch_ref/butterfly.py:1-33` | file shape |
| `_cuda_legacy/diag_mult.py` | `_cuda_legacy/butterfly.py:1-20` + `krylov.py:21-24` | shape + try-import |
| `tests/test_diag_mult.py` | `tests/test_dispatch.py:1-92` | pytest module style |

The single non-in-repo source is `04-COMPLEX-LAYOUT.md:58-76` (the 4-FMA template inside the Triton kernel). That document was Phase 4's deliverable specifically to be the canonical template for Phase 5+; treating it as an in-repo source is correct.

---

## Metadata

**Analog search scope:** `torch_structured/_ops.py`, `torch_structured/_torch_ref/`, `torch_structured/_cuda_legacy/`, `torch_structured/_triton/`, `torch_structured/structured/krylov.py`, `torch_structured/structured/hadamard.py`, `tests/conftest.py`, `tests/test_dispatch.py`, `csrc/diag_mult/`
**Files scanned:** 13 source files + 2 test files + 2 C++/CUDA files + 4 phase docs from Phase 4 + 2 phase docs from Phase 5
**Key analogs identified by full path:**
- `torch_structured/_ops.py:225-304` — Phase 4 demonstrator (literal template for `_triton/diag_mult/op.py`)
- `torch_structured/_ops.py:72-99` — probe pattern (template for `_has_cuda_legacy_diag_mult()`)
- `torch_structured/_ops.py:102-193` — `_resolve()` per-op binding loop (extension point for diag_mult)
- `torch_structured/_ops.py:11-39` — D-05 attribute-access call contract (load-bearing for krylov.py rewrite)
- `torch_structured/_torch_ref/butterfly.py:1-33` — file shape for `_torch_ref/diag_mult.py`
- `torch_structured/_cuda_legacy/butterfly.py:1-20` — file shape for `_cuda_legacy/diag_mult.py`
- `torch_structured/structured/krylov.py:21-24` — try-import idiom (the pattern being relocated to `_cuda_legacy/diag_mult.py`)
- `torch_structured/structured/krylov.py:325-339` — `CycleDownMultCuda` to delete
- `torch_structured/structured/krylov.py:342-344` — call site to rewrite
- `tests/conftest.py:1-21` — backend fixture to widen
- `tests/test_dispatch.py:1-92` — 5 demonstrator tests to remove + style template for `test_diag_mult.py`
- `csrc/diag_mult/diag_mult_cuda_kernel.cu:1-10` — C++ formula being ported (Triton kernel reference)
- `csrc/diag_mult/diag_mult_cuda.cpp:5-16` — `cycle_mult` op signature + `batchedSubdiag` flag (wrapper boundary reference)
- `04-COMPLEX-LAYOUT.md:32-50, 58-76` — wrapper boundary + 4-FMA kernel template (canonical for Phase 5)
- `04-PATTERNS.md` (Phase 4) — pattern-extraction methodology + shared-patterns conventions reused here

**Pattern extraction date:** 2026-05-27
