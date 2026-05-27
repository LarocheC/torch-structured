# Phase 6: hadamard Triton Port - Pattern Map

**Mapped:** 2026-05-27
**Files analyzed:** 10 (5 new + 5 modified)
**Analogs found:** 10 / 10 (every file has a Phase 5 mirror — Phase 6 is a near-verbatim transcription)
**Phase 5 status:** SHIPPED — its files are the literal templates. Verbatim file-shape copy with hadamard-specific substantive divergences flagged below.

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `torch_structured/_triton/hadamard_transform/__init__.py` (new) | package marker | n/a | `torch_structured/_triton/diag_mult/__init__.py:1-4` | exact (verbatim file shape) |
| `torch_structured/_triton/hadamard_transform/op.py` (new) | Triton kernel + autograd op | log_n-stage butterfly + request-response wrapper | `torch_structured/_triton/diag_mult/op.py:1-206` | exact-skeleton (only kernel body + backward formula + signature change) |
| `torch_structured/_torch_ref/hadamard.py` (new) | reference impl | request-response | `torch_structured/_torch_ref/diag_mult.py:1-56` (file shape) + `structured/hadamard.py:15-30` (function body) | exact (move + adapt) |
| `torch_structured/_cuda_legacy/hadamard.py` (new) | C++ extension passthrough | request-response | `torch_structured/_cuda_legacy/diag_mult.py:1-46` | exact (verbatim try-import + sentinel idiom) |
| `tests/structured/test_hadamard_triton.py` (new) | test module | test infra | `tests/test_diag_mult.py:1-119` | role-match (parametrized backend fixture + gradcheck; structural twin) |
| `torch_structured/_ops.py` (modified) | dispatch module | request-response + lazy import | self at `_ops.py:82-94` (probe pattern), `_ops.py:218-239` (diag_mult per-op binding) | exact (extend in place — add hadamard probe + per-op binding block) |
| `torch_structured/_torch_ref/__init__.py` (modified) | package init | n/a | self at `_torch_ref/__init__.py:1-5` | exact (extend `__all__`) |
| `torch_structured/structured/hadamard.py` (modified) | consumer + back-compat shim | request-response | `structured/krylov.py:16,321-339` (consumer delete + import pattern); 05-PATTERNS.md back-compat shim D-33d | exact (delete-then-shim) |
| `torch_structured/structured/fastfood.py` (modified) | consumer | request-response | `torch_structured/structured/krylov.py:16,333` (D-05 attribute-access call site) | exact (verbatim D-05 idiom) |
| `tests/conftest.py` (modified) | pytest fixture | test infra | self at `tests/conftest.py:18` | exact (widen skip-gate from per-op to any-Triton) |

---

## Pattern Assignments

### `torch_structured/_triton/hadamard_transform/__init__.py` (new, package marker)

**Analog:** `torch_structured/_triton/diag_mult/__init__.py:1-4`

**Pattern to copy** (verbatim):
```python
"""Triton kernel package for diag_mult — Phase 5 (TRI-01)."""
from .op import diag_mult  # noqa: F401 (re-exported)

__all__ = ["diag_mult"]
```

**Apply as** (substitute name + REQ id):
```python
"""Triton kernel package for hadamard_transform — Phase 6 (TRI-02)."""
from .op import hadamard_transform  # noqa: F401 (re-exported)

__all__ = ["hadamard_transform"]
```

**Notes:** The `_has_triton_kernel("hadamard_transform")` probe at `_ops.py:97-114` `importlib.import_module`s `torch_structured._triton.hadamard_transform.op` and checks `hasattr(mod, "hadamard_transform")` — this `__init__.py` is purely for ergonomic top-level access.

---

### `torch_structured/_triton/hadamard_transform/op.py` (new, Triton kernel + autograd op)

**Analog:** `torch_structured/_triton/diag_mult/op.py:1-206` — five-component skeleton (`@triton.jit` kernel + `@triton_op` wrapper + `_setup_context` + `_backward` + `@register_fake`) transcribes verbatim. The substantive divergences are flagged below.

**Imports pattern** (`_triton/diag_mult/op.py:21-26` — copy verbatim, substitute name):
```python
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from torch_structured._torch_ref.diag_mult import diag_mult as _diag_mult_torch  # backward oracle (D-26)
```

**Apply as:**
```python
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from torch_structured._torch_ref.hadamard import hadamard_transform_torch as _hadamard_transform_torch  # backward oracle (D-32, self-inverse)
```

**Kernel skeleton** (`_triton/diag_mult/op.py:29-94` — skeleton shape; the body diverges substantively):
```python
@triton.jit
def _cycle_mult_kernel(
    subdiag_ptr, v_ptr, out_ptr,
    n_batch, N, subdiag_batch_stride, shift_subdiag, shift_v,
    IS_COMPLEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    bid = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    pos = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pos < N
    # ... pointwise multiply with cyclic shift indexing ...
    a = tl.load(subdiag_ptr + sub_row_base + sub_idx, mask=mask)
    c = tl.load(v_ptr + v_row_base + v_idx, mask=mask)
    tl.store(out_ptr + v_row_base + pos, a * c, mask=mask)
```

**SUBSTANTIVE DIVERGENCE — hadamard kernel body** (per D-31, D-31a; not transcribed from `diag_mult/op.py`):

The hadamard kernel is **single-pass shared-memory** — one launch does all `log_n` butterfly stages. Grid is 1-D `(n_batch,)` (not 2-D — `BLOCK_SIZE` already covers a full row of N). Signature is `(u_ptr, out_ptr, n_batch, N, normalize_scale, BLOCK_SIZE: tl.constexpr, LOG_N: tl.constexpr)`. Body shape (planner authors fresh — no in-repo Triton analog yet):

```python
@triton.jit
def _hadamard_kernel(
    u_ptr, out_ptr,
    n_batch, N,
    BLOCK_SIZE: tl.constexpr,  # == N (power-of-2 padded; SC#1 caps N at 4096)
    LOG_N: tl.constexpr,
):
    bid = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    row_base = bid * N
    # Single tl.load: all log_n stages happen in registers/shared memory
    x = tl.load(u_ptr + row_base + offsets, mask=mask)
    # Unrolled log_n butterfly stages — Walsh-Hadamard FWT, pairs at stride k
    for k in tl.static_range(LOG_N):
        stride = 1 << k
        # Pair (i, i + stride): x[i], x[i + stride] -> (x[i] + x[i + stride], x[i] - x[i + stride])
        # In Triton this is done via xor: partner = x ^ stride; "lower half" mask via (offsets & stride) == 0
        # The exact masking pattern is the planner's call (see D-31b)
        ...
    tl.store(out_ptr + row_base + offsets, x, mask=mask)
```

Reference for the math (CUDA kernel `csrc/hadamard/hadamard_cuda_kernel.cu`, lines 39-86 of `fwtBatch1Kernel`) — Triton transcription is at planner discretion per D-31c. The CUDA two-pass mixed-radix `fwtBatch2Kernel` is explicitly out of scope (D-31c).

**`@triton_op` wrapper** (`_triton/diag_mult/op.py:97-162` — shape; signature differs):

Diag_mult wrapper (skeleton):
```python
@triton_op("torch_structured::diag_mult", mutates_args={})
def diag_mult(subdiag: torch.Tensor, v: torch.Tensor, shift_subdiag: int,
              shift_v: int) -> torch.Tensor:
    assert subdiag.dtype == v.dtype, ...  # D-20 mixed-dtype
    assert v.is_contiguous(), ...           # Pitfall 3
    assert subdiag.is_contiguous(), ...
    assert subdiag.size(-1) == v.size(-1), ...  # D-19b
    N = v.size(-1)
    n_batch = v.numel() // N
    # view_as_real for complex inputs
    if is_complex:
        v_work = torch.view_as_real(v)
        ...
    out_work = torch.empty_like(v_work)
    grid = lambda meta: (n_batch, triton.cdiv(N, meta["BLOCK_SIZE"]))
    wrap_triton(_cycle_mult_kernel)[grid](...)
    if is_complex:
        return torch.view_as_complex(out_work.contiguous())
    return out_work
```

**SUBSTANTIVE DIVERGENCES (hadamard wrapper):**
- Signature: `hadamard_transform(u: torch.Tensor, normalize: bool = False) -> torch.Tensor` (single tensor + bool, not 4-arg).
- Asserts (per D-31 / D-35 / WR-05 analog):
  - `assert u.dim() >= 1`, `assert u.size(-1) >= 1`
  - `assert u.dtype == torch.float32`, `f"hadamard kernel is fp32-only, got {u.dtype}"`  (no IS_COMPLEX branch — D-31c real-only)
  - `assert u.is_contiguous()` — Pitfall 3 analog
  - `n = u.size(-1)`, `log_n = int(n.bit_length() - 1)`, `assert n == 1 << log_n, f"n must be a power of 2, got {n}"` (matches `structured/hadamard.py:26` and `csrc/hadamard/hadamard_cuda.cpp:9` contract)
  - `assert log_n <= 12, f"single-pass kernel caps log_n at 12 (SC#1), got log_n={log_n}"` (D-31c)
- No `view_as_real` / `view_as_complex` round-trip (real-only).
- No `IS_COMPLEX: tl.constexpr` — kernel signature lacks this constexpr.
- `BLOCK_SIZE = n` (rounded up to next power-of-2 if needed, but n is already pow2). `LOG_N = log_n`.
- 1-D grid: `grid = lambda meta: (n_batch,)` — not 2-D; the kernel covers a whole row per block.
- Normalization (D-35): kernel itself is unnormalized. After `wrap_triton` returns, the wrapper applies `out = out_work / (2 ** (log_n / 2)) if normalize else out_work` — matches `structured/hadamard.py:58` exactly. **Do not rewrite as `math.sqrt(n)`** — preserves numerical parity with the existing path.
- `num_warps` defaults per D-31b: 4 for `log_n <= 8`, 8 for `log_n in {9..12}`. Fixed (no `@triton.autotune` for Phase 6).
- Op name: `"torch_structured::hadamard_transform"`.

**`_setup_context`** (`_triton/diag_mult/op.py:165-170` — same shape; saves the `normalize` flag for backward):

Diag_mult (skeleton):
```python
def _setup_context(ctx, inputs, output):
    subdiag, v, shift_subdiag, shift_v = inputs
    ctx.save_for_backward(subdiag, v)
    ctx.shift_subdiag = shift_subdiag
    ctx.shift_v = shift_v
```

**Apply as** (per D-32b — save `normalize` flag for chain-rule propagation):
```python
def _setup_context(ctx, inputs, output):
    u, normalize = inputs
    # No need to save_for_backward(u) — self-inverse means backward = forward applied to grad;
    # the saved input is unused.
    ctx.normalize = normalize
```

**`_backward`** (`_triton/diag_mult/op.py:173-190` — Wirtinger formula; hadamard is substantially simpler):

Diag_mult backward (skeleton, with Wirtinger `.conj()` and Pitfall 6 broadcast-sum):
```python
def _backward(ctx, grad_out):
    subdiag, v = ctx.saved_tensors
    s_sub, s_v = ctx.shift_subdiag, ctx.shift_v
    grad_subdiag = _diag_mult_torch(grad_out, v.conj(), -s_sub, s_v - s_sub)
    grad_v = _diag_mult_torch(subdiag.conj(), grad_out, s_sub - s_v, -s_v)
    if subdiag.shape != grad_subdiag.shape:
        ndims_to_sum = grad_subdiag.dim() - subdiag.dim()
        if ndims_to_sum > 0:
            grad_subdiag = grad_subdiag.sum(dim=tuple(range(ndims_to_sum)))
    return grad_subdiag, grad_v, None, None
```

**SUBSTANTIVE DIVERGENCE (per D-32, D-32a, D-32b) — self-inverse, single tensor, no Wirtinger, no broadcast-sum:**
```python
def _backward(ctx, grad_out):
    # Hadamard is self-inverse: d(H @ u)/du = H, so grad_u = H @ grad_out.
    # Route through _torch_ref oracle for fp64 gradcheck precision (D-32).
    # No .conj() (real-only), no broadcast-sum (single tensor input).
    grad_u = _hadamard_transform_torch(grad_out, normalize=ctx.normalize)
    return grad_u, None  # 2 returns matching 2 forward inputs (u + normalize bool)
```

**`register_autograd` + `register_fake`** (`_triton/diag_mult/op.py:193-205` — verbatim shape):

Diag_mult:
```python
diag_mult.register_autograd(_backward, setup_context=_setup_context)

@diag_mult.register_fake
def _diag_mult_fake(subdiag, v, shift_subdiag, shift_v):
    return torch.empty_like(v)
```

**Apply as** (signature substitution):
```python
hadamard_transform.register_autograd(_backward, setup_context=_setup_context)

@hadamard_transform.register_fake
def _hadamard_transform_fake(u, normalize):
    return torch.empty_like(u)
```

---

### `torch_structured/_torch_ref/hadamard.py` (new, reference impl)

**Primary analog (file shape):** `torch_structured/_torch_ref/diag_mult.py:1-56` — module docstring + single function + asserts per CLAUDE.md.

**Secondary analog (function body to relocate):** `torch_structured/structured/hadamard.py:15-30` — `hadamard_transform_torch(u, normalize=False)`. Move verbatim with one shape generalization (CONTEXT D-33d).

**Existing function** (`structured/hadamard.py:15-30`):
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
    batch_size, n = u.shape          # WARN: shape unpack assumes 2-D — preserve as-is per D-33d
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    x = u[..., np.newaxis]
    for d in range(m)[::-1]:
        x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
    return x.squeeze(-2) / 2**(m / 2) if normalize else x.squeeze(-2)
```

**File shape** (`_torch_ref/diag_mult.py:1-56`):
```python
"""Pure-PyTorch reference implementation of the cycle_mult primitive (D-19, D-26).
... module docstring ...
"""
import torch

def diag_mult(subdiag: torch.Tensor, v: torch.Tensor, shift_subdiag: int,
              shift_v: int) -> torch.Tensor:
    """Pure-PyTorch cycle_mult: ... docstring ..."""
    assert subdiag.dtype == v.dtype, ...
    assert subdiag.size(-1) == v.size(-1), ...
    return torch.roll(...) * torch.roll(...)
```

**Apply as** (relocate `structured/hadamard.py:15-30` into the `_torch_ref/diag_mult.py:1-56` file shape; keep numpy dependency for `np.log2` to preserve numerical parity with the original):
```python
"""Pure-PyTorch reference implementation of the Walsh-Hadamard transform (D-32, D-35c).

Moved verbatim from torch_structured/structured/hadamard.py:15-30 (per Phase 6 D-33d).

Three roles per the Phase 6 plan:
1. **Gradcheck oracle.** torch.autograd.gradcheck runs against this fp64-capable
   pure-PyTorch implementation; the Triton kernel's register_autograd backward
   callback also invokes this function (per D-32).
2. **Runtime fallback** for the D-22 asymmetric path: when TORCH_STRUCTURED_BACKEND=cuda
   is requested but _hadamard_cuda.so is not built, the resolver binds
   _ops.hadamard_transform to this function and emits a log.warning.
3. **Backend = "torch"** path: this is the implementation used when set_backend("torch")
   is in effect (no GPU required).

Self-inverse property (per ROADMAP SC#2): H @ (H @ u) = N * u (unnormalized) or u (normalized).
"""
import numpy as np
import torch


def hadamard_transform_torch(u, normalize=False):
    """Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
    n must be a power of 2.
    Parameters:
        u: Tensor of shape (batch_size, n)
        normalize: if True, divide the result by 2^{m/2} where m = log_2(n).
    Returns:
        product: Tensor of shape (batch_size, n)
    """
    batch_size, n = u.shape
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    x = u[..., np.newaxis]
    for d in range(m)[::-1]:
        x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
    return x.squeeze(-2) / 2**(m / 2) if normalize else x.squeeze(-2)
```

**SUBSTANTIVE DIVERGENCES vs `_torch_ref/diag_mult.py`:**
- Keeps `import numpy as np` (the existing function uses `np.log2`; do NOT rewrite as `n.bit_length()` — preserves numerical parity).
- Signature is `(u, normalize=False)` — single tensor + bool, not 4-arg.
- No `assert dtype == dtype` (single tensor).
- No `assert size(-1) == size(-1)` (single tensor).
- Function name is `hadamard_transform_torch` (with `_torch` suffix) — matches the existing function name; downstream imports via the back-compat shim (`structured/hadamard.py` D-33d) reference this exact name.

---

### `torch_structured/_torch_ref/__init__.py` (modified, package init)

**Analog (self):** `torch_structured/_torch_ref/__init__.py:1-5` (current state after Phase 5).

**Current state:**
```python
"""Pure-PyTorch reference implementations used by the dispatch fallback path."""
from .butterfly import butterfly_multiply_torch  # noqa: F401
from .diag_mult import diag_mult  # noqa: F401 (re-exported)

__all__ = ["butterfly_multiply_torch", "diag_mult"]
```

**Apply as** (extend `__all__` and add re-export — verbatim Phase 5 pattern):
```python
"""Pure-PyTorch reference implementations used by the dispatch fallback path."""
from .butterfly import butterfly_multiply_torch  # noqa: F401
from .diag_mult import diag_mult  # noqa: F401 (re-exported)
from .hadamard import hadamard_transform_torch  # noqa: F401 (re-exported)

__all__ = ["butterfly_multiply_torch", "diag_mult", "hadamard_transform_torch"]
```

---

### `torch_structured/_cuda_legacy/hadamard.py` (new, C++ extension passthrough)

**Analog:** `torch_structured/_cuda_legacy/diag_mult.py:1-46` — try-import + `HAS_CUDA_LEGACY_*` sentinel + defensive `RuntimeError` on call. Phase 6 transcribes verbatim with name substitutions.

**Pattern to copy** (verbatim, `_cuda_legacy/diag_mult.py:1-46`):
```python
"""Pass-through wrapper for the legacy ``_diag_mult_cuda`` pybind11 extension.

Asymmetry vs ``_cuda_legacy/butterfly.py``: the butterfly ``.so`` is loaded
eagerly into ``torch.ops.torch_structured.*`` by ``butterfly/__init__.py:22-39``
... (full docstring describes the honest-probe + RuntimeError-on-absent contract)
"""
from typing import Optional

import torch

try:
    from torch_structured import _diag_mult_cuda as _diag_mult_cuda_module
except (ImportError, RuntimeError):
    _diag_mult_cuda_module = None  # type: ignore[assignment]

HAS_CUDA_LEGACY_DIAG_MULT: bool = _diag_mult_cuda_module is not None


def diag_mult(subdiag: torch.Tensor, v: torch.Tensor, shift_subdiag: int,
              shift_v: int) -> torch.Tensor:
    """Pass-through to the compiled pybind11 ``cycle_mult`` op."""
    if _diag_mult_cuda_module is None:
        raise RuntimeError(
            "_diag_mult_cuda not built — caller should use "
            "_has_cuda_legacy_diag_mult() probe (D-22)"
        )
    return _diag_mult_cuda_module.cycle_mult(subdiag, v, shift_subdiag, shift_v)
```

**Apply as** (name + cpp-binding substitution per `csrc/hadamard/hadamard_cuda.cpp:16-18`):
```python
"""Pass-through wrapper for the legacy ``_hadamard_cuda`` pybind11 extension.

Same asymmetry rationale as ``_cuda_legacy/diag_mult.py`` — pybind11 extension
imported by name; honest-probe via the ``HAS_CUDA_LEGACY_HADAMARD`` sentinel.
"""
from typing import Optional

import torch

try:
    from torch_structured import _hadamard_cuda as _hadamard_cuda_module
except (ImportError, RuntimeError):
    _hadamard_cuda_module = None  # type: ignore[assignment]

HAS_CUDA_LEGACY_HADAMARD: bool = _hadamard_cuda_module is not None


def hadamard_transform(u: torch.Tensor) -> torch.Tensor:
    """Pass-through to the compiled pybind11 ``hadamard_transform`` op.

    Raises ``RuntimeError`` if the extension was not built — callers should
    probe ``HAS_CUDA_LEGACY_HADAMARD`` (or ``_ops._has_cuda_legacy_hadamard()``)
    first per the D-22 honest-fallback contract.
    """
    if _hadamard_cuda_module is None:
        raise RuntimeError(
            "_hadamard_cuda not built — caller should use "
            "_has_cuda_legacy_hadamard() probe (D-22)"
        )
    return _hadamard_cuda_module.hadamard_transform(u)
```

**SUBSTANTIVE DIVERGENCES:**
- Signature is `(u,)` only — the CPP op at `csrc/hadamard/hadamard_cuda.cpp:5-14` takes a single tensor; normalization happens in the Python wrapper (consistent with `_ops.hadamard_transform`'s wrapper-side scaling per D-35).
- CPP module name is `_hadamard_cuda` (not `_diag_mult_cuda`); function name is `hadamard_transform` (not `cycle_mult`) per `csrc/hadamard/hadamard_cuda.cpp:16-18` `PYBIND11_MODULE`.
- Sentinel constant is `HAS_CUDA_LEGACY_HADAMARD` (not `_DIAG_MULT`).
- Note: `_cuda_legacy/__init__.py` does NOT need to be modified per CONTEXT — the resolver imports `from torch_structured._cuda_legacy.hadamard import hadamard_transform` directly (Phase 5 set the precedent for `diag_mult` via `_cuda_legacy/__init__.py:15`, but Phase 6 CONTEXT does not require the umbrella re-export; planner's call).

---

### `torch_structured/_ops.py` (modified, dispatch module)

**Self-analog (probe):** `_ops.py:82-94` — `_has_cuda_legacy_diag_mult()` is the symmetric template. Phase 6 adds `_has_cuda_legacy_hadamard()` immediately below it.

**Self-analog (resolver per-op binding):** `_ops.py:213-239` — the Phase 5 `diag_mult` three-branch block. Phase 6 adds a structurally identical block for `hadamard_transform`. Also touches `_ops.py:236-239` (per-op `log.info` format string extension) and `_ops.py:241` (comment update).

**Probe pattern** (`_ops.py:82-94` — current state):
```python
def _has_cuda_legacy_diag_mult() -> bool:
    """Per-op honest probe (CHECKER B3) for the legacy ``_diag_mult_cuda`` extension.

    Symmetric to ``_has_cuda_legacy()`` but checks the pybind11 ``_diag_mult_cuda``
    extension (D-22). Returns the ``HAS_CUDA_LEGACY_DIAG_MULT`` sentinel from
    ``_cuda_legacy/diag_mult.py`` — True iff the ``.so`` was built and the
    top-of-module try-import succeeded. Never raises; returns a clean bool.
    """
    try:
        from torch_structured._cuda_legacy.diag_mult import HAS_CUDA_LEGACY_DIAG_MULT
        return HAS_CUDA_LEGACY_DIAG_MULT
    except ImportError:
        return False
```

**Apply as** (insert immediately after `_has_cuda_legacy_diag_mult` at ~line 95 — verbatim name substitution; per D-36a):
```python
def _has_cuda_legacy_hadamard() -> bool:
    """Per-op honest probe (CHECKER B3) for the legacy ``_hadamard_cuda`` extension.

    Symmetric to ``_has_cuda_legacy_diag_mult()``; returns the ``HAS_CUDA_LEGACY_HADAMARD``
    sentinel from ``_cuda_legacy/hadamard.py``. Never raises; returns a clean bool.
    """
    try:
        from torch_structured._cuda_legacy.hadamard import HAS_CUDA_LEGACY_HADAMARD
        return HAS_CUDA_LEGACY_HADAMARD
    except ImportError:
        return False
```

**Per-op resolver binding pattern** (`_ops.py:213-234` — current Phase 5 diag_mult block, verbatim):
```python
# diag_mult per-op binding (D-22 — asymmetric fallback). The coarse `actual`
# signals the user's intent; the per-op binding uses ``_has_triton_kernel`` /
# ``_has_cuda_legacy_diag_mult`` to honor honest availability. ``_diag_mult_backend``
# is local — the only consumer is the log.info line below; the module-level
# ``_BACKEND`` global stays coarse per D-22a recommendation A.
if actual == "triton" and _has_triton_kernel("diag_mult"):
    from torch_structured._triton.diag_mult.op import diag_mult as _triton_dm
    diag_mult = _triton_dm
    _diag_mult_backend = "triton"
elif actual == "cuda" and _has_cuda_legacy_diag_mult():
    from torch_structured._cuda_legacy.diag_mult import diag_mult as _cuda_dm
    diag_mult = _cuda_dm
    _diag_mult_backend = "cuda"
else:
    from torch_structured._torch_ref.diag_mult import diag_mult as _torch_dm
    diag_mult = _torch_dm
    _diag_mult_backend = "torch"
    if actual == "cuda":
        log.warning(
            "set_backend('cuda') requested but _diag_mult_cuda not built; "
            "falling back to torch_ref for diag_mult (D-22)"
        )
```

**Apply as** (insert structurally identical block immediately after the diag_mult block, before the `log.info` per-op summary at line 236-239; per D-36, D-36a, D-36c):
```python
# hadamard_transform per-op binding (D-22 / D-36 — same shape as diag_mult above).
if actual == "triton" and _has_triton_kernel("hadamard_transform"):
    from torch_structured._triton.hadamard_transform.op import hadamard_transform as _triton_ht
    hadamard_transform = _triton_ht
    _hadamard_transform_backend = "triton"
elif actual == "cuda" and _has_cuda_legacy_hadamard():
    from torch_structured._cuda_legacy.hadamard import hadamard_transform as _cuda_ht
    hadamard_transform = _cuda_ht
    _hadamard_transform_backend = "cuda"
else:
    from torch_structured._torch_ref.hadamard import hadamard_transform_torch as _torch_ht
    hadamard_transform = _torch_ht
    _hadamard_transform_backend = "torch"
    if actual == "cuda":
        log.warning(
            "set_backend('cuda') requested but _hadamard_cuda not built; "
            "falling back to torch_ref for hadamard_transform (D-22)"
        )
```

**Per-op log.info extension** (`_ops.py:236-239` — current Phase 5 log line):
```python
log.info(
    "torch_structured: per-op bindings: butterfly_multiply=%s, diag_mult=%s",
    actual, _diag_mult_backend,
)
```

**Apply as** (extend format string + args with hadamard_transform per D-36c):
```python
log.info(
    "torch_structured: per-op bindings: butterfly_multiply=%s, diag_mult=%s, hadamard_transform=%s",
    actual, _diag_mult_backend, _hadamard_transform_backend,
)
```

**Comment cleanup** (`_ops.py:241` — current state):
```python
# hadamard_transform: Phase 6 populates; stays None for now.
```

**Apply as** (delete this comment — hadamard is now populated; the Phase 5 placeholder is no longer accurate).

**Notes:**
- The `_BACKEND` coarse global stays unchanged per D-22a (research recommendation A — coarse global + per-op log.info is the diagnostic surface).
- `_has_any_triton_kernel()` at `_ops.py:117-129` ALREADY iterates `("butterfly_multiply", "diag_mult", "hadamard_transform")` — no Step 1 change needed (per D-36b). Phase 5's BLOCKER-1 fix already covered this.
- Resolver Step 1 (`_ops.py:149-188`) needs NO edits — when `_has_triton_kernel("hadamard_transform")` flips True in Phase 6, the `actual == "triton"` branch automatically lights up.
- `hadamard_transform = None` placeholder at `_ops.py:57` — no edit; remains the module-level placeholder rebound by `_resolve()`.

---

### `torch_structured/structured/hadamard.py` (modified, consumer + back-compat shim)

**Analog (delete pattern):** `torch_structured/structured/krylov.py` Phase 5 refactor — deleted the try-import, `CycleDownMultCuda` class, the module-level callable binding, and rewrote consumer call sites. Phase 6 mirrors but additionally keeps `hadamard_transform_torch` as a back-compat re-export shim (D-33d) because `tests/structured/test_hadamard.py:8` and `tests/structured/test_imports.py:7` import it from this module.

**Current state of `structured/hadamard.py`** (lines 1-62, full file):
```python
import numpy as np
import torch

use_hadamard_transform_cuda = True
try:
    from torch_structured import _hadamard_cuda as hadamard_cuda
except ImportError:
    use_hadamard_transform_cuda = False

from scipy.linalg import hadamard

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hadamard_transform_torch(u, normalize=False):
    """Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
    n must be a power of 2.
    ... (15-30) ...
    """
    batch_size, n = u.shape
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    x = u[..., np.newaxis]
    for d in range(m)[::-1]:
        x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
    return x.squeeze(-2) / 2**(m / 2) if normalize else x.squeeze(-2)


class HadamardTransformCuda(torch.autograd.Function):
    '''The unnormalized Hadamard transform ...'''
    @staticmethod
    def forward(ctx, u):
        return hadamard_cuda.hadamard_transform(u)
    @staticmethod
    def backward(ctx, grad):
        return HadamardTransformCuda.apply(grad)


def hadamard_transform_cuda(u, normalize=False):
    """Multiply H_n @ u ... (45-58) ..."""
    _, n = u.shape
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    output = HadamardTransformCuda.apply(u)
    return output / 2**(m / 2) if normalize else output


hadamard_transform = hadamard_transform_cuda if use_hadamard_transform_cuda else hadamard_transform_torch
```

**Krylov.py refactor pattern** (the delete + import-shim shape, `structured/krylov.py:16` + (deleted) lines 21-24):
```python
# DELETED: try-import block (lines 4-8) — _cuda_legacy/hadamard.py owns this.
# DELETED: HadamardTransformCuda class (lines 33-42).
# DELETED: hadamard_transform_cuda wrapper (lines 45-58).
# DELETED: module-level binding (line 61).
# KEPT: hadamard_transform_torch — but RE-EXPORTED from _torch_ref/hadamard.py per D-33d.
```

**Apply as** (Phase 6 D-33 / D-33a / D-33b / D-33c / D-33d — delete then shim):
```python
"""Back-compat shim — Phase 6 (TRI-02).

Per D-33d, the pure-PyTorch reference ``hadamard_transform_torch`` has been
relocated to ``torch_structured._torch_ref.hadamard`` (the cross-cutting oracle
home). This module re-exports it so that the existing import surface in
``tests/structured/test_hadamard.py:8`` and ``tests/structured/test_imports.py:7``
continues to work without edits.

The legacy ``HadamardTransformCuda(torch.autograd.Function)``, ``hadamard_transform_cuda``
wrapper, and module-level ``hadamard_transform = ...`` binding were deleted per
D-33 / D-33a / D-33b — ``torch_structured._ops.hadamard_transform`` now handles
the dispatch + autograd plumbing via ``register_autograd``.
"""
from torch_structured._torch_ref.hadamard import hadamard_transform_torch  # noqa: F401 — back-compat shim per D-33d
```

**SUBSTANTIVE NOTES (per CONTEXT D-33d + Integration Points + test_imports.py line 7-13):**
- `test_imports.py:7` imports BOTH `hadamard_transform` AND `hadamard_transform_torch` from this module and asserts `callable(hadamard_transform)`. The back-compat shim must therefore ALSO re-export a `hadamard_transform` callable. Options:
  - (a) Re-export via `import torch_structured; hadamard_transform = lambda *a, **k: torch_structured._ops.hadamard_transform(*a, **k)` — preserves D-05 attribute access (rebind-safe).
  - (b) Re-export `from torch_structured._ops import hadamard_transform` — captures the binding at import time (NOT rebind-safe; violates D-05). Reject.
  - **Recommended (a)** — planner verifies in scout. Either way the existing `test_imports.py` `callable(hadamard_transform)` assertion still passes.
- Remove unused imports: `numpy`, `scipy.linalg.hadamard` (the latter was unused — `device` variable was unused too).

---

### `torch_structured/structured/fastfood.py` (modified, consumer)

**Analog:** `torch_structured/structured/krylov.py:16,321-333` — the D-05 attribute-access pattern Phase 5 established. Per CONTEXT D-34, fastfood.py performs the identical refactor.

**Krylov pattern** (the Phase 5 in-place rewrite — `structured/krylov.py:16,332-333`):
```python
import torch_structured                                                    # line 16
...
    subdiag_extended = torch.cat((torch.tensor([upper_right_corner], ...), subdiag))
    return lambda v: torch_structured._ops.diag_mult(subdiag_extended, v, 0, -1)  # line 333
```

**Current state of `structured/fastfood.py`** (full file, 11 lines):
```python
from .hadamard import hadamard_transform    # line 1 — early binding (D-05 violation)


# S,G,B: diagonal
# P: permutation
# x: batch_size x n_features
def fastfood_multiply(S, G, B, P, x):
    HBx = hadamard_transform(B * x)         # line 8 — call site 1
    PHBx = HBx[:, P]
    HGPHBx = hadamard_transform(G * PHBx)   # line 10 — call site 2
    return S * HGPHBx
```

**Apply as** (per D-34 — drop early binding, rewrite both call sites to attribute-access form):
```python
import torch_structured


# S,G,B: diagonal
# P: permutation
# x: batch_size x n_features
def fastfood_multiply(S, G, B, P, x):
    HBx = torch_structured._ops.hadamard_transform(B * x)
    PHBx = HBx[:, P]
    HGPHBx = torch_structured._ops.hadamard_transform(G * PHBx)
    return S * HGPHBx
```

**Notes:**
- Line 1: replace `from .hadamard import hadamard_transform` with `import torch_structured` (drop early binding per D-05).
- Lines 8 & 10: substitute `hadamard_transform(...)` → `torch_structured._ops.hadamard_transform(...)`.
- The substantive divergence from krylov.py: fastfood.py has TWO call sites, krylov.py had one (the lambda inside `subdiag_linear_map_cuda`). Otherwise structurally identical.

---

### `tests/structured/test_hadamard_triton.py` (new, test module)

**Analog:** `tests/test_diag_mult.py:1-119` — verbatim structural twin: parametrized `backend` fixture, GPU skipif, 5 test functions, attribute-access call sites.

**Imports pattern** (`tests/test_diag_mult.py:13-19`):
```python
import itertools

import pytest
import torch

import torch_structured  # noqa: F401 — triggers extension load + _ops.py resolver
from torch_structured._torch_ref.diag_mult import diag_mult as diag_mult_ref


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="diag_mult tests require CUDA"
)
```

**Apply as** (substitute name + add scipy oracle import per CONTEXT D-37 cross-backend correctness against scipy reference):
```python
import itertools

import pytest
import torch
from scipy.linalg import hadamard as scipy_hadamard

import torch_structured  # noqa: F401 — triggers extension load + _ops.py resolver
from torch_structured._torch_ref.hadamard import hadamard_transform_torch as hadamard_ref


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="hadamard_transform tests require CUDA"
)
```

**Test function pattern** (`tests/test_diag_mult.py:27-36` — eager fp32 cross-backend allclose):
```python
def test_diag_mult_eager_fp32(backend):
    """Forward correctness vs torch_ref oracle, fp32, batched v + broadcast subdiag."""
    N, B = 128, 4
    s = torch.randn(N, device="cuda", dtype=torch.float32)
    v = torch.randn(B, N, device="cuda", dtype=torch.float32)
    out = torch_structured._ops.diag_mult(s, v, 0, -1)
    expected = diag_mult_ref(s, v, 0, -1)
    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-6), (
        f"fp32 mismatch (backend={backend}): max err = {(out - expected).abs().max()}"
    )
```

**Apply as** (5 tests per CONTEXT D-37 — `test_hadamard_eager_fp32`, `test_hadamard_normalize`, `test_hadamard_gradcheck_fp64`, `test_hadamard_self_inverse`, `test_hadamard_module_consumer`):

- `test_hadamard_eager_fp32(backend)` — log_n grid {2..12}, parametrized via `@pytest.mark.parametrize("log_n", [2,3,...,12])`; cross-backend allclose vs `hadamard_ref`. fp32, no normalize.
- `test_hadamard_normalize(backend)` — `log_n=10`; `normalize=True` correctness vs `hadamard_ref` (the `2 ** (m / 2)` scale per D-35a).
- `test_hadamard_gradcheck_fp64(backend)` — `gradcheck` against `torch_structured._ops.hadamard_transform` at fp64; small N (n=4 or 8); per D-32 the backward routes through `_torch_ref` so fp64 native gradcheck works.
- `test_hadamard_self_inverse(backend)` — composition `H(H(u)) ≈ N * u` (unnormalized) or `u` (normalized), fp32 noise floor. ROADMAP SC#2.
- `test_hadamard_module_consumer(backend)` — fastfood-style consumer chain via `fastfood_multiply` (or `Hadamard` nn.Module if `structured/layers.py` exposes one — scout reveals it does NOT; `structured/layers.py:118` references `ff.fastfood_multiply` directly, no `Hadamard` class). ROADMAP SC#3.

**Gradcheck pattern** (`tests/test_diag_mult.py:52-62` — fp64 gradcheck against the kernel under test):
```python
def test_diag_mult_gradcheck_fp64_real(backend):
    """fp64 gradcheck — real (D-26 acceptance gate for register_autograd plumbing)."""
    N = 8
    s = torch.randn(N, dtype=torch.float64, device="cuda", requires_grad=True)
    v = torch.randn(4, N, dtype=torch.float64, device="cuda", requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, b: torch_structured._ops.diag_mult(a, b, 0, -1),
        (s, v),
        eps=1e-6,
        atol=1e-5,
    )
```

**Apply as** (single tensor input, normalize=False):
```python
def test_hadamard_gradcheck_fp64(backend):
    """fp64 gradcheck — real-only (D-32 acceptance gate; backward routes through _torch_ref oracle).

    Note: the Triton kernel is fp32-only (D-31 / SC#1); fp64 gradcheck necessarily
    exercises the torch_ref backward leg per D-32. This validates that the
    register_autograd plumbing correctly routes grads through _torch_ref.
    """
    n = 8
    u = torch.randn(4, n, dtype=torch.float64, device="cuda", requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a: torch_structured._ops.hadamard_transform(a, normalize=False),
        (u,),
        eps=1e-6,
        atol=1e-5,
    )
```

**SUBSTANTIVE DIVERGENCES vs `tests/test_diag_mult.py`:**
- File path: `tests/structured/test_hadamard_triton.py` (per CONTEXT D-37 — mirrors existing `tests/structured/test_hadamard.py` layout, not `tests/test_diag_mult.py` top-level).
- No `test_*_eager_complex64` test (Phase 6 is real-only per ROADMAP "no complex" / D-31).
- No `test_*_shift_grid` test (no shift args).
- New `test_hadamard_self_inverse` test (no Phase 5 analog — exercises the self-inverse property per ROADMAP SC#2).
- New `test_hadamard_module_consumer` test using `fastfood_multiply` (no Phase 5 analog — exercises ROADMAP SC#3 consumer surface).
- `test_hadamard_eager_fp32` is additionally parametrized over `log_n ∈ {2..12}` (the full SC#1 N range).

---

### `tests/conftest.py` (modified, pytest fixture skip-gate)

**Analog (self):** `tests/conftest.py:1-23` — Phase 5 state. Phase 6 widens the skip-gate from `_has_triton_kernel("diag_mult")` to `_has_any_triton_kernel()` (per CONTEXT D-39) so the fixture skips only when NO Triton kernel is installed.

**Current state** (`tests/conftest.py:15-23`):
```python
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

**Apply as** (D-39 widening — substitute the per-op probe with the milestone-wide `_has_any_triton_kernel()`):
```python
@pytest.fixture(params=["torch", "triton"])
def backend(request):
    """Switch backend for the duration of a test, restore after.

    Phase 6 D-39: widened skip-gate from ``_has_triton_kernel("diag_mult")`` to
    ``_has_any_triton_kernel()`` — the fixture now skips ``triton`` only when
    NO per-op Triton kernel is installed. This lets phase-6 tests (which need
    only the hadamard_transform kernel) run on hosts that have hadamard but not
    diag_mult, and vice versa.
    """
    if request.param == "triton" and not torch_structured._ops._has_any_triton_kernel():
        pytest.skip("No Triton kernel installed (no CUDA or CPU-only runner)")
    original = torch_structured._ops._BACKEND
    chosen = torch_structured._ops.set_backend(request.param)
    yield chosen
    torch_structured._ops.set_backend(original)
```

**Notes:**
- Update the module docstring at `tests/conftest.py:1-9` to reflect the Phase 6 D-39 widening (currently it pins the skip-gate to `_has_triton_kernel("diag_mult")` per Phase 5).
- `_has_any_triton_kernel()` is already in place at `_ops.py:117-129` (Phase 5 BLOCKER-1 fix); no resolver change needed.

---

## Shared Patterns

### D-05 attribute-access contract (consumer-side)
**Source:** `torch_structured/_ops.py:11-39` (module docstring + the load-bearing CORRECT/WRONG examples)
**Applied to:** All consumer files (`structured/fastfood.py`, `structured/hadamard.py` back-compat shim, `tests/structured/test_hadamard_triton.py`).

Excerpt (`_ops.py:18-29`):
```python
# CORRECT — attribute access (re-reads binding on each call):
import torch_structured
def some_function(twiddle, x, ...):
    return torch_structured._ops.butterfly_multiply(twiddle, x, ...)
# WRONG — captures the CURRENT object at import time:
from torch_structured._ops import butterfly_multiply
def some_function(twiddle, x, ...):
    butterfly_multiply(twiddle, x, ...)   # set_backend() rebind invisible
```

**Apply as:** `import torch_structured` at module top; call sites use `torch_structured._ops.hadamard_transform(...)`.

### `assert` preconditions, no try/except in core lib (CLAUDE.md)
**Source:** `_triton/diag_mult/op.py:115-125` (wrapper-boundary asserts); `CLAUDE.md` §"Error Handling".
**Applied to:** `_triton/hadamard_transform/op.py` (kernel wrapper boundary).

Excerpt (`_triton/diag_mult/op.py:115-125`):
```python
assert subdiag.dtype == v.dtype, (...)
assert v.is_contiguous(), "v must be contiguous before view_as_real (Pitfall 3)"
assert subdiag.is_contiguous(), ...
assert subdiag.size(-1) == v.size(-1), (...)
```

**Apply as** (hadamard wrapper asserts per `csrc/hadamard/hadamard_cuda.cpp:6-9` + D-31c log_n cap):
```python
assert u.dim() >= 1
assert u.dtype == torch.float32, f"hadamard kernel is fp32-only, got {u.dtype}"
assert u.is_contiguous()
n = u.size(-1)
log_n = int(n.bit_length() - 1)
assert n == 1 << log_n, f"n must be a power of 2, got {n}"
assert log_n <= 12, f"single-pass kernel caps log_n at 12 (SC#1), got log_n={log_n}"
```

**Documented exception** (try-import only): `_cuda_legacy/diag_mult.py:24-27` honest-probe pattern. Applies verbatim to `_cuda_legacy/hadamard.py:24-27`.

### Honest probe + sentinel
**Source:** `_cuda_legacy/diag_mult.py:24-29` (try-import + `HAS_CUDA_LEGACY_DIAG_MULT: bool = ... is not None`); `_ops.py:82-94` (`_has_cuda_legacy_diag_mult()` resolver-side probe).
**Applied to:** `_cuda_legacy/hadamard.py` (sentinel) + `_ops.py` (`_has_cuda_legacy_hadamard()`).

### `register_autograd` + `register_fake` (260419-p27 fix)
**Source:** `_triton/diag_mult/op.py:193-205` (the two registration calls + the meta kernel).
**Applied to:** `_triton/hadamard_transform/op.py` (verbatim shape; `_hadamard_transform_fake(u, normalize)` returns `torch.empty_like(u)`).

### per-op log.info diagnostic surface
**Source:** `_ops.py:236-239` (current Phase 5 line).
**Applied to:** Extended format string adds `hadamard_transform=%s` and `_hadamard_transform_backend` arg per D-36c.

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| (kernel body of) `_triton/hadamard_transform/op.py` | log_n-stage Walsh-Hadamard butterfly | shared-memory single-pass | No in-repo Triton FWT exists. Reference for the math is the CUDA kernel at `csrc/hadamard/hadamard_cuda_kernel.cu:24-86` (`fwtBatch1Kernel`); the Triton implementation is planner-authored per D-31a using `tl.static_range(LOG_N)` for the unrolled stage loop. The two-pass mixed-radix `fwtBatch2Kernel` (CUDA lines 88-153) is explicitly out of scope per D-31c. |

All other files have a direct Phase 5 mirror or self-analog.

---

## Metadata

**Analog search scope:**
- `torch_structured/_triton/` (Phase 5 templates)
- `torch_structured/_torch_ref/` (Phase 5 templates)
- `torch_structured/_cuda_legacy/` (Phase 5 templates)
- `torch_structured/_ops.py` (resolver self-analog)
- `torch_structured/structured/` (consumer surfaces — `hadamard.py`, `fastfood.py`, `krylov.py`, `layers.py`)
- `tests/` and `tests/structured/` (test analogs)
- `csrc/hadamard/` (CPP signature + kernel reference)
- `.planning/phases/05-diag-mult-triton-port/` (Phase 5 plan + patterns mirror)

**Files scanned:** 18 (analogs + canonical references read end-to-end; CONTEXT and Phase 5 PATTERNS.md sections used for cross-validation)

**Pattern extraction date:** 2026-05-27

**Key insight:** Phase 6 is a near-verbatim Phase 5 transcription. The PATTERNS.md is structured so the planner can copy each Phase 5 file verbatim and substitute names + apply the four substantive divergences (kernel body, backward formula, no IS_COMPLEX, no shift args). The Phase 5 PLAN.md's 7-task shape (foundation → triton → cuda_legacy → resolver → consumer refactor → conftest → tests) is the literal template for the Phase 6 PLAN.md task structure.
