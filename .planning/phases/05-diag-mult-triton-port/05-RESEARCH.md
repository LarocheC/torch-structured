# Phase 5: diag_mult Triton Port - Research

**Researched:** 2026-05-27
**Domain:** Triton kernel port for a generic `cycle_mult(subdiag, v, shift_subdiag, shift_v)` primitive with fp32 + complex64 forward/backward, consuming the Phase 4 dispatch + autograd plumbing
**Confidence:** HIGH (math verified numerically against autograd; dispatch/wrapper pattern locked by Phase 4 verification; complex Wirtinger convention validated on fp64 reference)

## Summary

Phase 5 ports the `cycle_mult` primitive (the pointwise formula at `csrc/diag_mult/diag_mult_cuda_kernel.cu:8`) from C++/CUDA to a Triton `@triton.jit` kernel, exposes it as a generic four-argument op `_ops.diag_mult(subdiag, v, shift_subdiag, shift_v)`, refactors `structured/krylov.py` to consume the single dispatch import point, and deletes the Phase 4 demonstrator. Because CONTEXT.md already locks the API surface (D-19/D-19a/D-19b), the complex64 representation (D-20..D-20c via `view_as_real` + `IS_COMPLEX: tl.constexpr` + 4-FMA), the CUDA legacy wire-up (D-21..D-23), the krylov.py refactor (D-24..D-26), and the demonstrator cleanup (D-27..D-28), this research focuses exclusively on what the planner needs to *implement* those locked decisions correctly.

Three things were independently verified by running Python against the installed PyTorch 2.11.0 + Triton 3.6.0 stack on this CUDA-capable workstation:

1. **The `torch.roll` direction convention** — `torch.roll(t, -shift, dims=-1)` matches the C++ `(pos + shift + N) % N` indexing exactly. The `_torch_ref` formula is `torch.roll(subdiag, -shift_subdiag, dims=-1) * torch.roll(v, -shift_v, dims=-1)`. Verified for `(shift_subdiag, shift_v) ∈ {(0,-1), (1,1), (0,0), (-1,1), (1,-1)}`.
2. **The backward gradient formula** — derived symbolically from the forward, then validated against `torch.autograd.grad` on the `_torch_ref` impl at fp64. Errors are exactly 0.0 across the shift grid. The real-input formula matches the existing hand-derived backward at `krylov.py:336` for the `(0,-1)` specialization.
3. **The complex backward correction** — a naive port of the real formula gives errors ~2.0 on complex inputs because PyTorch's autograd uses the Wirtinger convention: `grad_x = conj(other) * grad_out` for a pointwise multiply. The corrected formula uses `.conj()` on the **other** operand. This is the single most planner-relevant finding that CONTEXT.md does not pre-specify; it is what makes SC#2 (fp64 complex gradcheck) achievable.

**Primary recommendation:** Implement `_triton/diag_mult/op.py` by transcribing the Phase 4 demonstrator op verbatim, replacing the identity kernel body with the four-FMA `cycle_mult` formula gated on `IS_COMPLEX: tl.constexpr`. Use `_torch_ref.diag_mult` for the `register_autograd` backward (per D-26) with `.conj()` on the non-grad operand in the complex case. Keep `_BACKEND` as a coarse global (defer the per-op `_BACKENDS: dict[str, str]` refactor to Phase 7 per D-22a's recommendation). Gradcheck in fp64 works because Triton's `tl.load` is dtype-polymorphic and the backward routes through `_torch_ref` which is fp64-capable.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Kernel arithmetic (cycle + pointwise mul) | Triton kernel (`_triton/diag_mult/op.py`) | — | Phase 4 locked `@triton_op + wrap_triton + register_autograd + register_fake` as the only viable wrapper pattern (PITFALLS §3, TRI-05) |
| Backend probe + binding | Resolver (`_ops.py:_resolve`) | — | Single dispatch point per DISP-03; D-22 extends per-op aware resolution for the asymmetric `_diag_mult.so`-absent case |
| Reference oracle for gradcheck | `_torch_ref/diag_mult.py` | — | Pure-PyTorch oracle per Pitfall 4; also doubles as runtime fallback when `BACKEND=cuda` requested but `.so` absent (D-22) |
| Complex64 boundary conversion | Wrapper boundary in `_triton/diag_mult/op.py` | — | Locked by D-20a/D-20b/`04-COMPLEX-LAYOUT.md`; `view_as_real` + `IS_COMPLEX` constexpr + 4-FMA at the kernel |
| Backward gradient derivation | `register_autograd` callback in `_triton/diag_mult/op.py` | `_torch_ref.diag_mult` (called by the callback) | D-26: backward expressed as two more `cycle_mult` calls with adjusted shifts; calling `_torch_ref` keeps fp64 gradcheck viable |
| Consumer refactor | `structured/krylov.py:325-350` | `torch_structured._ops` (attribute access) | D-24/D-25; deletes `CycleDownMultCuda` autograd Function; inlines `_ops.diag_mult(subdiag, v, 0, -1)` |
| Cross-backend test infra | `tests/conftest.py` `backend` fixture + new `tests/test_diag_mult.py` | — | D-29/D-30; widens fixture to `["torch", "triton"]` with `_has_triton_kernel("diag_mult")` skip-gate |

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-19 / D-19a / D-19b** — `_ops.diag_mult(subdiag, v, shift_subdiag: int, shift_v: int) -> Tensor`. Pointwise formula `out[pos] = subdiag[(pos + shift_subdiag + N) % N] * v[(pos + shift_v + N) % N]`. Broadcast or batched subdiag auto-detected at the wrapper (`subdiag.numel() == v.numel()` ⇒ batched). Triton specializes on `(IS_COMPLEX,)` constexpr; runtime ints for the shifts (no per-shift JIT explosion).
- **D-20 / D-20a / D-20b / D-20c** — Full complex × complex (4-FMA). Both inputs must share dtype: `assert subdiag.dtype == v.dtype` rejects mixed. `view_as_real` at the wrapper, `IS_COMPLEX: tl.constexpr` to the kernel, `view_as_complex(out.contiguous())` on the way back. `_torch_ref/diag_mult.py` accepts complex natively (no view games).
- **D-21 / D-22 / D-22a / D-23** — `_cuda_legacy/diag_mult.py` performs a top-of-module try-import of `torch_structured._diag_mult_cuda`; on `ImportError` the module-level `diag_mult` symbol is `None`. New `_has_cuda_legacy_diag_mult()` probe returns True iff non-None. When `set_backend("cuda")` and probe is False, resolver falls back to `_torch_ref` for **diag_mult only** with `log.warning`; does not affect other ops.
- **D-24 / D-25** — Delete `CycleDownMultCuda(torch.autograd.Function)` from `krylov.py:325-339`. Replace `cycle_down_mult` call sites with `torch_structured._ops.diag_mult(subdiag_extended, v, 0, -1)` (attribute access per Phase 4 D-05). Remove the `from torch_structured import _diag_mult_cuda as diag_mult_cuda` try-import at line 21-24.
- **D-26** — `register_autograd` callback derives gradients from the same `cycle_mult` primitive with adjusted shifts. Planner verifies via fp64 gradcheck.
- **D-27 / D-28** — Delete `_demo_identity_kernel`, `_demo_identity_op`, `_setup_context`, `_backward`, the `register_autograd` line, and the `register_fake` block from `_ops.py:225-304`. Keep the module-level `import triton / triton.language as tl / from torch.library import triton_op, wrap_triton`. Delete the 5 demonstrator tests from `tests/test_dispatch.py`.
- **D-29 / D-30** — New `tests/test_diag_mult.py` with `test_diag_mult_eager_fp32`, `test_diag_mult_eager_complex64`, `test_diag_mult_gradcheck_fp64_real`, `test_diag_mult_gradcheck_fp64_complex`, `test_diag_mult_shift_grid`. `conftest.py` `backend` fixture widens to `["torch", "triton"]` with skip-triton when `_has_triton_kernel("diag_mult")` is False (CPU runners).

### Claude's Discretion (planner choices within the locked envelope)

- Exact `BLOCK_SIZE` for the Triton kernel (research recommends 1024 — same as Phase 4 demonstrator; pointwise kernels are not block-size sensitive at these sizes).
- Whether to introduce `_BACKENDS: dict[str, str]` per-op resolution map or keep coarse `_BACKEND` global. **Research recommends coarse** (simpler; revisit in Phase 7 if needed — see "Per-Op Resolver Refactor Scope" below).
- Exact wording of the new `log.warning` when `cuda` falls back to `torch_ref` for `diag_mult`.
- Whether `tests/test_dispatch.py` is deleted outright or kept as a thin set_backend smoke test.

### Deferred Ideas (OUT OF SCOPE)

- `subdiagKrylov` op port (`csrc/diag_mult/diag_mult_cuda.cpp:18`) — zero Python consumers; deletion deferred to Phase 10.
- Per-op `_BACKENDS` dict — defer to Phase 7.
- Autotune over `BLOCK_SIZE` / `num_warps` — pointwise kernel, hand-picked default is fine.
- bf16 / fp16 support (TRI-FUT-01).
- CUDA backend axis in `backend` fixture — defer to Phase 9 per TEST-03.
- Resurrecting `_diag_mult.so` build — D-23 keeps existing conditional `setup.py:96-110` logic.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TRI-01 | `diag_mult` runs on Triton (forward + backward, fp32 + complex64) | All sections below. Forward formula transcribed from `csrc/diag_mult/diag_mult_cuda_kernel.cu`. Backward derivation in §"Backward Gradient Formula". Complex64 follows `04-COMPLEX-LAYOUT.md` template. |

## Standard Stack

[VERIFIED: installed packages — torch 2.11.0+cu130, triton 3.6.0]

### Core (already in `pyproject.toml`, no changes for Phase 5)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `torch` | >=2.6 (current 2.11.0) | `torch.library.triton_op`, `wrap_triton`, `register_autograd`, `register_fake` | Locked by Phase 4 D-11. Floor is 2.6 — first version shipping `triton_op` as stable. |
| `triton` | bundled with torch (current 3.6.0) | `@triton.jit` for kernel, `tl.load`/`tl.store`/`tl.constexpr` for layout | Bundled with PyTorch on CUDA Linux wheels; no `triton` in `dependencies` (per STACK.md note avoiding pip resolver fights). |

**Version verification (Bash output, this workstation):**
```
torch:  2.11.0+cu130
triton: 3.6.0
cuda available: True
```
[VERIFIED: `python3 -c "import torch, triton; print(torch.__version__, triton.__version__)"`]

**No new packages installed in Phase 5.** All dependencies already met by Phase 4.

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pytest.mark.skipif` | stdlib pytest | Skip triton-parametrized tests on CPU runners | Gate every Triton-touching test (STACK.md table) |
| `torch.autograd.gradcheck` | torch stdlib | fp64 numerical-vs-analytical gradient comparison | SC#2 acceptance gate; called against the wrapped `_demo_identity_op` analog |

### Alternatives Considered (and rejected)

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `_torch_ref.diag_mult` for the backward | Re-call Triton kernel with adjusted shifts (D-26 reads either way) | _torch_ref keeps fp64 gradcheck viable AND is what D-26 spec literally says (`_torch_ref.diag_mult(grad_out, v, ...)`); a Triton kernel call would also work since Triton tl.load is dtype-polymorphic, but it muddies the test/oracle separation per Pitfall 4 |
| One `@triton.jit` source with `IS_COMPLEX` constexpr | Two separate kernel sources `_cycle_mult_real`, `_cycle_mult_complex` | Phase 4 demonstrator validated the single-source + constexpr approach end-to-end (`view_as_real` round-trip works in eager + gradcheck + compile). Two-kernel split adds maintenance with no benefit. |
| Inline backward in `register_autograd` callback | Wrap backward in its own `@triton_op` | D-26 spec gives a closed-form algebraic recipe — call `_torch_ref.diag_mult` directly. A second `@triton_op` is overhead with no compile-graph benefit (the backward is already inside the autograd graph). |

## Architecture Patterns

### System Architecture Diagram

```
                     ┌────────────────────────────────────────────────┐
                     │   structured/krylov.py:347 subdiag_mult_cuda  │
                     │   structured/layers.py:237 LDRSubdiagonalC    │
                     └─────────────────────┬──────────────────────────┘
                                           │ torch_structured._ops.diag_mult(s, v, 0, -1)
                                           │ (attribute access; D-05 contract)
                                           ▼
                     ┌────────────────────────────────────────────────┐
                     │   torch_structured/_ops.py — module-level     │
                     │   `diag_mult` rebound by `_resolve()`         │
                     │   to one of:                                   │
                     └─────┬───────────────┬────────────────────┬─────┘
                           │               │                    │
                  actual="triton"   actual="cuda"        actual="torch"
                           │               │                    │
                           ▼               ▼                    ▼
            ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
            │ _triton/diag_   │  │ _cuda_legacy/   │  │ _torch_ref/      │
            │  mult/op.py     │  │  diag_mult.py   │  │  diag_mult.py    │
            │ @triton_op       │  │ try-import      │  │ pure PyTorch     │
            │ wraps @triton.jit│  │ _diag_mult_cuda│  │ torch.roll +     │
            │ + register_     │  │ (None if absent│  │ multiply         │
            │   autograd      │  │  → fallback to │  │                  │
            │ + register_fake │  │  _torch_ref    │  │                  │
            └────────┬─────────┘  │  per D-22)     │  └──────────────────┘
                     │            └─────────────────┘
                     │ wrap_triton(_cycle_mult_kernel)[grid](
                     │   subdiag_real, v_real, out_real, ...,
                     │   IS_COMPLEX, BLOCK_SIZE,
                     │   shift_subdiag, shift_v, N, n_batch)
                     ▼
            ┌──────────────────────────────────┐
            │ @triton.jit _cycle_mult_kernel  │
            │ — loads subdiag_real, v_real    │
            │ — if IS_COMPLEX: 4-FMA mul      │
            │   else: scalar mul              │
            │ — stores to out_real            │
            └──────────────────────────────────┘
```

Backward flow (executed when `loss.backward()` reaches a `diag_mult` node):

```
        grad_out (from downstream)
              │
              ▼
   register_autograd callback (_backward in op.py)
              │  retrieves saved (subdiag, v, shift_subdiag, shift_v)
              │  derives:
              │    grad_subdiag = _torch_ref.diag_mult(
              │                      grad_out, v.conj(),
              │                      -shift_subdiag, shift_v - shift_subdiag)
              │                   .sum_over_broadcast_dims_if_batched()
              │    grad_v       = _torch_ref.diag_mult(
              │                      subdiag.conj(), grad_out,
              │                      shift_subdiag - shift_v, -shift_v)
              ▼
   returns (grad_subdiag, grad_v, None, None)  # None for the two int args
```

### Recommended Project Structure

```
torch_structured/
├── _ops.py                          # MODIFIED: delete _demo_identity_*; bind diag_mult per-op
├── _torch_ref/
│   ├── __init__.py                  # MODIFIED: extend __all__ with diag_mult
│   └── diag_mult.py                 # NEW: pure-PyTorch oracle
├── _cuda_legacy/
│   ├── __init__.py                  # MODIFIED: re-export diag_mult (with try-import shape)
│   └── diag_mult.py                 # NEW: try-import _diag_mult_cuda; None on ImportError
├── _triton/
│   ├── __init__.py                  # UNCHANGED: HAS_TRITON sentinel
│   └── diag_mult/                   # NEW: nested package per _has_triton_kernel probe shape
│       ├── __init__.py
│       └── op.py                    # NEW: @triton.jit kernel + @triton_op + register_autograd + register_fake
└── structured/
    └── krylov.py                    # MODIFIED: delete CycleDownMultCuda; inline _ops.diag_mult
```

**Single-file vs. nested for `_triton/diag_mult/`:** Phase 4 probe at `_ops.py:96` does `importlib.import_module(f"torch_structured._triton.{op_name}.op")` — it expects the **nested** form (`_triton/diag_mult/op.py`), NOT a flat `_triton/diag_mult.py`. **The planner MUST create the nested directory.** Forward and backward fuse into one file (`op.py`) because diag_mult is small; Phase 6 (hadamard) and Phase 7 (butterfly) will split into `forward.py`/`backward.py`/`op.py` per the ARCHITECTURE.md template — that's correct because those kernels are larger. The `__init__.py` may be empty or re-export `diag_mult` for convenience; the probe only needs `op.py:diag_mult` to resolve.

### Pattern 1: triton_op + wrap_triton + register_autograd + register_fake (canonical, Phase 4 demonstrator template)

**What:** The exact wrapper shape Phase 4 demonstrated at `_ops.py:225-304`. Five components in this fixed order:

1. `@triton.jit` kernel (lines 225-233 in the demonstrator) — the actual GPU code
2. `@triton_op("torch_structured::<name>", mutates_args={})` decorator on the Python wrapper (line 236) — registers the op
3. `wrap_triton(kernel)[grid](args)` inside the wrapper body (line 271) — required for Inductor to see the kernel
4. `op.register_autograd(backward, setup_context=setup)` after the wrapper (line 292) — autograd plumbing
5. `@op.register_fake` decorator on the meta-kernel (line 295) — THE 260419-p27 fix; mandatory per Phase 4 D-12

**When to use:** Every Triton kernel port in Phases 5-8 follows this exact template. **Do not invent a different structure.**

**Example (literal Phase 5 transcription — replace `<...>` with concrete code):**

```python
# torch_structured/_triton/diag_mult/op.py
# Source: 04-COMPLEX-LAYOUT.md (lines 33-76); torch_structured/_ops.py:225-304 (Phase 4 demonstrator)

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from torch_structured._torch_ref.diag_mult import diag_mult as diag_mult_torch  # backward oracle


@triton.jit
def _cycle_mult_kernel(
    subdiag_ptr, v_ptr, out_ptr,
    n_batch, N,
    subdiag_batch_stride,  # 0 if subdiag is 1-D broadcast, else N (or 2*N for complex)
    shift_subdiag, shift_v,
    IS_COMPLEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Generic cycle_mult kernel — Phase 4 demonstrator template with cycle_mult math."""
    bid = tl.program_id(axis=0)   # batch index
    pid = tl.program_id(axis=1)   # block-of-N index
    pos = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pos < N

    # Index arithmetic — matches csrc/diag_mult/diag_mult_cuda_kernel.cu:8 verbatim:
    #   d_Sub[(pos + shift_subdiag + N) % N] * d_Src[(pos + shift_v + N) % N]
    sub_idx = (pos + shift_subdiag + N) % N
    v_idx   = (pos + shift_v        + N) % N

    if IS_COMPLEX:
        # 04-COMPLEX-LAYOUT.md lines 58-76: trailing-2 layout, 4-FMA multiply.
        # Each element occupies 2 contiguous reals (re, im) — pointer arithmetic doubles.
        bo = bid * N * 2  # batch offset in trailing-2 reals
        a_re = tl.load(subdiag_ptr + bid * subdiag_batch_stride + sub_idx * 2,     mask=mask)
        a_im = tl.load(subdiag_ptr + bid * subdiag_batch_stride + sub_idx * 2 + 1, mask=mask)
        c_re = tl.load(v_ptr + bo + v_idx * 2,     mask=mask)
        c_im = tl.load(v_ptr + bo + v_idx * 2 + 1, mask=mask)
        # (a + bi)(c + di) = (ac - bd) + (ad + bc)i   — exactly 04-COMPLEX-LAYOUT.md lines 68-69
        out_re = a_re * c_re - a_im * c_im
        out_im = a_re * c_im + a_im * c_re
        tl.store(out_ptr + bo + pos * 2,     out_re, mask=mask)
        tl.store(out_ptr + bo + pos * 2 + 1, out_im, mask=mask)
    else:
        bo = bid * N
        a = tl.load(subdiag_ptr + bid * subdiag_batch_stride + sub_idx, mask=mask)
        c = tl.load(v_ptr + bo + v_idx, mask=mask)
        tl.store(out_ptr + bo + pos, a * c, mask=mask)


@triton_op("torch_structured::diag_mult", mutates_args={})
def diag_mult(subdiag: torch.Tensor, v: torch.Tensor,
              shift_subdiag: int, shift_v: int) -> torch.Tensor:
    """Generic cycle_mult primitive — Triton kernel-backed."""
    # Pitfall 3: view_as_real requires contiguous source. assert before the view.
    # Also rejects mixed dtypes per D-20.
    assert subdiag.dtype == v.dtype, \
        f"subdiag dtype {subdiag.dtype} != v dtype {v.dtype} (D-20)"
    assert v.is_contiguous(), "v must be contiguous"
    assert subdiag.is_contiguous(), "subdiag must be contiguous"
    assert subdiag.size(-1) == v.size(-1), \
        f"trailing dim mismatch: subdiag {subdiag.size(-1)} vs v {v.size(-1)}"

    N = v.size(-1)
    n_batch = v.numel() // N
    # D-19b: batched-or-broadcast subdiag auto-detect (matches csrc batchedSubdiag)
    is_batched_subdiag = (subdiag.numel() == v.numel())

    is_complex = v.is_complex()
    if is_complex:
        v_real = torch.view_as_real(v)              # (..., N, 2)
        subdiag_real = torch.view_as_real(subdiag)  # (N, 2) or (..., N, 2)
        # batch stride in trailing-2-real units: 2*N if batched, else 0 (broadcast)
        subdiag_batch_stride = 2 * N if is_batched_subdiag else 0
    else:
        v_real = v
        subdiag_real = subdiag
        subdiag_batch_stride = N if is_batched_subdiag else 0

    out_real = torch.empty_like(v_real)
    BLOCK_SIZE = 1024
    grid = lambda meta: (n_batch, triton.cdiv(N, meta["BLOCK_SIZE"]))
    wrap_triton(_cycle_mult_kernel)[grid](
        subdiag_real, v_real, out_real,
        n_batch, N,
        subdiag_batch_stride,
        shift_subdiag, shift_v,
        IS_COMPLEX=is_complex,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return torch.view_as_complex(out_real.contiguous()) if is_complex else out_real


def _setup_context(ctx, inputs, output):
    subdiag, v, shift_subdiag, shift_v = inputs
    ctx.save_for_backward(subdiag, v)
    ctx.shift_subdiag = shift_subdiag
    ctx.shift_v = shift_v


def _backward(ctx, grad_out):
    subdiag, v = ctx.saved_tensors
    s_sub, s_v = ctx.shift_subdiag, ctx.shift_v
    # Derivation in §"Backward Gradient Formula" below.
    # Wirtinger convention: .conj() on the OTHER operand. Real ⇒ conj is no-op.
    grad_subdiag = diag_mult_torch(grad_out, v.conj(),         -s_sub,    s_v - s_sub)
    grad_v       = diag_mult_torch(subdiag.conj(), grad_out,    s_sub - s_v, -s_v)
    # If subdiag was 1-D broadcast, sum grad_subdiag over the broadcast batch dims.
    if subdiag.shape != grad_subdiag.shape:
        # subdiag is (N,) or broadcast — sum over leading dims of grad_subdiag
        ndims_to_sum = grad_subdiag.dim() - subdiag.dim()
        if ndims_to_sum > 0:
            grad_subdiag = grad_subdiag.sum(dim=tuple(range(ndims_to_sum)))
    return grad_subdiag, grad_v, None, None  # None for the two int args


diag_mult.register_autograd(_backward, setup_context=_setup_context)


@diag_mult.register_fake
def _diag_mult_fake(subdiag, v, shift_subdiag, shift_v):
    """Meta kernel — required by Phase 4 D-12. Resolves the 260419-p27 fake-tensor trace bug."""
    return torch.empty_like(v)
```

### Pattern 2: `_torch_ref` pure-PyTorch oracle

**What:** `_torch_ref/diag_mult.py` defines `diag_mult` (and re-exports as `diag_mult` for `__init__.py:__all__`) using `torch.roll`. Accepts complex tensors natively — no `view_as_real` games.

**When to use:** Backward oracle inside `_triton/diag_mult/op.py`; backend binding when `BACKEND=torch` or when `BACKEND=cuda` falls back per D-22; gradcheck oracle in tests.

**Example:**
```python
# torch_structured/_torch_ref/diag_mult.py
# Reference: csrc/diag_mult/diag_mult_cuda_kernel.cu:8.
# Source: torch.roll convention verified empirically (RESEARCH.md §"torch.roll Direction Sanity Check").

import torch


def diag_mult(subdiag: torch.Tensor, v: torch.Tensor,
              shift_subdiag: int, shift_v: int) -> torch.Tensor:
    """Pure-PyTorch oracle for cycle_mult.

    Implements ``out[pos] = subdiag[(pos + shift_subdiag + N) % N] * v[(pos + shift_v + N) % N]``
    via ``torch.roll(t, -shift, dims=-1)``. Accepts real or complex inputs; broadcasts a 1-D
    ``subdiag`` over ``v``'s leading dims.
    """
    assert subdiag.dtype == v.dtype, \
        f"subdiag dtype {subdiag.dtype} != v dtype {v.dtype}"
    assert subdiag.size(-1) == v.size(-1), \
        f"trailing dim mismatch: subdiag {subdiag.size(-1)} vs v {v.size(-1)}"
    return torch.roll(subdiag, -shift_subdiag, dims=-1) * torch.roll(v, -shift_v, dims=-1)
```

[CITED: torch.roll convention validated against C++ kernel by direct Python test — see Open Questions]

### Pattern 3: `_cuda_legacy` try-import + None probe

**What:** `_cuda_legacy/diag_mult.py` performs a top-of-module `try: from torch_structured import _diag_mult_cuda except ImportError: _diag_mult_cuda = None`. The module-level `diag_mult` symbol is either the bound callable or `None`. `_has_cuda_legacy_diag_mult()` in `_ops.py` checks for non-None.

**When to use:** This phase only — to satisfy D-21 + SC#3 (`_diag_mult.so` path remains selectable). Mirrors the pattern in `_cuda_legacy/butterfly.py` and `structured/krylov.py:21-24` (the latter we delete in Phase 5; the pattern lives on in `_cuda_legacy/`).

**Example:**
```python
# torch_structured/_cuda_legacy/diag_mult.py
# Try-import pattern per D-21. Mirrors structured/hadamard.py:1-8 and butterfly/__init__.py loader.

try:
    from torch_structured import _diag_mult_cuda as _diag_mult_cuda_module
except (ImportError, RuntimeError):
    _diag_mult_cuda_module = None


def diag_mult(subdiag, v, shift_subdiag, shift_v):
    """Pass-through to the compiled C++ cycle_mult. Returns None at module load
    time if the .so is absent; ``_has_cuda_legacy_diag_mult()`` in _ops.py
    detects that and falls back to _torch_ref per D-22.
    """
    if _diag_mult_cuda_module is None:
        raise RuntimeError(
            "_diag_mult_cuda not built — caller should use _has_cuda_legacy_diag_mult() probe"
        )
    return _diag_mult_cuda_module.cycle_mult(subdiag, v, shift_subdiag, shift_v)


# Module-level sentinel used by _ops.py probe:
HAS_CUDA_LEGACY_DIAG_MULT: bool = _diag_mult_cuda_module is not None
```

Note: this requires a small adaptation in `_ops.py` because `_has_cuda_legacy_diag_mult()` needs to import this module and check the sentinel — see "Per-Op Resolver Refactor Scope" below.

### Anti-Patterns to Avoid

- **Inventing a new wrapper shape.** The Phase 4 demonstrator is the literal template. Diverging (e.g., using `torch.autograd.Function`, or registering autograd via `_demo_identity_op.register_autograd` outside the file, or splitting `wrap_triton` outside the `@triton_op` body) breaks the contract that PITFALLS §3 calls out as the textbook regression.
- **Flat `_triton/diag_mult.py` instead of nested `_triton/diag_mult/op.py`.** The Phase 4 probe at `_ops.py:96` expects `torch_structured._triton.{op_name}.op`. Flat layout would silently make `_has_triton_kernel("diag_mult")` always return False — exactly the asymmetric trap D-22 is designed to handle, but unnecessarily triggered.
- **Forgetting `.conj()` in the complex backward.** A subtle Wirtinger-convention bug that passes the real test grid but silently fails on complex inputs (see "Backward Gradient Formula" below).
- **Skipping `register_fake`.** The Phase 4 demonstrator's gradcheck and `torch.compile` tests pass because of `register_fake`. Phase 5 inherits the requirement (D-12). Without it, downstream `torch.compile(model)` of a model containing `LDRSubdiagonalC` reproduces the 260419-p27 bug.
- **Routing the backward through Triton instead of `_torch_ref`.** D-26 says `_torch_ref.diag_mult(...)`. Calling Triton would work numerically (Triton's `tl.load` is dtype-polymorphic so fp64 gradcheck would still pass) but it (a) makes the test/oracle separation muddier per Pitfall 4 and (b) loses the fp64 fallback that lets the gradcheck math stay honest.

## Backward Gradient Formula

### Derivation

Let `s = subdiag`, `w = v`. The forward is:

```
out[i] = s[(i + s_sub) mod N] * w[(i + s_v) mod N]
```

**Real case:**

```
∂out[i]/∂s[k] = δ(k - (i + s_sub) mod N) * w[(i + s_v) mod N]
              = δ(i - (k - s_sub) mod N) * w[(k - s_sub + s_v) mod N]

grad_s[k] = Σᵢ grad_out[i] · ∂out[i]/∂s[k]
          = grad_out[(k - s_sub) mod N] * w[(k - s_sub + s_v) mod N]
```

Substituting `j = k` (output index for grad_s):

```
grad_s[j] = grad_out[(j - s_sub) mod N] * w[(j - s_sub + s_v) mod N]
         = cycle_mult(grad_out, w, -s_sub, s_v - s_sub) [j]
```

Symmetrically:

```
grad_w[j] = s[(j + s_sub - s_v) mod N] * grad_out[(j - s_v) mod N]
         = cycle_mult(s, grad_out, s_sub - s_v, -s_v) [j]
```

**Complex case (Wirtinger convention):** PyTorch defines `∂L/∂z` such that `z ← z - α · grad` performs steepest descent on the real-valued loss. For a holomorphic pointwise multiply `out = s ⊙ w`, the autograd convention gives `grad_w = conj(s) * grad_out` (NOT `s * grad_out`). Hence:

```
grad_s[j] = cycle_mult(grad_out, conj(w), -s_sub, s_v - s_sub) [j]
grad_w[j] = cycle_mult(conj(s), grad_out,  s_sub - s_v, -s_v)  [j]
```

For real inputs, `.conj()` is a no-op, so the same formula handles both cases.

### Numerical Validation [VERIFIED: Python on torch 2.11.0 + autograd]

```python
def cycle_mult_ref(subdiag, v, s_sub, s_v):
    return torch.roll(subdiag, -s_sub, dims=-1) * torch.roll(v, -s_v, dims=-1)

N = 8
for s_sub, s_v in [(0,-1), (1,1), (0,0), (-1,1), (1,-1)]:
    s = torch.randn(N, dtype=torch.float64, requires_grad=True)
    w = torch.randn(N, dtype=torch.float64, requires_grad=True)
    out = cycle_mult_ref(s, w, s_sub, s_v)
    grad_out = torch.randn(N, dtype=torch.float64)
    grad_s_auto, grad_w_auto = torch.autograd.grad(out, [s, w], grad_outputs=grad_out)

    pred_grad_s = cycle_mult_ref(grad_out, w.detach().conj(), -s_sub, s_v - s_sub)
    pred_grad_w = cycle_mult_ref(s.detach().conj(), grad_out,  s_sub - s_v, -s_v)
    err_s = (grad_s_auto - pred_grad_s).abs().max().item()
    err_w = (grad_w_auto - pred_grad_w).abs().max().item()
    # Result on real fp64 inputs: err_s == 0.0, err_w == 0.0 for ALL shift pairs
```

**Output (this workstation):**
```
s_sub=+0 s_v=-1: max_err(grad_s)=0.00e+00  max_err(grad_w)=0.00e+00
s_sub=+1 s_v=+1: max_err(grad_s)=0.00e+00  max_err(grad_w)=0.00e+00
s_sub=+0 s_v=+0: max_err(grad_s)=0.00e+00  max_err(grad_w)=0.00e+00
s_sub=-1 s_v=+1: max_err(grad_s)=0.00e+00  max_err(grad_w)=0.00e+00
s_sub=+1 s_v=-1: max_err(grad_s)=0.00e+00  max_err(grad_w)=0.00e+00
```

**Complex case [VERIFIED: same script, complex128 inputs]:**

Without the `.conj()` correction:
```
s_sub=+0 s_v=-1: err_s=3.52e+00 err_w=1.99e+00   ← WRONG
s_sub=+1 s_v=+1: err_s=1.75e+00 err_w=2.97e+00   ← WRONG
```

With `.conj()` on the other operand:
```
s_sub=+0 s_v=-1: err_s=0.00e+00 err_w=0.00e+00   ← CORRECT
s_sub=+1 s_v=+1: err_s=0.00e+00 err_w=0.00e+00   ← CORRECT
s_sub=+0 s_v=+0: err_s=0.00e+00 err_w=0.00e+00   ← CORRECT
```

### Cross-check Against `krylov.py:336` Manual Backward

The existing hand-derived backward in `CycleDownMultCuda.backward` at `structured/krylov.py:336` is:

```python
return diag_mult_cuda.cycle_mult(grad, v, 0, -1).sum(dim=0), diag_mult_cuda.cycle_mult(subdiag, grad, 1, 1)
```

For `(shift_subdiag, shift_v) = (0, -1)` our formula gives:
- `grad_subdiag = cycle_mult(grad_out, conj(v), -0, -1 - 0) = cycle_mult(grad, v, 0, -1)` (real ⇒ conj no-op) — **MATCHES**
- `grad_v       = cycle_mult(conj(subdiag), grad_out, 0 - (-1), -(-1)) = cycle_mult(subdiag, grad, 1, 1)` — **MATCHES**

The `.sum(dim=0)` in the krylov manual backward is the broadcast-batch reduction; our `_backward` callback handles the same case by checking `subdiag.shape != grad_subdiag.shape` and summing over the leading dims. **Confirmed equivalent.**

### Batched-Subdiag Broadcast Verification

When subdiag is `(N,)` and v is `(B, N)`, autograd correctly sums grad_subdiag over the broadcast B dim. Validated:

```
=== Batched v with 1-D subdiag (broadcast), B=4, N=8 ===
max_err(grad_s, sum-over-batch) = 0.00e+00
max_err(grad_w)                  = 0.00e+00
```

The `_backward` callback's `sum(dim=tuple(range(ndims_to_sum)))` matches this autograd behavior. (Note: PyTorch's autograd would emit the sum automatically in the `_torch_ref` path — i.e., a gradcheck under `set_backend('torch')` doesn't need the manual sum. But the Triton path's `register_autograd` does need it because PyTorch's broadcast-sum logic is only auto-applied for native ops, not for the `_torch_ref.diag_mult` call inside our callback.)

## torch.roll Direction Sanity Check [VERIFIED]

The C++ kernel formula `out[pos] = subdiag[(pos + shiftSubdiag + N) % N] * v[(pos + shiftV + N) % N]` matches `torch.roll(t, -shift, dims=-1)` exactly. Verified numerically:

```python
N = 4
v = torch.tensor([10., 20., 30., 40.])
subdiag = torch.tensor([1., 2., 3., 4.])
# C++ with shiftSubdiag=0, shiftV=-1:
#   out[0] = subdiag[0]*v[3] = 40
#   out[1] = subdiag[1]*v[0] = 20
#   out[2] = subdiag[2]*v[1] = 60
#   out[3] = subdiag[3]*v[2] = 120
# Expected: [40, 20, 60, 120]
# torch.roll(v, -shiftV) = torch.roll(v, 1) = [40, 10, 20, 30]
# subdiag * torch.roll(v, 1)                 = [40, 20, 60, 120]   ✓
```

Confirmed for `(s_sub=0, s_v=-1)`, `(s_sub=0, s_v=+1)`, `(s_sub=+1, s_v=+1)`. The `_torch_ref/diag_mult.py` formula is:

```python
torch.roll(subdiag, -shift_subdiag, dims=-1) * torch.roll(v, -shift_v, dims=-1)
```

## Per-Op Resolver Refactor Scope

D-22a explicitly asks the planner to decide between:
- **(A)** Coarse `_BACKEND` global, with a `log.info(per-op bindings)` line at import that prints the actual map.
- **(B)** Per-op `_BACKENDS: dict[str, str]` keyed by op name (e.g., `{"butterfly_multiply": "cuda", "diag_mult": "torch", "hadamard_transform": None}`).

**Recommendation: (A) — keep `_BACKEND` coarse.** Reasons:

1. **Phase 5 only has two ops with a real backend** (`butterfly_multiply` ← cuda from Phase 4; `diag_mult` ← new). The asymmetric case (cuda butterfly + torch diag_mult) is real but rare — it only happens when `_diag_mult.so` isn't built. On this workstation right now: `_butterfly.so` is built, `_diag_mult_cuda.so` is NOT built. So the asymmetric case is the *common* case for current dev environments — but the user-visible behavior is still "set_backend('cuda') gives you cuda where possible, torch_ref otherwise + log.warning". The dict makes that more honest but doesn't change observable behavior.

2. **The Phase 7 explosion is real but not yet here.** When Phase 7 adds a Triton butterfly kernel, `BACKEND=triton` on a host with `_diag_mult.so` absent will give: `butterfly_multiply ← triton, diag_mult ← torch, hadamard_transform ← <whatever Phase 6 ships>`. **The coarse `_BACKEND` will reflect "triton" (the primary) but lie about diag_mult.** This is the future cost — but adding the dict NOW also costs maintenance and is bigger than the current asymmetry warrants.

3. **A `log.info` listing the actual per-op bindings at import** gives users the same visibility for free, without restructuring. Suggested format:
   ```
   torch_structured: backend=cuda (import)
   torch_structured: per-op bindings: butterfly_multiply=cuda, diag_mult=torch (cuda .so absent), hadamard_transform=<unbound>
   ```

4. **Phase 7 entry criteria can revisit.** When the third op lands and the asymmetry is observably confusing in user reports, swap to the dict. Cost of the swap: a single search-replace in `_ops.py` + the per-op probe boolean already exists. No consumers touched.

**Concrete plan for Phase 5:** Keep `_BACKEND` as today. Extend `_resolve()` to bind `diag_mult` via three branches per the actual resolution. After binding, emit a single `log.info` line listing the actual per-op bindings (`log.info("torch_structured: per-op bindings: butterfly_multiply=%s, diag_mult=%s", ...)`).

The `_has_cuda_legacy_diag_mult()` probe is a small addition next to `_has_cuda_legacy()` at `_ops.py:72-79`:

```python
def _has_cuda_legacy_diag_mult() -> bool:
    """True iff _cuda_legacy/diag_mult.py imported the .so successfully."""
    try:
        from torch_structured._cuda_legacy.diag_mult import HAS_CUDA_LEGACY_DIAG_MULT
        return HAS_CUDA_LEGACY_DIAG_MULT
    except ImportError:
        return False
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Complex multiply in Triton | Hand-rolled `tl.complex64` shim | `IS_COMPLEX: tl.constexpr` + `view_as_real` + 4-FMA per `04-COMPLEX-LAYOUT.md` | PITFALLS §1: Triton has no native complex; locked in Phase 4 |
| Autograd Function for the op | `class CycleMult(torch.autograd.Function)` | `@triton_op + register_autograd` | PITFALLS §3: `autograd.Function` breaks `torch.compile`; deprecated path |
| Backward gradient derivation | Hand-derive a Triton kernel | `_torch_ref.diag_mult` (per D-26) | D-26 spec; the formula is closed-form in `cycle_mult` itself; Triton kernel for backward adds no value |
| Cycle indexing in Python | Index arithmetic with `unfold` / `gather` | `torch.roll` with sign verified | Pure-PyTorch reference is 2 lines (verified); anything more is over-engineering |
| Broadcast-sum logic in backward | Inline batch-dim probing | Standard `grad_subdiag.sum(dim=...)` based on shape diff | PyTorch's autograd does this natively for normal ops; we have to manually replicate inside `register_autograd` callbacks (small cost, well-trodden) |
| Single-source kernel that handles fp32 + fp64 | Two separate `@triton.jit` kernels | One `@triton.jit` with `IS_COMPLEX` constexpr; `tl.load` is dtype-polymorphic | The Phase 4 demonstrator's gradcheck passing in fp64 evidences this (kernel ran in fp64 — see Open Question 5 resolution below) |
| Conjugate-aware backward | Skip the `.conj()` for "obviously real" cases | Always include `.conj()` — no-op for real tensors | Single code path; correct for both dtypes (numerically verified) |

**Key insight:** Phase 5 has remarkably little to invent. Every problem has either a Phase 4 template, a derivable closed-form (the backward), or a one-line PyTorch primitive (`torch.roll`).

## Runtime State Inventory

Phase 5 is **not** a rename/refactor/migration of stored data. It introduces new code paths and consumes existing ones. The Inventory is included for completeness:

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — Phase 5 stores no new persistent state | None |
| Live service config | None — local library; no external services | None |
| OS-registered state | None — pure Python package | None |
| Secrets/env vars | `TORCH_STRUCTURED_BACKEND` (existing from Phase 4); no new secrets | None — env-var schema unchanged |
| Build artifacts | `_diag_mult_cuda.cpython-...-x86_64-linux-gnu.so` — currently NOT built on this workstation (verified via `ls /home/claroche/torch-structured/torch_structured/_diag_mult_cuda*` → no matches). D-21/D-22 handle the absence transparently. | None — `setup.py:96-110` conditional build logic stays as-is per D-23 |

## Common Pitfalls

### Pitfall 1: Forgetting `.conj()` in the Complex Backward [CRITICAL]

**What goes wrong:** Complex64 gradcheck silently fails. The real test grid (fp32, fp64) passes, but the moment a complex tensor enters with non-trivial imaginary part, the analytical gradient returned by the `register_autograd` callback disagrees with the numerical perturbation by ~order-of-magnitude.

**Why it happens:** PyTorch's complex autograd uses the Wirtinger convention. For `y = a * b` (pointwise mul), the gradient flowing to `b` is `conj(a) * grad_y`, not `a * grad_y`. Forgetting the conjugate is the canonical complex backward bug.

**How to avoid:** Always wrap the "other" operand with `.conj()` in the backward callback. For real tensors `.conj()` is a no-op (PyTorch optimizes it to a view-or-noop), so there is no perf cost. Single code path.

**Warning signs:**
- `test_diag_mult_gradcheck_fp64_real` PASSES but `test_diag_mult_gradcheck_fp64_complex` FAILS with `Jacobian mismatch for output 0 with respect to input 1`.
- Error magnitude is comparable to the tensor values themselves (not "off by precision").

### Pitfall 2: Confusing `_BACKEND` With Per-Op Backend [MODERATE]

**What goes wrong:** A user reads `_ops._BACKEND == "cuda"` and assumes `diag_mult` is the C++ path. On a workstation without `_diag_mult.so` built, `diag_mult` is actually `_torch_ref.diag_mult` per D-22 fallback — but the global says "cuda". User files a bug; we waste time explaining.

**Why it happens:** D-22a opts for the coarse global. The asymmetry is real but the global doesn't reflect it.

**How to avoid:** The `log.info("per-op bindings: ...")` line at import time makes the actual map visible. Users debugging should `pip install . -v` and check the log. Document this in CHANGELOG. (Phase 7 may upgrade to the dict if reports come in.)

**Warning signs:**
- User report: "I set `TORCH_STRUCTURED_BACKEND=cuda` but it's slow." (Their `.so` is missing; fallback warning was suppressed because they capture logs.)

### Pitfall 3: Contiguity After `torch.cat` [VERIFIED MINOR]

**What goes wrong:** `subdiag_linear_map_cuda` at `krylov.py:343` does `subdiag_extended = torch.cat((corner, subdiag))`. The wrapper at `_triton/diag_mult/op.py` asserts `subdiag.is_contiguous()`. If `torch.cat` returned a non-contiguous tensor, the assert would spuriously fire.

**Why it doesn't happen:** `torch.cat` returns a contiguous tensor (verified empirically: `torch.cat((t1, t2)).is_contiguous()` returns `True`). The assertion never fires for the krylov consumer; it stays as a safety net for other callers.

**How to avoid:** Keep the assert; document the krylov call site as known-safe. If a future caller passes a transposed/strided complex tensor, the assert catches it — exactly the Phase 4 D-01 contract.

### Pitfall 4: Missing `register_fake` Reproduces 260419-p27 Bug [CRITICAL]

**What goes wrong:** Wrapping `_ops.diag_mult` in `torch.compile(model)` raises `"The tensor has a non-zero number of elements, but its data is not allocated yet"` inside dynamo's fake-tensor tracing. Same failure mode as the Phase 4 demonstrator without `register_fake`.

**Why it happens:** Phase 4 D-12 mandates `register_fake` on every Triton op. Phase 5's `_triton/diag_mult/op.py` MUST include the `@diag_mult.register_fake` block.

**How to avoid:** Include `@diag_mult.register_fake` at the end of `op.py`, returning `torch.empty_like(v)`. Add a `test_diag_mult_compile_fake_tensor_trace` or extend `test_diag_mult_gradcheck` to call under `FakeTensorMode` if regression risk is high (the Phase 4 acceptance gate is already proven for the wrapper shape — fewer regression tests needed in Phase 5).

**Warning signs:**
- `test_demo_identity_compile_fake_tensor_trace`-style failure on any new test that exercises `torch.compile(model)` containing an `LDRSubdiagonalC` layer.

### Pitfall 5: Mixed-Dtype Inputs Silently Wrong [MODERATE]

**What goes wrong:** User calls `_ops.diag_mult(real_subdiag, complex_v, 0, -1)`. Without a guard, Triton's `tl.load` interprets the real subdiag tensor as if it were complex — reads garbage. Output is silently wrong.

**Why it happens:** Triton has no runtime dtype check; the kernel trusts the pointer.

**How to avoid:** D-20's `assert subdiag.dtype == v.dtype` at the wrapper boundary. Rejected loudly with `AssertionError` before the kernel launches.

**Warning signs:**
- A consumer module mixes dtypes (audit `LDRSubdiagonalC` — both `subd_A` and `x` come from the same module so this is unlikely, but a future consumer might).

### Pitfall 6: 1-D Subdiag Broadcast — Missing Sum Reduction in Backward [MODERATE]

**What goes wrong:** `_backward` returns `grad_subdiag` of shape `(B, N)` when `subdiag` was `(N,)`. PyTorch's autograd does NOT auto-sum over the broadcast B dim because the `register_autograd` callback is opaque — autograd just hands back whatever shape the callback returns. Training silently overwrites the subdiag gradient with the **last batch element's** contribution (or similar shape-mismatch wrongness).

**Why it happens:** Native PyTorch ops auto-sum broadcast gradients via their `derivatives.yaml` entries. Custom ops don't get this for free.

**How to avoid:** The `_backward` callback explicitly checks `subdiag.shape != grad_subdiag.shape` and sums over the leading broadcast dims. Validated against autograd of `_torch_ref` (which DOES auto-sum since it uses native PyTorch ops). The unit test should include a broadcast case to catch this.

**Warning signs:**
- `LDRSubdiagonal` (uses `(N-1,)` subdiag) training diverges or trains to a worse loss than the eager-CUDA path. Easy to detect with a 5-epoch sanity-train delta against `LDRSubdiagonalC` on the C++ backend.

## Code Examples

Each example below is referenced verbatim by the planner. URLs are intentionally absent — these are project-internal sources only (Phase 4 demonstrator and `04-COMPLEX-LAYOUT.md`).

### Example A: Wrapper-Boundary Complex Routing (from `04-COMPLEX-LAYOUT.md` lines 37-50, adapted)

```python
def wrapper(subdiag, v, shift_subdiag, shift_v):
    is_complex = v.is_complex()
    if is_complex:
        # Pitfall 3: non-contiguous → wrong strides. Assert before view_as_real.
        assert v.is_contiguous(), "v must be contiguous before view_as_real"
        assert subdiag.is_contiguous(), "subdiag must be contiguous before view_as_real"
        v_real = torch.view_as_real(v)
        subdiag_real = torch.view_as_real(subdiag)
    else:
        v_real, subdiag_real = v, subdiag
    out_real = _kernel_invoke(subdiag_real, v_real, IS_COMPLEX=is_complex, ...)
    return torch.view_as_complex(out_real.contiguous()) if is_complex else out_real
```

### Example B: 4-FMA Complex Multiply Inside `@triton.jit` (from `04-COMPLEX-LAYOUT.md` lines 58-76, instantiated for cycle_mult)

```python
if IS_COMPLEX:
    a_re = tl.load(subdiag_ptr + sub_idx * 2,     mask=mask)
    a_im = tl.load(subdiag_ptr + sub_idx * 2 + 1, mask=mask)
    c_re = tl.load(v_ptr + v_idx * 2,     mask=mask)
    c_im = tl.load(v_ptr + v_idx * 2 + 1, mask=mask)
    # (a + bi)(c + di) = (ac - bd) + (ad + bc)i — 4 FMAs
    out_re = a_re * c_re - a_im * c_im
    out_im = a_re * c_im + a_im * c_re
    tl.store(out_ptr + pos * 2,     out_re, mask=mask)
    tl.store(out_ptr + pos * 2 + 1, out_im, mask=mask)
else:
    a = tl.load(subdiag_ptr + sub_idx, mask=mask)
    c = tl.load(v_ptr + v_idx, mask=mask)
    tl.store(out_ptr + pos, a * c, mask=mask)
```

### Example C: `register_autograd` Setup Context and Backward Callback (Phase 4 template, adapted for 4-arg signature)

```python
def _setup_context(ctx, inputs, output):
    subdiag, v, shift_subdiag, shift_v = inputs  # positional unpack
    ctx.save_for_backward(subdiag, v)            # tensors via save_for_backward
    ctx.shift_subdiag = shift_subdiag            # ints as plain attributes
    ctx.shift_v = shift_v

def _backward(ctx, grad_out):
    subdiag, v = ctx.saved_tensors
    s_sub, s_v = ctx.shift_subdiag, ctx.shift_v
    grad_subdiag = diag_mult_torch(grad_out,        v.conj(), -s_sub,    s_v - s_sub)
    grad_v       = diag_mult_torch(subdiag.conj(),  grad_out,  s_sub - s_v, -s_v)
    if subdiag.shape != grad_subdiag.shape:
        ndims_to_sum = grad_subdiag.dim() - subdiag.dim()
        if ndims_to_sum > 0:
            grad_subdiag = grad_subdiag.sum(dim=tuple(range(ndims_to_sum)))
    return grad_subdiag, grad_v, None, None  # 4 returns matching 4 forward inputs

diag_mult.register_autograd(_backward, setup_context=_setup_context)
```

[CITED: PyTorch 2.9 torch.library docs — return `None` for non-tensor positions in backward]

### Example D: `register_fake` Meta Kernel (Phase 4 template, adapted)

```python
@diag_mult.register_fake
def _diag_mult_fake(subdiag, v, shift_subdiag, shift_v):
    """Meta kernel for dynamo fake-tensor tracing — THE 260419-p27 fix."""
    return torch.empty_like(v)  # output shape always matches v (subdiag broadcasts)
```

### Example E: `_torch_ref/diag_mult.py` Reference Oracle

```python
# torch_structured/_torch_ref/diag_mult.py
import torch

def diag_mult(subdiag: torch.Tensor, v: torch.Tensor,
              shift_subdiag: int, shift_v: int) -> torch.Tensor:
    """Pure-PyTorch oracle. ``out[pos] = subdiag[(pos+s_sub+N)%N] * v[(pos+s_v+N)%N]``."""
    assert subdiag.dtype == v.dtype
    assert subdiag.size(-1) == v.size(-1)
    return torch.roll(subdiag, -shift_subdiag, dims=-1) * torch.roll(v, -shift_v, dims=-1)
```

### Example F: `tests/conftest.py` Widening (Phase 4 → Phase 5)

```python
# tests/conftest.py — Phase 5 widens to ["torch", "triton"]
import pytest
import torch_structured

def _has_diag_mult_triton():
    return torch_structured._ops._has_triton_kernel("diag_mult")

@pytest.fixture(params=["torch", "triton"])
def backend(request):
    if request.param == "triton" and not _has_diag_mult_triton():
        pytest.skip("Triton kernel for diag_mult not installed (no CUDA or CPU-only runner)")
    original = torch_structured._ops._BACKEND
    chosen = torch_structured._ops.set_backend(request.param)
    yield chosen
    torch_structured._ops.set_backend(original)
```

### Example G: `tests/test_diag_mult.py` Skeleton (D-29)

```python
# tests/test_diag_mult.py
import itertools

import pytest
import torch

import torch_structured
from torch_structured._torch_ref.diag_mult import diag_mult as diag_mult_ref


def test_diag_mult_eager_fp32(backend):
    N, B = 128, 4
    subdiag = torch.randn(N, device="cuda")
    v = torch.randn(B, N, device="cuda")
    out = torch_structured._ops.diag_mult(subdiag, v, 0, -1)
    expected = diag_mult_ref(subdiag, v, 0, -1)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


def test_diag_mult_eager_complex64(backend):
    N, B = 128, 4
    subdiag = torch.randn(N, dtype=torch.complex64, device="cuda")
    v = torch.randn(B, N, dtype=torch.complex64, device="cuda")
    out = torch_structured._ops.diag_mult(subdiag, v, 0, -1)
    expected = diag_mult_ref(subdiag, v, 0, -1)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


def test_diag_mult_gradcheck_fp64_real(backend):
    N = 16
    subdiag = torch.randn(N, dtype=torch.float64, device="cuda", requires_grad=True)
    v = torch.randn(4, N, dtype=torch.float64, device="cuda", requires_grad=True)
    f = lambda s, x: torch_structured._ops.diag_mult(s, x, 0, -1)
    assert torch.autograd.gradcheck(f, (subdiag, v), eps=1e-6, atol=1e-5)


def test_diag_mult_gradcheck_fp64_complex(backend):
    N = 16
    subdiag = torch.randn(N, dtype=torch.complex128, device="cuda", requires_grad=True)
    v = torch.randn(4, N, dtype=torch.complex128, device="cuda", requires_grad=True)
    f = lambda s, x: torch_structured._ops.diag_mult(s, x, 0, -1)
    assert torch.autograd.gradcheck(f, (subdiag, v), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("s_sub,s_v",
    list(itertools.product([-1, 0, 1], [-1, 0, 1])))
def test_diag_mult_shift_grid(backend, s_sub, s_v):
    N, B = 64, 2
    subdiag = torch.randn(N, device="cuda")
    v = torch.randn(B, N, device="cuda")
    out = torch_structured._ops.diag_mult(subdiag, v, s_sub, s_v)
    expected = diag_mult_ref(subdiag, v, s_sub, s_v)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)
```

Note: `gradcheck` works with the **set_backend('triton')** path because Triton's `tl.load` is dtype-polymorphic — fp64 input ⇒ fp64 kernel execution. AND the `_backward` callback uses `_torch_ref.diag_mult` (per D-26) which also runs in fp64. Both legs of the gradcheck (forward through Triton, backward through `_torch_ref`) preserve fp64. No precision-conversion hack needed.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| C++ `torch::autograd::Function<CycleMult>` registration | Python `triton_op + register_autograd + register_fake` | PyTorch 2.6 (2024) | TRI-05 of v1.2 milestone; Phase 4 locked in |
| Single `_diag_mult_cuda` pybind module with hand-derived backward | `_torch_ref.diag_mult` (pure-PyTorch oracle) + `_triton/diag_mult/op.py` (Triton forward) + `register_autograd` callback for backward | Phase 5 | Decouples kernel from autograd; backward becomes auto-derivable from forward via two more `cycle_mult` calls |
| `torch.autograd.Function.apply` consumer pattern (`cycle_down_mult = CycleDownMultCuda.apply`) | `torch_structured._ops.diag_mult(...)` (attribute-access call) | Phase 5 (D-25) | Honors Phase 4 D-05 contract — `set_backend()` rebindings visible to consumers |

**Deprecated/outdated:**
- `from torch_structured import _diag_mult_cuda as diag_mult_cuda` at `krylov.py:21-24` — deleted per D-25; no replacement, all consumers go through `_ops.diag_mult`.
- `CycleDownMultCuda(torch.autograd.Function)` at `krylov.py:325-339` — deleted per D-24; replaced by `register_autograd` on `_ops.diag_mult`.

## Validation Architecture

[VERIFIED: `pyproject.toml` shows pytest in test extras; existing `tests/test_dispatch.py` and `tests/conftest.py` already follow the pattern.]

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (already present) |
| Config file | None (no `pytest.ini`, no `pyproject.toml` `[tool.pytest.ini_options]` block beyond what Phase 4 may have added; uses pytest defaults) |
| Quick run command | `pytest tests/test_diag_mult.py -v` |
| Full suite command | `pytest tests/` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| TRI-01 | `diag_mult` runs on Triton, fp32 forward | unit | `pytest tests/test_diag_mult.py::test_diag_mult_eager_fp32 -x` | ❌ Wave 0 |
| TRI-01 | `diag_mult` runs on Triton, complex64 forward | unit | `pytest tests/test_diag_mult.py::test_diag_mult_eager_complex64 -x` | ❌ Wave 0 |
| TRI-01 | `diag_mult` autograd correctness, fp64 real | gradcheck | `pytest tests/test_diag_mult.py::test_diag_mult_gradcheck_fp64_real -x` | ❌ Wave 0 |
| TRI-01 | `diag_mult` autograd correctness, fp64 complex | gradcheck | `pytest tests/test_diag_mult.py::test_diag_mult_gradcheck_fp64_complex -x` | ❌ Wave 0 |
| TRI-01 | `diag_mult` shift grid coverage | unit (parametrized 9-way) | `pytest tests/test_diag_mult.py::test_diag_mult_shift_grid -x` | ❌ Wave 0 |
| Phase 5 SC#3 | `LDRSubdiagonalC.forward+backward` still works (integration) | integration | `pytest tests/structured/ -k LDR -x` (if structured tests exist) | partial — check existing tests |
| Phase 5 SC#3 | `_ops.diag_mult` resolves under `BACKEND=cuda` when `.so` present | unit (skip when absent) | `pytest tests/test_diag_mult.py::test_diag_mult_cuda_backend_when_available -x` | ❌ Wave 0 (optional) |

### Sampling Rate

- **Per task commit:** `pytest tests/test_diag_mult.py -v` (≤ ~15 seconds on CUDA, fewer if Triton JIT cache is warm)
- **Per wave merge:** `pytest tests/ -v` (full suite; note pre-existing failures in `test_butterfly.py` / `test_multiply.py` documented in Phase 4 verification as environment-driven, NOT regression — preserve that posture)
- **Phase gate:** Full suite green (modulo Phase-4-documented pre-existing failures) before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `tests/test_diag_mult.py` — covers TRI-01 (forward fp32, forward complex64, gradcheck fp64 real, gradcheck fp64 complex, shift grid)
- [ ] `tests/conftest.py` — widen `backend` fixture from `["torch"]` to `["torch", "triton"]` with `_has_triton_kernel("diag_mult")` skip-gate
- [ ] `tests/test_dispatch.py` — delete 5 demonstrator tests; either keep file as a thin set_backend smoke test OR delete entirely (planner's call)
- [ ] No framework install needed (pytest already present)
- [ ] No `pytest.ini` / `conftest.py` global fixtures needed beyond the `backend` widening

## Security Domain

Phase 5 does NOT add or alter any input handling that crosses a trust boundary. The library is a research-grade CUDA/Triton kernel package consumed inside the Python process; no network, no file I/O, no user-provided code execution. ASVS categories:

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | n/a |
| V3 Session Management | no | n/a |
| V4 Access Control | no | n/a |
| V5 Input Validation | yes (lightly) | `assert` preconditions per CLAUDE.md (dtype match, contiguity, trailing-dim equality) |
| V6 Cryptography | no | n/a |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds Triton kernel load (wrong stride / mask) | Tampering | `mask = pos < N` on every `tl.load`/`tl.store`; `view_as_real`-introduced `(*, 2)` indexing inspected manually against the layout doc |
| Mixed-dtype confusion (real subdiag + complex v) | Tampering (silent wrong output) | `assert subdiag.dtype == v.dtype` at wrapper (D-20) |
| Backward shape mismatch on broadcast (1-D subdiag) | Tampering (silent wrong gradient) | Explicit `sum(dim=...)` in `_backward` callback (validated against autograd of `_torch_ref`) |
| Dynamo fake-tensor regression | Denial of Service (training pipeline fails to compile) | `@register_fake` on the op (D-12; Phase 4 acceptance gate) |

These are not adversarial threats; they are correctness footguns. The asserts and dtype guards are the standard mitigations.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `torch.cat` returns a contiguous tensor in the krylov call site | Pitfall 3 (verified) | None — verified empirically |
| A2 | Triton's `tl.load` is dtype-polymorphic and handles fp64 correctly on the diag_mult kernel | Validation Architecture, Open Question 8 | LOW — Phase 4 demonstrator gradcheck passed with fp64 input; `tl.load` on an fp64 pointer loads fp64. Pointwise multiply also supports fp64. Hardware: any sm_60+ supports fp64 (slower than fp32). |
| A3 | The Wirtinger `.conj()` convention applies to `torch.autograd.gradcheck` on complex inputs | Backward Gradient Formula (numerically validated) | LOW — `torch.autograd.grad` empirically returns the conjugate-corrected gradient on complex pointwise mul; `gradcheck` numerically perturbs and matches the same convention |
| A4 | `register_autograd` correctly handles 4-arg ops with 2 tensor + 2 int positional args | Pattern 1, Example C | LOW — PyTorch 2.9 docs example shows `return grad * ctx.y, None` for 1-tensor-1-float case; same pattern extends to 2-tensor-2-int |
| A5 | The hadamard probe / Phase 6 / Phase 7 will follow the same `_triton/<op>/op.py` shape | Recommended Project Structure | None for Phase 5 (cross-phase concern; not blocking) |
| A6 | The asymmetric `BACKEND=cuda + diag_mult fallback` case is rare enough to defer the dict refactor | Per-Op Resolver Refactor Scope | MEDIUM — on THIS workstation the `_diag_mult.so` is absent, so the asymmetric case IS the current case. Acceptable because `log.info(per-op bindings)` exposes it; revisit in Phase 7. |

## Open Questions (RESOLVED)

1. **Should `tests/test_dispatch.py` be deleted outright after removing the 5 demonstrator tests?**
   - What we know: the 5 demonstrator-specific tests (`test_demo_identity_*`) are tightly coupled to `_demo_identity_op` and have no other purpose.
   - What's unclear: whether any cross-cutting dispatch tests (set_backend round-trip, env-var override, ValueError on unknown backend, B3 honest-probe) are worth keeping as a thin file.
   - RESOLVED: keep `tests/test_dispatch.py` with 2-3 thin smoke tests (`test_set_backend_round_trip`, `test_unknown_backend_raises_value_error`, `test_has_triton_kernel_probe_returns_bool`). These are valuable regression nets for Phase 6/7 dispatch edits and they don't require GPU. If even those don't survive, delete the file.

2. **Should the wrapper boundary support non-contiguous subdiag via internal `.contiguous()` instead of asserting?**
   - What we know: the krylov call site provides contiguous tensors (verified via `torch.cat` semantics). Other callers may not.
   - What's unclear: whether the explicit `assert` is preferred over silent `.contiguous()` copy.
   - RESOLVED: follow Phase 4 D-01 / 04-COMPLEX-LAYOUT.md line 84 ("Warning sign: complex tests pass for non-transposed cases but fail for `Butterfly.forward(input, transpose=True, complex=True)`") — assert. A silent `.contiguous()` would hide perf regressions when a future caller passes a transposed tensor.

3. **For the `_has_cuda_legacy_diag_mult()` probe location — module-level boolean or runtime function?**
   - What we know: Phase 4 chose the runtime-function style (`_has_cuda_legacy()`, `_has_triton_kernel(op_name)`).
   - What's unclear: whether to keep symmetry with a function or shortcut to a module-level boolean.
   - RESOLVED: function style for symmetry. The function is a 3-line wrapper around `from torch_structured._cuda_legacy.diag_mult import HAS_CUDA_LEGACY_DIAG_MULT` inside a try-except.

4. **Should `test_diag_mult_gradcheck_fp64_complex` be conditionally skipped on backends that don't support complex64?**
   - What we know: Phase 5 ships fp32 + complex64; gradcheck uses fp64/complex128 (Triton supports both via `tl.load` polymorphism). `_torch_ref` accepts complex natively (D-20c).
   - What's unclear: whether gradcheck under `backend='triton'` will be fast enough — Triton kernels in fp64 are slow but functional.
   - RESOLVED: run it; if slow, mark `@pytest.mark.slow`. Gradcheck on N=16 is small; expected < 1s even in fp64.

5. **Krylov `subdiag_linear_map_cuda` (line 342) gets called from `LDRSubdiagonalC.forward` which constructs a Krylov matrix via repeated calls — how does this interact with the lambda capture of `subdiag_extended`?**
   - What we know: the lambda `return lambda v: torch_structured._ops.diag_mult(subdiag_extended, v, 0, -1)` captures `subdiag_extended` by closure; each call to the returned function uses the same captured tensor.
   - What's unclear: nothing critical — the closure semantics are standard Python and the captured tensor is contiguous from `torch.cat`.
   - RESOLVED: no action; covered by existing `LDRSubdiagonalC` integration tests (if any) and explicitly noted as the single user-facing change per the code_context section of CONTEXT.md.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.10+ | Test execution, package install | ✓ | 3.13 (workstation) / 3.11 (CI) | — |
| `torch>=2.6` | `triton_op`, `wrap_triton`, `register_autograd`, `register_fake` | ✓ | 2.11.0+cu130 | — (locked floor) |
| `triton>=3.x` | `@triton.jit` kernel | ✓ | 3.6.0 | `_torch_ref` fallback if absent |
| CUDA-capable GPU | Triton kernel execution | ✓ (workstation has) | — | `_torch_ref` fallback when CUDA unavailable |
| `_diag_mult_cuda.so` | `BACKEND=cuda` selection for diag_mult | **✗** (NOT built on workstation; verified via `ls torch_structured/_diag_mult_cuda*` → no matches) | — | `_torch_ref.diag_mult` per D-22 |
| `_butterfly.so` | `BACKEND=cuda` selection for butterfly_multiply | ✓ (built) | — | n/a for Phase 5 |
| pytest | Test execution | ✓ | — (test extra) | — |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:**
- `_diag_mult_cuda.so` — falls back transparently to `_torch_ref.diag_mult` per D-22 with `log.warning` heads-up. This IS the current state of THIS workstation; Phase 5 must ship the fallback path working correctly.

## Sources

### Primary (HIGH confidence)

- `csrc/diag_mult/diag_mult_cuda.cpp:5-16` — `cycle_mult` op signature + `batchedSubdiag` detection (C++ contract being ported)
- `csrc/diag_mult/diag_mult_cuda_kernel.cu:1-17` — pointwise formula `d_Sub[(pos + shiftSubdiag + N) % N] * d_Src[(pos + shiftV + N) % N]`
- `torch_structured/_ops.py:225-304` — Phase 4 demonstrator (the literal template Phase 5 transcribes)
- `torch_structured/_ops.py:42-213` — current resolver implementation (Phase 4); D-22/D-22a edit point
- `torch_structured/structured/krylov.py:21-24, 325-350` — call sites being refactored
- `04-COMPLEX-LAYOUT.md` — canonical `view_as_real + IS_COMPLEX + 4-FMA` template (Phase 4 lock)
- `04-VERIFICATION.md` — Phase 4 line-number map confirming the wrapper shape works end-to-end
- `tests/test_dispatch.py:74-92` — `test_demo_identity_compile_fake_tensor_trace` confirms `register_fake` resolves 260419-p27
- `tests/conftest.py:1-22` — `backend` fixture to extend
- Python numerical verification (this RESEARCH.md run, 2026-05-27) — torch.roll convention, real & complex backward gradient formula validated against `torch.autograd.grad`
- [PyTorch 2.9 torch.library docs](https://docs.pytorch.org/docs/2.9/library.html) — `triton_op`, `register_autograd`, `register_fake` contract; non-Tensor positional args → `None` returns

### Secondary (MEDIUM confidence)

- `.planning/research/PITFALLS.md` §1, §3, §11 — complex layout, triton_op pattern, view_as_real strides
- `.planning/research/ARCHITECTURE.md` — `_triton/<op>/{forward,backward,op}.py` layout convention
- `.planning/research/STACK.md` — Triton 3.x dtype support; `tl.load` polymorphism
- `.planning/quick/260419-p27-extend-recurrent-poc-torch-compile-track/260419-p27-SUMMARY.md` — `register_fake` is the fix

### Tertiary (LOW confidence, flagged for in-implementation validation)

- The exact behavior of `register_autograd` when the op signature includes Python ints — verified via PyTorch 2.9 docs (HIGH→MEDIUM) but not exercised in Phase 4 (the demonstrator has only 1 tensor arg). Mitigation: the planner runs `test_diag_mult_gradcheck_fp64_real` early in execution to detect any mismatch.

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — verified versions installed; locked since Phase 4
- Architecture (`_triton/diag_mult/op.py` shape): HIGH — Phase 4 demonstrator is the literal template; verified by `04-VERIFICATION.md` and the 5/5 passing tests on CUDA
- Backward gradient formula: HIGH — derived symbolically AND numerically validated (error = 0.0) against `torch.autograd.grad` for real and complex cases across the shift grid
- torch.roll direction convention: HIGH — verified empirically against C++ kernel formula
- Pitfalls (complex Wirtinger, broadcast sum, register_fake): HIGH — each has a known regression or test gate
- Per-op resolver refactor recommendation: MEDIUM — defensible argument for coarse global, but the dict form is the more honest long-term answer; revisit at Phase 7

**Research date:** 2026-05-27
**Valid until:** 30 days (stable phase; only changes if Phase 4 contract evolves or `04-COMPLEX-LAYOUT.md` is amended)
