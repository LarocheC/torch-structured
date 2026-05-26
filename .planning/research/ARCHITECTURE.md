# Architecture Research — v1.2 Triton Migration

**Domain:** PyTorch C++/CUDA → Triton kernel migration for a structured-matrix library
**Researched:** 2026-05-26
**Confidence:** HIGH for the PyTorch API decision and integration points (verified against official docs + multiple production examples). MEDIUM for the per-kernel performance characteristics (Triton autotune behaviour depends on workload, not just API).

## Standard Architecture

### Current System (as-of v1.1)

```
┌───────────────────────────────────────────────────────────────────────┐
│                    Public Python API (umbrella)                        │
│   torch_structured/__init__.py  →  Butterfly, ButterflyBmm, LRU, …    │
├───────────────────────────────────────────────────────────────────────┤
│                    Subpackages (nn.Module surface)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ butterfly/  │  │ structured/ │  │  monarch/    │  │ recurrent/  │  │
│  │ Butterfly   │  │ Hadamard,   │  │ Blockdiag…   │  │   LRU       │  │
│  │ +special    │  │ Krylov, LDR │  │ Flash MM     │  │             │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘  └─────────────┘  │
│         │ butterfly_multiply  │ hadamard_transform                     │
│         │ (calls torch.ops…)  │ (calls _hadamard_cuda.…)               │
├─────────┼─────────────────────┼─────────────────────┼─────────────────┤
│         ↓ glob+load_library   ↓ pybind11 import     ↓ pybind11 import │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │              Native extensions (compiled C++/CUDA)                │ │
│  │  _butterfly.so  _version.so    _hadamard_cuda.so   _diag_mult_…  │ │
│  │  (TORCH_LIBRARY registers ops:                                    │ │
│  │   butterfly_multiply_fw / _bw / butterfly_multiply +              │ │
│  │   torch::autograd::Function inside C++)                           │ │
│  └──────────────────────────────────────────────────────────────────┘ │
├───────────────────────────────────────────────────────────────────────┤
│           csrc/ — built by setup.py via torch.utils.cpp_extension     │
│  butterfly.cpp (dispatch + autograd)  cuda/butterfly_cuda.cu          │
│  hadamard/*.cu  diag_mult/*.cu  cpu/butterfly_cpu.cpp                 │
└───────────────────────────────────────────────────────────────────────┘
```

### Target System (v1.2 end state)

```
┌───────────────────────────────────────────────────────────────────────┐
│              Public Python API (unchanged from v1.1)                   │
│   torch_structured/__init__.py  →  Butterfly, ButterflyBmm, LRU, …    │
├───────────────────────────────────────────────────────────────────────┤
│                  Subpackages (nn.Module surface, unchanged)            │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ butterfly/  │  │ structured/ │  │  monarch/    │  │ recurrent/  │  │
│  │ Butterfly   │  │ Hadamard,   │  │ Blockdiag…   │  │   LRU       │  │
│  │ +special    │  │ Krylov, LDR │  │ (no kernel)  │  │             │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────────┘  └─────────────┘  │
│         │ butterfly_multiply  │ hadamard_transform                     │
│         │ (calls _ops.…)      │ (calls _ops.…)                         │
├─────────┴─────────────────────┴────────────────────────────────────────┤
│   torch_structured/_ops.py — single import point for kernels           │
│   Re-exports butterfly_multiply, hadamard_transform, diag_mult, …       │
│   Each = a `@torch.library.triton_op` op with `register_autograd`      │
│   AND a meta/fake kernel for `torch.compile` / dynamo.                 │
├───────────────────────────────────────────────────────────────────────┤
│   torch_structured/_triton/ — pure-Python kernels (lazy import)        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │ butterfly/                hadamard/             diag_mult/      │  │
│  │   forward.py (jit)          forward.py (jit)      kernel.py     │  │
│  │   backward.py (jit)         backward.py (jit)                   │  │
│  │   op.py     (triton_op)     op.py    (triton_op)  op.py         │  │
│  │ _common/                                                         │  │
│  │   autotune_cache.py  dispatch.py (has_triton, has_cuda, …)      │  │
│  └─────────────────────────────────────────────────────────────────┘  │
├───────────────────────────────────────────────────────────────────────┤
│   torch_structured/_torch_ref/  — pure-PyTorch correctness oracle      │
│   butterfly_multiply_torch (already exists; moved here)                │
│   hadamard_transform_torch  (already exists; moved here)               │
│   Always available; runtime fallback when Triton & CUDA both fail.     │
└───────────────────────────────────────────────────────────────────────┘
```

`csrc/`, `setup.py` build shim, MANIFEST.in, and all `.so` artifacts are **deleted** at the end of v1.2. `pyproject.toml` becomes the only build file (with `[build-system] requires = ["hatchling"]` — no compilation step).

### Component Responsibilities

| Component | Responsibility | Implementation |
|-----------|----------------|----------------|
| `torch_structured/_triton/<op>/forward.py` | Triton forward kernel (`@triton.jit`) | Pure Python, no PyTorch imports inside the kernel function body |
| `torch_structured/_triton/<op>/backward.py` | Triton backward kernel(s) (`@triton.jit`) | Same; uses `tl.atomic_add` into the twiddle-grad output |
| `torch_structured/_triton/<op>/op.py` | `@triton_op` wrapper + `register_autograd` + meta kernel | Tensor allocation, grid computation, `wrap_triton(kernel)[grid](…)` calls |
| `torch_structured/_ops.py` | Single re-export point for callers; runs dispatch logic on import | `from ._triton.butterfly.op import butterfly_multiply` (or fallback) |
| `torch_structured/_torch_ref/*.py` | Pure-PyTorch reference (existing `_torch` functions) | Same code that's currently `butterfly_multiply_torch` |
| `torch_structured/butterfly/butterfly.py` (and other nn.Modules) | Public `nn.Module` surface, unchanged | Calls `butterfly_multiply` from `_ops.py` rather than from `.multiply` |
| `torch_structured/_triton/_common/dispatch.py` | Backend probing (`has_triton()`, `has_cuda()`, env-var overrides) | Module-level constants computed once at import |

## Recommended Project Structure

```
torch_structured/
├── __init__.py                  # public re-exports — unchanged surface
├── factory.py                   # unchanged
├── _ops.py                      # NEW — single import point for kernels
├── _torch_ref/                  # NEW — pure-PyTorch reference impls
│   ├── __init__.py
│   ├── butterfly.py             # butterfly_multiply_torch
│   ├── hadamard.py              # hadamard_transform_torch
│   └── diag_mult.py             # subdiag_mult_torch (currently lives in krylov.py)
├── _triton/                     # NEW — Triton kernels live here
│   ├── __init__.py              # lazy: try-import triton, expose HAS_TRITON
│   ├── _common/
│   │   ├── dispatch.py          # backend probing + env-var TORCH_STRUCTURED_BACKEND
│   │   └── autotune.py          # shared autotune configs / cache helpers
│   ├── butterfly/
│   │   ├── __init__.py
│   │   ├── forward.py           # @triton.jit forward kernel(s)
│   │   ├── backward.py          # @triton.jit backward kernel(s) (atomic_add into d_twiddle)
│   │   └── op.py                # @triton_op + register_autograd + register_fake
│   ├── hadamard/
│   │   ├── __init__.py
│   │   ├── forward.py           # @triton.jit Walsh-Hadamard
│   │   └── op.py                # @triton_op (self-inverse, so backward = forward)
│   └── diag_mult/
│       ├── __init__.py
│       ├── kernel.py            # @triton.jit (single kernel, fw and bw share)
│       └── op.py                # @triton_op + register_autograd
├── butterfly/                   # unchanged file paths; multiply.py shrinks
│   ├── __init__.py              # remove _load_extension + check_cuda_version
│   ├── multiply.py              # thin re-export from _ops; keep `butterfly_multiply_torch` re-export
│   ├── butterfly.py             # unchanged (still imports butterfly_multiply)
│   ├── butterfly_base4.py       # unchanged
│   └── …                        # complex_utils, combine, permutation, special — unchanged
├── structured/
│   ├── hadamard.py              # change: hadamard_transform_cuda → hadamard_transform from _ops
│   ├── krylov.py                # change: diag_mult_cuda → from _ops import diag_mult
│   └── …                        # rest unchanged
├── monarch/                     # unchanged; flash_mm.py keeps raising NotImplementedError
└── recurrent/                   # unchanged
```

### Structure Rationale

- **`_triton/` is a top-level peer of `butterfly/`, `structured/`, …** rather than `butterfly/_triton/` because (a) `diag_mult` and `hadamard` are consumed from `structured/`, not `butterfly/` — putting them under a subpackage they don't own is misleading; (b) a single backend folder makes it trivial to add/remove backends in the future (CUDA-Triton, ROCm-Triton, native-PyTorch fallback all become sibling directories); (c) mirrors how Liger-Kernel organises (`liger_kernel/ops/` flat, by op, not by consuming module).
- **`_ops.py` is the single import point** so each consumer (`butterfly.py`, `hadamard.py`, `krylov.py`) imports `from torch_structured._ops import butterfly_multiply` and never touches `_triton/` directly. This makes the dispatch logic (Triton vs torch-fallback) live in exactly one place and makes test parametrisation trivial (`monkeypatch._ops.butterfly_multiply`).
- **`_torch_ref/` is its own folder** rather than scattered in subpackages because it serves a cross-cutting role (correctness oracle in tests + runtime fallback). Keeping it in one folder makes the "find me all pure-PyTorch reference impls" question answerable with one `ls`.
- **Leading underscore (`_triton`, `_ops`, `_torch_ref`)** marks all of this as private. The public API does not include any of these module paths — only the re-exports through `torch_structured.butterfly.*` and the top-level `torch_structured.*` continue to be supported.
- **No `torch_structured/_triton/butterfly/butterfly.py`**: the file holding the `nn.Module` (`butterfly/butterfly.py`) does not move. Triton holds *only* the kernel and the op registration.

## Architectural Patterns

### Pattern 1: `torch.library.triton_op` + `wrap_triton` + `register_autograd`

**What:** Each kernel is registered as a *transparent* custom op via `@torch.library.triton_op("torch_structured::butterfly_multiply", mutates_args={})`. Inside, `wrap_triton(my_kernel)[grid](…)` invokes the actual `@triton.jit` function. Autograd is added through `op.register_autograd(backward, setup_context=…)`. A meta/fake kernel is registered through `op.register_kernel("meta", …)` so `torch.compile` and `dynamo` can trace through it without a real device.

**When to use:** PyTorch >= 2.6 (which the project already targets via the "PyTorch 2.x" constraint, comfortably below the current stable 2.9+). For every kernel that has gradients (butterfly forward, diag_mult). For kernels that need to be visible to `torch.compile`.

**Trade-offs:**

| Aspect | `triton_op` (recommended) | Plain `torch.autograd.Function` wrapping `@triton.jit` | `torch.library.custom_op` |
|--------|---------------------------|--------------------------------------------------------|---------------------------|
| `torch.compile` traces *into* the op | YES (designed for it) | NO — opaque boundary | NO — opaque by design |
| Autograd | `register_autograd` (no Function subclass) | Native | `register_autograd` |
| `torch.export` / AOT-Inductor | Native | Limited | Native |
| Documented as the recommended path for Triton in 2026 | YES | Legacy | "Use this when you can't compile" |
| Effort | One decorator + one backward fn | Subclass with `forward`/`backward` | Same as `triton_op` but opaque |
| Compatibility with dynamo fake-tensor tracing | Native (when fake kernel registered) | Requires manual `register_fake` (we already hit this — see PITFALLS) | Requires `register_fake` |

The dynamo fake-tracing footgun that bit us in the recurrent POC (see `.planning/quick/260419-p27-…/260419-p27-SUMMARY.md`: *"The butterfly torch.ops…butterfly_multiply custom op raises inside dynamo's fake-tensor tracing"*) is **resolved for free** by `triton_op` because the op exposes its implementation to dynamo. This is a load-bearing reason to pick `triton_op` over either alternative.

**Example (butterfly forward + backward):**
```python
# torch_structured/_triton/butterfly/op.py
import torch
from torch.library import triton_op, wrap_triton
from .forward import _butterfly_fw_kernel
from .backward import _butterfly_bw_kernel

@triton_op("torch_structured::butterfly_multiply", mutates_args={})
def butterfly_multiply(twiddle: torch.Tensor, x: torch.Tensor,
                       increasing_stride: bool,
                       output_size: int | None = None) -> torch.Tensor:
    # validate, allocate output, compute grid
    out = torch.empty(..., device=x.device, dtype=x.dtype)
    grid = lambda meta: (batch_size, nstacks, triton.cdiv(n, meta["BLOCK_N"]))
    wrap_triton(_butterfly_fw_kernel)[grid](twiddle, x, out, ...,
                                            BLOCK_N=128)
    return out

def _setup_context(ctx, inputs, output):
    twiddle, x, increasing_stride, _ = inputs
    ctx.save_for_backward(twiddle, x)
    ctx.increasing_stride = increasing_stride

def _backward(ctx, grad):
    twiddle, x = ctx.saved_tensors
    d_twiddle = torch.zeros_like(twiddle)   # atomic_add target — must be zero-init
    d_x = torch.empty_like(x)
    grid = ...
    wrap_triton(_butterfly_bw_kernel)[grid](
        twiddle, x, grad, d_twiddle, d_x, ...,
        INCREASING_STRIDE=ctx.increasing_stride)
    return d_twiddle, d_x, None, None

butterfly_multiply.register_autograd(_backward, setup_context=_setup_context)

@butterfly_multiply.register_kernel("meta")
def _meta(twiddle, x, increasing_stride, output_size):
    n = 1 << twiddle.shape[2]
    out_size = output_size if output_size is not None else n
    return x.new_empty(x.shape[0], x.shape[1], out_size)
```

### Pattern 2: Single dispatch point, lazy probe at import

**What:** `_ops.py` runs a one-time probe (`HAS_TRITON = ...`, `HAS_CUDA = torch.cuda.is_available()`), reads `TORCH_STRUCTURED_BACKEND` env var, and binds the public names (`butterfly_multiply`, `hadamard_transform`, `diag_mult`) to the chosen implementation. No per-call branching.

**When to use:** Always for this project. The dispatch decision is global (you don't mix backends per-call) and doesn't depend on tensor shape — so per-call cost is wasted.

**Trade-offs:**
- Per-call dispatch (the flash-attention style: every call checks `x.device`, `triton_available`, etc.): correct but adds overhead and complicates `torch.compile` tracing. Reject.
- Module-level toggle (chosen): one source of truth, easy to override in tests via `monkeypatch.setattr("torch_structured._ops.butterfly_multiply", _torch_ref.butterfly_multiply_torch)`.
- `torch.backends.*` registration: only meaningful for backends PyTorch itself knows about (cuDNN, MIOpen, …). Not applicable.

**Example:**
```python
# torch_structured/_ops.py
import os
import torch
from . import _torch_ref

_BACKEND = os.environ.get("TORCH_STRUCTURED_BACKEND", "auto").lower()

try:
    import triton  # noqa: F401
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if _BACKEND == "torch" or not HAS_TRITON or not torch.cuda.is_available():
    butterfly_multiply = _torch_ref.butterfly_multiply_torch
    hadamard_transform = _torch_ref.hadamard_transform_torch
    diag_mult = _torch_ref.subdiag_mult_torch
else:
    from ._triton.butterfly.op import butterfly_multiply
    from ._triton.hadamard.op import hadamard_transform
    from ._triton.diag_mult.op import diag_mult
```

### Pattern 3: Backward via separate kernel with `tl.atomic_add` into `d_twiddle`

**What:** Two kernels per op with gradient (forward and backward), not one fused kernel. Backward `d_twiddle` is the only term needing cross-thread accumulation (each input element touches each twiddle factor across batch and stack dims). Use `tl.atomic_add(d_twiddle_ptr, value, mask=…)` inside backward. `d_x` is element-wise local and needs no atomics.

**When to use:** Default for v1.2. A fused forward+backward (the "compute fw, save residuals, run bw in same kernel") only helps when activation memory is the bottleneck; for butterfly it isn't — `twiddle` is tiny, `x` and `grad` are the activations and they're not big in modern training.

**Trade-offs vs. the existing C++/CUDA backward:**
- The current C++/CUDA kernel uses hand-written reductions across `MAX_BLOCK_SIZE` with painstaking `WARP_SIZE` / `ITEMS_PER_THREAD` constants per `log_n` (see `csrc/cuda/butterfly_cuda.cu`). Triton's autotuner subsumes this — define a few candidate `BLOCK_N` / `num_warps` / `num_stages` and let Triton pick per (`log_n`, GPU arch) at first call.
- `atomicAdd` correctness was already a load-bearing assumption in the C++ version (see `THCAtomics.cuh` include); Triton's `tl.atomic_add` on `float32` is supported on all PTX targets we care about (sm70+).
- One regression risk: complex tensors. The current C++ kernel does atomic_add on `c10::complex<float>` via PyTorch's custom helper. Triton has limited complex support — recommendation: **decompose complex twiddle into two `float32` tensors (real, imag)** at the op boundary, run two real kernels, recombine. This matches how Triton tutorials handle complex.

**Example (backward kernel sketch):**
```python
# torch_structured/_triton/butterfly/backward.py
import triton, triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64},  num_warps=2),
        triton.Config({"BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_N": 256}, num_warps=8),
    ],
    key=["log_n"],
)
@triton.jit
def _butterfly_bw_kernel(
    twiddle_ptr, x_ptr, grad_ptr, d_twiddle_ptr, d_x_ptr,
    batch_size, nstacks, log_n,
    INCREASING_STRIDE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # … load tile of twiddle, x, grad …
    # … compute d_x in registers (no atomics) …
    # tl.store(d_x_ptr + offs, d_x_vals, mask=…)
    # … compute d_twiddle contribution for this tile …
    # tl.atomic_add(d_twiddle_ptr + tw_offs, d_tw_vals, mask=…)
```

## Data Flow

### Forward call flow (after v1.2)

```
user code
    │
    ▼
nn.Module.Butterfly.forward()           # torch_structured/butterfly/butterfly.py — UNCHANGED
    │  output = butterfly_multiply(twiddle, x, increasing_stride, output_size)
    ▼
torch_structured._ops.butterfly_multiply   # NEW — bound at import to one of:
    │
    ├──→  _triton.butterfly.op.butterfly_multiply         (Triton path, default on CUDA)
    │       │  @triton_op  →  wrap_triton(_fw_kernel)[grid](…)
    │       ▼
    │     _fw_kernel  (PTX, JIT-compiled by Triton at first call, cached)
    │
    └──→  _torch_ref.butterfly_multiply_torch              (fallback, CPU or TORCH_STRUCTURED_BACKEND=torch)
```

### Backward flow

```
loss.backward()
    │
    ▼
dispatcher looks up registered autograd
    │  (registered via `butterfly_multiply.register_autograd(_backward, setup_context=…)`)
    ▼
_backward(ctx, grad_out)
    │  d_twiddle = torch.zeros_like(twiddle)            # zero-init for atomics
    │  d_x = torch.empty_like(x)
    │  wrap_triton(_bw_kernel)[grid](…)                 # atomic_add into d_twiddle, store into d_x
    ▼
returns (d_twiddle, d_x, None, None)
```

### Compilation/caching flow (Triton's responsibility, not ours)

```
first call to butterfly_multiply(…) on a GPU
    │
    ▼
@triton.autotune runs all configs once on representative input
    │  picks winner for (log_n, dtype, GPU arch) key
    ▼
@triton.jit compiles winning config to PTX  →  caches in ~/.triton/cache/
    │
    ▼
subsequent calls (in this process or any future process on same GPU+config)
    │
    ▼
load cached PTX, launch.   Total overhead ~ a few hundred µs once, ~0 thereafter.
```

This replaces the current "compile to .so at `pip install` time, link arch-specific code via `TORCH_CUDA_ARCH_LIST`" model. **`TORCH_CUDA_ARCH_LIST` becomes irrelevant** (Triton recompiles per actual GPU). **The CUDA version mismatch check in `butterfly/__init__.py` becomes irrelevant** (no C++ compiled against a specific CUDA version).

## Build Order

The migration must respect three dependencies:

1. **Infrastructure first.** `_torch_ref/` must exist and be wired in *before* any Triton kernel lands — it's the rollback for every kernel.
2. **Forward before backward.** A Triton forward with a torch-reference backward is a valid intermediate state. The reverse (Triton backward with torch forward) is not (the backward derivation matches the forward's algorithm).
3. **Smallest kernel first.** Build muscle memory on a kernel that's easy to verify before tackling butterfly.

Recommended phase ordering (the roadmap will turn each into a phase or sub-phase):

| Order | Step | Depends on | What lands | Why this order |
|-------|------|-----------|------------|----------------|
| **0** | Infrastructure: `_torch_ref/`, `_ops.py`, `TORCH_STRUCTURED_BACKEND` env var, backend probe, test harness that parametrises `[triton, cuda, torch]` | nothing | Folder structure, dispatch table, three-way test fixture | Everything below assumes this exists. No new kernels yet. |
| **1** | Port `diag_mult` | step 0 | `_triton/diag_mult/{kernel.py, op.py}`; `krylov.py` switches its `diag_mult_cuda` import to `_ops.diag_mult` | Smallest kernel (~10 lines of CUDA). Single-pass, no reductions, no atomics. Validates the `triton_op`+autograd plumbing end-to-end with minimal risk. |
| **2** | Port `hadamard` forward | step 1 | `_triton/hadamard/{forward.py, op.py}`; `structured/hadamard.py` switches | Self-inverse: backward = forward, so the "two-kernel" pattern collapses to one. Two-pass (mixed-radix) gives experience with shared memory / `tl.dot`-style reductions without atomic gradients. |
| **3** | Port `butterfly_multiply` **forward** | step 2 | `_triton/butterfly/{forward.py, op.py}`; `_ops.butterfly_multiply` now uses Triton forward + `_torch_ref.butterfly_multiply_torch` backward (via plain `register_autograd`) | Largest forward kernel. Validates that the existing C++ backward can be kept temporarily while forward moves. Tests still pass because backward correctness comes from `_torch_ref`. |
| **4** | Port `butterfly_multiply` **backward** | step 3 | `_triton/butterfly/backward.py`; replace torch-ref backward with Triton backward via `register_autograd` | Highest-risk kernel: atomic gradient accumulation into `d_twiddle`. Has the most autotune surface (per-`log_n` config). Now the C++ extension is no longer called for anything. |
| **5** | Deletion phase: remove `csrc/`, `setup.py`, MANIFEST.in, `_load_extension`, `check_cuda_version`. Switch `pyproject.toml` to hatchling. | step 4 | Pure-Python package; `pip install .` no longer compiles | Cannot delete until every kernel + every consumer has switched. The CUDA-version check in `butterfly/__init__.py` and the `_load_extension` glob also disappear here. |

**Critical invariant:** at the end of each step, the test suite passes on a CUDA machine AND the existing C++/CUDA build still works. This is what the parallel-paths strategy in `PROJECT.md` ("Triton kernels live alongside existing CUDA during migration; a runtime flag selects one") buys us. Each step is independently revertable.

A reasonable shortcut if pre-step audit shows `_diag_mult_cuda` and `_hadamard_cuda` are not actually on any hot path that users hit (the imports are `try/except ImportError` and the consumers degrade gracefully): port `diag_mult` and `hadamard` in step 1 in parallel rather than serially. Butterfly cannot be shortcut.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| Single GPU, single config | The autotuner runs once on first call, caches to `~/.triton/cache/`. Subsequent processes on the same machine reuse the cache. No adjustments needed. |
| Multiple GPU architectures on one machine (e.g., dev box with H100, CI with A100) | Triton autotune caches per (op, key, GPU arch) so this is handled. Just make sure `key=` arguments to `@triton.autotune` include the dimensions that matter (`log_n`, `nstacks`, dtype). |
| CI / cold cache | First-call autotune cost is real (~1-10s for a kernel with many configs). Recommendation: keep autotune `configs=` list short (3-5 options). Don't enumerate every BLOCK_SIZE × num_warps combination. |
| Many small calls (e.g., tight loop calling butterfly per-batch) | `@triton_op` adds dispatcher overhead vs. raw `kernel[grid](…)`. Negligible if `n >= 256`; measurable at `n == 4`. Not a concern for the actual butterfly use case (`n` is power-of-2 dimension, typically 128+). |

### Scaling Priorities

1. **First bottleneck — autotune time on cold cache.** Keep the autotune config list small. Document the warm-up cost in README.
2. **Second bottleneck — kernel launch overhead for small `n`.** Not a v1.2 concern. If users complain, the answer is "fall back to `TORCH_STRUCTURED_BACKEND=torch`" because at very small `n` the Triton overhead exceeds the gain.

## Anti-Patterns

### Anti-Pattern 1: Subclass `torch.autograd.Function` and call Triton inside

**What people do:** Write `class ButterflyFn(torch.autograd.Function): def forward(ctx, …): kernel[grid](…)` — i.e., the legacy autograd pattern with Triton inside.

**Why it's wrong:**
1. Documented as the *not-recommended* path in PyTorch 2.6+ docs. "Prefer `register_autograd` to using `torch.autograd.Function` (which has various composability footguns with `torch.compile`)".
2. Opaque to `torch.compile` / dynamo — the very fake-tensor tracing bug we already hit (`260419-p27`).
3. Cannot register a fake/meta kernel cleanly.

**Do this instead:** `@triton_op` + `register_autograd`.

### Anti-Pattern 2: Per-call backend dispatch (`if has_triton(): … else: …` inside every Module forward)

**What people do:** Each `nn.Module.forward` does a runtime check of which backend to call.

**Why it's wrong:** Adds Python overhead, complicates `torch.compile` tracing, scatters the dispatch decision across N files. When you want to change the rule, you change N files.

**Do this instead:** Bind once in `_ops.py` at import time. Consumers import the name and call it.

### Anti-Pattern 3: Keep `butterfly_multiply_torch` only as a test oracle and delete the import path

**What people do:** Move the pure-PyTorch reference to `tests/` and remove it from the installable package.

**Why it's wrong:** Users who pip-install on a CUDA-less machine (CPU laptop, CI without GPU runner, Apple Silicon) need *some* working path. Triton doesn't ship Windows or macOS wheels and only supports NVIDIA + AMD GPUs. Without a runtime CPU fallback, those users can't `import torch_structured` at all.

**Do this instead:** Keep `_torch_ref/` shipped in the package. It's already written, tested, slow-but-correct. Cost of including it: a few hundred lines of pure-Python. Benefit: cross-platform install without errors.

### Anti-Pattern 4: Eagerly import Triton at top-level `__init__.py`

**What people do:** `import triton` at the top of `torch_structured/__init__.py`.

**Why it's wrong:** Triton import is non-trivial (loads LLVM bits). Eager import slows `import torch_structured` for users who only want, e.g., the `LRU` module. Also breaks installs without Triton (Windows, macOS, CPU-only Linux).

**Do this instead:** Triton is imported lazily inside `_ops.py` behind a `try/except`. The umbrella `torch_structured/__init__.py` should never itself touch `_triton/`.

## Integration Points

### External Services (toolchain)

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Triton (>=3.0) | `pyproject.toml` install_requires; lazy import in `_ops.py` | No build step. Wheels available on PyPI for Linux x86_64 (NVIDIA) and Linux for ROCm. **Windows: triton-windows fork.** **macOS: no support — fall back to `_torch_ref`.** |
| PyTorch (>=2.6) | Already installed by user; `_ops.py` uses `torch.library.triton_op` | Version bump: PROJECT currently says ">=2.0" but `triton_op` needs 2.6. The roadmap must bump this constraint. |
| CUDA toolkit | **No longer needed at install time.** | Removed: `nvcc`, `TORCH_CUDA_ARCH_LIST`, `FORCE_CUDA`, `FORCE_CPU`, `BUILD_DOCS`. Runtime: CUDA driver + libs are bundled with the user's PyTorch wheel. |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `butterfly/butterfly.py` → kernel | `from torch_structured._ops import butterfly_multiply` | Single import line change vs. today's `from .multiply import butterfly_multiply`. Same callable signature, same behaviour. |
| `structured/hadamard.py` → kernel | `from torch_structured._ops import hadamard_transform` | Replaces today's `from torch_structured import _hadamard_cuda as hadamard_cuda` plus the `try/except ImportError` toggle. |
| `structured/krylov.py` → kernel | `from torch_structured._ops import diag_mult` | Replaces `from torch_structured import _diag_mult_cuda`. |
| `_ops.py` → backend | Module-level binding at import. Re-import (importlib.reload) is unsupported; tests use monkeypatch instead. | Crisp: dispatch is a one-time decision. |
| `_triton/<op>/op.py` → `@triton.jit` kernel | `wrap_triton(_kernel)[grid](…)` inside the `@triton_op`-decorated function | Mandatory wrapping — raw `_kernel[grid](…)` calls outside `wrap_triton` are not traceable. |
| `tests/` → backend selection | `pytest -k backend_triton`, `-k backend_torch`, `-k backend_cuda` via parametrize fixture | One fixture in `conftest.py` parametrises a `backend` value; the test uses `monkeypatch.setattr("torch_structured._ops.butterfly_multiply", impl_for[backend])`. |

### Public API stability

The `nn.Module` surface — `Butterfly`, `ButterflyBmm`, `ButterflyBase4`, `ButterflyUnitary`, the `special.fft`/`dct`/`hadamard` factory functions, `LRU`, `make_linear` — **must not change**. Their constructors, attributes (`twiddle`, `bias`, `_is_structured`), state-dict keys, and `forward` signatures stay byte-identical. The only internal change: where they get `butterfly_multiply` from.

Top-level `torch_structured.butterfly_multiply` (re-exported in `__init__.py`) also stays callable with the same signature `(twiddle, input, increasing_stride, output_size=None) -> Tensor`. The implementation behind it changes; the contract doesn't.

### Tests architecture

| File | What it covers | New parametrisation |
|------|---------------|---------------------|
| `tests/conftest.py` | NEW — defines `backend` fixture parametrised over `["triton", "torch"]` (and "cuda" only if the legacy `.so` is present, for as long as it is) | All cross-backend tests pull this fixture |
| `tests/test_multiply.py` | Existing — currently iterates `for device in ['cpu', 'cuda']` and compares to `butterfly_multiply_torch` | Add `backend` parametrisation; existing oracle (`butterfly_multiply_torch`) becomes the authoritative reference |
| `tests/test_butterfly.py`, `test_butterfly_base4.py`, `test_special.py` | Existing — exercise `nn.Module` surface | No change needed; they go through `_ops` automatically. Verifies the API-stability invariant. |
| `tests/structured/test_hadamard.py` | Existing — currently has `cuda_ext = pytest.importorskip("torch_structured._hadamard_cuda")` | Add a `triton` branch; `_hadamard_cuda` branch deprecated after step 5 |
| `tests/_triton/test_butterfly_kernel.py` | NEW — kernel-level tests (correct grid, dtype handling, edge cases like `n == 1`, `output_size < n`, complex inputs decomposed as real pair) | Run only when `HAS_TRITON and torch.cuda.is_available()` |
| `tests/_triton/test_dispatch.py` | NEW — verifies `TORCH_STRUCTURED_BACKEND=torch` overrides Triton selection | Run on any platform |

The key insight: by making every consumer go through `_ops.py`, we get *N* tests (one per consumer) parametrising backend for free — without touching N test files. The cost is one fixture in `conftest.py`.

## Sources

- [PyTorch — torch.library documentation (2.9, current stable)](https://docs.pytorch.org/docs/2.9/library.html) — `triton_op`, `wrap_triton`, `register_autograd`, `register_kernel` reference
- [PyTorch 2.6 Release Blog](https://pytorch.org/blog/pytorch2-6/) — `triton_op` introduced as stable feature in 2.6
- [Using User-Defined Triton Kernels with torch.compile — PyTorch tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html) — canonical example of `triton_op` + `wrap_triton` + `register_autograd` with forward and backward Triton kernels
- [PyTorch dev-discuss: Custom Ops Under torch.compile — autograd.Function vs torch.library.custom_op](https://dev-discuss.pytorch.org/t/custom-ops-under-torch-compile-autograd-function-vs-torch-library-custom-op/3338) — the dynamo / fake-tensor footgun and why `triton_op` is the recommended path in 2026
- [Liger-Kernel — LinkedIn's production Triton kernel library](https://github.com/linkedin/Liger-Kernel) — reference for `ops/<op>.py` per-op layout (kernel + autograd in one file) and minimal-deps (torch + triton) packaging
- [Liger-Kernel `rms_norm.py`](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rms_norm.py) — `@triton.jit` kernel + `torch.autograd.Function` co-located, validates the "kernel + op wrapper next to each other" pattern
- [Flash-Linear-Attention `fla-org/flash-linear-attention`](https://github.com/fla-org/flash-linear-attention) — split package: `fla-core` (kernels + torch + triton + einops) vs `flash-linear-attention` (high-level layers + transformers). Confirms separation of kernels from modules.
- [Triton documentation — installation and platform support](https://triton-lang.org/main/getting-started/installation.html) — Linux-only PyPI wheels (NVIDIA + ROCm); macOS unsupported. Justifies the `_torch_ref/` runtime fallback.
- Internal: `.planning/quick/260419-p27-extend-recurrent-poc-torch-compile-track/260419-p27-SUMMARY.md` — documents the dynamo fake-tensor bug on the current C++ `butterfly_multiply` op; `triton_op` + meta kernel is the fix.
- Internal: `.planning/research/PITFALLS.md` line 86 — flagged the modern `torch.library` migration as a "larger refactor" in v1.0 research; v1.2 is that refactor.

---
*Architecture research for: v1.2 Triton Migration of torch_structured*
*Researched: 2026-05-26*
