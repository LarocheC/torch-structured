# Stack Research — Triton Migration

**Domain:** PyTorch C++/CUDA extension → Triton kernel port (structured-matrix primitives)
**Researched:** 2026-05-26
**Confidence:** HIGH (Triton + PyTorch version pinning, autograd APIs, install paths verified against current
official docs and PyPI; MEDIUM on tooling-choice opinions where multiple sane options exist)

## Recommended Stack

Migration is **additive** to the v1.1 stack. Nothing in `pyproject.toml` actually has to *change* on day 1:
the `triton` Python package is already a transitive dependency of `torch` on CUDA Linux wheels. New items
below are integration points and tooling, not new core deps.

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `triton` (the language/compiler from triton-lang) | 3.5.x or 3.6.x (whatever Torch pins) | Replace `csrc/cuda/*.cu` JIT-compiled at first use | Bundled with PyTorch 2.9+ CUDA wheels; no toolchain or compile step at install. Triton 3.x is the standard authoring path for custom GPU kernels and the language `torch.library.triton_op` is built for. |
| `torch` | **>=2.6** (recommend >=2.9 floor; keep `>=2.0` as a lower-bound for the Python-only paths) | Provides `torch.library.triton_op`, `torch.library.wrap_triton`, `torch.library.register_autograd` | `triton_op` and `wrap_triton` first shipped in PyTorch 2.6; they are the post-2024 idiomatic way to attach a Triton kernel to autograd *and* keep `torch.compile` tracing through it. The legacy `torch.autograd.Function` + `TORCH_LIBRARY` C++ shim we have today is no longer the recommended pattern. |
| `torch.library.triton_op` decorator | (Torch built-in, ≥2.6) | Define `torch_structured::butterfly_multiply` etc. as a real op with FakeTensor/Meta and Inductor-traceable kernel calls | `torch.compile` traces *into* `triton_op`-wrapped kernels (vs treating them as opaque). Lets users get fusion when they `torch.compile` a model containing our layers. Mandatory for the FX-graph-friendly story. |
| `torch.library.wrap_triton` | (Torch built-in, ≥2.6) | Wraps each `@triton.jit` kernel call site inside the `triton_op` function body | Required marker so Inductor can lift the kernel into the compiled graph instead of breaking on it. |
| `torch.library.register_autograd` | (Torch built-in) | Replaces `torch::autograd::Function<ButterflyMultiply>` in `csrc/butterfly.cpp` lines 99–125 | Pure-Python autograd registration; setup_context + backward (which itself calls another `triton_op` whose body uses `wrap_triton(butterfly_bw_kernel)`). |
| Pure-PyTorch reference (`butterfly_multiply_torch` in `multiply.py:28`) | existing | Correctness oracle; CPU fallback; reference for `register_fake` shape contract | Already exists, already tested. The Triton port must match it numerically. Doubles as the CPU path once `_butterfly.so` is gone — Triton has no production CPU backend. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `triton.autotune` (in-tree) | n/a | Sweep `BLOCK_SIZE`, `num_warps`, `num_stages` keyed on `(n, log_n, nblocks, nstacks, dtype)` | Use on each kernel during initial port. Be aware of pitfall: autotune key explosion is real for our 6-D twiddle (see PITFALLS.md). |
| `triton.heuristics` | n/a | Set `BLOCK_SIZE = next_power_of_2(...)` deterministically | Preferred over `autotune` for the small-`n` butterfly cases where the optimal config is mathematically derivable from `log_n`; cheaper than a full sweep. |
| `triton.testing.do_bench` | bundled with `triton` | Microbenchmark replacement for hand-rolled CUDA timing | Standard tool: median + p20/p80 with proper GPU sync and L2 cache flush. Use in `tests/benchmarks/`. |
| `triton.testing.assert_close` | bundled with `triton` | Numerical comparison against `butterfly_multiply_torch` reference | Handles fp16/bf16 tolerances; use in pytest assertions instead of hand-rolled `torch.allclose`. |
| `pytest` + `pytest.mark.parametrize` | existing | Parametrize over `(n, nblocks, nstacks, dtype, increasing_stride, device)` | Already in the dev deps. Standard pattern: parametrize, run both Triton and Torch-reference, `assert_close`. |
| `pytest.mark.skipif(not torch.cuda.is_available())` | stdlib pytest | Gate Triton tests | Triton kernels can't run on CPU; gate every Triton-touching test. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `TRITON_CACHE_DIR` env var | Persist JIT compilation artifacts across runs | First-call kernel compile is the user-visible cost we're trading the install-time compile for. Make sure CI doesn't wipe it between jobs. Default: `~/.triton/cache`. |
| `TRITON_PRINT_AUTOTUNING=1` | Surfaces the chosen autotune config and the time spent picking it | Run during phase development; do not enable by default in tests. |
| `TRITON_INTERPRET=1` | Runs `@triton.jit` kernels under a Python interpreter (no GPU compile) | Lets us debug kernel logic with `print` and a debugger. Slow; only for development. |
| `triton.testing.perf_report` | Plots/CSV from a sweep of input sizes | Use once per kernel to verify the Triton port is ≥ existing CUDA path before flipping the dispatch default. |

## Installation

No new install commands. The existing `uv pip install .` (or `-e .`) keeps working; once the C++/CUDA path
is fully removed, the `setup.py` shim and `csrc/` go with it (see Roadmap).

```bash
# Today (parallel paths)
uv pip install -e .                # still compiles csrc/ during migration
# Verify Triton is importable from torch's bundle:
python -c "import triton, torch; print(triton.__version__, torch.__version__)"

# End state (after csrc/ is deleted)
uv pip install torch-structured    # pure-Python wheel; Triton kernels JIT on first call
```

### pyproject.toml deltas

Minimum, end-state:

```toml
[project]
requires-python = ">=3.10"
dependencies = [
    "torch>=2.6",      # bump from >=2.0 — needed for torch.library.triton_op
    "numpy",
    "scipy",
    "einops",
    "opt_einsum",
]

# [build-system] becomes:
[build-system]
requires = ["setuptools>=64"]      # drop torch, ninja, wheel — no compile step
build-backend = "setuptools.build_meta"
```

Note we do **not** add `triton` to `dependencies`. `torch` already drags it in on the CUDA Linux wheel,
and adding it ourselves can fight pip's resolver against the `pytorch-triton` / `triton` package name
split (see PITFALLS.md). Instead, import-guard at runtime and raise a clear error if missing.

### During migration (parallel paths)

`requires-python` and `torch>=2.6` should bump *before* the first Triton kernel ships, because
`torch.library.triton_op` is the integration API. Keep `setup.py`, `ninja`, `wheel`, and the CUDA build
inputs until the last `csrc/` source is removed.

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `@triton_op` + `register_autograd` + `wrap_triton` | `torch.autograd.Function` wrapping raw `@triton.jit` calls | If you genuinely don't need `torch.compile` to trace through it. Forward only / pure eager. We *do* care about compile-friendliness for downstream users, so we use `triton_op`. |
| `@triton_op` + `register_autograd` + `wrap_triton` | `torch.library.custom_op` (opaque) | If the kernel is a black box that should never be fused. We want fusion when possible, so `triton_op` is better. |
| `triton.autotune` with a short key list | Hand-tuned per-shape configs (a `dict[(n, log_n) -> Config]`) | When autotune sweep is too slow at first call and shapes are known. For the butterfly kernel where `log_n` and `nblocks` fully determine the loop nest, **prefer `triton.heuristics`** over `autotune` — it's deterministic and free. |
| `triton.testing.do_bench` | `torch.utils.benchmark.Timer` | When measuring whole-Module wall time (not single-kernel). `do_bench` is the right tool for inside-the-kernel comparisons; `torch.utils.benchmark` for layer-level. Both have a place. |
| Drop `_flashmm` | Triton port of `_flashmm` | `_flashmm` uses cuBLASDx / MathDx tensor-core tuning we cannot reproduce in Triton without a multi-quarter project. PROJECT.md already lists this as out-of-scope for v1.2. |
| Drop CPU support for the C++ ops | Keep `csrc/cpu/butterfly_cpu.cpp` | Triton has no production CPU backend. The pure-PyTorch `butterfly_multiply_torch` already provides a CPU path; keep it as *the* CPU path. |
| One Triton kernel that templates over `nsteps` via Python `for _ in range(log_n)` inside `@triton.jit` | Port the 1..5-step C++ templates verbatim into 5 Triton kernels | Triton's `tl.static_range` + `constexpr` lets one kernel cover all step counts. The five-way template explosion in `csrc/cuda/butterfly_cuda.cu` was a CUDA-template workaround, not a fundamental algorithmic choice. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `torch::autograd::Function` in C++ (current `butterfly.cpp:99`) | The whole reason we're migrating: drags in a C++ build step. Also not the recommended autograd integration path post-2024 even if we kept C++. | `torch.library.register_autograd` (Python). |
| `TORCH_LIBRARY(torch_structured, m) { m.def(...); }` block | Same as above — requires compiled `.so`. | `@triton_op("torch_structured::butterfly_multiply", ...)` registers the op under the same namespace, no compile. |
| `torch.jit.script` on the Python wrappers (`multiply.py:8,15,21`) | TorchScript is deprecated as of PyTorch 2.10. `triton_op` already gives us a real op that `torch.compile` understands. | Plain Python wrappers; let `torch.compile` (Inductor) be the compile path. |
| `tl.make_block_ptr` for new code | Deprecated in Triton 3.7 (emits warning). Replaced by tensor-descriptor API. | Plain `tl.load(ptr + offsets, mask=...)` for our kernels — butterfly's 2×2 access pattern is small enough that block-pointers don't help. If we ever port a matmul-shaped kernel, use `tl.make_tensor_descriptor` (still tagged experimental in Triton 3.6/3.7 but stable in practice on Hopper+). |
| `triton-windows` as a hard dep | Our users are Linux-only per Docker history; Windows Triton requires a separate fork (`woct0rdho/triton-windows`) with its own bundled CUDA. | Document Linux-only; let users on Windows install `triton-windows` themselves. Don't pin it. |
| Triton CPU backend | Marked "under development" in the official README as of 2026-Q2. Not production-ready. | The existing pure-PyTorch `butterfly_multiply_torch` is the CPU path. |
| Wide `autotune` config sweep with `key=["n"]` only | `key` decides when to re-tune; missing `nblocks`/`dtype` means stale configs get reused for different shapes. | `key=["n", "nblocks", "nstacks", "input.dtype"]` (and consider `triton.heuristics` instead for fully-deterministic shapes). |

## Stack Patterns by Variant

**If we want to ship a non-Triton fallback path for old GPUs (sm_70/sm_75):**
- Keep the `csrc/butterfly.cpp` autograd Function alive behind a runtime flag
  (`TORCH_STRUCTURED_USE_TRITON=0`).
- Document sm_80+ as the supported-via-Triton tier; sm_70/sm_75 users need to build from source.
- This matches what PyTorch 2.x itself does: Inductor falls back to ATen kernels when Triton can't compile.

**If sm_70/sm_75 support gets dropped (project decision):**
- Triton 3.3+ already dropped Turing (sm_75). Volta (sm_70) was already shaky.
- Single Triton path. Delete `csrc/cuda/` outright. This is the "clean" end state and what PROJECT.md
  implies by "delete `csrc/` and `setup.py` shim".

**If a user runs CPU-only:**
- They get `butterfly_multiply_torch` (already the case today via the `if input.device.is_cuda` dispatch
  in `csrc/butterfly.cpp:44`). Nothing changes for them.

**If a user runs ROCm/AMD:**
- Triton 3.3+ supports ROCm 6.2+, so this *might* work for free, but **we don't test it and don't claim
  support in v1.2**. Mark as experimental; revisit if there's demand. (Today's CUDA path doesn't support
  ROCm either, so this is at worst neutral.)

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `torch==2.6` | `triton==3.2.x` | First version with `torch.library.triton_op` and `wrap_triton`. **Floor for the Triton path.** |
| `torch==2.7` | `triton==3.3.x` | Triton 3.3 drops Turing (sm_75). |
| `torch==2.9` | `triton==3.5.x` | Triton 3.5: warp specialization on AMD, ragged TMA, `tl.unsplat`. |
| `torch==2.10` | `triton==3.6.x` | Constexpr-propagation BC break in Triton 3.6; verify our kernels still compile. |
| `torch>=2.11` | `triton==3.7.x` | **`make_block_ptr` deprecation warning**. Don't use it in new kernels. |
| Triton 3.x on NVIDIA | **CC 8.0+ (Ampere) required** | Per Triton 3.x README. Volta/Turing users must keep the CUDA C++ path or pin `triton<3.3`. |
| Triton 3.x on AMD | **ROCm 6.2+** | Untested by us in v1.2. |
| Triton wheels | **CPython 3.10–3.14, Linux x86_64/ARM64 only** | No macOS, no official Windows. Matches our existing `requires-python = ">=3.10"`. |
| Patch-version pinning | All `3.5.x` interchangeable with all Torch `2.9.x` | Torch pins exact patch (e.g. 2.9.0 → triton 3.5.0) but minor-series compatibility holds. Don't pin Triton ourselves. |

## What the User Has to Give Up

Spelled out explicitly because PROJECT.md's quality gate asks for it:

1. **Volta (sm_70) and Turing (sm_75) support via the Triton path.** Triton 3.3+ requires CC 8.0+.
   The existing CUDA C++ path covers them today (we ship `7.0 8.0 9.0+PTX`). End state: dropped, or
   gated behind a `TORCH_STRUCTURED_USE_TRITON=0` build-from-source escape hatch.
2. **CPU compiled-extension path.** `csrc/cpu/butterfly_cpu.cpp` goes away. Pure-PyTorch
   `butterfly_multiply_torch` already exists and stays as the CPU answer — same correctness, lower
   performance than the C++ CPU code (which itself was never the hot path). Acceptable trade.
3. **macOS support for the kernel layer.** Already gone in practice (no `_butterfly.dylib` was being
   built); now formalized.
4. **Windows support without third-party Triton.** Users need `triton-windows`. We document this; we
   don't carry it.
5. **`_flashmm` MathDx tensor-core kernel.** Cannot be replicated in Triton at the same perf level
   without a major effort. PROJECT.md already declared this out-of-scope.
6. **First-call latency.** The install-time compile (~5 minutes) becomes a per-kernel JIT compile on
   first call (seconds per kernel, then cached in `~/.triton/cache`). Different cost profile; users
   running long training jobs won't notice, users running smoke tests will.
7. **Deterministic-build reproducibility for the kernel binaries.** The Triton compiler may emit
   slightly different PTX across Triton versions even for the same Python source. Numerical bit-exact
   reproducibility across versions isn't guaranteed (it isn't today either with NVCC, but the surface
   area is bigger now). Always-on numerical tolerance in tests.

## Integration Points (concrete)

Mapping from current code → new pattern (this is what the roadmap will hand to phase planning):

| Current (v1.1) | New (v1.2) |
|---|---|
| `csrc/butterfly.cpp` `ButterflyMultiply::forward/backward` (lines 99–125) | `@triton_op("torch_structured::butterfly_multiply", mutates_args={})` in `torch_structured/butterfly/triton_kernels/butterfly.py` |
| `TORCH_LIBRARY(torch_structured, m) { m.def("butterfly_multiply", ...); }` (line 127) | Op name registers automatically via the `triton_op` decorator. `torch.ops.torch_structured.butterfly_multiply` keeps working — no caller-side changes. |
| `torch.ops.load_library(...)` for `_butterfly.so` / `_version.so` in `torch_structured/butterfly/__init__.py:38-39` | Removed. Replaced by `from . import triton_kernels  # registers ops at import` |
| `csrc/cuda/butterfly_cuda.cu` (647 lines, templated on `nsteps∈{1..5}` × `increasing_stride`) | One `@triton.jit` kernel parametrized on `LOG_N: tl.constexpr`, `INCREASING_STRIDE: tl.constexpr`. Loop unrolled by `tl.static_range`. |
| `csrc/diag_mult/`, `csrc/hadamard/` pybind extensions | Plain `@triton.jit` kernels invoked from `torch_structured/structured/triton_kernels/`. No autograd wrapping needed for `hadamard` (orthogonal/involutory). `diag_mult`: backward is just elementwise mul, register the same way. |
| `setup.py` (170 lines) | Deleted. `pyproject.toml` is enough. |
| `MANIFEST.in` | Deleted (only existed to package `csrc/`). |
| Dispatch flag for parallel-paths during migration | New env var `TORCH_STRUCTURED_BACKEND ∈ {auto, triton, cuda}`, default `auto` (Triton if importable, else CUDA, else pure-PyTorch reference). Implementation: small `_dispatch.py` module that the public `butterfly_multiply` calls into. |

## Sources

- [Triton GitHub README](https://github.com/triton-lang/triton/blob/main/README.md) — supported platforms (Linux only, sm_80+, ROCm 6.2+, CPython 3.10–3.14). HIGH confidence.
- [Triton on PyPI](https://pypi.org/project/triton/) — current version 3.7.0 (May 2026), Linux x86_64/aarch64 wheels only. HIGH confidence.
- [PyTorch tutorial: User-Defined Triton Kernels with torch.compile](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html) — `triton_op`, `wrap_triton`, `register_autograd` usage; verified via WebFetch. HIGH confidence.
- [PyTorch dev-discuss: Custom Ops Under torch.compile — autograd.Function vs torch.library.custom_op](https://dev-discuss.pytorch.org/t/custom-ops-under-torch-compile-autograd-function-vs-torch-library-custom-op/3338) — official recommendation to migrate off `autograd.Function`. HIGH confidence.
- [pytorch/pytorch#154025 — Block ptrs are being removed from Triton](https://github.com/pytorch/pytorch/issues/154025) — `make_block_ptr` deprecation tracker. MEDIUM confidence (tracking issue, status changes).
- [Triton releases page](https://github.com/triton-lang/triton/releases) — 3.5/3.6/3.7 feature matrix. HIGH confidence.
- [PyTorch 2.10 release blog](https://pytorch.org/blog/pytorch-2-10-release-blog/) — TorchScript deprecation, Triton-3.6 pairing. HIGH confidence.
- [triton.testing.do_bench docs](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html) — benchmarking API. HIGH confidence.
- [Red Hat Emerging Tech — Understanding Triton Cache](https://next.redhat.com/2025/05/16/understanding-triton-cache-optimizing-gpu-kernel-compilation/) — `TRITON_HOME` / `TRITON_CACHE_DIR` semantics. MEDIUM confidence.
- [triton-lang/triton#4020 — autotuner deja-vu RFC](https://github.com/triton-lang/triton/issues/4020) — autotune persistence considerations. MEDIUM confidence.
- [Compatibility table — vici0549 (Medium)](https://medium.com/@vici0549/the-definitive-guide-to-pytorch-cuda-and-flash-attention-compatibility-ebec1161ec10) — Torch↔Triton patch pinning. MEDIUM confidence (community-maintained, but matches the release notes).

---
*Stack research for: torch_structured v1.2 Triton migration*
*Researched: 2026-05-26*
