# Project Research Summary — v1.2 Triton Migration

**Project:** torch_structured v1.2 — Triton Migration
**Domain:** PyTorch C++/CUDA → Triton kernel port (structured-matrix primitives with autograd)
**Researched:** 2026-05-26
**Confidence:** HIGH on API choice and dispatch architecture; MEDIUM on per-kernel performance ceilings

## Executive Summary

v1.2 ports torch_structured's three native kernels (`butterfly_multiply`, `hadamard`, `diag_mult`) from hand-tuned C++/CUDA to Triton, eliminating the install-time compile step that v1.0/v1.1 did not solve. The end state is a wheel-free, source-only PyPI package that JIT-compiles GPU kernels on first call. All four research streams converged on the same API choice: `@torch.library.triton_op` + `register_autograd` + `wrap_triton` is the only viable wrapper pattern — it's the only one that (a) plays correctly with `torch.compile` and dynamo fake-tensor tracing, (b) replaces the existing C++ `torch::autograd::Function` cleanly, and (c) lets each Triton kernel become a fusable, FX-traceable op. This API requires **PyTorch >=2.6** (we currently target >=2.0 — the floor must bump in Phase 1 before any kernel lands).

The migration is brownfield: v1.0 (build modernization) and v1.1 (cleanup) already shipped. Phase numbering 1–N within v1.2; the roadmapper will normalize to absolute numbering. All four researchers proposed identical ordering: infrastructure & dispatch → `diag_mult` (warm-up) → `hadamard` (no atomics, no complex) → butterfly forward → butterfly backward → integration hardening → delete `csrc/`. Seven work phases plus a deletion phase, with the deletion gated by a two-release deprecation cadence.

Two correctness risks dominate and must be designed for, not discovered: **complex64 has no native Triton support** (must use real/imag-split arithmetic — this is mandatory for FFT/DCT/circulant/Toeplitz/ButterflyUnitary/LRU to keep working), and **`tl.atomic_add` on bf16/fp16 is unsupported or broken** (must accumulate into an fp32 scratch tensor regardless of input dtype). Both are decided in Phase 1, implemented in Phases 4–5. The performance budget is **60–90% of the existing hand-tuned CUDA kernel** — accept that the 5-stage multi-stage tile may not be matchable in Triton and defer or drop it rather than chasing parity. The pure-PyTorch `butterfly_multiply_torch` is preserved in both roles: correctness oracle for every test and runtime fallback for CPU / Windows / macOS / no-Triton environments.

## Key Findings

### Recommended Stack

The migration is additive to the v1.1 stack — nothing in `pyproject.toml` *removes* deps; the integration is via PyTorch's built-in `torch.library` API and the `triton` package that PyTorch already drags in on CUDA Linux wheels. The bigger story is what *goes away* on the cleanup side: `setup.py`, `MANIFEST.in`, `ninja`, `wheel`, `TORCH_CUDA_ARCH_LIST`, `FORCE_CUDA`, `FORCE_CPU`, the CUDA version compatibility check, and the entire `csrc/` tree — replaced by Triton's JIT cache at `~/.triton/cache`.

**Core technologies:**
- **`torch.library.triton_op`** (PyTorch >=2.6): Decorator that registers `torch_structured::butterfly_multiply` etc. as a real op with FakeTensor/Meta and Inductor-traceable kernel calls. **Hard floor: must bump `torch>=2.0` to `torch>=2.6` in Phase 1.**
- **`torch.library.wrap_triton`** (PyTorch >=2.6): Required marker inside the `triton_op` body around every `@triton.jit` kernel invocation; without it Inductor cannot lift the kernel into the compiled graph.
- **`torch.library.register_autograd`**: Pure-Python replacement for `torch::autograd::Function<ButterflyMultiply>` in `csrc/butterfly.cpp:99-125`. Backward function calls another `triton_op`-wrapped kernel whose body uses `wrap_triton(butterfly_bw_kernel)`.
- **`triton` 3.x (bundled with torch on CUDA Linux)**: Authors the kernels. Requires **CC 8.0+ (Ampere+)** since Triton 3.3 — Volta sm_70 and Turing sm_75 are dropped via the Triton path.
- **`butterfly_multiply_torch`** (pure-PyTorch, exists in `multiply.py:28`): Dual role — correctness oracle in tests AND runtime fallback for CPU / Windows / macOS / no-Triton.
- **Tooling**: `triton.testing.do_bench` and `triton.testing.assert_close` (not hand-rolled); `triton.heuristics` preferred over `triton.autotune` where shapes are deterministic from `log_n`; `TRITON_CACHE_DIR` persisted in CI.

See [STACK.md](./STACK.md) for full version compatibility matrix.

### Expected Features

**Must have (table stakes):**
- Triton port of `butterfly_multiply_fw` and `butterfly_multiply_bw` (heart of the library)
- Triton port of `hadamard` and `diag_mult` (anything `.cu` remaining blocks the wheel-free promise)
- Runtime backend selector (`TORCH_STRUCTURED_BACKEND={triton,cuda,torch,auto}` env var)
- Correctness gate vs `butterfly_multiply_torch` for every Triton kernel
- gradcheck vs `autograd.grad` of `_torch_fw` (NOT vs the CUDA reference)
- fp32 + complex64 dtype support (complex64 is mandatory — FFT/DCT/circulant/Toeplitz/ButterflyUnitary/LRU all depend on it)
- CPU fallback preserved via `butterfly_multiply_torch`
- Twiddle layout `(nstacks, nblocks, log_n, n/2, 2, 2)` unchanged (breaking layout breaks every saved checkpoint)
- `csrc/` + `setup.py` + `MANIFEST.in` deleted at end of milestone (gated by deprecation cadence)

**Should have (differentiators):**
- bf16/fp16 support (existing CUDA path is fp32+complex64 only)
- `triton.autotune` over `BLOCK_SIZE` / `num_warps` / `num_stages`
- ROCm support (free with Triton; advertise as "untested")
- `torch.compile` integration (free with `triton_op` — resolves the dynamo fake-tensor bug from `260419-p27`)

**Defer (v1.3+):**
- bf16/fp16 dtype matrix (ship fp32+complex first)
- Fused diag×butterfly or perm×butterfly kernels
- fp64

**Anti-features (do NOT promise):**
- Native `tl.complex64` (Triton has none — must use real/imag split)
- `_flashmm` port (MathDx tensor-core tuning isn't replicable in Triton)
- Outperforming the existing CUDA kernel (target is **60–90% of CUDA**, not parity)
- Removing `butterfly_multiply_torch` (oracle AND CPU fallback — never delete)
- Pre-compiled wheels

See [FEATURES.md](./FEATURES.md) for full table.

### Architecture Approach

A new `_triton/` peer package holds all kernels; a new `_torch_ref/` peer package holds pure-PyTorch references (moved, not rewritten); and a new `_ops.py` single dispatch point binds public names to one backend at import time. Every nn.Module consumer changes one import line. Public nn.Module API surface stays byte-identical so saved checkpoints from v1.0/v1.1 still load.

**Major components:**
1. **`torch_structured/_ops.py`** — Single import-time dispatch. Reads `TORCH_STRUCTURED_BACKEND`, probes `HAS_TRITON`/`HAS_CUDA`, binds public names exactly once. No per-call branching.
2. **`torch_structured/_triton/<op>/{forward,backward,op}.py`** — Per-op trio: `@triton.jit` kernels + `@triton_op` wrapper with `register_autograd` and meta kernel.
3. **`torch_structured/_triton/_common/{dispatch,autotune}.py`** — Shared probes and helpers.
4. **`torch_structured/_torch_ref/{butterfly,hadamard,diag_mult}.py`** — Pure-PyTorch references. Always shipped.
5. **Existing nn.Module surface** (`butterfly/`, `structured/`, `monarch/`, `recurrent/`) — unchanged except single `import` line per consumer.
6. **`pyproject.toml`** — switches to no-compile build backend.

Resolves an existing dynamo fake-tensor bug from `260419-p27-SUMMARY.md` for free (the C++ op currently raises inside dynamo's fake-tensor tracing; `triton_op` exposes the impl to dynamo).

See [ARCHITECTURE.md](./ARCHITECTURE.md).

### Critical Pitfalls

All four research streams flagged the same top three risks.

1. **Triton has no native complex64.** ButterflyUnitary's `U U^* = I` check silently fails because the conjugate path degenerates to a transpose. Decision in Phase 1: split tensors (recommended — carry `real` and `imag` as two fp32 tensors) or packed last-dim. Implement complex multiply as explicit 4-FMA helper. **Mandatory work, not optional.**
2. **`tl.atomic_add` on bf16/fp16 destination is unsupported or non-deterministic.** Naive backward gives gradients 10–50% off the reference at batch_size >=1024. Always accumulate into an fp32 scratch tensor, then cast to user dtype. Pre-allocate scratch from Python; do `tl.sum` block-reduce first.
3. **Using `torch.autograd.Function` instead of `triton_op` is the textbook regression.** Works in eager, breaks `torch.compile` opaquely. PyTorch dev-discuss explicitly recommends `triton_op` + `register_autograd` post-2024. Lock the pattern in at Phase 2.
4. **Comparing Triton to the CUDA reference is the wrong oracle.** Two correct kernels can differ by 1e-5 from FMA ordering. `butterfly_multiply_torch` is the only acceptable correctness gate. For gradcheck: `autograd.grad(butterfly_multiply_torch(...), inputs)` — NOT a hand-derived backward.
5. **Removing `csrc/` before Triton reaches parity.** Minimum two-release cadence: v1.2 ships both paths with Triton default and `DeprecationWarning` on CUDA; v1.3 default-disables CUDA build; v1.4 actually deletes `csrc/`.

See [PITFALLS.md](./PITFALLS.md).

## Implications for Roadmap

Phase numbering 1–N within v1.2; roadmapper will normalize to absolute numbering. All four researchers proposed identical ordering.

### Phase 1 — Triton Dispatch Infrastructure & Foundational Decisions
**Rationale:** Every subsequent phase depends on the wrapper pattern, the backend selector, and the complex layout being decided up front. No-kernels phase. Also bumps `torch>=2.6` and writes the deprecation plan.
**Delivers:** `_torch_ref/` package; `_ops.py` with `TORCH_STRUCTURED_BACKEND`; `_triton/_common/dispatch.py`; `triton_op` + `register_autograd` + `register_fake` skeleton; CI `TRITON_CACHE_DIR` persistence; tests `conftest.py` with `backend` fixture parametrized over `["triton", "torch", "cuda"]`; complex64 layout decision documented; deprecation plan written.

### Phase 2 — diag_mult Triton Port
**Rationale:** Smallest kernel (~10 lines today). Validates `triton_op` + autograd plumbing end-to-end with minimal kernel risk.
**Delivers:** `_triton/diag_mult/{kernel.py, op.py}`; `structured/krylov.py` switches to `_ops.diag_mult`; first end-to-end dispatch+autograd test.

### Phase 3 — hadamard Triton Port
**Rationale:** Self-inverse (backward = forward), so two-kernel pattern collapses to one. No atomics, no complex. Reference pattern exists (arthurfeeney/fwht).
**Delivers:** `_triton/hadamard/{forward.py, op.py}`; `structured/hadamard.py` switches to `_ops.hadamard_transform`.

### Phase 4 — butterfly Forward (Triton) with Torch-Reference Backward
**Rationale:** Largest forward kernel. Forward-only is a stable intermediate state — keep CUDA or `_torch_ref` backward via `register_autograd`. Complex64 via real/imag split lands here. Start with 3-stage tile; defer 5-stage.
**Delivers:** `_triton/butterfly/{forward.py, op.py}`; forward correctness at all dtypes including complex64; n=1, n=2, output_size!=n edge cases covered.
**Perf budget:** 60–90% of existing CUDA on log_n ∈ [8, 11].

### Phase 5 — butterfly Backward (Triton) + register_autograd Replacement
**Rationale:** Highest-risk kernel. atomicAdd into `d_twiddle` is *the* Triton footgun. Mandatory fp32 scratch + block-level `tl.sum` then single atomicAdd per block. 3-layer gradcheck pattern is a phase entry gate.
**Delivers:** `_triton/butterfly/backward.py` with fp32 scratch reduction; complex backward via real/imag split; determinism warning in Python wrapper; full pytest pass on Triton path; benchmark sweep vs CUDA published.

### Phase 6 — Integration Hardening (`torch.compile`, DDP, checkpoint, FSDP)
**Rationale:** All kernels exist; verify cross-cutting integrations. Validates that `triton_op` choice paid off — `torch.compile` should trace through cleanly.
**Delivers:** `torch.compile(model)` smoke test; gradient checkpointing perf check; 2-GPU FSDP smoke test with twiddle no-shard hint; deterministic-mode tests; perf grid published (log_n × batch × dtype); ROCm "untested" disclaimer added to README.

### Phase 7 — Deprecate (and Eventually Delete) csrc/, setup.py, MANIFEST.in
**Rationale:** Actual milestone goal: wheel-free, source-only install. **Gated by two-release deprecation cadence** — v1.2 ships both paths with Triton default and `DeprecationWarning` on CUDA; v1.3 default-disables CUDA build; v1.4 (later milestone) deletes the source. **Phase 7 of v1.2 may only land the deprecation portion;** actual deletion may be a v1.3 phase. Roadmapper should split if needed.

### Phase Ordering Rationale

- **Infrastructure precedes any kernel** because every kernel inherits the wrapper pattern, backend selector, and complex64 layout decision.
- **Smallest kernel before largest** (diag_mult → hadamard → butterfly) builds muscle memory on cheap kernels first; if the dispatch pattern is wrong, it surfaces cheaply.
- **Forward before backward**: Triton forward with torch-reference backward is a valid intermediate. The reverse isn't.
- **Integration hardening after all kernels exist**: cross-cutting tests only make sense once every consumer is on the new path.
- **Deletion last and gated by cadence**: the deprecation cadence is the single hardest discipline — none of the prior phases are reversible if csrc/ is gone too early.

### Research Flags

**Phases likely needing focused research during planning:**
- **Phase 4 (butterfly forward):** Exact CUDA algorithm, 5-stage vs 3-stage tile decision, complex64 split layout.
- **Phase 5 (butterfly backward):** 3-layer gradcheck pattern, fp32 scratch reduction layout, deterministic-backward opt-in. Single highest correctness risk.
- **Phase 6 (integration hardening):** FSDP and gradient checkpointing interactions documented at MEDIUM confidence only.

**Phases with well-documented patterns (likely skip focused research):**
- Phase 1: canonical PyTorch tutorial. Phase 2 (`diag_mult`): trivial elementwise. Phase 3 (`hadamard`): reference at arthurfeeney/fwht. Phase 7 (deletion): careful execution.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | API choice verified against official PyTorch tutorial, dev-discuss recommendation, four production codebases. PyTorch >=2.6 floor is hard-verified. |
| Features | MEDIUM-HIGH | Table-stakes set unambiguous (mirrors existing CUDA path). Differentiator costs estimated from analogous projects. |
| Architecture | HIGH | `_ops.py` + `_triton/<op>/` + `_torch_ref/` design straight from Liger-Kernel and fla-org. nn.Module API surface stability is constraint-driven. |
| Pitfalls | HIGH | Top 3 verified against open Triton issues and PyTorch official docs. Existing test tolerances already evidence atomicAdd precision pattern. |

**Overall confidence:** HIGH on API and architecture; MEDIUM on absolute performance numbers.

### Gaps to Address

- **Exact perf numbers vs the existing CUDA kernel unknown until Phase 4–5 benchmarks land.** 60–90% is target; if Phase 4 surfaces >40% regression on a hot shape, runtime selector lets CUDA path stay live for that shape.
- **ROCm validation deferred.** v1.2 documents as experimental; real hardware pass is a v1.3 candidate.
- **Volta sm_70 and Turing sm_75 users dropped from the Triton path.** Decision in Phase 1: escape-hatch env var, or pin to v1.1? Recommend the latter (cleaner end state).
- **First-call JIT cost in CI is a known UX regression.** Phase 1 persists `TRITON_CACHE_DIR`; Phase 6 publishes wall-time numbers.
- **The 5-stage multi-stage tile may be unreachable in Triton.** Phase 4 ships 3-stage; perf-critical shapes may route to CUDA via backend selector indefinitely.
- **Determinism story.** PyTorch's `use_deterministic_algorithms(True)` is invisible to custom Triton ops; Phase 5 wires manual guards; deterministic backward is an opt-in that may need its own kernel.

## Sources

### Primary (HIGH confidence)
- Triton GitHub README — platform support, sm_80+, ROCm 6.2+, CPython 3.10–3.14
- PyTorch tutorial — User-defined Triton kernels with torch.compile
- PyTorch torch.library 2.9 docs — `triton_op`, `register_autograd`, `register_kernel("meta")`
- PyTorch dev-discuss — Custom Ops Under torch.compile
- Triton issue #1687 — Complex number support (still open)
- Triton issue #2834 — bf16 atomic_add unsupported
- Triton issue #891 — fp16 atomic_add non-deterministic segfaults
- PyTorch 2.6 Release Blog — `triton_op` introduced as stable
- Liger-Kernel — reference for per-op file layout
- arthurfeeney/fwht — Triton FWHT reference
- Project's `csrc/cuda/butterfly_cuda.cu:40-58` — hand-tuned constants evidencing 5-stage gap
- Project's `tests/test_multiply.py:16-17, 56-57` — tolerance scaling evidencing atomicAdd precision sensitivity

### Secondary (MEDIUM confidence)
- Triton autotune docs — `cache_results=True`, config key semantics
- Triton issue #5283 — FMA tolerance differences vs cuBLAS
- PyTorch issue #125489 — Compiled autograd + user-defined Triton kernel
- PyTorch issue #162687 — Default arguments break compile on custom ops
- Red Hat — Triton-vs-CUDA benchmark (basis for 60–90% perf target)
- Mamba selective_scan in Triton — production log-N op with bf16
- FlashFFTConv — reference dtype envelope for FFT-like kernels

### Internal
- `.planning/quick/260419-p27-extend-recurrent-poc-torch-compile-track/` — dynamo fake-tensor bug documented
- `.planning/research/STACK.md`, `FEATURES.md`, `ARCHITECTURE.md`, `PITFALLS.md`
