# Feature Landscape

**Domain:** Triton ports of structured-matrix GPU kernels (butterfly / Hadamard / diagonal)
**Project:** torch_structured v1.2 — Triton migration
**Researched:** 2026-05-26
**Confidence (overall):** MEDIUM — patterns from analogous projects (FlashFFTConv, fast-hadamard-transform, Mamba SSD, arthurfeeney/fwht, Flash-Attention Triton) are well-documented; no public Triton port of the *specific* recursive-butterfly kernel exists to copy from.

---

## How to read this file

The downstream consumer is **requirements scoping**. Every row is sized so a reader can decide "is this in scope?" by reading the row alone.

Three categories:

1. **Table stakes** — must ship to call the Triton path "done"
2. **Differentiators** — features the Triton port can plausibly add that the CUDA path does not have, and that are cheap enough to justify shipping
3. **Anti-features** — explicitly out of scope; do not promise these

A feature is table stakes if dropping it breaks the migration's stated value ("a wheel-free install that just works"). Differentiators are upside; anti-features are landmines flagged before someone wastes a phase on them.

---

## Table Stakes

Features required to ship the v1.2 Triton path. Missing any of these and the milestone fails its own success criteria (parallel paths, correctness oracle reachable, install-from-source-only).

| # | Feature | Why required | Complexity | Depends on |
|---|---------|--------------|------------|------------|
| 1 | **Triton port of `butterfly_multiply_fw`** (forward) | The single largest kernel in `csrc/cuda/butterfly_cuda.cu` (647 lines, templated over `nsteps ∈ {1..5}` and `increasing_stride`). Everything in `torch_structured/butterfly/butterfly.py`, `factory.py`, and the `special.py` transforms (FFT, DCT, DST, Hadamard, circulant, Toeplitz) routes through this op. Without it, no module in the library works on GPU. | **High** | `butterfly_multiply_torch` reference oracle in `multiply.py` |
| 2 | **Triton port of `butterfly_multiply_bw`** (backward) | The autograd path. The existing C++ `torch::autograd::Function` in `csrc/butterfly.cpp` lines 99–125 is replaced by a Python `torch.autograd.Function` whose `forward` and `backward` call the Triton kernels. Without backward, the library is inference-only — which contradicts its core use case (learnable twiddles, `_is_structured` weight-decay flag, training research code). | **High** | Feature 1; same correctness oracle |
| 3 | **Triton port of `hadamard`** | Standalone fast-Walsh-Hadamard kernel (`csrc/hadamard/hadamard_cuda_kernel.cu`, 153 lines, two-pass mixed-radix-2/4 with shared-memory tile). Used by `torch_structured.structured` and the `hadamard()` special transform. arthurfeeney/fwht demonstrates this is well within Triton's capability. | **Medium** | none (independent kernel) |
| 4 | **Triton port of `diag_mult`** | Trivial elementwise (`csrc/diag_mult/diag_mult_cuda_kernel.cu`, 17 lines). Should be a 10-line Triton kernel; included because the migration goal is "no CUDA toolchain needed," so any remaining `.cu` file blocks that goal. | **Low** | none |
| 5 | **Runtime backend selector (Python-level dispatch)** | The migration strategy is **parallel paths**: Triton and CUDA coexist, a flag picks one. Without this, you cannot land Triton incrementally (per-kernel) or roll back if a Triton port regresses. **Recommendation: environment variable `TORCH_STRUCTURED_BACKEND={triton,cuda,torch,auto}` with `auto` as default.** Env vars are the established convention in this space — see `FLASH_ATTENTION_TRITON_AMD_AUTOTUNE`, `FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON`, `TRITON_CACHE_DIR`. Also expose as a Python API (`torch_structured.set_backend(...)`) for tests. | **Low–Medium** | none (cross-cutting) |
| 6 | **Correctness gate against `butterfly_multiply_torch`** | The pure-PyTorch reference in `multiply.py` is the existing oracle and `PROJECT.md` explicitly preserves it as such. Every Triton kernel must pass `torch.allclose(triton_out, torch_out, atol, rtol)` with dtype-keyed tolerances (fp32: `atol=1e-5, rtol=1e-4`; fp16/bf16: `atol=1e-2, rtol=1e-2`; complex64: `atol=1e-4, rtol=1e-3`). Pattern matches the Flash-Attention Triton tutorial (`triton/python/tutorials/06-fused-attention.py`) and is standard practice. | **Low** (test scaffolding) | Features 1, 2, 3, 4 |
| 7 | **Gradient check vs pure-PyTorch reference** | Backward correctness cannot be verified by forward `allclose` alone. Existing test files (`tests/test_butterfly.py`) already do reference comparisons; add a path that exercises Triton specifically. Use `torch.autograd.gradcheck` only with fp64 (gradcheck assumes fp64 numerics); for fp32/fp16/bf16 use perturbation tests or compare backward outputs to the pure-PyTorch reference's autograd. | **Low–Medium** | Feature 2, Feature 6 |
| 8 | **fp32 + complex64 dtype support** | These are the dtypes the existing CUDA kernel supports (per the dispatch macro at `butterfly_cuda.cu:9–23`: `Float` and `ComplexFloat`). The Butterfly module's complex code path (`butterfly.py:42, 86–90, 121–122, 227, 237`) and the special transforms (FFT/IFFT init in `butterfly.py:96–109`) require complex64. **Dropping complex64 silently breaks `fft()`, `ifft()`, `dct()`, `dst()`, `circulant()`, `toeplitz()`, `ButterflyUnitary`, and the LRU recurrent layer's complex eigenvalue path.** Complex must be implemented via real/imag-split arithmetic since Triton has no first-class complex type (see Pitfalls). | **High** for complex64 (real/imag manual decomposition); **Medium** for fp32 alone | Features 1, 2 |
| 9 | **CPU fallback preserved (no Triton on CPU)** | Triton is GPU-only. The existing `cpu/butterfly_cpu.cpp` path is invoked when `input.device().is_cuda() == false`. Two options: (a) keep `cpu/*.cpp` compiled when no GPU and ship `butterfly_multiply_torch` as the CPU implementation; (b) drop the cpp_extension for CPU entirely and have CPU dispatch through `butterfly_multiply_torch`. **Recommendation: option (b)** — it is the only path consistent with "zero compilation step." The torch reference is slower but on CPU nobody cares about peak perf. | **Low** | Feature 5 |
| 10 | **Twiddle layout preserved** `(nstacks, nblocks, log_n, n/2, 2, 2)` | The Python module layer (`butterfly.py`, `combine.py`, `permutation.py`, `factory.py`, `monarch/`, `recurrent/`) and the saved checkpoints assume this exact layout. The Triton kernel must consume it directly. Changing layout = breaking change = scope creep. | **Low** (constraint, not work) | Features 1, 2 |
| 11 | **Support for `increasing_stride` flag and `nblocks > 1`** | These are first-class arguments. The CUDA kernel templates on `increasing_stride`; in Triton this becomes either a `tl.constexpr` (compile-time specialization, one kernel per value) or a runtime branch (one kernel handling both). `nblocks > 1` is a Python-side outer loop in the reference and can stay a Python-side loop calling the kernel `nblocks` times. | **Medium** | Features 1, 2 |
| 12 | **`output_size` argument supported (output trimming)** | The signature is `butterfly_multiply(twiddle, input, increasing_stride, output_size)`. Output truncation lets `Butterfly(in_size, out_size)` produce non-power-of-2 outputs. Trim post-kernel in Python is fine. | **Low** | Feature 1 |
| 13 | **First-call JIT compile cost is acceptable and cacheable** | Triton JIT-compiles kernels on first invocation per `(shape, dtype, num_warps, num_stages)` signature. With autotune this can mean a multi-second pause on first import-then-call. Mitigate by setting `TRITON_CACHE_DIR` and documenting it. Without this, the "wheel-free" pitch feels like a different kind of slow startup. | **Low** (mostly documentation) | Feature 5 |
| 14 | **Documented dispatch behaviour for `_flashmm`** | `_flashmm` is the MathDx tensor-core kernel and is in **Out of Scope** per `PROJECT.md`. The dispatcher must know not to route to it. Either keep the CUDA `_flashmm` path opt-in via a separate flag, or remove it entirely (the cleaner option for v1.2). | **Low** (decision, not code) | Feature 5 |
| 15 | **Build files removed** (`csrc/`, `setup.py` shim, `MANIFEST.in`) once parity reached | The explicit endpoint of the milestone per `PROJECT.md`: "Deprecate and remove `csrc/`, `setup.py` build shim, and MANIFEST.in." This is what makes the install wheel-free; without this step, the "no compilation" promise is unkept. | **Low** (deletion + pyproject cleanup) | Features 1–4 complete & validated |

---

## Differentiators

Capabilities the Triton port can plausibly add. These are upside, not commitments. Include in the milestone only if a port is already underway and the marginal cost is low.

| # | Feature | Value | Complexity | Honest take |
|---|---------|-------|------------|-------------|
| D1 | **bf16 / fp16 support** | The existing CUDA kernel is fp32 + complex64 only (the dispatch macro in `butterfly_cuda.cu:18–19` explicitly lists only `Float` and `ComplexFloat`; complex32 / half / bfloat16 cases are not generated, with the comment "Only support float (not double) for now to speed up compilation time"). Adding bf16 is the single biggest functional win Triton brings — it matches what `fast-hadamard-transform` ships (fp32 / fp16 / bf16) and what FlashFFTConv requires (fp16 / bf16 only). | **Medium** — Triton's bf16 has known quirks (`tl.dot` and `tl.atomic_add` did not historically accept bf16; intermediate accumulation should stay in fp32). Butterfly does not use `tl.dot` so the worst footgun is avoided; the backward's atomicAdd on `d_twiddle` is the real concern (see Pitfalls). | High value, real risk. Recommend: ship fp32 first, add bf16/fp16 in a follow-up phase once the fp32 port is stable. |
| D2 | **Autotune surface (`triton.autotune` over `BLOCK_SIZE`, `num_warps`, `num_stages`)** | Lets the same kernel cover the wide `log_n ∈ {1..20+}` range the library targets without hand-tuning the lookup tables that the existing CUDA kernel uses (`ITEMS_PER_THREAD_FORWARD[14]`, `MIN_BLOCKS_PER_MP_FORWARD[14]`, etc., at `butterfly_cuda.cu:42–58`). Removes ~20 magic numbers. | **Medium** — define candidate configs, pick `key=['log_n', 'nstacks']` so autotune re-runs when shapes change. Cache to `TRITON_CACHE_DIR`. Standard pattern from Flash-Attention Triton. | Worth doing. The hand-tuned tables are a maintenance liability; autotune trades one-time per-shape warmup cost for code clarity. |
| D3 | **Compile-time specialization on `nsteps` and `increasing_stride` via `tl.constexpr`** | Same idea as the existing C++ template dispatch (`Dispatch<n_max, Function>::call`, `butterfly_cuda.cu:87–96`) but expressed natively. Multiple kernel binaries get compiled by Triton, one per specialization. | **Low–Medium** — `tl.constexpr` is the idiomatic mechanism. | Should be the default approach; explicitly *not* a "differentiator," but worth calling out so reviewers don't expect a single mega-kernel. |
| D4 | **Single-kernel fused diagonal × butterfly** | The Python `combine.py` already fuses a diagonal into the first/last twiddle factor of the butterfly (in-parameter-space, not in-kernel). A Triton kernel could fuse `diag_mult ∘ butterfly` without modifying parameters. | **Medium** | Niche. The parameter-space fusion already exists and works. Skip unless a downstream model needs the runtime fusion. |
| D5 | **Single-kernel fused permutation + butterfly** | `FixedPermutation` is a separate `nn.Module` (`permutation.py`) with its own kernel launch. Fusing it into the first twiddle stage would cut launch overhead and remove a memory round-trip. | **High** | Defer. Permutations are buffer-permutes, not on the perf-critical path of large models. |
| D6 | **AMD ROCm support** | Triton runs on ROCm. The existing CUDA path does not. This is a free side-benefit of porting to Triton — no AMD-specific work needed beyond having a ROCm-installed Triton. | **Low** (mostly testing) | Worth advertising in the README. Do not gate the milestone on access to AMD hardware; ship as "should work on ROCm, untested." |
| D7 | **fp64 support** | Pure-Python reference handles double; existing C++ has a `<double>` specialization in `maxstep` but the dispatch macro filters it out. Triton supports fp64. | **Low** | Cheap to add but unclear demand. Add only if the autotune-config explosion is bounded. |
| D8 | **Backward in pure Python (no `torch::autograd::Function`)** | Replacing `ButterflyMultiply : torch::autograd::Function` (in `butterfly.cpp:99–125`) with a Python `torch.autograd.Function` makes the autograd path debuggable, profileable, and compatible with `torch.compile`'s `triton_op` integration (which traces *into* Triton kernels). | **Low** | This is implicit in Feature 2 but worth calling out: removing the C++ autograd glue is part of the win, not an extra. |
| D9 | **`torch.compile` integration via `triton_op`** | PyTorch 2.x exposes `torch.library.triton_op` which lets `torch.compile` look *through* Triton kernels for fusion. Custom C++ ops are opaque to the compiler. | **Medium** | Future-facing. Document as supported; do not promise full TorchInductor parity. |

---

## Anti-Features

Explicitly **out of scope** for v1.2. Listed here so they do not creep in during phase planning.

| # | Anti-feature | Why not | What to do instead |
|---|--------------|---------|---------------------|
| A1 | **`_flashmm` Triton port** | `_flashmm` is a MathDx tensor-core kernel (`csrc/flashmm/`). MathDx wraps cuFFTDx/cuBLASDx tensor-core building blocks that Triton has no equivalent for. Per `PROJECT.md`: "`_flashmm` MathDx kernel port — Triton cannot replicate MathDx tensor-core tuning; drop instead." | Delete `csrc/flashmm/` along with `csrc/` in the cleanup phase. Users who need that path can pin to v1.1. |
| A2 | **Pre-compiled wheels** | `PROJECT.md` explicitly out-of-scopes wheel distribution. The whole *point* of moving to Triton is that JIT compilation replaces wheels. | Source-only PyPI release. `pip install torch-structured` ships pure Python; Triton does its own JIT on first call. |
| A3 | **Native complex64 in Triton kernels** | Triton has no `complex64` dtype. Every project doing FFT-like work in Triton (FlashFFTConv, FNet variants, Mamba SSD with complex states) uses real/imag-split arithmetic: store complex as `(..., 2)` real tensor or as two separate real tensors, expand `(a+bi)(c+di) = (ac−bd) + (ad+bc)i` manually. | Implement complex path by splitting the trailing complex dim into two real components inside the kernel. `view_as_real` / `view_as_complex` at the Python boundary. **This is mandatory work for Feature 8, not optional.** |
| A4 | **fp8 / int8 / quantized paths** | The existing library has no quantized story. Adding one is a separate research project, not a build migration. | If demand surfaces post-v1.2, file a v1.3 milestone. HadaCore is the reference if it ever happens. |
| A5 | **Multi-GPU / NCCL kernels** | Library has no distributed story today. Out of scope. | Defer indefinitely. |
| A6 | **Outperforming the existing CUDA kernel** | The existing kernel is hand-tuned with per-`log_n` shared-memory budgets, register pressure tuning (`MAX5_FORWARD_BLOCK_SIZE`), shuffle-based reductions, and a multi-stage variant with explicit shared-memory tiling (`butterfly_cuda.cu:42–58`). Beating it in Triton is a research project. Public Triton-vs-handwritten-CUDA numbers show 78–82% of CUDA performance on H100/A100 for matmul/attention; for irregular log-N stride patterns the gap can be wider. | **Target: 60–90% of the existing CUDA perf in fp32.** Anything better is a bonus. Document this expectation up front so "Triton is 20% slower" doesn't read as a regression. |
| A7 | **Removing `butterfly_multiply_torch`** | It is the correctness oracle and CPU fallback. `PROJECT.md` preserves it. Tempting to delete after Triton lands; don't. | Keep, document role, add tests that explicitly compare Triton output to it. |
| A8 | **Changing twiddle layout to something Triton likes better** | Breaks every saved checkpoint and every Python module. | Adapt the kernel to the layout, not the other way around. If the layout is genuinely a perf problem, file a v1.3 task with a migration plan. |
| A9 | **In-place kernel variants** | The C++ path is functional. Triton ports should stay functional. In-place adds aliasing concerns the existing autograd glue does not handle. | Out of scope. |
| A10 | **Custom C++ dispatch via `TORCH_LIBRARY`** | The whole point of moving to Triton is dropping the C++ ABI surface. Re-introducing `TORCH_LIBRARY` registration for the Triton path defeats the migration. | Use a plain Python `torch.autograd.Function` (or `torch.library.triton_op` for compile-friendliness). |

---

## Feature Dependencies

```
F9 CPU fallback (torch reference)
   │
   └── F6 Correctness oracle ────────────┐
                                         │
F10 Twiddle layout (constraint)          │
   │                                     │
   ├── F1 Triton FW kernel ──── F11 increasing_stride / nblocks
   │     │                      │
   │     └── F12 output_size    │
   │                            │
   ├── F2 Triton BW kernel ───── F7 gradcheck
   │     │
   │     └── D8 pure-Python autograd Function (implicit)
   │
F3 Hadamard Triton (independent)
F4 diag_mult Triton (independent)

F5 backend selector (cross-cutting; gates rollback)
   │
   ├── F14 _flashmm dispatch decision
   └── F13 JIT cache documentation

F8 dtype matrix (fp32 + complex64 required; bf16/fp16 = D1)
   └── A3 (anti) complex64-via-real-split is the only way

F15 csrc/ cleanup (final step; gated on F1–F4 + correctness)
```

The critical path is **F9 → F1 → F2 → F6 → F7 → F15**. Hadamard (F3) and diag_mult (F4) are parallelizable side quests. F5 (dispatch) should land early so each kernel port is shippable independently.

---

## MVP Recommendation

Minimum shippable v1.2 Triton path:

1. **F5 backend selector** — first, so everything else can land incrementally.
2. **F4 diag_mult** — smallest kernel, tests the autograd/dispatch plumbing end-to-end.
3. **F3 hadamard** — second-smallest, independent of butterfly, matches `fast-hadamard-transform`'s known-working Triton design (per `arthurfeeney/fwht`).
4. **F1 butterfly forward fp32 real** — the heart of the milestone.
5. **F8a complex64 via real/imag split** for the forward.
6. **F2 butterfly backward fp32 real + complex64**.
7. **F6 + F7 correctness gates** (continuously, not at the end).
8. **F15 csrc/ deletion** — last, after all four kernels are at parity.

Defer to v1.3 or a follow-up phase:

- **D1 bf16 / fp16** — high value, but adds a dtype matrix that quadruples test surface. Land after fp32+complex are bedded down.
- **D6 ROCm advertising** — test once on borrowed hardware before promising it.
- **D2 autotune** — start with fixed `Config(BLOCK_SIZE=…, num_warps=4, num_stages=3)` per `log_n` bucket; turn on autotune once the kernel is correct.

Anything outside this list either is in Anti-Features or is genuine v1.3+ research.

---

## Risk-Weighted Complexity Summary

| Risk band | Features |
|-----------|----------|
| **Low risk, low effort** | F4 (diag_mult), F5 (backend selector), F9 (CPU fallback), F10/F12/F14 (constraints/decisions), F15 (deletion), A1–A10 (just don't do them) |
| **Medium risk, medium effort** | F3 (hadamard — proven pattern from arthurfeeney/fwht), F11 (constexpr specialization), F13 (JIT cache UX), D2 (autotune), D7 (fp64), D8/D9 (compile integration) |
| **High risk, high effort** | F1 (forward kernel — 647 lines of CUDA to replicate semantically), F2 (backward — atomicAdd on twiddles in fp16/bf16 is the textbook Triton footgun), F8 (complex64 via real-split), D1 (bf16/fp16 with all the Triton-bf16 quirks) |

The two genuinely hard features are **F1+F2 (the butterfly kernels themselves)** and **F8 (complex64 emulation)**. Everything else is either trivial, well-trodden, or a "just don't do it" decision.

---

## Sources

- [FlashFFTConv — HazyResearch/flash-fft-conv (C++/CUDA, fp16/bf16 only)](https://github.com/HazyResearch/flash-fft-conv) — confirms the dtype envelope that production "FFT-like" kernels target; confirms that FFT-style projects skip fp32.
- [fast-hadamard-transform — Dao-AILab](https://github.com/Dao-AILab/fast-hadamard-transform) — CUDA reference for Hadamard; supports fp32/fp16/bf16 up to dim=32768; the dtype baseline to match.
- [HadaCore: Tensor Core Accelerated Hadamard Transform Kernel (PyTorch blog)](https://pytorch.org/blog/hadacore/) — 1.1–1.4× to 3.5× over Dao-AILab CUDA; confirms tensor-core fusion is its own research direction (anti-feature A1).
- [arthurfeeney/fwht — Triton FWHT implementation](https://github.com/arthurfeeney/fwht) — open-source Triton port of the exact pattern we need; confirms block sizes must be power-of-2 and zero-padding can be done inside the kernel.
- [Mamba selective_scan / SSD in Triton — state-spaces/mamba](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_combined.py) — reference for a production Triton implementation of a state-space (log-N-style) op with bf16 support.
- [Monarch: Expressive Structured Matrices — arxiv:2204.00595](https://arxiv.org/pdf/2204.00595) — confirms the "fuse via Triton" framing for block-diagonal × permutation × block-diagonal pipelines.
- [Triton bf16 limitations — triton-lang/triton#2834](https://github.com/openai/triton/issues/2834) — `tl.dot` and `tl.atomic_add` historically reject bf16. Relevant to F2 backward and D1.
- [Triton autotune docs](https://triton-lang.org/main/python-api/generated/triton.autotune.html) — `BLOCK_SIZE`, `num_warps`, `num_stages` config surface for D2.
- [Flash-Attention Triton tutorial — triton-lang/triton/python/tutorials/06-fused-attention.py](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py) — reference for testing patterns and tolerance values (F6).
- [Triton-vs-CUDA performance gap (Red Hat benchmark, 2026)](https://next.redhat.com/2026/02/12/from-hand-tuned-to-generated-a-reproducible-triton-gpu-kernel-benchmark-across-different-vendors/) — empirical "78–82% of CUDA on H100/A100"; basis for A6's 60–90% target.
- [PyTorch `triton_op` tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html) — supports D9 (`torch.compile` integration story).
- [Flash-Attention AMD Triton env var conventions (`FLASH_ATTENTION_TRITON_AMD_AUTOTUNE`, `FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON`)](https://github.com/Dao-AILab/flash-attention) — basis for F5's env-var naming.
- Existing code read: `csrc/butterfly.cpp` (autograd glue), `csrc/cuda/butterfly_cuda.cu` (kernel structure, dtype dispatch, autotune table), `csrc/hadamard/hadamard_cuda_kernel.cu` (mixed-radix-2/4 reference), `csrc/diag_mult/diag_mult_cuda_kernel.cu` (trivial elementwise), `torch_structured/butterfly/multiply.py` (op wrapping + reference oracle), `torch_structured/butterfly/butterfly.py` (consumer modules), `.planning/PROJECT.md` (scope + migration strategy).
