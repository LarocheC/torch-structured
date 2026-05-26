# Domain Pitfalls

**Domain:** Porting hand-tuned CUDA kernels (structured-matrix ops with autograd) to Triton
**Project:** torch_structured v1.2 Triton Migration
**Researched:** 2026-05-26
**Confidence:** HIGH (verified against Triton issue tracker, PyTorch documentation, and the project's existing CUDA code/test tolerances)

## Reading Guide

Pitfalls are ranked by blast radius. Critical = silent correctness or rewrite required. Moderate = visible failure or perf cliff with a known workaround. Minor = annoying but easy to fix once known. Each pitfall states the phase that should own the mitigation. See "Pitfall-to-Phase Mapping" at the bottom for the full matrix.

Phase numbering uses the working set assumed in PROJECT.md v1.2:

- **Phase 1 — Triton dispatch infrastructure** (runtime selection, env flag, kernel registration shape)
- **Phase 2 — diag_mult Triton** (warm-up kernel, autograd shape established)
- **Phase 3 — hadamard Triton** (no atomicAdd, no complex)
- **Phase 4 — butterfly forward Triton** (multi-stage tile, log-N stages, complex support)
- **Phase 5 — butterfly backward + autograd Triton** (atomicAdd into d_twiddle, the heavy one)
- **Phase 6 — torch.compile + integration hardening** (triton_op wrapping, DDP/checkpointing, determinism)
- **Phase 7 — Remove csrc/ and CUDA build path** (deprecation cadence, wheel-free release)

---

## Critical Pitfalls

### Pitfall 1: Assuming Triton Has Native Complex64 Support

**What goes wrong:** Triton does not have a stable `tl.complex64` dtype. `torch.complex64` tensors passed into a Triton kernel either get rejected at compile time or — worse — get reinterpreted as a `(real, imag)` pair the kernel does not know how to handle. The forward looks correct on a smoke test, but conjugates flip sign, dot products lose the cross term, and the unitary butterfly test (`test_butterfly.py:234`, the `twiddle_matrix @ twiddle_matrix.T.conj()` check) fails because the conjugate path silently degenerated to a transpose.

**Why it happens:** Complex is a long-standing open feature request on Triton ([issue #1687](https://github.com/triton-lang/triton/issues/1687)); the language exposes only real scalar types, and `tl.dot` does not have a complex code path. Engineers porting from CUDA — where `c10::complex<float>` and the `complex_utils.cuh` header in this codebase Just Work — assume the same will hold in Triton.

**Consequences:** The library's most distinctive use case (FFT / DCT / circulant / Toeplitz via `Butterfly(complex=True)`) is broken on the Triton path. The CUDA tests in `test_butterfly.py` and `test_special.py` exercise complex twiddles heavily — they will fail. If the failure is missed, downstream code (LRU layer, FFT-as-butterfly experiments) produces silently wrong gradients.

**Warning signs:**
- Triton compile error mentioning unsupported dtype on a `complex64` tensor
- `torch.allclose(out, out_torch, ...)` passes on simple inputs but fails on `init='fft_no_br'` or `init='ortho'` complex twiddle initializations
- `torch.view_as_real` / `torch.view_as_complex` showing up in kernel call sites — usually a sign someone is trying to paper over this without committing to a layout
- Backward gradients on complex twiddles are off by a conjugate (real part right, imag part sign-flipped)

**Prevention:**
1. **Decide layout upfront in Phase 1.** Two viable strategies:
   - **Split tensors (recommended):** Carry real and imag as two separate fp32 tensors through the kernel boundary. Convert at the Python wrapper using `t.real.contiguous()` / `t.imag.contiguous()`. Reassemble with `torch.complex(out_re, out_im)`.
   - **Packed last-dim:** Reinterpret `complex64` tensor as `float32` with a trailing dim of 2 via `torch.view_as_real`. Kernel indexes `[..., 0]` for real and `[..., 1]` for imag. Avoids an extra tensor allocation but requires the kernel to know the stride trick.
2. **Implement complex multiply as a 4-FMA helper** inside the kernel (`(a+bi)(c+di) = (ac-bd) + (ad+bc)i`). Do not try to template it — write two explicit Triton functions: `_butterfly_mul_real` and `_butterfly_mul_complex`. Dispatch in the Python wrapper.
3. **Treat the unitary-butterfly correctness test as a gate.** `test_butterfly.py:234` (the `U U^* = I` assertion) is the cheapest detector of a wrong complex code path.
4. **Do not attempt to use `tl.complex64` or any "experimental" Triton complex API** — even if it parses, semantics are not guaranteed across Triton versions.

**Phase to address:** Decided in Phase 1 (dispatch design), implemented in Phase 4 (forward). Backward (Phase 5) inherits the layout.

**Confidence:** HIGH ([Triton issue #1687](https://github.com/triton-lang/triton/issues/1687) is still open; no `tl.complex*` in the [Triton language reference](https://triton-lang.org/main/python-api/triton.language.html)).

---

### Pitfall 2: Skipping FP32 Accumulator on the Backward atomicAdd

**What goes wrong:** The CUDA backward kernel reduces `d_twiddle` across the batch dimension via atomicAdd. A naive Triton port writes `tl.atomic_add(d_twiddle_ptr, partial, mask=...)` where `d_twiddle_ptr` is fp16 or bf16. The result is non-deterministically wrong: large batches see catastrophic cancellation in the reduction and `d_twiddle` gradients diverge by 10-50% from the pure-PyTorch reference.

**Why it happens:** Triton's `tl.atomic_add` on bf16 is unsupported (issue [#2834](https://github.com/openai/triton/issues/2834)), and fp16 atomic_add has historically been brittle ([issue #891](https://github.com/openai/triton/issues/891) — non-deterministic compile segfaults). Even where it compiles, the hardware atomic on fp16/bf16 has only ~3 ulp accuracy per add, which compounds badly when N (batch × n/2) is in the thousands.

**Consequences:** This is the single highest-risk correctness issue in the whole migration. The existing test suite already detects it: `test_multiply.py:56-57` deliberately multiplies tolerances by 10× when batch_size > 1024 — that is the live signature of atomic reduction noise. A naive Triton port will need 100× tolerance scaling, or fail outright.

**Warning signs:**
- `d_twiddle` test passes with rtol=1e-3 at batch_size=8 but fails at batch_size=4096
- Re-running the same test twice produces different gradient values (atomicAdd ordering)
- `torch.use_deterministic_algorithms(True)` is set in some downstream test and the test errors out with "no deterministic implementation"
- bf16/fp16 input dtype shows much larger gradient error than fp32 input dtype (more than 10× the ratio you'd expect from precision alone)

**Prevention:**
1. **Always atomically accumulate into a `float32` scratch tensor**, regardless of input dtype. Cast to the user's dtype only once, in a separate final-pass kernel that reads from the scratch and writes to `d_twiddle`.
2. **Pre-allocate the scratch in the Python wrapper** as `torch.zeros_like(twiddle, dtype=torch.float32)`. Do not allocate inside the kernel.
3. **Tile the batch dimension large enough that each block does its partial reduction in registers/shared memory first** (a `tl.sum` over a block), and only emit a single atomicAdd per block instead of one per batch element. Cuts atomic traffic by `BLOCK_M`× and reduces ULP error.
4. **Lock down a regression test** that compares Triton `d_twiddle` to `butterfly_multiply_torch.backward()` at batch_size=4096 with rtol=1e-3, fp32 inputs. If this passes, the reduction is sound.
5. **Do not use `tl.atomic_add` on a bf16/fp16 destination, period.** If a user passes those dtypes, accumulate in fp32 internally.

**Phase to address:** Phase 5 (backward). The fp32 scratch pattern is the first design decision when starting the backward kernel.

**Confidence:** HIGH (existing test tolerances confirm this is already a known noise source in the CUDA path; Triton makes it worse).

---

### Pitfall 3: Using torch.autograd.Function Instead of torch.library.triton_op

**What goes wrong:** Engineer ports the existing `ButterflyMultiply : public torch::autograd::Function` (`csrc/butterfly.cpp:99`) directly to a Python `torch.autograd.Function` subclass that calls a Triton kernel in `forward()` and another in `backward()`. It works in eager mode. Then someone wraps a model that uses `Butterfly` in `torch.compile` and gets opaque graph breaks, recompiles on every call, or just-plain-wrong outputs.

**Why it happens:** `torch.autograd.Function` is a documented foot-gun under `torch.compile`. The PyTorch dev-discuss post ["Custom Ops Under torch.compile: autograd.Function vs torch.library.custom_op"](https://dev-discuss.pytorch.org/t/custom-ops-under-torch-compile-autograd-function-vs-torch-library-custom-op/3338) and the official Triton+compile tutorial both explicitly recommend `torch.library.triton_op` + `register_autograd` instead. The C++ `autograd::Function` *was* the right call in the CUDA world; the Python equivalent is not.

**Consequences:**
- `torch.compile` falls back to eager on every Butterfly call — defeats the point of compile.
- Dynamo cache thrashes if the kernel autotunes on different shapes.
- AOTInductor / export emits unfusable opaque ops.
- The kernel cannot be inlined into a larger compiled graph, so fusion with surrounding ops (the typical perf win of `torch.compile` on memory-bound kernels) is lost.

**Warning signs:**
- `TORCH_LOGS="graph_breaks"` shows breaks at every butterfly call
- `torch.compile(model)(x)` is slower than eager
- Stack traces under compile mention "speculate_subgraph on bw failed" ([issue #125489](https://github.com/pytorch/pytorch/issues/125489))
- The `mutates_args` argument is missing or wrong on a kernel that writes to `d_twiddle`

**Prevention:**
1. **Use `torch.library.triton_op` from day one.** Even Phase 2 (`diag_mult`, the simplest kernel) should be wrapped with `triton_op` so the pattern is locked in before the heavy kernels.
2. **Use `register_autograd` to attach the backward**, not a custom `autograd.Function`. The backward function itself must call `triton_op`-wrapped kernels (not raw `@triton.jit` functions) if you want compile to see through the backward.
3. **Register a `register_fake` ("meta kernel") for every op.** Without it, `torch.compile` cannot trace shape propagation. For butterfly: input `(B, S, K)` + twiddle `(S, blocks, log_n, n/2, 2, 2)` → output `(B, S, output_size)`. Trivial to write but easy to forget.
4. **Declare `mutates_args=()` explicitly.** Butterfly forward is pure; backward mutates the pre-allocated `d_twiddle` scratch and `d_input`, so its `mutates_args` should list those.
5. **Avoid default argument values on the op.** [Issue #162687](https://github.com/pytorch/pytorch/issues/162687) breaks compile when callers pass the default value explicitly. Make every argument required at the C++/Python op boundary; default in a Python wrapper above the op.

**Phase to address:** Phase 1 (decide the wrapping pattern) and enforced in every phase 2-5. Phase 6 verifies compile composability end-to-end.

**Confidence:** HIGH ([official PyTorch tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html) and [dev-discuss post](https://dev-discuss.pytorch.org/t/custom-ops-under-torch-compile-autograd-function-vs-torch-library-custom-op/3338) both state this explicitly).

---

### Pitfall 4: Comparing Triton Output to the CUDA Reference Instead of the Pure-PyTorch Reference

**What goes wrong:** Engineer runs `butterfly_multiply_triton(...)` and `butterfly_multiply_cuda(...)` side-by-side, sees them disagree at the 1e-4 level, panics, and rewrites the Triton kernel chasing a bug that is not there. Meanwhile the actual bug (a missing twiddle dimension swap) is masked because both kernels handle small inputs identically.

**Why it happens:** The CUDA kernel uses warp shuffles, `__fmaf_rn`, and a specific reduction order. The Triton kernel uses `tl.sum` (a tree reduction) and may emit different FMA patterns. Two correct kernels can differ by ~1e-5 relative error on fp32 just from FMA ordering and rounding. Worse, the CUDA kernel's atomicAdd order is non-deterministic, so it is not even a stable reference.

**Consequences:** Wasted weeks chasing phantom bugs. Or — opposite failure — accepting a Triton kernel that happens to match CUDA on small cases but is actually wrong on larger ones, because both share a common-mode error (e.g., both pad the input dim the same wrong way).

**Warning signs:**
- Tolerance keeps creeping up to match what the diff is, rather than what the math requires
- "Why does this test fail at n=512 but pass at n=256?"
- The CUDA path also has nondeterministic test failures at large batch (see Pitfall 2's tolerance scaling)

**Prevention:**
1. **`butterfly_multiply_torch` is the only correctness oracle.** It is deterministic, dtype-stable, and the math is auditable in 20 lines of Python (`multiply.py:28-49`). Every Triton kernel — forward and backward — is graded against it, not against CUDA.
2. **Compare CUDA to `_torch` as a baseline.** Establish what tolerance `_torch` vs `_cuda` currently has. Then require `_torch` vs `_triton` to be at least that tight. If you cannot match CUDA's tolerance, that is information — surface it, do not paper over it.
3. **Use `torch.testing.assert_close`** instead of `torch.allclose` in new tests. It produces actionable diff messages (max element error, indices).
4. **Test in fp64 first.** Run the entire Triton kernel in fp64 input/twiddle and compare to `_torch` in fp64. If fp64 fails, the math is wrong, not the precision. Catch this before debugging fp32 rounding.
5. **Realistic tolerances:** fp32 forward `rtol=1e-5, atol=1e-6` is achievable. Backward `d_input` similarly. Backward `d_twiddle` with batch=4096 atomicAdd reduction realistically needs `rtol=1e-3, atol=1e-4` even with a fp32 scratch — match the existing CUDA test tolerances (`test_multiply.py:16-17`) as a starting point.

**Phase to address:** Phase 2 onwards — set the testing pattern at the very first kernel (`diag_mult`) and reuse it.

**Confidence:** HIGH ([Triton issue #5283](https://github.com/triton-lang/triton/issues/5283) discusses exactly this — Triton matmul tutorial tolerances had to be relaxed because of FMA differences with cuBLAS).

---

## Moderate Pitfalls

### Pitfall 5: First-Call JIT and Autotune Cost Tanks CI Wall Time

**What goes wrong:** Every test process pays the Triton JIT cost for every kernel × every config. With aggressive `@triton.autotune` configs, the first call can take 30-120 seconds for the multi-stage butterfly forward (5-stage tile × multiple block sizes × num_warps × num_stages). The pytest suite that used to take 60s now takes 20 minutes in CI.

**Why it happens:** Triton compiles lazily on first call per signature. `@triton.autotune` benchmarks every config in the search space on first call ([Triton autotune docs](https://triton-lang.org/main/python-api/generated/triton.autotune.html)). Cache is keyed per-process by default and not persisted across CI runs unless explicitly configured.

**Consequences:** Developer-experience regression. CI becomes the slowest part of iteration. Engineers start `-x`-ing tests to keep the loop tight, which masks regressions.

**Warning signs:**
- First test of the suite hangs for 30+ seconds without output
- CI total time jumps 10×
- `TRITON_CACHE_DIR` is unset in CI environment

**Prevention:**
1. **Keep the autotune surface small.** Phase 1 prescribes the search space: 2-4 block sizes, 2 `num_warps` values, 1-2 `num_stages` values. Total configs per kernel ≤ 8. Resist the urge to autotune over 24+ configs.
2. **Persist `TRITON_CACHE_DIR` in CI.** Use the GitHub Actions cache action keyed on (Python version, torch version, triton version, kernel source hash). Subsequent CI runs skip recompile + autotune entirely.
3. **Use `cache_results=True`** on `@triton.autotune` ([feature request #4020](https://github.com/triton-lang/triton/issues/4020) is closed, this exists). Persists timings to disk so a fresh process reuses winning configs.
4. **Pre-warm in a session-scoped pytest fixture.** A `conftest.py` fixture calls each kernel once with representative shapes before any test runs, paying the JIT cost once per pytest session.
5. **Have a `--skip-triton` pytest flag** for CPU-only iteration. The `_torch` reference is usable for shape/API tests without needing the GPU path.
6. **Do not autotune the backward.** It runs less often than forward in practice (typical training step is 1 fw : 1 bw), and the autotune surface compounds. Pick the best config from forward autotune as a starting heuristic and hard-code it.

**Phase to address:** Phase 1 (CI infra: cache dir setup + autotune config policy) and Phase 6 (full integration check).

**Confidence:** MEDIUM (cache-key shape is well documented; specific CI wall-time numbers are project-dependent).

---

### Pitfall 6: torch.use_deterministic_algorithms(True) Breaks the Backward

**What goes wrong:** Downstream user sets `torch.use_deterministic_algorithms(True)` for reproducible training. The Triton backward kernel uses `tl.atomic_add` on `d_twiddle`. PyTorch's determinism guard does not see the atomic_add inside the kernel (it cannot — Triton is opaque to it), so it does not error or warn, but the user's overall reproducibility silently breaks: two runs with the same seed produce slightly different gradients.

**Why it happens:** PyTorch's `use_deterministic_algorithms` checks a list of known non-deterministic ops. A custom triton_op is not on that list. The user trusts the global flag; the flag silently does not cover the new code.

**Consequences:** Reproducibility bug that is invisible until a user reports "I cannot reproduce my paper's numbers." Common in research codebases (which is this library's target audience). Worse than crashing — it's a silent integrity failure.

**Warning signs:**
- Setting `torch.use_deterministic_algorithms(True, warn_only=True)` in a test does not warn at all on a butterfly backward call
- Reproducibility tests (run-twice-and-compare) start failing only with Triton path enabled
- Bit-exact tests on CPU pass; CUDA path differs run-to-run beyond fp rounding noise

**Prevention:**
1. **Manually check the determinism flag inside the Python wrapper.** `if torch.are_deterministic_algorithms_enabled(): raise RuntimeError("butterfly_multiply Triton backward uses atomicAdd and is not deterministic. Set deterministic=False or use the CUDA fallback path.")`
2. **Respect `warn_only=True`** with `warnings.warn(...)` instead of raising.
3. **Provide a deterministic backward path** as an opt-in: a kernel that does the per-batch partial reductions via `tl.sum` with no inter-block atomic — slower but deterministic. Behind a flag like `torch_structured.set_deterministic(True)`.
4. **Document this explicitly in the README** under "Reproducibility": "butterfly backward uses GPU atomics; setting torch.use_deterministic_algorithms requires explicit opt-in via torch_structured.set_deterministic(True)."

**Phase to address:** Phase 5 (when backward + atomicAdd lands) and re-verified in Phase 6.

**Confidence:** MEDIUM-HIGH ([torch.use_deterministic_algorithms docs](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html) confirm the list of monitored ops is static; custom ops are not auto-tracked).

---

### Pitfall 7: Multi-Stage Shared-Memory Tile Doesn't Map to Triton's Implicit Allocation

**What goes wrong:** The CUDA kernel (`butterfly_cuda.cu:40-58`) defines a 5-stage tile that holds twiddle and intermediate values in shared memory across 5 butterfly stages, hand-tuned for sm_70+ with `MIN_BLOCKS_PER_MP` annotations and explicit `ITEMS_PER_THREAD` schedules. The Triton port tries to replicate by loading 5 stages worth of twiddle into a register-resident tile. The autotuner cannot find a working config — either spills to local memory (DRAM), drops occupancy to 1 block/SM, or fails to compile.

**Why it happens:** Triton manages shared memory and registers implicitly via `num_warps`, `num_stages`, and `BLOCK_*` constexpr. The CUDA kernel's specific 5-stage layout (`MAX5_FORWARD_BLOCK_SIZE = 512`, `ITEMS_PER_THREAD_FORWARD_MAX5[]`) embeds knowledge that no Triton autotune config can express. H100/Blackwell SRAM ceilings (~228KB H100, ~256KB SM100) get hit, and Triton silently spills.

**Consequences:** Forward perf is 50-80% of the CUDA kernel even on a "successful" port. Or worse — the 5-stage variant works at small n but breaks at n ≥ 2048 because the tile no longer fits.

**Warning signs:**
- `ncu` shows local memory loads on the Triton kernel (register spills to DRAM)
- Autotune picks `num_warps=1, BLOCK_SIZE=64` as the winner (small tile = no spill but no perf)
- Achieved bandwidth < 40% of peak on a memory-bound kernel
- Compile error: "shared memory size exceeded" — the autotuner blacklisted larger configs

**Prevention:**
1. **Do not try to replicate the 5-stage tile on day one.** Phase 4 ports the *3-stage* variant of the forward (simpler tile, similar perf for typical n=512-2048 sizes). Phase 4b (or deferred) attempts the 5-stage if the perf gap is unacceptable.
2. **Use `tl.constexpr` for `LOG_N`, `NSTAGES_PER_TILE`, `BLOCK_BATCH`.** Constant propagation frees registers; the compiler can unroll the inner loop. Forgetting this is a common cause of "Triton port is 3× slower than CUDA."
3. **Profile early with Nsight Compute, not just wall-clock.** The Nsight metrics to watch: achieved occupancy, shared memory per block, register count per thread, local memory load/store (= spill). If local memory > 0, you have a spill.
4. **Have a fallback per-stage kernel.** A kernel that does ONE butterfly stage per launch (`log_n` launches total) is bandwidth-bound and easy to get right. Slower than the multi-stage tile but always works. Ship this first, optimize later.
5. **Set explicit ceilings in autotune configs.** `num_stages ∈ {1, 2, 3}`, `num_warps ∈ {4, 8}`, `BLOCK_BATCH ∈ {16, 32, 64}` keeps the search small and avoids the "shared memory exceeded" black hole.

**Phase to address:** Phase 4 (forward) — start with 3-stage tile, defer 5-stage. Phase 5 (backward) — same: per-stage kernel first.

**Confidence:** HIGH (the CUDA code's hand-tuned constants visible at `butterfly_cuda.cu:40-58` make this gap explicit; [register spill discussion](https://www.spheron.network/blog/openai-triton-kernel-gpu-cloud-2026/) confirms the symptom pattern).

---

### Pitfall 8: gradcheck Fails Because the Reference Backward and the Triton Backward Were Both Hand-Derived

**What goes wrong:** Engineer hand-derives the butterfly backward math, implements it in Triton, runs `torch.autograd.gradcheck` against `butterfly_multiply_torch.backward()`, and gets a "Numerical gradient does not match analytical gradient" failure with a confusing error pointing at element [3, 1, 7, 0, 1, 0] of `d_twiddle`. The fix is non-obvious: either the math derivation is wrong (transpose of a 2×2 block matters), or the reference `_torch.backward` is wrong (autograd's automatic backward of `_torch` is the ground truth, not a hand-derived one).

**Why it happens:** The CUDA backward is a *hand-coded* derivative — `butterfly_cuda.cu` contains explicit derivative arithmetic. Translating that arithmetic into Triton, *with* a different memory layout for atomicAdd, is error-prone. There are at least three layouts where a sign flip or transpose can hide.

**Consequences:** Days of debugging. The wrong fix (loosening tolerances) ships a subtly-wrong backward to production. Real users hit it as "my model trains 10% worse with the Triton path."

**Warning signs:**
- gradcheck fails but `torch.allclose(d_twiddle_triton, d_twiddle_torch_autograd, rtol=1e-3)` *passes* — means the hand-derived `_torch.backward` is wrong (use autograd of `_torch` forward as the oracle instead)
- The error scales linearly with batch size (atomicAdd noise, not a math bug)
- Error is concentrated on the last stage of butterfly twiddles (off-by-one in `increasing_stride` flip)
- Only the complex path fails, real path passes (conjugate in the derivative is wrong)

**Prevention:**
1. **The oracle is `autograd.grad(butterfly_multiply_torch(...), inputs)`, NOT a hand-derived reference.** PyTorch's autograd applied to the pure-Python forward is the ground truth — it derives the backward from chain rule, not from human algebra. Trust it.
2. **Test in fp64.** `torch.autograd.gradcheck` defaults assume fp64. Run the whole Triton path in fp64 for gradcheck. If gradcheck passes in fp64, the math is right and any fp32 disagreement is precision noise.
3. **Test the backward in three layers, in order:**
   1. **gradcheck on n=4, batch=1, log_n=2** — smallest case that exercises one full butterfly tile. If this fails, the math is wrong.
   2. **`torch.allclose(d_input_triton, autograd.grad(_torch_fw, input))` at n=256, batch=8** — exercises stride logic.
   3. **`torch.allclose(d_twiddle_triton, autograd.grad(_torch_fw, twiddle))` at n=512, batch=4096** — exercises the atomic reduction.
4. **Bisect by replacing one piece at a time.** Triton forward + autograd backward of `_torch`. If that gradcheck passes, the forward is right; bug is in the Triton backward. If it fails, the forward is wrong.
5. **Keep the CUDA backward available during Phase 5.** Run both in parallel; if Triton disagrees with CUDA *and* with autograd-of-`_torch`, both kernels probably have the same bug.

**Phase to address:** Phase 5 (backward). The 3-layer gradcheck pattern is a phase entry gate.

**Confidence:** HIGH (canonical advice; matches [PyTorch gradcheck docs](https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html) and the project's existing test structure in `test_multiply.py:56-57`).

---

### Pitfall 9: Removing the CUDA Path Before Triton Reaches Parity

**What goes wrong:** Phase 5 ships, all tests pass, the team celebrates and deletes `csrc/`. Two weeks later a user reports a 3× slowdown on a specific shape (n=1024, batch=1, the latency-sensitive case the 5-stage CUDA kernel was tuned for) or a hard crash on an old GPU (sm_60) where Triton silently emits broken PTX. There is no rollback path because the CUDA code is gone.

**Why it happens:** Pressure to "finish the migration." Test coverage is shape-based and may not exhibit the regression. Triton's GPU support matrix is narrower than what the CUDA kernel targeted (the CUDA build targets sm_35+ per `setup.py`).

**Consequences:** Emergency revert PR, or worse, a "fix it forward" rush. User trust erodes. The `csrc/` deletion git commit becomes legendary in retrospectives.

**Warning signs:**
- "We ported it, the tests pass, let's clean up" — said before benchmarks on diverse shapes/GPUs
- No deprecation period was scheduled
- No telemetry on what fraction of users are on the CUDA path
- The CUDA path was never gated behind a deprecation warning in any release

**Prevention:**
1. **Two-release deprecation cadence.** v1.2 ships with both paths and `torch_structured.use_triton=True` as default. v1.2 emits a `DeprecationWarning` when CUDA path is selected. v1.3 default-disables the CUDA build but keeps the source. v1.4 removes `csrc/` only after at least one release cycle with no user-reported regressions.
2. **Benchmark across the full shape sweep before declaring parity.** The shape grid in `benchmark_utils.py` covers n ∈ {64, 128, ..., 8192}, batch ∈ {1, 8, 64, 1024, 4096}. Triton path must be within 0.9× of CUDA on every cell, AND faster on at least 30% of cells. If not, the perf-critical shapes are still served by CUDA via a runtime selector.
3. **Test on at least three GPU generations.** sm_70 (V100), sm_80 (A100), sm_90 (H100). Triton has known regressions on older arches.
4. **Keep `csrc/` for one full release after disabling its build.** Source remains, build is off by default. Re-enable by env var. This catches issues a user can report on without a forensic git archaeology session.
5. **Phase 7 has an explicit "removal gate" review** — not just "tests pass" but "benchmarks pass, 2 release cycles elapsed, no open issues mentioning the CUDA path."

**Phase to address:** Phase 7. The discipline is set in Phase 1 (have a deprecation plan written before starting).

**Confidence:** HIGH (general software migration wisdom; specific to this codebase given the perf-sensitive nature of the kernels).

---

## Minor Pitfalls

### Pitfall 10: Forgetting Power-of-2 Edge Cases (n=1, n=2)

**What goes wrong:** The CUDA kernel is templated/specialized for small n. Triton's BLOCK_SIZE must be a power of 2 — fine when n=512, but a Triton kernel that hard-codes `BLOCK_SIZE = n // 2` breaks at n=1 (BLOCK_SIZE=0, invalid) and at n=2 (BLOCK_SIZE=1, also invalid in some Triton versions).

**Why it happens:** The corner case is rare in practice but `Butterfly(in_size=1)` is legal in the existing API. The test suite exercises n=2 in `test_multiply.py`.

**Prevention:**
- Minimum `BLOCK_SIZE=4` (or fall back to the pure-PyTorch path for n < 4)
- Explicit early-return in the Python wrapper for n=1 (identity) and n=2 (single butterfly factor)
- Add n=1 and n=2 to the test parameterization explicitly

**Phase to address:** Phase 4 (forward) — surfaces during the first port.

**Confidence:** HIGH (verified against [Triton issue #1966](https://github.com/triton-lang/triton/issues/1966) — BLOCK_SIZE must be power-of-2 constexpr).

---

### Pitfall 11: `view_as_real` / `view_as_complex` Non-Contiguous Strides

**What goes wrong:** `torch.view_as_real(complex_tensor)` returns a non-contiguous view. Passing it to a Triton kernel that assumes contiguous strides reads garbage. The kernel does not crash — outputs are just wrong.

**Why it happens:** Triton kernels typically use `tl.load(ptr + offsets)` with a flat offset assumption. Strides do not propagate automatically.

**Prevention:**
- Always `.contiguous()` after `view_as_real` if the consumer expects a packed (real, imag) layout
- Pass strides explicitly as kernel arguments and use them in offset arithmetic
- Add a `assert tensor.is_contiguous()` precondition in the Python wrapper for any tensor that the kernel will treat as such

**Phase to address:** Phase 4 (complex forward).

**Confidence:** HIGH.

---

### Pitfall 12: FSDP Sharding the Twiddle Parameter

**What goes wrong:** A user wraps a model containing `Butterfly` with FSDP. FSDP shards the `twiddle` parameter across ranks. The Triton kernel expects the *full* twiddle tensor on each rank. On forward, the kernel reads only a fraction of the twiddle and produces wrong output. Different ranks get different wrong outputs.

**Why it happens:** Custom autograd ops don't automatically participate in FSDP's all-gather. FSDP assumes the op uses standard PyTorch ops that hook into its sharding plumbing.

**Consequences:** Silent correctness bug in distributed training. Loss diverges or trains to a different local minimum per run.

**Prevention:**
- Mark the twiddle parameter with `param._is_structured = True` (already done in `butterfly.py:57`) — extend with a `_no_fsdp_shard = True` hint and document that FSDP users should add `twiddle` to the no-shard list
- Test with a 2-GPU FSDP wrapping of a Butterfly-containing model in Phase 6
- Document FSDP/DDP behavior in the README

**Phase to address:** Phase 6 (integration hardening).

**Confidence:** MEDIUM (cross-cutting; not unique to this kernel — affects any custom op + FSDP combo).

---

### Pitfall 13: Gradient Checkpointing Re-runs the Forward — and Re-pays JIT Cost

**What goes wrong:** User wraps a Butterfly call in `torch.utils.checkpoint.checkpoint(...)`. During backward, the forward is re-run, which triggers a fresh Triton kernel launch. If autotune is configured eagerly, the second forward triggers re-autotuning (because the activation tensors during recompute have different memory addresses → potentially different cache key). Backward pass becomes 5-10× slower.

**Why it happens:** Some Triton autotune cache keys include pointer addresses or strides that change under checkpointing. The recompute path looks like a "new shape" to the cache.

**Prevention:**
- Verify Triton autotune cache key uses tensor shape/dtype, not addresses (default behavior in recent Triton is correct; verify the version)
- Test gradient checkpointing of a Butterfly module in Phase 6 — wall-time should not blow up
- If it does, pre-warm the cache or pin the config (`@triton.jit` without autotune, hard-coded best config)

**Phase to address:** Phase 6.

**Confidence:** MEDIUM ([discussion](https://discuss.pytorch.org/t/what-is-the-current-future-best-practice-for-custom-autograd-functions-with-triton-parts/207810) notes Triton+checkpoint interactions are subtle but typically fine if `triton_op` is used).

---

### Pitfall 14: Memory-Bound Kernel That's Actually Compute-Bound on Small N

**What goes wrong:** Engineer profiles the multi-stage butterfly on n=4096, sees it's memory-bound, optimizes for bandwidth. Same kernel on n=64 is compute-bound (FFT/DCT has high arithmetic intensity at small n), and the bandwidth-optimized config picks a low `num_warps` that starves the SMs of compute. Small-n perf regresses 3-5× compared to CUDA.

**Why it happens:** The CUDA kernel has per-n-bucket tuning (`ITEMS_PER_THREAD_FORWARD[14]` — different schedule for each log_n from 1 to 14). Triton autotune sees the kernel as one config space and picks an average-best.

**Prevention:**
- Bucket autotune by log_n: separate `@triton.autotune` configs for log_n ≤ 6, 6 < log_n ≤ 10, log_n > 10. Effectively three kernels under one dispatcher.
- Benchmark sweep: report perf at each log_n × batch_size cell, not just one or two
- Profile both regimes with `ncu` — memory utilization vs SM throughput tells you which regime you are in

**Phase to address:** Phase 4 (forward), surfaces in Phase 6 perf review.

**Confidence:** HIGH (the CUDA constants in `butterfly_cuda.cu:42-58` directly evidence this regime split).

---

## Technical Debt Patterns

Shortcuts that look reasonable but cost dearly later.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Wrap kernel with `torch.autograd.Function` instead of `triton_op` | Simpler code, fewer files | Breaks `torch.compile`, opaque to fusion, deprecation pain later | Never — use `triton_op` from Phase 2 onwards |
| Skip the fp32 scratch in backward atomicAdd | Saves a kernel launch and memory allocation | Wrong gradients at batch ≥ 1024; debugging cost is days | Never |
| Disable autotune entirely and hard-code one config | Eliminates JIT cost in CI | Perf is 30-60% of optimal across shape grid | OK as a temporary measure during Phase 2-3 (simple kernels); not for Phase 4-5 |
| "We'll loosen tolerances to make tests pass" | Tests turn green immediately | Hides real bugs; ships subtly-wrong gradients | Only when the loosening is explained by a documented precision source (fp32 atomic ULP, FMA) and is bounded by what the CUDA path already needs |
| Keep complex unsupported on the Triton path "for now" | Ships Phase 4 faster | Library's primary value prop (FFT/DCT/circulant) becomes CUDA-only and migration is incomplete | Phase 4 can ship real-only, but Phase 4b must complete complex before Phase 7 deprecation |
| Delete `csrc/` in the same PR that lands Phase 5 | "Clean migration" optics | No rollback, no perf comparison after release | Never — minimum 1 release cycle of parallel paths |
| Hand-derive the backward math in Triton without an autograd-of-`_torch` oracle | Feels rigorous | High risk of sign / transpose / conjugate errors that survive testing | Only with the 3-layer gradcheck pattern from Pitfall 8 |

---

## Integration Gotchas

Common mistakes when connecting to PyTorch subsystems and tooling.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| `torch.compile` | Wrapping Triton in `torch.autograd.Function` | Use `torch.library.triton_op` + `register_autograd`; register a `fake` (meta) kernel; no default arguments |
| DDP | Assuming standard hooks work | Confirmed compatible if op is via `triton_op` and `mutates_args` is correct; test in Phase 6 |
| FSDP | Sharding the twiddle parameter | Document FSDP no-shard guidance for twiddle; or shard with explicit all-gather in forward |
| Gradient checkpointing | Re-pays JIT on recompute | Pre-warm cache; verify autotune cache key uses shape not pointer |
| `use_deterministic_algorithms(True)` | Custom op is invisible to the determinism guard | Manually check the flag in Python wrapper; provide deterministic backward as opt-in |
| `torch.jit.script` | Triton kernels are not scriptable | The existing `multiply.py` uses `@torch.jit.script` wrappers; these must change to call the `triton_op` directly (jit.script is incompatible with Triton kernels) |
| `torch.fx` tracing | Triton ops break tracing | `triton_op` is FX-traceable as opaque; fine for typical use |
| `torch.cuda.amp.autocast` | Mixed precision casts inputs unexpectedly | Inside the op, accept any input dtype, accumulate atomic in fp32 regardless |
| Multi-stream | Kernel launches default stream | Use `torch.cuda.current_stream()` and pass to Triton via `stream=` if calling raw |

---

## Performance Traps

Patterns that pass small-scale benchmarks but break under load.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Single autotune over the full log_n range | Some shapes 3× slower than CUDA | Bucket autotune by log_n (small/medium/large) | log_n ≤ 6 (compute-bound) and log_n ≥ 11 (memory-bound) — opposite regimes |
| atomicAdd directly to bf16/fp16 `d_twiddle` | Gradient noise at batch ≥ 1024; nondeterministic test failures | fp32 scratch + final cast | batch_size × n/2 ≳ 65k atomic operations per twiddle slot |
| One atomicAdd per batch element instead of per block | Atomic traffic 100× higher than needed; bandwidth-starved | `tl.sum` block-reduce first, single atomicAdd per block | batch ≥ 128, any n |
| Block size in shared memory > 64 KB | Compile errors or 1 block/SM | Cap autotune block size; profile with ncu | n ≥ 4096 with multi-stage tile |
| Re-allocating scratch tensors inside kernel call | Allocator pressure, fragmented memory | Pass pre-allocated scratch from Python wrapper | High-frequency calls (training inner loop) |
| Not pinning torch.compile cache key | Recompiles every iteration | Use `triton_op`; explicit shape/dtype in op signature | Variable batch sizes in training |
| Autotune surface too large | First call takes 60+ seconds | ≤ 8 configs per kernel; `cache_results=True` | CI and fresh-process startup |
| Treating Triton as a drop-in for warp shuffles | Triton lacks fine warp control; perf gap on multi-stage kernels stays open | Accept 70-90% of CUDA perf on the most-tuned kernels; do not over-invest | Multi-stage butterfly forward, log_n ∈ [8, 11] |

---

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing a critical piece. Each kernel phase should pass all of these before moving on.

- [ ] **Forward kernel:** Often missing complex support — verify `Butterfly(complex=True)` produces correct output AND correct conjugate (`test_butterfly.py:151,154`)
- [ ] **Forward kernel:** Often missing `output_size != n` truncation — verify both n==output_size and n>output_size
- [ ] **Forward kernel:** Often missing `increasing_stride=False` path — verify both stride directions
- [ ] **Forward kernel:** Often missing n=1, n=2 corner cases — verify against `_torch` reference
- [ ] **Backward kernel:** Often missing fp32 scratch — verify gradient parity at batch=4096
- [ ] **Backward kernel:** Often missing both `d_input` AND `d_twiddle` paths — verify `ctx.needs_input_grad` is honored (autograd will silently skip if not)
- [ ] **Backward kernel:** Often missing deterministic-mode guard — verify `torch.use_deterministic_algorithms(True)` raises or warns
- [ ] **Autograd wrapper:** Often missing `register_fake` — verify `torch.compile` does not graph-break
- [ ] **Autograd wrapper:** Often missing `mutates_args` declaration — verify Inductor does not produce wrong code
- [ ] **Autotune:** Often missing `cache_results=True` — verify second process startup is fast
- [ ] **CI:** Often missing persistent `TRITON_CACHE_DIR` — verify GHA cache hit rate
- [ ] **Tests:** Often missing fp64 gradcheck — verify math correctness independent of precision
- [ ] **Tests:** Often missing test on 3+ GPU generations — verify sm_70, sm_80, sm_90 all pass
- [ ] **Deprecation:** Often missing DeprecationWarning on the old CUDA path — verify v1.2 release emits the warning when CUDA is selected
- [ ] **Docs:** Often missing reproducibility caveat — verify README mentions atomicAdd nondeterminism

---

## Recovery Strategies

When a pitfall lands despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Complex not supported (P1) | MEDIUM | Add `view_as_real` path; revert to CUDA for `complex=True` via runtime selector; ship Phase 4b later |
| Backward atomicAdd precision (P2) | MEDIUM | Add fp32 scratch buffer; new kernel + final cast pass; preserves API |
| Used `torch.autograd.Function` (P3) | MEDIUM-HIGH | Refactor to `triton_op` + `register_autograd`; touches every callsite of the op |
| Triton vs CUDA tolerance drift (P4) | LOW | Switch reference to `autograd.grad(_torch_fw, ...)`; relax tolerances within documented ULP bounds |
| CI wall time tanked (P5) | LOW | Cap autotune configs; enable `cache_results`; persist cache dir |
| Determinism guard missed (P6) | LOW | Add manual flag check in Python wrapper; ship in patch release |
| Multi-stage tile won't autotune (P7) | MEDIUM | Fall back to 3-stage or per-stage kernel; defer 5-stage to a later phase |
| Backward gradcheck fails (P8) | HIGH | Bisect with autograd-of-`_torch` oracle; fp64 first; isolate forward vs backward |
| csrc/ removed too early (P9) | VERY HIGH | Revert; restore from git; postpone deprecation by one release; never repeat |
| Power-of-2 edge case (P10) | LOW | Add Python wrapper early-return for n < 4; add tests |
| view_as_real strides (P11) | LOW | `.contiguous()` in Python wrapper |
| FSDP shards twiddle (P12) | MEDIUM | Document; add no-shard hint on parameter; or implement all-gather in op |
| Checkpointing JIT thrash (P13) | LOW | Pre-warm cache or pin config |
| Compute vs memory-bound regime mismatch (P14) | MEDIUM | Split autotune by log_n bucket |

---

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls. "Prevent in" = phase that must design for it. "Verify in" = phase whose test gate proves it was prevented.

| Pitfall | Severity | Prevent In | Verify In | Verification |
|---------|----------|------------|-----------|--------------|
| P1: Complex unsupported | CRITICAL | Phase 1 (layout decision) | Phase 4 | Unitary butterfly test (`U U^* = I`) passes with complex=True |
| P2: bf16/fp16 atomicAdd | CRITICAL | Phase 5 | Phase 5 | d_twiddle parity vs `_torch` at batch=4096 |
| P3: autograd.Function vs triton_op | CRITICAL | Phase 1 | Phase 6 | `torch.compile(model)` shows no graph breaks |
| P4: Wrong correctness oracle | CRITICAL | Phase 2 (testing pattern) | All phases | Every kernel test compares to `_torch`, not to CUDA |
| P5: Autotune/JIT CI cost | MODERATE | Phase 1 (CI infra) | Phase 6 | CI wall time within 1.5× of pre-Triton baseline |
| P6: Determinism guard | MODERATE | Phase 5 | Phase 6 | `use_deterministic_algorithms(True)` raises or warns on backward |
| P7: Multi-stage tile mismatch | MODERATE | Phase 4 (defer 5-stage) | Phase 4 | 3-stage kernel within 0.9× of CUDA on log_n ∈ [8, 11] |
| P8: gradcheck failures | MODERATE | Phase 5 (3-layer test pattern) | Phase 5 | gradcheck passes in fp64; allclose passes in fp32 |
| P9: csrc/ removed early | CRITICAL | Phase 1 (deprecation plan) | Phase 7 | 2-release cycle elapsed; benchmark gates passed |
| P10: Power-of-2 edges | MINOR | Phase 4 | Phase 4 | n=1, n=2 tests pass |
| P11: view_as_real strides | MINOR | Phase 4 | Phase 4 | Tests on complex with non-contiguous inputs |
| P12: FSDP sharding | MINOR | Phase 6 | Phase 6 | 2-GPU FSDP smoke test |
| P13: Checkpoint JIT thrash | MINOR | Phase 6 | Phase 6 | gradient_checkpointing perf within 2× of non-checkpointed |
| P14: Regime mismatch | MODERATE | Phase 4 | Phase 6 | Perf grid: every log_n × batch cell within 0.9× of CUDA |

---

## Sources

### Primary (HIGH confidence)

- [Triton issue #1687 — Complex number support feature request](https://github.com/triton-lang/triton/issues/1687) — confirms no native complex
- [Triton issue #2834 — Bf16 with tl.dot and tl.atomic_add](https://github.com/openai/triton/issues/2834) — atomic_add bf16 unsupported
- [Triton issue #891 — atomic_add for fp16 non-deterministic compile segfaults](https://github.com/openai/triton/issues/891) — fp16 atomic_add fragility
- [PyTorch tutorial — User-defined Triton kernels with torch.compile](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html) — triton_op vs custom_op recommendation
- [PyTorch dev-discuss — Custom Ops Under torch.compile](https://dev-discuss.pytorch.org/t/custom-ops-under-torch-compile-autograd-function-vs-torch-library-custom-op/3338) — explicit recommendation to use triton_op
- [torch.use_deterministic_algorithms docs](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html) — determinism guard semantics
- [Triton issue #5283 — Numerical accuracy test failing tolerances](https://github.com/triton-lang/triton/issues/5283) — FMA precision differences
- [Triton issue #1966 — BLOCK_SIZE constexpr power-of-2](https://github.com/triton-lang/triton/issues/1966) — block size constraint
- Project's own `csrc/cuda/butterfly_cuda.cu` lines 40-58 — evidence of hand-tuned constants that Triton autotune cannot replicate directly
- Project's own `tests/test_multiply.py` lines 16-17, 56-57 — existing tolerance pattern confirms atomicAdd precision sensitivity

### Secondary (MEDIUM confidence)

- [Triton autotune docs](https://triton-lang.org/main/python-api/generated/triton.autotune.html) — cache_results parameter
- [Triton issue #4020 — Persistent autotune cache RFC](https://github.com/triton-lang/triton/issues/4020) — cache persistence strategy
- [PyTorch issue #125489 — Compiled autograd + user-defined Triton kernel](https://github.com/pytorch/pytorch/issues/125489) — speculate_subgraph bw failures
- [PyTorch issue #162687 — custom op default arguments break compile](https://github.com/pytorch/pytorch/issues/162687) — no defaults on op signature
- [PyTorch forum — best practice for custom autograd with Triton parts](https://discuss.pytorch.org/t/what-is-the-current-future-best-practice-for-custom-autograd-functions-with-triton-parts/207810) — current recommendations
- [PyTorch forum — DDP with custom gradient computations](https://discuss.pytorch.org/t/ddp-does-not-work-with-custom-gradient-backward-computations/222882) — DDP + custom autograd limitations
- [Spheron — Triton Kernel Development Guide 2026](https://www.spheron.network/blog/openai-triton-kernel-gpu-cloud-2026/) — register spill and shared memory ceiling guidance

### Cross-reference

- `.planning/research/STACK.md` — Triton + torch.library decisions live here
- `.planning/research/ARCHITECTURE.md` — kernel dispatch architecture, where the runtime selector lives
- `.planning/research/FEATURES.md` — feature parity matrix between CUDA and Triton paths
