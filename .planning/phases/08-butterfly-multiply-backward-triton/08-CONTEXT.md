# Phase 8: butterfly_multiply Backward (Triton) - Context

**Gathered:** 2026-05-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace Phase 7's `_backward` callback (which delegates to `torch.autograd.grad(_torch_ref.butterfly_multiply_torch(...), [twiddle, input], grad_out)` per D-47) with a real **Triton backward kernel** that computes `(d_twiddle, d_input)` natively. Backward uses a **pre-allocated fp32 scratch accumulator** for `d_twiddle` with **block-level `tl.sum` reduce + single `tl.atomic_add` per program** (SC#3). Backward kernel-launch structure **mirrors Phase 7's 3-stage-tile pattern in reverse stage-group order**. Backward callback **materializes the full per-stage activation trail once** by re-running the Triton forward into a `(log_n * nblocks, batch, nstacks, n)` fp32 trail buffer, then walks back. Two plans split by dtype, mirroring Phase 7's 07-01/07-02 shape: 08-01 fp32 backward + three-layer gradcheck; 08-02 complex64 backward + perf baseline extension.

**In scope:**
- New Triton backward kernel `_butterfly_backward_kernel` in `torch_structured/_triton/butterfly/op.py` — `@triton.jit` 3-stage-tile structure mirroring `_butterfly_kernel`, but walking stages in reverse and accumulating `d_twiddle` via fp32 scratch + block-level reduce + single `tl.atomic_add` per program (SC#3).
- Replace the existing `_backward` callback body (currently at `torch_structured/_triton/butterfly/op.py:516-543`) — the new body:
  1. Allocates the fp32 activation trail: `trail = torch.empty(log_n * nblocks, batch_size, nstacks, n, dtype=torch.float32, device=...)` (or 2x that for IS_COMPLEX, via the `view_as_real` flatten).
  2. Re-runs the Triton forward stage-group launches writing each post-stage result into `trail[i]` (i indexes the stage, 0..log_n*nblocks-1).
  3. Allocates the fp32 `d_twiddle` scratch: `d_twiddle_scratch = torch.zeros_like(twiddle, dtype=torch.float32)` (or fp32-real-mirror shape for complex64; see D-50 below).
  4. Allocates `d_input` buffer: `d_input = torch.empty_like(input)`.
  5. Issues `ceil(log_n / 3)` backward stage-group launches per nblock in **reverse** order — `(9,10,11)` first, then `(6,7,8)`, then `(3,4,5)`, then `(0,1,2)` for `log_n=11`. Reverse nblock order across blocks. Each launch reads `trail[i-1]` as the input to the stage and `grad_out_current` as the gradient flowing back. After each backward launch the new `grad_out_current = d_input_partial` for the next reverse stage group.
  6. Returns `(d_twiddle_scratch.to(twiddle.dtype), d_input, None, None)` matching the 4 forward-input contract from D-47.
- Pre-wire `IS_COMPLEX: tl.constexpr` in 08-01 with `tl.static_assert(not IS_COMPLEX, ...)` gate (per D-41a — zero kernel-signature refactor between plans).
- Backend rollout — **hard switch on `TORCH_STRUCTURED_BACKEND=triton`** (no second knob, no opt-in env var):
  - As soon as Plan 08-01 ships, `BACKEND=triton` routes both forward AND backward through Triton.
  - The Phase 7 `torch.autograd.grad(_torch_ref.butterfly_multiply_torch(...))` callback body is **deleted** in Plan 08-01 and replaced with the new kernel-backed body.
  - `BACKEND=torch` still uses the torch oracle for both forward and backward (TRI-07 preserved).
  - Small-N fallback (D-42a) inherits from Phase 7: `if log_n <= 1: return _butterfly_multiply_torch(...)` in the wrapper. The backward callback also inherits — `log_n <= 1` routes through the torch oracle's autograd graph (D-49 below).
- Tests (extending `tests/test_butterfly_triton.py` from Phase 7):
  - **Plan 08-01:** `test_butterfly_backward_gradcheck_fp64` (the three-layer SC#1 gradcheck — fp64 `gradcheck` on n=4, batch=1, log_n=2 against `autograd.grad(butterfly_multiply_torch, ...)`); `test_butterfly_dinput_allclose_fp32` (`d_input` allclose at n=256, batch=8); `test_butterfly_dtwiddle_allclose_fp32` (`d_twiddle` allclose at n=512, batch=4096 within rtol=1e-3, atol=1e-4 — D-52); `test_butterfly_backward_no_cpp_symbol` (SC#4 — D-53).
  - **Plan 08-02:** `test_butterfly_backward_complex64` (`d_input` + `d_twiddle` allclose for complex64 at n=512, batch=4096 within rtol=1e-3, atol=1e-4 — SC#2 verbatim); extend the dense smoke + sparse comprehensive tiers (Phase 7 D-43a) to include backward at all log_n × dtype combos; extend `07-BASELINE.json` schema with backward p50/p95 entries.
- `_ops.py` resolver — no change. The `butterfly_multiply` resolver block at `_ops.py:204-228` already routes `BACKEND=triton` to the Triton `triton_op`. Phase 8 only changes the `_backward` callback body inside `_triton/butterfly/op.py` — the dispatch surface is untouched.

**Out of scope:**
- **5-stage tile** (deferred to Phase 9 per Phase 7 D-40). Phase 8's backward stays at 3-stage tile for kernel-shape symmetry with Phase 7's forward.
- **Save-during-forward activation strategy** (rejected in discussion — adds ~720MB at log_n=11/nblocks=2/batch=4096 and heavies `_setup_context`).
- **Fused forward-backward single-launch kernel** (rejected — CUDA's `butterfly_multiply_untied_forward_backward_max5_fast_cuda_kernel` at `csrc/cuda/butterfly_cuda.cu:497` is the pattern, but no Phase 6/7 Triton analog and harder to gradcheck; deferred to Phase 9 as a perf optimization candidate).
- **Gated opt-in via `TORCH_STRUCTURED_BACKEND_BW=triton`** (rejected — clean break, single backend selector).
- **bf16/fp16 backward** (TRI-FUT-01, deferred). Phase 8 is fp32 + complex64.
- **log_n > 11** — kernel works in principle; test surface and perf baseline only exercise up to 11 per ROADMAP SC#1.
- **`@triton.autotune` over `num_warps` / `TILE_N`** — fixed values from Phase 7 D-40d carry over; Phase 9 revisits.
- **Touching `csrc/butterfly.cpp` / `butterfly_cuda.cu`** — Phase 10 deletion candidates. Phase 8 verifies these are NOT invoked at runtime (SC#4 via D-53) but does not delete them.
- **Editing `_setup_context`** — stays as Phase 7 wrote it (saves `(twiddle, input)`, `ctx.increasing_stride`, `ctx.output_size`). Re-running the forward inside `_backward` only needs these four values.
- **Editing `_torch_ref/butterfly.py`** — the oracle stays untouched (TRI-07).

</domain>

<decisions>
## Implementation Decisions

### Activation strategy — materialize full per-stage trail once (User choice, locked)

- **D-49:** Backward callback **re-runs the Triton forward inside the backward callback** to materialize per-stage activations. Phase 7's `_setup_context` is NOT modified — it continues saving only `(twiddle, input)` and the two scalar flags. The new `_backward` body:
  ```python
  def _backward(ctx, grad_out):
      twiddle, input_ = ctx.saved_tensors
      increasing_stride, output_size = ctx.increasing_stride, ctx.output_size

      batch_size, nstacks, input_size = input_.shape
      nblocks, log_n = twiddle.shape[1], twiddle.shape[2]
      n = 1 << log_n

      # Small-N fallback (D-42a inheritance) — D-49b
      if log_n <= 1:
          twiddle_d = twiddle.detach().requires_grad_(True)
          input_d = input_.detach().requires_grad_(True)
          with torch.enable_grad():
              out = _butterfly_multiply_torch(twiddle_d, input_d, increasing_stride, output_size)
          grad_twiddle, grad_input = torch.autograd.grad(out, [twiddle_d, input_d], grad_out)
          return grad_twiddle, grad_input, None, None

      # Allocate fp32 trail: (log_n * nblocks, batch, nstacks, n)
      # IS_COMPLEX=True: trail dtype is fp32 but n axis is doubled (view_as_real flatten — D-44)
      # Phase 8 — recompute forward writing to trail[i] at end of each stage
      trail = _allocate_trail(twiddle, input_, log_n, nblocks, batch_size, nstacks, n, is_complex)
      _recompute_forward_into_trail(...)  # uses Phase 7's stage-group launches but redirects out_ptr

      # Allocate fp32 d_twiddle scratch (SC#3) + d_input buffer
      d_twiddle_scratch = torch.zeros_like(twiddle, dtype=torch.float32)  # IS_COMPLEX: 2x size via view_as_real (D-50)
      d_input = torch.empty_like(input_)

      # Walk reverse stage-groups, accumulating d_twiddle via atomicAdd, propagating d_input
      grad_current = _pad_grad(grad_out, n, output_size)  # mirror Phase 7's pad/slice
      for block in reversed(range(nblocks)):
          # ... reverse stage-group launches per block, alternating cur_increasing_stride
          # Each launch: reads trail[i-1], grad_current, twiddle; writes d_input_partial + atomic_add into d_twiddle_scratch
          pass

      d_twiddle = d_twiddle_scratch.to(twiddle.dtype)  # fp32 -> twiddle.dtype cast at the boundary (D-50)
      d_input = d_input[:, :, :input_size]  # trim back to input_size
      return d_twiddle, d_input, None, None
  ```
  Peak memory: `log_n * nblocks * batch * nstacks * n * 4 bytes` fp32 ≈ 88MB at `log_n=11, nblocks=2, batch=4096, nstacks=1` (manageable; well under the ~720MB save-during-forward alternative).

- **D-49a:** **Trail recomputation reuses Phase 7's stage-group launches.** The recompute path does NOT introduce a second forward kernel — it issues the same `_butterfly_kernel` launches Phase 7 already wrote, but writes each launch's output to `trail[i]` instead of ping-ponging between two output buffers. Concretely: factor the existing forward stage-group launch loop in `_triton/butterfly/op.py:322-501` into a helper `_run_forward_stage_groups(twiddle, input, ..., trail_out=None)` where `trail_out=None` preserves Phase 7 forward behavior (ping-pong) and `trail_out` not None writes each stage's output into `trail_out[stage_idx]` instead. Zero behavioral change to Phase 7 forward when called the normal way (verified by re-running Phase 7's `test_butterfly_eager_fp32` + comprehensive grid after refactor).

- **D-49b:** **Small-N fallback inherits from Phase 7 (D-42a).** When `log_n <= 1`, the backward callback delegates to `torch.autograd.grad(_butterfly_multiply_torch(...))` exactly as Phase 7's body does today. Rationale: at `log_n=1` there's a single butterfly stage — the Triton backward's trail allocation + atomicAdd machinery has higher overhead than the torch oracle's two-line autograd. Tests at `log_n=1` exercise this path explicitly.

### Backward kernel launch shape — mirror Phase 7 in reverse stage-group order (User choice, locked)

- **D-50:** **`ceil(log_n / 3)` backward stage-group launches per nblock, traversed in REVERSE order.** For `log_n=11`: launches walk stages `(9,10,11)` first, then `(6,7,8)`, then `(3,4,5)`, then `(0,1,2)`. Reverse `nblock` order across blocks. Same `TILE_N` schedule as Phase 7 D-40b (`TILE_N = 1 << (max(group_stages) + 1)`). Same 2-D grid `(n_row_tiles, batch_size * nstacks)`. Same `num_warps` schedule from D-40d (4 / 8 / 16 by `TILE_N` band). Zero structural divergence from Phase 7 forward — only the kernel body changes (running stages in reverse + accumulating `d_twiddle` via atomicAdd).
- **D-50a:** **Per-program `d_twiddle` reduce + single atomicAdd (SC#3 verbatim).** Within each program's tile of `TILE_N` consecutive elements × `(batch * nstacks)` row, the kernel:
  1. Loads input from `trail[i-1]`, grad_out_current, and twiddle into registers.
  2. Computes per-pair `d_twiddle_local = grad_out * conj(input)` (real path: drop conj) and `d_input_local = twiddle.T @ grad_out` for each of the 3 stages in the group, register-resident.
  3. **`tl.sum`-reduces `d_twiddle_local` across the batch-row dimension within the program** to a per-twiddle-element local accumulator.
  4. Single `tl.atomic_add` per program writes the reduced local into `d_twiddle_scratch[stack, block, stage, n//(2*stride), 2, 2]` at the appropriate offset.
  5. `tl.store` of `d_input_local` into `d_input` for that row tile.
  **Critical:** `tl.atomic_add` writes into the **fp32 scratch buffer**, never into a bf16/fp16 buffer (SC#3 verbatim). The final fp32→twiddle.dtype cast happens **outside the kernel** at the very end of the backward callback (`d_twiddle = d_twiddle_scratch.to(twiddle.dtype)`).
- **D-50b:** **Complex64 d_twiddle scratch via view_as_real flatten.** For `IS_COMPLEX=True`, `d_twiddle_scratch` is allocated as `torch.zeros((nstacks, nblocks, log_n, n//2, 2, 2, 2), dtype=torch.float32, device=...)` — the extra trailing `2` is the real/imag axis via `view_as_real` (Phase 4 D-44). The kernel's `tl.atomic_add` writes the real and imag parts as separate fp32 atomics into the corresponding slots; the final `.to(torch.complex64)` cast at the callback boundary uses `torch.view_as_complex(d_twiddle_scratch.contiguous())`.
- **D-50c:** **Complex64 conjugate in d_twiddle formula.** For complex inputs, `d_twiddle = grad_out * conj(input)` — the conjugate is applied via the 4-FMA pattern adapted for complex multiply with conjugate: `(a+bi) * conj(c+di) = (ac + bd) + (bc - ad)i`. The kernel branches on `IS_COMPLEX` (the same flag from Phase 7 D-44 / D-41a) to swap between the real path (`grad_out * input`) and the conjugate path. The complex backward formula is **load-bearing for the complex64 gradcheck** — getting the conjugate sign wrong silently passes fp32 tests but fails complex64 tests.

### Plan split — 2 plans by dtype, mirroring Phase 7 (User choice, locked)

- **D-51:** **Two plans split by dtype**, transcribing Phase 7's 07-01/07-02 shape (Phase 7 D-41):
  - **08-01: fp32 backward + three-layer gradcheck + d_input/d_twiddle correctness + SC#4 stack-trace verification.** Kernel includes `IS_COMPLEX: tl.constexpr` flag but the `IS_COMPLEX=True` branch contains only `tl.static_assert(not IS_COMPLEX, 'complex64 backward lands in 08-02')`. Backward callback asserts `input.dtype == torch.float32 and twiddle.dtype == torch.float32` (NOT complex64). All fp32 backward tests + SC#4 verification.
  - **08-02: complex64 backward + perf baseline extension.** Removes the `tl.static_assert` gate; implements the `IS_COMPLEX=True` branch verbatim per D-50b/D-50c (view_as_real flatten + 4-FMA conjugate). Wrapper gates `view_as_real(d_twiddle_scratch)` + `view_as_real(input)` + `view_as_real(grad_out)` with `assert input.is_contiguous()`. Extend Phase 7's `tests/test_butterfly_triton.py` complex64 surface with backward correctness. Extend `07-BASELINE.json` schema with backward p50/p95 entries at `log_n ∈ {8,9,10,11} × {fp32, complex64}`.
- **D-51a:** **Pre-wire IS_COMPLEX in 08-01 (mirrors D-41a verbatim).** Plan 08-01 writes the backward kernel signature with `IS_COMPLEX: tl.constexpr` already present and the `view_as_real` machinery in the callback already in place but gated by `assert not is_complex` until 08-02 removes the gate. **Zero kernel-signature refactor between plans — Plan 08-02 only adds code, never changes signatures.** Eliminates the integration risk of "the kernel that worked yesterday looks different today".

### Atomic-add noise envelope — rtol=1e-3, atol=1e-4 (User choice, locked)

- **D-52:** **d_twiddle gradcheck rtol=1e-3, atol=1e-4 at batch=4096 (fp32 and complex64).** Reuses the rtol/atol that SC#2 already locks for complex64 d_twiddle at batch=4096 — fp32 and complex64 share one envelope for downstream consistency. Calibrated for the `~sqrt(batch) * machine_eps_fp32 ≈ 1e-4` atomicAdd reorder noise at batch=4096.
- **D-52a:** **Three-layer gradcheck pattern (SC#1 verbatim).** Layer (a): fp64 `gradcheck` on the smallest case (n=4, batch=1, log_n=2, nstacks=1, nblocks=1) against `torch.autograd.grad(_butterfly_multiply_torch, ...)`. Layer (b): `torch.allclose(d_input_triton, d_input_torch_ref, rtol=1e-5, atol=1e-6)` at n=256, batch=8 (standard fp32 forward-correctness rtol — d_input is a sum of products, not an atomicAdd-noisy reduction). Layer (c): `torch.allclose(d_twiddle_triton, d_twiddle_torch_ref, rtol=1e-3, atol=1e-4)` at n=512, batch=4096 (the SC#3 atomicAdd path — looser envelope).
- **D-52b:** **Reference oracle is `torch.autograd.grad(_butterfly_multiply_torch, ...)`** (not the CUDA legacy backward). Mirrors TEST-02 verbatim: "Backward correctness validated via `gradcheck` against `autograd.grad(_torch_fw, ...)` — not against the CUDA reference." The CUDA legacy backward at `csrc/cuda/butterfly_cuda.cu:497` (`butterfly_multiply_untied_forward_backward_max5_fast_cuda_kernel`) is documented as a reference point in the comments (for the 5-stage future variant) but is NOT a numerical oracle.

### SC#4 verification — assert `_butterfly.so` symbols not invoked (User choice, locked)

- **D-53:** **Runtime assertion that no symbol from `_butterfly.so` is invoked during a full forward+backward step under `BACKEND=triton`.** The Plan 08-01 test `test_butterfly_backward_no_cpp_symbol`:
  ```python
  def test_butterfly_backward_no_cpp_symbol():
      torch_structured.set_backend('triton')
      # Build a model that uses Butterfly nn.Module
      m = Butterfly(in_size=16, out_size=16, bias=False, complex=False, increasing_stride=True).cuda()
      x = torch.randn(8, 16, requires_grad=True, device='cuda')
      # Probe BEFORE: snapshot torch.ops.torch_structured ops
      ops_before = set(dir(torch.ops.torch_structured))
      # Run forward + backward
      out = m(x)
      loss = out.sum()
      loss.backward()
      # Probe AFTER: no new C++ ops registered AND no _butterfly.so symbol invoked
      ops_after = set(dir(torch.ops.torch_structured))
      # Hard assertion: butterfly_multiply_fw / butterfly_multiply_bw from csrc/butterfly.cpp must not be invokable
      # Either by absence (if .so never loaded) OR by use_count check on torch.ops registry
      # The detailed probe form is up to the planner — the contract is: assert no csrc/butterfly.cpp symbol invoked
      assert not _was_butterfly_cpp_invoked(), \
          'BACKEND=triton must not invoke any symbol from csrc/butterfly.cpp at runtime (SC#4)'
  ```
  Concrete verification mechanism (planner's discretion): check `sys.modules` for `_butterfly` module not present, OR check `torch.ops.torch_structured.butterfly_multiply_fw._opname` registry call-count delta is zero across the forward+backward span, OR use Python's `sys.settrace` to monitor C-extension calls during `loss.backward()` (the test must NOT be marked slow — runs on every CI).
- **D-53a:** **Build-time guarantee is NOT the primary mechanism.** Phase 8 does NOT modify CI to skip `setup.py` extension build — the existing v1.2 milestone keeps `_butterfly.so` building as an opt-in fallback for CUDA-only users (DEPR-03). SC#4 is enforced as a runtime assertion that the path is not taken under `BACKEND=triton`, not by hiding the path. This composes correctly with TRI-07 (torch oracle stays as fallback) and DEPR-01 (Triton is the v1.2 default backend, CUDA available via `BACKEND=cuda`).

### Inherited from prior phases (NOT re-discussed — locked upstream)

- **D-54 (inherits Phase 4 D-01..D-03 / Phase 7 D-44):** Complex64 layout via `torch.view_as_real()` at the wrapper boundary (zero-copy on contiguous input). Twiddle layout `(nstacks, nblocks, log_n, n/2, 2, 2)` reinterpretable as `(nstacks, nblocks, log_n, n/2, 2, 2, 2)` real via the same view. Kernel uses `IS_COMPLEX: tl.constexpr` flag — same `@triton.jit` source specializes per dtype at JIT time. Phase 8 extends to the `d_twiddle_scratch` allocation: under `IS_COMPLEX=True`, allocate the scratch as fp32 with the doubled trailing axis (D-50b).
- **D-55 (inherits Phase 5 D-21, D-22 / Phase 7 D-45):** `_cuda_legacy/butterfly.py` already exists with try-import + sentinel pattern. `_has_cuda_legacy()` probe at `_ops.py` already exists. When `BACKEND=cuda` AND `_butterfly.so` is missing, `_ops.py:217-218` falls back to `_torch_ref` with the existing `log.warning` (D-22 asymmetric fallback). Phase 8 does NOT touch `_cuda_legacy/butterfly.py` or the cuda fallback wiring.
- **D-56 (inherits Phase 5 D-25, D-26 / Phase 7 D-46):** Consumer call sites use D-05 attribute access. The `Butterfly`, `ButterflyBmm`, `ButterflyUnitary`, `ButterflyBase4` nn.Modules already call `torch_structured._ops.butterfly_multiply` via the dispatch surface. Phase 8 does NOT refactor consumer code — switching to Triton backward happens transparently via the resolver `set_backend()`, and the `register_autograd` callback at `_triton/butterfly/op.py:546` continues to wire `_backward` into the autograd graph.
- **D-57 (inherits Phase 5/6 D-32 / Phase 7 D-47):** `register_autograd` + `register_fake` + `triton_op` skeleton — five-component pattern. Backward callback for butterfly is two-input: returns `(grad_twiddle, grad_input, None, None)`. Phase 8 swaps the **body** of `_backward` (the kernel-backed path) but keeps the **signature** (4-tuple return, `(twiddle, input)` saved tensors, `(increasing_stride, output_size)` scalar attributes). The number of returns must match the 4 forward inputs `(twiddle, input, increasing_stride, output_size)`; last two are `None` (non-tensor int/bool args).
- **D-58 (inherits Phase 6 D-39 / Phase 7 D-48):** `tests/conftest.py` `backend` fixture skip-gate already uses `_has_any_triton_kernel()` which iterates `butterfly_multiply`. No conftest changes needed.
- **D-59 (inherits Phase 7 D-42, D-42a):** Pad/trim and small-N fallback. Pad-on-input + trim-on-output stay in the wrapper (Phase 7's behavior preserved verbatim for forward, replicated in the backward callback for `grad_out` padding and `d_input` trimming). Small-N fallback `log_n <= 1` routes through the torch oracle's `autograd.grad` for BOTH forward and backward — backward inherits via D-49b.

### Claude's Discretion

Areas where Claude (planner / executor) has flexibility:
- **Exact SC#4 probe mechanism** in `test_butterfly_backward_no_cpp_symbol` — `sys.modules` check, `torch.ops` registry call-count delta, or `sys.settrace` C-extension monitor. Pick the cheapest that's deterministic across CI runs. Recommend `sys.modules['_butterfly']` absence check + assertion against `torch.ops.torch_structured.butterfly_multiply_fw` not being callable (the latter only registered when `_butterfly.so` loaded).
- **Trail buffer reuse vs. per-call allocation** — at the cost of one extra Python attribute on `ctx`, the trail buffer could be saved between forward calls. Phase 8 ships per-call allocation (simpler, matches CUDA pattern); Phase 9 perf gate may revisit if `torch.empty` overhead at log_n=11 dominates.
- **Per-stage-group `_backward_kernel` signature** — whether to use a single kernel with `STAGE_COUNT: tl.constexpr ∈ {1,2,3}` (uniform launches with degenerate last group, D-40d recommendation) or to have separate kernels per `STAGE_COUNT`. Recommend uniform `STAGE_COUNT` constexpr per Phase 7 D-40d.
- **Exact ordering of recompute-then-walk-back inside `_backward`** — could be (i) all recompute first then all backward, or (ii) recompute one group, do backward on that group, recompute prior group, etc. Recommend (i) — simpler, matches the CUDA legacy pattern, and the trail buffer is allocated once anyway.
- **Whether `d_twiddle_scratch` is allocated via `torch.zeros_like(twiddle, dtype=torch.float32)` or `torch.zeros((shape), dtype=torch.float32)` from explicit shape** — both work; pick `zeros_like` for consistency with the `d_input = torch.empty_like(input_)` pattern.
- **Order of nblock iteration** — reverse `for block in reversed(range(nblocks))` is mandatory (the chain rule walks the forward chain in reverse). Recommend explicit `range(nblocks - 1, -1, -1)` for readability over `reversed(range(...))`.
- **Whether `d_input` is allocated with `empty_like` or `zeros_like`** — `empty_like` is sufficient since every element of `d_input` is written by the final-stage-group backward launch. Recommend `empty_like`.
- **Whether to ping-pong d_input between two buffers across reverse stage-groups** — yes (mirrors Phase 7 forward ping-pong at `_triton/butterfly/op.py:322-501`); each backward stage-group writes its output `d_input` to the buffer the next reverse-stage-group reads from. Recommend ping-pong for memory efficiency.
- **Whether trail buffer is `torch.empty` or `torch.zeros`** — `empty` is sufficient (each element is written by the corresponding forward stage launch before being read by the backward stage launch). Recommend `empty`.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 8 charter
- `.planning/ROADMAP.md` §"Phase 8" — phase goal, depends on Phase 7, 4 success criteria, 2 plan slots
- `.planning/REQUIREMENTS.md` §"v1.2 Requirements" → TRI-04 (the sole REQ this phase covers — `butterfly_multiply` backward on Triton + fp32 scratch accumulator)
- `.planning/REQUIREMENTS.md` §"Correctness & Performance Gates" → TEST-02 (gradcheck against `autograd.grad(_torch_fw, ...)`, not CUDA reference — Phase 8 satisfies the backward-correctness half)
- `.planning/REQUIREMENTS.md` §"Traceability" — confirms TRI-04 mapped to Phase 8

### Phase 7 hand-off (LOCKED — load-bearing for Phase 8)
- `.planning/phases/07-butterfly-multiply-forward-triton/07-CONTEXT.md` — **CRITICAL.** D-40..D-48 (multi-launch 3-stage tile, 2-plan dtype split, IS_COMPLEX pre-wire, register_autograd two-input pattern, edge cases, tiered test surface, perf baseline). Phase 8 inherits all of these and extends to the backward direction.
- `.planning/phases/07-butterfly-multiply-forward-triton/07-01-PLAN.md` — Plan template for the forward-fp32 deliverable. Phase 8's 08-01 mirrors its task structure (kernel + autograd + tests + verification) with the backward-specific deliverables.
- `.planning/phases/07-butterfly-multiply-forward-triton/07-01-SUMMARY.md` — concrete forward kernel + wrapper code lines. The recompute-forward path (D-49a) reuses the launch loop documented here verbatim.
- `.planning/phases/07-butterfly-multiply-forward-triton/07-02-PLAN.md` — Plan template for complex64 forward. Phase 8's 08-02 mirrors its complex64-extension structure (gate removal + 4-FMA branch fill-in + perf baseline JSON).
- `.planning/phases/07-butterfly-multiply-forward-triton/07-02-SUMMARY.md` — complex64 view_as_real pattern + 4-FMA template lines. Phase 8's complex64 backward adapts the same template with the conjugate sign flip (D-50c).
- `.planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json` — perf baseline schema. Phase 8 extends with backward p50/p95 entries at the same log_n × dtype grid.
- `.planning/phases/07-butterfly-multiply-forward-triton/07-VERIFICATION.md` — verification report style. Phase 8 aims for the same `passed`-after-gap-closure pattern.

### Phase 4 hand-off (LOCKED — view_as_real + 4-FMA template)
- `.planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/04-COMPLEX-LAYOUT.md` — **CRITICAL.** D-01 `view_as_real` at wrapper boundary; D-02 wrapper template (copy verbatim for Plan 08-02 backward); D-03 twiddle layout invariant. Kernel-side `IS_COMPLEX: tl.constexpr` 4-FMA template. **Phase 8 adapts the 4-FMA template for the conjugate path** `(a+bi) * conj(c+di) = (ac + bd) + (bc - ad)i` (D-50c). Contiguity Gotcha (Pitfall 3) — Plan 08-02 callback MUST assert `input.is_contiguous() and twiddle.is_contiguous() and grad_out.is_contiguous()` before `view_as_real`.
- `.planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/04-CONTEXT.md` — D-04..D-08 (dispatch + set_backend); D-09..D-10 (`_torch_ref/` layout); D-11..D-12 (torch>=2.6, triton_op pattern); D-13 (register_autograd + register_fake); D-15 (deprecation plan); D-16 (CI cache)
- `.planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/04-DEPRECATION-PLAN.md` — Phase 10 implements; Phase 8's `BACKEND=cuda` fallback to `_torch_ref` uses `log.warning`, not `DeprecationWarning`

### Phase 5 hand-off (LOCKED — pattern source for two-input gradient backward)
- `.planning/phases/05-diag-mult-triton-port/05-CONTEXT.md` — D-21 (try-import + sentinel), D-22 (per-op asymmetric fallback), **D-26 (backward via `_torch_ref` oracle — Phase 8 evolves this to the Triton-native backward)**, D-27/D-28 (test surface pattern)
- `.planning/phases/05-diag-mult-triton-port/05-01-PLAN.md` — 7-task template; Phase 8's two plans each adopt a subset (kernel + autograd + tests, then complex extension + baseline)
- `.planning/phases/05-diag-mult-triton-port/05-01-SUMMARY.md` — concrete delta lines for the `register_autograd` callback pattern with conjugate-aware closed-form backward. Phase 5's diag_mult used direct Wirtinger formulas; Phase 8's butterfly backward uses kernel-backed direct gradients (D-50a/D-50c), not closed-form Python.

### Phase 6 hand-off (LOCKED — most recent atomic-free precedent for op kernel + test surface)
- `.planning/phases/06-hadamard-triton-port/06-CONTEXT.md` — D-31..D-39: single-pass shared-memory pattern (Phase 8 diverges to atomicAdd — hadamard backward had no twiddle reduction, butterfly backward has `d_twiddle` reduction across batch); D-37 test surface pattern (Phase 8 follows verbatim).
- `.planning/phases/06-hadamard-triton-port/06-01-SUMMARY.md` — `tl.debug_barrier()` lesson (Phase 8 backward avoids — register-resident tiles per D-40b carry through; no inter-stage barrier needed); the `normalize=False` default in `register_fake` lesson does NOT apply (no schema-default issue for `_backward` callback — its inputs are tensors).

### Research outputs (milestone-wide — load-bearing for Phase 8)
- `.planning/research/PITFALLS.md` §1 — **CRITICAL.** Complex64 in Triton: no `tl.complex64` dtype; `view_as_real` + 4-FMA + conjugate variant is the only viable path. Phase 8 d_twiddle complex path uses the conjugate 4-FMA per D-50c.
- `.planning/research/PITFALLS.md` §3 — Phase 8 challenge is "atomicAdd into a reduced fp32 buffer with deterministic precision envelope" (resolved via D-50a/D-52).
- `.planning/research/STACK.md` — `@triton.jit` + `wrap_triton` + `register_autograd` + `register_fake` API contract
- `.planning/research/ARCHITECTURE.md` — `_triton/<op>/op.py` layout pattern

### Project-level constraints
- `.planning/PROJECT.md` §"Current Milestone: v1.2" — `butterfly_multiply_torch` preserved as oracle + runtime fallback (TRI-07 already locked); v1.2 default backend = Triton (DEPR-01)
- `./CLAUDE.md` (project root) — `assert` for preconditions, no try/except in core lib (one exception: `_cuda_legacy/*.py` try-imports — documented honest-probe pattern from Phase 5 D-21)
- `/home/claroche/CLAUDE.md` (user-level) — `bd` for task tracking, NOT TaskCreate/TodoWrite

### Code-level references (read before editing)
- `torch_structured/_torch_ref/butterfly.py:1-34` — the pure-PyTorch oracle. Phase 8 backward gradcheck (D-52a) uses `torch.autograd.grad(butterfly_multiply_torch, ...)` as the numerical reference (TEST-02 verbatim).
- `torch_structured/_triton/butterfly/op.py:77-321` — Phase 7's `_butterfly_kernel` (`@triton.jit` forward kernel). Phase 8 inherits the launch shape (D-50); recompute path (D-49a) reuses the launches verbatim.
- `torch_structured/_triton/butterfly/op.py:322-501` — Phase 7's `butterfly_multiply` `@triton_op` wrapper with the stage-group launch loop and ping-pong buffer logic. Phase 8 factors this into `_run_forward_stage_groups(..., trail_out=None)` (D-49a).
- `torch_structured/_triton/butterfly/op.py:503-543` — Phase 7's `_setup_context` + `_backward` callback. Phase 8 **leaves `_setup_context` unchanged** and **replaces the body of `_backward`** (D-49 / D-57). The `register_autograd` registration line at `:546` stays as-is.
- `torch_structured/_triton/butterfly/op.py:549-567` — Phase 7's `_butterfly_multiply_fake` meta kernel. Phase 8 does NOT touch.
- `torch_structured/_cuda_legacy/butterfly.py` — already exists. Phase 8 does NOT touch.
- `torch_structured/_ops.py:204-228` — existing `butterfly_multiply` resolver block. Phase 8 does NOT touch — the dispatch surface is unchanged; only the `_backward` callback body in `_triton/butterfly/op.py` is replaced.
- `csrc/cuda/butterfly_cuda.cu:497-540` — `butterfly_multiply_untied_forward_backward_max5_fast_cuda_kernel` and `b_untied_forward_backward_shared_twiddle` template — the CUDA reference for the forward-backward-fused pattern. Phase 8 ports the **3-stage** version (not 5-stage); the math at `:453-467` (`d_twiddle_val[0] += grad_val * conj_wrapper(input_val); gpuAtomicAdd(&s_d_twiddle[step][0][...], d_twiddle_val[0])`) is the algorithmic blueprint adapted to Triton's `tl.atomic_add` into the **per-program reduce + single atomic** path (D-50a).
- `tests/test_butterfly_triton.py` — Phase 7's test file. Phase 8 extends with backward tests; same parametrize axes (log_n, nstacks, nblocks, increasing_stride, output_size, dtype), same dense smoke / sparse comprehensive tiered structure (D-43a inheritance).
- `tests/test_butterfly.py:234` (or equivalent line) — `Butterfly(complex=True)` unitary test. Phase 7 already verifies this on the Triton forward; Phase 8 verifies it ALSO passes when backward is invoked (e.g., a backward through a `Butterfly` nn.Module on a complex64 input produces correct gradients via the new Triton path).
- `tests/test_diag_mult.py:1-119` — Phase 5 test skeleton (closed-form backward correctness). Phase 8 uses the same allclose pattern with looser tolerance for d_twiddle (D-52).
- `tests/conftest.py` — `backend` fixture (Phase 6 D-39 widened). Phase 8 does NOT touch.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`butterfly_multiply_torch` at `_torch_ref/butterfly.py:13-33`** — the verbatim oracle. Phase 8 uses `torch.autograd.grad(butterfly_multiply_torch(twiddle, input, ...), [twiddle, input], grad_out)` as the numerical reference for d_input and d_twiddle correctness (TEST-02). Also the small-N fallback target via D-49b (when `log_n <= 1`).
- **Phase 7's `_butterfly_kernel` at `_triton/butterfly/op.py:77-321`** — the forward Triton kernel. Phase 8 recompute path (D-49a) reuses this kernel with the same launch shape but writes each stage's output to `trail[i]` instead of ping-ponging — the kernel itself is unchanged.
- **Phase 7's stage-group launch loop at `_triton/butterfly/op.py:322-501`** — the wrapper that issues `ceil(log_n / 3)` launches per nblock. Phase 8 factors this into a helper with an optional `trail_out` parameter (D-49a).
- **Phase 7's `register_autograd` callback wiring at `_triton/butterfly/op.py:503-546`** — `_setup_context` + `_backward` + `butterfly_multiply.register_autograd(_backward, setup_context=_setup_context)`. Phase 8 replaces ONLY the body of `_backward` (`:516-543`); `_setup_context` and the `register_autograd` registration line stay as-is.
- **Phase 7's small-N fallback at `_triton/butterfly/op.py` wrapper** — `if log_n <= 1: return _butterfly_multiply_torch(...)`. Phase 8's backward callback adds a symmetric small-N branch that routes through `torch.autograd.grad(_butterfly_multiply_torch(...))` for `log_n <= 1`.
- **`_cuda_legacy/butterfly.py`** — already exists. Try-import + sentinel pattern from Phase 5 D-21. Phase 8 does NOT touch.

### Established Patterns
- **3-stage register-resident tile** (Phase 7 D-40b) — backward inherits the same tile width schedule (`TILE_N = 1 << (max(group_stages) + 1)`) and the same `num_warps` schedule (D-40d: 4 / 8 / 16 by `TILE_N` band).
- **2-D grid `(n_row_tiles, batch_size * nstacks)`** (Phase 7 D-40c) — backward uses the identical grid shape per stage-group launch.
- **`IS_COMPLEX: tl.constexpr` pre-wire pattern** (Phase 7 D-41a) — Plan 08-01 writes the backward kernel with the `IS_COMPLEX` flag and the `view_as_real` machinery in the callback already in place but gated; Plan 08-02 fills in the conjugate-4-FMA branch.
- **`register_autograd` + `register_fake` + `triton_op`** (Phase 4 D-13 / Phase 5 D-32) — Phase 7 wired this for butterfly. Phase 8 only touches the `_backward` callback body, not the registrations.
- **Tiered test surface** (Phase 7 D-43a) — dense smoke (~5-10 cases per dtype) + sparse comprehensive (`@pytest.mark.slow`, hundreds of cases). Phase 8 adds backward tests at the same tier structure.
- **D-05 attribute access** — `Butterfly`, `ButterflyBmm`, etc. nn.Modules already call `torch_structured._ops.butterfly_multiply`. Phase 8 does NOT refactor consumer code — switching to Triton backward happens transparently via the `register_autograd` callback already wired in Phase 7.

### Integration Points
- **`torch_structured.butterfly` legacy package nn.Modules** — `Butterfly`, `ButterflyBmm`, `ButterflyUnitary`, `ButterflyBase4` already route through `torch_structured._ops.butterfly_multiply`. Phase 8 verifies that `loss.backward()` on these modules under `BACKEND=triton` produces correct gradients via the new Triton path — and verifies SC#4 (no C++ symbol invoked).
- **Phase 7's existing `tests/test_butterfly_triton.py`** — Phase 8 extends this file with backward tests. Same parametrize axes and tiered structure (D-43a inheritance). No new test file needed.
- **Phase 7's `07-BASELINE.json` schema** — Phase 8 extends with backward p50/p95 entries at the same `log_n × dtype` grid. Phase 9 perf gate consumes both directions.
- **`_ops.py:204-228` resolver block** — already routes `BACKEND=triton` to the `triton_op`. Phase 8 does NOT touch.

</code_context>

<specifics>
## Specific Ideas

- **The `_backward` callback body replacement is THE Phase 8 deliverable.** Phase 7's body at `_triton/butterfly/op.py:516-543` calls `torch.autograd.grad(_butterfly_multiply_torch(...), [twiddle_d, input_d], grad_out)`. Phase 8's body recomputes the forward into a trail buffer, then issues reverse stage-group launches accumulating `d_twiddle` via atomicAdd into an fp32 scratch buffer. Everything else (`_setup_context`, `register_autograd` registration, `register_fake`, the forward kernel, the `_ops.py` resolver, the `Butterfly` nn.Module surface) stays unchanged.

- **Recompute-then-walk-back template (illustrative — planner's exact form):**
  ```python
  def _backward(ctx, grad_out):
      twiddle, input_ = ctx.saved_tensors
      increasing_stride, output_size = ctx.increasing_stride, ctx.output_size
      batch_size, nstacks, input_size = input_.shape
      nblocks, log_n = twiddle.shape[1], twiddle.shape[2]
      n = 1 << log_n
      is_complex = twiddle.is_complex()

      # Small-N fallback (D-49b inheritance)
      if log_n <= 1:
          twiddle_d = twiddle.detach().requires_grad_(True)
          input_d = input_.detach().requires_grad_(True)
          with torch.enable_grad():
              out = _butterfly_multiply_torch(twiddle_d, input_d, increasing_stride, output_size)
          gt, gi = torch.autograd.grad(out, [twiddle_d, input_d], grad_out)
          return gt, gi, None, None

      # Allocate fp32 trail + d_twiddle scratch
      trail_dtype = torch.float32  # IS_COMPLEX path: trail width = n*2 via view_as_real flatten
      trail_n = n * (2 if is_complex else 1)
      trail = torch.empty(log_n * nblocks, batch_size, nstacks, trail_n, dtype=trail_dtype, device=input_.device)
      d_twiddle_scratch_shape = twiddle.shape + ((2,) if is_complex else ())
      d_twiddle_scratch = torch.zeros(d_twiddle_scratch_shape, dtype=torch.float32, device=twiddle.device)
      d_input = torch.empty_like(input_)

      # Recompute forward into trail (reuse Phase 7 stage-group launches)
      _run_forward_stage_groups(twiddle, input_, increasing_stride, output_size, trail_out=trail, is_complex=is_complex)

      # Pad grad_out from output_size up to n
      grad_full = F.pad(grad_out, (0, n - output_size)) if output_size < n else grad_out

      # Walk reverse stage-groups, atomicAdd into d_twiddle_scratch + write d_input
      _run_backward_stage_groups(twiddle, trail, grad_full, d_twiddle_scratch, d_input,
                                 nblocks, log_n, increasing_stride, is_complex)

      # Cast fp32 scratch back to twiddle dtype + complex-view
      if is_complex:
          d_twiddle = torch.view_as_complex(d_twiddle_scratch.contiguous())
      else:
          d_twiddle = d_twiddle_scratch.to(twiddle.dtype)
      d_input = d_input[:, :, :input_size]
      return d_twiddle, d_input, None, None
  ```

- **Per-program d_twiddle reduce + atomic (illustrative kernel body sketch):**
  ```python
  @triton.jit
  def _butterfly_backward_kernel(
      twiddle_ptr, trail_in_ptr, grad_in_ptr,
      d_twiddle_scratch_ptr, d_input_out_ptr,
      n, nstacks, block_idx, nblocks,
      STAGE_START: tl.constexpr, STAGE_COUNT: tl.constexpr,
      INCREASING_STRIDE: tl.constexpr, LOG_N: tl.constexpr,
      IS_COMPLEX: tl.constexpr, TILE_N: tl.constexpr,
  ):
      # 1. Load tile from trail_in_ptr (post-stage activation, INPUT to this backward step)
      # 2. Load tile from grad_in_ptr (gradient flowing into this stage from "later" stages)
      # 3. For stage in reversed(stage_range_in_group):
      #    - Load 2x2 twiddle for stage from twiddle_ptr
      #    - Compute d_input_local += twiddle.T @ grad   (or complex 4-FMA conjugate)
      #    - Compute d_twiddle_local = grad * conj(input)  (or real * input)
      #    - REDUCE d_twiddle_local across the batch axis within this program (tl.sum)
      #    - SINGLE tl.atomic_add per program into d_twiddle_scratch at the appropriate offset
      # 4. tl.store d_input_local into d_input_out_ptr (no atomic — each program writes a disjoint row tile)
  ```

- **Complex64 conjugate path (D-50c, illustrative):** For `IS_COMPLEX=True`, `d_twiddle = grad * conj(input)` translates to:
  ```
  out_re = grad_re * input_re + grad_im * input_im    (= grad * conj(input) .real)
  out_im = grad_im * input_re - grad_re * input_im    (= grad * conj(input) .imag)
  ```
  Note the sign flip on `out_im` compared to the forward 4-FMA (which had `out_im = a_re * c_im + a_im * c_re`). Getting this sign wrong silently passes fp32 tests (where input.imag is zero) but breaks complex64 tests.

- **SC#4 verification mechanism (D-53, illustrative):**
  ```python
  def test_butterfly_backward_no_cpp_symbol():
      import torch_structured as ts
      from torch_structured.butterfly import Butterfly
      ts.set_backend('triton')

      # Build model + run forward+backward
      m = Butterfly(in_size=16, out_size=16, bias=False, complex=False).cuda()
      x = torch.randn(8, 16, requires_grad=True, device='cuda')
      loss = m(x).sum()
      loss.backward()

      # SC#4: no symbol from csrc/butterfly.cpp invoked. Mechanism (planner's discretion):
      # (option A — recommended) Assert _butterfly module not loaded.
      assert '_butterfly' not in sys.modules, \
          'SC#4: csrc/butterfly.cpp must not be loaded under BACKEND=triton'
      # (option B — alternative) Assert torch.ops.torch_structured.butterfly_multiply_fw not callable.
      # Either form is acceptable; planner picks the one that's most deterministic in the test env.
  ```

- **Plan structure transcription from Phase 7:** Plan 08-01 transcribes Phase 7 07-01's task structure verbatim with the backward-specific deliverables substituted. Plan 08-02 transcribes 07-02's structure — gate removal + branch fill-in + complex64 test extension + perf baseline JSON extension.

- **Forward-backward asymmetry: trail buffer is fp32, even when twiddle is fp32.** The trail just stores forward activations — no atomicAdd, no precision concern. The fp32 typing is for storage consistency and avoids a dtype mismatch when reading into the backward kernel under `BACKEND=cuda` (where the kernel expects fp32). For complex64, the trail is stored as fp32 via the `view_as_real` flatten — same trick as Phase 7's forward complex path.

</specifics>

<deferred>
## Deferred Ideas

- **5-stage tile backward** — Phase 9 (TEST-04 perf gate) decides whether to land the 5-stage CUDA-parity variant. Phase 8 stays at 3-stage.
- **Save-during-forward activation strategy** — rejected for Phase 8 (~720MB at log_n=11/nblocks=2/batch=4096). May revisit in Phase 9 if recompute overhead dominates.
- **Fused forward-backward single-launch kernel** — the CUDA reference pattern at `csrc/cuda/butterfly_cuda.cu:497`. Deferred to Phase 9 as a perf optimization candidate; Phase 8 stays with the simpler recompute-then-walk-back structure.
- **Gated opt-in via `TORCH_STRUCTURED_BACKEND_BW=triton`** — rejected; clean hard switch on `BACKEND=triton`.
- **bf16 / fp16 backward** (TRI-FUT-01). Phase 8 is fp32 + complex64. The fp32 scratch accumulator pattern (SC#3) is the foundation that bf16/fp16 backward will build on (atomicAdd into fp32, cast at boundary).
- **`log_n > 11`** — kernel works in principle for any `log_n`; test surface and perf baseline only exercise up to 11 per ROADMAP SC#1.
- **`_butterfly.so` build verification in CI** — Phase 8 verifies the runtime path is NOT taken under `BACKEND=triton` (SC#4 via D-53) but does NOT modify the CI build matrix. Phase 9 may add a CI matrix entry that explicitly builds + tests the `.so` under `BACKEND=cuda`; Phase 8 stays out.
- **`@triton.autotune` over `num_warps` / `TILE_N`** — fixed values from Phase 7 D-40d carry over. Phase 9 may revisit.
- **CUDA backend in `backend` conftest fixture** — Phase 5 D-30 / Phase 6 deferred the `"cuda"` param to Phase 9 per TEST-03. Phase 8 inherits this deferral.
- **Build-time guarantee for SC#4** — Phase 8 deliberately keeps `_butterfly.so` building (DEPR-03 / TRI-07 compose). The runtime assertion at D-53 is the primary mechanism. Phase 9/10 may revisit when the deprecation cadence completes.
- **Trail buffer reuse across calls** — Phase 8 ships per-call allocation. Phase 9 perf gate may revisit if `torch.empty(log_n * nblocks, batch, nstacks, n)` allocation dominates at log_n=11.

### Reviewed Todos (not folded)
None — no pending todos surfaced for Phase 8.

</deferred>

---

*Phase: 8-butterfly_multiply Backward (Triton)*
*Context gathered: 2026-05-28*
