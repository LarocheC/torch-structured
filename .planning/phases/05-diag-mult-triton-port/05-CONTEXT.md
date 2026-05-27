# Phase 5: diag_mult Triton Port - Context

**Gathered:** 2026-05-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Port the `cycle_mult` primitive (cyclic-shifted pointwise multiply — the kernel underlying `LDRSubdiagonalC`'s `cycle_down_mult`) from C++/CUDA to Triton, expose it as `_ops.diag_mult(subdiag, v, shift_subdiag, shift_v)` (forward + backward, fp32 + complex64), refactor `structured/krylov.py` to consume the dispatch single import point, and delete the Phase 4 demonstrator (`_demo_identity_op`). This phase validates that the Phase 4 dispatch + autograd plumbing carries a real Triton kernel end-to-end.

**In scope:**
- New `torch_structured/_triton/diag_mult/` package with the `@triton.jit` kernel, `@triton_op` wrapper, `register_autograd` backward, `register_fake` meta kernel.
- New `torch_structured/_torch_ref/diag_mult.py` reference impl (pure-PyTorch oracle).
- New `torch_structured/_cuda_legacy/diag_mult.py` thin try-import wrapper around `torch_structured._diag_mult_cuda.cycle_mult` (no-op if `.so` is absent).
- `_ops.py` resolver edits: bind `diag_mult` from chosen backend; light up `_has_triton_kernel("diag_mult")` for the auto path; light up the corresponding D-08 INFO heads-up dormant in Phase 4.
- `_ops.py` cleanup: delete `_demo_identity_op` + `_demo_identity_kernel` + `_setup_context` + `_backward` + the `register_fake` shim (per Phase 4 D-13).
- `structured/krylov.py` refactor: delete `CycleDownMultCuda(torch.autograd.Function)`; replace `cycle_down_mult = CycleDownMultCuda.apply` with inline `_ops.diag_mult(subdiag, v, 0, -1)` call sites; the existing `subdiag_mult_cuda` autograd path now relies on `_ops.diag_mult`'s `register_autograd`.
- `tests/conftest.py` extension: `backend` fixture params widen from `["torch"]` to `["torch", "triton"]` (skip "triton" when `_has_triton_kernel("diag_mult")` is False — i.e., CPU-only runners).
- New `tests/test_diag_mult.py` (cross-backend correctness + gradcheck + complex64).
- `tests/test_dispatch.py` adjustment: remove the 5 demonstrator-specific tests; keep any cross-cutting dispatch tests (set_backend round-trip, env-var override) if they still apply.

**Out of scope:**
- `hadamard` port (Phase 6).
- `butterfly_multiply` forward/backward (Phases 7, 8).
- The `subdiagKrylov` op in `csrc/diag_mult/diag_mult_cuda.cpp:18` — pybind-exported but has zero Python consumers (dead code; deletion deferred to Phase 10).
- bf16/fp16 dtype matrix (TRI-FUT-01).
- Autotune over `BLOCK_SIZE` / `num_warps` — pointwise kernel; a hand-picked default is fine (autotune belongs in butterfly forward / Phase 7).
- Resurrecting the build of `_diag_mult_cuda.so` for environments where it's absent — fall back to `_torch_ref` instead (see D-22).
- Touching `LDRSubdiagonal` / `LDRSubdiagonalC` / `LDRTridiagonal` (`structured/layers.py:200-268`) public API — krylov.py refactor is internal.

</domain>

<decisions>
## Implementation Decisions

### API surface of `_ops.diag_mult` (Claude's discretion, locked)

- **D-19:** `_ops.diag_mult` is the **generic `cycle_mult` primitive**: `diag_mult(subdiag: Tensor, v: Tensor, shift_subdiag: int, shift_v: int) -> Tensor`. Pointwise formula matching `csrc/diag_mult/diag_mult_cuda_kernel.cu:8`: `out[pos] = subdiag[(pos + shift_subdiag + N) % N] * v[(pos + shift_v + N) % N]`, with `N = v.size(-1)` and broadcasting over leading dims.
- **D-19a:** Krylov.py's forward (`shift=(0,-1)`) and backward (`shift=(0,-1)` and `shift=(1,1)`) all call this one op with different shift args. Triton specializes per `(IS_COMPLEX,)` constexpr but treats the int shifts as runtime args (no per-shift JIT explosion).
- **D-19b:** `subdiag` may be 1-D `(N,)` (broadcast across batch) or batched `(*, N)` matching `v`'s leading shape — auto-detected at the wrapper boundary, mirroring the existing C++ `batchedSubdiag` flag.

### Complex64 representation (Claude's discretion, locked — inherits Phase 4 D-01..D-03)

- **D-20:** `diag_mult` supports **complex × complex (full 4-FMA)**: both `subdiag` and `v` may be `complex64`. Mixed dtypes (e.g., real subdiag, complex v) are rejected at the wrapper with an `assert subdiag.dtype == v.dtype` — keeps the kernel surface minimal and matches Phase 7's planned butterfly behavior.
- **D-20a:** Wrapper boundary follows the canonical template from `04-COMPLEX-LAYOUT.md`: `view_as_real` on both inputs (with `.contiguous()` guard per Pitfall 3), `IS_COMPLEX: tl.constexpr` to the kernel, `view_as_complex(out.contiguous())` on the way back.
- **D-20b:** Inside the kernel, the `IS_COMPLEX = True` branch implements `(a+bi)(c+di) = (ac - bd) + (ad + bc)i` as four FMAs against the trailing-2 layout. The same `@triton.jit` source compiles into both real-only and complex specializations — exactly the pattern Phase 4 demonstrator validated.
- **D-20c:** `_torch_ref/diag_mult.py` accepts complex inputs natively (no `view_as_real` games in the reference — `torch.roll(v, -shift_v)` works on complex). The kernel-side complex math is what's being validated against this oracle.

### CUDA legacy backend wire-up (Claude's discretion, locked)

- **D-21:** `_cuda_legacy/diag_mult.py` performs a top-of-module **try-import** of `torch_structured._diag_mult_cuda` (the pybind module from `csrc/diag_mult/`). On `ImportError`, it defines the module-level `diag_mult` symbol as `None` (rather than raising). The new `_has_cuda_legacy_diag_mult()` probe in `_ops.py` returns `True` iff that symbol is non-None — honors Phase 4 CHECKER B3 honest-resolver pattern.
- **D-22:** When `set_backend("cuda")` is requested AND `_has_cuda_legacy_diag_mult()` is False, the resolver falls back to `"torch"` for `diag_mult` (binding it to `_torch_ref.diag_mult`), emits a `log.warning("set_backend('cuda') requested but _diag_mult_cuda not built; falling back to torch_ref for diag_mult")`, and **does not affect other ops** — `butterfly_multiply` may still resolve to `cuda` if `_has_cuda_legacy()` is True. The resolver becomes per-op aware.
- **D-22a:** This means `_BACKEND` (the module-level string) may need to become per-op (e.g., `_BACKENDS: dict[str, str]` keyed by op name) — OR we keep `_BACKEND` as a coarse global and accept that it reflects the "primary" backend (the one chosen for `butterfly_multiply`, the heaviest op). **Planner decides:** the per-op dict is more honest but a bigger refactor; the coarse global is simpler. Recommend coarse + a `log.info(per-op bindings)` line at import that prints the actual map.
- **D-23:** SC#3 (\"CUDA `_diag_mult.so` path remains selectable\") is satisfied: when the `.so` IS built, `BACKEND=cuda` produces bit-exact existing behavior; when it isn't, the fall-back is transparent and explicit. No `setup.py` change in Phase 5 — the existing conditional build (`setup.py:98-99`) already handles \"build if dir exists\".

### krylov.py consumer refactor (Claude's discretion, locked)

- **D-24:** Delete `CycleDownMultCuda(torch.autograd.Function)` at `structured/krylov.py:325-339` and the module-level `cycle_down_mult = CycleDownMultCuda.apply` (line 339). The hand-rolled autograd is now redundant — `_ops.diag_mult.register_autograd` handles it.
- **D-25:** Replace `cycle_down_mult` call sites:
  - `subdiag_linear_map_cuda` (line 342-344): `return lambda v: cycle_down_mult(subdiag_extended, v)` → `return lambda v: torch_structured._ops.diag_mult(subdiag_extended, v, 0, -1)` (attribute-access form per Phase 4 D-05).
  - The `from torch_structured import _diag_mult_cuda as diag_mult_cuda` try-import at line 22 → removed (no longer needed; `_ops.diag_mult` covers it).
- **D-26:** The `register_autograd` callback for `_ops.diag_mult` derives gradients from the same `cycle_mult` primitive with adjusted shifts:
  - `grad_subdiag` at index `(i - shift_subdiag) mod N` accumulates `grad_out[i] * v[(i + shift_v) mod N]` — implementable as `_torch_ref.diag_mult(grad_out, v, -shift_subdiag, shift_v - shift_subdiag).sum_over_batch_dims()`.
  - `grad_v` at index `(i - shift_v) mod N` accumulates `grad_out[i] * subdiag[(i + shift_subdiag) mod N]` — implementable as `_torch_ref.diag_mult(subdiag, grad_out, shift_subdiag - shift_v, -shift_v)`.
  - **Planner verifies via fp64 `gradcheck`** that this formula is correct against the existing manual backward at `krylov.py:336` (which only covered the `shift=(0,-1)` specialization).

### Demonstrator cleanup (Phase 4 D-13 follow-through)

- **D-27:** Delete from `torch_structured/_ops.py`: the `_demo_identity_kernel` `@triton.jit` function (lines 225-233), the `_demo_identity_op` `@triton_op` (lines 236-278), `_setup_context` (lines 281-284), `_backward` (lines 287-289), the `register_autograd` line (292), and the `register_fake` block (295-304). Keep the module-level `import triton` and `import triton.language as tl` and `from torch.library import triton_op, wrap_triton` — Phase 5+ kernel imports need them.
- **D-28:** Delete `tests/test_dispatch.py`'s 5 demonstrator tests (`test_demo_identity_*`). Replace with cross-cutting dispatch tests if any remain useful (set_backend round-trip, env-var override, ValueError on unknown backend, B3 honest-probe). If nothing is portable, delete the file. The replacement `tests/test_diag_mult.py` carries the kernel correctness load.

### Test surface (Claude's discretion within scope)

- **D-29:** New `tests/test_diag_mult.py` contains: (a) `test_diag_mult_eager_fp32` cross-backend allclose vs `_torch_ref`; (b) `test_diag_mult_eager_complex64` same; (c) `test_diag_mult_gradcheck_fp64_real` against `autograd.grad(_torch_ref.diag_mult, ...)` per TRI-spec gradcheck pattern; (d) `test_diag_mult_gradcheck_fp64_complex` same with complex inputs; (e) `test_diag_mult_shift_grid` covering `shift_subdiag, shift_v ∈ {-1, 0, 1}` (the only shifts krylov.py actually uses, but worth pinning the contract).
- **D-30:** `tests/conftest.py` `backend` fixture widens to `params=["torch", "triton"]`. Phase 5 conditionally skips `"triton"` when `not torch_structured._ops._has_triton_kernel("diag_mult")` (CPU runners, no-Triton envs). The `"cuda"` param is **deferred to Phase 9** per the milestone-wide TEST-03 (full backend axis at integration hardening).

### Claude's Discretion

All four selected gray areas were resolved as Claude's discretion at the user's request:
- API surface: generic `cycle_mult(subdiag, v, shift_subdiag, shift_v)` primitive.
- Complex64: full complex × complex (4-FMA) via Phase 4's `view_as_real` + `IS_COMPLEX` template.
- CUDA legacy: try-import + honest probe + transparent fallback.
- krylov.py refactor: delete `CycleDownMultCuda`; inline `_ops.diag_mult` call sites.

Remaining planner-flexible items:
- Exact `BLOCK_SIZE` for the Triton kernel (recommend 1024, same as Phase 4 demonstrator — pointwise kernels are not block-size sensitive at these sizes).
- Whether to introduce a `_BACKENDS: dict[str, str]` per-op resolution map or keep the coarse `_BACKEND` global (recommend coarse — simpler; revisit in Phase 7 if needed).
- The exact wording of the new `log.warning` when `cuda` falls back to `torch_ref` for `diag_mult` (planner tightens copy).
- Whether `tests/test_dispatch.py` is deleted outright or kept as a thin set_backend smoke test (planner's call after seeing what's portable).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 5 charter
- `.planning/ROADMAP.md` §"Phase 5" — phase goal, depends-on Phase 4, 3 success criteria, single plan slot
- `.planning/REQUIREMENTS.md` §"v1.2 Requirements" → TRI-01 only (sole REQ this phase covers)
- `.planning/REQUIREMENTS.md` §"Traceability" — confirms TRI-01 mapped to Phase 5

### Phase 4 hand-off (locked, READ BEFORE PLANNING)
- `.planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/04-CONTEXT.md` — D-01..D-16 are the dispatch + complex layout + autograd contract. Phase 5 inherits all of them; D-04..D-08, D-09..D-10, D-11..D-12, D-13 are load-bearing.
- `.planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/04-COMPLEX-LAYOUT.md` — canonical `view_as_real` + `IS_COMPLEX: tl.constexpr` + 4-FMA template. Phase 5 implements this verbatim for `diag_mult`.
- `.planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/04-DEPRECATION-PLAN.md` — does NOT apply yet (Phase 10), but Phase 5's cuda-legacy fallback log.warning should NOT be confused with the DeprecationWarning that DEPR-02 reserves for Phase 10.
- `.planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/04-VERIFICATION.md` — confirms which Phase 4 artifacts exist and how they wire (especially the `_ops.py` line-number map at SC2/SC3 evidence).

### Research outputs (milestone-wide)
- `.planning/research/SUMMARY.md` §"Architecture Approach" + §"Critical Pitfalls" — anchor for kernel + dispatch design
- `.planning/research/STACK.md` §"triton_op" + §"version matrix" — PyTorch >=2.6 floor, wrap_triton requirement
- `.planning/research/PITFALLS.md` §1 (complex layout) + §3 (triton_op only viable wrapper) — load-bearing for the diag_mult Triton kernel
- `.planning/research/ARCHITECTURE.md` `_triton/<op>/{forward,backward,op}.py` layout — Phase 5 establishes `_triton/diag_mult/`

### Project-level constraints
- `.planning/PROJECT.md` §"Current Milestone: v1.2" — parallel paths, `butterfly_multiply_torch` preserved (same pattern applies: `diag_mult_torch` preserved as runtime fallback)
- `CLAUDE.md` (project root) — `assert` for preconditions, no exceptions; pyproject.toml as build SoT
- `/home/claroche/CLAUDE.md` (user-level) — beads (`bd`) for task tracking, NOT TaskCreate/TodoWrite

### Code-level references (read before editing)
- `csrc/diag_mult/diag_mult_cuda.cpp:5-16` — `cycle_mult` op signature + `batchedSubdiag` detection (the C++ contract the Triton port matches)
- `csrc/diag_mult/diag_mult_cuda_kernel.cu:1-17` — pointwise formula `d_Sub[(pos + shiftSubdiag + N) % N] * d_Src[(pos + shiftV + N) % N]`
- `torch_structured/_ops.py:42-213` — current resolver implementation (Phase 4); D-22/D-22a edit the resolver to be per-op aware
- `torch_structured/_ops.py:216-304` — demonstrator op to delete per D-27
- `torch_structured/structured/krylov.py:21-24` — try-import to remove per D-25
- `torch_structured/structured/krylov.py:325-339` — `CycleDownMultCuda` + `cycle_down_mult` to delete per D-24/D-25
- `torch_structured/structured/krylov.py:342-344` — `subdiag_linear_map_cuda` call site to rewrite per D-25
- `torch_structured/_cuda_legacy/__init__.py` + `_cuda_legacy/butterfly.py` — pattern to mirror for `_cuda_legacy/diag_mult.py`
- `torch_structured/_torch_ref/__init__.py` + `_torch_ref/butterfly.py` — pattern to mirror for `_torch_ref/diag_mult.py`
- `tests/conftest.py:1-22` (Phase 4) — `backend` fixture to extend per D-30
- `tests/test_dispatch.py` — to delete or trim per D-28
- `setup.py:96-110` — existing `_diag_mult_cuda` conditional build (no edits in Phase 5; D-21/D-23 rely on this staying as-is)

### Prior-art / known issues
- `.planning/quick/260419-p27-extend-recurrent-poc-torch-compile-track/260419-p27-SUMMARY.md` — `register_fake` is the fix; Phase 5's `_triton/diag_mult/op.py` MUST include `register_fake` (Phase 4 D-12 enforces this)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`csrc/diag_mult/diag_mult_cuda_kernel.cu`** — the float-only CUDA kernel; the formula transcribes directly to Triton with `IS_COMPLEX: tl.constexpr` added.
- **`csrc/diag_mult/diag_mult_cuda.cpp`** — the pybind boundary; `_cuda_legacy/diag_mult.py` is a thin Python passthrough wrapping `torch_structured._diag_mult_cuda.cycle_mult`.
- **`torch_structured/_ops.py:225-304`** — Phase 4 demonstrator op. Even though D-27 deletes it, the *pattern* (`@triton.jit` kernel + `@triton_op` wrapper with `view_as_real`/`view_as_complex` + `register_autograd` + `register_fake`) is the literal template Phase 5 transcribes for `diag_mult`. Do not invent a new pattern — copy this one.
- **`torch_structured/_cuda_legacy/butterfly.py`** — exact analog for `_cuda_legacy/diag_mult.py` (de-jitted, plain Python passthrough, `Optional[int]` for backward-compat shifts).
- **`torch_structured/_torch_ref/butterfly.py`** — exact analog for `_torch_ref/diag_mult.py` (relocate-style, not rewrite — but `_torch_ref/diag_mult.py` is a fresh write since `cycle_mult` has no Python reference impl today; use `torch.roll(v, -shift_v) * torch.roll(subdiag, -shift_subdiag)`).

### Established Patterns
- **Try-import + module-level boolean** (`torch_structured/structured/hadamard.py:1-8`, `torch_structured/structured/krylov.py:21-24`) — Phase 5's `_cuda_legacy/diag_mult.py` follows this idiom: try-import `_diag_mult_cuda`; on `ImportError`, set the module symbol to `None`.
- **B3 honest probe** (Phase 4 `_ops.py:82-99`) — `_has_triton_kernel("diag_mult")` probes `torch_structured._triton.diag_mult.op` for the `diag_mult` attribute. The new `_has_cuda_legacy_diag_mult()` probe at the same indentation level checks whether `_cuda_legacy.diag_mult.diag_mult is not None`.
- **Module-level rebindable callable** (`_ops.py:56-58`, `_ops.py:114`) — `diag_mult = None` placeholder declared at module scope; `_resolve()` mutates it via `global`. Consumers (krylov.py) call `torch_structured._ops.diag_mult(...)` (attribute access — per Phase 4 D-05 contract).
- **`# noqa: F401 (re-exported)`** — `_torch_ref/__init__.py` extends `__all__` with `diag_mult_torch` re-export.
- **`assert` preconditions** (CLAUDE.md §"Error Handling", `butterfly/multiply.py:33-36`) — Phase 5 uses `assert v.is_contiguous()` (Pitfall 3 guard before `view_as_real`), `assert subdiag.dtype == v.dtype` (mixed-dtype reject per D-20), `assert subdiag.size(-1) == v.size(-1)`.

### Integration Points
- **`structured/krylov.py:342-350` (`subdiag_mult_cuda`)** — the only call-site that drives `cycle_down_mult` today; rewriting it to call `_ops.diag_mult(...)` is the single user-facing change.
- **`structured/layers.py:237-240` (`LDRSubdiagonalC.forward`)** — does NOT change; routes through `kry.subdiag_mult_cuda` which now wraps `_ops.diag_mult`. End-to-end correctness covered by an `LDRSubdiagonalC` forward+backward test (recommend adding one to `tests/test_diag_mult.py` as an integration sanity check).
- **`_ops.py` resolver step 2 (binding)** — currently has three branches for `butterfly_multiply`; D-22 extends each branch to also bind `diag_mult` (with its own per-op probe). `hadamard_transform` continues to stay None until Phase 6.
- **`_ops.py` resolver step 3 (D-08 heads-up)** — currently dormant in Phase 4 (`actual == "triton"` never fires). Phase 5 will exercise it for the first time on a CUDA host with both `_diag_mult.so` AND the new `_triton/diag_mult/`. The INFO log text composed in Phase 4 stays — no rewording.

</code_context>

<specifics>
## Specific Ideas

- The Phase 4 demonstrator op in `_ops.py:225-304` IS the template — Phase 5's `_triton/diag_mult/op.py` transcribes it nearly verbatim with the identity kernel replaced by the `cycle_mult` formula. Same `view_as_real` guard, same `assert x.is_contiguous()`, same `register_autograd` + `register_fake` shape. Do not invent a different structure.
- The 4-FMA complex multiply inside the kernel must follow `04-COMPLEX-LAYOUT.md` lines 58-76 — both `subdiag` and `v` loaded as `(re, im)` pairs, output stored as `(re, im)`. The order of FMA operations must match: `out_re = a_re*c_re - a_im*c_im`, `out_im = a_re*c_im + a_im*c_re`.
- `_torch_ref/diag_mult.py` is the gradcheck oracle. Implement it as `subdiag.roll(-shift_subdiag, dims=-1) * v.roll(-shift_v, dims=-1)` (the `.roll` direction follows from `(pos + shift + N) % N` = "pos shifted right by `-shift`"). Verify the sign convention on a small numeric example before writing the kernel; an off-by-one rolls-direction bug here would silently pass single-shift tests but fail the (1,1) backward case.
- The autograd backward derivation must be tested via fp64 `gradcheck` (D-26). The hand-written backward at `krylov.py:336` only validates `shift=(0,-1)` — Phase 5 generalizes to arbitrary `(shift_subdiag, shift_v)` so gradcheck across the `{-1, 0, 1}^2` grid is the literal acceptance gate.

</specifics>

<deferred>
## Deferred Ideas

- **`subdiagKrylov` op port** — exported in `csrc/diag_mult/diag_mult_cuda.cpp:18` but has zero Python consumers (grep confirms). Either drop it from `csrc/` in Phase 10 (along with the rest of `csrc/diag_mult/`) or fold its loop into Python over the new `_ops.diag_mult`. Out of Phase 5 scope.
- **Per-op `_BACKENDS: dict[str, str]` map** — D-22a notes the coarse-vs-fine resolver choice. The dict form is more honest but a larger refactor; defer to Phase 7 if the asymmetry between `butterfly_multiply` (Triton) and `diag_mult` (Triton fallback to torch_ref) becomes confusing.
- **Autotune over `BLOCK_SIZE` / `num_warps`** — pointwise kernel; not perf-critical. Worth revisiting if Phase 9's perf grid (`triton.testing.do_bench`) shows diag_mult is a bottleneck on small N. Otherwise drop.
- **bf16 / fp16 support** — TRI-FUT-01; same deferral logic as butterfly.
- **CUDA backend axis in `backend` conftest fixture** — D-30 defers to Phase 9 per the milestone-wide TEST-03 ("full backend axis at integration hardening"). Phase 5's fixture stays `["torch", "triton"]`.
- **Resurrect `_diag_mult.so` build for shipped artifacts** — D-23 keeps the existing conditional `setup.py` logic; environments that have CUDA + nvcc + `csrc/diag_mult/` get the `.so`. Adding a CI matrix entry that explicitly verifies the `.so` builds is a Phase 9 integration concern.

### Reviewed Todos (not folded)
None — no pending todos surfaced by `cross_reference_todos` for Phase 5.

</deferred>

---

*Phase: 5-diag_mult Triton Port*
*Context gathered: 2026-05-27*
