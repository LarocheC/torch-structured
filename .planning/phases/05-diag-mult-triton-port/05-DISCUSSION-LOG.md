# Phase 5: diag_mult Triton Port - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-27
**Phase:** 5-diag_mult Triton Port
**Areas discussed:** API surface of `_ops.diag_mult`, Complex64 semantics, CUDA legacy backend, krylov.py consumer refactor

---

## API surface of `_ops.diag_mult`

| Option | Description | Selected |
|--------|-------------|----------|
| Generic `cycle_mult` primitive | `_ops.diag_mult(subdiag, v, shift_subdiag: int, shift_v: int)` — matches C++ verbatim. krylov.py's fwd (0,-1) and bwd (0,-1)+(1,1) all call the same op with different shifts. Cleanest dispatch parity (cuda backend wires through 1:1). | ✓ (Claude's discretion) |
| Specialized autograd-wrapped op (shift=0,-1) | `_ops.diag_mult(subdiag, v)` — shift hardcoded, autograd via register_autograd. Matches krylov.py's hot path. But krylov.py still needs the (1,1) shift form internally, so we'd need a separate non-autograd primitive too — two surfaces. | |
| Drop the two-shift API; expose `roll`-based op | `_ops.diag_mult(subdiag, v)` does `subdiag * torch.roll(v, -1)` only. Idiomatic PyTorch; loses generality. Krylov.py's bwd `cycle_mult(subdiag, grad, 1, 1)` would need a separate op or get inlined. | |

**User's choice:** "I have no idea. Use your best guess on how to handle this." → deferred to Claude's discretion.
**Notes:** Locked as Option A (generic primitive). Rationale: matches C++ kernel surface 1:1 so the `cuda` backend wires through with zero adaptation; krylov.py's three shift call sites (fwd `(0,-1)`, bwd `(0,-1)` and `(1,1)`) all share one op; Triton compiles per `IS_COMPLEX` constexpr anyway so shift specialization at the Python boundary is free.

---

## Complex64 semantics

| Option | Description | Selected |
|--------|-------------|----------|
| Complex × complex (full 4-FMA) | Both `subdiag` and `v` can be complex64. Kernel does `(a+bi)(c+di) = (ac-bd) + (ad+bc)i` per pair. Most general; future-proofs for any complex-LDR consumer. Matches Phase 4 D-01 template. | ✓ (Claude's discretion) |
| Complex `v` only; subdiag stays real | Only `v` can be complex64; `subdiag` must be real. Kernel is real-scalar × (re, im) — 2 FMAs. Lighter kernel; matches the spectral-radius use case for LDR. | |
| Real only — raise on complex inputs | Implement fp32 only; raise `NotImplementedError` for complex. Violates TRI-01 and the roadmap SC#1. Not viable. | |
| You decide | Let Claude pick based on what's cleanest. | (user-selected escape) |

**User's choice:** "You decide" → deferred to Claude's discretion.
**Notes:** Locked as Option A (complex × complex, 4-FMA). Rationale: Phase 4 D-01 + `04-COMPLEX-LAYOUT.md` already specify the 4-FMA kernel-side template; following it lets Phase 7 (butterfly) reuse the same Triton helper. The `IS_COMPLEX: tl.constexpr` flag makes both specializations zero-cost at runtime. Mixed dtypes (real subdiag, complex v) rejected via `assert subdiag.dtype == v.dtype` for minimal surface area.

---

## CUDA legacy backend — preserve or drop?

| Option | Description | Selected |
|--------|-------------|----------|
| Resurrect `.so` build, wire `_cuda_legacy/diag_mult.py` | Add `setup.py`-driven build; SC#3 satisfied literally. Cost: CI compiles, one more `.so` maintained through Phase 10. | |
| Wire `_cuda_legacy/diag_mult.py` only if `.so` import succeeds | Try-import `_diag_mult_cuda`; if it loads, wire it; if not, `_has_cuda_legacy_diag_mult()` returns False and `BACKEND=cuda` falls back to `torch_ref` with a log.warning. B3 honest-probe pattern. | ✓ (Claude's discretion) |
| Drop the cuda path for diag_mult entirely | `BACKEND=cuda` silently falls back to `torch_ref`; no `.so` build, no `_cuda_legacy/diag_mult.py`. Violates SC#3 strictly. | |

**User's choice:** "Pick your best guess" → deferred to Claude's discretion.
**Notes:** Locked as Option B (try-import + honest probe + log.warning). Rationale: matches Phase 4 CHECKER B3 honest-resolver pattern; pragmatic about the current "no `.so` shipped on this box" reality; SC#3 stays satisfied because the cuda path *is* selectable — it just transparently falls back when not available, with a single log line. Anyone who chooses to build the `.so` (via the existing `setup.py:96-110` conditional) gets bit-exact behavior.

---

## krylov.py consumer refactor scope

| Option | Description | Selected |
|--------|-------------|----------|
| Delete `CycleDownMultCuda`; call `_ops.diag_mult` directly | Remove the hand-rolled autograd Function. Replace `cycle_down_mult = CycleDownMultCuda.apply` with inline `_ops.diag_mult(s, v, 0, -1)`. Cleaner; one fewer abstraction layer. | ✓ (Claude's discretion) |
| Keep `CycleDownMultCuda` shape; route through `_ops.diag_mult` | Keep the `cycle_down_mult` symbol but rewrite the autograd Function to call `_ops.diag_mult(...)` internally. Preserves the krylov.py internal API surface. | |
| Hybrid — forward via `_ops.diag_mult`, keep manual backward | Forward delegates; backward stays in `CycleDownMultCuda`. Complicates `set_backend('triton')` reasoning. | |

**User's choice:** "Do what make the most sense" → deferred to Claude's discretion.
**Notes:** Locked as Option A (delete `CycleDownMultCuda`, inline). Rationale: `_ops.diag_mult` is already autograd-aware via `register_autograd` — wrapping it in another `torch.autograd.Function` is redundant double-wrapping. `cycle_down_mult` is internal to krylov.py (grep confirms no external consumers), so the replacement is purely refactor. Also gives clean B3 semantics: `set_backend('triton')` re-binds `_ops.diag_mult` and consumers auto-pick up the new binding via attribute access (Phase 4 D-05 contract).

---

## Claude's Discretion

All four selected gray areas were resolved as Claude's discretion at the user's request (the user responded "I have no idea / pick your best guess / you decide / do what makes the most sense" for each).

Additional planner-flexible items captured in CONTEXT.md:
- Exact `BLOCK_SIZE` for the Triton kernel (recommend 1024).
- Whether `_BACKEND` global becomes a per-op dict `_BACKENDS` or stays coarse (recommend coarse).
- Wording of the new `log.warning` when `cuda` falls back to `torch_ref` for `diag_mult`.
- Whether `tests/test_dispatch.py` is deleted outright or kept as a thin set_backend smoke test.

## Deferred Ideas

- `subdiagKrylov` op port — exported in `csrc/diag_mult/diag_mult_cuda.cpp:18` but has zero Python consumers. Defer to Phase 10 deletion sweep.
- Per-op `_BACKENDS: dict[str, str]` resolver map — defer to Phase 7 if asymmetry between ops becomes confusing.
- Autotune over `BLOCK_SIZE` / `num_warps` for diag_mult — pointwise kernel; revisit only if Phase 9 perf grid flags it.
- bf16 / fp16 support — TRI-FUT-01.
- CUDA backend axis in `backend` conftest fixture — deferred to Phase 9 per milestone-wide TEST-03.
- Resurrect `_diag_mult.so` shipped builds — existing conditional `setup.py` logic preserved; verifying it builds on CI is a Phase 9 concern.
