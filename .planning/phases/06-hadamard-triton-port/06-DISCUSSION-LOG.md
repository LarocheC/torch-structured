# Phase 6: hadamard Triton Port - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-27
**Phase:** 6-hadamard Triton Port
**Areas discussed:** Triton kernel design (single-pass vs mixed-radix), Backward routing for self-inverse, structured/hadamard.py + fastfood.py refactor scope, normalize=True kwarg handling

---

## Triton kernel design — single-pass vs mixed-radix two-pass

| Option | Description | Selected |
|--------|-------------|----------|
| Single-pass shared-mem | One `@triton.jit` kernel; BLOCK_SIZE=N (max 4096); all log_n stages in shared memory. Matches SC#1's log_n≤12 cap. | ✓ |
| Mixed-radix two-pass (matches ROADMAP wording) | Two `@triton.jit` kernels: shared-mem inner + global-mem outer. Faithful port of CUDA `fwtBatch1Kernel` + `fwtBatch2Kernel`. | |
| Single-stage × log_n launches | One kernel per butterfly stage; Python wrapper launches log_n times. Easy correctness; ~log_n launches per call. | |

**User's choice:** Selected Option A (single-pass shared-mem) directly from the menu.
**Notes:** SC#1 caps log_n at 12 → n=4096 → 16KB fp32 shared mem per block, fits comfortably in any GPU with >=48KB shared mem. The mixed-radix two-pass is deferred to a future milestone if log_n > 12 ever becomes a real requirement. Documented in CONTEXT.md D-31c.

---

## Backward routing for self-inverse op

| Option | Description | Selected |
|--------|-------------|----------|
| Dispatch-aware via `_ops.hadamard_transform(grad)` | Backward callback uses attribute access on `_ops`; honors `set_backend` rebinding at backward time. | |
| Oracle via `_torch_ref.hadamard.hadamard_transform_torch(grad)` (Phase 5 D-26 convention) | Backward routes through the pure-PyTorch oracle. Deterministic, fp64-capable for gradcheck. | ✓ (Claude's discretion) |
| Self-call via direct Triton kernel | Backward calls the Triton kernel directly. Fastest but breaks fp64 gradcheck since Triton kernel is fp32-only. | |

**User's choice:** "Select the option that make the most sense" → deferred to Claude's discretion.
**Notes:** Locked as Option B. Rationale: Phase 5 D-26 set the convention (backward through `_torch_ref` for fp64 gradcheck precision and dispatch-independent determinism). The Triton kernel for hadamard is fp32-only per ROADMAP "no complex" wording (and the underlying butterfly arithmetic is real-valued only), so a self-call to the Triton kernel would fail fp64 gradcheck. Phase 7/8 will inherit this convention.

---

## `structured/hadamard.py` + `fastfood.py` refactor scope

| Option | Description | Selected |
|--------|-------------|----------|
| Phase 5 mirror — delete `HadamardTransformCuda`; inline `_ops.hadamard_transform` calls | Match Phase 5 D-24/D-25 pattern: delete the autograd Function wrapper; inline attribute-access call sites in fastfood.py; move `hadamard_transform_torch` to `_torch_ref/`; keep back-compat shim. | ✓ |
| Conservative — keep wrappers; route through `_ops` internally | Keep `HadamardTransformCuda` and `hadamard_transform_cuda` symbols, rewrite to call `_ops.hadamard_transform(...)` internally. fastfood.py keeps early-binding (D-05 violation). | |
| Minimal — don't touch `structured/hadamard.py` or `fastfood.py` | Add `_ops.hadamard_transform` binding only; structured module untouched. Cleanest single-plan scope; consumer surface inconsistent with Phase 5. | |

**User's choice:** Selected Option A (Phase 5 mirror) directly from the menu.
**Notes:** Consistent with Phase 5's krylov.py refactor. Two call sites in fastfood.py rewrite to `torch_structured._ops.hadamard_transform(...)` per D-05 attribute-access contract. Back-compat shim in `structured/hadamard.py` preserves `tests/structured/test_hadamard.py:8` (`from torch_structured.structured.hadamard import hadamard_transform_torch`) without test edits.

---

## `normalize=True` kwarg handling

| Option | Description | Selected |
|--------|-------------|----------|
| Python wrapper-side scaling | Triton kernel is unnormalized; Python wrapper applies `out / 2**(m/2)` after the kernel returns. Matches existing `structured/hadamard.py:58` convention. | ✓ |
| Kernel-side scaling via `NORMALIZE: tl.constexpr` | Triton kernel takes a NORMALIZE constexpr; multiplies by `1/sqrt(N)` at the final tl.store. One fewer GPU launch; 2 JIT-compiled variants per shape. | |
| Drop `normalize=` from `_ops.hadamard_transform` | Always return unnormalized; consumers handle scaling. Cleanest API; breaks back-compat. | |

**User's choice:** Selected Option A (Python wrapper-side scaling) directly from the menu.
**Notes:** Simplest kernel; matches existing semantics exactly (including the odd-m `2 ** (m/2) = sqrt(2) * 2 ** ((m-1)/2)` case). Backward applies the same scale per chain rule (D-32b). `setup_context` saves the `normalize` flag.

---

## Claude's Discretion

Three of four areas were selected directly from the menu; one (backward routing) was deferred to Claude. Locked decisions captured in CONTEXT.md D-31..D-39.

Additional planner-flexible items:
- Exact `BLOCK_SIZE` / `num_warps` choice (recommend num_warps=4 for log_n ≤ 8, num_warps=8 for log_n ∈ {9..12}).
- Test file location: `tests/structured/test_hadamard_triton.py` vs `tests/test_hadamard_triton.py` (planner picks for symmetry).
- Exact wording of D-22 fallback `log.warning` text.
- Whether to autotune via `@triton.autotune` (recommend fixed configs for Phase 6; autotune in Phase 9 if needed).
- `_torch_ref` file name: `_torch_ref/hadamard.py` (recommended; matches existing `_torch_ref/butterfly.py` style) vs `_torch_ref/hadamard_transform.py`.

## Deferred Ideas

- Mixed-radix two-pass Triton kernel — log_n > 12 only; future milestone.
- Autotune (`@triton.autotune`) — defer to Phase 9 perf grid.
- bf16/fp16 hadamard — TRI-FUT-01.
- Complex hadamard — explicitly out of scope per ROADMAP "no complex".
- Resurrecting the `_hadamard_cuda.so` build — Phase 9 CI matrix concern.
- CUDA backend axis in `backend` conftest fixture — Phase 9 / TEST-03.
- `Hadamard` nn.Module factory in `structured/layers.py` — planner verifies during scout; if it imports `hadamard_transform` directly, the fastfood.py D-34 refactor extends to it.
