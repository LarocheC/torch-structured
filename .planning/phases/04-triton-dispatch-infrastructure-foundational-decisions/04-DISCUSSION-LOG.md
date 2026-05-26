# Phase 4: Triton Dispatch Infrastructure & Foundational Decisions - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-26
**Phase:** 4-Triton Dispatch Infrastructure & Foundational Decisions
**Areas discussed:** Complex64 representation, set_backend() semantics, _torch_ref/ package layout, auto precedence on upgrade

---

## Complex64 representation

| Option | Description | Selected |
|--------|-------------|----------|
| A — Separate-tensor pairs at op boundary | Every op takes (re, im) args explicitly. More verbose but no reinterpret magic. nn.Module call sites pack/unpack. | |
| B — Packed last-dim layout | Op contract is trailing-2 real tensor. Conflicts with existing twiddle layout, breaks checkpoint compat — not really viable. | |
| C — `view_as_real` at wrapper boundary | Public API keeps complex64. Wrappers reinterpret to trailing-2 real before kernel call. Zero-copy. Preserves COMPAT-01/02 and saved checkpoints. Kernel gets an `IS_COMPLEX` constexpr flag. | ✓ |

**User's choice:** Option C — `view_as_real` at wrapper boundary.
**Notes:** Chosen because it cleanly preserves COMPAT-01 (public nn.Module API unchanged) and COMPAT-02 (twiddle layout untouched) while keeping the kernel simple. The `IS_COMPLEX` constexpr flag inside the kernel selects between the real-FMA and the 4-FMA complex-multiply path with no runtime branch. To be documented in `04-COMPLEX-LAYOUT.md` so Phase 7 has a concrete spec to implement.

---

## set_backend() semantics

| Option | Description | Selected |
|--------|-------------|----------|
| A — Reassign + nn.Modules use `_ops.x(...)` form | Module-level reassignment in `_ops.py`. nn.Module call sites use `torch_structured._ops.butterfly_multiply(...)`. set_backend() takes effect immediately everywhere. Honors DISP-03 literally (one attribute access per call, not a branch). | ✓ |
| B — Reassign + "re-instantiate after switching" caveat | nn.Modules keep top-level imports. set_backend() works for new instances but existing instances keep their original binding. Simpler import style, surprising semantics. | |
| C — Context manager + thread-local | `with use_backend('torch'): ...` — most expressive for tests, but every call reads a thread-local (~50ns/call). Strictly violates "no per-call branching". | |

**User's choice:** Option A — Reassign + nn.Modules call via `_ops.x(...)`.
**Notes:** Each kernel call site is one Python attribute lookup, not a conditional. ~5 nn.Module call sites to migrate (small). Tests can call `_ops.set_backend("torch")` and subsequent operations route correctly without re-instantiating modules. This is the cleanest reading of "no per-call branching" + "set_backend at runtime for tests".

---

## _torch_ref/ package layout

| Option | Description | Selected |
|--------|-------------|----------|
| A — Move to `_torch_ref/` + thin shim at old location | `butterfly_multiply_torch` lives in `_torch_ref/butterfly.py`. `butterfly/multiply.py` keeps a re-export shim. Phase 5/6 add `_torch_ref/diag_mult.py` and `hadamard.py`. Clean architecture, zero test breakage. | ✓ |
| B — Move and update test imports in Phase 4 | Cleanest result, no shim. 3 test files need an import line updated. No back-compat for external users (if any). | |
| C — Keep in place; `_ops.py` imports from `butterfly/multiply.py` | No move. `butterfly/multiply.py` mixes reference impl and Triton dispatch wrappers from Phase 7 onward. `_torch_ref/` becomes a fictional architecture concept. | |

**User's choice:** Option A — Move + thin shim.
**Notes:** The shim is a single-line re-export with `# noqa: F401`. No behavior change for any existing caller. Phase 5/6 will add sibling files (`diag_mult.py`, `hadamard.py`) to the same `_torch_ref/` package as those kernels are ported, building out the full reference-implementation surface.

---

## auto precedence on upgrade

| Option | Description | Selected |
|--------|-------------|----------|
| A — Strict Triton-first + nudge message on `.so` detection | Resolution: Triton → CUDA `.so` → torch reference. One-time INFO log when auto resolves to Triton on a machine with leftover `.so`. Honors DEPR-01/02; gives upgrade users a clear out. | ✓ |
| B — Strict Triton-first, silent | Same precedence, no extra messaging. Upgrade users who hit a perf cliff or correctness diff have to discover the env-var fallback themselves. | |
| C — CUDA-first when `.so` loadable | Revise DEPR-01: existing users stay on CUDA; new users default to Triton. v1.2 becomes nearly invisible to existing users — defeats the migration goal. | |

**User's choice:** Option A — Triton-first + nudge message.
**Notes:** INFO log (not DeprecationWarning) keeps the signaling distinct: the auto-switch heads-up tells upgrade users "you just changed paths," while DeprecationWarning (from DEPR-02) tells explicit-CUDA users "you should migrate." Two different audiences, two different messages.

---

## Claude's Discretion

- **Demonstrator op location** — placed at `torch_structured/_ops.py` as `_demo_identity_op` (private). Test at `tests/test_dispatch.py`. Deleted at start of Phase 5.
- **DeprecationWarning ergonomics for DEPR-02** — `warnings.warn(..., DeprecationWarning, stacklevel=2)` with `warnings.simplefilter("once", DeprecationWarning)` in the `_cuda_legacy` import block. Phase 10 implements per the `04-DEPRECATION-PLAN.md` written in Phase 4.
- **CI cache concrete config** — left to planner; reuse whatever CI mechanism the repo already has (likely GitHub Actions `actions/cache@v4` keyed on `torch.__version__` + git SHA of `_triton/`).
- **Top-level `torch_structured.set_backend` export** — recommend yes for ergonomics; planner verifies no circular-import issue (likely safe since `_ops` doesn't import from public modules).
- **Demonstrator op exercises complex64** — recommend including a complex input path even though it's a no-op identity, because complex64 routing is on the critical path for Phase 7.
- **Internal naming** — `_resolve` / `_pick_backend` / etc. are planner's choice.

## Deferred Ideas

- AOT-compiled Triton bytecode shipped in wheel for common shapes (v1.3+ optimization; no requirement).
- `TRITON_INTERPRET=1` debugging guide in CONTRIBUTING.md (mention but not a Phase 4 deliverable).
- `torch.backends.torch_structured` namespace registration (no standard PyTorch mechanism for third-party libs; reconsider if/when one exists).
- Bf16/fp16 support in the demonstrator (deferred to TRI-FUT-01 / post-v1.2 when real kernels gain bf16/fp16).
