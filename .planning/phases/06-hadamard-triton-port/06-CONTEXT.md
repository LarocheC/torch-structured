# Phase 6: hadamard Triton Port - Context

**Gathered:** 2026-05-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Port the fast Walsh-Hadamard transform (`hadamard_transform(x)`) from C++/CUDA to a Triton `@triton.jit` kernel, expose it as `_ops.hadamard_transform(u, normalize=False)` with self-inverse autograd via `register_autograd`, refactor `structured/hadamard.py` + `structured/fastfood.py` to consume the single dispatch import point, and delete the existing `HadamardTransformCuda` autograd wrapper. fp32 only; no complex; no atomics; log_n ∈ {2..12}.

**In scope:**
- New `torch_structured/_triton/hadamard_transform/{__init__.py, op.py}` — single-pass shared-memory `@triton.jit` kernel + `@triton_op` wrapper + `register_autograd` backward (self-inverse: `H @ grad`) + `register_fake` meta kernel.
- New `torch_structured/_torch_ref/hadamard.py` — pure-PyTorch reference (move from `structured/hadamard.py:15-30`); `_torch_ref/__init__.py` extends `__all__` with `hadamard_transform_torch`.
- New `torch_structured/_cuda_legacy/hadamard.py` — thin try-import wrapper around `torch_structured._hadamard_cuda.hadamard_transform` (graceful `None` on ImportError, `HAS_CUDA_LEGACY_HADAMARD` sentinel for honest probe).
- `_ops.py` resolver edits: bind `hadamard_transform` per-op in `_resolve()` Step 2 (matching Phase 5's per-op three-branch pattern); add `_has_cuda_legacy_hadamard()` probe at `_ops.py:~81` (after `_has_cuda_legacy_diag_mult`); `_has_any_triton_kernel()` already iterates `hadamard_transform` per `_ops.py:126` — no resolver Step 1 changes needed.
- `structured/hadamard.py` refactor: delete `HadamardTransformCuda` (lines 33-42), delete `hadamard_transform_cuda` wrapper (lines 45-58), delete the module-level `hadamard_transform = ...` binding (line 61), delete the `use_hadamard_transform_cuda` try-import (lines 4-8); keep `hadamard_transform_torch` as a back-compat shim re-exporting from `_torch_ref.hadamard` (`from torch_structured._torch_ref.hadamard import hadamard_transform_torch  # noqa: F401`).
- `structured/fastfood.py` refactor: rewrite line 1 from `from .hadamard import hadamard_transform` to `import torch_structured` (no early binding); rewrite call sites at lines 8 and 10 from `hadamard_transform(...)` to `torch_structured._ops.hadamard_transform(...)` (D-05 attribute access).
- `tests/conftest.py` extension: `backend` fixture already widened in Phase 5 to `["torch", "triton"]` with skip-gate via `_has_triton_kernel(...)` — Phase 6 widens the skip-gate to OR over both diag_mult AND hadamard_transform (or use `_has_any_triton_kernel()`).
- New `tests/structured/test_hadamard_triton.py` — cross-backend correctness (vs scipy reference + vs `_torch_ref.hadamard.hadamard_transform_torch`); fp32 only; log_n grid {2..12}; gradcheck fp64 on `_torch_ref` path (Triton kernel is fp32-only); self-inverse composition test (`hadamard ∘ hadamard ≈ N · I` to fp32 noise floor); a `Hadamard` / `hadamard()` consumer-surface integration test on the Triton backend per ROADMAP SC#3.
- `tests/structured/test_hadamard.py` minor update if needed (the existing tests reference `hadamard_transform_torch` directly — the back-compat shim covers them).
- `tests/structured/test_imports.py` — update import surface if Phase 6 changes the public-symbol set (current line 7 imports `hadamard_transform` from `structured.hadamard` — this still works via the shim re-routing through `_ops`, but planner verifies).

**Out of scope:**
- Backward kernel (self-inverse — backward IS forward applied to grad).
- Complex64 support (ROADMAP says "no complex" — fp32 only).
- Atomics / `tl.atomic_add` (Hadamard butterfly writes to disjoint output positions per stage; no reduction needed).
- bf16/fp16 (TRI-FUT-01).
- `log_n > 12` (ROADMAP caps at n=4096; the mixed-radix two-pass that the CUDA kernel uses for larger N is deferred; Phase 6 ships the single-pass shared-memory variant that covers SC#1 exactly).
- `_has_cuda_legacy_hadamard()` resurrection — `_hadamard_cuda.so` isn't currently built (only `_butterfly.so` and `_version.so` exist on disk; same situation as Phase 5 for diag_mult). The try-import gracefully returns `HAS_CUDA_LEGACY_HADAMARD = False`; `BACKEND=cuda` falls back to `_torch_ref` for hadamard with a `log.warning` (D-22 pattern from Phase 5).
- Touching `csrc/hadamard/hadamard_cuda.cpp` / `_kernel.cu` (Phase 10 deletion candidates per DEPR-03/04).
- Editing `setup.py` (existing conditional `_hadamard_cuda` build at `setup.py:85-92` already handles "build if dir exists" — no change needed).
- Public `Butterfly`/`LDR*`/`make_linear` API surface — byte-identical preserved.

</domain>

<decisions>
## Implementation Decisions

### Triton kernel design — single-pass shared-memory (Claude's discretion, locked)

- **D-31:** The Triton kernel is **single-pass shared-memory**: one `@triton.jit` kernel does all `log_n` butterfly stages in shared memory within one launch. Grid is `(batch_size,)` where `batch_size = x.numel() // n`. `BLOCK_SIZE: tl.constexpr` = `N` (power-of-2; max 4096 per SC#1's log_n≤12 cap). `LOG_N: tl.constexpr` controls the unrolled stage loop.
- **D-31a:** Kernel body executes the Hadamard butterfly: for each `stride` in `(1, 2, 4, ..., N/2)`, pairs at indices `(i0, i0+stride)` are replaced by `(x[i0] + x[i0+stride], x[i0] - x[i0+stride])`. All log_n stages fit in shared memory (single `tl.load` at start, single `tl.store` at end; intermediate state lives in registers + shared scratch).
- **D-31b:** Recommended `num_warps` defaults: 4 for log_n ≤ 8 (N ≤ 256), 8 for log_n in {9..12} (N in {512..4096}). Planner may use `@triton.autotune` with a small config space; not perf-critical for Phase 6 (perf gate is Phase 9 / TEST-04).
- **D-31c:** Mixed-radix two-pass (the CUDA structure at `csrc/hadamard/hadamard_cuda_kernel.cu:24-132`) is **explicitly deferred** to a future milestone for log_n > 12. ROADMAP SC#1 caps at log_n=12; the single-pass kernel covers this fully. Document the deferral in the phase SUMMARY for Phase 9's perf grid to verify.

### Self-inverse backward — `_torch_ref` oracle convention (Claude's discretion, locked)

- **D-32:** The `register_autograd` backward callback routes `grad` through `_torch_ref.hadamard.hadamard_transform_torch(grad)` — the pure-PyTorch oracle. Same convention as Phase 5 D-26 (backward through torch_ref for fp64 gradcheck precision and deterministic backward independent of which backend forward used).
- **D-32a:** Self-inverse property: `H @ (H @ u) = N · u` (unnormalized) or `H @ (H @ u) = u` (normalized). So `d(H @ u)/du = H` and backward of `out = H @ u` is `grad_u = H @ grad_out`. No separate derivation needed — the same formula transcribes; no `.conj()` (real-only kernel).
- **D-32b:** The `register_autograd` backward includes the `normalize=` argument propagation: if forward was called with `normalize=True`, backward also normalizes the gradient (chain rule on a scalar multiply). Saved via `setup_context(ctx, inputs, output)` capturing `inputs[1]` (the normalize flag).
- **D-32c:** `register_fake` meta kernel: returns `torch.empty_like(u)` — shape, dtype, device preserved. Same as the Phase 5 `register_fake` for `diag_mult`.

### `structured/hadamard.py` + `fastfood.py` refactor (Claude's discretion, locked — Phase 5 mirror)

- **D-33:** Delete `HadamardTransformCuda(torch.autograd.Function)` at `structured/hadamard.py:33-42`. Redundant: `_ops.hadamard_transform`'s `register_autograd` now handles autograd.
- **D-33a:** Delete `hadamard_transform_cuda` wrapper at `structured/hadamard.py:45-58`. Its `normalize` handling is absorbed by `_ops.hadamard_transform(u, normalize=...)`.
- **D-33b:** Delete the module-level binding `hadamard_transform = hadamard_transform_cuda if use_hadamard_transform_cuda else hadamard_transform_torch` at line 61. Replace with attribute access at consumer call sites per D-05.
- **D-33c:** Delete the try-import block at lines 4-8 (`use_hadamard_transform_cuda = True; try: from torch_structured import _hadamard_cuda; except: use_hadamard_transform_cuda = False`). The new `_cuda_legacy/hadamard.py` owns this probe.
- **D-33d:** Keep `hadamard_transform_torch` exposed at `structured/hadamard.py` via re-export shim: `from torch_structured._torch_ref.hadamard import hadamard_transform_torch  # noqa: F401`. This preserves `tests/structured/test_hadamard.py:8` (`from torch_structured.structured.hadamard import hadamard_transform_torch`) and `tests/structured/test_imports.py:7` import surface.
- **D-34:** Rewrite `structured/fastfood.py`:
  - Line 1: `from .hadamard import hadamard_transform` → `import torch_structured` (drop early binding per D-05).
  - Line 8: `hadamard_transform(B * x)` → `torch_structured._ops.hadamard_transform(B * x)`.
  - Line 10: `hadamard_transform(G * PHBx)` → `torch_structured._ops.hadamard_transform(G * PHBx)`.

### `normalize=True` — wrapper-side scaling (Claude's discretion, locked)

- **D-35:** `_ops.hadamard_transform(u, normalize: bool = False)` — the kernel itself is unnormalized; the Python wrapper applies `out / (2 ** (m / 2))` where `m = log_n` after the kernel returns. Matches the existing `structured/hadamard.py:58` convention.
- **D-35a:** The scale factor `1.0 / sqrt(N)` is computed as `2 ** (m / 2)` to match the existing semantics exactly (avoids `math.sqrt` import variation). For odd `m`, `2 ** (m / 2)` is `sqrt(2) * 2 ** ((m-1) / 2)` — same as the existing code.
- **D-35b:** Normalization applies symmetrically to backward (D-32b): `setup_context` saves the `normalize` flag; backward applies the same scale to the gradient.
- **D-35c:** `_torch_ref.hadamard.hadamard_transform_torch(u, normalize=False)` — same signature; identical normalization semantics.

### Per-op resolver wiring (inherits Phase 5 D-22 pattern, no new decisions)

- **D-36:** `_ops.py` resolver Step 2 binding for `hadamard_transform` follows the existing three-branch shape (mirrors the `diag_mult` block added in Phase 5 Task 4):
  - `if actual == "triton" and _has_triton_kernel("hadamard_transform")`: bind to `_triton.hadamard_transform.op.hadamard_transform`.
  - `elif actual == "cuda" and _has_cuda_legacy_hadamard()`: bind to `_cuda_legacy.hadamard.hadamard_transform`.
  - `else`: bind to `_torch_ref.hadamard.hadamard_transform_torch`; emit a `log.warning` when `actual == "cuda"` AND `_has_cuda_legacy_hadamard()` is False (D-22 asymmetric fallback pattern).
- **D-36a:** `_has_cuda_legacy_hadamard()` probe at `_ops.py:~81` (after `_has_cuda_legacy_diag_mult`): returns `True` iff `_cuda_legacy.hadamard.hadamard_transform is not None`. Honest-probe pattern mirrors Phase 5 D-21.
- **D-36b:** The `_has_any_triton_kernel()` helper already iterates `("butterfly_multiply", "diag_mult", "hadamard_transform")` at `_ops.py:126` — no Step 1 changes needed for Phase 6 to make `actual == "triton"` reachable.
- **D-36c:** The per-op `log.info` line at the end of `_resolve()` (added in Phase 5 Task 4) automatically includes `hadamard_transform=<actual>` once Phase 6 lights it up.

### Test surface (Claude's discretion within scope)

- **D-37:** New `tests/structured/test_hadamard_triton.py` (or `tests/test_hadamard_triton.py` — planner picks based on test layout convention). Covers:
  - `test_hadamard_eager_fp32` cross-backend allclose vs `_torch_ref.hadamard.hadamard_transform_torch`, log_n grid {2..12}, parametrized over the `backend` conftest fixture.
  - `test_hadamard_normalize` — `normalize=True` correctness vs `_torch_ref` at log_n=10 (cross-backend).
  - `test_hadamard_gradcheck_fp64` — `torch.autograd.gradcheck` of `_ops.hadamard_transform` at fp64 against `autograd.grad(_torch_ref.hadamard_transform_torch, ...)`. Per D-32, backward routes through `_torch_ref` so fp64 gradcheck passes natively. Small N (n=4 or n=8) for tractability.
  - `test_hadamard_self_inverse` — composition `H ∘ H` ≈ `N · I` (unnormalized) or `I` (normalized), bit-equivalent to fp32 noise floor. Per ROADMAP SC#2.
  - `test_hadamard_module_consumer` — integration sanity: a small fastfood-style consumer chain (or `Hadamard` nn.Module if one exists in `structured/layers.py` — verify in scout) that routes through `_ops.hadamard_transform` produces correct outputs. Per ROADMAP SC#3.
- **D-38:** `tests/structured/test_hadamard.py` (existing) requires no changes if the back-compat shim at D-33d works; planner verifies the existing 3 tests still pass.
- **D-39:** `tests/conftest.py` `backend` fixture skip-gate: widen the `triton` skip predicate from `_has_triton_kernel("diag_mult")` to use `_has_any_triton_kernel()` (or OR over both) so the fixture skips only when NO Triton kernel is installed. Phase 5's exact wording per `tests/conftest.py:18` may have used a hardcoded probe — recheck and generalize.

### Claude's Discretion

All four selected gray areas resolved as Claude's discretion at user's request. Remaining planner-flexible items:
- Exact `BLOCK_SIZE` / `num_warps` choice (recommend 4 for log_n ≤ 8, 8 for log_n in {9..12}; defer autotune to Phase 9 if needed).
- Whether the new test file is `tests/structured/test_hadamard_triton.py` (mirrors existing `tests/structured/test_hadamard.py`) or `tests/test_hadamard_triton.py` (top-level, mirrors Phase 5's `tests/test_diag_mult.py`). Either works; planner picks for symmetry.
- The exact wording of `log.warning` text for the D-22 fallback case ("BACKEND=cuda requested but `_hadamard_cuda` not built; falling back to torch_ref for hadamard").
- Whether to autotune the kernel via `@triton.autotune` or use fixed `num_warps` configs. Recommend fixed for Phase 6 (autotune perf gain is Phase 9).
- File layout for `_torch_ref/hadamard.py` vs `_torch_ref/hadamard_transform.py` — recommend `_torch_ref/hadamard.py` exporting `hadamard_transform_torch` (matches existing function name + the `_torch_ref/butterfly.py` parallel from Phase 4).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 6 charter
- `.planning/ROADMAP.md` §"Phase 6" — phase goal, depends on Phase 5, 3 success criteria, 1 plan slot
- `.planning/REQUIREMENTS.md` §"v1.2 Requirements" → TRI-02 (sole REQ this phase covers)
- `.planning/REQUIREMENTS.md` §"Traceability" — confirms TRI-02 mapped to Phase 6

### Phase 4 hand-off (LOCKED — inherited)
- `.planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/04-CONTEXT.md` — D-04..D-08 (dispatch + set_backend), D-09..D-10 (`_torch_ref/` layout), D-11..D-12 (torch>=2.6, triton_op pattern), D-15 (deprecation plan), D-16 (CI cache)
- `.planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/04-COMPLEX-LAYOUT.md` — does NOT apply (Phase 6 is real-only, no complex)
- `.planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/04-DEPRECATION-PLAN.md` — Phase 10 implements; Phase 6's cuda-legacy fallback is `log.warning`, not `DeprecationWarning`

### Phase 5 hand-off (LOCKED — inherited)
- `.planning/phases/05-diag-mult-triton-port/05-CONTEXT.md` — especially D-21 (try-import + sentinel), D-22 (per-op fallback), D-24/D-25 (consumer refactor pattern), D-26 (backward via _torch_ref oracle), D-27/D-28 (test surface), D-30 (conftest widening)
- `.planning/phases/05-diag-mult-triton-port/05-01-PLAN.md` — Phase 5's task structure is the literal template for Phase 6 (7 tasks: foundation → triton → cuda_legacy → resolver → consumer refactor → conftest → tests). The `_has_any_triton_kernel()` helper added in Phase 5 Task 4 is already in place.
- `.planning/phases/05-diag-mult-triton-port/05-RESEARCH.md` — backward formula derivation pattern (Phase 6 is simpler — self-inverse means backward = forward applied to grad; no Wirtinger needed since no complex)
- `.planning/phases/05-diag-mult-triton-port/05-01-SUMMARY.md` — concrete delta lines and code excerpts for the `_ops.py` resolver edits, the `_cuda_legacy/diag_mult.py` shape, and the `_triton/diag_mult/op.py` skeleton (transcribe with hadamard formula substituted)
- `.planning/phases/05-diag-mult-triton-port/05-VERIFICATION.md` — verification pattern for goal-backward checks
- `.planning/phases/05-diag-mult-triton-port/05-REVIEW.md` — code review findings; especially WR-02 (`is_conj()`-flagged tensors) and WR-05 (no shift bounds checks). WR-02 doesn't apply (no complex in Phase 6). WR-05's analog: `n` must be power-of-2, asserted at wrapper.

### Research outputs (milestone-wide)
- `.planning/research/SUMMARY.md` §"Architecture Approach" — overall `_triton/<op>/op.py` layout
- `.planning/research/PITFALLS.md` §3 — Phase 6 hadamard challenge: "two-pass mixed-radix shared-memory pattern in Triton without atomics or complex"; we explicitly opt for single-pass per D-31c
- `.planning/research/STACK.md` — `@triton.jit` + `wrap_triton` + `register_autograd` + `register_fake` API contract
- `.planning/research/ARCHITECTURE.md` — `_triton/<op>/op.py` layout pattern

### Project-level constraints
- `.planning/PROJECT.md` §"Current Milestone: v1.2" — parallel paths; `hadamard_transform_torch` preserved as oracle + runtime fallback
- `CLAUDE.md` (project root) — `assert` for preconditions, no try/except in core lib (one exception: `_cuda_legacy/*.py` try-imports — documented honest-probe pattern from Phase 5 D-21)
- `/home/claroche/CLAUDE.md` (user-level) — `bd` for task tracking, NOT TaskCreate/TodoWrite

### Code-level references (read before editing)
- `csrc/hadamard/hadamard_cuda.cpp:5-14` — `hadamard_transform(x)` C++ signature: takes a CUDA tensor, asserts last-dim power-of-2, returns transform. fp32-only.
- `csrc/hadamard/hadamard_cuda_kernel.cu:24-153` — CUDA kernel structure (mixed-radix two-pass via `fwtBatch1Kernel` + `fwtBatch2Kernel`). Phase 6 doesn't transcribe this verbatim — opts for single-pass per D-31. Reference for correctness checking only.
- `torch_structured/structured/hadamard.py:1-62` — current consumer module. Lines 33-42 + 45-58 + 61 to delete; lines 4-8 to delete; lines 15-30 (`hadamard_transform_torch`) to move to `_torch_ref/hadamard.py` with back-compat shim.
- `torch_structured/structured/fastfood.py:1,8,10` — consumer rewrite per D-34 (the only Python consumer of `hadamard_transform` outside test files).
- `torch_structured/_ops.py:81-99` — `_has_cuda_legacy_diag_mult` + `_has_triton_kernel` patterns to mirror for hadamard.
- `torch_structured/_ops.py:101-130` — `_has_any_triton_kernel()` helper (already iterates `"hadamard_transform"`); no change needed.
- `torch_structured/_ops.py:144-240` — `_resolve()` Step 2 binding logic (Phase 5 added the `diag_mult` per-op branch; Phase 6 mirrors for `hadamard_transform`).
- `torch_structured/_torch_ref/__init__.py` + `_torch_ref/butterfly.py` + `_torch_ref/diag_mult.py` — analogs for new `_torch_ref/hadamard.py`.
- `torch_structured/_cuda_legacy/__init__.py` + `_cuda_legacy/butterfly.py` + `_cuda_legacy/diag_mult.py` — analogs for new `_cuda_legacy/hadamard.py`.
- `torch_structured/_triton/__init__.py` + `_triton/diag_mult/op.py` — analog for new `_triton/hadamard_transform/op.py` (transcribe with hadamard formula; drop complex branch; backward is self-inverse).
- `tests/conftest.py:13-22` — `backend` fixture (Phase 5 added skip-gate via `_has_triton_kernel("diag_mult")`); Phase 6 widens to `_has_any_triton_kernel()` per D-39.
- `tests/structured/test_hadamard.py` — existing tests; should pass unchanged via back-compat shim.
- `tests/structured/test_imports.py:7-13` — public-symbol import checks; verify pass after D-33 refactor.
- `setup.py:75-92` — existing conditional `_hadamard_cuda` build (no change in Phase 6; D-21 honest-probe pattern handles missing `.so`).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`hadamard_transform_torch` at `structured/hadamard.py:15-30`** — pure-PyTorch reference (`torch.cat` interleaved butterfly); relocate verbatim to `_torch_ref/hadamard.py` (per D-33d back-compat shim). 16-line function, no logic change.
- **`csrc/hadamard/hadamard_cuda.cpp`** — pybind boundary; `_cuda_legacy/hadamard.py` is a thin Python passthrough wrapping `torch_structured._hadamard_cuda.hadamard_transform` (graceful `None` on ImportError per D-21).
- **Phase 5's `_triton/diag_mult/op.py`** — literal template for `_triton/hadamard_transform/op.py`. Replace the pointwise `cycle_mult` body with the in-shared-memory log_n-stage Hadamard butterfly; drop the IS_COMPLEX branch (no complex per ROADMAP); drop the four-arg signature (just `(u,)` for hadamard); backward callback becomes `_torch_ref.hadamard.hadamard_transform_torch(grad)` (self-inverse per D-32).
- **`_has_any_triton_kernel()` at `_ops.py:101-130`** — already iterates `hadamard_transform`; the BLOCKER-1 fix from Phase 5 means Phase 6 doesn't need to touch resolver Step 1.

### Established Patterns
- **Phase 5's per-op three-branch resolver binding** (`_ops.py:144-240`) — Phase 6 adds a `hadamard_transform` block in the same shape:
  ```python
  # In Step 2:
  if actual == "triton" and _has_triton_kernel("hadamard_transform"):
      from torch_structured._triton.hadamard_transform.op import hadamard_transform as _triton_ht
      hadamard_transform = _triton_ht
      _hadamard_transform_backend = "triton"
  elif actual == "cuda" and _has_cuda_legacy_hadamard():
      from torch_structured._cuda_legacy.hadamard import hadamard_transform as _cuda_ht
      hadamard_transform = _cuda_ht
      _hadamard_transform_backend = "cuda"
  else:
      from torch_structured._torch_ref.hadamard import hadamard_transform_torch
      hadamard_transform = hadamard_transform_torch
      _hadamard_transform_backend = "torch"
      if actual == "cuda":
          log.warning("set_backend('cuda') requested but _hadamard_cuda not built; falling back to torch_ref for hadamard")
  ```
- **Phase 5's try-import + sentinel idiom** (`_cuda_legacy/diag_mult.py`) — Phase 6's `_cuda_legacy/hadamard.py` mirrors exactly:
  ```python
  HAS_CUDA_LEGACY_HADAMARD = True
  try:
      from torch_structured import _hadamard_cuda
  except ImportError:
      _hadamard_cuda = None
      HAS_CUDA_LEGACY_HADAMARD = False

  def hadamard_transform(u: torch.Tensor) -> torch.Tensor:
      if _hadamard_cuda is None:
          raise RuntimeError("_hadamard_cuda not built")  # documented exception
      assert u.dtype == torch.float32, "_hadamard_cuda is fp32-only"
      assert u.is_cuda, "_hadamard_cuda requires CUDA tensor"
      return _hadamard_cuda.hadamard_transform(u)
  ```
- **D-05 attribute-access contract** — fastfood.py rewrite per D-34 follows Phase 5's krylov.py D-25 pattern: `import torch_structured` at top, `torch_structured._ops.hadamard_transform(...)` at call sites.
- **`assert` preconditions** — wrapper-boundary asserts in the Triton op: `assert u.dim() >= 1 and u.size(-1) >= 1`, `assert (u.size(-1) & (u.size(-1) - 1)) == 0` (power-of-2 check), `assert u.dtype in (torch.float32, torch.float64)` (allow fp64 for gradcheck oracle leg; kernel itself is fp32 but the wrapper accepts fp64 and routes to torch_ref via the backward callback).

### Integration Points
- **`structured/fastfood.py`** — the only non-test Python consumer (lines 8, 10). D-34 rewrites.
- **`structured/hadamard.py`** — keeps the back-compat shim for `hadamard_transform_torch` (D-33d) so `tests/structured/test_hadamard.py:8` and `test_imports.py:7` continue to work without edits.
- **`_ops.py` resolver Step 3 (per-op log.info)** — Phase 5 added a line like `log.info("torch_structured: per-op bindings: butterfly_multiply=%s, diag_mult=%s, hadamard_transform=%s", ...)`. Phase 6 extends the format string args; the line was authored in Phase 5 expecting hadamard to light up here.

</code_context>

<specifics>
## Specific Ideas

- **The Phase 5 `_triton/diag_mult/op.py` IS the template** — copy its structure (`@triton.jit` kernel + `@triton_op` wrapper + `_setup_context` + `_backward` + `register_fake`). The hadamard kernel body is the only substantive divergence; backward is simpler (no Wirtinger `.conj()`, no shift arguments, single-tensor input).
- **Hadamard butterfly inner loop in Triton** — same structure as the CUDA `fwtBatch1Kernel:39-86`: for `stride` in `(1, 2, 4, ..., N/2)`, do the in-place add/sub on pairs. In Triton, this is unrolled at JIT time via the `LOG_N: tl.constexpr` flag with `tl.static_range(LOG_N)` (or `for k in range(LOG_N):` since `LOG_N` is constexpr). All log_n stages live in shared memory; the kernel does one `tl.load` at start, one `tl.store` at end, intermediate state in registers.
- **Self-inverse acceptance test** — ROADMAP SC#2 says "Composing `hadamard ∘ hadamard` is bit-equivalent (within fp32 noise) to identity on any input shape". For unnormalized: `H @ (H @ u) = N · u`. For normalized: `H @ (H @ u) = u`. Use this as a no-derivative consistency check that catches sign errors in the kernel.
- **Normalization scale invariant** — `2 ** (m / 2)` matches the existing `structured/hadamard.py:58` exactly, including the odd-m case where Python computes `2 ** 0.5 = 1.4142...`. Don't rewrite as `math.sqrt(N)` or similar — keeps numerical parity with the existing path.
- **`fastfood_multiply` integration test** — after refactor, `fastfood_multiply(S, G, B, P, x)` should produce identical outputs to the pre-refactor code on a small random input. Use this as the consumer-surface regression gate.

</specifics>

<deferred>
## Deferred Ideas

- **Mixed-radix two-pass Triton kernel** (D-31c) — needed only for log_n > 12; out of Phase 6 scope per SC#1. Revisit in a follow-up milestone if larger-N hadamard becomes a real consumer requirement.
- **Autotune over `num_warps` / `BLOCK_SIZE`** — defer to Phase 9 if the perf grid shows hadamard is a bottleneck. Phase 6 uses fixed `num_warps` defaults per D-31b.
- **bf16 / fp16 hadamard kernel** — TRI-FUT-01; same deferral logic as butterfly + diag_mult.
- **Complex hadamard** — out of scope per ROADMAP "no complex". If a real consumer ever needs it, the Phase 4 `view_as_real` + `IS_COMPLEX: tl.constexpr` template from `04-COMPLEX-LAYOUT.md` applies.
- **Resurrect `_hadamard_cuda.so` build** — existing conditional `setup.py:85-92` already handles this. Phase 9 (TEST-04 perf gate) can add a CI matrix entry that verifies the `.so` builds. Out of Phase 6 scope.
- **CUDA backend axis in `backend` conftest fixture** — D-30 (Phase 5) defers the `"cuda"` param to Phase 9 per TEST-03 ("full backend axis at integration hardening"). Phase 6's fixture stays `["torch", "triton"]`.
- **`Hadamard` nn.Module factory** — `structured/layers.py` may have a `Hadamard` nn.Module wrapping `hadamard_transform`. ROADMAP SC#3 implies it should route through `_ops.hadamard_transform`. Planner verifies during scout — if it exists and imports `hadamard_transform` directly, the `fastfood.py` D-34 refactor extends to that file too.

### Reviewed Todos (not folded)
None — no pending todos surfaced for Phase 6.

</deferred>

---

*Phase: 6-hadamard Triton Port*
*Context gathered: 2026-05-27*
