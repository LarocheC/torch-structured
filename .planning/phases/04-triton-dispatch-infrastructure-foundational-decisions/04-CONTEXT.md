# Phase 4: Triton Dispatch Infrastructure & Foundational Decisions - Context

**Gathered:** 2026-05-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Lock in the dispatch infrastructure and foundational decisions that every subsequent kernel port (Phases 5-8) inherits. No Triton kernels for real ops yet — but the wrapper pattern, env-var backend selector, complex64 representation, and `_torch_ref/` package layout are all decided and demonstrated here via a no-op proof-of-concept op that exercises the full `triton_op + register_autograd + register_fake` pipeline.

**In scope:** `TORCH_STRUCTURED_BACKEND` env var; `torch_structured/_ops.py` single dispatch point; `torch_structured/_torch_ref/` peer package with `butterfly_multiply_torch` moved into it; `set_backend()` Python API; PyTorch >=2.6 floor in `pyproject.toml`; a demonstrator op proving the wrapper pattern survives `torch.compile` and `gradcheck`; CI `TRITON_CACHE_DIR` persistence; written complex64 layout decision; written CUDA deprecation plan for Phase 10.

**Out of scope:** any real Triton kernel for butterfly/hadamard/diag_mult (those are Phases 5-8); 5-stage tile design (Phase 7); bf16/fp16 support (TRI-FUT-01); `_flashmm` removal (Phase 10); actual `csrc/` deletion (post-v1.2 milestone).

</domain>

<decisions>
## Implementation Decisions

### Complex64 representation (TRI-06)

- **D-01:** Complex64 inputs are reinterpreted via `torch.view_as_real()` inside the `_ops.py` wrapper boundary (zero-copy). Triton kernels receive trailing-2 real tensors and a `IS_COMPLEX: tl.constexpr` flag. Output is `torch.view_as_complex(...)` back to the caller. Public API and nn.Module call sites continue to accept and return `complex64` exactly as today.
- **D-02:** This decision is written up in a `04-COMPLEX-LAYOUT.md` companion doc in this phase directory before the demonstrator op is built — it must be referenceable by Phase 7 when the butterfly forward kernel actually consumes the layout.
- **D-03:** The twiddle layout `(nstacks, nblocks, log_n, n/2, 2, 2)` is **not** touched (COMPAT-02). Complex64 twiddles already use this layout via `c10::complex<float>` storage; the same memory aliases to `(nstacks, nblocks, log_n, n/2, 2, 2, 2)` real (final 2 = re/im) under `view_as_real`.

### Backend dispatch and `set_backend()` (DISP-01..05)

- **D-04:** `torch_structured/_ops.py` exposes module-level callable attributes (`butterfly_multiply`, `hadamard_transform`, `diag_mult`). At import time, an internal `_resolve(env_var)` function picks one of three backend impl modules (`_triton`, `_cuda_legacy`, `_torch_ref`) and assigns its callables to the module-level names.
- **D-05:** `set_backend(name)` is a `global`-mutating function in `_ops.py` that re-runs `_resolve(name)` and reassigns the same module-level names. nn.Module consumers MUST call via `torch_structured._ops.butterfly_multiply(...)` (NOT `from torch_structured._ops import butterfly_multiply`) so the re-binding takes effect for already-loaded modules. This is a documented call-site contract — the planner adds it to the migration guidance for Phase 5 onward.
- **D-06:** `set_backend()` is intended primarily for tests. Each call site is one Python attribute access (no per-call conditional branching) — honors DISP-03 literally.
- **D-07:** `auto` precedence: Triton if importable AND CUDA device available → existing CUDA `.so` if `torch.ops.torch_structured.butterfly_multiply` is registered → pure-PyTorch `_torch_ref`. CPU-only machines skip the first two and land on `_torch_ref` directly.
- **D-08:** When `auto` resolves to Triton AND a CUDA `.so` is detected on disk (upgrade signal), `_ops.py` emits a one-time `logging.info(...)` message: *"torch_structured: selecting Triton backend; the compiled CUDA backend is still available via TORCH_STRUCTURED_BACKEND=cuda. See README for the deprecation timeline."* This is a heads-up, NOT a `DeprecationWarning` — that's reserved for explicit CUDA backend selection (DEPR-02).

### `_torch_ref/` package layout (TRI-07)

- **D-09:** Create new package `torch_structured/_torch_ref/` with `butterfly.py` containing the moved `butterfly_multiply_torch`. The old location `torch_structured/butterfly/multiply.py:28` keeps a thin shim: `from torch_structured._torch_ref.butterfly import butterfly_multiply_torch  # noqa: F401`. Existing test imports keep working unchanged.
- **D-10:** Phase 5 and Phase 6 will add `_torch_ref/diag_mult.py` and `_torch_ref/hadamard.py` (pure-PyTorch reference impls of the kernels being ported). Phase 4 only creates the package + moves butterfly.

### PyTorch floor and `triton_op` wrapper pattern (COMPAT-05, TRI-05)

- **D-11:** `pyproject.toml` bumps `dependencies = ["torch>=2.6", ...]` (was `>=2.0`). This is the `triton_op` floor and is non-negotiable.
- **D-12:** All future Triton kernels register via `@torch.library.triton_op("torch_structured::<name>", mutates_args={})` + `wrap_triton(kernel)[(grid,)](...)` + `register_autograd(backward_fn, setup_context=...)` + `register_fake(...)`. The `register_fake` meta kernel is mandatory — it's what fixes the 260419-p27 dynamo bug. `torch.autograd.Function` is forbidden for Triton paths.

### Demonstrator op (Phase 4 SC#3)

- **D-13 (Claude's discretion):** The demonstrator op is a no-op identity wrapped via the full `triton_op + register_autograd + register_fake` pipeline. Lives at `torch_structured/_ops.py` as `_demo_identity_op` (private, leading underscore) so the tests can import it. Deleted at the start of Phase 5 once `diag_mult` proves the same pattern works on a real kernel.
- **D-14 (Claude's discretion):** The demonstrator's test lives at `tests/test_dispatch.py` and covers: (a) `torch.compile(model)` traces cleanly with no graph break, (b) `gradcheck` passes, (c) the bug from 260419-p27 (`tensor has non-zero number of elements but data not allocated yet`) does NOT reproduce when the op is invoked under dynamo fake-tensor tracing.

### Deprecation plan for Phase 10 (DEPR-01..05 groundwork)

- **D-15:** Phase 4 writes a `04-DEPRECATION-PLAN.md` companion doc that Phase 10 implements verbatim. It specifies: when the `DeprecationWarning` fires (only on explicit `TORCH_STRUCTURED_BACKEND=cuda`), once per process via `warnings.simplefilter("once", DeprecationWarning)` in the `_cuda_legacy` module's import block, with `stacklevel=2`. The warning text references v1.3 (default-disabled CUDA build) and v1.4+ (csrc/ deletion).

### CI cache (TEST-05)

- **D-16 (Claude's discretion):** Planner detail — use whichever CI cache mechanism the repo already uses (GitHub Actions `actions/cache@v4` keyed on `torch.__version__` + git SHA of `_triton/` directory). If no CI config exists yet, planner creates one minimal `.github/workflows/test.yml` in Phase 4.

### Claude's Discretion

- Exact internal naming of the resolver function (`_resolve`, `_pick_backend`, etc.) — planner choice.
- Whether `set_backend()` lives at `torch_structured.set_backend` (top-level re-export) or only `torch_structured._ops.set_backend` — recommend top-level for ergonomics, but planner can revisit if it creates circular import issues.
- The exact INFO log format string from D-08 — planner can tighten the wording.
- Whether the demonstrator op's no-op identity is fp32-only or also exercises the `IS_COMPLEX` flag with a complex input — recommend both, since complex64 routing is on the critical path for Phase 7.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 4 charter (this milestone)
- `.planning/ROADMAP.md` §"Phase 4" — phase goal, depends-on, success criteria, plan count
- `.planning/REQUIREMENTS.md` §"v1.2 Requirements" — full text of DISP-01..05, COMPAT-05, TRI-05, TRI-06, TRI-07, TEST-05

### Research outputs (v1.2)
- `.planning/research/SUMMARY.md` — milestone-wide synthesis; especially "Architecture Approach" and "Critical Pitfalls"
- `.planning/research/STACK.md` — Triton/PyTorch version matrix, `triton_op` API, `triton.heuristics` vs `triton.autotune` guidance
- `.planning/research/ARCHITECTURE.md` — `_ops.py` + `_triton/<op>/` + `_torch_ref/` layout pattern; existing nn.Module surface preservation contract
- `.planning/research/PITFALLS.md` §1 (complex layout) and §3 (`triton_op` is the only viable wrapper) are load-bearing for Phase 4

### Project-level constraints
- `.planning/PROJECT.md` §"Current Milestone: v1.2" — migration strategy (parallel paths, butterfly_multiply_torch preserved)
- `.planning/PROJECT.md` §"Out of Scope" — `_flashmm` not ported; native `tl.complex64` not used

### Code-level references
- `torch_structured/butterfly/multiply.py:28` — `butterfly_multiply_torch` (the artifact being moved into `_torch_ref/`)
- `torch_structured/butterfly/butterfly.py:9` — current re-export site of `butterfly_multiply_torch`
- `torch_structured/butterfly/__init__.py:19-39` — current `torch.ops.load_library` + `check_cuda_version` (Phase 4 leaves these in place; Phase 10 deprecates)
- `csrc/butterfly.cpp:99-131` — current C++ autograd registration to be left intact in Phase 4 (the `_cuda_legacy` backend continues using it)
- `pyproject.toml` — to be edited (`torch>=2.0` → `torch>=2.6`)

### Prior-art / known issues
- `.planning/quick/260419-p27-extend-recurrent-poc-torch-compile-track/260419-p27-SUMMARY.md` — Lessons-Learned line 177: documents the dynamo fake-tensor bug the demonstrator must reproduce + fix
- PyTorch tutorial: "User-defined Triton kernels with torch.compile" (referenced in STACK.md sources)
- PyTorch dev-discuss: "Custom Ops Under torch.compile — autograd.Function vs torch.library.custom_op" (referenced in STACK.md sources)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- `butterfly_multiply_torch` at `torch_structured/butterfly/multiply.py:28` — the pure-PyTorch reference implementation that becomes the `_torch_ref` backend's `butterfly_multiply`. Not rewritten, just moved.
- Existing `torch.ops.load_library` + glob discovery in `torch_structured/butterfly/__init__.py:22-33` is the model for the new `_cuda_legacy` backend module's loader logic. Phase 4 may refactor that loader into a `_cuda_legacy/loader.py` helper if it makes the `auto` precedence cleaner.
- `check_cuda_version()` in `torch_structured/butterfly/__init__.py:42-66` — emits a warning when PyTorch/torch_structured CUDA versions diverge. Phase 4 leaves it alone; Phase 10 absorbs it into the `_cuda_legacy` backend and the `DeprecationWarning` swallows it.

### Established Patterns

- The package already uses lazy subpackage imports (`from . import combine` in `butterfly/__init__.py`) — `_ops.py` follows the same idiom for backend resolution: import the chosen backend module lazily, not all three.
- Existing tests use `pytest` parametrize but DON'T have a `backend` axis yet — Phase 4 adds a `conftest.py` fixture that parametrizes on `["torch"]` only (Triton kernels don't exist yet); Phase 5+ extend it to `["torch", "triton", "cuda"]` as kernels land.

### Integration Points

- `torch_structured/__init__.py` re-exports `Butterfly`, `ButterflyBmm`, `ButterflyBase4`, `ButterflyUnitary`, `butterfly_multiply` from `.butterfly`. Phase 4 leaves these unchanged. The `butterfly` subpackage `__init__.py` re-routes its `butterfly_multiply` import from `.multiply` to `torch_structured._ops` once Phase 5 lands the first real kernel. Phase 4 doesn't touch this routing — the existing `torch.ops.torch_structured.butterfly_multiply` path keeps serving real callers.
- `torch_structured.set_backend(...)` may need to be exported at the top level (`torch_structured/__init__.py`) for ergonomics. Planner verifies this doesn't create a circular import (likely OK because `_ops` doesn't import from the public modules).

</code_context>

<specifics>
## Specific Ideas

- Complex64 layout decision must be documented in `04-COMPLEX-LAYOUT.md` companion doc — Phase 7 reads this verbatim. Include a concrete code snippet showing the `view_as_real` reinterpret at the wrapper, the `IS_COMPLEX: tl.constexpr` flag at the kernel, and how the complex multiply is implemented inline as 4 FMAs.
- Deprecation plan must be documented in `04-DEPRECATION-PLAN.md` companion doc — Phase 10 implements it verbatim. Include the exact `warnings.warn(...)` text, the `stacklevel=2` requirement, and the `warnings.simplefilter("once", DeprecationWarning)` setup.
- The 260419-p27 dynamo bug is a specific failure mode to reproduce-then-fix: the test must call the demonstrator op inside a `torch.compile`-wrapped function and assert no `"The tensor has a non-zero number of elements, but its data is not allocated yet"` error. The fix is `register_fake`.

</specifics>

<deferred>
## Deferred Ideas

- **Top-level `torch_structured.set_backend(...)` re-export** — if the planner discovers a circular-import issue, defer to a follow-up plan; users can call `torch_structured._ops.set_backend(...)` in the meantime. Document either way.
- **AOT compilation cache shipping** — research mentioned shipping pre-compiled Triton bytecode in the wheel for common shapes. Out of scope for v1.2; would be a v1.3+ optimization (no requirement covers it).
- **Triton "interpret mode" debugging setup** — `TRITON_INTERPRET=1` is useful for development; mention in CONTRIBUTING.md but don't make it a Phase 4 deliverable.
- **Bf16/fp16 support in the demonstrator** — keep the demonstrator op fp32 + complex64 only. Bf16 lands when butterfly kernels do (TRI-FUT-01 / post-v1.2).
- **`torch.backends.torch_structured` namespace registration** — research called this out as not a standard pattern for third-party libs. Reconsider if PyTorch adds a public extension mechanism in the future.

</deferred>

---

*Phase: 4-Triton Dispatch Infrastructure & Foundational Decisions*
*Context gathered: 2026-05-26*
