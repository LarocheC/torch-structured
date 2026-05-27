# Phase 4: Triton Dispatch Infrastructure & Foundational Decisions - Research

**Researched:** 2026-05-27
**Domain:** PyTorch custom-op wrapping (torch.library.triton_op pattern), Python dispatch architecture, complex64 ABI for Triton, deprecation warnings hygiene, GitHub Actions caching for Triton JIT
**Confidence:** HIGH on the `torch.library` API contract, complex64 view semantics, Python warnings/dispatch idioms, and `actions/cache` shape. MEDIUM on autotune-vs-heuristics styling for the demonstrator (mostly aesthetic).

## Summary

Phase 4 has zero new GPU kernels — it locks in the *shapes* every subsequent kernel phase inherits. The milestone-level research (`.planning/research/`) already established the strategic direction: `torch.library.triton_op` + `register_autograd` + `register_fake` is the only viable wrapper API, real/imag-split via `view_as_real` is the only viable complex64 path, and `_torch_ref/` peer package is the only viable CPU fallback strategy. This document does the per-decision drilling the planner needs to write task-level instructions without re-research.

The drillings produced eight load-bearing artifacts: (1) the exact `triton_op` + `register_autograd` + `register_fake` skeleton, with confirmation that `op.register_fake(fn)` is a method on the resulting `CustomOpDef` (it is, verified from PyTorch source); (2) the exact `view_as_real` / `view_as_complex` autograd-and-stride contract, with the gotcha that `view_as_complex` requires stride-1 last-dim of size 2 and the twiddle re-interpret IS legal under that contract because `(nstacks, nblocks, log_n, n/2, 2, 2)` complex64 storage *already* aliases to `(..., 2, 2, 2)` real with stride-1 last dim; (3) the `warnings.simplefilter("once", DeprecationWarning)` + `stacklevel=2` incantation with the (message, category) suppression semantics verified against Python 3.14 docs; (4) the `triton.heuristics` decorator signature for the demonstrator's BLOCK_SIZE; (5) the `actions/cache@v4` (v5 also valid) YAML template keyed on `(os, python, torch, triton, kernel-source-hash)`; (6) the `set_backend()` module-mutation idiom and the corresponding call-site contract (`torch_structured._ops.butterfly_multiply(...)`, NOT `from torch_structured._ops import butterfly_multiply`); (7) the existing test idiom in `tests/test_multiply.py` is `unittest.TestCase` with hand-rolled nested-loop parametrize (not pytest fixtures), so adding a pytest-style `backend` fixture in a new `conftest.py` is purely additive; (8) the 260419-p27 dynamo bug ("The tensor has a non-zero number of elements, but its data is not allocated yet") is the literal acceptance gate for the demonstrator op test.

**Primary recommendation:** Plan 1 = `_torch_ref/` extraction + `_ops.py` resolver + `_cuda_legacy/` wrapper + `pyproject.toml` torch>=2.6 bump + companion docs (`04-COMPLEX-LAYOUT.md`, `04-DEPRECATION-PLAN.md`) so the structural changes ship in one reviewable unit. Plan 2 = demonstrator op (`_demo_identity_op`) + `tests/test_dispatch.py` + CI cache config so the structure is verified end-to-end before Phase 5 starts.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Complex64 representation (TRI-06)**
- **D-01:** Complex64 inputs are reinterpreted via `torch.view_as_real()` inside the `_ops.py` wrapper boundary (zero-copy). Triton kernels receive trailing-2 real tensors and a `IS_COMPLEX: tl.constexpr` flag. Output is `torch.view_as_complex(...)` back to the caller. Public API and nn.Module call sites continue to accept and return `complex64` exactly as today.
- **D-02:** This decision is written up in a `04-COMPLEX-LAYOUT.md` companion doc in this phase directory before the demonstrator op is built — it must be referenceable by Phase 7 when the butterfly forward kernel actually consumes the layout.
- **D-03:** The twiddle layout `(nstacks, nblocks, log_n, n/2, 2, 2)` is **not** touched (COMPAT-02). Complex64 twiddles already use this layout via `c10::complex<float>` storage; the same memory aliases to `(nstacks, nblocks, log_n, n/2, 2, 2, 2)` real (final 2 = re/im) under `view_as_real`.

**Backend dispatch and `set_backend()` (DISP-01..05)**
- **D-04:** `torch_structured/_ops.py` exposes module-level callable attributes (`butterfly_multiply`, `hadamard_transform`, `diag_mult`). At import time, an internal `_resolve(env_var)` function picks one of three backend impl modules (`_triton`, `_cuda_legacy`, `_torch_ref`) and assigns its callables to the module-level names.
- **D-05:** `set_backend(name)` is a `global`-mutating function in `_ops.py` that re-runs `_resolve(name)` and reassigns the same module-level names. nn.Module consumers MUST call via `torch_structured._ops.butterfly_multiply(...)` (NOT `from torch_structured._ops import butterfly_multiply`) so the re-binding takes effect for already-loaded modules. This is a documented call-site contract — the planner adds it to the migration guidance for Phase 5 onward.
- **D-06:** `set_backend()` is intended primarily for tests. Each call site is one Python attribute access (no per-call conditional branching) — honors DISP-03 literally.
- **D-07:** `auto` precedence: Triton if importable AND CUDA device available → existing CUDA `.so` if `torch.ops.torch_structured.butterfly_multiply` is registered → pure-PyTorch `_torch_ref`. CPU-only machines skip the first two and land on `_torch_ref` directly.
- **D-08:** When `auto` resolves to Triton AND a CUDA `.so` is detected on disk (upgrade signal), `_ops.py` emits a one-time `logging.info(...)` message. This is a heads-up, NOT a `DeprecationWarning` — that's reserved for explicit CUDA backend selection (DEPR-02).

**`_torch_ref/` package layout (TRI-07)**
- **D-09:** Create new package `torch_structured/_torch_ref/` with `butterfly.py` containing the moved `butterfly_multiply_torch`. The old location `torch_structured/butterfly/multiply.py:28` keeps a thin shim: `from torch_structured._torch_ref.butterfly import butterfly_multiply_torch  # noqa: F401`. Existing test imports keep working unchanged.
- **D-10:** Phase 5 and Phase 6 will add `_torch_ref/diag_mult.py` and `_torch_ref/hadamard.py`. Phase 4 only creates the package + moves butterfly.

**PyTorch floor and `triton_op` wrapper pattern (COMPAT-05, TRI-05)**
- **D-11:** `pyproject.toml` bumps `dependencies = ["torch>=2.6", ...]` (was `>=2.0`). Non-negotiable.
- **D-12:** All future Triton kernels register via `@torch.library.triton_op("torch_structured::<name>", mutates_args={})` + `wrap_triton(kernel)[(grid,)](...)` + `register_autograd(backward_fn, setup_context=...)` + `register_fake(...)`. `register_fake` is mandatory — it's what fixes the 260419-p27 dynamo bug. `torch.autograd.Function` is forbidden for Triton paths.

**Demonstrator op (Phase 4 SC#3)**
- **D-13:** The demonstrator op is a no-op identity wrapped via the full `triton_op + register_autograd + register_fake` pipeline. Lives at `torch_structured/_ops.py` as `_demo_identity_op` (private, leading underscore). Deleted at the start of Phase 5.
- **D-14:** Test lives at `tests/test_dispatch.py` and covers: (a) `torch.compile(model)` traces cleanly with no graph break, (b) `gradcheck` passes, (c) the 260419-p27 bug does NOT reproduce under dynamo fake-tensor tracing.

**Deprecation plan for Phase 10 (DEPR-01..05 groundwork)**
- **D-15:** Phase 4 writes a `04-DEPRECATION-PLAN.md` companion doc that Phase 10 implements verbatim. It specifies: when the `DeprecationWarning` fires (only on explicit `TORCH_STRUCTURED_BACKEND=cuda`), once per process via `warnings.simplefilter("once", DeprecationWarning)` in the `_cuda_legacy` module's import block, with `stacklevel=2`. The warning text references v1.3 and v1.4+.

**CI cache (TEST-05)**
- **D-16:** Use whichever CI cache mechanism the repo already uses (GitHub Actions `actions/cache@v4` keyed on `torch.__version__` + git SHA of `_triton/` directory). If no CI config exists yet, planner creates one minimal `.github/workflows/test.yml` in Phase 4.

### Claude's Discretion

- Exact internal naming of the resolver function (`_resolve`, `_pick_backend`, etc.) — planner choice.
- Whether `set_backend()` lives at `torch_structured.set_backend` (top-level re-export) or only `torch_structured._ops.set_backend` — recommend top-level for ergonomics, but planner can revisit if it creates circular import issues.
- The exact INFO log format string from D-08 — planner can tighten the wording.
- Whether the demonstrator op's no-op identity is fp32-only or also exercises the `IS_COMPLEX` flag with a complex input — recommend both, since complex64 routing is on the critical path for Phase 7.

### Deferred Ideas (OUT OF SCOPE)

- **Top-level `torch_structured.set_backend(...)` re-export** — if circular-import issue surfaces, defer to follow-up plan.
- **AOT compilation cache shipping** — pre-compiled Triton bytecode in the wheel for common shapes. v1.3+.
- **Triton "interpret mode" debugging setup** — mention in CONTRIBUTING.md but don't make it a Phase 4 deliverable.
- **Bf16/fp16 support in the demonstrator** — fp32 + complex64 only. Bf16 is TRI-FUT-01 / post-v1.2.
- **`torch.backends.torch_structured` namespace registration** — not a standard pattern for third-party libs.

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DISP-01 | User can select backend via `TORCH_STRUCTURED_BACKEND` env var (values: `triton`, `cuda`, `torch`, `auto`) | "`set_backend()` and `_ops.py` Resolver Pattern" section: `os.environ.get(...).lower()` + `_resolve()` with explicit value validation |
| DISP-02 | `auto` mode selects Triton → CUDA `.so` → pure-PyTorch in that order | Same section: `auto` branch tries imports in precedence order, catches `ImportError` / `AttributeError` per tier |
| DISP-03 | Backend selected once at import time via a single `_ops.py` dispatch module (no per-call branching) | Module-level assignment idiom; `global` statement in `set_backend()`; no `if` inside the bound callables |
| DISP-04 | `torch_structured.set_backend("triton"\|"cuda"\|"torch")` callable at runtime | `set_backend()` mutates module-level names; call-site contract (D-05) ensures already-loaded modules see the new binding |
| DISP-05 | Library logs selected backend at import time | `logging.info("torch_structured: backend=%s", _BACKEND)` in `_ops.py` after `_resolve()` completes |
| COMPAT-05 | PyTorch minimum bumped from `>=2.0` to `>=2.6` | "PyTorch 2.6 floor" section: `triton_op` is a 2.6 beta feature; documented in PyTorch 2.6 release blog |
| TRI-05 | All Triton kernels registered via `torch.library.triton_op` + `register_autograd` + `wrap_triton` (not `torch.autograd.Function`) | "Triton Op Skeleton" section: complete copy-paste template with verified API shapes from PyTorch source |
| TRI-06 | Complex64 implemented via real/imag-split arithmetic, layout decision documented in Phase 4 | "Complex64 ABI: `view_as_real` Contract" section: stride/contiguity requirements, autograd preservation, gotchas |
| TRI-07 | `butterfly_multiply_torch` remains as runtime fallback — not deleted | "`_torch_ref/` Migration" section: thin shim re-export from old location preserves all existing test imports |
| TEST-05 | CI persists `TRITON_CACHE_DIR` between runs | "GitHub Actions Cache for `~/.triton/cache`" section: `actions/cache@v4` YAML template with cache key composition |

</phase_requirements>

## Project Constraints (from CLAUDE.md)

The repository root `/home/claroche/torch-structured/CLAUDE.md` contains these load-bearing directives for Phase 4 planning:

- **Build system:** `pyproject.toml` is single source of truth for packaging — `pyproject.toml` edit (torch>=2.6) is the right place for the bump.
- **UV compatibility:** must work with `uv pip install` without conda. Phase 4 leaves the existing build flow intact (no Triton install in Phase 4); CI still uses `pip install -e .`.
- **CUDA support:** must retain CUDA extension compilation via `torch.utils.cpp_extension`. Phase 4 does NOT touch `csrc/`, `setup.py`, or the build-time extension compile. The `_cuda_legacy/` Python wrapper consumes the *already-compiled* `_butterfly.so`.
- **Python >=3.10, <4** — already enforced; Phase 4 does not change this.
- **PyTorch >=2.0 → >=2.6** — this is exactly the COMPAT-05 bump.

The parent-user CLAUDE.md (`~/CLAUDE.md`) also constrains: "use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists." The planner should issue beads tasks per plan, not maintain a TODO.md.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Backend selection (env var read, resolution logic) | `_ops.py` module-level | `_cuda_legacy/` loader, `_triton/` lazy-import probe | Single source of truth (DISP-03) lives at the dispatch boundary; backend modules expose callables but don't decide. |
| Complex64 ABI conversion (`view_as_real` / `view_as_complex`) | `_ops.py` wrapper boundary | n/a | Kernels receive real tensors only; the wrapper is the only legal place for the reinterpret because that's where output type contract is restored. |
| CUDA `.so` runtime loading | `_cuda_legacy/__init__.py` | `torch.ops.load_library` | Phase 4 moves the existing `_load_extension` glob into `_cuda_legacy/` — the legacy package owns the legacy artifact. |
| Pure-PyTorch reference (`butterfly_multiply_torch`) | `_torch_ref/butterfly.py` | shim at old location | Per D-09 — moved, not rewritten. Old location stays as a re-export shim. |
| `DeprecationWarning` on CUDA backend selection | `_cuda_legacy/__init__.py` (Phase 10) | n/a | Phase 4 writes the plan; Phase 10 implements. Per D-15. |
| `set_backend()` test API | `_ops.py` module-level callable | top-level `torch_structured.set_backend` re-export | Tests need a Python entry point; nn.Module consumers go through `_ops.X` (D-05). |
| `register_fake` meta kernel for the demonstrator | `_ops.py` (where `_demo_identity_op` lives) | n/a | Phase 4 only has one op; co-location is fine. Phase 5+ will move to `_triton/<op>/op.py`. |
| `nn.Module` surface (`Butterfly`, `LRU`, etc.) | `butterfly/`, `recurrent/`, `structured/` | n/a — UNCHANGED in Phase 4 | Public API stays byte-identical (COMPAT-01); only Phase 5+ swaps the consumer import lines. |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `torch` | `>=2.6` (verified current PyPI is 2.12 as of 2026-05) | `torch.library.triton_op`, `wrap_triton`, `register_autograd`, `register_fake` | `triton_op` shipped as Beta in PyTorch 2.6 ([PyTorch 2.6 release blog][1]); is *the* idiomatic post-2024 path. `[CITED: docs.pytorch.org/docs/2.6/library.html]` |
| `triton` | `>=3.2` (bundled with torch>=2.6 on CUDA Linux wheels) | Demonstrator `@triton.jit` kernel + `@triton.heuristics` for `BLOCK_SIZE` | Already pinned by torch — no need to add to `pyproject.toml` `dependencies` per `STACK.md` D-11 advice. `[CITED: pypi.org/project/triton/]` |
| `pytest` | existing dev dep | `tests/test_dispatch.py` + new `conftest.py` `backend` fixture | Already in `[project.optional-dependencies] dev`. `[VERIFIED: pyproject.toml line 35-36]` |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Python `warnings` (stdlib) | n/a | `simplefilter("once", DeprecationWarning)` for Phase 10's CUDA-import warning (planning documented in `04-DEPRECATION-PLAN.md`) | Standard idiom for once-per-process deprecation. `[CITED: docs.python.org/3/library/warnings.html]` |
| Python `logging` (stdlib) | n/a | DISP-05 INFO log at import time stating selected backend | `logging.getLogger("torch_structured").info(...)` — does not pollute stderr for users who configure logging properly. `[ASSUMED]` (standard Python practice) |
| Python `os` / `os.environ` (stdlib) | n/a | Read `TORCH_STRUCTURED_BACKEND` env var | Trivial. `[VERIFIED]` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `torch.library.triton_op` + `register_autograd` | `torch.autograd.Function` wrapping raw `@triton.jit` | Rejected by milestone research (PITFALLS.md §3); is the 260419-p27 footgun root cause. |
| `op.register_fake(fn)` method form | `@torch.library.register_fake("torch_structured::name")` decorator form | Both work; the milestone-research example used the decorator. The method form (used in PyTorch's own `triton_op` docstring example) keeps registration co-located with the op definition. **Recommend method form for the demonstrator** since the op definition is short. `[VERIFIED: torch/_library/triton.py source]` |
| `triton.heuristics({'BLOCK_SIZE': lambda args: ...})` | `triton.autotune(configs=[...], key=[...])` | Heuristics for fully-deterministic block sizes (this is the butterfly case where `n` and `log_n` fix everything); autotune for shape-dependent. **For the no-op demonstrator, neither is strictly needed — a hardcoded `BLOCK_SIZE=128` is sufficient** but adding `@triton.heuristics` exercises the full Phase 5+ pattern. `[CITED: triton-lang.org/main/python-api/generated/triton.heuristics.html]` |
| `pytest` fixtures for the new `backend` axis | `unittest.TestCase` nested loops (existing `tests/test_multiply.py` idiom) | Existing tests use nested for-loops, not pytest parametrize. Adding `conftest.py` with a pytest-style `backend` fixture is purely additive — `tests/test_dispatch.py` is a new file, can use pytest from day one. Existing files unchanged in Phase 4. `[VERIFIED: tests/test_multiply.py line 19-32]` |

**Installation:** No new packages. `torch>=2.6` and `triton` (bundled) are the only deltas.

**Version verification:**
```bash
# As of 2026-05-27, latest stable:
python -c "import torch; print(torch.__version__)"   # 2.12.x
python -c "import triton; print(triton.__version__)" # 3.7.x (bundled)
# pyproject.toml should bump to torch>=2.6 (the floor where triton_op landed)
```
`[CITED: pypi.org/project/torch/, pypi.org/project/triton/]`

## Architecture Patterns

### System Architecture Diagram

```
Phase 4 end-state — runtime call flow (after the demonstrator lands)
────────────────────────────────────────────────────────────────────

user code
    │
    ▼
nn.Module.Butterfly.forward()                           # UNCHANGED in Phase 4
    │  # Still calls butterfly_multiply from torch_structured.butterfly.multiply
    ▼
torch_structured.butterfly.multiply.butterfly_multiply  # UNCHANGED in Phase 4
    │  # Still routes through @torch.jit.script → torch.ops.torch_structured.butterfly_multiply
    ▼
C++ ButterflyMultiply::apply (existing _butterfly.so)   # UNCHANGED in Phase 4

────────────────────────────────────────────────────────────────────
Phase 4 NEW infrastructure (sitting alongside, not yet on the hot path)
────────────────────────────────────────────────────────────────────

import torch_structured                                 # triggers _ops.py import
    │
    ▼
torch_structured/_ops.py                                # NEW
    │  1. Read TORCH_STRUCTURED_BACKEND env var
    │  2. _resolve("triton"|"cuda"|"torch"|"auto")
    │  3. Bind module-level names (currently only `_demo_identity_op`)
    │  4. logging.info("backend=%s") per DISP-05
    │
    ├──> torch_structured/_triton/ (Phase 5+ kernel home; empty placeholder in Phase 4)
    ├──> torch_structured/_cuda_legacy/__init__.py  # NEW — wraps existing _butterfly.so loader
    └──> torch_structured/_torch_ref/butterfly.py   # NEW — butterfly_multiply_torch moved here

torch_structured.butterfly.multiply
    │  (thin shim) from torch_structured._torch_ref.butterfly import butterfly_multiply_torch  # noqa: F401
    ▼  # Pre-existing imports keep working: torch_structured.butterfly.multiply.butterfly_multiply_torch

torch.compile(model_with_demo_op)                       # NEW test
    │
    ▼
@triton_op("torch_structured::_demo_identity", ...)     # NEW — proves the pattern
    │  wrap_triton(_demo_identity_kernel)[grid](...)
    │  .register_autograd(_backward, setup_context=_setup_context)
    │  .register_fake(_fake_kernel)                     # the 260419-p27 fix lives here
    ▼
@triton.jit _demo_identity_kernel                       # NEW — JIT compiled on first call
```

### Recommended Project Structure

```
torch_structured/
├── __init__.py                  # +1 line: re-export set_backend (D's discretion)
├── _ops.py                      # NEW — dispatch entry point + _demo_identity_op
├── _torch_ref/                  # NEW
│   ├── __init__.py
│   └── butterfly.py             # butterfly_multiply_torch moved here from butterfly/multiply.py:28
├── _cuda_legacy/                # NEW (Phase 4 minimal version)
│   ├── __init__.py              # wraps the .so loader; exposes butterfly_multiply callable
│   └── loader.py                # OPTIONAL refactor of butterfly/__init__.py:22-33 _load_extension
├── _triton/                     # NEW empty placeholder (kernels land in Phase 5+)
│   └── __init__.py              # HAS_TRITON probe lives here
├── butterfly/
│   ├── __init__.py              # UNCHANGED (still loads _butterfly.so directly)
│   ├── multiply.py              # MODIFIED: butterfly_multiply_torch becomes a re-export shim
│   └── ... (everything else UNCHANGED)
├── structured/ recurrent/ monarch/   # UNCHANGED in Phase 4
└── factory.py                   # UNCHANGED

tests/
├── conftest.py                  # NEW — `backend` fixture parametrized over ["torch"] only in Phase 4
├── test_dispatch.py             # NEW — demonstrator op tests (compile, gradcheck, fake-tensor bug)
└── ... (everything else UNCHANGED — they still test the existing path)

.planning/phases/04-.../
├── 04-CONTEXT.md                # existing
├── 04-COMPLEX-LAYOUT.md         # NEW — per D-02
└── 04-DEPRECATION-PLAN.md       # NEW — per D-15

.github/workflows/
└── test.yml                     # NEW (or modified if exists) — actions/cache for TRITON_CACHE_DIR
```

### Pattern 1: Triton Op Skeleton (the canonical template Phase 5+ will copy)

**What:** The complete `@triton_op` + `wrap_triton` + `register_autograd` + `register_fake` registration. This is the template that gets copied into `_triton/diag_mult/op.py`, `_triton/hadamard/op.py`, `_triton/butterfly/op.py` in Phases 5-8.

**When to use:** Every kernel with gradients. The demonstrator op uses this exact skeleton.

**Example (combining the official PyTorch tutorial pattern with the `register_fake` form verified from `torch/_library/triton.py`):**

```python
# torch_structured/_ops.py  (Phase 4 demonstrator section)
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

# ─── Triton kernel (private) ────────────────────────────────────────────
@triton.heuristics(values={'BLOCK_SIZE': lambda args: triton.next_power_of_2(args['n_elements'])})
@triton.jit
def _demo_identity_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)

# ─── triton_op wrapper ──────────────────────────────────────────────────
@triton_op("torch_structured::_demo_identity", mutates_args={})
def _demo_identity_op(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = (triton.cdiv(n_elements, 1),)   # heuristics will refine via BLOCK_SIZE
    wrap_triton(_demo_identity_kernel)[grid](x, out, n_elements)
    return out

# ─── Autograd backward (identity → grad pass-through) ───────────────────
def _backward(ctx, grad):
    # Identity op: d(output)/d(input) = 1, so d(input) = grad
    return grad

def _setup_context(ctx, inputs, output):
    # No tensors to save for an identity op
    x, = inputs
    # ctx.save_for_backward(x)   # not needed for pass-through, but here for shape reference

_demo_identity_op.register_autograd(_backward, setup_context=_setup_context)

# ─── Fake/meta kernel (THE 260419-p27 fix) ──────────────────────────────
@_demo_identity_op.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)
```

**Verified signatures** `[CITED: docs.pytorch.org/docs/2.6/library.html, torch/_library/triton.py docstring]`:

- `triton_op(name, fn=None, /, *, mutates_args, schema=None)` — `name` is `"namespace::opname"`; `mutates_args={}` for a pure op; `schema` inferred from type annotations.
- `wrap_triton(kernel)[grid](...)` — `grid` can be a tuple or a callable returning a tuple. Inside `@triton_op`-decorated function only.
- `op.register_autograd(backward, setup_context=...)` — `setup_context(ctx, inputs, output)` where `inputs` is a tuple matching the op's positional args; `backward(ctx, grad)` for single-output ops, `backward(ctx, *grads)` for multi-output.
- `op.register_fake(fn)` — method on the resulting `CustomOpDef` object. Equivalent form: `@torch.library.register_fake("torch_structured::_demo_identity")` as a free-standing decorator. **The method form is verified to exist in PyTorch source** (`torch/_library/triton.py` shows `result.register_fake(fn)` in implementation).

### Pattern 2: `set_backend()` and `_ops.py` Resolver

**What:** Module-level callable attributes that get reassigned at import time (and again on `set_backend()` calls), so consumers see the latest binding through attribute access.

**When to use:** This is THE pattern for DISP-03/04/05. The planner must ensure every Phase 5+ task uses the `torch_structured._ops.X(...)` call site (not `from torch_structured._ops import X`) because module-attribute rebinding does NOT affect already-imported local names.

**Example (the `_ops.py` skeleton):**

```python
# torch_structured/_ops.py
"""Single dispatch point for kernel-backed ops.

Call sites MUST use attribute access:
    torch_structured._ops.butterfly_multiply(twiddle, x, ...)
NOT:
    from torch_structured._ops import butterfly_multiply
    butterfly_multiply(twiddle, x, ...)   # WRONG — won't see set_backend() rebind
"""
import logging
import os

import torch

log = logging.getLogger("torch_structured")

# Public, module-level callables — rebound by _resolve()
butterfly_multiply = None      # type: ignore[assignment]
hadamard_transform = None      # type: ignore[assignment]
diag_mult = None               # type: ignore[assignment]

_BACKEND = "uninitialized"

def _has_triton() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()

def _has_cuda_legacy() -> bool:
    # The C++ op is registered iff _butterfly.so loaded successfully
    return hasattr(torch.ops.torch_structured, "butterfly_multiply")

def _resolve(name: str) -> str:
    """Pick a backend and bind module-level names. Returns the chosen name."""
    global butterfly_multiply, hadamard_transform, diag_mult, _BACKEND
    name = (name or "auto").lower()
    if name not in ("triton", "cuda", "torch", "auto"):
        raise ValueError(f"Unknown backend {name!r}; expected triton|cuda|torch|auto")

    chosen = name
    if name == "auto":
        if _has_triton():
            chosen = "triton"
        elif _has_cuda_legacy():
            chosen = "cuda"
        else:
            chosen = "torch"

    if chosen == "triton":
        # In Phase 4 the _triton package is empty — fall through to torch ref for real ops.
        from torch_structured._torch_ref.butterfly import butterfly_multiply_torch
        butterfly_multiply = butterfly_multiply_torch
        # Phase 5 will replace with: from torch_structured._triton.butterfly.op import butterfly_multiply
    elif chosen == "cuda":
        from torch_structured._cuda_legacy import butterfly_multiply as _cuda_bm
        butterfly_multiply = _cuda_bm
    elif chosen == "torch":
        from torch_structured._torch_ref.butterfly import butterfly_multiply_torch
        butterfly_multiply = butterfly_multiply_torch

    # hadamard_transform, diag_mult: Phase 6, Phase 5 respectively — stub for now.

    # DISP-08 heads-up log when auto picks triton but a .so is also present
    if name == "auto" and chosen == "triton" and _has_cuda_legacy():
        log.info(
            "torch_structured: selecting Triton backend; the compiled CUDA backend is "
            "still available via TORCH_STRUCTURED_BACKEND=cuda. "
            "See README for the deprecation timeline."
        )

    _BACKEND = chosen
    return chosen

def set_backend(name: str) -> str:
    """Public API: switch backend at runtime (primarily for tests).

    Returns the resolved backend name (useful when name='auto').
    """
    chosen = _resolve(name)
    log.info("torch_structured: backend=%s (set_backend)", chosen)
    return chosen

# Import-time resolution (DISP-03)
_initial = os.environ.get("TORCH_STRUCTURED_BACKEND", "auto")
_resolve(_initial)
log.info("torch_structured: backend=%s (import)", _BACKEND)   # DISP-05
```

**Why the call-site contract matters:** Python's `from X import Y` binds `Y` in the caller's namespace at import time. If `X.Y` is later reassigned, the caller's `Y` still points at the old object. Module attribute lookup (`X.Y`) re-reads the binding on each access. The `set_backend()` semantics require attribute access at every call site. The planner must document this in every Phase 5+ task that touches a consumer (`butterfly/butterfly.py`, `structured/hadamard.py`, etc.).

### Pattern 3: Complex64 ABI (`view_as_real` / `view_as_complex` at the Wrapper Boundary)

**What:** Inside the `_ops.py` wrapper for any complex-accepting op, the wrapper does `x_real = torch.view_as_real(x_complex).contiguous()` if needed, calls a real-only Triton kernel, then `out_complex = torch.view_as_complex(out_real)` to restore the public type.

**Verified contract** `[CITED: docs.pytorch.org/docs/2.6/generated/torch.view_as_real.html, view_as_complex.html, torch/derivatives.yaml]`:

1. **Zero-copy:** Both return *views* sharing storage. Per derivatives.yaml: `view_as_real` backward is `at::view_as_complex(grad.contiguous())`, and `view_as_complex` backward is `at::view_as_real(grad.contiguous().resolve_conj())` — confirming both are differentiable views with explicit autograd wiring.
2. **Stride contract on `view_as_complex`:** "the tensor must have a stride of 1 for its last dimension. The strides of all other dimensions must be even numbers." And the last dim size must be exactly 2.
3. **`view_as_real` accepts any complex tensor** and appends a trailing dim of size 2. The result has stride-1 on the new last dim; *other dims' strides are doubled* (because each complex element is 2 reals).
4. **Autograd flows through both** — they are listed as "view functions with metadata change" in PyTorch's autograd system. `[VERIFIED: PyTorch derivatives.yaml]`

**The twiddle re-interpret IS legal** under (2): the twiddle layout `(nstacks, nblocks, log_n, n/2, 2, 2)` complex64 has last-dim stride 1 (it's the innermost 2×2 block stored contiguously). Reinterpreting as `(nstacks, nblocks, log_n, n/2, 2, 2, 2)` real via `view_as_real` is the canonical case. Re-packing the kernel output via `view_as_complex` requires the output tensor to be allocated with `.contiguous()` in the wrapper.

**Critical gotcha (PITFALLS Pitfall 11):** Sometimes a user passes a non-contiguous complex tensor (e.g., after a `.transpose()` in `butterfly.py:126`). The wrapper MUST `.contiguous()` before `view_as_real`, or the trailing-2 view will have wrong strides and the kernel will read garbage. Verified by:

```python
import torch
x = torch.randn(4, 4, dtype=torch.complex64)
xt = x.transpose(0, 1)         # non-contiguous
torch.view_as_real(xt)         # works but stride is the transposed pattern
# Right thing: torch.view_as_real(xt.contiguous())  # forces a copy if needed
```

**Recommendation for the demonstrator's complex pass:** Add a single test case `_demo_identity_op(complex_input)` where `complex_input = torch.randn(N, dtype=torch.complex64, device='cuda')`. The wrapper does `view_as_real → kernel → view_as_complex`. This proves Phase 7's critical path is sound BEFORE Phase 7 starts.

### Pattern 4: `DeprecationWarning` — `once` Semantics

**What:** Phase 10's `_cuda_legacy/__init__.py` (to be implemented per Phase 4's `04-DEPRECATION-PLAN.md` doc) will emit a single `DeprecationWarning` per process when a user explicitly selects `TORCH_STRUCTURED_BACKEND=cuda`.

**Verified incantation** `[CITED: docs.python.org/3/library/warnings.html, Python 3.14]`:

```python
# torch_structured/_cuda_legacy/__init__.py (Phase 10 — Phase 4 only documents)
import warnings

# Module-level filter setup: "once" suppresses repeats based on (message, category)
# IGNORING module and line number. So even if multiple call sites import this
# module, the warning fires exactly once per process.
warnings.simplefilter("once", DeprecationWarning)

warnings.warn(
    "torch_structured: the CUDA C++ backend (csrc/) is deprecated and will be "
    "default-disabled in v1.3, with full removal in v1.4+. "
    "Switch to TORCH_STRUCTURED_BACKEND=triton (default in v1.2). "
    "See the v1.2 release notes for migration guidance.",
    DeprecationWarning,
    stacklevel=2,
)
```

**Why `stacklevel=2`:** With stacklevel=1 (default), the warning is attributed to the line *inside* `_cuda_legacy/__init__.py` containing `warnings.warn(...)`. With `stacklevel=2`, it's attributed to the *importer* — which is `_ops.py`'s `from torch_structured._cuda_legacy import ...`. This makes the warning actionable: the user sees their import line, not Python guts.

**The `__init__.py` gotcha** (from Python docs): When `stacklevel=2` is used at module top level (not inside a function), the attribution is to *whoever imported the module*, which for a once-per-process import is exactly what we want. But if `_cuda_legacy/__init__.py` *re-runs* the warn at every access (it shouldn't — it's at module top level so only fires on first import), `stacklevel=2` would attribute it to the importer of *that* call. Since module top-level code runs exactly once, this is the correct behavior. `[CITED: Python warnings docs § 'Repeated Warning Suppression Criteria']`

**`simplefilter("once", DeprecationWarning)` scope:** "A warning is considered a repeat if the (message, category) are the same, ignoring the module and line number." This is *per process*, not per session. Calling `simplefilter` at module top-level installs the filter for the rest of the process. `[CITED: Python warnings docs § 'The Warnings Filter']`

**Document this verbatim in `04-DEPRECATION-PLAN.md`** so Phase 10 doesn't re-research.

### Pattern 5: GitHub Actions Cache for `~/.triton/cache`

**What:** Persist Triton's JIT cache across CI runs so first-call compile cost doesn't compound for every PR. Without this, PITFALLS Pitfall 5 ("First-Call JIT and Autotune Cost Tanks CI Wall Time") bites the moment Phase 5 lands.

**Triton cache details** `[CITED: next.redhat.com/2025/05/16/understanding-triton-cache..., github.com/triton-lang/triton/issues/4265]`:

- **Default location:** `~/.triton/cache/` (can be overridden via `TRITON_CACHE_DIR` env var).
- **Contents:** TTIR, TTGIR, LLIR (Triton/LLVM IR), platform code (PTX/AMDGCN), compiled binaries (CUBIN/HSACO), metadata JSON.
- **Cache key:** Includes Triton version hash, function signature, constants, GPU backend options, and selected env vars. Upgrading Triton invalidates everything.
- **Autotune timings:** NOT stored in this cache by default — `cache_results=True` on `@triton.autotune` persists timings separately. Phase 4 demonstrator doesn't autotune.

**Recommended `.github/workflows/test.yml` snippet** `[CITED: docs.github.com/actions/cache]`:

```yaml
name: Test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Cache Triton JIT artifacts
        uses: actions/cache@v4
        with:
          path: ~/.triton/cache
          key: triton-${{ runner.os }}-py${{ env.PYTHON_VERSION }}-torch${{ env.TORCH_VERSION }}-${{ hashFiles('torch_structured/_triton/**/*.py') }}
          restore-keys: |
            triton-${{ runner.os }}-py${{ env.PYTHON_VERSION }}-torch${{ env.TORCH_VERSION }}-
            triton-${{ runner.os }}-py${{ env.PYTHON_VERSION }}-
      - run: pip install -e .[test]
      - run: pytest tests/ -x
```

**Cache key composition rationale:**
- `runner.os` — Linux/macOS/Windows have different PTX.
- `python-version` — Triton compiles different bytecode for different Python ABI.
- `torch-version` — Triton is bundled with torch; bumping torch invalidates.
- `hashFiles('torch_structured/_triton/**/*.py')` — Source-hash component so kernel edits invalidate the cache (Phase 5+).
- `restore-keys` — Falls back to older caches on miss (e.g., partial match on torch version).

**Two important verifications:**
1. **`actions/cache@v4` is fully backward compatible with v3**; v5 also exists (released April 2026, Node 24 runtime) but v4 is the de facto standard. The CONTEXT.md D-16 specifies v4 explicitly — honor it.  `[CITED: github.com/actions/cache]`
2. **Cache does NOT survive `pip install -U torch_structured`** for users (only matters for our CI). Triton key includes Triton version, so version bumps auto-invalidate.

**Phase 4 reality check:** Phase 4 has NO real Triton kernels — the demonstrator's compile cost is trivial. The cache infrastructure exists in Phase 4 because (a) it's TEST-05 requirement, (b) Phase 5 will immediately exercise it with the first real kernel.

### Anti-Patterns to Avoid

- **`from torch_structured._ops import butterfly_multiply` at consumer sites** — breaks `set_backend()` because the name binds to the *current* object, not the *future* one after rebinding. Always use `torch_structured._ops.butterfly_multiply(...)`. Phase 5 task author must enforce this.
- **`torch.autograd.Function` wrapping the demonstrator** — Phase 4's whole point is proving `triton_op` works. Falling back to `autograd.Function` defeats the demonstrator.
- **Eagerly importing `_triton` at top-level `torch_structured/__init__.py`** — Triton import is expensive (loads LLVM). Lazy-import only inside `_ops.py` behind `try/except`. `__init__.py` should NEVER touch `_triton/`.
- **Skipping `register_fake`** — this is the literal 260419-p27 bug fix. The demonstrator MUST register it; the planner's verification must assert the dynamo-tracing path doesn't raise.
- **Adding `triton` to `pyproject.toml` `dependencies`** — `torch>=2.6` already drags it in on CUDA Linux. Adding our own pin fights pip's resolver against the `pytorch-triton` / `triton` package name split. Import-guard at runtime instead. `[VERIFIED: STACK.md "What NOT to Use"]`

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Custom-op + autograd + compile composability | Subclass `torch.autograd.Function`, override `forward`/`backward`, hope `torch.compile` traces it | `@torch.library.triton_op` + `op.register_autograd(...)` + `op.register_fake(...)` | Documented foot-gun. Causes 260419-p27. Two-line difference in the wrapper saves weeks of dynamo debugging. |
| Complex64 in Triton | Write a `tl.complex64` shim, pack `(re, im)` manually with hand-coded stride math | `torch.view_as_real(x).contiguous()` at the wrapper boundary | Triton has no `tl.complex*`; view_as_real is zero-copy with explicit autograd backward in derivatives.yaml. |
| Once-per-process deprecation warning | Module-level `_warned = False` + `if not _warned: warn(...); _warned = True` | `warnings.simplefilter("once", DeprecationWarning)` + plain `warnings.warn(...)` | Python's warnings module already implements the per-process dedup with (message, category) keying. Custom flag misses `warn_only=True` semantics, doesn't compose with pytest's warning capture, doesn't integrate with `python -W` CLI flags. |
| BLOCK_SIZE selection for the demonstrator | Hardcode 128, hope for the best on edge sizes | `@triton.heuristics(values={'BLOCK_SIZE': lambda args: triton.next_power_of_2(args['n_elements'])})` | Built-in, deterministic, no autotune cost. Sets the pattern Phase 5+ will reuse. |
| CI cache for Triton JIT | Custom rsync-to-S3 / GHA artifact dance | `actions/cache@v4` with `path: ~/.triton/cache` | Standard GHA tooling; restore-keys provides graceful fallback on partial matches. |
| Backend probe at every call | `if has_triton(): ... else: ...` inside every consumer's forward | Module-level binding in `_ops.py` resolved once at import | DISP-03 mandates "no per-call branching." Also makes test parametrization a one-line `_ops.set_backend("torch")` instead of monkeypatching every consumer. |
| Reading `TORCH_STRUCTURED_BACKEND` from inside individual modules | Each consumer does its own `os.environ.get(...)` | `_ops.py` reads once at import; consumers just call the bound name | DISP-01..03 single dispatch point. |

**Key insight:** Phase 4's deliverables are *plumbing*. Hand-rolling any of the above moves Phase 4 from "infrastructure that holds up for 6 more phases" to "infrastructure that breaks the moment Phase 5 stresses it." Use the standard tools.

## Runtime State Inventory

Phase 4 is NOT a rename/refactor — it's *additive* infrastructure plus one file move (`butterfly_multiply_torch` from `butterfly/multiply.py:28` to `_torch_ref/butterfly.py`). However, the file move has a state-inventory dimension because the existing test suite imports the old location.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — no databases, no persistent state. | None. |
| Live service config | None — no live services. | None. |
| OS-registered state | None — no OS-level registrations. | None. |
| Secrets / env vars | `TORCH_STRUCTURED_BACKEND` is a NEW env var being introduced. No pre-existing one. No SOPS/secrets implications. | None — purely additive. |
| Build artifacts / installed packages | The existing `_butterfly.so` and `_version.so` in the installed package directory MUST keep working after Phase 4 — `_cuda_legacy/` consumes them. **Stale `.egg-info` from prior `pip install -e .` should be re-installed if `pyproject.toml` torch bump changes resolution.** | Document in Phase 4 plan task: after `pyproject.toml` edit, run `pip install -e . --force-reinstall --no-deps` to refresh `.egg-info`. |
| Test imports referencing the old location | `tests/test_multiply.py` line 40, 78: `torch_structured.butterfly.multiply.butterfly_multiply_torch(...)` — qualified path through the old location | **Handled by the shim** (D-09): `butterfly/multiply.py` keeps a `from torch_structured._torch_ref.butterfly import butterfly_multiply_torch  # noqa: F401`. Verified by `grep -rn 'butterfly_multiply_torch' tests/` shows usage stays valid. |
| Test imports referencing `butterfly.multiply` directly | `tests/test_multiply_base4.py:11`: `from torch_structured.butterfly.multiply import butterfly_multiply_torch` | Same shim covers this. The shim re-exports the moved name. |

**The canonical question — answer:** After moving `butterfly_multiply_torch` to `_torch_ref/`, every existing test import (`torch_structured.butterfly.multiply.butterfly_multiply_torch` and `from torch_structured.butterfly.multiply import butterfly_multiply_torch`) keeps working because `butterfly/multiply.py` re-exports the name. The planner's first verification on Plan 1 is `pytest tests/ -k "test_multiply"` should pass unchanged.

## Common Pitfalls

### Pitfall 1: Forgetting `register_fake` on the Demonstrator

**What goes wrong:** The demonstrator op gets a `triton_op` decorator and `register_autograd`, but the planner forgets `register_fake`. The demonstrator works in eager. The 260419-p27 test (D-14c) fails with *exactly* the bug we're trying to fix: "The tensor has a non-zero number of elements, but its data is not allocated yet" inside dynamo fake-tensor tracing.

**Why it happens:** The official PyTorch tutorial for `triton_op` does NOT include `register_fake` in its example. A planner who copies from the tutorial verbatim will miss this. The Lei Mao blog post also omits it. Only the Torch-TensorRT docs and the PyTorch source-code docstring for `triton_op` mention it.

**How to avoid:** Treat `register_fake` as MANDATORY in every Phase 5+ task. The Phase 4 demonstrator is the template — if it's missing here, every kernel that copies the pattern misses it too.

**Warning signs:** `torch.compile(model_with_demo_op)(x)` raises `"data is not allocated yet"`, or graph-breaks with "speculate_subgraph on bw failed."

### Pitfall 2: `from torch_structured._ops import X` in Phase 5+ Consumer

**What goes wrong:** Phase 5 lands `diag_mult` Triton kernel. The consumer (`structured/krylov.py`) does `from torch_structured._ops import diag_mult`. Now in a test, `_ops.set_backend("torch")` rebinds `_ops.diag_mult` to the torch-ref impl — but `structured/krylov.py`'s local `diag_mult` still points at the Triton version. Tests pass under default backend, fail under `set_backend("torch")` with confusing "module not the one I thought" errors.

**Why it happens:** Python `from X import Y` semantics: `Y` is bound in the caller's namespace at import time. Subsequent rebinds to `X.Y` are invisible to the caller.

**How to avoid:** The planner MUST document this contract in the Phase 5 plan (and every subsequent kernel-port phase). The Phase 4 plan should add the contract to the `04-COMPLEX-LAYOUT.md` companion or to a top-level CONTRIBUTING.md note. Recommended idiom for consumers:

```python
# CORRECT — at the top of structured/krylov.py
from torch_structured import _ops

def some_function(x):
    return _ops.diag_mult(x, diag)   # attribute access — sees rebinding

# WRONG — same file
from torch_structured._ops import diag_mult   # binds the CURRENT object
```

**Warning signs:** Tests pass with `TORCH_STRUCTURED_BACKEND=triton` (the default) but fail when explicitly setting `TORCH_STRUCTURED_BACKEND=torch` in CI.

### Pitfall 3: `view_as_real` on a Non-Contiguous Tensor

**What goes wrong:** A consumer passes a complex tensor that's the result of a `.transpose(-1, -2)` (this happens in `butterfly/butterfly.py:126` for the transpose path). The wrapper does `view_as_real(x)` without `.contiguous()`. The trailing-2 view has stride pattern inherited from the transpose. The kernel reads garbage; output is silently wrong.

**Why it happens:** `view_as_real` and `view_as_complex` are *views* — they don't force layout. The kernel expects packed (re, im) in the last dim with stride 1.

**How to avoid:** ALWAYS `.contiguous()` before `view_as_real` in the wrapper. Add a `assert x.is_contiguous(), "complex input must be contiguous before view_as_real"` precondition. Document this in `04-COMPLEX-LAYOUT.md`.

**Warning signs:** Complex tests pass for non-transposed cases but fail for `Butterfly.forward(input, transpose=True, complex=True)`.

### Pitfall 4: `set_backend()` Doesn't Reload `_triton.*` Submodules

**What goes wrong:** First `set_backend("triton")` succeeds — `_resolve()` imports `_triton.butterfly.op` and binds the name. Second `set_backend("torch")` rebinds to `_torch_ref`. Now `set_backend("triton")` again — the import statement is a no-op (Python's module cache), so the binding picks up the *same* function object as the first time. This is actually FINE for our use case (the Triton functions are stateless), but it would be a bug if `_triton.butterfly.op` had module-level state. **No action needed in Phase 4** but document the assumption.

**Why it happens:** Python caches imported modules in `sys.modules`. `import X` is cheap on repeat but doesn't re-execute X's top-level code.

**How to avoid:** Don't put mutable state in `_triton/` module top-levels. Keep kernel registration idempotent. This is the natural pattern anyway.

### Pitfall 5: Logging Output Pollution in Tests

**What goes wrong:** DISP-05's `logging.info("torch_structured: backend=%s", ...)` fires on every test that imports `torch_structured`. CI logs get cluttered; pytest captures it but it shows on failure.

**Why it happens:** Without a handler configured, Python's `logging.info` goes to a `lastResort` handler that prints to stderr. This is correct behavior for end users (they want to see the backend) but noisy for tests.

**How to avoid:** Use `logging.getLogger("torch_structured").info(...)` (named logger). The default handler is the `lastResort` only if no parent logger has a handler — applications can suppress with `logging.getLogger("torch_structured").setLevel(logging.WARNING)` if desired. In `tests/conftest.py`, optionally add `logging.getLogger("torch_structured").setLevel(logging.WARNING)` for the test session. `[ASSUMED]` — standard Python logging idiom.

### Pitfall 6: CI Cache Hit Rate is 0% Because Key Includes Random Hashes

**What goes wrong:** Planner writes a cache key like `triton-${{ runner.os }}-${{ github.sha }}` thinking it'll be useful. Every commit has a different `github.sha` → 0% hit rate → cache is effectively never used.

**Why it happens:** Cache keys must be *stable across the dimension you want to share*. Source hash should only be in the key for files that, when changed, *invalidate* the cache (i.e., Triton kernel source). Commit SHA changes per-commit — wrong.

**How to avoid:** Use `hashFiles('torch_structured/_triton/**/*.py')` for the source-hash portion, NOT `github.sha`. Cache hits across commits that don't touch Triton kernel sources. Verified pattern shown in Pattern 5 above.

## Code Examples

### Complete demonstrator op (copy-paste skeleton for Phase 4 Plan 2)

```python
# torch_structured/_ops.py — Phase 4 demonstrator section
# Source: composed from official PyTorch tutorial
# https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html
# and verified register_fake method from torch/_library/triton.py docstring

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.heuristics(values={'BLOCK_SIZE': lambda args: triton.next_power_of_2(min(args['n_elements'], 1024))})
@triton.jit
def _demo_identity_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)


@triton_op("torch_structured::_demo_identity", mutates_args={})
def _demo_identity_op(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    wrap_triton(_demo_identity_kernel)[grid](x, out, n_elements)
    return out


def _setup_context(ctx, inputs, output):
    pass  # identity op needs nothing saved


def _backward(ctx, grad):
    return grad   # d(out)/d(in) = 1


_demo_identity_op.register_autograd(_backward, setup_context=_setup_context)


@_demo_identity_op.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)
```

### Complete test for the demonstrator (copy-paste skeleton for Phase 4 Plan 2)

```python
# tests/test_dispatch.py
# Source: combines decisions D-13 and D-14
import pytest
import torch

from torch_structured._ops import _demo_identity_op


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="demo op is GPU-only")


def test_demo_identity_eager_fp32():
    x = torch.randn(128, device='cuda', requires_grad=True)
    y = _demo_identity_op(x)
    assert torch.equal(y, x)


def test_demo_identity_eager_complex64():
    # Exercises the view_as_real path through the wrapper boundary
    x = torch.randn(128, dtype=torch.complex64, device='cuda', requires_grad=True)
    # If the demonstrator's wrapper handles complex via view_as_real:
    y = _demo_identity_op(x)   # may need wrapper to recognise complex
    assert torch.equal(y, x)


def test_demo_identity_gradcheck():
    # gradcheck uses fp64 by default — declare the op accepts it via type promotion
    x = torch.randn(16, dtype=torch.float64, device='cuda', requires_grad=True)
    assert torch.autograd.gradcheck(_demo_identity_op, (x,), eps=1e-6, atol=1e-5)


def test_demo_identity_compile_no_graph_break():
    """The 260419-p27 fix verification: register_fake must let dynamo trace through."""
    @torch.compile(fullgraph=True)   # fullgraph=True raises on graph break
    def f(x):
        return _demo_identity_op(x) * 2

    x = torch.randn(128, device='cuda', requires_grad=True)
    y = f(x)
    assert torch.allclose(y, x * 2)


def test_demo_identity_compile_fake_tensor_trace():
    """Explicit fake-tensor trace — the literal 260419-p27 reproducer."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x = torch.empty(128, device='cuda', requires_grad=True)
        # MUST NOT raise "The tensor has a non-zero number of elements, but its data is not allocated yet"
        y = _demo_identity_op(x)
        assert y.shape == x.shape
        assert y.dtype == x.dtype
```

### Minimal `_torch_ref/butterfly.py` (Plan 1)

```python
# torch_structured/_torch_ref/butterfly.py
# Moved verbatim from torch_structured/butterfly/multiply.py:28
# Source: existing repo, no logic change

import torch
from torch.nn import functional as F


def butterfly_multiply_torch(twiddle, input, increasing_stride=True, output_size=None):
    batch_size, nstacks, input_size = input.shape
    nblocks = twiddle.shape[1]
    log_n = twiddle.shape[2]
    n = 1 << log_n
    assert twiddle.shape == (nstacks, nblocks, log_n, n // 2, 2, 2)
    input = F.pad(input, (0, n - input_size)) if input_size < n else input[:, :, :n]
    output_size = n if output_size is None else output_size
    assert output_size <= n
    output = input.contiguous()
    cur_increasing_stride = increasing_stride
    for block in range(nblocks):
        for idx in range(log_n):
            log_stride = idx if cur_increasing_stride else log_n - 1 - idx
            stride = 1 << log_stride
            t = twiddle[:, block, idx].view(
                nstacks, n // (2 * stride), stride, 2, 2).permute(0, 1, 3, 4, 2)
            output_reshape = output.view(
                batch_size, nstacks, n // (2 * stride), 1, 2, stride)
            output = (t * output_reshape).sum(dim=4)
        cur_increasing_stride = not cur_increasing_stride
    return output.view(batch_size, nstacks, n)[:, :, :output_size]
```

### Re-export shim at `butterfly/multiply.py` (Plan 1)

```python
# torch_structured/butterfly/multiply.py — Phase 4 edit
# Move butterfly_multiply_torch to _torch_ref/, keep a shim here for test imports.

import math
from typing import Tuple, Optional

import torch
from torch.nn import functional as F

# Phase 4: butterfly_multiply_torch now lives in _torch_ref/. Re-export here
# so existing test imports (torch_structured.butterfly.multiply.butterfly_multiply_torch)
# keep working unchanged.
from torch_structured._torch_ref.butterfly import butterfly_multiply_torch  # noqa: F401


@torch.jit.script
def butterfly_multiply_fw(twiddle: torch.Tensor, input: torch.Tensor, increasing_stride: bool,
                          output_size: Optional[int] = None) -> torch.Tensor:
    return torch.ops.torch_structured.butterfly_multiply_fw(twiddle, input, increasing_stride,
                                                            output_size)


@torch.jit.script
def butterfly_multiply_bw(twiddle: torch.Tensor, input: torch.Tensor, grad: torch.Tensor,
                          increasing_stride: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_structured.butterfly_multiply_bw(twiddle, input, grad, increasing_stride)


@torch.jit.script
def butterfly_multiply(twiddle: torch.Tensor, input: torch.Tensor, increasing_stride: bool,
                       output_size: Optional[int] = None) -> torch.Tensor:
    return torch.ops.torch_structured.butterfly_multiply(twiddle, input, increasing_stride,
                                                          output_size)
```

### `conftest.py` with `backend` fixture (Plan 2)

```python
# tests/conftest.py
# Phase 4: backend fixture parametrized over ["torch"] only.
# Phase 5+ will extend to ["torch", "triton", "cuda"] as kernels land.
import pytest

import torch_structured


@pytest.fixture(params=["torch"])
def backend(request, monkeypatch):
    """Switch backend for the duration of a test, restore after."""
    original = torch_structured._ops._BACKEND
    chosen = torch_structured._ops.set_backend(request.param)
    yield chosen
    torch_structured._ops.set_backend(original)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torch.autograd.Function` + raw `@triton.jit` calls in `forward`/`backward` | `@torch.library.triton_op` + `register_autograd` + `register_fake` | PyTorch 2.6 (Dec 2024) | The 260419-p27 dynamo bug; required floor bump. |
| `torch::autograd::Function` in C++ (`csrc/butterfly.cpp:99`) | `register_autograd` (pure Python) | Same as above | Removes need for compiled extensions for autograd. Out-of-scope for Phase 4 (Phase 10 cleanup). |
| `torch.jit.script` decorators on Python wrappers (`multiply.py:8,15,21`) | Plain Python wrappers + `torch.compile` (Inductor) | TorchScript deprecated PyTorch 2.10 | Phase 4 leaves them in place because `multiply.py` is on the *existing* CUDA path which Phase 4 doesn't touch. Phase 5+ replaces. |
| `torch.ops.load_library` + glob discovery in `butterfly/__init__.py:22-33` | Same loader moved into `_cuda_legacy/loader.py` | Phase 4 D-09 | Refactor only; no behavior change. |
| `TORCH_CUDA_ARCH_LIST` build-time arch selection | Triton JIT compiles for the running GPU's arch at first call | Phase 5+ when Triton kernels land | Phase 4 still uses ARCH_LIST for the existing `.so` build. |
| Module-level `triton_op` example missing `register_fake` (official tutorial as of 2026-05) | `register_fake` is mandatory for compile-friendliness | n/a — official tutorial just hasn't been updated | Phase 4 demonstrator MUST include it. |

**Deprecated/outdated:**
- **`torch.autograd.Function` for custom ops with Triton:** PyTorch dev-discuss explicitly recommends `triton_op` post-2024. Don't use the old pattern in any new code.
- **`make_block_ptr`:** Deprecated in Triton 3.7 (emits warning). Use plain `tl.load/tl.store` with `mask=`. Not relevant for Phase 4 demonstrator.
- **`TORCH_LIBRARY(...)` C++ block for new ops:** Pure-Python `triton_op` is the recommended path for kernels that have a Triton implementation.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `logging.getLogger("torch_structured").info(...)` is the right idiom for DISP-05 (vs `print` or root logger) | Pattern 2, Pitfall 5 | LOW — at worst, log output is mildly noisier than intended; easy to tweak post-merge. Standard Python practice. |
| A2 | The demonstrator's `register_fake` method-form (vs `@torch.library.register_fake("ns::name")` decorator form) is preferred for co-location | Standard Stack alternatives table | LOW — both forms work; decorator form is documented equally well. Planner can swap if planner prefers. |
| A3 | The grad rule for identity is literal `return grad` (no copy needed) for an identity op via `triton_op` | Code Examples / `_backward` | LOW — verified semantically against the PyTorch tutorial's `mysin` example pattern. |
| A4 | `triton.heuristics` decorator goes ABOVE `@triton.jit` (not below) | Pattern 1 demonstrator | VERIFIED via official Triton docs (`docs.pytorch.org` and `triton-lang.org/main/python-api/generated/triton.heuristics.html` example shows `@triton.heuristics` then `@triton.jit`). Not actually an assumption. |
| A5 | The exact `triton` minimum version pinned by `torch>=2.6` is `triton>=3.2` (not 3.0 or 3.1) | Standard Stack | LOW — verified via the version compatibility matrix in STACK.md and PyPI. If wrong by one minor version, no impact on Phase 4 (no real kernels). |
| A6 | `actions/cache@v4` is preferred over `v5` for Phase 4 because CONTEXT.md D-16 names v4 explicitly | Pattern 5 | LOW — both versions have the same YAML surface for this use case. CONTEXT.md is authoritative. |
| A7 | Editable install (`pip install -e .`) does NOT need rebuilding after the torch>=2.6 pin bump as long as torch is already 2.6+ in the environment | Project Constraints | MEDIUM — true for `pip install -e .` semantics. If user's env has torch==2.5, `pip install -e .` will fail with resolver error — which is exactly the desired hard-floor behavior. |
| A8 | The `_torch_ref/` move + shim does NOT break `tests/test_butterfly_base4.py:11` import | Runtime State Inventory | VERIFIED via `grep`: `tests/test_multiply_base4.py:11` does `from torch_structured.butterfly.multiply import butterfly_multiply_torch` — the shim re-exports the symbol at the same path. |

**Open question requiring user confirmation:** A1 and A6 are stylistic; A2 is a planner-discretion choice; the rest are verified. No load-bearing assumptions need user signoff before planning.

## Open Questions (RESOLVED)

1. **Should `_demo_identity_op` exercise complex64 in Phase 4 or only fp32?**
   - What we know: D-14 (Claude's discretion) recommends exercising complex64 too, since complex routing is on Phase 7's critical path.
   - What's unclear: doing so requires the wrapper to do `view_as_real → kernel → view_as_complex`, which is mini-Pattern-3 in miniature. Adds ~10 LOC to the demonstrator.
   - Recommendation: **DO include complex64**, because the wrapper boundary is exactly what's being demonstrated. Phase 7 reading `04-COMPLEX-LAYOUT.md` will appreciate having a working example. The planner should add a fourth task to Plan 2: "demonstrator op accepts complex64 via view_as_real and survives gradcheck." If the planner's task budget is tight, defer to a Phase 5 task.
   - **RESOLVED:** include complex64 in the demonstrator op via the view_as_real/view_as_complex branch — Plan 04-02 Task 1 will bake this into `_demo_identity_op` unconditionally.

2. **Top-level `torch_structured.set_backend` re-export — does it create a circular import?**
   - What we know: `torch_structured/__init__.py` imports from `.butterfly`, which imports from `.multiply`, which (after Phase 4) imports from `._torch_ref.butterfly`. None of these touch `_ops`. So adding `from ._ops import set_backend` to `__init__.py` should be safe.
   - What's unclear: `_ops._resolve()` at import time calls `from ._torch_ref.butterfly import butterfly_multiply_torch` — which is a sibling package. As long as `_torch_ref/` doesn't import back from `_ops`, no cycle.
   - Recommendation: **Add the top-level re-export.** Verify by `python -c "import torch_structured; torch_structured.set_backend('torch')"` as a Phase 4 task verification.
   - **RESOLVED:** add the top-level `from ._ops import set_backend` re-export in Plan 04-01 Task 2; the verification command is part of that task's acceptance criteria — no cycle exists.

3. **Should `_cuda_legacy/` be a full refactor of `butterfly/__init__.py:22-33` _load_extension, or a minimal wrapper that imports from there?**
   - What we know: `butterfly/__init__.py` already loads the `.so`. Duplicating loader logic risks divergence.
   - Recommendation: **Minimal wrapper** for Phase 4. `_cuda_legacy/__init__.py` does `from torch_structured.butterfly.multiply import butterfly_multiply as butterfly_multiply` (the C++ path) and re-exposes. Phase 10 does the full refactor when `butterfly/__init__.py` collapses.
   - **RESOLVED:** Plan 04-01 Task 1 ships the minimal wrapper (`_cuda_legacy/butterfly.py` is a pass-through to `torch.ops.torch_structured.butterfly_multiply`); the full collapse of `butterfly/__init__.py` is deferred to Phase 10 per D-15.

4. **What's the expected behavior of `set_backend("triton")` in Phase 4 when no real Triton kernel exists?**
   - What we know: `_resolve("triton")` per the Pattern 2 sketch falls through to `_torch_ref` for `butterfly_multiply` (with a comment saying Phase 5 will replace). The demonstrator op itself IS a real Triton op, so `_demo_identity_op` works under `set_backend("triton")`.
   - Recommendation: Document this in `_resolve()` source comments. Test parametrization stays at `["torch"]` only in Phase 4's `conftest.py`. Phase 5 extends.
   - **RESOLVED:** `_resolve()` is honest about backend availability — `_has_triton_kernel(op_name)` probes the `_triton/` package; in Phase 4 that package is empty so the probe returns False for every op. `set_backend('triton')` therefore falls back to torch-ref (or cuda if .so present) AND emits a `log.warning("set_backend('triton') requested but no Triton kernel installed; falling back to %s", actual_backend)`. The D-08 INFO heads-up only fires when the actual binding is `triton` AND `_cuda_legacy` is detected — dormant in Phase 4, exercised first in Phase 5.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | Everything | ✓ | 3.x (per repo `requires-python = ">=3.10"`) | — |
| `torch` (>=2.6) | DISP-*, TRI-05 | UNKNOWN — depends on user/CI env | Will be enforced by `pyproject.toml` bump | — (hard floor) |
| `triton` (bundled with torch on CUDA Linux) | TRI-05 demonstrator | UNKNOWN — only present on CUDA Linux installs | bundled | Skip demonstrator GPU tests on non-CUDA machines (`pytest.mark.skipif(not torch.cuda.is_available())`) |
| CUDA driver + GPU | Demonstrator runtime (Triton needs sm_80+ Ampere) | UNKNOWN | — | If absent, tests skip; library still imports via `_torch_ref` fallback |
| `pytest` | Tests | ✓ | per `pyproject.toml` `[project.optional-dependencies]` | — |
| `actions/cache@v4` (GHA) | TEST-05 CI cache | ✓ (GitHub Actions universal) | v4 | v5 (forward compat) |
| `git` | CI checkout | ✓ | universal | — |

**Missing dependencies with no fallback:** None — every dependency either exists or has a documented fallback / skip path.

**Missing dependencies with fallback:**
- Triton/CUDA absent → demonstrator tests skip; `_ops.py` resolves to `_torch_ref` path; library works for CPU consumers.

**CI environment check the planner should add as a verification step:** `python -c "import torch; assert torch.__version__ >= '2.6'; import triton; print(triton.__version__)"` — should succeed on CUDA Linux runners after pyproject bump.

## Validation Architecture

> SKIPPED — `workflow.nyquist_validation` is explicitly set to `false` in `.planning/config.json`. `[VERIFIED: .planning/config.json line 12]`

## Security Domain

> SKIPPED — Phase 4 changes are: a Python dispatch module, a pyproject.toml version bump, a documentation file, a no-op demonstrator op, and a GHA cache config. None of these have user-input, network, persistence, or auth surface. No new attack surface introduced; no ASVS categories apply to test-only / build-only changes in a research library that exposes no network endpoints.
>
> The CONTEXT.md and CLAUDE.md contain no `security_enforcement` directive, so this section is omitted per protocol.

## Sources

### Primary (HIGH confidence)

- [PyTorch 2.6 Release Blog — triton_op Beta announcement](https://pytorch.org/blog/pytorch2-6/) — confirms `torch.library.triton_op` shipped Beta in 2.6, links to library docs
- [torch.library — PyTorch 2.6 documentation](https://docs.pytorch.org/docs/2.6/library.html) — verified `triton_op(name, fn=None, /, *, mutates_args, schema=None)` signature
- [Using User-Defined Triton Kernels with torch.compile — PyTorch tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html) — canonical `triton_op` + `wrap_triton` + `register_autograd` example (note: tutorial omits `register_fake`)
- [pytorch/pytorch — torch/_library/triton.py](https://raw.githubusercontent.com/pytorch/pytorch/main/torch/_library/triton.py) — verified `op.register_fake(fn)` method exists on the resulting `CustomOpDef`
- [Torch-TensorRT: Custom Kernel Plugins example](https://docs.pytorch.org/TensorRT/tutorials/_rendered_examples/dynamo/custom_kernel_plugins.html) — verified `@torch.library.register_fake("ns::name")` decorator form
- [Python warnings docs (3.14)](https://docs.python.org/3/library/warnings.html) — verified `simplefilter("once", DeprecationWarning)` per-(message, category) semantics and `stacklevel=2` attribution
- [GitHub Actions Cache (actions/cache)](https://github.com/actions/cache) — verified `@v4` is fully backward-compatible with v3; `@v5` exists (April 2026) for Node 24 runtime
- [Triton heuristics docs](https://triton-lang.org/main/python-api/generated/triton.heuristics.html) — verified decorator signature and order relative to `@triton.jit`
- [Red Hat — Understanding Triton Cache](https://next.redhat.com/2025/05/16/understanding-triton-cache-optimizing-gpu-kernel-compilation/) — verified `~/.triton/cache` default location and stored artifact types
- Existing repo: `tests/test_multiply.py` lines 19-32 — verified existing test idiom is `unittest.TestCase` with nested loops, not pytest fixtures (so new `conftest.py` is purely additive)
- Existing repo: `torch_structured/butterfly/__init__.py` lines 22-39 — verified the `_load_extension` glob loader pattern to wrap in `_cuda_legacy/`
- Existing repo: `csrc/butterfly.cpp` lines 99-131 — verified the C++ autograd::Function code that stays untouched in Phase 4
- Existing repo: `.planning/quick/260419-p27-…/260419-p27-SUMMARY.md` line 177 — verified the dynamo fake-tensor bug literal error string
- Existing repo: `.planning/config.json` — verified `nyquist_validation: false` to skip that section

### Secondary (MEDIUM confidence)

- [PyTorch dev-discuss — Custom Ops Under torch.compile](https://dev-discuss.pytorch.org/t/custom-ops-under-torch-compile-autograd-function-vs-torch-library-custom-op/3338) — official recommendation to use `triton_op` over `autograd.Function`
- [Triton issue #9368 — non-deterministic behavior with TRITON_CACHE_DIR set](https://github.com/triton-lang/triton/issues/9368) — context on cache behavior in CI
- [Triton issue #4265 — environment variables for cache locations](https://github.com/triton-lang/triton/issues/4265) — `TRITON_CACHE_DIR` semantics
- [PyTorch forum — How might autograd differentiate complex functions](https://discuss.pytorch.org/t/how-might-autograd-differentiate-complex-functions-that-are-not-complex-differentiable/105602) — Wirtinger calculus context for complex backward
- [PyTorch derivatives.yaml (in pytorch/pytorch repo)](https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml) — verified `view_as_real` / `view_as_complex` have explicit autograd backward rules
- [Lei Mao — PyTorch Triton Kernel Transparent Tracing and Compilation](https://leimao.github.io/blog/PyTorch-Triton-Kernel-Transparent-Tracing-and-Compilation/) — alternate `triton_op` worked example (no `register_fake` shown)

### Internal (research from prior phases)

- `.planning/research/SUMMARY.md` — milestone synthesis (Architecture Approach, Critical Pitfalls)
- `.planning/research/STACK.md` — Triton/PyTorch version matrix, `triton_op` API
- `.planning/research/ARCHITECTURE.md` — `_ops.py` + `_triton/<op>/` + `_torch_ref/` layout
- `.planning/research/PITFALLS.md` §1 (complex), §3 (`triton_op` is the only viable wrapper), §11 (`view_as_real` strides)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — versions and APIs verified against PyTorch 2.6 source code (`torch/_library/triton.py`) and PyPI
- Architecture: HIGH — directly inherits the milestone-research `_ops.py` + `_torch_ref/` design; Phase 4-specific details (D-04..D-16) verified against existing repo structure
- Pitfalls: HIGH — top 3 pitfalls are the same ones flagged in milestone PITFALLS.md plus the call-site contract from D-05
- Complex64 ABI: HIGH — `view_as_real`/`view_as_complex` autograd is documented in PyTorch's `derivatives.yaml`; stride contract is in the public docs
- Deprecation warning incantation: HIGH — verified against Python 3.14 stdlib docs
- CI cache template: HIGH — `actions/cache@v4` YAML is documented; `~/.triton/cache` default is documented
- Demonstrator op skeleton: HIGH — composed from official PyTorch tutorial + verified `register_fake` form from source

**Research date:** 2026-05-27
**Valid until:** 2026-06-26 (30 days; APIs are stable post-2.6, but PyTorch minor releases happen ~monthly — re-check `torch.library.triton_op` docs if Phase 5 starts after this window)

[1]: https://pytorch.org/blog/pytorch2-6/
