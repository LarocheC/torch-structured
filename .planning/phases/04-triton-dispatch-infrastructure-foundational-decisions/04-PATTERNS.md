# Phase 4: Triton Dispatch Infrastructure & Foundational Decisions - Pattern Map

**Mapped:** 2026-05-27
**Files analyzed:** 14 (10 created + 4 modified)
**Analogs found:** 12 / 14 (the 2 doc files + the `.github/workflows/test.yml` have no in-repo analog; standard upstream templates apply)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `torch_structured/_torch_ref/__init__.py` (new) | package init | n/a | `torch_structured/recurrent/__init__.py` | exact (one-line re-export idiom) |
| `torch_structured/_torch_ref/butterfly.py` (new) | reference impl | request-response (tensor in/out) | `torch_structured/butterfly/multiply.py:28-49` (the function being relocated) | exact (verbatim move) |
| `torch_structured/_ops.py` (new) | dispatch module | request-response + lazy import | `torch_structured/butterfly/__init__.py:22-39` (loader + bind) + `torch_structured/structured/hadamard.py:1-8,61` (try-import + bind) | role-match (no exact 1:1 in repo; assembled from two analogs) |
| `torch_structured/_cuda_legacy/__init__.py` (new) | thin wrapper | request-response | `torch_structured/structured/hadamard.py:1-8` (try-import existing `torch.ops` / `_xxx_cuda` artifact) | exact |
| `torch_structured/_cuda_legacy/butterfly.py` (new) | op wrapper | request-response | `torch_structured/butterfly/multiply.py:21-25` (`@torch.jit.script` wrapping `torch.ops.torch_structured.butterfly_multiply`) | exact |
| `tests/conftest.py` (new) | pytest fixture | test infra | none in repo (no existing conftest.py); the closest *style* anchor is `tests/test_lru.py:6-8` (module-import side-effects pattern) | no-analog (purely additive) |
| `tests/test_dispatch.py` (new) | test module | test infra | `tests/test_lru.py:1-12` + `tests/test_multiply.py:1-12` (unittest+import style) | role-match (Phase 4 chooses pytest-style for the new file per RESEARCH §"Standard Stack Alternatives") |
| `.github/workflows/test.yml` (new if absent) | CI config | event-driven | none in repo (no `.github/` directory exists) | no-analog (standard `actions/cache@v4` template from RESEARCH Pattern 5) |
| `04-COMPLEX-LAYOUT.md` (new) | doc | n/a | `.planning/research/PITFALLS.md` (companion-doc style) | role-match |
| `04-DEPRECATION-PLAN.md` (new) | doc | n/a | `.planning/research/PITFALLS.md` (companion-doc style) | role-match |
| `pyproject.toml` (modified) | build config | n/a | `pyproject.toml:24-25` (current `torch>=2.0` line — one-character edit) | exact |
| `torch_structured/butterfly/multiply.py` (modified) | re-export shim | n/a | `torch_structured/butterfly/butterfly.py:8-9` (existing `# noqa: F401 (re-exported)` shim line) | exact |
| `torch_structured/butterfly/butterfly.py` (modified, conditional) | consumer import | n/a | `torch_structured/butterfly/butterfly.py:9` (no edit needed if shim covers it — verify only) | exact |
| `torch_structured/__init__.py` (modified) | public API surface | n/a | `torch_structured/__init__.py:26-34` (existing `from .butterfly import ...` re-export block) | exact |

---

## Pattern Assignments

### `torch_structured/_torch_ref/__init__.py` (package init)

**Analog:** `torch_structured/recurrent/__init__.py` (full file, 4 lines)

**Pattern to copy** (`torch_structured/recurrent/__init__.py:1-4`):
```python
"""Recurrent layers built on torch_structured structured-matrix primitives."""
from .lru import LRU  # noqa: F401

__all__ = ["LRU"]
```

**Apply as:**
```python
"""Pure-PyTorch reference implementations used by the dispatch fallback path."""
from .butterfly import butterfly_multiply_torch  # noqa: F401

__all__ = ["butterfly_multiply_torch"]
```

**Notes:** Trivial init — matches the minimal subpackage idiom already established by `recurrent/`. Phase 5/6 will extend `__all__` with `diag_mult` and `hadamard_transform` when those reference impls land.

---

### `torch_structured/_torch_ref/butterfly.py` (reference impl, request-response)

**Analog:** `torch_structured/butterfly/multiply.py:28-49` — the function being moved verbatim.

**Imports pattern** (mirror current file, drop the unused jit imports):
```python
import torch
from torch.nn import functional as F
```
*(Note: `import math` in the source file is not used by `butterfly_multiply_torch` itself — drop it. `Tuple, Optional` typing imports are also for the `_fw`/`_bw` jit wrappers, not this function.)*

**Core function pattern** (`torch_structured/butterfly/multiply.py:28-49`, copy verbatim):
```python
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

**Notes:** Zero logic change. The function is being relocated, not rewritten (D-09). Existing `assert` precondition style matches CLAUDE.md "Use `assert` statements for preconditions (not exceptions)" convention.

---

### `torch_structured/_ops.py` (dispatch module, request-response + lazy import)

**Primary analog (try-import + module-level bind):** `torch_structured/structured/hadamard.py:1-8,61`

**Pattern to copy** (`torch_structured/structured/hadamard.py:1-8`):
```python
import numpy as np
import torch

use_hadamard_transform_cuda = True
try:
    from torch_structured import _hadamard_cuda as hadamard_cuda
except ImportError:
    use_hadamard_transform_cuda = False
```

**Final binding line** (`torch_structured/structured/hadamard.py:61`):
```python
hadamard_transform = hadamard_transform_cuda if use_hadamard_transform_cuda else hadamard_transform_torch
```

**Secondary analog (extension presence probe):** `torch_structured/butterfly/__init__.py:22-39` shows the `glob` + `torch.ops.load_library` pattern. For `_ops.py`'s `_has_cuda_legacy()` probe, the simpler `hasattr(torch.ops.torch_structured, "butterfly_multiply")` form is correct (the `.so` is already loaded by `butterfly/__init__.py` at top-level import — `_ops.py` only needs to detect whether the registration succeeded):

```python
def _has_cuda_legacy() -> bool:
    return hasattr(torch.ops.torch_structured, "butterfly_multiply")
```

**Try-import fallback pattern for Triton** (mirror `hadamard.py:5-8` ImportError handling, plus the CUDA-available gate):
```python
def _has_triton() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()
```

**Module-level callable rebinding pattern (new — no direct in-repo analog, assembled from RESEARCH Pattern 2):**
```python
butterfly_multiply = None      # type: ignore[assignment]
# ... rebound by _resolve()

def _resolve(name: str) -> str:
    global butterfly_multiply, _BACKEND
    # ... pick chosen backend
    if chosen == "torch":
        from torch_structured._torch_ref.butterfly import butterfly_multiply_torch
        butterfly_multiply = butterfly_multiply_torch
    # ...
```

**Notes:** This is the most novel file in Phase 4. The two analogs cover (a) the try-import + bind idiom and (b) the existing extension detection. The `global`-mutating `_resolve` + `set_backend` pattern is *not* present in the current repo — it's lifted verbatim from RESEARCH.md Pattern 2 (lines 302-391 of RESEARCH.md). The planner MUST also implement the demonstrator op (`_demo_identity_op`) in this same file per D-13; the demonstrator skeleton lives at RESEARCH.md lines 611-657.

**CRITICAL idiom to preserve:** the demonstrator op uses `op.register_fake(fn)` (NOT just `register_autograd`). RESEARCH.md Pitfall 1 (lines 542-551) calls this out as the 260419-p27 bug fix; omitting it silently breaks `torch.compile` traceability.

---

### `torch_structured/_cuda_legacy/__init__.py` (thin wrapper)

**Analog:** `torch_structured/structured/hadamard.py:1-8` (try-import a compiled artifact).

**Pattern to copy / adapt:**
```python
"""Wrapper around the already-loaded torch.ops.torch_structured.* C++ ops.

This module exists so the _ops.py resolver can do
`from torch_structured._cuda_legacy import butterfly_multiply` uniformly,
regardless of whether butterfly's compiled .so loaded successfully.
"""
import torch

# The .so was loaded eagerly by torch_structured.butterfly's __init__.py at
# package import time (see torch_structured/butterfly/__init__.py:22-39).
# If it failed to register, this import path will surface AttributeError when
# the resolver probes _has_cuda_legacy().
from .butterfly import butterfly_multiply  # noqa: F401

__all__ = ["butterfly_multiply"]
```

**Notes:** Per RESEARCH §"Open Questions 3" (lines 852-854), Phase 4 uses the **minimal wrapper** strategy — do NOT duplicate `_load_extension` here; consume the side-effects of `butterfly/__init__.py` import. Phase 10 may absorb the loader into `_cuda_legacy/` when `butterfly/__init__.py` collapses.

---

### `torch_structured/_cuda_legacy/butterfly.py` (op wrapper, request-response)

**Analog:** `torch_structured/butterfly/multiply.py:21-25`

**Pattern to copy** (`torch_structured/butterfly/multiply.py:21-25`):
```python
@torch.jit.script
def butterfly_multiply(twiddle: torch.Tensor, input: torch.Tensor, increasing_stride: bool,
                       output_size: Optional[int] = None) -> torch.Tensor:
    return torch.ops.torch_structured.butterfly_multiply(twiddle, input, increasing_stride,
                                                          output_size)
```

**Apply as** (`_cuda_legacy/butterfly.py`):
```python
from typing import Optional
import torch


def butterfly_multiply(twiddle: torch.Tensor, input: torch.Tensor, increasing_stride: bool,
                       output_size: Optional[int] = None) -> torch.Tensor:
    """Pass-through to the compiled C++ op (already loaded by butterfly/__init__.py)."""
    return torch.ops.torch_structured.butterfly_multiply(twiddle, input, increasing_stride,
                                                          output_size)
```

**Notes:** Drop the `@torch.jit.script` decorator — the dispatched callable is invoked from `_ops.butterfly_multiply` which may itself be inside a `torch.compile` graph (TorchScript and Inductor compose poorly; RESEARCH.md "State of the Art" table notes TorchScript is deprecated in PyTorch 2.10). Keep the `Optional[int]` typing for API compatibility but the wrapper is a plain Python function. Phase 10 may absorb this into the deprecation-warning module.

---

### `tests/conftest.py` (pytest fixture, test infra)

**Analog:** No existing `conftest.py` in the repo (`find . -name "conftest.py"` returned empty). The closest style anchor for "test imports `torch_structured` to trigger extension load" is `tests/test_lru.py:6-8`:

```python
import torch_structured  # noqa: F401 — triggers extension load
from torch_structured import LRU
from torch_structured.recurrent import lru as _lru_mod
```

**Pattern to apply** (copied from RESEARCH.md lines 791-807, no in-repo analog):
```python
# tests/conftest.py
# Phase 4: backend fixture parametrized over ["torch"] only.
# Phase 5+ will extend to ["torch", "triton", "cuda"] as kernels land.
import pytest

import torch_structured


@pytest.fixture(params=["torch"])
def backend(request):
    """Switch backend for the duration of a test, restore after."""
    original = torch_structured._ops._BACKEND
    chosen = torch_structured._ops.set_backend(request.param)
    yield chosen
    torch_structured._ops.set_backend(original)
```

**Notes:** Purely additive — no existing test uses pytest fixtures (existing files use `unittest.TestCase` with hand-rolled nested loops; see RESEARCH.md Pitfall §"existing test idiom" line 11). The `noqa: F401 — triggers extension load` comment style from `test_lru.py:6` is the established convention for explaining import side-effects.

---

### `tests/test_dispatch.py` (test module, test infra)

**Analog (import style):** `tests/test_lru.py:1-12` and `tests/test_multiply.py:1-12`.

**Imports pattern (mixed: existing convention + Phase 4's pytest-style new-file allowance):**

Existing convention (`tests/test_multiply.py:1-11`):
```python
import math
import unittest

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import torch_structured
```

Phase 4 new-file pattern (per RESEARCH.md "Standard Stack Alternatives" table, line 135, and the test skeleton at RESEARCH.md lines 661-714):
```python
import pytest
import torch

from torch_structured._ops import _demo_identity_op

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="demo op is GPU-only")
```

**Core test pattern** (copy verbatim from RESEARCH.md lines 673-713 — five tests):
- `test_demo_identity_eager_fp32` — basic call works in eager mode
- `test_demo_identity_eager_complex64` — exercises `view_as_real` wrapper path
- `test_demo_identity_gradcheck` — `torch.autograd.gradcheck` passes
- `test_demo_identity_compile_no_graph_break` — `@torch.compile(fullgraph=True)` traces cleanly
- `test_demo_identity_compile_fake_tensor_trace` — explicit `FakeTensorMode` reproducer for the 260419-p27 bug

**Notes:** This is a NEW file, so it may use pytest-style without disrupting the existing `unittest.TestCase` convention. RESEARCH.md "Pitfall 1" (line 542) makes the `register_fake` test (the last two cases) load-bearing — they are the literal acceptance gate.

---

### `.github/workflows/test.yml` (CI config, event-driven)

**Analog:** None — `find .github -type f` returned no results; the repo has no CI configured yet.

**Pattern to apply** (lifted verbatim from RESEARCH.md Pattern 5, lines 466-487, with the `actions/cache@v4` version honoring CONTEXT.md D-16):
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

**Notes:** Pitfall 6 (RESEARCH.md line 599) warns against keying on `github.sha` (would yield 0% hit rate). Use `hashFiles('torch_structured/_triton/**/*.py')`. The `_triton/` directory will be empty in Phase 4 but the cache infra must exist now per TEST-05.

---

### `pyproject.toml` (build config, modified)

**Analog:** Current file (`pyproject.toml:24-25`) — the line being edited.

**Current state** (`pyproject.toml:24-30`):
```toml
dependencies = [
    "torch>=2.0",
    "numpy",
    "scipy",
    "einops",
    "opt_einsum",
]
```

**Edit pattern:** Bump `"torch>=2.0"` → `"torch>=2.6"` on line 25. Also bump the `[build-system].requires` line 2 if it pins torch (`pyproject.toml:2` currently reads `requires = ["setuptools>=64", "torch>=2.0", "ninja", "wheel"]` — bump that too for consistency).

**Notes:** One-line edit. CLAUDE.md ("Build system: Must use pyproject.toml as the single source of truth") confirms this is the right and only place. RESEARCH.md "Assumption A7" (line 835) notes that existing editable installs on torch>=2.6 envs won't need rebuild; envs with torch==2.5 will hard-fail at install resolution (which is the desired floor behavior).

---

### `torch_structured/butterfly/multiply.py` (re-export shim, modified)

**Analog:** `torch_structured/butterfly/butterfly.py:8-9` — existing in-repo example of a `# noqa: F401 (re-exported)` pass-through.

**Pattern to copy** (`torch_structured/butterfly/butterfly.py:8-9`):
```python
from .multiply import butterfly_multiply
from .multiply import butterfly_multiply_torch  # noqa: F401 (re-exported)
```

**Apply as** (add to top of `torch_structured/butterfly/multiply.py`, after the existing imports but BEFORE the `@torch.jit.script` decorators):
```python
# Phase 4: butterfly_multiply_torch now lives in _torch_ref/. Re-export here
# so existing test imports (torch_structured.butterfly.multiply.butterfly_multiply_torch)
# keep working unchanged.
from torch_structured._torch_ref.butterfly import butterfly_multiply_torch  # noqa: F401
```

Then **delete** the original `def butterfly_multiply_torch(...)` block (`torch_structured/butterfly/multiply.py:28-49`). The remaining `butterfly_multiply_fw` / `butterfly_multiply_bw` / `butterfly_multiply` `@torch.jit.script` wrappers stay UNCHANGED — Phase 4 does not touch the existing CUDA path.

**Notes:** RESEARCH.md "Runtime State Inventory" (lines 528-538) verifies this shim covers both existing usage patterns: `torch_structured.butterfly.multiply.butterfly_multiply_torch(...)` (qualified path) and `from torch_structured.butterfly.multiply import butterfly_multiply_torch` (used by `tests/test_multiply_base4.py:10`).

---

### `torch_structured/butterfly/butterfly.py` (consumer import, verify only)

**Analog:** Self-reference — `torch_structured/butterfly/butterfly.py:9` is already structured as a re-export line.

**Current state** (`torch_structured/butterfly/butterfly.py:8-9`):
```python
from .multiply import butterfly_multiply
from .multiply import butterfly_multiply_torch  # noqa: F401 (re-exported)
```

**Edit pattern:** No edit required. After Phase 4's shim lands in `multiply.py`, `from .multiply import butterfly_multiply_torch` continues to resolve correctly (now through the shim re-export). Verification step: `python -c "from torch_structured.butterfly.butterfly import butterfly_multiply_torch; print(butterfly_multiply_torch)"` should succeed and print the function object from `_torch_ref.butterfly`.

**Notes:** Listed in CONTEXT.md "Files to modify" but RESEARCH.md "Runtime State Inventory" + the shim design makes this a verify-only target. Planner should add the verification as a Plan 1 task acceptance check.

---

### `torch_structured/__init__.py` (public API surface, modified)

**Analog:** `torch_structured/__init__.py:26-34` — the existing re-export block.

**Pattern to copy** (`torch_structured/__init__.py:26-44`):
```python
from .butterfly import (
    Butterfly,
    ButterflyBmm,
    ButterflyBase4,
    ButterflyUnitary,
    butterfly_multiply,
)
from .factory import make_linear
from .recurrent import LRU

__all__ = [
    'Butterfly',
    'ButterflyBmm',
    'ButterflyBase4',
    'ButterflyUnitary',
    'butterfly_multiply',
    'LRU',
    'make_linear',
    '__version__',
]
```

**Apply as** (add after the existing imports block, before `__all__`):
```python
from ._ops import set_backend  # noqa: F401
```

And extend `__all__` with `'set_backend'`.

**Notes:** RESEARCH.md "Open Question 2" (lines 847-851) verifies no circular import: `torch_structured/__init__.py` → `.butterfly` → `.multiply` → (after Phase 4) `._torch_ref.butterfly`, none of which import `_ops`. Adding `from ._ops import set_backend` is safe. The import of `_ops` will trigger `_resolve()` at import time, which logs the chosen backend per DISP-05. Planner verification: `python -c "import torch_structured; torch_structured.set_backend('torch')"` succeeds.

---

## Shared Patterns

### Try-import + boolean flag (compiled-extension presence detection)

**Source:** `torch_structured/structured/hadamard.py:1-8`
**Apply to:** `torch_structured/_ops.py` (`_has_triton`, `_has_cuda_legacy`), `torch_structured/_cuda_legacy/__init__.py` (the `from .butterfly import butterfly_multiply` will surface ImportError if the underlying `.so` failed to register)

```python
use_hadamard_transform_cuda = True
try:
    from torch_structured import _hadamard_cuda as hadamard_cuda
except ImportError:
    use_hadamard_transform_cuda = False
```

**Variation for `_ops.py` (function-form, lazy, returns bool):**
```python
def _has_triton() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()
```

---

### `# noqa: F401` re-export comment style

**Source:** `torch_structured/butterfly/butterfly.py:9` (`# noqa: F401 (re-exported)`) and `torch_structured/recurrent/__init__.py:2` (`# noqa: F401`)
**Apply to:** Every package init in Phase 4 (`_torch_ref/__init__.py`, `_cuda_legacy/__init__.py`) and the shim in `butterfly/multiply.py`.

**Convention:** Use `# noqa: F401` to suppress unused-import warnings for re-exports. The repo uses both bare `# noqa: F401` (recurrent/) and annotated `# noqa: F401 (re-exported)` (butterfly/) forms — either is acceptable; prefer the annotated form for new shims so the intent is obvious.

---

### `assert` for preconditions (no exceptions, no validators)

**Source:** `torch_structured/butterfly/multiply.py:33-36` (`assert twiddle.shape == ...; assert output_size <= n`), and project convention (CLAUDE.md §"Error Handling")
**Apply to:** Any new wrapper that validates tensor shape/dtype in `_ops.py` (e.g., the `view_as_real` contiguity check from RESEARCH.md Pitfall 3 line 579: `assert x.is_contiguous(), "complex input must be contiguous before view_as_real"`).

```python
assert twiddle.shape == (nstacks, nblocks, log_n, n // 2, 2, 2)
```

**Do NOT** introduce `if x: raise ValueError(...)` patterns — they conflict with the established codebase convention.

---

### Module-import-as-side-effect comment

**Source:** `tests/test_lru.py:6` (`import torch_structured  # noqa: F401 — triggers extension load`)
**Apply to:** `tests/conftest.py` and `tests/test_dispatch.py` if they need to force the `_ops.py` import side-effect before any backend probe.

```python
import torch_structured  # noqa: F401 — triggers extension load + _ops.py resolver
```

---

## No Analog Found

Files with no close match in the codebase (planner should use RESEARCH.md patterns instead):

| File | Role | Data Flow | Reason / Source |
|------|------|-----------|-----------------|
| `tests/conftest.py` | pytest fixture | test infra | No existing `conftest.py` in repo (`find` confirmed). Use RESEARCH.md lines 791-807 verbatim. |
| `.github/workflows/test.yml` | CI config | event-driven | No `.github/` directory exists. Use RESEARCH.md Pattern 5 (lines 466-487) verbatim with CONTEXT.md D-16's `actions/cache@v4` pin. |
| `04-COMPLEX-LAYOUT.md` | doc | n/a | New companion doc class. Follow `.planning/research/PITFALLS.md` style: short prose + concrete code snippet showing `view_as_real → kernel → view_as_complex` per CONTEXT.md "Specific Ideas". |
| `04-DEPRECATION-PLAN.md` | doc | n/a | New companion doc class. Use the exact `warnings.simplefilter("once", DeprecationWarning)` + `stacklevel=2` template from RESEARCH.md Pattern 4 (lines 426-443). |

For the **demonstrator op skeleton** inside `_ops.py` (`_demo_identity_op` + `_demo_identity_kernel` + `_setup_context` + `_backward` + `register_fake`), there is also **no in-repo analog** — Phase 4 is the first place the repo uses `torch.library.triton_op`. Use RESEARCH.md lines 611-657 (Code Examples §"Complete demonstrator op") verbatim. RESEARCH.md "Pitfall 1" makes `register_fake` non-negotiable.

---

## Metadata

**Analog search scope:** `torch_structured/`, `tests/`, `.github/` (empty), `pyproject.toml`
**Files scanned:** 14 source files + 2 config files + ~15 test files (sampled `test_multiply.py`, `test_lru.py`, `test_multiply_base4.py`)
**Key analogs identified by full path:**
- `torch_structured/butterfly/multiply.py:28-49` — `butterfly_multiply_torch` (relocates verbatim)
- `torch_structured/butterfly/multiply.py:21-25` — `@torch.jit.script` op wrapper (de-jitted for `_cuda_legacy/butterfly.py`)
- `torch_structured/butterfly/__init__.py:22-39` — `_load_extension` loader + `torch.ops.load_library` (Phase 4 reuses by side-effect, not by duplication)
- `torch_structured/structured/hadamard.py:1-8,61` — try-import + module-level callable bind pattern (the closest in-repo analog for `_ops.py`'s resolver-bind loop)
- `torch_structured/structured/krylov.py:21-24` — try-import with `(ImportError, RuntimeError)` exception spec (variant `_ops.py` may use for cuda detection robustness)
- `torch_structured/butterfly/butterfly.py:8-9` — `# noqa: F401 (re-exported)` shim style
- `torch_structured/recurrent/__init__.py:1-4` — minimal subpackage `__init__.py` template
- `torch_structured/__init__.py:26-44` — top-level re-export + `__all__` extension template
- `tests/test_multiply.py:1-12`, `tests/test_lru.py:1-12` — existing `unittest.TestCase` import style (Phase 4 deviates per RESEARCH §"Standard Stack Alternatives" — new file `test_dispatch.py` uses pytest style)
- `pyproject.toml:2,25` — `torch>=2.0` pin (one-line bump to `torch>=2.6`)

**Pattern extraction date:** 2026-05-27
