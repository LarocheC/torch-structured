# Phase 9: Integration Hardening & Correctness Gates - Pattern Map

**Mapped:** 2026-05-28
**Files analyzed:** 18 (10 modified + 8 new)
**Analogs found:** 17 / 18

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `torch_structured/butterfly/multiply.py` (MOD, 09-01, RESEARCH §0 LANDMINE) | back-compat shim | delegator / request-response | `torch_structured/structured/hadamard.py` | exact (D-05 attribute-access delegator pattern) |
| `tests/conftest.py` (MOD, 09-01) | test fixture | parametrized setup/teardown | `tests/conftest.py` (current Phase 6 D-39 implementation) | exact (extend in place) |
| `torch_structured/_ops.py` (MOD x3 across 09-01/09-02/09-03) | dispatch / config | resolver + per-op probe + flag setter | `torch_structured/_ops.py` (existing `_has_cuda_legacy_diag_mult` / `_has_cuda_legacy_hadamard` / `set_backend`) | exact (clone-and-extend siblings already in file) |
| `torch_structured/__init__.py` (MOD, 09-02) | public API | symbol re-export | `torch_structured/__init__.py:35` (existing `set_backend` export) | exact |
| `torch_structured/_triton/butterfly/op.py` (MOD, 09-02) | autograd wrapper | request-response gate | `torch_structured/_triton/butterfly/op.py:1377-1383` (existing small-N oracle fallback in `_backward`) | exact (clone the small-N fallback pattern) |
| `torch_structured/_routing.json` (NEW, 09-03) | static config | data lookup | `.planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json` | role-match (JSON-on-disk; different schema) |
| `.planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json` (MOD, 09-03) | perf data | schema extension | self (16 existing rows) | exact (extend each row with `reference_cuda_p50`) |
| `tests/_baseline_butterfly.py` (MOD, 09-03) | perf harness | timed bench | `tests/_baseline_butterfly.py` (current forward harness) | exact (add CUDA branch) |
| `tests/_baseline_butterfly_backward.py` (MOD, 09-03) | perf harness | timed bench | `tests/_baseline_butterfly_backward.py` (current backward harness) | exact (add CUDA branch) |
| `tests/test_butterfly_triton.py` (MOD, 09-03, only `test_butterfly_backward_no_cpp_symbol`) | test | dispatch-binding + monkey-patch | `tests/test_butterfly_triton.py:705-774` (current SC#4 test) | exact (update in place per D-61a) |
| `tests/test_phase9_integration.py` (NEW, 09-01) | test | backend-agreement + smoke | `tests/test_butterfly_triton.py:60-127` (parametrized `backend` fixture tests) | exact (use widened `backend` fixture) |
| `tests/test_torch_compile_triton.py` (NEW, 09-02) | test | torch.compile trace + FakeTensorMode | `tests/test_dispatch.py` (Phase 4 demonstrator — for FakeTensorMode contract) | role-match (current test_dispatch.py covers dispatch-contract, not compile-end-to-end; copy the import/imports pattern + add `fullgraph=True` calls) |
| `tests/test_distributed_triton.py` (NEW, 09-02) | test | DDP/FSDP/checkpoint | `tests/test_butterfly_triton.py` (parametrized backend + cuda guard) | role-match (existing file is the closest backend-parametrized integration test; FSDP/DDP idioms are new in this repo) |
| `tests/test_deterministic_mode.py` (NEW, 09-02) | test | flag toggle + bit-identity | `tests/test_dispatch.py:17-30` (set_backend round-trip test) | exact (same shape: toggle, assert state, restore) |
| `tests/test_perf_grid.py` (NEW, 09-03) | test | JSON read + ratio gate | none in repo (new shape) | no analog — JSON-driven gate is novel; pattern documented inline |
| `scripts/regenerate_routing_table.py` (NEW, 09-03; scripts/ dir does not exist) | utility script | one-shot JSON transform | `tests/_baseline_butterfly.py` (CLI script shape: `def main() -> int` + `sys.exit(main())`) | role-match (different work, same CLI shell) |
| `README.md` (MOD, 09-03) | docs | narrative | self | exact (extend existing file) |
| `CHANGELOG.md` (NEW, 09-03; does not exist) | docs | release notes | none in repo | no analog — Keep a Changelog v1.1 format per RESEARCH §11 |
| `.github/workflows/test.yml` or new `test-gpu.yml` (MOD/NEW, 09-03) | CI config | matrix job | `.github/workflows/test.yml` (current single-job CPU workflow) | role-match (extend the existing single-job shape to multi-job) |

---

## Pattern Assignments

### `torch_structured/butterfly/multiply.py` (back-compat shim — 09-01 LANDMINE fix)

**Analog:** `torch_structured/structured/hadamard.py` (lines 1-35) — already implements the exact `_ops` attribute-access delegator pattern that RESEARCH §0 prescribes for this file.

**Current state (the LANDMINE — lines 13-30):** All three top-level callables are `@torch.jit.script`-decorated wrappers that call `torch.ops.torch_structured.*` (the C++ ops) DIRECTLY, bypassing `_ops`. Under `BACKEND=triton`, the legacy nn.Modules (`Butterfly.forward`) STILL route through the C++ path because they import `butterfly_multiply` from this module, not from `_ops`.

**Target pattern — copy from `torch_structured/structured/hadamard.py:20-34` VERBATIM (adjust names):**

```python
"""Back-compat shim — Phase 9 (D-05 LANDMINE fix per RESEARCH §0).

The legacy ``Butterfly.forward`` / ``ButterflyBmm.forward`` call sites at
``torch_structured/butterfly/butterfly.py:124,128,239,243`` import
``butterfly_multiply`` from this module. Until Phase 9, that binding resolved
to a ``@torch.jit.script``-decorated wrapper that called
``torch.ops.torch_structured.butterfly_multiply`` (the C++ op) DIRECTLY,
bypassing ``_ops`` — so under ``BACKEND=triton`` the nn.Module surface still
went to the C++ path.

Phase 9 09-01 replaces the wrappers with D-05 attribute-access delegators
(mirrors ``torch_structured/structured/hadamard.py:25-34``) so the nn.Module
surface honors ``set_backend()``. The historical import surface
``from torch_structured.butterfly.multiply import butterfly_multiply``
keeps working; the binding is now re-read on every call.
"""
import torch_structured  # noqa: F401 — needed so the shims below can attribute-access _ops

# Phase 4: butterfly_multiply_torch lives in _torch_ref/ — re-export for back-compat.
from torch_structured._torch_ref.butterfly import butterfly_multiply_torch  # noqa: F401


def butterfly_multiply(*args, **kwargs):
    """Back-compat shim — delegates to ``torch_structured._ops.butterfly_multiply``.

    Preserves the historical
    ``torch_structured.butterfly.multiply.butterfly_multiply`` import surface
    while honoring the D-05 attribute-access contract: the binding is re-read
    on every call so ``set_backend()`` rebindings take effect transparently.
    """
    return torch_structured._ops.butterfly_multiply(*args, **kwargs)


def butterfly_multiply_fw(*args, **kwargs):
    """Back-compat shim — re-checked for the same LANDMINE pattern.

    Routes through ``torch.ops.torch_structured.butterfly_multiply_fw`` (the
    forward-only C++ op). Phase 9 keeps this routed to the C++ op directly
    (no Triton equivalent for the `_fw`-only entry point exists — the Triton
    path goes through ``butterfly_multiply`` which is autograd-aware). The
    @torch.jit.script decorator is REMOVED so monkey-patching from
    tests/test_butterfly_triton.py:756-757 (Phase 8 SC#4) keeps working.
    """
    return torch.ops.torch_structured.butterfly_multiply_fw(*args, **kwargs)


def butterfly_multiply_bw(*args, **kwargs):
    """Same as butterfly_multiply_fw — C++ backward op direct call.

    Phase 9 leaves this on the C++ op (no Triton equivalent — Triton backward
    is exposed only through ``butterfly_multiply`` + ``register_autograd``).
    """
    return torch.ops.torch_structured.butterfly_multiply_bw(*args, **kwargs)
```

**Note on `@torch.jit.script` removal:** Phase 8's `test_butterfly_backward_no_cpp_symbol` (`tests/test_butterfly_triton.py:756-757`) monkey-patches `_legacy_mod_for_sc4.butterfly_multiply_fw` and `_bw`. The `@torch.jit.script` decorator makes a scripted callable that resists attribute reassignment on the module — RESEARCH §0 Pitfall 6 notes this constraint. Removing the decorator preserves monkey-patchability; the C++ op call is still the underlying dispatch.

**Import keyword:** `import torch` must be added (only `from torch.nn import functional as F` and `from typing import Tuple, Optional` are imported today; the new function bodies call `torch.ops.torch_structured.*`).

---

### `torch_structured/_ops.py` (modified across all three waves)

**Analog:** `torch_structured/_ops.py` itself — clone-and-extend the existing per-op probes and `set_backend`.

**09-01 addition — `_has_cuda_legacy_for_op(op_name: str) -> bool`**

Copy structure from `_has_cuda_legacy_diag_mult` (lines 82-94) and `_has_cuda_legacy_hadamard` (lines 97-107). Existing pattern (lines 82-94 verbatim):

```python
def _has_cuda_legacy_diag_mult() -> bool:
    """Per-op honest probe (CHECKER B3) for the legacy ``_diag_mult_cuda`` extension.

    Symmetric to ``_has_cuda_legacy()`` but checks the pybind11 ``_diag_mult_cuda``
    extension (D-22). Returns the ``HAS_CUDA_LEGACY_DIAG_MULT`` sentinel from
    ``_cuda_legacy/diag_mult.py`` — True iff the ``.so`` was built and the
    top-of-module try-import succeeded. Never raises; returns a clean bool.
    """
    try:
        from torch_structured._cuda_legacy.diag_mult import HAS_CUDA_LEGACY_DIAG_MULT
        return HAS_CUDA_LEGACY_DIAG_MULT
    except ImportError:
        return False
```

**New helper — multiplex over op_name (sibling to `_has_triton_kernel(op_name)` at lines 119-139):**

```python
def _has_cuda_legacy_for_op(op_name: str) -> bool:
    """Per-op honest probe (CHECKER B3) for the legacy CUDA backend (D-62a).

    Multiplexes the per-op `_has_cuda_legacy_*` probes — symmetric to
    ``_has_triton_kernel(op_name)``. Returns True iff the specific CUDA legacy
    .so is importable for ``op_name``.

      - "butterfly_multiply" -> _has_cuda_legacy()      (loads on import of
                                 torch_structured.butterfly via _butterfly.so)
      - "diag_mult"          -> _has_cuda_legacy_diag_mult()
      - "hadamard_transform" -> _has_cuda_legacy_hadamard()

    Never raises; returns a clean bool. Used by ``tests/conftest.py`` to
    per-test skip the 'cuda' axis when the op's .so isn't built (D-62).
    """
    if op_name == "butterfly_multiply":
        return _has_cuda_legacy()
    if op_name == "diag_mult":
        return _has_cuda_legacy_diag_mult()
    if op_name == "hadamard_transform":
        return _has_cuda_legacy_hadamard()
    return False
```

**09-02 addition — `_DETERMINISTIC` flag + `set_deterministic` setter (D-63):**

Mirror `set_backend(name) -> str` shape at lines 303-314, but with a `bool` flag instead of a string-state machine:

```python
# ── Deterministic-mode flag (Phase 9 D-63) ───────────────────────────────
_DETERMINISTIC: bool = False


def _is_deterministic_mode_active() -> bool:
    """True iff EITHER torch_structured._DETERMINISTIC OR
    ``torch.are_deterministic_algorithms_enabled()`` (D-63b — additive OR).
    Consumed by ``_triton/butterfly/op.py`` ``_backward`` wrapper gate.
    """
    return _DETERMINISTIC or torch.are_deterministic_algorithms_enabled()


def set_deterministic(value: bool) -> bool:
    """Public API: toggle deterministic-mode for torch_structured Triton kernels.

    Returns the **previous** value (save/restore pattern, mirrors
    ``set_backend()`` at lines 303-314). Default ``False``.

    Under deterministic mode, ``butterfly_multiply`` backward routes through
    ``butterfly_multiply_torch`` (the pure-PyTorch oracle) — slower but
    deterministic because there is no atomicAdd reduction. Forward path is
    unaffected (no atomicAdd in forward; deterministic-by-construction via
    the multi-launch tile structure).

    Composes additively with ``torch.use_deterministic_algorithms(True)``
    (D-63b): deterministic path is active when EITHER flag is True.
    """
    global _DETERMINISTIC
    prev = _DETERMINISTIC
    _DETERMINISTIC = bool(value)
    return prev
```

**09-03 addition — `_should_route_to_cuda` + `_ROUTING_TABLE` + `_DISABLE_ROUTING` (D-66):**

Pattern is novel-in-file (load JSON at module-import time, expose a per-call selector that takes (op_name, shape, dtype, direction) and returns bool). Module-level state mirrors the existing `_BACKEND` global (line 60) and `_TRITON_PACKAGE_NAMES` dict (lines 114-116):

```python
import json
import pathlib

# ── Runtime routing table (Phase 9 D-66) ─────────────────────────────────
_ROUTING_DISABLED: bool = False
_routing_log_emitted: set = set()


def _load_routing_table() -> dict:
    """Load _routing.json at module-import time. Returns an empty dict when
    the file is absent (Phase 9 09-03 generates this; pre-generation runs
    fall back to the empty table — selector never routes).
    """
    routing_path = pathlib.Path(__file__).parent / "_routing.json"
    if not routing_path.exists():
        return {}
    with open(routing_path) as f:
        return json.load(f).get("rules", {})


_ROUTING_TABLE: dict = _load_routing_table()


def _should_route_to_cuda(op_name: str, shape: tuple, dtype: torch.dtype,
                          direction: str) -> bool:
    """Consult the static routing table — True iff this cell has
    ``route_to_cuda: true`` in _routing.json (D-66a).

    The cell key is "<op>::<log_n>::<dtype>::<direction>" (D-66b — keyed
    object for O(1) lookup, RESEARCH §9). When ``_ROUTING_DISABLED`` is
    True (set by ``set_routing_enabled(False)`` for SC#4 test isolation
    per D-61a), returns False unconditionally.

    On first routing decision per (op, log_n, dtype, direction) tuple,
    emits ``log.info`` once (D-66d). Subsequent calls with the same tuple
    do not re-emit.
    """
    if _ROUTING_DISABLED:
        return False
    # _shape_to_log_n: input shape is (batch, nstacks, n) — log_n = log2(n).
    # Twiddle shape is (nstacks, nblocks, log_n, n//2, 2, 2) — log_n at axis 2.
    # Both shapes appear in the resolver — the selector must accept either.
    # Concrete derivation lives in the helper below.
    log_n = _shape_to_log_n(shape)
    dtype_str = "fp32" if dtype == torch.float32 else (
        "complex64" if dtype == torch.complex64 else str(dtype)
    )
    key = f"{op_name}::{log_n}::{dtype_str}::{direction}"
    cell = _ROUTING_TABLE.get(key)
    if cell is None or not cell.get("route_to_cuda", False):
        return False
    # First-route logging — once per process per (op, log_n, dtype, direction).
    if key not in _routing_log_emitted:
        ratio = cell.get("triton_cuda_ratio_p50", 0.0)
        log.info(
            "torch_structured: routing %s(log_n=%d, dtype=%s, direction=%s) "
            "to CUDA (Triton/CUDA ratio %.2fx > 1.67x threshold)",
            op_name, log_n, dtype_str, direction, ratio,
        )
        _routing_log_emitted.add(key)
    return True


def set_routing_enabled(value: bool) -> bool:
    """Test-time override (NOT exported as user API — internal to _ops).

    Disables the routing selector for the duration of a test (D-66c). Phase 8
    SC#4 test uses this to assert pure-Triton execution (D-61a). Returns
    previous value. Mirrors ``set_backend()`` save/restore pattern.
    """
    global _ROUTING_DISABLED
    prev = _ROUTING_DISABLED
    _ROUTING_DISABLED = not bool(value)
    return not prev
```

**Resolver hook at `_ops.py:220-224` (the `butterfly_multiply` triton branch):**

Wrap the bound `_triton_bm` with a thin shim that consults `_should_route_to_cuda` per call. Current code (lines 216-230):

```python
if actual == "triton":
    if _has_triton_kernel("butterfly_multiply"):
        from torch_structured._triton.butterfly.op import (  # type: ignore[import-not-found]
            butterfly_multiply as _triton_bm,
        )
        butterfly_multiply = _triton_bm
    elif _has_cuda_legacy():
        # ... existing fallback ...
```

Replace with a wrapping closure that intercepts the call:

```python
if actual == "triton":
    if _has_triton_kernel("butterfly_multiply"):
        from torch_structured._triton.butterfly.op import (
            butterfly_multiply as _triton_bm,
        )
        # Phase 9 D-66b: wrap with routing selector. When the cell is marked
        # route_to_cuda AND _has_cuda_legacy_for_op('butterfly_multiply') is
        # True, route the call to the CUDA legacy path (with one-time log).
        # Fallback chain (D-61b): when CUDA .so is missing, fall back to
        # Triton (NOT to torch oracle).
        if _has_cuda_legacy():
            from torch_structured._cuda_legacy import butterfly_multiply as _cuda_bm

            def _routed_butterfly_multiply(twiddle, input_, increasing_stride,
                                            output_size=None):
                # Forward direction at call time; backward direction is
                # selected by register_autograd's _backward callback, which
                # consults the same selector independently.
                if _should_route_to_cuda(
                    "butterfly_multiply", input_.shape, twiddle.dtype, "forward"
                ):
                    return _cuda_bm(twiddle, input_, increasing_stride, output_size)
                return _triton_bm(twiddle, input_, increasing_stride, output_size)

            butterfly_multiply = _routed_butterfly_multiply
        else:
            butterfly_multiply = _triton_bm
    elif _has_cuda_legacy():
        # ... existing fallback unchanged ...
```

---

### `tests/conftest.py` (modified — 09-01 widening to 3-axis)

**Analog:** `tests/conftest.py` itself (current 43-line Phase 6 implementation).

**Current state (lines 22-42):**

```python
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: opt-in comprehensive parameter grid (Phase 7 D-43a)",
    )


@pytest.fixture(params=["torch", "triton"])
def backend(request):
    """Switch backend for the duration of a test, restore after."""
    if request.param == "triton" and not torch_structured._ops._has_any_triton_kernel():
        pytest.skip("No Triton kernel installed (no CUDA or CPU-only runner)")
    original = torch_structured._ops._BACKEND
    chosen = torch_structured._ops.set_backend(request.param)
    yield chosen
    torch_structured._ops.set_backend(original)
```

**Phase 9 09-01 widens to `["torch", "triton", "cuda"]` with per-op skip-gate (D-62b — marker-based op-name detection, RESEARCH-recommended):**

```python
def pytest_configure(config):
    """Register custom markers to silence PytestUnknownMarkWarning."""
    config.addinivalue_line(
        "markers",
        "slow: opt-in comprehensive parameter grid (Phase 7 D-43a)",
    )
    config.addinivalue_line(
        "markers",
        "multigpu: requires >=2 GPUs (Phase 9 D-64 — opt-in via torchrun --nproc_per_node=2)",
    )
    config.addinivalue_line(
        "markers",
        "op(name): bind a test to a specific op for per-op cuda skip-gate "
        "(Phase 9 D-62b; e.g., @pytest.mark.op('butterfly_multiply'))",
    )


@pytest.fixture(params=["torch", "triton", "cuda"])
def backend(request):
    """Switch backend for the duration of a test, restore after.

    Phase 6 D-39 introduced the per-op `_has_any_triton_kernel()` skip-gate
    for the 'triton' axis. Phase 9 D-62 adds the symmetric 'cuda' axis: when
    the test's @pytest.mark.op('<op_name>') marker names an op whose CUDA
    legacy .so is not installed, the cuda parametrize value is skipped.

    Marker-less tests do NOT skip the cuda axis — they're treated as
    backend-agnostic (e.g., tests that don't directly call an _ops surface).
    """
    param = request.param
    if param == "triton" and not torch_structured._ops._has_any_triton_kernel():
        pytest.skip("No Triton kernel installed (no CUDA or CPU-only runner)")
    if param == "cuda":
        op_marker = request.node.get_closest_marker("op")
        op_name = op_marker.args[0] if op_marker and op_marker.args else None
        if op_name and not torch_structured._ops._has_cuda_legacy_for_op(op_name):
            pytest.skip(f"No CUDA legacy .so for {op_name}")
    original = torch_structured._ops._BACKEND
    chosen = torch_structured._ops.set_backend(param)
    yield chosen
    torch_structured._ops.set_backend(original)
```

---

### `torch_structured/__init__.py` (modified — 09-02 export `set_deterministic`)

**Analog:** itself, lines 35, 37-47 — already exports `set_backend` from `_ops`.

**Pattern — append to existing imports + `__all__`:**

```python
from ._ops import set_backend, set_deterministic  # noqa: F401

__all__ = [
    'Butterfly',
    'ButterflyBmm',
    'ButterflyBase4',
    'ButterflyUnitary',
    'butterfly_multiply',
    'LRU',
    'make_linear',
    'set_backend',
    'set_deterministic',
    '__version__',
]
```

---

### `torch_structured/_triton/butterfly/op.py` (modified — 09-02 `_backward` deterministic gate)

**Analog:** `torch_structured/_triton/butterfly/op.py:1377-1383` (existing small-N oracle fallback inside `_backward`). Phase 9 09-02 prepends a SECOND oracle-routing branch at the top of `_backward` (D-63a — wrapper-level gate, NOT kernel-side constexpr).

**Current small-N fallback pattern (lines 1377-1383 — exact code to clone):**

```python
# D-49b small-N fallback (mirror Phase 7 wrapper's log_n <= 1 branch via
# torch.autograd.grad over _butterfly_multiply_torch). Triton launch
# overhead and the trail/scratch machinery dominate at trivial sizes.
if log_n <= 1:
    twiddle_d = twiddle.detach().requires_grad_(True)
    input_d = input_.detach().requires_grad_(True)
    with torch.enable_grad():
        out = _butterfly_multiply_torch(twiddle_d, input_d, increasing_stride, output_size)
    gt, gi = torch.autograd.grad(out, [twiddle_d, input_d], grad_out)
    return gt, gi, None, None
```

**Phase 9 09-02 prepends a deterministic-mode gate BEFORE the small-N branch — same pattern, different predicate:**

```python
def _backward(ctx, grad_out):
    """Plan 08-01 (fp32) + Plan 08-02 (complex64) Triton-backed two-input backward."""
    twiddle, input_ = ctx.saved_tensors
    increasing_stride = ctx.increasing_stride
    output_size = ctx.output_size

    # Phase 9 D-63a: deterministic-mode gate (wrapper-level oracle fallback).
    # When set_deterministic(True) OR torch.use_deterministic_algorithms(True)
    # is active (D-63b additive OR), route through the pure-PyTorch oracle —
    # deterministic by virtue of having no atomicAdd reordering. Identical
    # body shape to the small-N fallback below (lines 1377-1383).
    from torch_structured._ops import _is_deterministic_mode_active
    if _is_deterministic_mode_active():
        twiddle_d = twiddle.detach().requires_grad_(True)
        input_d = input_.detach().requires_grad_(True)
        with torch.enable_grad():
            out = _butterfly_multiply_torch(twiddle_d, input_d, increasing_stride, output_size)
        gt, gi = torch.autograd.grad(out, [twiddle_d, input_d], grad_out)
        return gt, gi, None, None

    # ── existing body unchanged from here ────────────────────────────────
    batch_size, nstacks, input_size = input_.shape
    # ... rest of _backward as it is today (lines 1368-1571) ...
```

---

### `tests/_baseline_butterfly.py` and `tests/_baseline_butterfly_backward.py` (modified — 09-03 CUDA branch)

**Analog:** the existing harness body (forward at lines 102-153; backward at lines 120-190).

**Current Triton measurement pattern (forward, lines 121-127):**

```python
# Triton backend
torch_structured._ops.set_backend("triton")
def triton_call():
    return torch_structured._ops.butterfly_multiply(
        twiddle, input_, True, n
    )
p50_triton, p95_triton = measure_p50_p95(triton_call)
```

**Phase 9 09-03 adds a sibling CUDA measurement (after the Triton block, before writing the row):**

```python
# CUDA backend (Phase 9 D-65a — extends 16 rows with reference_cuda_p50).
# When _butterfly.so is not built, leave reference_cuda_p50: null and the
# perf gate (D-65b) loosens to "Triton >= 60% of reference_torch_ref_p50"
# with the 5x threshold per Phase 7's documented torch-ref weaker gate.
p50_cuda = None
if torch_structured._ops._has_cuda_legacy():
    torch_structured._ops.set_backend("cuda")
    def cuda_call():
        return torch_structured._ops.butterfly_multiply(
            twiddle, input_, True, n,
        )
    p50_cuda, _ = measure_p50_p95(cuda_call)

# Row schema extended per D-65a — null when CUDA legacy not built.
row = {
    "kernel": "butterfly_multiply",
    "dtype": dtype_name,
    "log_n": log_n,
    "nstacks": nstacks,
    "nblocks": nblocks,
    "wall_ms_p50": round(p50_triton, 6),
    "wall_ms_p95": round(p95_triton, 6),
    "reference_torch_ref_p50": round(p50_ref, 6),
    "reference_cuda_p50": (round(p50_cuda, 6) if p50_cuda is not None else None),
    "measured_at": datetime.datetime.now(datetime.timezone.utc)
        .isoformat().replace("+00:00", "Z"),
    "gpu": gpu_name,
    "direction": "forward",
}
```

Backward harness mirrors this — same pattern, `direction="backward"` and the backward-call shape (`out.backward(grad_out, retain_graph=True)` per the existing closure at lines 150-157).

---

### `tests/test_butterfly_triton.py` (modified — 09-03 SC#4 reconciliation)

**Analog:** `tests/test_butterfly_triton.py:705-774` (the existing `test_butterfly_backward_no_cpp_symbol` body — read fully above).

**Phase 9 D-61a updates required:**

1. Choose `log_n = 4` (verified low enough that Triton is faster than CUDA → selector won't route).
2. Add `_DISABLE_ROUTING` flag during the test body via `set_routing_enabled(False)` in a try/finally.
3. Update docstring + add a comment referencing `_routing.json`.

**Pattern (modify the existing test in place — lines 705-774):**

```python
def test_butterfly_backward_no_cpp_symbol():
    """SC#4 / D-53 — under BACKEND=triton AND DEFAULT routing, no
    csrc/butterfly.cpp symbol is invoked.

    Phase 9 D-61a reconciliation: with the runtime selector active
    (torch_structured/_routing.json), BACKEND=triton CAN invoke csrc for
    shapes where the selector routes to CUDA per D-66. This test asserts
    the **default routing** path is pure-Triton, by:

      1. Using log_n=4 — well above the small-N fallback at log_n<=1, but
         below any cell where Triton trails CUDA (the routing table marks
         only log_n in {8,9,10,11} cells; log_n=4 is never routed).
      2. Asserting at setup time that the chosen log_n is NOT in the
         route-to-cuda list of _routing.json (regression detector — if a
         future routing table regen marks log_n=4, this assertion fires).
      3. Disabling routing for the test duration via set_routing_enabled
         (D-66c test-time override) as belt-and-braces.

    Shapes where the runtime selector intentionally routes to CUDA are
    documented at torch_structured/_routing.json and are exercised by
    tests/test_perf_grid.py::test_routing_selector_*.
    """
    torch_structured._ops.set_backend("triton")

    # D-61a precondition: assert log_n=4 is not in the route-to-cuda list.
    log_n = 4
    routing_key = f"butterfly_multiply::{log_n}::fp32::backward"
    assert not torch_structured._ops._ROUTING_TABLE.get(routing_key, {}).get(
        "route_to_cuda", False
    ), (
        f"SC#4: log_n={log_n} unexpectedly marked route_to_cuda in _routing.json. "
        f"Choose a different log_n for this test or regenerate the routing table."
    )

    # D-66c: disable routing during the test (belt-and-braces).
    prev_routing = torch_structured._ops.set_routing_enabled(False)

    # Part (a): dispatch-binding assertion (cheap, deterministic) — but the
    # binding is now the routing-shim closure (Phase 9 D-66b), not _triton_bm.
    # Note for the planner: this `is`-check needs to be reconsidered against
    # the new resolver. Either:
    #   - Drop the `is`-check and rely on the monkey-patch (Part b) alone
    #   - Wrap _triton_bm via routing-shim AND assert on
    #     ``_ops.butterfly_multiply.__closure__[?].cell_contents is _triton_bm``
    # Recommended: drop the `is`-check entirely (it was a Phase 8 indirection
    # check that conflicts with the Phase 9 routing shim).

    # Part (b): runtime invocation tracking via monkey-patch shim — UNCHANGED.
    raised_calls = []
    original_fw = _legacy_mod_for_sc4.butterfly_multiply_fw
    original_bw = _legacy_mod_for_sc4.butterfly_multiply_bw

    def _fail_fw(*a, **kw):
        raised_calls.append("fw")
        raise AssertionError(
            "SC#4: csrc/butterfly.cpp::butterfly_multiply_fw invoked under BACKEND=triton"
        )

    def _fail_bw(*a, **kw):
        raised_calls.append("bw")
        raise AssertionError(
            "SC#4: csrc/butterfly.cpp::butterfly_multiply_bw invoked under BACKEND=triton"
        )

    _legacy_mod_for_sc4.butterfly_multiply_fw = _fail_fw
    _legacy_mod_for_sc4.butterfly_multiply_bw = _fail_bw
    try:
        n = 1 << log_n  # log_n=4 -> n=16
        twiddle = torch.randn(
            1, 1, log_n, n // 2, 2, 2,
            device="cuda", dtype=torch.float32, requires_grad=True,
        )
        x = torch.randn(4, 1, n, device="cuda", dtype=torch.float32, requires_grad=True)
        out = torch_structured._ops.butterfly_multiply(twiddle, x, True, n)
        loss = out.sum()
        loss.backward()
        assert raised_calls == [], f"SC#4 violated: legacy symbols invoked: {raised_calls}"
        assert twiddle.grad is not None
        assert x.grad is not None
    finally:
        _legacy_mod_for_sc4.butterfly_multiply_fw = original_fw
        _legacy_mod_for_sc4.butterfly_multiply_bw = original_bw
        torch_structured._ops.set_routing_enabled(prev_routing)
```

---

### `tests/test_phase9_integration.py` (NEW — 09-01)

**Analog:** `tests/test_butterfly_triton.py:60-127` (parametrized `backend` tests + `pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), ...)`).

**Imports pattern** (mirror `tests/test_butterfly_triton.py:37-46`):

```python
"""Phase 9 09-01 integration tests — backend-agreement + checkpoint round-trip
+ make_linear + LRU smoke + public API regression detector.

Tests parametrized via the widened `backend` fixture (tests/conftest.py
Phase 9 D-62, axis ["torch", "triton", "cuda"]). The cuda axis is
per-test-skipped via @pytest.mark.op('<op_name>') markers (D-62b).

Tolerances (D-62c, inherited Phase 7+8):
  fp32: rtol=1e-5, atol=1e-6
  complex64 d_twiddle: rtol=1e-3, atol=1e-4
  complex64 d_input: rtol=1e-5, atol=1e-6
"""
import inspect
import itertools

import pytest
import torch

import torch_structured  # noqa: F401
from torch_structured import Butterfly, ButterflyBmm, ButterflyBase4, ButterflyUnitary
from torch_structured import LRU, make_linear


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Phase 9 integration tests require CUDA"
)
```

**Backend-agreement pattern (clone from `tests/test_butterfly_triton.py:60-72`):**

```python
@pytest.mark.op("butterfly_multiply")
@pytest.mark.parametrize("log_n", [2, 4, 8])
def test_backend_agreement_butterfly_fp32(backend, log_n):
    """All three backends produce equal outputs within Phase 7+8 tolerances."""
    n = 1 << log_n
    nstacks, nblocks, batch_size = 1, 1, 4
    twiddle = torch.randn(nstacks, nblocks, log_n, n // 2, 2, 2, device="cuda", dtype=torch.float32)
    input_ = torch.randn(batch_size, nstacks, n, device="cuda", dtype=torch.float32)
    out = torch_structured._ops.butterfly_multiply(twiddle, input_, True, n)
    # Compare against the torch oracle (the parametrize value is the backend
    # under test; the oracle is the fixed reference).
    from torch_structured._torch_ref.butterfly import butterfly_multiply_torch
    expected = butterfly_multiply_torch(twiddle, input_, True, n)
    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-6), (
        f"backend={backend} log_n={log_n} fp32 disagreement"
    )
```

**Checkpoint round-trip pattern (RESEARCH §7, lines 632-680 — copy that test verbatim).**

**`make_linear`/`LRU` smoke tests (RESEARCH §10, lines 914-957 — copy with the corrected `make_linear("butterfly", 256, 256)` signature, NOT `structure='butterfly'`).**

**Public API regression detector (RESEARCH §13, lines 1112-1144 — `inspect.signature` snapshot pattern):**

```python
def test_public_api_butterfly_init_signature_unchanged():
    """COMPAT-01 — Butterfly.__init__ signature byte-identical to v1.1.

    The exact signature string must be captured by reading
    torch_structured/butterfly/butterfly.py before the test is written
    (RESEARCH §13 — do not rely on training data; planner does the capture).
    """
    # Planner captures from the live file at task time.
    expected_sig = "..."  # filled in by planner
    actual = str(inspect.signature(Butterfly.__init__))
    assert actual == expected_sig, (
        f"COMPAT-01: Butterfly.__init__ signature changed.\n  v1.1: {expected_sig}\n  now:  {actual}"
    )
```

---

### `tests/test_torch_compile_triton.py` (NEW — 09-02)

**Analog:** RESEARCH §2 + §3 (lines 219-343 — verified `fullgraph=True` + FakeTensorMode patterns).

**No direct test-file analog in the repo** — `tests/test_dispatch.py` is the closest (Phase 4 dispatch contract tests) but covers `set_backend` round-trips, not `torch.compile`. Phase 9 09-02 introduces the pattern. Use RESEARCH §2-§3 verbatim:

```python
"""Phase 9 09-02 torch.compile + FakeTensorMode integration tests."""
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import torch_structured  # noqa: F401
from torch_structured import Butterfly, ButterflyBmm, LRU, make_linear


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="torch.compile tests require CUDA"
)


@pytest.mark.op("butterfly_multiply")
def test_torch_compile_butterfly_fullgraph(backend):
    """fullgraph=True asserts NO graph breaks anywhere in the trace.
    Resolves 260419-p27 end-to-end through the nn.Module surface (per
    LANDMINE §0 fix to torch_structured/butterfly/multiply.py).
    """
    if backend != "triton":
        pytest.skip("torch.compile no-break gate is meaningful only for triton path")

    m = Butterfly(in_size=256, out_size=256, bias=False, complex=False).cuda()
    m_compiled = torch.compile(m, fullgraph=True)
    x = torch.randn(8, 256, device="cuda", requires_grad=True)
    y = m_compiled(x)
    loss = y.sum()
    loss.backward()  # fullgraph=True covers AOT'd backward — see RESEARCH Pitfall 3.
    assert x.grad is not None and torch.isfinite(x.grad).all()


# Plus: test_torch_compile_butterfly_bmm, test_torch_compile_lru,
# test_torch_compile_make_linear, test_butterfly_under_fake_tensor_mode
# (260419-p27 acceptance gate per RESEARCH §3, lines 287-318).
```

---

### `tests/test_distributed_triton.py` (NEW — 09-02 DDP/FSDP/checkpoint)

**Analog:** RESEARCH §1 (FSDP1 pattern, lines 167-198) and §6 (gradient checkpoint pattern, lines 575-605). No closer in-repo analog — DDP/FSDP idioms are new in this repo.

**Key patterns to copy:**

- **FSDP1 with `ignored_modules`** — RESEARCH §1, lines 167-198 (must use NCCL backend, must opt-in via `@pytest.mark.multigpu`).
- **DDP single-process smoke** — RESEARCH ~§1.5 / CONTEXT.md D-64c (gloo backend, init_process_group rank=0 world_size=1).
- **Gradient checkpointing** — RESEARCH §6, lines 575-605 (`use_reentrant=False`, rtol=1e-5 atol=1e-6).

---

### `tests/test_deterministic_mode.py` (NEW — 09-02)

**Analog:** `tests/test_dispatch.py:17-30` (set_backend round-trip test — same toggle-assert-restore shape).

**Pattern from `test_dispatch.py` (verified existing pattern for state-mutation tests):**

```python
def test_set_backend_round_trip():
    original = torch_structured._ops._BACKEND
    chosen = torch_structured.set_backend("torch")
    assert chosen == "torch"
    assert torch_structured._ops._BACKEND == "torch"
    torch_structured.set_backend(original)
```

**Phase 9 09-02 clones this shape for `set_deterministic`:**

```python
def test_set_deterministic_round_trip():
    """API contract — toggle returns previous value, default is False."""
    original = torch_structured._ops._DETERMINISTIC
    prev = torch_structured.set_deterministic(True)
    assert prev is original
    assert torch_structured._ops._DETERMINISTIC is True
    prev2 = torch_structured.set_deterministic(False)
    assert prev2 is True
    assert torch_structured._ops._DETERMINISTIC is False
    torch_structured.set_deterministic(original)


def test_deterministic_dtwiddle_bit_identical(...):
    # Full pattern at RESEARCH §4, lines 425-447 — copy verbatim.
    ...
```

---

### `tests/test_perf_grid.py` (NEW — 09-03)

**Analog:** none in repo. Pattern from CONTEXT.md `<specifics>` (lines 348-361) and RESEARCH §9.

**No prior pattern to copy** — JSON-driven gate is novel. The pattern is documented inline:

```python
"""Phase 9 09-03 perf gate — TEST-04 (>=60% of CUDA throughput)."""
import json
from pathlib import Path

import pytest


BASELINE_PATH = Path(".planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json")
THRESHOLD = 1.0 / 0.60  # = 1.6667


@pytest.mark.gpu_required
def test_perf_gate_triton_at_60pct_cuda():
    """Read 07-BASELINE.json, assert every cell has triton_p50 / cuda_p50 <= 1.67."""
    baseline = json.loads(BASELINE_PATH.read_text())
    failures = []
    for row in baseline["rows"]:
        if row.get("reference_cuda_p50") is None:
            continue  # cell without CUDA measurement — weaker fallback gate
        ratio = row["wall_ms_p50"] / row["reference_cuda_p50"]
        if ratio > THRESHOLD:
            failures.append((row["kernel"], row["log_n"], row["dtype"],
                             row["direction"], round(ratio, 3)))
    # Sort descending by ratio (worst-perf cells first — RESEARCH discretion).
    failures.sort(key=lambda f: -f[-1])
    assert not failures, f"TEST-04 below 60% of CUDA on cells (sorted by ratio): {failures}"
```

---

### `scripts/regenerate_routing_table.py` (NEW — 09-03)

**Analog:** `tests/_baseline_butterfly.py` (CLI script shell — `def main() -> int` + `sys.exit(main())` at lines 90, 168-169).

**Pattern: copy the CLI shell from `_baseline_butterfly.py`:**

```python
#!/usr/bin/env python
"""Regenerate torch_structured/_routing.json from 07-BASELINE.json.

For each (op, log_n, dtype, direction) cell, computes
ratio = wall_ms_p50_triton / reference_cuda_p50 and sets route_to_cuda
when ratio > 1.67 (= 1/0.60). When reference_cuda_p50 is None, falls back
to comparing against reference_torch_ref_p50 with the 5.0x threshold
(Phase 7's documented torch-ref weaker gate).

Run: python scripts/regenerate_routing_table.py
"""
import datetime
import json
import sys
from pathlib import Path


BASELINE = Path(".planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json")
ROUTING = Path("torch_structured/_routing.json")
THRESHOLD = 1.0 / 0.60  # = 1.6667
TORCH_REF_THRESHOLD = 5.0


def main() -> int:
    rows = json.loads(BASELINE.read_text())["rows"]
    rules = {}
    for row in rows:
        if row.get("reference_cuda_p50"):
            ratio = row["wall_ms_p50"] / row["reference_cuda_p50"]
            route = ratio > THRESHOLD
        else:
            ratio = row["wall_ms_p50"] / row["reference_torch_ref_p50"]
            route = ratio > TORCH_REF_THRESHOLD
        key = f"{row['kernel']}::{row['log_n']}::{row['dtype']}::{row['direction']}"
        rules[key] = {
            "triton_cuda_ratio_p50": round(ratio, 4),
            "route_to_cuda": route,
        }

    output = {
        "schema_version": 1,
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
            .isoformat().replace("+00:00", "Z"),
        "generated_from": str(BASELINE),
        "threshold_ratio": THRESHOLD,
        "rules": rules,
    }
    ROUTING.parent.mkdir(parents=True, exist_ok=True)
    ROUTING.write_text(json.dumps(output, indent=2) + "\n")
    routed = sum(1 for r in rules.values() if r["route_to_cuda"])
    print(f"Wrote {ROUTING} with {len(rules)} rules; {routed} marked route_to_cuda")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

### `torch_structured/_routing.json` (NEW — 09-03)

**Analog:** `.planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json` (existing JSON-on-disk pattern, top-level `{"rows": [...]}`).

**Schema (RESEARCH §9, keyed-object form for O(1) lookup):**

```json
{
  "schema_version": 1,
  "generated_at": "2026-05-28T...",
  "generated_from": ".planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json",
  "threshold_ratio": 1.6667,
  "rules": {
    "butterfly_multiply::8::fp32::forward":    {"triton_cuda_ratio_p50": 0.85, "route_to_cuda": false},
    "butterfly_multiply::8::complex64::forward":{"triton_cuda_ratio_p50": 1.10, "route_to_cuda": false},
    "butterfly_multiply::11::complex64::backward":{"triton_cuda_ratio_p50": 1.92, "route_to_cuda": true}
  }
}
```

---

### `.github/workflows/test.yml` (modified) or `test-gpu.yml` (NEW — 09-03)

**Analog:** `.github/workflows/test.yml` (current single-job 49-line CPU workflow).

**Pattern from current file (verified — lines 18-48):** A single `test` job on `ubuntu-latest` with Triton cache + `pip install -e .[test]` + `pytest tests/test_dispatch.py -v`.

**Phase 9 09-03 extension pattern (RESEARCH §12, lines 1062-1093):**

```yaml
jobs:
  test-cpu:
    # existing job — keep unchanged, but expand pytest command to
    # `pytest tests/ -m "not multigpu and not gpu_required"` so non-GPU
    # tests run in CPU CI.
    ...

  test-triton:
    runs-on: [self-hosted, gpu, ampere]
    if: ${{ vars.ENABLE_GPU_CI == 'true' }}
    env:
      TORCH_STRUCTURED_BACKEND: triton
    steps:
      # ... checkout, setup-python, install ...
      - uses: actions/cache@v4
        with:
          path: ~/.triton/cache
          key: triton-self-hosted-py3.11-torch${{ env.TORCH_VERSION }}-${{ hashFiles('torch_structured/_triton/**/*.py') }}
      - run: pytest tests/ -m "not multigpu" -v

  test-multigpu:
    runs-on: [self-hosted, gpu, multi-gpu]
    if: ${{ vars.ENABLE_MULTIGPU_CI == 'true' }}
    needs: test-triton
    env:
      TORCH_STRUCTURED_BACKEND: triton
    steps:
      # ... same setup as test-triton, SHARE the cache key ...
      - run: torchrun --nproc_per_node=2 -m pytest tests/test_distributed_triton.py -v -m multigpu
```

Pitfall (RESEARCH §12 line 1095): `pytest -m multigpu` requires the marker registered in `conftest.py:pytest_configure` — 09-01 adds the registration; without it, `pytest -m multigpu` runs nothing AND emits a warning.

---

## Shared Patterns

### Pattern 1: D-05 attribute-access delegator
**Source:** `torch_structured/structured/hadamard.py:25-34` (verified — the canonical Phase 6 implementation)
**Apply to:** `torch_structured/butterfly/multiply.py` (09-01 LANDMINE fix), any future back-compat shim that must re-route through `_ops`

```python
import torch_structured  # noqa: F401

def <op_name>(*args, **kwargs):
    """Back-compat shim — delegates to torch_structured._ops.<op_name>.
    Re-reads the binding on every call so set_backend() rebindings take effect.
    """
    return torch_structured._ops.<op_name>(*args, **kwargs)
```

**Why:** Python's `from X import Y` captures the current object at import time. Attribute access on the module re-reads the binding on every call — load-bearing for `set_backend()` to take effect at consumer call sites.

### Pattern 2: Per-op honest probe (CHECKER B3)
**Source:** `torch_structured/_ops.py:82-107` (existing `_has_cuda_legacy_diag_mult` and `_has_cuda_legacy_hadamard` — verified)
**Apply to:** `_has_cuda_legacy_for_op(op_name)` (09-01 D-62a)

```python
def _has_cuda_legacy_<op>() -> bool:
    """Per-op honest probe (CHECKER B3) — returns True iff <op>'s legacy .so
    extension was built and importable. Never raises; returns a clean bool.
    """
    try:
        from torch_structured._cuda_legacy.<op> import HAS_CUDA_LEGACY_<OP>
        return HAS_CUDA_LEGACY_<OP>
    except ImportError:
        return False
```

**Why:** `_cuda_legacy/*.py` is the ONE place in the codebase where try/except is sanctioned (CLAUDE.md exception per Phase 5 D-21 honest-probe pattern). All other code uses `assert` for preconditions and never wraps imports.

### Pattern 3: save/restore setter shape (mirror `set_backend`)
**Source:** `torch_structured/_ops.py:303-314` (existing `set_backend(name: str) -> str` — verified)
**Apply to:** `set_deterministic(value: bool) -> bool` (09-02 D-63), `set_routing_enabled(value: bool) -> bool` (09-03 D-66c)

```python
def set_<flag>(value: <T>) -> <T>:
    """Public API: toggle <flag>. Returns the previous value for the
    save/restore pattern. Mirrors set_backend() shape.
    """
    global _<FLAG>
    prev = _<FLAG>
    _<FLAG> = <T>(value)
    return prev
```

**Why:** Returning the previous value supports the test fixture save/restore pattern (`prev = set_X(True); try: ...; finally: set_X(prev)`) — used everywhere in `tests/conftest.py::backend` fixture lines 39-42.

### Pattern 4: pytestmark CUDA skip
**Source:** `tests/test_butterfly_triton.py:49-51` (verified — exact 3-line skip block)
**Apply to:** all new test files in 09-01, 09-02 (`test_phase9_integration.py`, `test_torch_compile_triton.py`, `test_distributed_triton.py`, `test_deterministic_mode.py`)

```python
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="<phase>'s tests require CUDA"
)
```

**Why:** Module-level pytestmark applies to every test in the file. CUDA-only tests share the same skip predicate; no per-test boilerplate.

### Pattern 5: Backend fixture for `_ops` attribute access
**Source:** `tests/conftest.py:34-42` (verified — current fixture)
**Apply to:** Every test that uses `torch_structured._ops.*` and wants backend-axis coverage

```python
@pytest.fixture(params=["torch", "triton", "cuda"])  # Phase 9 — 3-axis
def backend(request):
    # ... skip-gate logic ...
    original = torch_structured._ops._BACKEND
    chosen = torch_structured._ops.set_backend(request.param)
    yield chosen
    torch_structured._ops.set_backend(original)
```

**Why:** Save/restore is critical because pytest runs tests in undefined order; a test that leaves `_BACKEND = "cuda"` would break subsequent tests that assume default.

### Pattern 6: Oracle fallback inside `_backward` (D-49b small-N pattern)
**Source:** `torch_structured/_triton/butterfly/op.py:1377-1383` (verified Phase 8 small-N fallback)
**Apply to:** Phase 9 D-63a deterministic-mode gate (09-02)

```python
if <gate_predicate>:
    twiddle_d = twiddle.detach().requires_grad_(True)
    input_d = input_.detach().requires_grad_(True)
    with torch.enable_grad():
        out = _butterfly_multiply_torch(twiddle_d, input_d, increasing_stride, output_size)
    gt, gi = torch.autograd.grad(out, [twiddle_d, input_d], grad_out)
    return gt, gi, None, None
```

**Why:** The pattern is exact-reusable because both the small-N branch and the deterministic-mode branch want the same behavior — route through `_butterfly_multiply_torch` via `torch.autograd.grad`. The oracle is deterministic by virtue of pure-PyTorch tensor ops (no atomicAdd).

### Pattern 7: `assert` preconditions (no try/except in core lib)
**Source:** `CLAUDE.md` project-level rule + verified throughout `torch_structured/_triton/butterfly/op.py:1387-1390` (the dtype-gate assert)
**Apply to:** All new core-lib code in `_ops.py` and `_triton/butterfly/op.py`

**Exception:** `_cuda_legacy/*.py` honest-probe try-imports — sanctioned by D-21. The Phase 9 `_load_routing_table()` function uses `pathlib.Path.exists()` check (NOT try/except) to gracefully handle the pre-generation case.

---

## No Analog Found

Files with no close in-repo match:

| File | Role | Data Flow | Reason / Reference |
|------|------|-----------|--------------------|
| `tests/test_perf_grid.py` | test | JSON read + ratio gate | No prior JSON-driven gate test in repo; pattern documented in CONTEXT.md `<specifics>` (illustrative example at lines 348-361). |
| `torch_structured/_routing.json` | static data | data lookup | Schema is novel; defined in RESEARCH §9 (keyed-object form for O(1) lookup). |
| `CHANGELOG.md` | docs | release notes | File does not exist (verified via `ls`); RESEARCH §11 prescribes Keep a Changelog v1.1 format. |
| `tests/test_distributed_triton.py` | test | DDP/FSDP/checkpoint | No prior DDP/FSDP test in repo; FSDP1 pattern at RESEARCH §1 lines 167-198 is verbatim from PyTorch 2.6 docs. |

---

## Metadata

**Analog search scope:** `torch_structured/**/*.py`, `tests/**/*.py`, `.github/workflows/*.yml`, `.planning/phases/0[5-8]-*/0[5-8]-*.md`
**Files scanned:** ~80 (Python core + tests + research docs + CI workflows)
**Pattern extraction date:** 2026-05-28

## PATTERN MAPPING COMPLETE
