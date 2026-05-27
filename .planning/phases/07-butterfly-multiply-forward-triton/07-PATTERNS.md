# Phase 7: butterfly_multiply Forward (Triton) - Pattern Map

**Mapped:** 2026-05-27
**Files analyzed:** 4 new (1 marker, 1 op, 1 test, 1 baseline-data)
**Analogs found:** 4 / 4 (all exact or role-match; baseline JSON has no in-repo precedent and gets a structural-only assignment)

This phase is the third "real op port" in the Triton-migration milestone (after Phase 5 diag_mult and Phase 6 hadamard_transform). The infrastructure surface — `_ops.py` resolver block, `_cuda_legacy/butterfly.py`, `_torch_ref/butterfly.py`, `tests/conftest.py` `backend` fixture, `_has_any_triton_kernel()` probe — is **all pre-wired** by Phases 4–6. Phase 7's authoring surface collapses to **one Triton op file + one test file + one baseline JSON**.

## File Classification

| New File | Role | Data Flow | Closest Analog | Match Quality |
|----------|------|-----------|----------------|---------------|
| `torch_structured/_triton/butterfly/__init__.py` | package-marker | re-export | `torch_structured/_triton/diag_mult/__init__.py` (and `_triton/hadamard_transform/__init__.py` — identical shape) | exact (3-line file; all three Triton-op packages share this shape verbatim) |
| `torch_structured/_triton/butterfly/op.py` | service / Triton kernel + autograd op | compute (GPU kernel) + autograd (two-input gradient via oracle) + meta (register_fake) | `_triton/diag_mult/op.py` (IS_COMPLEX scaffolding + 4-FMA template + two-tensor `save_for_backward` + Wirtinger backward via `_torch_ref`) AND `_triton/hadamard_transform/op.py` (most recent, register-resident tile pattern, normalize-style wrapper-side post-processing, `tl.static_range(LOG_N)` loop) | role-match — **substantive kernel-body divergence per D-40b multi-launch 3-stage tile**; structural skeleton transcribes verbatim |
| `tests/test_butterfly_triton.py` | test | request-response (function call) | `tests/test_diag_mult.py` (top-level, fixture-parametrized, `torch_structured._ops.<op>` attribute access, pytest skipif on CUDA, two-input gradcheck pattern) | exact — top-level test file mirroring Phase 5 layout (NOT `tests/structured/test_hadamard_triton.py` which sits under `tests/structured/` because Phase 6 mirrored an existing `tests/structured/test_hadamard.py`; no `tests/structured/test_butterfly_triton.py` analog exists, so Phase 7 lands at `tests/test_butterfly_triton.py`) |
| `.planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json` | config / data artifact | one-shot write | none in-repo | no-analog — Phase 7 is the first phase to emit a perf-baseline JSON; schema is locked in 07-CONTEXT.md D-43b; assignment is structural-only |

## Pattern Assignments

### `torch_structured/_triton/butterfly/__init__.py` (package-marker, re-export)

**Analog:** `torch_structured/_triton/diag_mult/__init__.py` (3 lines; identical pattern at `_triton/hadamard_transform/__init__.py`)

**Pattern to copy verbatim** (with name substitution `diag_mult` → `butterfly_multiply`):

```python
"""Triton kernel package for butterfly_multiply — Phase 7 (TRI-03)."""
from .op import butterfly_multiply  # noqa: F401 (re-exported)

__all__ = ["butterfly_multiply"]
```

**Why it matters:** The `_has_triton_kernel("butterfly_multiply")` probe at `_ops.py:104-114` imports `torch_structured._triton.butterfly.op` and checks `hasattr(mod, "butterfly_multiply")` — only `op.py` is strictly required. The `__init__.py` is the ergonomic top-level access pattern; mirroring it across all three Triton-op packages keeps the per-op skeleton uniform.

---

### `torch_structured/_triton/butterfly/op.py` (service, Triton kernel + autograd + meta)

**Analogs (transcribed jointly):**
- **`_triton/diag_mult/op.py`** — IS_COMPLEX scaffolding + 4-FMA template + two-tensor `save_for_backward` + backward via `_torch_ref` oracle.
- **`_triton/hadamard_transform/op.py`** — most recent op port; `tl.static_range(LOG_N)` butterfly-stage loop; register-resident tile concept (closest in spirit to Phase 7's 3-stage tile though Phase 7 multi-launches instead of single-pass).

**Imports pattern** (verbatim transcription from `_triton/diag_mult/op.py:21-26`, name-substituted; note D-40a wrapper also needs `from torch.nn import functional as F` for the input pad):

```python
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from torch.nn import functional as F  # NEW vs prior templates: input pad per D-42

from torch_structured._torch_ref.butterfly import butterfly_multiply_torch as _butterfly_multiply_torch  # backward oracle (D-47, two-input)
```

**Kernel signature template** (combining `_triton/diag_mult/op.py:29-41` IS_COMPLEX + `_triton/hadamard_transform/op.py:42-49` static_range loop):

```python
@triton.jit
def _butterfly_kernel(
    twiddle_ptr,
    input_ptr,
    output_ptr,
    batch_size_x_nstacks,
    N,
    # Per-launch stage-group constexprs (D-40a)
    STAGE_START: tl.constexpr,
    STAGE_COUNT: tl.constexpr,
    INCREASING_STRIDE: tl.constexpr,
    # Phase 4 / D-40b layout flag (gated off in 07-01 via tl.static_assert; lit in 07-02)
    IS_COMPLEX: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """3-stage register-resident butterfly tile (D-40, D-40b).

    Each program loads TILE_N elements per (batch, nstack) row, runs
    STAGE_COUNT (<=3) butterfly stages on the in-register tile via tl.where
    partner-swap, and stores once at the end. No tl.debug_barrier needed —
    intra-launch state stays in registers (lesson learned from Phase 6,
    06-01-SUMMARY.md).
    """
    ...
```

**Plan 07-01 IS_COMPLEX gate (the load-bearing 07-01 → 07-02 transition pattern):**

```python
# Inside the kernel body, immediately at function entry:
tl.static_assert(not IS_COMPLEX, "complex64 lands in 07-02 (D-41a pre-wiring)")
# Plan 07-02 removes ONLY this line and implements the 4-FMA branch per
# .planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/04-COMPLEX-LAYOUT.md
# lines 58-76. Zero kernel-signature refactor between plans.
```

**Wrapper pattern** (combining `_triton/diag_mult/op.py:97-162` assert-block + IS_COMPLEX gating + `_triton/hadamard_transform/op.py:113-145` wrapper-side post-processing pattern):

Verbatim assert + view_as_real boundary (copy from `_triton/diag_mult/op.py:113-139`):

```python
@triton_op("torch_structured::butterfly_multiply", mutates_args={})
def butterfly_multiply(
    twiddle: torch.Tensor,
    input: torch.Tensor,
    increasing_stride: bool = True,
    output_size: Optional[int] = None,
) -> torch.Tensor:
    # Wrapper-boundary preconditions (Phase 5 D-20 + Pitfall 3 + 07-CONTEXT.md "code_context"):
    assert input.dim() == 3, f"input must be (batch, nstacks, input_size), got dim={input.dim()}"
    assert twiddle.dtype == input.dtype, (
        f"twiddle.dtype ({twiddle.dtype}) must equal input.dtype ({input.dtype})"
    )
    assert input.is_contiguous(), "input must be contiguous (Pitfall 3)"
    assert twiddle.is_contiguous(), "twiddle must be contiguous (Pitfall 3)"
    # Plan 07-01 fp32-only gate; Plan 07-02 relaxes to {float32, complex64}.
    assert input.dtype == torch.float32, (
        f"Plan 07-01: fp32-only (complex64 lands in 07-02); got {input.dtype}"
    )
    batch_size, nstacks, input_size = input.shape
    nblocks = twiddle.shape[1]
    log_n = twiddle.shape[2]
    n = 1 << log_n
    assert twiddle.shape == (nstacks, nblocks, log_n, n // 2, 2, 2)
    output_size_actual = n if output_size is None else output_size
    assert output_size_actual <= n
```

**Small-N fallback pattern (D-42a)** — Phase 7's unique divergence; no prior Triton op had this guard:

```python
    # D-42a: small-N fallback. Bypasses kernel for n=1, n=2 where Triton launch
    # overhead dominates and the smallest 3-stage tile (TILE_N=8) is larger than n.
    if log_n <= 1:
        return _butterfly_multiply_torch(twiddle, input, increasing_stride, output_size)
```

**Pad/trim wrapping (D-42)** — mirrors `_torch_ref/butterfly.py:18, 33` verbatim:

```python
    # D-42: wrapper-side pad/trim (mirrors _torch_ref/butterfly.py:18 + 33).
    input = F.pad(input, (0, n - input_size)) if input_size < n else input[:, :, :n]
    input = input.contiguous()  # F.pad already contiguous; explicit for kernel pointer math
```

**Python-side nblocks loop + stage groups (D-40a)** — mirrors `_torch_ref/butterfly.py:22-32` shape (the verbatim oracle) with kernel launches replacing the inner `(t * output_reshape).sum(dim=4)` body:

```python
    output = input  # carries through stage-group launches
    cur_increasing_stride = increasing_stride
    is_complex = input.is_complex()

    # Phase 4 04-COMPLEX-LAYOUT.md D-02 wrapper boundary (gated off in 07-01)
    if is_complex:
        output_work = torch.view_as_real(output)
        twiddle_work = torch.view_as_real(twiddle)
    else:
        output_work = output
        twiddle_work = twiddle

    for block in range(nblocks):
        # Per-block: ceil(log_n / 3) launches, each handling up to 3 stages.
        stage_order = list(range(log_n)) if cur_increasing_stride else list(reversed(range(log_n)))
        for group_start in range(0, log_n, 3):
            group_stages = stage_order[group_start:group_start + 3]
            tile_n = 1 << (max(group_stages) + 1)
            n_row_tiles = n // tile_n
            grid = (n_row_tiles, batch_size * nstacks)  # D-40c 2-D grid
            num_warps = _pick_num_warps(tile_n)  # D-40d
            wrap_triton(_butterfly_kernel)[grid](
                twiddle_work, output_work, output_work,  # in-place per-stage-group
                batch_size * nstacks, n,
                STAGE_START=group_stages[0],
                STAGE_COUNT=len(group_stages),
                INCREASING_STRIDE=cur_increasing_stride,
                IS_COMPLEX=is_complex,
                TILE_N=tile_n,
                num_warps=num_warps,
            )
        cur_increasing_stride = not cur_increasing_stride  # toggle (mirrors _torch_ref:32)

    if is_complex:
        output = torch.view_as_complex(output_work.contiguous())
    else:
        output = output_work

    return output[:, :, :output_size_actual]  # D-42 trim (mirrors _torch_ref:33)
```

**`_setup_context` pattern** (copy from `_triton/diag_mult/op.py:165-170`; the two-tensor `save_for_backward` is the only Phase 7-relevant element — the 4-input signature requires saving both `twiddle` and `input` plus the two non-tensor flags):

```python
def _setup_context(ctx, inputs, output):
    twiddle, input_, increasing_stride, output_size = inputs
    ctx.save_for_backward(twiddle, input_)
    ctx.increasing_stride = increasing_stride
    ctx.output_size = output_size
```

**`_backward` pattern (D-47 two-input variant)** — substantive divergence from Phase 5/6:

- Phase 5 (`diag_mult`): two-input but uses Wirtinger formulas + `.conj()` (real-only no-op).
- Phase 6 (`hadamard_transform`): self-inverse, single tensor in, single `_hadamard_transform_torch(grad_out, ...)` call.
- Phase 7 (`butterfly_multiply`): two-input via `torch.autograd.grad(...)` on the oracle. This pattern is **NEW** in the codebase — no prior Triton op uses `torch.autograd.grad` in the backward callback.

Reference template (from 07-CONTEXT.md `<specifics>` lines 188-199, transcribe verbatim):

```python
def _backward(ctx, grad_out):
    twiddle, input_ = ctx.saved_tensors
    # Detach + re-enable grad so torch.autograd.grad traces both inputs through
    # the oracle. The saved tensors come from the forward graph; cloning to
    # leaf tensors with requires_grad=True is the canonical pattern.
    twiddle_d = twiddle.detach().requires_grad_(True)
    input_d = input_.detach().requires_grad_(True)
    with torch.enable_grad():
        out = _butterfly_multiply_torch(
            twiddle_d, input_d, ctx.increasing_stride, ctx.output_size
        )
    grad_twiddle, grad_input = torch.autograd.grad(
        out, [twiddle_d, input_d], grad_out, retain_graph=False
    )
    # 4 returns matching 4 forward inputs (twiddle, input, increasing_stride, output_size).
    return grad_twiddle, grad_input, None, None
```

**`register_autograd` + `register_fake` pattern** (copy from `_triton/diag_mult/op.py:193-205` with shape substitution):

```python
butterfly_multiply.register_autograd(_backward, setup_context=_setup_context)


@butterfly_multiply.register_fake
def _butterfly_multiply_fake(twiddle, input, increasing_stride=True, output_size=None):
    """Meta kernel — Phase 4 D-12 mandate (the literal 260419-p27 fix).

    The default values for increasing_stride and output_size are LOAD-BEARING
    (Phase 6 06-01-SUMMARY.md lesson learned: FakeTensorMode elides
    default-valued scalar args before calling the fake impl, so missing
    defaults raise TypeError: missing positional argument).
    """
    batch_size, nstacks, _ = input.shape
    log_n = twiddle.shape[2]
    n = 1 << log_n
    output_size_actual = n if output_size is None else output_size
    return torch.empty(
        batch_size, nstacks, output_size_actual,
        dtype=input.dtype, device=input.device,
    )
```

**num_warps helper (D-40d)** — Phase 7-specific; no analog. Define at module scope above the wrapper:

```python
def _pick_num_warps(tile_n: int) -> int:
    """D-40d: fixed num_warps schedule by tile_n. Phase 9 may revisit."""
    if tile_n <= 64:
        return 4
    if tile_n <= 1024:
        return 8
    return 16
```

---

### `tests/test_butterfly_triton.py` (test, request-response)

**Analog:** `tests/test_diag_mult.py` (119 lines, top-level, fixture-parametrized, two-input gradcheck — verbatim structural template)

**Header pattern (verbatim from `tests/test_diag_mult.py:1-24` with name substitution):**

```python
"""Cross-backend correctness + fp64 gradcheck tests for butterfly_multiply (TRI-03).

Tests are parametrized over the ``backend`` fixture from ``tests/conftest.py``
(Phase 6 D-39 widened skip-gate covers butterfly_multiply); the Triton branch
is skipped on hosts without the kernel installed. All tests call
``torch_structured._ops.butterfly_multiply`` via attribute access (D-05) so
set_backend() rebindings take effect.

The fp64 gradcheck tests are the load-bearing acceptance gates per D-47:
* ``test_butterfly_gradcheck_fp64``: validates two-input register_autograd
  plumbing via torch.autograd.grad(_torch_ref, [twiddle, input], grad_out).
* ``test_butterfly_unitary`` (Plan 07-02): the U U^* = I gate per PITFALLS §1.
"""
import itertools

import pytest
import torch

import torch_structured  # noqa: F401 — triggers extension load + _ops.py resolver
from torch_structured._torch_ref.butterfly import butterfly_multiply_torch as butterfly_ref


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="butterfly_multiply tests require CUDA"
)
```

**Eager fp32 test pattern (mirror `tests/test_diag_mult.py:27-36`):**

Note the **dense smoke / sparse comprehensive** tiering per D-43a — Phase 5 didn't tier (it was a small surface); Phase 7's parameter grid is much larger so the slow marker is a planner-required addition.

```python
@pytest.mark.parametrize("log_n", [2, 4, 8, 10])  # dense smoke tier (D-43a)
def test_butterfly_eager_fp32(backend, log_n):
    """Forward correctness vs torch_ref oracle, fp32, dense-smoke parameter set."""
    n = 1 << log_n
    nstacks, nblocks, batch_size = 1, 1, 4
    twiddle = torch.randn(nstacks, nblocks, log_n, n // 2, 2, 2, device="cuda", dtype=torch.float32)
    input_ = torch.randn(batch_size, nstacks, n, device="cuda", dtype=torch.float32)
    out = torch_structured._ops.butterfly_multiply(twiddle, input_, True, n)
    expected = butterfly_ref(twiddle, input_, True, n)
    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-6), (
        f"fp32 mismatch (backend={backend}, log_n={log_n}): max err = {(out - expected).abs().max()}"
    )
```

**Sparse comprehensive tier — `@pytest.mark.slow`** (NEW pattern vs Phase 5/6; the tiered parametrize approach is a Phase 7 introduction; tests/conftest.py already supports the marker because it's a standard pytest marker — but the planner may need to register it in conftest if a cleaner CLI is desired):

```python
@pytest.mark.slow
@pytest.mark.parametrize("log_n,nstacks,nblocks,increasing_stride,output_size_ratio", list(
    itertools.product(
        range(2, 12),       # log_n ∈ {2..11}
        [1, 2, 3],          # nstacks
        [1, 2],             # nblocks
        [True, False],      # increasing_stride
        ["n", "half", "n-1"],  # output_size variants (decoded inside the test body)
    )
))
def test_butterfly_comprehensive(backend, log_n, nstacks, nblocks, increasing_stride, output_size_ratio):
    n = 1 << log_n
    output_size = {"n": n, "half": n // 2, "n-1": n - 1}[output_size_ratio]
    # ... (same shape as eager_fp32 with parametrized axes)
```

**Two-input gradcheck pattern (D-47 acceptance gate)** — diverges from `tests/test_diag_mult.py:52-62` only in the call signature (4 args vs 4 args, but two are non-tensor):

```python
def test_butterfly_gradcheck_fp64(backend):
    """fp64 gradcheck — two-input (D-47 acceptance gate for register_autograd plumbing)."""
    log_n = 3
    n = 1 << log_n
    nstacks, nblocks, batch_size = 1, 1, 2
    twiddle = torch.randn(nstacks, nblocks, log_n, n // 2, 2, 2,
                          dtype=torch.float64, device="cuda", requires_grad=True)
    input_ = torch.randn(batch_size, nstacks, n,
                         dtype=torch.float64, device="cuda", requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda t, x: torch_structured._ops.butterfly_multiply(t, x, True, n),
        (twiddle, input_),
        eps=1e-6,
        atol=1e-5,
    )
```

**Small-N fallback test pattern (D-42a-specific, Phase 7 unique):**

```python
def test_butterfly_smallN_fallback(backend):
    """log_n ≤ 1 routes through _torch_ref via the D-42a fallback."""
    for log_n in (0, 1):
        n = 1 << log_n
        twiddle = torch.randn(1, 1, log_n, max(n // 2, 1), 2, 2, device="cuda", dtype=torch.float32)
        input_ = torch.randn(2, 1, n, device="cuda", dtype=torch.float32)
        # Just exercising the path — the value match against torch_ref is implied
        # because the fallback IS torch_ref.
        out = torch_structured._ops.butterfly_multiply(twiddle, input_, True, n)
        expected = butterfly_ref(twiddle, input_, True, n)
        assert torch.allclose(out, expected, rtol=1e-5, atol=1e-6)
```

**Plan 07-02 additions — complex64 + unitary:**

```python
@pytest.mark.parametrize("log_n", [2, 4, 8, 10])
def test_butterfly_eager_complex64(backend, log_n):
    """Forward correctness, complex64; dtype preserved through view_as_real round-trip."""
    n = 1 << log_n
    twiddle = torch.randn(1, 1, log_n, n // 2, 2, 2, device="cuda", dtype=torch.complex64)
    input_ = torch.randn(4, 1, n, device="cuda", dtype=torch.complex64)
    out = torch_structured._ops.butterfly_multiply(twiddle, input_, True, n)
    expected = butterfly_ref(twiddle, input_, True, n)
    assert out.dtype == torch.complex64
    assert torch.allclose(out, expected, rtol=1e-4)


def test_butterfly_unitary(backend):
    """U U^* = I — the load-bearing complex-correctness detector per PITFALLS §1.

    Builds a ButterflyUnitary(complex=True) module, materializes its matrix
    representation, asserts unitarity to fp32 tolerance.
    """
    from torch_structured.butterfly import ButterflyUnitary  # legacy import surface
    n = 1 << 4
    b = ButterflyUnitary(in_size=n, out_size=n, bias=False, complex=True, increasing_stride=True)
    b = b.to("cuda")
    # The exact API to materialize U is module-specific; planner must verify.
    # Pattern: feed identity-vector batch, collect outputs as columns of U.
    eye = torch.eye(n, dtype=torch.complex64, device="cuda").unsqueeze(1)  # (n, 1, n)
    U = b(eye).squeeze(1)  # (n, n) — rows are U @ e_i, so U.t() is the matrix
    UUH = U @ U.conj().T
    assert torch.allclose(UUH, torch.eye(n, dtype=torch.complex64, device="cuda"), atol=1e-4)
```

**Sub-analog for the shift-grid loop in `tests/test_diag_mult.py:82-118`** — Phase 7 does NOT have shift-grid axes; the equivalent surface is the `output_size` grid + `increasing_stride` grid, both folded into `test_butterfly_comprehensive` above. No additional separate-function pattern needed.

---

### `.planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json` (data artifact)

**Analog:** None in-repo.

**Pattern derivation:** Schema is locked in 07-CONTEXT.md D-43b. Phase 7 is the first phase to emit a perf-baseline JSON; Phase 9's TEST-04 parity gate reads this verbatim.

**Schema (locked in CONTEXT.md, transcribe verbatim):**

```json
{
  "rows": [
    {
      "kernel": "butterfly_multiply",
      "dtype": "fp32",
      "log_n": 8,
      "nstacks": 1,
      "nblocks": 1,
      "wall_ms_p50": 0.0,
      "wall_ms_p95": 0.0,
      "reference_torch_ref_p50": 0.0,
      "measured_at": "2026-MM-DD",
      "gpu": "<output of torch.cuda.get_device_name(0)>"
    }
  ]
}
```

**Generation pattern:** A pytest harness with a custom marker (e.g., `@pytest.mark.baseline`) or a standalone Python script under `tests/` that writes the JSON via `json.dump`. The CONTEXT.md `<specifics>` block (line 26) suggests `pytest tests/test_butterfly_triton.py -m baseline --baseline-out 07-BASELINE.json` but the exact CLI is **planner discretion** (07-CONTEXT.md `<decisions>` D-43b). The harness MUST cover `log_n ∈ {8,9,10,11} × {fp32, complex64}` with `batch_size=64, nstacks=1, nblocks=1, increasing_stride=True, output_size=n`.

**Measurement pattern (recommended):**

```python
def measure_p50_p95(fn, *args, warmup=10, n_iter=100):
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    # Measure with cuda events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(n_iter):
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2], times[int(len(times) * 0.95)]
```

## Shared Patterns

### Wrapper-boundary asserts (applies to `_triton/butterfly/op.py`)

**Source:** `_triton/diag_mult/op.py:113-125` + `_triton/hadamard_transform/op.py:113-125`

**Apply to:** Plan 07-01 wrapper preamble (verbatim transcribe + add the input-dim + twiddle-shape asserts unique to butterfly).

```python
assert input.dim() == 3, f"input must be (batch, nstacks, input_size), got dim={input.dim()}"
assert twiddle.dtype == input.dtype, f"twiddle.dtype ({twiddle.dtype}) must equal input.dtype ({input.dtype})"
assert input.is_contiguous(), "input must be contiguous (Pitfall 3)"
assert twiddle.is_contiguous(), "twiddle must be contiguous (Pitfall 3)"
assert twiddle.shape == (nstacks, nblocks, log_n, n // 2, 2, 2)
assert output_size_actual <= n
```

Note: `n == 1 << log_n` is **implicit** from `log_n = twiddle.shape[2]` + the twiddle shape assert — well-formed twiddle implies n is a power of 2 by construction (this matches the `_torch_ref/butterfly.py:17` convention; do NOT add a separate `assert n == 1 << log_n` line — would be redundant). However, Plan 07-01 SHOULD add the fp32-only gate (relaxed to `{fp32, complex64}` in Plan 07-02):

```python
# Plan 07-01 only — Plan 07-02 removes:
assert input.dtype == torch.float32, f"Plan 07-01: fp32-only (complex64 lands in 07-02); got {input.dtype}"
```

### IS_COMPLEX kernel scaffolding (applies to `_triton/butterfly/op.py`)

**Source:** Plan 04-COMPLEX-LAYOUT.md lines 58-76 (canonical 4-FMA template) + `_triton/diag_mult/op.py:29-95` (full kernel example with IS_COMPLEX branching).

**Apply to:** Plan 07-01 includes the `IS_COMPLEX: tl.constexpr` flag in the kernel signature with `tl.static_assert(not IS_COMPLEX, ...)` at function entry. Plan 07-02 removes ONLY the static_assert and implements the 4-FMA branch — **zero kernel-signature refactor between plans** (D-41a load-bearing).

Verbatim 4-FMA from `04-COMPLEX-LAYOUT.md:58-76`:

```python
if IS_COMPLEX:
    a_re, a_im = tl.load(in_ptr + off_re), tl.load(in_ptr + off_im)
    c_re, c_im = tl.load(twiddle_ptr + t_re), tl.load(twiddle_ptr + t_im)
    out_re = a_re * c_re - a_im * c_im
    out_im = a_re * c_im + a_im * c_re
    tl.store(out_ptr + off_re, out_re)
    tl.store(out_ptr + off_im, out_im)
else:
    a = tl.load(in_ptr + off)
    c = tl.load(twiddle_ptr + t)
    tl.store(out_ptr + off, a * c)
```

For the butterfly kernel body, the 4-FMA is applied **per-stage inside the static-range loop** on the in-register tile — not per-element on a flat pointer like diag_mult. The Phase 7 planner adapts: replace `a * c` with the butterfly-stage `tl.where(lower_mask, x + partner, x - partner)` formulation (real branch) and the 4-FMA twiddle-multiplied version for complex.

### View_as_real wrapper-boundary pattern (applies to `_triton/butterfly/op.py`)

**Source:** `04-COMPLEX-LAYOUT.md:33-50` (D-02 canonical template) + `_triton/diag_mult/op.py:132-139, 160-162` (verbatim implementation).

**Apply to:** Plan 07-01 includes the `view_as_real` machinery in the wrapper but the gate is inert because the fp32-only assert rejects complex inputs. Plan 07-02 removes the fp32 assert and the view_as_real path lights up.

```python
if is_complex:
    assert input.is_contiguous(), "complex input must be contiguous before view_as_real (Pitfall 3)"
    assert twiddle.is_contiguous(), "complex twiddle must be contiguous before view_as_real (Pitfall 3)"
    output_work = torch.view_as_real(output)
    twiddle_work = torch.view_as_real(twiddle)
else:
    output_work = output
    twiddle_work = twiddle
# ... kernel launches operate on *_work ...
if is_complex:
    output = torch.view_as_complex(output_work.contiguous())
```

### `register_autograd` + `register_fake` five-component skeleton (applies to `_triton/butterfly/op.py`)

**Source:** `_triton/diag_mult/op.py:165-205` (the Phase 5 reference skeleton; Phase 4 demonstrator deleted at D-27).

**Apply to:** Plan 07-01 ships the full skeleton:
1. `@triton.jit` kernel (`_butterfly_kernel`)
2. `@triton_op("torch_structured::butterfly_multiply", mutates_args={})` wrapper
3. `_setup_context(ctx, inputs, output)` — saves both `twiddle` and `input` tensors + 2 non-tensor flags
4. `_backward(ctx, grad_out)` — **two-input via `torch.autograd.grad`** (D-47, Phase 7 unique)
5. `@butterfly_multiply.register_fake` meta — returns `torch.empty(batch_size, nstacks, output_size, dtype=input.dtype, device=input.device)` with **load-bearing defaults** for `increasing_stride=True` and `output_size=None` (Phase 6 06-01-SUMMARY.md lesson learned)

### Attribute-access dispatch in tests (applies to `tests/test_butterfly_triton.py`)

**Source:** `tests/test_diag_mult.py:18, 32, 44, 57, 73, 91, 103, 108` (every call site uses `torch_structured._ops.diag_mult(...)`).

**Apply to:** Every Phase 7 test invocation MUST use `torch_structured._ops.butterfly_multiply(...)` (NOT `from torch_structured._ops import butterfly_multiply` — that would early-bind to whichever backend was active at import time and miss `set_backend()` rebinds). D-05 attribute access is the load-bearing pattern that lets the `backend` fixture parametrize over `['torch', 'triton']`.

### Skip-gate pattern (already in place — no edits needed)

**Source:** `tests/test_diag_mult.py:22-24` (module-level pytestmark on CUDA availability).

**Apply to:** Plan 07-01 transcribes the module-level `pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), ...)`. The conftest `backend` fixture handles the Triton-specific skip-gate via `_has_any_triton_kernel()` (Phase 6 D-39 already widened).

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `.planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json` | data artifact | one-shot write | First phase to emit a perf-baseline JSON. Schema is locked in 07-CONTEXT.md D-43b; planner picks the harness invocation pattern (pytest marker vs standalone script). |

## Substantive Divergences Summary (planner cross-reference)

These are the 7 places Plan 07-01 / 07-02 cannot transcribe verbatim from prior templates and must author new logic. Cross-reference 07-CONTEXT.md `<domain>` line 9 and `<decisions>` D-40..D-48 for the full rationale.

1. **Kernel body** — multi-launch 3-stage register-resident tile (D-40, D-40b). No in-repo analog at the body level; only the kernel **structural skeleton** transcribes. Math reference: `csrc/cuda/butterfly_cuda.cu:288` `butterfly_multiply_untied_forward_max5_fast_cuda_kernel` (5-stage variant; Phase 7 caps at 3).
2. **Two-input backward via `torch.autograd.grad`** (D-47). NEW pattern in the codebase. Phase 5 used Wirtinger formulas + `.conj()`; Phase 6 used self-inverse single-tensor. Phase 7 delegates the entire two-input gradient to the oracle via `torch.autograd.grad(out, [twiddle, input], grad_out)`.
3. **Wrapper-side `F.pad` + small-N fallback** (D-42, D-42a). Phase 7 unique. Phase 5 had pointer-math broadcast detection; Phase 6 had wrapper-side `/ (2 ** (log_n / 2))` post-processing; Phase 7's pad+trim mirrors `_torch_ref/butterfly.py:18, 33` verbatim.
4. **Python-side nblocks loop + `cur_increasing_stride` toggle** (D-40a). Phase 7 unique. No prior op had a multi-stage Python-driven launch loop (diag_mult is one launch; hadamard is one launch).
5. **2-D grid `(n_row_tiles, batch_size * nstacks)` per stage-group launch** (D-40c). Phase 5 used `(n_batch, cdiv(N, BLOCK_SIZE))`; Phase 6 used `(n_batch,)` 1-D grid. Phase 7's 2-D-with-row-tiles shape is new.
6. **IS_COMPLEX pre-wiring with `tl.static_assert(not IS_COMPLEX, ...)` gate** (D-41a). NEW pattern: 07-01 pre-wires the flag with a static_assert gate; 07-02 removes ONLY the gate. Eliminates kernel-signature refactor between plans. Phase 5 lit up IS_COMPLEX immediately; Phase 6 was real-only and omitted the flag entirely.
7. **Test file location** — `tests/test_butterfly_triton.py` (top-level), NOT `tests/structured/test_butterfly_triton.py`. Phase 6 lived under `tests/structured/` because an existing `tests/structured/test_hadamard.py` drove the layout. There is no existing `tests/structured/test_butterfly_triton.py` to mirror; the closest analog is `tests/test_diag_mult.py` at the top level.

## Metadata

**Analog search scope:** `torch_structured/_triton/`, `torch_structured/_torch_ref/`, `torch_structured/_cuda_legacy/`, `tests/`, `tests/structured/`, `.planning/phases/05-diag-mult-triton-port/`, `.planning/phases/06-hadamard-triton-port/`, `.planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/`.
**Files scanned:** 9 (3 op.py templates + 2 test templates + 1 oracle + 1 complex layout + 2 plan files).
**Pattern extraction date:** 2026-05-27
