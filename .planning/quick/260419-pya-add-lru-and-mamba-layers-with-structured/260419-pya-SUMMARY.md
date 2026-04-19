---
phase: 260419-pya
plan: 01
subsystem: experiments/recurrent_poc
tags: [recurrent, lru, mamba, ssm, structured-linear, benchmark]
requires:
  - experiments/recurrent_poc/layers.py (make_linear factory)
  - experiments/recurrent_poc/gru.py (StackedGateGRUCell, unroll_cell baseline)
  - experiments/recurrent_poc/lin_rnn.py (associative_scan probe pattern)
provides:
  - experiments.recurrent_poc.lru.LRU (complex-diagonal linear recurrent unit)
  - experiments.recurrent_poc.mamba.Mamba (simplified S6 SSM, naive loop)
  - experiments/recurrent_poc/bench_recurrent.py (CLI bench)
affects:
  - None — strictly additive; no edits to existing files.
tech_stack_added:
  - torch.associative_scan on complex64 (verified working; naive fallback kept)
  - torch.nn.Parameter (complex state via torch.complex of real params)
  - F.softplus / softplus-inverse init for Δ projection (Mamba)
patterns:
  - Drop-in nn.GRU forward contract (batch_first, bidirectional, num_layers, h_0)
  - Structured B/C via make_linear; kind / kind_B / kind_C kwargs
  - Flat ModuleList of length num_layers * num_directions indexed as l*D+d
key_files:
  created:
    - experiments/recurrent_poc/lru.py
    - experiments/recurrent_poc/mamba.py
    - experiments/recurrent_poc/bench_recurrent.py
  modified: []
decisions:
  - Use the SAME associative_scan probe pattern as lin_rnn.py (hasattr + private
    fallback) plus a tri-state _SCAN_COMPLEX_OK cache so a single first-call
    failure on complex tensors falls back permanently to the naive loop without
    re-raising on every forward.
  - LRU B_re/B_im/C_re/C_im are four INDEPENDENT real make_linear modules (not
    one complex module) — make_linear is real-valued and this keeps each
    structured variant (butterfly/monarch/circulant) usable unchanged.
  - Mamba's in_proj is always dense nn.Linear (not make_linear) — per plan
    context, structuring this is orthogonal to the A/B/C comparison and
    complicates shapes.
  - Mamba's h_0 is accepted but IGNORED with a single warning — real Mamba
    state is (B, H, d_state) per layer, and a (L*D, B, H) h_0 is
    underdetermined; bending it back to d_state would be arbitrary.
  - h_n for both modules is a lossy real reduction of the internal state
    (real(h_T) for LRU, h.mean(-1) for Mamba) to match nn.GRU's (L*D, B, H)
    return-shape contract. Documented in each docstring.
metrics:
  duration_min: ~15
  completed: 2026-04-19
  files_created: 3
  total_loc: 647
  budget: 650
---

# Phase 260419-pya Plan 01: Add LRU and Mamba Layers with Structured B/C — Summary

**One-liner:** Added LRU (Orvieto-2023 complex-diagonal linear recurrence with
parallel associative scan) and Mamba (simplified S6 selective SSM with naive
time loop) as drop-in nn.GRU peers, plus a CLI benchmark comparing them
against cuDNN nn.GRU and StackedGateGRUCell — all with structured B/C via
the existing `make_linear` factory.

## What Changed

Three new files in `experiments/recurrent_poc/`, zero edits to existing files:

- **`lru.py` (242 LOC)** — Orvieto-2023 LRU. `h_t = a ⊙ h_{t-1} + γ ⊙ B(x_t)`
  with `|a| = exp(-exp(ν))`, `arg(a) = exp(θ)` (stable reparam), `γ =
  sqrt(1-|a|²)`. Parallel `torch.associative_scan` on complex64 state with
  combine `(a1,b1)⊕(a2,b2) = (a1·a2, a2·b1 + b2)`; naive Python-loop
  fallback cached via tri-state flag. B_re/B_im/C_re/C_im routed through
  `make_linear`. Full nn.GRU surface: `num_layers`, `bidirectional`,
  `batch_first`, `h_0`, `dropout`. `h_n` is `real(h_T)` (documented lossy).

- **`mamba.py` (215 LOC)** — Simplified S6. Per-step selective `B_t, C_t, Δ_t`
  from hidden `x`; ZOH-ish discretization `A_bar = exp(Δ ⊗ A)`, `B_bar =
  Δ ⊗ B_t`; naive Python time loop with state `(B, H, d_state)`. `A` is
  the standard `-exp(A_log)` with `A_log = log(1..N)` init; Δ uses low-rank
  `dt_rank = max(H//16, 1)` with softplus-inverse bias so softplus starts
  at ~1e-3. Same nn.GRU surface; `h_0` accepted-but-ignored with one-time
  warning; `h_n = h.mean(-1)` (lossy). `expand` kwarg accepted for API
  parity but unused.

- **`bench_recurrent.py` (190 LOC)** — Argparse CLI with
  `--models/--kinds/--hiddens/--seq-lens/--num-layers/--bidirectional/--quick`.
  Reuses `time_fn` from `bench_gru.py` (cuda.synchronize + median). SKIPPED
  rows for cuDNN-non-dense, GRU-cell-multi-layer-or-bidir, and
  shape-incompatible structured kinds (e.g. circulant non-square). Fwd-only
  and fwd+bwd columns.

## Scan Correctness Probe

Before committing LRU, verified the combine-op direction against a naive
Python loop on a random `(B=2, T=16, H=8)` complex problem, with and without
`h_0`:

```
parallel max diff:       2.67e-07
last state diff:         1.49e-07
no-h0 parallel max diff: 2.67e-07
```

Well under 1e-5. Direction `(a2*a1, a2*b1 + b2)` is correct.

## Verification

All three plan sanity checks pass on CUDA (RTX 4090, sm_89):

| Check | Expected | Actual |
| --- | --- | --- |
| LRU butterfly (4, 200, 64) → out, h_n | `(4,200,256) (4,4,128) float32` + backward | **PASS** |
| Mamba dense (4, 200, 64) → out, h_n | `(4,200,128) (2,4,128)` + backward | **PASS** |
| bench_recurrent --quick | 1 row per {cudnn, gru, lru, mamba} | **PASS** |

Quick-mode bench output on the machine:

```
model | kind  | H  | T  | layers | bi | params | fwd_ms | fwd_bwd_ms
cudnn | dense | 64 | 32 | 1      | no | 24960  | 0.253  | 0.598
gru   | dense | 64 | 32 | 1      | no | 24960  | 2.196  | 5.787
lru   | dense | 64 | 32 | 1      | no | 16704  | 1.365  | 2.592
mamba | dense | 64 | 32 | 1      | no | 7872   | 1.034  | 4.383
```

Additional manual verification of SKIPPED rendering (structured kinds +
bidirectional + num-layers=2 + circulant-non-square) — all SKIPPED rows
show the correct reason string and leave timing columns blank.

## LOC Budget

| File | LOC | Guidance | Status |
| --- | ---: | ---: | --- |
| `lru.py` | 242 | ~220 | over guidance, kept under total budget |
| `mamba.py` | 215 | ~270 | under |
| `bench_recurrent.py` | 190 | ~220 | under |
| **Total** | **647** | **< 650** | **PASS** |

An initial LRU draft at 275 LOC was trimmed to 242 by collapsing redundant
config-attr-storage lines and helper docstrings. The strict constraint is
the 650-total budget; individual-file guidance is advisory and we stayed
within it for Mamba and bench.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking] Under-budget trim of LRU after first verification**
- **Found during:** Task 3 (after bench_recurrent was written, total hit 680)
- **Issue:** Total LOC was 30 over the 650 budget (lru=275, mamba=215,
  bench=190).
- **Fix:** Trimmed lru.py docstrings and paired attribute assignments;
  smoke test re-run and still passes.
- **Commit:** f5c0a03

No other deviations. All three tasks executed as written; all three sanity
checks passed on first attempt after trim.

## Key Decisions

- **Fallback cache in LRU scan** — a module-level tri-state
  `_SCAN_COMPLEX_OK` means a single runtime failure on complex64 silently
  flips the layer to the naive loop for the rest of the process, rather
  than spamming warnings every forward. In this PyTorch build (2.11.0
  +cu130) the parallel path works, so the fallback never triggers; it's
  purely safety net.
- **`h_n` is lossy by design** — both modules reduce internal state to
  `(L*D, B, H)` for nn.GRU-shape compat. LRU drops `imag(h_T)`; Mamba
  averages over `d_state`. Each docstring spells this out so callers who
  need exact state carry-over know to maintain the complex / `(H, N)`
  state themselves.
- **Mamba in_proj is always dense** — plan context was explicit that
  structuring the input projection is orthogonal to the per-step A/B/C
  comparison.
- **Four real linears for B/C in LRU (not one complex)** — keeps
  `make_linear` (real-valued by construction) usable for every structured
  kind without a complex detour.

## Commits

| Commit | Message |
| --- | --- |
| fd0c083 | `feat(260419-pya-01): add LRU layer with Orvieto-2023 complex diagonal recurrence` |
| 13ad17f | `feat(260419-pya-01): add simplified Mamba S6 layer with selective per-step state` |
| f5c0a03 | `chore(260419-pya-01): tighten LRU docstrings/comments to fit LOC budget` |
| 0725d2f | `feat(260419-pya-01): add bench_recurrent.py comparing cuDNN/gru/lru/mamba` |

Branch: `feat/recurrent-poc-extensions` (unchanged).

## Follow-ups (out of scope for this task)

- Wire LRU/Mamba into the repo's speech-enhancement stack as drop-in GRU
  replacements.
- Add a full-sweep reference run of `bench_recurrent.py` at
  `H∈{256,512}, T∈{500,1000}, kinds={dense,butterfly}` and commit the
  table to a notes directory (not committed in this task — plan says no
  bench output commits).
- Investigate whether Mamba's naive Python loop can be replaced with
  `torch.associative_scan` over the `(B, T, H, N)` state (the recurrence
  `h_t = A_bar_t * h_{t-1} + B_bar_t * x_t` is associative in the same
  way as LRU's).
- Consider adding a fused-linear variant of Mamba's per-step B_t, C_t,
  Δ_t projections to cut 3 GEMMs per step to 1.

## Self-Check: PASSED

- [x] `experiments/recurrent_poc/lru.py` exists, contains `class LRU`
- [x] `experiments/recurrent_poc/mamba.py` exists, contains `class Mamba`
- [x] `experiments/recurrent_poc/bench_recurrent.py` exists, contains `def main`
- [x] Commit `fd0c083` (LRU feat) present in `git log`
- [x] Commit `13ad17f` (Mamba feat) present in `git log`
- [x] Commit `f5c0a03` (LRU trim chore) present in `git log`
- [x] Commit `0725d2f` (bench feat) present in `git log`
- [x] All three sanity checks pass on CUDA
- [x] Total LOC 647 < 650 budget
- [x] No modifications to `layers.py`, `gru.py`, `lin_rnn.py`, `bench_gru.py`, `bench_lin_rnn.py`
- [x] Branch still `feat/recurrent-poc-extensions`
