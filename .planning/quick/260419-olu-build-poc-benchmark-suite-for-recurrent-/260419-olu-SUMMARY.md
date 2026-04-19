---
phase: quick-260419-olu
plan: 01
subsystem: experiments/recurrent_poc
tags: [poc, benchmark, recurrent, butterfly, monarch, circulant]
dependency-graph:
  requires:
    - torch_structured.Butterfly
    - torch_structured.monarch.blockdiag_linear.BlockdiagLinear
  provides:
    - experiments.recurrent_poc.layers.make_linear
    - experiments.recurrent_poc.gru.StackedGateGRUCell
    - experiments.recurrent_poc.lin_rnn.LinearDiagRNN
  affects: []
tech-stack:
  added: []
  patterns:
    - "Factory dispatch by string kind with explicit ValueError on unknown"
    - "Thin (x)->y wrappers normalizing structured-linear forward signatures"
    - "Median-of-N timing with cuda.synchronize"
key-files:
  created:
    - experiments/__init__.py
    - experiments/recurrent_poc/__init__.py
    - experiments/recurrent_poc/layers.py
    - experiments/recurrent_poc/gru.py
    - experiments/recurrent_poc/bench_gru.py
    - experiments/recurrent_poc/lin_rnn.py
    - experiments/recurrent_poc/bench_lin_rnn.py
  modified: []
decisions:
  - "Implemented circulant from scratch via torch.fft.rfft instead of wrapping torch_structured.butterfly.special.circulant: avoids per-forward sub-module rebuild, ~10 LOC, all-float32 path."
  - "Used combine_mode='generic' for torch.associative_scan so the scan path runs on CPU as well as CUDA (pointwise mode is CUDA-only)."
  - "Probed associative_scan via two paths: torch.associative_scan attribute, then torch._higher_order_ops.associative_scan; fallback warns once and uses naive."
  - "_MonarchLinear defaults nblocks=min(4, in, out) to avoid crashes when H<4; honors caller-supplied nblocks if passed."
metrics:
  duration: ~25 minutes
  completed: 2026-04-19
  files: 7
  tasks: 3
---

# Quick Task 260419-olu: Recurrent POC Benchmark Suite Summary

**One-liner:** Self-contained POC benchmark suite measuring GRU + linear-RNN training-step wall-clock when dense linears are swapped for `torch_structured` primitives (butterfly, monarch blockdiag, circulant), benchmarked against cuDNN `nn.GRU` and a naive scan.

## Files Created

| Path                                              | LOC | Purpose                                         |
| ------------------------------------------------- | --: | ----------------------------------------------- |
| `experiments/__init__.py`                         |   1 | Namespace stub                                  |
| `experiments/recurrent_poc/__init__.py`           |   1 | Namespace stub                                  |
| `experiments/recurrent_poc/layers.py`             | 140 | `make_linear` factory + dense/butterfly/monarch/circulant wrappers |
| `experiments/recurrent_poc/gru.py`                |  70 | `StackedGateGRUCell` (matches `nn.GRUCell` math) |
| `experiments/recurrent_poc/bench_gru.py`          | 156 | CLI sweep over (kind, H, T) with cuDNN reference |
| `experiments/recurrent_poc/lin_rnn.py`            | 122 | `LinearDiagRNN` with naive + associative-scan paths |
| `experiments/recurrent_poc/bench_lin_rnn.py`     | 118 | CLI comparing naive vs scan timings              |

All files have module docstrings; no README; no JSON/file output from any script.

## Sweep Grid Used in `--quick` Smoke Tests

- `bench_gru.py --quick`: H=64, T=32, batch=32, kinds={dense, butterfly}, warmup=1, iters=1
- `bench_lin_rnn.py --quick`: H=64, T=32, batch=32, kind=dense, warmup=1, iters=1

## Per-Kind Timing Excerpt (RTX 4090, batch=32, T=32, H=64)

```
kind      | H  | T  | params | fwd_ms | fwd_bwd_ms
----------+----+----+--------+--------+-----------
dense     | 64 | 32 | 24960  | 2.16   | 6.13
butterfly | 64 | 32 |  4992  | 3.33   | 8.28
```

Wider sweep at batch=8, T=32, H ∈ {64, 100, 128, 256}, iters=2 (excerpt):

```
kind      | H   | params | fwd_ms | fwd_bwd_ms
----------+-----+--------+--------+-----------
dense     | 256 | 394752 | 2.20   | 5.93
butterfly | 256 |  26112 | 3.07   | 7.88
monarch   | 256 |  99840 | 5.08   | 8.58
cudnn     | 256 | 394752 | 0.27   | 0.61
```

`bench_lin_rnn.py --quick`:

```
kind  | H  | T  | naive_ms | scan_ms | speedup
------+----+----+----------+---------+--------
dense | 64 | 32 | 0.74     | 1.77    | 0.42x
```

(Scan is slower than naive at this scale: kernel-launch overhead of the
`generic` associative_scan dominates the savings from parallel reduction.
The crossover is expected at larger T; `--quick` is intentionally tiny.)

## Shape-Incompatible Combos Observed

The GRU's hidden gate linear is `H -> 3*H`, which is fundamentally non-square.
**Circulant** therefore SKIPs *every* row in any `bench_gru.py` sweep — this is
a property of the GRU formulation, not a bug. SKIP message clearly states the
mismatch:

```
SKIPPED — circulant requires square power-of-2 sizes; got in=64, out=192
```

If a future experiment wants to bench circulant in a recurrent setting, the
right fix is to use a *linear* RNN with square `B` and `C` (which is exactly
what `LinearDiagRNN` provides — it does not stack three gates).

Other observations:

- `Butterfly` and `BlockdiagLinear` both work for non-power-of-2 `H` (they
  pad internally); no SKIPs for those kinds.
- `cuDNN nn.GRU` is ~20-40x faster than the unrolled `StackedGateGRUCell`
  even with `dense` linears, as expected — the `nn.GRU` path fuses the cell
  and the time-step loop into a single CUDA kernel.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 — bug] `torch.associative_scan` requires `combine_mode='generic'` outside CUDA**
- **Found during:** Task 3 verification.
- **Issue:** Default `combine_mode='pointwise'` raises `ValueError: For combine_mode='pointwise', all input tensors need to be on CUDA or XPU` when `forward_scan` runs on CPU. The plan didn't mention this constraint.
- **Fix:** Pass `combine_mode='generic'` explicitly. Generic mode is slightly slower but works on both CPU and CUDA, and was verified to match `forward_naive` to ~1e-7 absolute error on both code paths (h0=None and h0!=None).
- **Files modified:** `experiments/recurrent_poc/lin_rnn.py`
- **Commit:** included in `0e72282` (Task 3 — fixed before commit, no separate fix commit).

No other deviations: the plan executed cleanly.

## Authentication Gates

None.

## Sanity Checks Run

All four sanity commands from `must_haves.truths` pass:

1. `make_linear('butterfly', 64, 64)` returns a printable nn.Module — OK
2. `StackedGateGRUCell(16, 32, kind='dense')(x, h)` returns shape `(2, 32)` — OK
3. `bench_gru.py --quick` prints a table and exits 0 in ~5s on RTX 4090 — OK
4. `bench_lin_rnn.py --quick` prints naive vs scan timings and exits 0 — OK

Plus a wider sweep (`bench_gru.py --hiddens 64,100,128,256 --kinds dense,butterfly,monarch,circulant,cudnn`) completed all 20 combos without uncaught exceptions; circulant's 4 expected SKIPs all printed cleanly.

## Self-Check: PASSED

Verified on disk:
- `experiments/__init__.py` — FOUND
- `experiments/recurrent_poc/__init__.py` — FOUND
- `experiments/recurrent_poc/layers.py` — FOUND (140 lines)
- `experiments/recurrent_poc/gru.py` — FOUND (70 lines)
- `experiments/recurrent_poc/bench_gru.py` — FOUND (156 lines)
- `experiments/recurrent_poc/lin_rnn.py` — FOUND (122 lines)
- `experiments/recurrent_poc/bench_lin_rnn.py` — FOUND (118 lines)

Verified in git log:
- Commit `eaeb6ad` — FOUND (Task 1: factory + GRU cell)
- Commit `8c8457f` — FOUND (Task 2: bench_gru)
- Commit `0e72282` — FOUND (Task 3: lin_rnn + bench)
