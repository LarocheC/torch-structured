---
phase: quick-260419-olu
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - experiments/__init__.py
  - experiments/recurrent_poc/__init__.py
  - experiments/recurrent_poc/layers.py
  - experiments/recurrent_poc/gru.py
  - experiments/recurrent_poc/lin_rnn.py
  - experiments/recurrent_poc/bench_gru.py
  - experiments/recurrent_poc/bench_lin_rnn.py
autonomous: true
requirements:
  - QUICK-260419-olu-01  # make_linear factory
  - QUICK-260419-olu-02  # StackedGateGRUCell
  - QUICK-260419-olu-03  # bench_gru sweep
  - QUICK-260419-olu-04  # LinearDiagRNN + scan bench

must_haves:
  truths:
    - "User can call make_linear(kind, in, out) for kind in {dense, butterfly, monarch, circulant} and get back a working nn.Module with a plain (x) -> y forward signature"
    - "User can instantiate StackedGateGRUCell(input_size, hidden_size, kind=...) and call it as (x_t, h_{t-1}) -> h_t with correct output shape"
    - "User can run `.venv/bin/python experiments/recurrent_poc/bench_gru.py --quick` and it prints a table without crashing"
    - "User can run `.venv/bin/python experiments/recurrent_poc/bench_lin_rnn.py --quick` and it prints naive-vs-parallel scan timings"
    - "Unsupported kinds raise ValueError listing supported kinds; shape-incompatible combos in the sweep print 'skipped: <reason>' instead of crashing"
  artifacts:
    - path: "experiments/recurrent_poc/layers.py"
      provides: "make_linear factory + thin wrappers for butterfly/monarch/circulant"
      min_lines: 60
    - path: "experiments/recurrent_poc/gru.py"
      provides: "StackedGateGRUCell with fused input and hidden gate linears"
      min_lines: 40
    - path: "experiments/recurrent_poc/lin_rnn.py"
      provides: "LinearDiagRNN with naive and (optional) parallel scan"
      min_lines: 50
    - path: "experiments/recurrent_poc/bench_gru.py"
      provides: "CLI bench sweeping hidden size x seq len x kind with cuDNN GRU reference"
      min_lines: 80
    - path: "experiments/recurrent_poc/bench_lin_rnn.py"
      provides: "CLI bench comparing naive vs parallel scan"
      min_lines: 40
  key_links:
    - from: "experiments/recurrent_poc/gru.py"
      to: "experiments/recurrent_poc/layers.py"
      via: "linear_factory callable passed into StackedGateGRUCell.__init__"
      pattern: "linear_factory\\(.*3 ?\\*"
    - from: "experiments/recurrent_poc/bench_gru.py"
      to: "experiments/recurrent_poc/gru.py"
      via: "import StackedGateGRUCell and build cells per kind"
      pattern: "StackedGateGRUCell"
    - from: "experiments/recurrent_poc/layers.py"
      to: "torch_structured.Butterfly / torch_structured.monarch.blockdiag_linear.BlockdiagLinear / torch_structured.butterfly.special.circulant"
      via: "direct import + wrap"
      pattern: "from torch_structured"
---

<objective>
Build a small, self-contained proof-of-concept benchmark suite under
`experiments/recurrent_poc/` that lets the user measure training-step wall-clock
for recurrent models whose dense linears are swapped for torch_structured
primitives (butterfly, monarch blockdiag, circulant). The suite covers both a
GRU cell unrolled in Python and a diagonal-state linear RNN, benchmarked
against cuDNN `nn.GRU` and a naive scan respectively.

Purpose: give the user a reproducible "which structured layer is actually
faster at what (H, T)?" table on their RTX 4090 before investing in a real
training run. This is scaffold code, not a library feature — keep it small,
docstring-only, no README.

Output: five Python source files in `experiments/recurrent_poc/` plus
package-init stubs, all runnable end-to-end on the existing `.venv` with CUDA.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@CLAUDE.md
@torch_structured/__init__.py
@torch_structured/butterfly/__init__.py
@torch_structured/monarch/blockdiag_linear.py
@torch_structured/monarch/structured_linear.py
@torch_structured/butterfly/special.py

<interfaces>
<!-- Pre-extracted so the executor does not need to spelunk the codebase. -->

From torch_structured/__init__.py (top-level re-exports):
```python
from .butterfly import (
    Butterfly,          # drop-in for nn.Linear, pads in_size to next pow2 internally
    ButterflyBmm,
    ButterflyBase4,
    ButterflyUnitary,
    butterfly_multiply,
)
```

From torch_structured/butterfly/butterfly.py:
```python
class Butterfly(nn.Module):
    def __init__(self, in_size, out_size, bias=True, complex=False,
                 increasing_stride=True, init='randn', nblocks=1): ...
    def forward(self, input, transpose=False, conjugate=False, subtwiddle=False): ...
```
NOTE: `Butterfly.forward` accepts extra kwargs; for the GRU cell we only
pass `input` positionally, so it behaves like `nn.Linear.forward`. No
separate power-of-2 padding wrapper is required at the call site — the
class handles it internally. However, because the forward signature is
`(input, transpose=False, ...)` rather than `(input)`, do NOT chain it via
`nn.Sequential` expecting pure `(x) -> y`; wrap it in a trivial
`nn.Module` that calls `self.b(x)` if you need a clean signature.

From torch_structured/monarch/blockdiag_linear.py:
```python
class BlockdiagLinear(StructuredLinear):
    def __init__(self, *args, nblocks=4, shuffle=False, **kwargs): ...
    # StructuredLinear base __init__(in_features, out_features, bias=True, device=None, dtype=None)
    # forward(x) returns (output + bias) — clean drop-in for nn.Linear
```
This is the Monarch class to use. `nblocks=4` default; for small hidden
sizes we will use `nblocks = min(4, in_features, out_features)` so it
does not crash when H < 4. It pads internally via `preprocess/postprocess`.

From torch_structured/butterfly/special.py:
```python
def circulant(col, transposed=False, separate_diagonal=True) -> nn.Module:
    # col: Tensor of shape (n,). Returns nn.Sequential wrapping real->complex,
    # Butterfly (FFT), Diagonal (eigenvalues = FFT(col)), Butterfly (iFFT), complex->real.
```
To make circulant *learnable* in the (in, out) factory style, the wrapper
must (a) require in == out and be a power of 2, (b) hold a learnable
`nn.Parameter` for `col` of shape (n,), (c) rebuild the circulant
`nn.Module` on every forward using the current `col` (because `circulant`
bakes `col` into the module at construction time — it doesn't take col
as a forward argument). This means each forward runs an FFT over `col`;
that's fine for a POC bench, we're measuring the structured cost.
An alternative, simpler approach: implement circulant *from scratch* via
`torch.fft.fft`: `y = ifft(fft(col) * fft(x))` with a learnable `col`.
Prefer this approach — it's ~10 lines, avoids rebuilding sub-modules per
forward, and still exercises the "circulant via FFT" story that the
torch_structured README advertises.
</interfaces>

<environment>
- Python venv: `/home/clement/torch-structured/.venv/bin/python`
- CUDA extension already built and importable; known benign warning
  "Detected that PyTorch and torch_structured were compiled with different
  CUDA versions" — suppress in scripts via
  `warnings.filterwarnings("ignore", message=".*different CUDA versions.*")`.
- GPU: RTX 4090 (compute capability 8.9), `torch.cuda.is_available()` is True.
- Do NOT commit benchmark output (`*.json`, `bench_*.txt`) — scripts must
  only print to stdout, never write files.
</environment>
</context>

<tasks>

<task type="auto">
  <name>Task 1: make_linear factory + StackedGateGRUCell</name>
  <files>
    experiments/__init__.py,
    experiments/recurrent_poc/__init__.py,
    experiments/recurrent_poc/layers.py,
    experiments/recurrent_poc/gru.py
  </files>
  <action>
Create the two namespace `__init__.py` files as empty modules with a one-line
module docstring each ("POC experiments — not part of the installed package.").

Create `experiments/recurrent_poc/layers.py` implementing:

1. A thin `_ButterflyLinear(nn.Module)` wrapper that holds a
   `torch_structured.Butterfly(in_features, out_features, bias=bias)` and
   exposes `forward(x)` calling the butterfly with no extra kwargs. This
   normalizes the forward signature to `(x) -> y`.

2. A `_CirculantLinear(nn.Module)` implemented from scratch (NOT via
   `torch_structured.butterfly.special.circulant`, for the reasons in the
   interfaces block):
   - Requires `in_features == out_features` and that value be a power of 2.
     If not, raise `ValueError("circulant requires square power-of-2 sizes; got in={in}, out={out}")`.
   - Stores `self.col = nn.Parameter(torch.randn(n) / math.sqrt(n))`.
   - Optional `self.bias = nn.Parameter(torch.zeros(n))` if bias=True else None.
   - Forward: `col_f = torch.fft.rfft(self.col); x_f = torch.fft.rfft(x); y = torch.fft.irfft(col_f * x_f, n=n); return y + bias`.
   - This is O(n log n) per forward and uses the real FFT path so it
     stays in float32 throughout — matches how a real circulant layer
     would be written in production.

3. A `_MonarchLinear(nn.Module)` wrapper around
   `torch_structured.monarch.blockdiag_linear.BlockdiagLinear`:
   - `nblocks = min(4, in_features, out_features)` — prevents crashes when
     H is small. Do NOT silently change nblocks if the user passes one in
     via **kwargs; prefer their value.
   - Forward is `self.bd(x)` (BlockdiagLinear already has a clean
     (x) -> y forward including bias).

4. `make_linear(kind: str, in_features: int, out_features: int, bias: bool = True, **kwargs) -> nn.Module`:
   - `kind == "dense"` -> `torch.nn.Linear(in_features, out_features, bias=bias)`.
   - `kind == "butterfly"` -> `_ButterflyLinear(in_features, out_features, bias)`.
   - `kind == "monarch"` -> `_MonarchLinear(in_features, out_features, bias, **kwargs)`.
   - `kind == "circulant"` -> `_CirculantLinear(in_features, out_features, bias)`.
   - `kind in {"ldr", "fastfood"}` -> `raise NotImplementedError(f"{kind} not wired up in the POC factory; see torch_structured.structured.layers if you want to add it")`.
   - Any other kind -> `raise ValueError(f"unknown kind={kind!r}; supported: dense, butterfly, monarch, circulant")`.

Create `experiments/recurrent_poc/gru.py` implementing `StackedGateGRUCell(nn.Module)`:

- Constructor: `(input_size, hidden_size, kind="dense", bias=True, linear_factory=None)`.
  If `linear_factory is None`, use `functools.partial(make_linear, kind=kind, bias=bias)`.
  Passing `linear_factory` overrides `kind` (lets tests inject a mock).
- Creates `self.ih = linear_factory(in_features=input_size, out_features=3*hidden_size)`
  and `self.hh = linear_factory(in_features=hidden_size, out_features=3*hidden_size)`.
  The factory must be called with *both* in_features and out_features as kwargs so
  `functools.partial` composition is unambiguous.
- Forward: `(x_t, h_prev) -> h_t` with standard GRU math:
    gi = self.ih(x_t); gh = self.hh(h_prev)
    i_r, i_z, i_n = gi.chunk(3, dim=-1)
    h_r, h_z, h_n = gh.chunk(3, dim=-1)
    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    return (1 - z) * n + z * h_prev
  (This matches PyTorch's `nn.GRUCell` formulation — specifically the
  `r * h_n` placement, not `r * (i_n + h_n)`. Verify against the
  nn.GRUCell docstring if in doubt.)

Each file ends with a small `if __name__ == "__main__":` smoke block printing
shapes for a dense instance, so a bare `python layers.py` / `python gru.py`
invocation also serves as a self-test.
  </action>
  <verify>
    <automated>
      .venv/bin/python -c "
      import warnings; warnings.filterwarnings('ignore');
      from experiments.recurrent_poc.layers import make_linear;
      import torch;
      for kind in ['dense','butterfly','monarch','circulant']:
          m = make_linear(kind, 64, 64);
          y = m(torch.randn(2, 64));
          assert y.shape == (2, 64), (kind, y.shape);
          print(kind, 'ok', tuple(y.shape));
      try:
          make_linear('circulant', 64, 128);
          assert False, 'should have raised'
      except ValueError as e:
          print('circulant-mismatch raised:', str(e)[:60])
      try:
          make_linear('bogus', 4, 4)
          assert False
      except ValueError as e:
          print('unknown kind raised:', str(e)[:60])
      from experiments.recurrent_poc.gru import StackedGateGRUCell;
      c = StackedGateGRUCell(16, 32, kind='dense');
      h = c(torch.randn(2,16), torch.zeros(2,32));
      assert h.shape == (2,32); print('gru dense ok', tuple(h.shape))
      c2 = StackedGateGRUCell(16, 32, kind='butterfly');
      h2 = c2(torch.randn(2,16), torch.zeros(2,32));
      assert h2.shape == (2,32); print('gru butterfly ok', tuple(h2.shape))
      "
    </automated>
  </verify>
  <done>
    All four kinds instantiate and produce correct (B, out) shapes; invalid
    kinds raise ValueError with a helpful message; circulant rejects
    non-square sizes; GRU cell returns (B, hidden_size) with both dense and
    butterfly factories. All checks above print "ok" lines.
  </done>
</task>

<task type="auto">
  <name>Task 2: bench_gru.py — GRU sweep with cuDNN reference</name>
  <files>experiments/recurrent_poc/bench_gru.py</files>
  <action>
Create `experiments/recurrent_poc/bench_gru.py`, a standalone CLI that
benchmarks `StackedGateGRUCell` unrolled across T time steps against
cuDNN `nn.GRU`. Target <150 LOC.

Module docstring: document expected run command and the known CUDA-version
warning suppression (see environment block in context).

Imports: `argparse, math, time, statistics, warnings, torch, torch.nn as nn`
plus the local factory and cell. Suppress the cuda_version warning at
import time:
    warnings.filterwarnings("ignore", message=".*different CUDA versions.*")

Top-level structure:

1. `time_fn(fn, *, warmup, iters, device) -> float` — runs `fn()` for
   `warmup` un-timed iterations, then `iters` timed iterations, returns
   the *median* wall-clock seconds per iteration. Uses
   `torch.cuda.synchronize()` before/after each timed iteration when
   `device.type == 'cuda'`, plain `time.perf_counter()` otherwise.

2. `run_cell(cell, x_bt, h0) -> h_T` — unrolls the cell:
       h = h0
       for t in range(x_bt.size(1)):
           h = cell(x_bt[:, t], h)
       return h

3. `make_cudnn_gru(input_size, hidden_size, device)` — returns
   `nn.GRU(input_size, hidden_size, batch_first=True).to(device)`.

4. `bench_one(kind, H, T, batch, device, warmup, iters, include_backward, compile_cell)`:
   - If `kind == "cudnn"`, build nn.GRU; forward is `gru(x)`, taking the
     first return value; backward sums the output and calls `.backward()`.
   - Else build `StackedGateGRUCell(input_size=H, hidden_size=H, kind=kind)`
     on `device`. If `compile_cell`, wrap via `torch.compile(cell, mode="reduce-overhead")`.
     For structured kinds, wrap the unroll in a try/except that catches
     `RuntimeError` and `AssertionError` and returns the dict
     `{"status": "skipped", "reason": str(exc)[:80]}` so one bad combo
     doesn't crash the sweep.
   - Build fresh input `x = torch.randn(batch, T, H, device=device)` and
     `h0 = torch.zeros(batch, H, device=device)`.
   - Define `fwd = lambda: run_cell(cell, x, h0)` (or the cudnn path).
   - Define `fwd_bwd = lambda: _wrap_backward(...)` which calls fwd then
     `out.sum().backward()` and zero_grad before the next iter. Use
     `cell.zero_grad(set_to_none=True)` between iters — do NOT share state.
   - Measure forward-only and forward+backward times. Return
     `{"status": "ok", "kind": kind, "H": H, "T": T,
       "params": sum(p.numel() for p in cell.parameters()),
       "t_fwd_ms": 1e3 * t_fwd, "t_fwd_bwd_ms": 1e3 * t_fwd_bwd}`.

5. `main()` with argparse:
   - `--device {cuda,cpu}` (default: cuda if available else cpu)
   - `--batch` (default 32)
   - `--warmup` (default 3), `--iters` (default 5)
   - `--hiddens` (default "100,300,1000" — comma-separated)
   - `--seq-lens` (default "200,500,1000")
   - `--kinds` (default "dense,butterfly,monarch,circulant,cudnn")
   - `--compile` (bool flag, default False)
   - `--no-backward` (bool flag, default False)
   - `--quick` (bool flag: override hiddens=64, seq_lens=32, kinds=dense,butterfly, warmup=1, iters=1)
   - Iterate over the sweep grid; for each combo, call `bench_one` and
     append to a results list. Print a progress line to stderr per combo.
   - At the end, print a plain-text table (pipe-separated, aligned):
       kind | H | T | params | fwd_ms | fwd_bwd_ms
     Skipped combos print "SKIPPED — <reason>" in the timing columns.

6. Guard: `if device.type == "cuda" and not torch.cuda.is_available(): raise SystemExit("CUDA requested but not available")`.

At the bottom: `if __name__ == "__main__": main()`.

Quality constraints: no matplotlib, no JSON dumping, no file writes, only
print to stdout/stderr. Circulant with non-power-of-2 H will raise inside
`make_linear` — the try/except in `bench_one` converts that to a skip.
  </action>
  <verify>
    <automated>
      .venv/bin/python experiments/recurrent_poc/bench_gru.py --quick 2>&1 | tee /tmp/bench_quick.out
      grep -E "^kind|dense|butterfly" /tmp/bench_quick.out >/dev/null
    </automated>
  </verify>
  <done>
    `--quick` run finishes in under 30 seconds on the RTX 4090 (or under
    2 minutes on CPU), prints a header row and at least one timing row
    per kind that was not skipped, and does NOT raise uncaught exceptions
    for any combo in the sweep grid. Shape-incompatible combos (e.g.
    circulant with non-power-of-2 H=100) print "SKIPPED".
  </done>
</task>

<task type="auto">
  <name>Task 3: LinearDiagRNN + bench_lin_rnn.py</name>
  <files>
    experiments/recurrent_poc/lin_rnn.py,
    experiments/recurrent_poc/bench_lin_rnn.py
  </files>
  <action>
Create `experiments/recurrent_poc/lin_rnn.py`:

`LinearDiagRNN(nn.Module)` implementing `h_t = a * h_{t-1} + B(x_t)` with a
learnable diagonal `a` of shape (H,) and output `y_t = C(h_t)`:

- Constructor: `(input_size, hidden_size, kind="dense", output_size=None, bias=True)`.
  `output_size` defaults to `hidden_size`.
- `self.a = nn.Parameter(torch.full((hidden_size,), 0.9))` — diagonal state
  init near 1 so the RNN is stable by default.
- `self.B = make_linear(kind, input_size, hidden_size, bias=bias)`
- `self.C = make_linear(kind, hidden_size, output_size, bias=bias)`

Two forward methods, both accept `x` of shape `(batch, T, input_size)` and
an optional `h0` of shape `(batch, hidden_size)`:

- `forward_naive(x, h0=None)` — pure Python for-loop over T:
    u = self.B(x)                          # (B, T, H)
    h = torch.zeros(...) if h0 is None else h0
    outs = []
    for t in range(T):
        h = self.a * h + u[:, t]
        outs.append(self.C(h))
    return torch.stack(outs, dim=1)         # (B, T, out)

- `forward_scan(x, h0=None)` — parallel scan using
  `torch.associative_scan` if available. Check once at module load:
    _HAS_SCAN = hasattr(torch, "associative_scan")
  If `_HAS_SCAN`:
    Define a lambda `combine((a1, b1), (a2, b2)) = (a2 * a1, a2 * b1 + b2)`
    broadcasting over the hidden dimension, seeded with `(self.a, u[:, t])`
    per t, scanned over dim=1. Apply `self.C` to the resulting h sequence.
  Else: print a warning *once* via `warnings.warn(..., stacklevel=2)` on
  first call and fall back to `forward_naive`.

Default `forward(x, h0=None)` calls `forward_naive` (safe, deterministic).

Create `experiments/recurrent_poc/bench_lin_rnn.py`:

CLI that reuses `time_fn` (copy-paste a minimal version here or import
from `bench_gru` — prefer import: `from .bench_gru import time_fn` via
`sys.path.insert(0, os.path.dirname(...))` since this is a script, not
an installed package. The simpler option: duplicate the ~15-line
`time_fn` to keep each file self-contained per the quality constraint).

- argparse: `--device`, `--hiddens` (default "100,300,1000"),
  `--seq-lens` (default "200,500,1000"), `--batch` (default 32),
  `--kind` (default "dense"), `--warmup` (3), `--iters` (5),
  `--quick` (H=64, T=32, 1/1, kind=dense).
- For each (H, T): instantiate `LinearDiagRNN(input_size=H, hidden_size=H, kind=kind)`,
  build `x = torch.randn(batch, T, H, device=device)`, time both
  `forward_naive` and `forward_scan`, print one row:
    kind | H | T | naive_ms | scan_ms | speedup
  If `torch.associative_scan` is unavailable, report scan_ms as "N/A"
  and speedup as "N/A" (single line, not a loud warning).
- Uncaught exceptions on one combo should not kill the sweep —
  wrap in try/except and print "SKIPPED — <reason>".

Docstrings on both files; no README.

Suppress the cuda-version warning at import time in both files, as in
Task 2.
  </action>
  <verify>
    <automated>
      .venv/bin/python -c "
      import warnings; warnings.filterwarnings('ignore');
      import torch;
      from experiments.recurrent_poc.lin_rnn import LinearDiagRNN;
      m = LinearDiagRNN(8, 8, kind='dense');
      x = torch.randn(2, 5, 8);
      y1 = m.forward_naive(x);
      y2 = m.forward_scan(x);
      assert y1.shape == (2,5,8), y1.shape;
      assert y2.shape == (2,5,8), y2.shape;
      # naive and scan should agree (within tolerance) when scan is available
      if hasattr(torch, 'associative_scan'):
          assert torch.allclose(y1, y2, atol=1e-5), 'naive vs scan disagree'
      print('lin_rnn ok');
      " && .venv/bin/python experiments/recurrent_poc/bench_lin_rnn.py --quick 2>&1 | tee /tmp/bench_lin.out && grep -E "H.*T|dense" /tmp/bench_lin.out >/dev/null
    </automated>
  </verify>
  <done>
    `LinearDiagRNN` instantiates with kind="dense" and returns shape
    `(B, T, H)` from both `forward_naive` and `forward_scan`. When
    `torch.associative_scan` exists, naive and scan outputs match within
    1e-5. `bench_lin_rnn.py --quick` prints at least one timing row
    without crashing.
  </done>
</task>

</tasks>

<verification>
Run all three sanity-check commands from the quick-task description:

1. `.venv/bin/python -c "from experiments.recurrent_poc.layers import make_linear; m = make_linear('butterfly', 64, 64); print(m)"` — prints a module repr, no error.
2. `.venv/bin/python -c "from experiments.recurrent_poc.gru import StackedGateGRUCell; import torch; c = StackedGateGRUCell(16, 32, kind='dense'); x = torch.randn(2, 16); h = torch.zeros(2, 32); print(c(x, h).shape)"` — prints `torch.Size([2, 32])`.
3. `.venv/bin/python experiments/recurrent_poc/bench_gru.py --quick` — prints a table; exits 0.

Plus `bench_lin_rnn.py --quick` must also exit 0 and print a table.

No files written under `experiments/recurrent_poc/` beyond the six source
files listed in `files_modified`. No `*.json`, no cached artifacts.
</verification>

<success_criteria>
- Factory dispatches all four supported kinds and raises ValueError /
  NotImplementedError with clear messages for the rest.
- `StackedGateGRUCell` matches the standard PyTorch `nn.GRUCell` math
  (specifically `r * h_n`, not `r * (i_n + h_n)`).
- `bench_gru.py --quick` runs to completion on the RTX 4090 in <30s and
  prints a readable table including a cuDNN reference row.
- `bench_lin_rnn.py --quick` runs to completion and shows naive vs scan
  timings (or "N/A" if `torch.associative_scan` is unavailable).
- No combo crashes the sweep; shape-incompatible combos print
  "SKIPPED — <reason>".
- Every source file has a module docstring; no README.md exists under
  `experiments/recurrent_poc/`.
- Each `.py` file is <150 LOC (verify with `wc -l`).
</success_criteria>

<output>
After completion, create `.planning/quick/260419-olu-build-poc-benchmark-suite-for-recurrent-/260419-olu-SUMMARY.md`
summarizing: files created, sweep grid used in the quick smoke test,
any shape-incompatible combos observed, and one-line per-kind timing
excerpt from the quick bench (for later reference when expanding the
sweep).
</output>
