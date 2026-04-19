---
phase: 260419-pya
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - experiments/recurrent_poc/lru.py
  - experiments/recurrent_poc/mamba.py
  - experiments/recurrent_poc/bench_recurrent.py
autonomous: true
requirements:
  - QUICK-260419-pya
must_haves:
  truths:
    - "LRU(input_size, hidden_size, num_layers, batch_first, bidirectional, kind='butterfly') runs forward on CUDA and returns correct (out, h_n) shapes matching nn.GRU's contract."
    - "Mamba(input_size, hidden_size, num_layers, batch_first, kind='dense') runs forward on CUDA and returns correct (out, h_n) shapes."
    - "LRU's output projection is the real part of the complex state projection; the model emits float32 outputs."
    - "Both LRU and Mamba accept kind / kind_B / kind_C and route them through the existing make_linear factory from layers.py."
    - "bench_recurrent.py --quick prints a table with at least one row per model in {cudnn, gru, lru, mamba} and skips shape-incompatible combos with a SKIPPED row."
  artifacts:
    - path: "experiments/recurrent_poc/lru.py"
      provides: "LRU nn.Module with Orvieto-2023 stability parameterization, complex diagonal scan, bidirectional stacking."
      contains: "class LRU"
    - path: "experiments/recurrent_poc/mamba.py"
      provides: "Simplified Mamba S6 SSM with selective B/C/Δ, ZOH discretization, naive Python time loop."
      contains: "class Mamba"
    - path: "experiments/recurrent_poc/bench_recurrent.py"
      provides: "CLI benchmark comparing cuDNN nn.GRU vs StackedGateGRUCell vs LRU vs Mamba."
      contains: "def main"
  key_links:
    - from: "experiments/recurrent_poc/lru.py"
      to: "experiments/recurrent_poc/layers.make_linear"
      via: "kind_B / kind_C passed through to B_re, B_im, C_re, C_im projections"
      pattern: "make_linear\\(kind_[BC]"
    - from: "experiments/recurrent_poc/mamba.py"
      to: "experiments/recurrent_poc/layers.make_linear"
      via: "per-step B_t, C_t projections built via make_linear"
      pattern: "make_linear\\(kind_[BC]"
    - from: "experiments/recurrent_poc/bench_recurrent.py"
      to: "experiments/recurrent_poc/{lru,mamba,gru}.py + torch.nn.GRU"
      via: "imports LRU, Mamba, StackedGateGRUCell, unroll_cell"
      pattern: "from experiments.recurrent_poc"
---

<objective>
Add two modern recurrent layers (LRU and Mamba) to experiments/recurrent_poc/
whose constructor and forward contract mirror torch.nn.GRU (drop-in peers)
but whose input/output projections route through the existing make_linear
factory so they can be structured (butterfly/monarch/circulant) just like
the existing StackedGateGRUCell. Add a bench script that compares them
against cuDNN nn.GRU and the existing StackedGateGRUCell across a small
sweep.

Purpose: Unblock downstream speech-enhancement experiments that want
sub-quadratic linear-recurrence layers as GRU replacements. Keep the
implementations compact (POC grade) and match the existing file style.

Output: three new files in experiments/recurrent_poc/, no tests, no
commits of bench output. Total new code under 650 LOC.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@CLAUDE.md
@.planning/STATE.md
@experiments/recurrent_poc/layers.py
@experiments/recurrent_poc/lin_rnn.py
@experiments/recurrent_poc/gru.py
@experiments/recurrent_poc/bench_gru.py
@experiments/recurrent_poc/bench_lin_rnn.py

<interfaces>
<!-- Key contracts the executor needs from the existing codebase. -->
<!-- No codebase exploration needed beyond these. -->

From experiments/recurrent_poc/layers.py:
```python
_SUPPORTED = ("dense", "butterfly", "monarch", "circulant")

def make_linear(
    kind: str,
    in_features: int,
    out_features: int,
    bias: bool = True,
    **kwargs,
) -> nn.Module:
    """Returns an nn.Module with forward(x) -> y of shape (..., out_features).
    Raises ValueError on unknown kind; NotImplementedError on unwired kinds;
    may raise ValueError internally (e.g. circulant requires square pow-2)."""
```

From experiments/recurrent_poc/lin_rnn.py (associative_scan probe pattern, reuse verbatim):
```python
_associative_scan = None
try:
    if hasattr(torch, "associative_scan"):
        _associative_scan = torch.associative_scan  # type: ignore[attr-defined]
    else:
        from torch._higher_order_ops.associative_scan import associative_scan as _as
        _associative_scan = _as
except Exception:
    _associative_scan = None

_HAS_SCAN = _associative_scan is not None
```

From experiments/recurrent_poc/gru.py:
```python
class StackedGateGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kind="dense", bias=True, linear_factory=None): ...
    def forward(self, x_t, h_prev): ...

def unroll_cell(cell, x_seq, h0):
    """x_seq: (B, T, input_size), h0: (B, hidden_size) -> h_seq: (B, T, hidden_size)."""
```

From experiments/recurrent_poc/bench_gru.py (time_fn pattern, reuse verbatim):
```python
def time_fn(fn, *, warmup: int, iters: int, device: torch.device) -> float:
    """Median seconds per iteration with cuda.synchronize on CUDA."""
    ...
```

torch.nn.GRU contract we must mirror:
- __init__(input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
          dropout=0.0, bidirectional=False)
- forward(input, h_0=None) -> (output, h_n)
  - input: (T, B, input_size) if batch_first=False else (B, T, input_size)
  - output: (T, B, D*hidden_size) or (B, T, D*hidden_size); D = 2 if bidirectional else 1
  - h_n: (num_layers * D, B, hidden_size)
</interfaces>

<environment_notes>
- venv at .venv/ is CUDA-built (sm_89, RTX 4090). Any torch_structured code
  needs CUDA_HOME=/home/clement/torch-structured/.venv/lib/python3.11/site-packages/nvidia/cu13
- On branch feat/recurrent-poc-extensions — DO NOT switch branches. DO NOT
  create a worktree (venv is not reachable from one).
- torch 2.11.0+cu130 supports torch.associative_scan with combine_mode='generic'
  and complex tensors — verify with a quick probe before committing to the
  scan path. If the complex scan fails at runtime, fall back to naive loop
  with a warnings.warn().
- Suppress "different CUDA versions" warning via
  warnings.filterwarnings("ignore", message=".*different CUDA versions.*")
  in every new file, matching existing convention.
- Match existing file style: module docstring, absolute imports for scripts
  via sys.path.insert(0, repo_root) shim, argparse for CLIs.
</environment_notes>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create LRU layer (experiments/recurrent_poc/lru.py)</name>
  <files>experiments/recurrent_poc/lru.py</files>
  <action>
Create experiments/recurrent_poc/lru.py implementing the Linear Recurrent
Unit from Orvieto et al. 2023 as a drop-in nn.GRU peer. Target ~200 LOC.

File structure:
1. Module docstring explaining: Orvieto-2023 LRU, drop-in GRU peer, structured
   B/C via make_linear, complex state, stability parameterization. Cite:
   "# Orvieto et al. 2023 — https://arxiv.org/abs/2303.06349"
2. Suppress "different CUDA versions" warning at import (match lin_rnn.py).
3. Standard imports: math, warnings, torch, torch.nn as nn. Local import:
   from .layers import make_linear
4. Probe torch.associative_scan availability with the SAME pattern as
   lin_rnn.py (hasattr -> private _higher_order_ops fallback -> None).
   Store _HAS_SCAN and a module-level _warned_no_scan flag.

Define private _LRULayer(nn.Module) — single-direction, single-layer LRU:
  __init__(input_size, hidden_size, bias, kind_B, kind_C, r_min, r_max, max_phase):
    - Store config on self.
    - Init log-radius nu and log-phase theta parameters of shape (H,):
        u1 = torch.rand(H); u2 = torch.rand(H)
        nu_init    = torch.log(-0.5 * torch.log(u1 * (r_max**2 - r_min**2) + r_min**2))
        theta_init = torch.log(max_phase * u2)
      self.nu    = nn.Parameter(nu_init)
      self.theta = nn.Parameter(theta_init)
      # This makes |a_i| uniform in [r_min, r_max] and arg(a_i) uniform in [0, max_phase]
      # via the reparameterization |a| = exp(-exp(nu)), arg(a) = exp(theta).
    - Build input projection: B_re = make_linear(kind_B, input_size, hidden_size, bias=bias)
                              B_im = make_linear(kind_B, input_size, hidden_size, bias=bias)
    - Build output projection (no bias on these, real bias handled separately):
        C_re = make_linear(kind_C, hidden_size, hidden_size, bias=False)
        C_im = make_linear(kind_C, hidden_size, hidden_size, bias=False)
    - If bias=True: self.out_bias = nn.Parameter(torch.zeros(hidden_size))
      else register_parameter("out_bias", None).

  def _compute_a(self):
    # a = exp(-exp(nu) + i * exp(theta)), shape (H,), complex64
    a_mag = torch.exp(-torch.exp(self.nu))
    a_phase = torch.exp(self.theta)
    return torch.complex(a_mag * torch.cos(a_phase), a_mag * torch.sin(a_phase))

  def _compute_gamma(self):
    # gamma = sqrt(1 - |a|^2), shape (H,), real
    # |a|^2 = exp(-2*exp(nu))
    return torch.sqrt(1.0 - torch.exp(-2.0 * torch.exp(self.nu)))

  def forward(self, x, h0=None):
    # x: (B, T, input_size). h0: (B, H) complex or None.
    # Returns: (y, h_T) where y: (B, T, H) float32, h_T: (B, H) complex64.
    a = self._compute_a()                             # (H,) complex
    gamma = self._compute_gamma()                     # (H,) real
    b_re = self.B_re(x)                               # (B, T, H) real
    b_im = self.B_im(x)                               # (B, T, H) real
    u = torch.complex(b_re, b_im) * gamma             # (B, T, H) complex

    # Scan: expand a to (B, T, H). Try associative_scan on complex first; if
    # it raises at runtime (which we'll also cache via a module-level flag),
    # fall back to a Python loop.
    h_seq, h_last = _lru_scan(a, u, h0)               # helper below

    # Output projection: real(C h) = C_re(real(h)) - C_im(imag(h)) + bias
    y_re = self.C_re(h_seq.real) - self.C_im(h_seq.imag)
    if self.out_bias is not None:
        y_re = y_re + self.out_bias
    return y_re, h_last

Helper function _lru_scan(a, u, h0) at module scope:
  # a: (H,) complex; u: (B, T, H) complex; h0: (B, H) complex or None
  # Uses _HAS_SCAN + a runtime-try cache:
  global _SCAN_COMPLEX_OK  # tri-state: None (untried), True, False
  if _HAS_SCAN and (_SCAN_COMPLEX_OK is None or _SCAN_COMPLEX_OK):
      try:
          return _lru_scan_parallel(a, u, h0)
      except Exception as e:
          _SCAN_COMPLEX_OK = False
          warnings.warn(f"associative_scan failed on complex ({type(e).__name__}); "
                        f"falling back to naive LRU loop.", stacklevel=3)
  return _lru_scan_naive(a, u, h0)

_lru_scan_parallel(a, u, h0):
  # Broadcast a to (B, T, H); combine op (a1, b1), (a2, b2) -> (a1*a2, a2*b1 + b2)
  # Prepend (1, h0) if h0 provided, then slice off first element.
  Bsz, T, H = u.shape
  a_bt = a.expand(Bsz, T, H).contiguous()
  def combine(left, right):
      a1, b1 = left; a2, b2 = right
      return (a2 * a1, a2 * b1 + b2)
  if h0 is not None:
      ones_a = torch.ones_like(a_bt[:, :1])
      a_in = torch.cat([ones_a, a_bt], dim=1)
      u_in = torch.cat([h0.unsqueeze(1), u], dim=1)
      scanned = _associative_scan(combine, (a_in, u_in), dim=1, combine_mode="generic")
      h_seq = scanned[1][:, 1:]
  else:
      scanned = _associative_scan(combine, (a_bt, u), dim=1, combine_mode="generic")
      h_seq = scanned[1]
  return h_seq, h_seq[:, -1]

_lru_scan_naive(a, u, h0):
  Bsz, T, H = u.shape
  h = h0 if h0 is not None else torch.zeros(Bsz, H, dtype=u.dtype, device=u.device)
  outs = []
  for t in range(T):
      h = a * h + u[:, t]
      outs.append(h)
  h_seq = torch.stack(outs, dim=1)
  return h_seq, h

Define public LRU(nn.Module):
  __init__(input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
           dropout=0.0, bidirectional=False, *, kind=None, kind_B="dense",
           kind_C="dense", r_min=0.0, r_max=1.0, max_phase=2*math.pi):
    - if kind is not None: kind_B = kind_C = kind
    - D = 2 if bidirectional else 1
    - Build an nn.ModuleList of _LRULayer's. layer 0 takes input_size; layer l>0
      takes D * hidden_size. If bidirectional, store TWO _LRULayer per layer
      index (fwd, bwd) — use nn.ModuleList of nn.ModuleList, or a flat list of
      length num_layers * D with indexing i * D + d.
    - self.dropout = nn.Dropout(dropout) if dropout > 0 else None

  def forward(self, input, h_0=None):
    # Canonicalize to (B, T, *). If not batch_first, transpose.
    # h_0 may be (num_layers * D, B, H) real (from nn.GRU contract) or None.
    # If h_0 is real, lift to complex with zero imaginary part.
    # Iterate layers:
    #   For each layer l, run fwd _LRULayer on the current input;
    #     if bidirectional, run bwd _LRULayer on time-reversed input and
    #     time-reverse its output; concat along last dim.
    #   Collect the h_T's in order [layer0_fwd, layer0_bwd, layer1_fwd, ...].
    #   Apply self.dropout between layers if set (not after last layer).
    # Output:
    #   out: (B, T, D*H). If not batch_first, transpose back to (T, B, D*H).
    #   h_n: stack of collected h_T's, shape (num_layers * D, B, H).
    # h_T projection to real for h_n: use .real of the complex state.
    #   Document: "h_n returns real(h_T) — loses the imaginary component.
    #   For exact state carry-over across forward calls, track the complex
    #   state yourself." This is the explicit information-loss note the
    #   planning context calls out.

Edge cases / hygiene:
- If batch_first=False and num_layers > 1, always work on (B, T, *) internally
  and transpose once on entry and once on exit.
- Dropout goes on the OUTPUT of each stacked layer except the last (matches
  nn.GRU semantics).
- No __main__ smoke block is strictly required, but add a tiny one that
  builds LRU(8, 8, num_layers=2, batch_first=True, bidirectional=True) and
  prints output shapes — mirrors lin_rnn.py style.

Style: match lin_rnn.py and gru.py — module docstring, 4-space indent,
warnings.filterwarnings at top, no type stub files, type hints sparse.
Do NOT introduce new external deps.
  </action>
  <verify>
    <automated>CUDA_HOME=/home/clement/torch-structured/.venv/lib/python3.11/site-packages/nvidia/cu13 /home/clement/torch-structured/.venv/bin/python -c "
from experiments.recurrent_poc.lru import LRU
import torch
m = LRU(64, 128, num_layers=2, batch_first=True, bidirectional=True, kind='butterfly').cuda()
x = torch.randn(4, 200, 64, device='cuda')
out, hn = m(x)
assert out.shape == (4, 200, 256), f'out shape {out.shape}'
assert hn.shape == (4, 4, 128), f'hn shape {hn.shape}'
assert out.dtype == torch.float32, f'out dtype {out.dtype}'
out.sum().backward()
print('LRU ok:', out.shape, hn.shape, out.dtype)
"</automated>
  </verify>
  <done>
LRU smoke runs on CUDA, shapes are (4, 200, 256) for out and (4, 4, 128)
for h_n, out.dtype is float32, backward runs without error. No import-time
errors. File under ~220 LOC.
  </done>
</task>

<task type="auto">
  <name>Task 2: Create Mamba layer (experiments/recurrent_poc/mamba.py)</name>
  <files>experiments/recurrent_poc/mamba.py</files>
  <action>
Create experiments/recurrent_poc/mamba.py implementing a simplified Mamba
S6 SSM as a drop-in nn.GRU peer. Target ~250 LOC.

File structure:
1. Module docstring explaining: simplified S6-style selective SSM, drop-in
   GRU peer, per-step input-dependent B/C/Δ, ZOH discretization, naive
   Python time loop (NO selective-scan kernel — document this explicitly).
   Cite: "# Gu & Dao 2023 — https://arxiv.org/abs/2312.00752"
   Include the note verbatim:
   "# Real Mamba uses a fused CUDA selective_scan kernel; this is the
   reference Python loop — correct but slow. Expect ~10x slower than LRU
   at T=1000."
2. Suppress "different CUDA versions" warning at import.
3. Imports: math, warnings, torch, torch.nn as nn, torch.nn.functional as F.
   Local: from .layers import make_linear

Define private _MambaLayer(nn.Module) — single-direction, single-layer SSM:
  __init__(input_size, hidden_size, bias, kind_B, kind_C, d_state, dt_rank):
    - Store config on self. dt_rank resolution: if dt_rank == "auto":
      dt_rank = max(hidden_size // 16, 1); else int(dt_rank).
    - in_proj: self.in_proj = nn.Linear(input_size, hidden_size, bias=bias)
      # Always dense — per the plan context, structuring this is out of scope.
    - Per-step B, C (map hidden -> d_state): these are applied per time step to
      x_t already in hidden space.
        self.B_proj = make_linear(kind_B, hidden_size, d_state, bias=False)
        self.C_proj = make_linear(kind_C, hidden_size, d_state, bias=False)
    - dt_proj (low-rank): two dense Linears.
        self.dt_proj1 = nn.Linear(hidden_size, dt_rank, bias=False)
        self.dt_proj2 = nn.Linear(dt_rank, hidden_size, bias=True)
      Init self.dt_proj2.bias with softplus-inverse of a small value (e.g. 1e-3)
      so softplus(bias) starts ~1e-3 — matches Mamba's init.
        with torch.no_grad():
            inv_dt = math.log(math.expm1(1e-3))
            self.dt_proj2.bias.fill_(inv_dt)
    - A_log: learnable (hidden_size, d_state). Init via the S4/Mamba
      convention: A = -exp(A_log), so A_log init = log of 1..d_state repeated.
        A_log_init = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
        A_log_init = A_log_init.unsqueeze(0).expand(hidden_size, d_state).contiguous()
        self.A_log = nn.Parameter(A_log_init.clone())
    - D (skip, optional): for simplicity omit D. (Spec explicitly says OK to
      omit; keeps the drop-in story simple.)
    - out_bias: if bias, self.out_bias = nn.Parameter(torch.zeros(hidden_size));
      else register_parameter("out_bias", None). Applied once per time step on y.

  def forward(self, x_in, h0=None):
    # x_in: (B, T, input_size). h0 in (B, H, d_state) or None.
    # Returns: (y, h_last_avg) where y: (B, T, H), h_last_avg: (B, H).
    # h_last_avg is the mean-over-d_state projection documented in the spec.
    Bsz, T, _ = x_in.shape
    x = self.in_proj(x_in)                       # (B, T, H)
    B_t = self.B_proj(x)                         # (B, T, N)  where N = d_state
    C_t = self.C_proj(x)                         # (B, T, N)
    dt = F.softplus(self.dt_proj2(self.dt_proj1(x)))  # (B, T, H)
    A = -torch.exp(self.A_log)                    # (H, N)

    # Discretize:
    # A_bar = exp(dt[..., None] * A[None, None, :, :])   shape (B, T, H, N)
    # B_bar = dt[..., None] * B_t[..., None, :]          shape (B, T, H, N)
    A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
    B_bar = dt.unsqueeze(-1) * B_t.unsqueeze(-2)

    H = x.shape[-1]; N = A.shape[-1]
    h = h0 if h0 is not None else torch.zeros(Bsz, H, N, device=x.device, dtype=x.dtype)
    ys = []
    for t in range(T):
        # h: (B, H, N). A_bar[:, t]: (B, H, N). B_bar[:, t]: (B, H, N). x[:, t]: (B, H).
        h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
        # y_t = sum_n C_t[:, t, n] * h[:, :, n] -> (B, H)
        y_t = (C_t[:, t].unsqueeze(1) * h).sum(dim=-1)  # (B, H)
        ys.append(y_t)
    y = torch.stack(ys, dim=1)                    # (B, T, H)
    if self.out_bias is not None:
        y = y + self.out_bias
    # Project final h to (B, H) via mean over d_state as specified.
    return y, h.mean(dim=-1)

Define public Mamba(nn.Module):
  __init__(input_size, hidden_size, num_layers=1, bias=True, batch_first=False,
           dropout=0.0, bidirectional=False, *, kind=None, kind_B="dense",
           kind_C="dense", d_state=16, dt_rank="auto", expand=2):
    - `expand` is accepted for API compat with real Mamba but NOT used
      internally — document this in the docstring: "expand is reserved for
      API compatibility with the reference Mamba implementation; this POC
      does not widen the hidden dimension internally."
    - if kind is not None: kind_B = kind_C = kind
    - D = 2 if bidirectional else 1
    - Stack num_layers of _MambaLayer. Layer 0 takes input_size; layer l>0
      takes D * hidden_size as its input_size. If bidirectional, two per
      layer index (fwd, bwd). Flat ModuleList of length num_layers * D.
    - self.dropout = nn.Dropout(dropout) if dropout > 0 else None

  def forward(self, input, h_0=None):
    # Canonicalize to (B, T, *) via transpose if not batch_first.
    # h_0 accepted but documented: "h_0 is ignored in this POC — Mamba's
    # real state is (B, H, d_state) per layer; providing a (L*D, B, H) h_0
    # would require lifting each layer's h_0 back to d_state, which is
    # underdetermined. Accepted for API compat, not used." (Implement: warn
    # once via warnings.warn if h_0 is not None.)
    # Loop layers exactly like LRU:
    #   fwd output, optional bwd output on time-reversed input and reverse back;
    #   concat along last dim; collect h_last_avg's per (layer, direction);
    #   dropout between layers.
    # h_n: stack of the collected (B, H) tensors -> (L * D, B, H).
    # Transpose output back to (T, B, *) if not batch_first.

Edge cases / hygiene:
- Use F.softplus (not torch.nn.functional.softplus — they're the same but
  F is already imported).
- Keep the Python time loop simple; no .jit, no torch.compile.
- Add a tiny __main__ smoke: Mamba(8, 8, num_layers=2, batch_first=True).
  print output shapes.
- Do NOT introduce new external deps.

Style: match lru.py and gru.py conventions.
  </action>
  <verify>
    <automated>CUDA_HOME=/home/clement/torch-structured/.venv/lib/python3.11/site-packages/nvidia/cu13 /home/clement/torch-structured/.venv/bin/python -c "
from experiments.recurrent_poc.mamba import Mamba
import torch
m = Mamba(64, 128, num_layers=2, batch_first=True, kind='dense').cuda()
x = torch.randn(4, 200, 64, device='cuda')
out, hn = m(x)
assert out.shape == (4, 200, 128), f'out shape {out.shape}'
assert hn.shape == (2, 4, 128), f'hn shape {hn.shape}'
out.sum().backward()
print('Mamba ok:', out.shape, hn.shape)
"</automated>
  </verify>
  <done>
Mamba smoke runs on CUDA, shapes are (4, 200, 128) for out and (2, 4, 128)
for h_n, backward runs without error, no import-time errors, file under
~270 LOC.
  </done>
</task>

<task type="auto">
  <name>Task 3: Create bench_recurrent.py CLI</name>
  <files>experiments/recurrent_poc/bench_recurrent.py</files>
  <action>
Create experiments/recurrent_poc/bench_recurrent.py comparing cuDNN nn.GRU,
StackedGateGRUCell (from gru.py), LRU (from lru.py) and Mamba (from
mamba.py) across a small sweep. Target ~200 LOC.

File structure:
1. Module docstring: purpose + 2-3 example invocations (match bench_gru.py style).
2. Suppress "different CUDA versions" warning.
3. sys.path.insert(0, repo_root) shim identical to bench_gru.py (three
   dirname calls up from __file__ to get the repo root).
4. Standard imports: argparse, os, statistics, sys, time, warnings, torch,
   from torch import nn.
5. Local imports:
     from experiments.recurrent_poc.gru import StackedGateGRUCell, unroll_cell
     from experiments.recurrent_poc.lru import LRU
     from experiments.recurrent_poc.mamba import Mamba
6. Reuse time_fn verbatim from bench_gru.py.

CLI: argparse with flags
  --device (cuda|cpu, default cuda if available)
  --batch (int, default 32)
  --warmup (int, default 3)
  --iters (int, default 5)
  --hiddens (default "256,512")
  --seq-lens (default "500,1000")
  --models (default "cudnn,gru,lru,mamba")
  --kinds (default "dense,butterfly")
  --num-layers (int, default 1)
  --bidirectional (flag, default False)
  --quick (flag): overrides hiddens=64, seq-lens=32, kinds=dense, models=cudnn,gru,lru,mamba,
                  warmup=1, iters=1, batch=4, num-layers=1, bidirectional=False.

Per-combo bench_one(model, kind, H, T, batch, num_layers, bidirectional, device,
                    warmup, iters):
  Build the model for this combo:
    if model == "cudnn":
        if kind != "dense":
            return SKIPPED("cudnn only supports dense")
        m = nn.GRU(H, H, num_layers=num_layers, batch_first=True,
                   bidirectional=bidirectional).to(device)
        D = 2 if bidirectional else 1
        fwd = lambda: m(x)[0]
        def fwd_bwd():
            m.zero_grad(set_to_none=True)
            out, _ = m(x); out.sum().backward()
    elif model == "gru":
        if num_layers != 1 or bidirectional:
            return SKIPPED("gru cell POC is single-layer, unidirectional only")
        cell = StackedGateGRUCell(H, H, kind=kind).to(device)
        h0 = torch.zeros(batch, H, device=device)
        fwd = lambda: unroll_cell(cell, x, h0)
        def fwd_bwd():
            cell.zero_grad(set_to_none=True)
            unroll_cell(cell, x, h0).sum().backward()
        m = cell
    elif model == "lru":
        m = LRU(H, H, num_layers=num_layers, batch_first=True,
                bidirectional=bidirectional, kind=kind).to(device)
        fwd = lambda: m(x)[0]
        def fwd_bwd():
            m.zero_grad(set_to_none=True)
            out, _ = m(x); out.sum().backward()
    elif model == "mamba":
        m = Mamba(H, H, num_layers=num_layers, batch_first=True,
                  bidirectional=bidirectional, kind=kind).to(device)
        fwd = lambda: m(x)[0]
        def fwd_bwd():
            m.zero_grad(set_to_none=True)
            out, _ = m(x); out.sum().backward()
    else:
        return SKIPPED("unknown model")

  Wrap the whole thing (construction + timing) in try/except (RuntimeError,
  AssertionError, ValueError) and return a SKIPPED row with a truncated
  reason string (80 chars) — mirrors bench_gru.py / bench_lin_rnn.py style.

Return dict with keys: status, model, kind, H, T, num_layers, bidirectional,
params, t_fwd_ms, t_fwd_bwd_ms (ok) or status=skipped + reason.

_print_table(rows):
  columns: model | kind | H | T | layers | bi | params | fwd_ms | fwd_bwd_ms
  'bi' cell prints "yes"/"no".
  SKIPPED rows: fill model/kind/H/T/layers/bi cells, then "SKIPPED — {reason}"
  in params column, leave the two timing columns blank. Match the dash-padding
  idiom from bench_gru.py _print_table.

main():
  Parse args, apply --quick overrides, parse comma lists, iterate
    for model in models:
      for kind in kinds:
        for H in hiddens:
          for T in seq_lens:
            # Skip impossible combos cleanly (handled inside bench_one).
            print progress to stderr like bench_gru.py does
            rows.append(bench_one(...))
  _print_table(rows).

Hygiene:
- Create the input tensor x = torch.randn(batch, T, H, device=device) inside
  bench_one (fresh per combo).
- Do not compute model params for SKIPPED rows.
- No file writes; stdout only.
- No commits of bench output (executor: do not run full sweep, only --quick).

Style: mirror bench_gru.py precisely — same column alignment, same time_fn,
same warnings filter, same sys.path trick.
  </action>
  <verify>
    <automated>CUDA_HOME=/home/clement/torch-structured/.venv/lib/python3.11/site-packages/nvidia/cu13 /home/clement/torch-structured/.venv/bin/python /home/clement/torch-structured/experiments/recurrent_poc/bench_recurrent.py --quick 2>&1 | tee /tmp/bench_recurrent_quick.txt &amp;&amp; grep -q cudnn /tmp/bench_recurrent_quick.txt &amp;&amp; grep -q gru /tmp/bench_recurrent_quick.txt &amp;&amp; grep -q lru /tmp/bench_recurrent_quick.txt &amp;&amp; grep -q mamba /tmp/bench_recurrent_quick.txt &amp;&amp; echo 'bench_recurrent --quick ok'</automated>
  </verify>
  <done>
bench_recurrent.py --quick runs without raising, prints a table containing
at least one row per model in {cudnn, gru, lru, mamba}. SKIPPED rows for
incompatible combos render cleanly. No file writes. File under ~220 LOC.
  </done>
</task>

</tasks>

<verification>
Phase-level checks (executor must run all three after all tasks complete):

1. LRU smoke (exact command from the planning context):
   CUDA_HOME=/home/clement/torch-structured/.venv/lib/python3.11/site-packages/nvidia/cu13 \
     .venv/bin/python -c "from experiments.recurrent_poc.lru import LRU; import torch; \
       m = LRU(64, 128, num_layers=2, batch_first=True, bidirectional=True, kind='butterfly').cuda(); \
       x = torch.randn(4, 200, 64, device='cuda'); out, hn = m(x); \
       print('LRU ok:', out.shape, hn.shape, out.dtype)"
   Expect: out.shape == (4, 200, 256), hn.shape == (4, 4, 128), out.dtype == torch.float32.

2. Mamba smoke:
   CUDA_HOME=... .venv/bin/python -c "from experiments.recurrent_poc.mamba import Mamba; import torch; \
     m = Mamba(64, 128, num_layers=2, batch_first=True, kind='dense').cuda(); \
     x = torch.randn(4, 200, 64, device='cuda'); out, hn = m(x); \
     print('Mamba ok:', out.shape, hn.shape)"
   Expect: out.shape == (4, 200, 128), hn.shape == (2, 4, 128).

3. Bench smoke:
   CUDA_HOME=... .venv/bin/python experiments/recurrent_poc/bench_recurrent.py --quick
   Expect: a table with at least one row per model.

4. LOC budget:
   wc -l experiments/recurrent_poc/{lru,mamba,bench_recurrent}.py
   Expect: sum < 650.

5. No unintended edits:
   git status should show only the three new files (no modifications to
   layers.py, gru.py, lin_rnn.py, bench_gru.py, bench_lin_rnn.py).
</verification>

<success_criteria>
- experiments/recurrent_poc/lru.py exists, class LRU matches nn.GRU's
  constructor signature plus LRU-specific kwargs; forward I/O shapes
  match the LRU smoke expectations.
- experiments/recurrent_poc/mamba.py exists, class Mamba matches nn.GRU's
  constructor signature plus Mamba-specific kwargs; forward I/O shapes
  match the Mamba smoke expectations.
- experiments/recurrent_poc/bench_recurrent.py --quick runs and prints a
  table with >= 1 row per {cudnn, gru, lru, mamba}.
- kind_B / kind_C route through make_linear for both new layers.
- Total new code (wc -l of the three files) < 650 LOC.
- No modifications to any other file in experiments/recurrent_poc/.
- No benchmark output files committed.
- Branch remains feat/recurrent-poc-extensions.
</success_criteria>

<output>
After all tasks pass verification, create:
.planning/quick/260419-pya-add-lru-and-mamba-layers-with-structured/260419-pya-SUMMARY.md
using the standard GSD summary template.
</output>
