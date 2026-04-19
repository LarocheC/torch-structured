"""Simplified Mamba (S6) SSM — drop-in nn.GRU peer with selective per-step state.

Selective SSM: for each time step t compute input-dependent B_t, C_t, and
Δ_t from x_t; discretize (A, B) via ZOH to (A_bar_t, B_bar_t); step the
state h_t = A_bar_t ⊙ h_{t-1} + B_bar_t ⊙ x_t; read out y_t = C_t · h_t.
A is a static (H, d_state) negative-exp parameter; B_t, C_t come from
projections on x (hidden space); Δ_t uses a low-rank (dt_rank) projection
followed by softplus.

# Real Mamba uses a fused CUDA selective_scan kernel; this is the
# reference Python loop — correct but slow. Expect ~10x slower than LRU
# at T=1000.

# Gu & Dao 2023 — https://arxiv.org/abs/2312.00752
"""

import math
import warnings

warnings.filterwarnings("ignore", message=".*different CUDA versions.*")

import torch
from torch import nn
from torch.nn import functional as F

from .layers import make_linear


_h0_warned = False


class _MambaLayer(nn.Module):
    """Single-direction, single-layer simplified S6. Internal helper for Mamba."""

    def __init__(self, input_size, hidden_size, bias=True, kind_B="dense",
                 kind_C="dense", d_state=16, dt_rank="auto"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.d_state = d_state
        self.kind_B = kind_B
        self.kind_C = kind_C

        if dt_rank == "auto":
            dt_rank = max(hidden_size // 16, 1)
        self.dt_rank = int(dt_rank)

        # Project input to hidden space (dense by design — orthogonal to B/C
        # comparison, keeps per-step shapes simple).
        self.in_proj = nn.Linear(input_size, hidden_size, bias=bias)

        # Per-step B_t, C_t: hidden -> d_state. Structured via make_linear.
        self.B_proj = make_linear(kind_B, hidden_size, d_state, bias=False)
        self.C_proj = make_linear(kind_C, hidden_size, d_state, bias=False)

        # Δ_t: low-rank (H -> dt_rank -> H), with a softplus that starts
        # around 1e-3 thanks to the softplus-inverse bias init.
        self.dt_proj1 = nn.Linear(hidden_size, self.dt_rank, bias=False)
        self.dt_proj2 = nn.Linear(self.dt_rank, hidden_size, bias=True)
        with torch.no_grad():
            inv_dt = math.log(math.expm1(1e-3))
            self.dt_proj2.bias.fill_(inv_dt)

        # A: (H, d_state). S4/Mamba init: A = -exp(A_log), A_log = log(1..N).
        A_log_init = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
        A_log_init = A_log_init.unsqueeze(0).expand(hidden_size, d_state).contiguous()
        self.A_log = nn.Parameter(A_log_init.clone())

        if bias:
            self.out_bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter("out_bias", None)

    def forward(self, x_in, h0=None):
        """x_in: (B, T, input_size). h0: (B, H, d_state) or None.
        Returns (y, h_last_avg) with y: (B, T, H), h_last_avg: (B, H).
        h_last_avg = h.mean(dim=-1) — a lossy reduction for nn.GRU-shape compat.
        """
        Bsz, T, _ = x_in.shape
        x = self.in_proj(x_in)                         # (B, T, H)
        B_t = self.B_proj(x)                           # (B, T, N)
        C_t = self.C_proj(x)                           # (B, T, N)
        dt = F.softplus(self.dt_proj2(self.dt_proj1(x)))  # (B, T, H)
        A = -torch.exp(self.A_log)                     # (H, N)

        # Discretize: A_bar = exp(dt ⊗ A), B_bar = dt ⊗ B_t (linear approx).
        A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, T, H, N)
        B_bar = dt.unsqueeze(-1) * B_t.unsqueeze(-2)                      # (B, T, H, N)

        H = x.shape[-1]
        N = A.shape[-1]
        if h0 is None:
            h = torch.zeros(Bsz, H, N, device=x.device, dtype=x.dtype)
        else:
            h = h0

        ys = []
        for t in range(T):
            # h, A_bar[:,t], B_bar[:,t]: (B, H, N); x[:,t]: (B, H).
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            # y_t = sum_n C_t[:, t, n] * h[:, :, n] -> (B, H)
            y_t = (C_t[:, t].unsqueeze(1) * h).sum(dim=-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)                     # (B, T, H)
        if self.out_bias is not None:
            y = y + self.out_bias
        # Reduce final (B, H, N) state to (B, H) via mean over d_state.
        return y, h.mean(dim=-1)


class Mamba(nn.Module):
    """Stacked, optionally bidirectional simplified Mamba matching nn.GRU's contract.

    Parameters:
        input_size, hidden_size: feature dims.
        num_layers: stack depth (each stack element has D=1 or 2 directions).
        bias: include bias on in_proj and output.
        batch_first: if True, I/O are (B, T, *); else (T, B, *).
        dropout: applied on output of each layer except the last.
        bidirectional: if True, run a second independent layer on the
            time-reversed input and concat along the feature axis.
        kind: shorthand that sets both kind_B and kind_C.
        kind_B, kind_C: make_linear kinds for the per-step B, C projections.
        d_state: internal SSM state dim N.
        dt_rank: low-rank size for Δ projection; "auto" -> max(H//16, 1).
        expand: accepted for API compat with the reference Mamba; NOT used
            internally (this POC does not widen the hidden dimension).

    forward(input, h_0=None):
        input: (T, B, input_size) or (B, T, input_size).
        h_0: accepted but IGNORED (real Mamba state is (B, H, d_state) per
             layer; a (L*D, B, H) h_0 is underdetermined). Warn once if
             supplied.
        output: (T, B, D*H) or (B, T, D*H).
        h_n: (num_layers * D, B, H) = stack of h.mean(dim=-1) per layer/dir.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.0, bidirectional=False,
                 *, kind=None, kind_B="dense", kind_C="dense",
                 d_state=16, dt_rank="auto", expand=2):
        super().__init__()
        if kind is not None:
            kind_B = kind_C = kind
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.kind_B = kind_B
        self.kind_C = kind_C
        self.d_state = d_state
        self.expand = expand  # reserved for API parity only

        D = 2 if bidirectional else 1
        self.num_directions = D

        layers = []
        for l in range(num_layers):
            layer_in = input_size if l == 0 else D * hidden_size
            for _ in range(D):
                layers.append(_MambaLayer(
                    layer_in, hidden_size, bias=bias,
                    kind_B=kind_B, kind_C=kind_C,
                    d_state=d_state, dt_rank=dt_rank,
                ))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, input, h_0=None):
        global _h0_warned
        if h_0 is not None and not _h0_warned:
            warnings.warn(
                "Mamba.forward: h_0 is ignored in this POC — real Mamba state is "
                "(B, H, d_state) per layer; providing a (L*D, B, H) h_0 is "
                "underdetermined. Accepted for API compat only.",
                stacklevel=2,
            )
            _h0_warned = True

        if self.batch_first:
            x = input
        else:
            x = input.transpose(0, 1).contiguous()
        D = self.num_directions

        h_T_list = []
        cur = x
        for l in range(self.num_layers):
            outs = []
            for d in range(D):
                layer = self.layers[l * D + d]
                inp = cur if d == 0 else cur.flip(1)
                y, h_last_avg = layer(inp, None)
                if d == 1:
                    y = y.flip(1)
                outs.append(y)
                h_T_list.append(h_last_avg)
            cur = torch.cat(outs, dim=-1) if D > 1 else outs[0]
            if self.dropout is not None and l < self.num_layers - 1:
                cur = self.dropout(cur)

        out = cur
        if not self.batch_first:
            out = out.transpose(0, 1).contiguous()
        h_n = torch.stack(h_T_list, dim=0)
        return out, h_n


if __name__ == "__main__":
    m = Mamba(8, 8, num_layers=2, batch_first=True)
    x = torch.randn(2, 5, 8)
    out, hn = m(x)
    print(f"Mamba: x={tuple(x.shape)} -> out={tuple(out.shape)}, hn={tuple(hn.shape)}")
