"""Linear Recurrent Unit (LRU) — drop-in nn.GRU peer with complex diagonal state.

    h_t = a * h_{t-1} + gamma * B(x_t),    y_t = real(C h_t) + b

`a` is a complex diagonal reparameterized as |a_i| = exp(-exp(nu_i)) and
arg(a_i) = exp(theta_i) so |a_i| ~ U[r_min, r_max] and arg(a_i) ~ U[0,
max_phase] at init. gamma = sqrt(1 - |a|^2) normalizes input so the
stationary hidden variance stays O(1). B and C are routed through
make_linear (structurable). State is complex64 internally; outputs are
float32. The h_n in the nn.GRU-compatible return value is real(h_T) — the
imaginary state is discarded; track the complex state yourself if needed.

# Orvieto et al. 2023 — https://arxiv.org/abs/2303.06349
"""

import math
import warnings

warnings.filterwarnings("ignore", message=".*different CUDA versions.*")

import torch
from torch import nn

from .layers import make_linear


# Probe associative_scan availability once at import (same pattern as lin_rnn.py).
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
# Tri-state: None (untried), True (works on complex), False (fell back).
_SCAN_COMPLEX_OK = None


def _lru_scan_parallel(a, u, h0):
    """Parallel scan of h_t = a*h_{t-1} + u_t. a:(H,) u:(B,T,H) h0:(B,H)|None.
    Combine: (a1, b1) ⊕ (a2, b2) = (a1*a2, a2*b1 + b2)."""
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


def _lru_scan_naive(a, u, h0):
    """Naive Python for-loop; fallback if parallel scan raises on complex."""
    Bsz, T, H = u.shape
    h = torch.zeros(Bsz, H, dtype=u.dtype, device=u.device) if h0 is None else h0
    outs = []
    for t in range(T):
        h = a * h + u[:, t]
        outs.append(h)
    return torch.stack(outs, dim=1), h


def _lru_scan(a, u, h0):
    """Dispatch: try parallel scan on complex, cache failure, fall back."""
    global _SCAN_COMPLEX_OK
    if _HAS_SCAN and (_SCAN_COMPLEX_OK is None or _SCAN_COMPLEX_OK):
        try:
            out = _lru_scan_parallel(a, u, h0)
            if _SCAN_COMPLEX_OK is None:
                _SCAN_COMPLEX_OK = True
            return out
        except Exception as e:  # noqa: BLE001 - scan failures are varied
            _SCAN_COMPLEX_OK = False
            warnings.warn(
                f"associative_scan failed on complex ({type(e).__name__}: {e}); "
                f"falling back to naive LRU loop.",
                stacklevel=3,
            )
    return _lru_scan_naive(a, u, h0)


class _LRULayer(nn.Module):
    """Single-direction, single-layer LRU. Internal helper for LRU."""

    def __init__(self, input_size, hidden_size, bias=True, kind_B="dense",
                 kind_C="dense", r_min=0.0, r_max=1.0, max_phase=2 * math.pi):
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.kind_B, self.kind_C = kind_B, kind_C
        # Orvieto-2023 reparam: |a_i| ~ U[r_min, r_max], arg(a_i) ~ U[0, max_phase].
        u1, u2 = torch.rand(hidden_size), torch.rand(hidden_size)
        self.nu = nn.Parameter(torch.log(
            -0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2 + 1e-8)))
        self.theta = nn.Parameter(torch.log(max_phase * u2 + 1e-8))
        # B and C: real/imag halves of the complex projections via make_linear.
        self.B_re = make_linear(kind_B, input_size, hidden_size, bias=bias)
        self.B_im = make_linear(kind_B, input_size, hidden_size, bias=bias)
        self.C_re = make_linear(kind_C, hidden_size, hidden_size, bias=False)
        self.C_im = make_linear(kind_C, hidden_size, hidden_size, bias=False)
        if bias:
            self.out_bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter("out_bias", None)

    def _compute_a(self):
        # a = exp(-exp(nu) + i * exp(theta)), shape (H,) complex64.
        a_mag = torch.exp(-torch.exp(self.nu))
        a_phase = torch.exp(self.theta)
        return torch.complex(a_mag * torch.cos(a_phase), a_mag * torch.sin(a_phase))

    def _compute_gamma(self):
        # gamma = sqrt(1 - |a|^2), shape (H,) real; |a|^2 = exp(-2 exp(nu)).
        return torch.sqrt(1.0 - torch.exp(-2.0 * torch.exp(self.nu)))

    def forward(self, x, h0=None):
        """x: (B, T, input_size). h0: (B, H) complex or None.
        Returns (y, h_last) with y: (B, T, H) float32, h_last: (B, H) complex.
        """
        a = self._compute_a()                         # (H,) complex
        gamma = self._compute_gamma()                 # (H,) real
        b_re = self.B_re(x)                           # (B, T, H) real
        b_im = self.B_im(x)                           # (B, T, H) real
        u = torch.complex(b_re, b_im) * gamma         # (B, T, H) complex

        h_seq, h_last = _lru_scan(a, u, h0)

        # Real output projection: real(C h) = C_re(real(h)) - C_im(imag(h)).
        y_re = self.C_re(h_seq.real) - self.C_im(h_seq.imag)
        if self.out_bias is not None:
            y_re = y_re + self.out_bias
        return y_re, h_last


class LRU(nn.Module):
    """Stacked, optionally bidirectional LRU matching nn.GRU's forward contract.

    Args mirror nn.GRU (input_size, hidden_size, num_layers, bias, batch_first,
    dropout, bidirectional) plus: kind / kind_B / kind_C route through
    make_linear; r_min, r_max, max_phase control eigenvalue init.

    forward(input, h_0=None):
        input: (T, B, input_size) or (B, T, input_size) per batch_first.
        h_0: (num_layers * D, B, hidden_size) real, optional (lifted to
             complex with zero imag). None -> zero state.
        output: (T, B, D*H) or (B, T, D*H); h_n: (L*D, B, H) = real(h_T).
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0.0, bidirectional=False,
                 *, kind=None, kind_B="dense", kind_C="dense",
                 r_min=0.0, r_max=1.0, max_phase=2 * math.pi):
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

        D = 2 if bidirectional else 1
        self.num_directions = D

        layers = []
        for l in range(num_layers):
            layer_in = input_size if l == 0 else D * hidden_size
            for _ in range(D):
                layers.append(_LRULayer(
                    layer_in, hidden_size, bias=bias,
                    kind_B=kind_B, kind_C=kind_C,
                    r_min=r_min, r_max=r_max, max_phase=max_phase,
                ))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, input, h_0=None):
        # Canonicalize to (B, T, *).
        if self.batch_first:
            x = input
        else:
            x = input.transpose(0, 1).contiguous()
        Bsz = x.size(0)
        D = self.num_directions
        H = self.hidden_size

        # Lift h_0 to complex if supplied. nn.GRU's h_0 is real (L*D, B, H).
        h0_list = [None] * (self.num_layers * D)
        if h_0 is not None:
            if not torch.is_complex(h_0):
                h_0c = torch.complex(h_0, torch.zeros_like(h_0))
            else:
                h_0c = h_0
            for i in range(self.num_layers * D):
                h0_list[i] = h_0c[i]

        h_T_list = []
        cur = x
        for l in range(self.num_layers):
            outs = []
            for d in range(D):
                layer = self.layers[l * D + d]
                inp = cur if d == 0 else cur.flip(1)
                y, h_last = layer(inp, h0_list[l * D + d])
                if d == 1:
                    y = y.flip(1)
                outs.append(y)
                h_T_list.append(h_last)
            cur = torch.cat(outs, dim=-1) if D > 1 else outs[0]
            if self.dropout is not None and l < self.num_layers - 1:
                cur = self.dropout(cur)

        out = cur  # (B, T, D*H)
        if not self.batch_first:
            out = out.transpose(0, 1).contiguous()

        # h_n: stack of real(h_T_list) -> (L*D, B, H).
        h_n = torch.stack([h.real for h in h_T_list], dim=0)
        return out, h_n


if __name__ == "__main__":
    m = LRU(8, 8, num_layers=2, batch_first=True, bidirectional=True, kind="dense")
    x = torch.randn(2, 5, 8)
    out, hn = m(x)
    print(f"LRU: x={tuple(x.shape)} -> out={tuple(out.shape)}, hn={tuple(hn.shape)}")
    print(f"associative_scan available: {_HAS_SCAN}")
