"""Linear RNN with diagonal state transition: h_t = a * h_{t-1} + B(x_t).

Provides both a naive Python for-loop unroll and an optional parallel
associative-scan implementation (when torch.associative_scan or its
private _higher_order_ops counterpart is available in this PyTorch build).

Use cases:
    - Microbenchmark naive vs scan throughput on the recurrence sweep.
    - Sanity-check that the scan formulation matches the naive output
      to within float32 tolerance.
"""

import warnings

warnings.filterwarnings("ignore", message=".*different CUDA versions.*")

import torch
from torch import nn

from .layers import make_linear


# Probe associative_scan availability once at import.
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
_warned_no_scan = False


class LinearDiagRNN(nn.Module):
    """h_t = a * h_{t-1} + B(x_t); y_t = C(h_t).

    Parameters:
        input_size: input feature dim
        hidden_size: hidden state dim (also the diagonal length)
        kind: shorthand that sets both kind_B and kind_C (overrides them).
        kind_B: structured-linear kind for the input projection B.
        kind_C: structured-linear kind for the output projection C.
        output_size: y_t dim, defaults to hidden_size
        bias: include bias on B and C

    The `a` parameter stays dense (shape (H,)) regardless of kind_B/kind_C.
    """

    def __init__(self, input_size: int, hidden_size: int,
                 kind: str = None, kind_B: str = "dense", kind_C: str = "dense",
                 output_size: int = None, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size if output_size is not None else hidden_size
        if kind is not None:
            kind_B = kind_C = kind
        self.kind_B = kind_B
        self.kind_C = kind_C
        # Init near 1 so the recurrence is stable by default (||a|| < 1, ~0.9).
        self.a = nn.Parameter(torch.full((hidden_size,), 0.9))
        self.B = make_linear(kind_B, input_size, hidden_size, bias=bias)
        self.C = make_linear(kind_C, hidden_size, self.output_size, bias=bias)

    def forward_naive(self, x, h0=None):
        """x: (B, T, input_size). Returns (B, T, output_size)."""
        u = self.B(x)  # (B, T, H)
        B, T, H = u.shape
        if h0 is None:
            h = torch.zeros(B, H, device=u.device, dtype=u.dtype)
        else:
            h = h0
        outs = []
        for t in range(T):
            h = self.a * h + u[:, t]
            outs.append(self.C(h))
        return torch.stack(outs, dim=1)

    def forward_scan(self, x, h0=None):
        """Parallel associative scan; falls back to naive if scan is unavailable."""
        global _warned_no_scan
        if not _HAS_SCAN:
            if not _warned_no_scan:
                warnings.warn(
                    "torch.associative_scan unavailable in this PyTorch build; "
                    "falling back to forward_naive.",
                    stacklevel=2,
                )
                _warned_no_scan = True
            return self.forward_naive(x, h0)

        u = self.B(x)  # (B, T, H)
        Bsz, T, H = u.shape
        # Broadcast a to (B, T, H) so every time-step shares the same diagonal.
        a_bt = self.a.expand(Bsz, T, H).contiguous()

        # combine: ((a1, b1), (a2, b2)) -> (a2 * a1, a2 * b1 + b2). Scanning
        # this over t with seed (a, u_t) gives h_t = sum_{s<=t} (prod_{r>s} a) * u_s.
        def combine(left, right):
            a1, b1 = left
            a2, b2 = right
            return (a2 * a1, a2 * b1 + b2)

        # combine_mode="generic" so this works on CPU too; pointwise requires CUDA.
        if h0 is not None:
            ones_a = torch.ones_like(a_bt[:, :1])
            a_in = torch.cat([ones_a, a_bt], dim=1)
            u_in = torch.cat([h0.unsqueeze(1), u], dim=1)
            scanned = _associative_scan(combine, (a_in, u_in), dim=1,
                                        combine_mode="generic")
            h_seq = scanned[1][:, 1:]
        else:
            scanned = _associative_scan(combine, (a_bt, u), dim=1,
                                        combine_mode="generic")
            h_seq = scanned[1]
        return self.C(h_seq)

    def forward(self, x, h0=None):
        return self.forward_naive(x, h0)


if __name__ == "__main__":
    m = LinearDiagRNN(8, 8, kind="dense")
    x = torch.randn(2, 5, 8)
    y = m.forward_naive(x)
    print(f"lin_rnn naive: x={tuple(x.shape)} -> y={tuple(y.shape)}")
    m2 = LinearDiagRNN(8, 8, kind_B="butterfly", kind_C="dense")
    y2 = m2.forward_naive(x)
    print(f"lin_rnn asymmetric (B=butterfly, C=dense): x={tuple(x.shape)} -> y={tuple(y2.shape)}")
    print(f"associative_scan available: {_HAS_SCAN}")
