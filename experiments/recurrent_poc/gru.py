"""StackedGateGRUCell — a GRU cell whose two linears can be any structured kind.

Matches PyTorch's nn.GRUCell formulation:
    r = sigmoid(W_ir x + W_hr h)
    z = sigmoid(W_iz x + W_hz h)
    n = tanh   (W_in x + r * (W_hn h))
    h' = (1 - z) * n + z * h

The "stacked gate" name refers to fusing all three gate projections of the
input (resp. hidden) side into a single Linear of width 3*hidden_size, which
is what nn.GRUCell does internally.
"""

import functools
import warnings

warnings.filterwarnings("ignore", message=".*different CUDA versions.*")

import torch
from torch import nn

from .layers import make_linear


class StackedGateGRUCell(nn.Module):
    """GRU cell with structured input and hidden gate linears.

    Parameters:
        input_size: input feature dim
        hidden_size: hidden feature dim
        kind: one of make_linear's supported kinds ("dense", "butterfly", ...)
        bias: include bias on both linears
        linear_factory: optional callable(in_features=, out_features=) -> nn.Module.
            If provided, overrides `kind`. Useful for tests/mock injection.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kind: str = "dense",
        bias: bool = True,
        linear_factory=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kind = kind
        if linear_factory is None:
            linear_factory = functools.partial(make_linear, kind=kind, bias=bias)
        self.ih = linear_factory(in_features=input_size, out_features=3 * hidden_size)
        self.hh = linear_factory(in_features=hidden_size, out_features=3 * hidden_size)

    def forward(self, x_t, h_prev):
        gi = self.ih(x_t)
        gh = self.hh(h_prev)
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return (1 - z) * n + z * h_prev


if __name__ == "__main__":
    cell = StackedGateGRUCell(16, 32, kind="dense")
    x = torch.randn(4, 16)
    h = torch.zeros(4, 32)
    h_next = cell(x, h)
    print(f"dense GRU cell: x={tuple(x.shape)}, h={tuple(h.shape)} -> {tuple(h_next.shape)}")
