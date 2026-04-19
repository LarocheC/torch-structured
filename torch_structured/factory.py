"""Factory for (x) -> y linear-like modules backed by torch_structured primitives.

Supported kinds:
    - "dense"     : torch.nn.Linear
    - "butterfly" : torch_structured.Butterfly (signature normalized to (x) -> y)
    - "monarch"   : torch_structured.monarch.blockdiag_linear.BlockdiagLinear
    - "circulant" : custom rfft-based circulant; requires square power-of-2 size

Unknown kinds raise ValueError; "ldr"/"fastfood" raise NotImplementedError to
surface that they exist in torch_structured.structured but are not wired up
in this factory.
"""

import math
import warnings

warnings.filterwarnings("ignore", message=".*different CUDA versions.*")

import torch
from torch import nn

from torch_structured import Butterfly
from torch_structured.monarch.blockdiag_linear import BlockdiagLinear


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


class _ButterflyLinear(nn.Module):
    """Wrap torch_structured.Butterfly so its forward is plain (x) -> y.

    Butterfly.forward accepts (input, transpose=False, conjugate=False, ...);
    nn.Sequential / GRU code expects exactly one positional tensor, so we
    discard the extra kwargs at the call site.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.b = Butterfly(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.b(x)


class _CirculantLinear(nn.Module):
    """Square circulant via real FFT: y = irfft(rfft(col) * rfft(x)).

    Requires in == out and a power-of-2 size. ~10 lines, O(n log n) per
    forward, all-float32 path. Mirrors how a production circulant layer
    would be written without going through torch_structured.special.circulant
    (which bakes col into a sub-module at construction time).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if in_features != out_features or not _is_pow2(in_features):
            raise ValueError(
                f"circulant requires square power-of-2 sizes; "
                f"got in={in_features}, out={out_features}"
            )
        n = in_features
        self.n = n
        self.col = nn.Parameter(torch.randn(n) / math.sqrt(n))
        if bias:
            self.bias = nn.Parameter(torch.zeros(n))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        col_f = torch.fft.rfft(self.col)
        x_f = torch.fft.rfft(x, dim=-1)
        y = torch.fft.irfft(col_f * x_f, n=self.n, dim=-1)
        if self.bias is not None:
            y = y + self.bias
        return y


class _MonarchLinear(nn.Module):
    """Wrap BlockdiagLinear with a sane default for nblocks on small H.

    nblocks defaults to min(4, in, out) so H<4 doesn't crash. If the caller
    explicitly passes nblocks via kwargs, we honor their choice.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, **kwargs):
        super().__init__()
        if "nblocks" not in kwargs:
            kwargs["nblocks"] = min(4, in_features, out_features)
        self.bd = BlockdiagLinear(in_features, out_features, bias=bias, **kwargs)

    def forward(self, x):
        return self.bd(x)


_SUPPORTED = ("dense", "butterfly", "monarch", "circulant")
_NOT_WIRED = ("ldr", "fastfood")


def make_linear(
    kind: str,
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
    **kwargs,
) -> nn.Module:
    """Build an (x) -> y linear-like nn.Module of the requested kind.

    Parameters:
        kind: one of {"dense", "butterfly", "monarch", "circulant"}.
        in_features, out_features: standard nn.Linear sizes.
        bias: include additive bias (keyword-only).
        **kwargs: forwarded to the underlying constructor (e.g. nblocks for monarch).

    Returns:
        nn.Module with forward(x) -> y of shape (..., out_features).
    """
    if kind == "dense":
        return nn.Linear(in_features, out_features, bias=bias)
    if kind == "butterfly":
        return _ButterflyLinear(in_features, out_features, bias=bias)
    if kind == "monarch":
        return _MonarchLinear(in_features, out_features, bias=bias, **kwargs)
    if kind == "circulant":
        return _CirculantLinear(in_features, out_features, bias=bias)
    if kind in _NOT_WIRED:
        raise NotImplementedError(
            f"{kind} not wired up in the POC factory; see torch_structured.structured.layers "
            f"if you want to add it"
        )
    raise ValueError(
        f"unknown kind={kind!r}; supported: " + ", ".join(_SUPPORTED)
    )
