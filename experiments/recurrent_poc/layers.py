"""Deprecated shim. make_linear moved to torch_structured.factory.

Kept so existing bench scripts (bench_gru.py, bench_lin_rnn.py,
bench_recurrent.py) and the sibling modules (gru.py, lin_rnn.py, mamba.py)
that do `from .layers import make_linear` keep working.
"""

from torch_structured.factory import (  # noqa: F401
    make_linear,
    _ButterflyLinear,
    _MonarchLinear,
    _CirculantLinear,
    _is_pow2,
    _SUPPORTED,
    _NOT_WIRED,
)
