"""Deprecated shim. LRU moved to torch_structured.recurrent.lru.

Kept so bench_recurrent.py's `from experiments.recurrent_poc.lru import LRU`
keeps working. New code should import directly from torch_structured.
"""

from torch_structured.recurrent.lru import (  # noqa: F401
    LRU,
    _HAS_SCAN,
)
