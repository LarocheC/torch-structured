"""Pure-PyTorch reference implementations used by the dispatch fallback path."""
from .butterfly import butterfly_multiply_torch  # noqa: F401

__all__ = ["butterfly_multiply_torch"]
