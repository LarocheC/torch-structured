"""Pure-PyTorch reference implementations used by the dispatch fallback path."""
from .butterfly import butterfly_multiply_torch  # noqa: F401
from .diag_mult import diag_mult  # noqa: F401 (re-exported)
from .hadamard import hadamard_transform_torch  # noqa: F401 (re-exported)

__all__ = ["butterfly_multiply_torch", "diag_mult", "hadamard_transform_torch"]
