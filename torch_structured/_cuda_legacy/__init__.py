"""Wrapper around the already-loaded torch.ops.torch_structured.* C++ ops.

The .so was loaded eagerly by torch_structured.butterfly's __init__.py at
package import time (see torch_structured/butterfly/__init__.py:22-39). If it
failed to register, this import path will surface AttributeError when the
resolver probes _has_cuda_legacy().

This module exists so the _ops.py resolver can do
``from torch_structured._cuda_legacy import butterfly_multiply`` uniformly,
regardless of whether butterfly's compiled .so loaded successfully. Phase 10
may absorb the loader into _cuda_legacy/ when butterfly/__init__.py collapses
(per 04-DEPRECATION-PLAN.md).
"""
import warnings

# DeprecationWarning installation (Phase 10 D-74 / DEPR-02; verbatim from
# 04-DEPRECATION-PLAN.md). The "once" filter suppresses repeats based on
# (message, category) IGNORING module and line number — even if multiple call
# sites import this module, the warning fires exactly once per process.
# stacklevel=2 attributes the warning to the importer (e.g., _ops.py's
# `from torch_structured._cuda_legacy import butterfly_multiply` line) rather
# than to the warning line inside this module.
warnings.simplefilter("once", DeprecationWarning)

warnings.warn(
    "torch_structured: the CUDA C++ backend (csrc/) is deprecated and will be "
    "default-disabled in v1.3, with full removal in v1.4+. "
    "Switch to TORCH_STRUCTURED_BACKEND=triton (default in v1.2). "
    "See the v1.2 release notes for migration guidance.",
    DeprecationWarning,
    stacklevel=2,
)

from .butterfly import butterfly_multiply  # noqa: F401, E402
from .diag_mult import diag_mult  # noqa: F401, E402  — may raise RuntimeError if .so absent
from .hadamard import hadamard_transform  # noqa: F401, E402  — may raise RuntimeError if .so absent

__all__ = ["butterfly_multiply", "diag_mult", "hadamard_transform"]
