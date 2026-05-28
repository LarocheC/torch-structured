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
# 04-DEPRECATION-PLAN.md). Once-per-process gating is implemented via a
# module-level _WARNED flag rather than warnings.simplefilter("once") because
# simplefilter() PREPENDS to warnings.filters and would override any prior
# warnings.catch_warnings() + simplefilter("ignore", DeprecationWarning) wrap
# (D-74b probe-silencing in _ops._has_cuda_legacy_for_op). With the flag
# pattern, the probe's outer 'ignore' filter is still the first-match when
# warnings.warn() runs, so the probe stays silent — and once the user-facing
# path emits the warning, _WARNED=True prevents re-emission for the rest of
# the process.
# stacklevel=2 attributes the warning to the importer (e.g., _ops.py's
# `from torch_structured._cuda_legacy import butterfly_multiply` line) rather
# than to the warning line inside this module.
_WARNED = False
if not _WARNED:
    warnings.warn(
        "torch_structured: the CUDA C++ backend (csrc/) is deprecated and will be "
        "default-disabled in v1.3, with full removal in v1.4+. "
        "Switch to TORCH_STRUCTURED_BACKEND=triton (default in v1.2). "
        "See the v1.2 release notes for migration guidance.",
        DeprecationWarning,
        stacklevel=2,
    )
    _WARNED = True

from .butterfly import butterfly_multiply  # noqa: F401, E402
from .diag_mult import diag_mult  # noqa: F401, E402  — may raise RuntimeError if .so absent
from .hadamard import hadamard_transform  # noqa: F401, E402  — may raise RuntimeError if .so absent

__all__ = ["butterfly_multiply", "diag_mult", "hadamard_transform"]
