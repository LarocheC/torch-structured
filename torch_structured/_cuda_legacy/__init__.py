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

# DeprecationWarning emitter (Phase 10 D-74 / DEPR-02; message verbatim from
# 04-DEPRECATION-PLAN.md). The warning is emitted EXPLICITLY via
# ``warn_cuda_deprecation()`` from the resolver's explicit cuda-selection paths
# (set_backend('cuda') / auto-with-no-triton in _ops._resolve) — NOT at module
# import time. Decoupling the emission from import timing closes the DEPR-02
# leak: the routing-fallback closure in _ops.py imports this module on the
# DEFAULT triton backend, and an import-time warning would fire spuriously on a
# bare ``import torch_structured`` whenever a CUDA build is present. Importing
# _cuda_legacy (the suppressed leak-site import, or a per-op probe) is now
# side-effect-free w.r.t. the warning.
#
# Once-per-process gating is implemented via a module-level _WARNED flag rather
# than warnings.simplefilter("once") because simplefilter() PREPENDS to
# warnings.filters and would override any caller's warnings.catch_warnings() +
# simplefilter("ignore", DeprecationWarning) wrap (D-74b probe-silencing in
# _ops._has_cuda_legacy_for_op). With the flag pattern, a caller's outer
# 'ignore' filter is still the first-match when warnings.warn() runs, so a
# suppressed call stays silent — and once an explicit-cuda selection emits the
# warning, _WARNED=True prevents re-emission for the rest of the process.
# stacklevel=2 attributes the warning to the caller of warn_cuda_deprecation()
# (i.e., the _ops.py explicit-cuda binding site) rather than to this module.
_WARNED = False


def warn_cuda_deprecation():
    """Emit the legacy-CUDA DeprecationWarning at most once per process.

    Called explicitly from ``_ops._resolve`` on the explicit cuda-selection
    binding paths (D-74b / DEPR-02-leak). Idempotent via the module-level
    ``_WARNED`` gate, so all three per-op cuda bindings in a single
    ``_resolve('cuda')`` collapse to a single warning. Importing this module is
    side-effect-free w.r.t. the warning.
    """
    global _WARNED
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

__all__ = ["butterfly_multiply", "diag_mult", "hadamard_transform", "warn_cuda_deprecation"]
