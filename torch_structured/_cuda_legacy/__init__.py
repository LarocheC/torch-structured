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
from .butterfly import butterfly_multiply  # noqa: F401

__all__ = ["butterfly_multiply"]
