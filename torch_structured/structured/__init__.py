"""Structured-matrix layers with low displacement rank.

Ported from https://github.com/HazyResearch/structured-nets (PyTorch side),
accompanying Thomas, Gu, Dao, Rudra, Re, *Learning Compressed Transforms with
Low Displacement Rank*, NeurIPS 2018.

This subpackage is intentionally namespaced so it does not collide with the
native-complex utilities in the top-level ``torch_structured`` package.
"""

from . import _compat  # noqa: F401 - patches torch.rfft/irfft on import
