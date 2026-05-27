"""Triton kernel home. Empty in Phase 4; populated by Phase 5+ kernel ports.

This package is the future home for ported Triton kernels (Phase 5: diag_mult,
Phase 6: hadamard, Phase 7: butterfly forward, Phase 8: butterfly backward).
In Phase 4 the directory exists as a placeholder so that
``_ops._has_triton_kernel(op_name)`` has a parent package to probe — the
per-op probe will return ``False`` because no submodule exists yet.

Do NOT import this package from ``torch_structured/__init__.py``. The lazy
probe in ``_ops.py._has_triton()`` is the only intended call site for ``import
triton``. In Phase 4 the demonstrator op (``_ops._demo_identity_op``) ALSO
imports triton at the top of ``_ops.py``; that cost is unavoidable while the
demonstrator lives and disappears when Phase 5 deletes it (per D-13).
"""
try:
    import triton  # noqa: F401 — exposes Triton package as importable, but contents land in Phase 5+

    HAS_TRITON: bool = True
except ImportError:
    HAS_TRITON: bool = False
