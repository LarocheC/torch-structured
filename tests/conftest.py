"""Phase 6: backend fixture skip-gate widened to _has_any_triton_kernel() per D-39.

Phase 5 narrowed the skip-gate to the per-op diag_mult probe; Phase 6 widens
it to ``_has_any_triton_kernel()`` so the fixture skips the ``triton`` param
ONLY when NO per-op Triton kernel is installed. This lets
Phase 6 tests (which need only the ``hadamard_transform`` kernel) run on hosts
that have hadamard but not diag_mult, and vice versa; it also future-proofs
against Phase 7 (butterfly_multiply) lighting up its kernel.

Phase 7+ will extend ``params`` to ``["torch", "triton", "cuda"]`` once the
CUDA backend axis is added per the milestone-wide TEST-03 (integration
hardening); for Phase 6 we stay at ``["torch", "triton"]``. The fixture
captures the current ``_BACKEND``, sets the requested one, yields the
actually-resolved name, then restores the original on teardown so tests are
order-independent.
"""
import pytest

import torch_structured  # noqa: F401 — triggers extension load + _ops.py resolver


@pytest.fixture(params=["torch", "triton"])
def backend(request):
    """Switch backend for the duration of a test, restore after."""
    if request.param == "triton" and not torch_structured._ops._has_any_triton_kernel():
        pytest.skip("No Triton kernel installed (no CUDA or CPU-only runner)")
    original = torch_structured._ops._BACKEND
    chosen = torch_structured._ops.set_backend(request.param)
    yield chosen
    torch_structured._ops.set_backend(original)
