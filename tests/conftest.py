"""Phase 5: backend fixture widened to ["torch", "triton"] per D-30.

Phase 7+ will extend to ``["torch", "triton", "cuda"]`` once the CUDA backend
axis is added per the milestone-wide TEST-03 (integration hardening). The
Triton parametrization is skipped on hosts without a registered Triton
diag_mult kernel (CPU-only runners, no-Triton envs). The fixture captures the
current ``_BACKEND``, sets the requested one, yields the actually-resolved
name, then restores the original on teardown so tests are order-independent.
"""
import pytest

import torch_structured  # noqa: F401 — triggers extension load + _ops.py resolver


@pytest.fixture(params=["torch", "triton"])
def backend(request):
    """Switch backend for the duration of a test, restore after."""
    if request.param == "triton" and not torch_structured._ops._has_triton_kernel("diag_mult"):
        pytest.skip("Triton kernel for diag_mult not installed (no CUDA or CPU-only runner)")
    original = torch_structured._ops._BACKEND
    chosen = torch_structured._ops.set_backend(request.param)
    yield chosen
    torch_structured._ops.set_backend(original)
