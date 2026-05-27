"""Phase 4: backend fixture parametrized over ["torch"] only.

Phase 5+ will extend the ``params=`` list to ``["torch", "triton"]`` (when
the first real Triton kernel lands AND CUDA is available) and Phase 7+ to
``["torch", "triton", "cuda"]`` as the kernel set fills out. The fixture
captures the current ``_BACKEND``, sets the requested one, yields the
actually-resolved name, then restores the original on teardown so tests are
order-independent.
"""
import pytest

import torch_structured  # noqa: F401 — triggers extension load + _ops.py resolver


@pytest.fixture(params=["torch"])
def backend(request):
    """Switch backend for the duration of a test, restore after."""
    original = torch_structured._ops._BACKEND
    chosen = torch_structured._ops.set_backend(request.param)
    yield chosen
    torch_structured._ops.set_backend(original)
