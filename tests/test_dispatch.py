"""Five-test acceptance suite for the Phase 4 demonstrator op.

These tests are the literal acceptance gate for Phase 4 per
``<verification_pattern>`` in the planning context. If
``test_demo_identity_compile_fake_tensor_trace`` fails with the original
260419-p27 error string ("The tensor has a non-zero number of elements, but
its data is not allocated yet"), Phase 4 has not shipped its promised fix.

Tests cover (per D-14 in 04-CONTEXT.md):

* (D-14a) ``torch.compile(fullgraph=True)`` traces cleanly with no graph break
* (D-14b) ``torch.autograd.gradcheck`` passes (autograd wiring correct)
* (D-14c) Explicit ``FakeTensorMode`` trace does not raise the 260419-p27 bug
* Eager fp32 + complex64 paths through the wrapper boundary
"""
import pytest
import torch

from torch_structured._ops import _demo_identity_op


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="demo op is GPU-only")


def test_demo_identity_eager_fp32():
    """Plain fp32 eager identity copy — sanity check on the wrap_triton plumbing."""
    x = torch.randn(128, device="cuda", requires_grad=True)
    y = _demo_identity_op(x)
    assert torch.equal(y, x)


def test_demo_identity_eager_complex64():
    """Exercises the view_as_real/view_as_complex path through the wrapper boundary.

    D-14 recommendation; planned-in feature per CHECKER B2 fix and RESOLVED
    Open Question 1 in 04-RESEARCH.md. The wrapper unconditionally routes
    complex inputs via ``view_as_real → kernel → view_as_complex`` so Phase 7
    can inherit a working reference.
    """
    x = torch.randn(128, dtype=torch.complex64, device="cuda", requires_grad=True)
    y = _demo_identity_op(x)
    assert torch.equal(y, x)
    assert y.dtype == torch.complex64


def test_demo_identity_gradcheck():
    """Verify autograd wiring via gradcheck (D-14b).

    gradcheck uses fp64 by default — declare the op accepts it via type
    promotion. The identity op's analytical gradient is the identity, and
    gradcheck's numerical perturbation will agree.
    """
    x = torch.randn(16, dtype=torch.float64, device="cuda", requires_grad=True)
    assert torch.autograd.gradcheck(_demo_identity_op, (x,), eps=1e-6, atol=1e-5)


def test_demo_identity_compile_no_graph_break():
    """The 260419-p27 fix verification: register_fake must let dynamo trace through.

    ``fullgraph=True`` raises on graph break — this is the strict gate per
    D-14a. Without ``register_fake`` on ``_demo_identity_op``, dynamo would
    fail to resolve the meta kernel and either graph-break or raise the
    "data is not allocated yet" error.
    """
    @torch.compile(fullgraph=True)
    def f(x):
        return _demo_identity_op(x) * 2

    x = torch.randn(128, device="cuda", requires_grad=True)
    y = f(x)
    assert torch.allclose(y, x * 2)


def test_demo_identity_compile_fake_tensor_trace():
    """Explicit fake-tensor trace — the literal 260419-p27 reproducer.

    THIS IS THE BLOCKING ACCEPTANCE GATE per planning context
    ``<verification_pattern>``. The assertion is that NO
    ``"The tensor has a non-zero number of elements, but its data is not
    allocated yet"`` error is raised. The ``register_fake`` decorator on
    ``_demo_identity_op`` is what makes this test pass — without it, this
    test reproduces the original bug.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x = torch.empty(128, device="cuda", requires_grad=True)
        # MUST NOT raise "The tensor has a non-zero number of elements,
        # but its data is not allocated yet"
        y = _demo_identity_op(x)
        assert y.shape == x.shape
        assert y.dtype == x.dtype
