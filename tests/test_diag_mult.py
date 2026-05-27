"""Cross-backend correctness + fp64 gradcheck tests for diag_mult (TRI-01).

Tests are parametrized over the ``backend`` fixture from ``tests/conftest.py``
(``['torch', 'triton']`` per D-30); the Triton branch is skipped on hosts
without the kernel installed. All tests call ``torch_structured._ops.diag_mult``
via attribute access (D-05) so set_backend() rebindings take effect.

The fp64 gradcheck tests are the load-bearing acceptance gates per D-26:
* ``test_diag_mult_gradcheck_fp64_real``: validates register_autograd plumbing
* ``test_diag_mult_gradcheck_fp64_complex``: the Wirtinger acceptance gate;
  fails with errors ~2.0 if ``.conj()`` is missing from the backward callback
"""
import itertools

import pytest
import torch

import torch_structured  # noqa: F401 — triggers extension load + _ops.py resolver
from torch_structured._torch_ref.diag_mult import diag_mult as diag_mult_ref


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="diag_mult tests require CUDA"
)


def test_diag_mult_eager_fp32(backend):
    """Forward correctness vs torch_ref oracle, fp32, batched v + broadcast subdiag."""
    N, B = 128, 4
    s = torch.randn(N, device="cuda", dtype=torch.float32)
    v = torch.randn(B, N, device="cuda", dtype=torch.float32)
    out = torch_structured._ops.diag_mult(s, v, 0, -1)
    expected = diag_mult_ref(s, v, 0, -1)
    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-6), (
        f"fp32 mismatch (backend={backend}): max err = {(out - expected).abs().max()}"
    )


def test_diag_mult_eager_complex64(backend):
    """Forward correctness vs torch_ref oracle, complex64; dtype preserved (D-20c)."""
    N, B = 128, 4
    s = torch.randn(N, device="cuda", dtype=torch.complex64)
    v = torch.randn(B, N, device="cuda", dtype=torch.complex64)
    out = torch_structured._ops.diag_mult(s, v, 0, -1)
    expected = diag_mult_ref(s, v, 0, -1)
    assert out.dtype == torch.complex64
    assert torch.allclose(out, expected, rtol=1e-4), (
        f"complex64 mismatch (backend={backend}): max err = {(out - expected).abs().max()}"
    )


def test_diag_mult_gradcheck_fp64_real(backend):
    """fp64 gradcheck — real (D-26 acceptance gate for register_autograd plumbing)."""
    N = 8
    s = torch.randn(N, dtype=torch.float64, device="cuda", requires_grad=True)
    v = torch.randn(4, N, dtype=torch.float64, device="cuda", requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, b: torch_structured._ops.diag_mult(a, b, 0, -1),
        (s, v),
        eps=1e-6,
        atol=1e-5,
    )


def test_diag_mult_gradcheck_fp64_complex(backend):
    """fp64 gradcheck — complex (D-26 Wirtinger acceptance gate).

    Fails with errors ~2.0 if ``.conj()`` is missing from the backward callback
    (verified numerically in 05-RESEARCH.md §"Backward Gradient Formula").
    """
    N = 8
    s = torch.randn(N, dtype=torch.complex128, device="cuda", requires_grad=True)
    v = torch.randn(4, N, dtype=torch.complex128, device="cuda", requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, b: torch_structured._ops.diag_mult(a, b, 0, -1),
        (s, v),
        eps=1e-6,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "shift_subdiag,shift_v", list(itertools.product([-1, 0, 1], repeat=2))
)
def test_diag_mult_shift_grid(backend, shift_subdiag, shift_v):
    """Forward + backward correctness over the {-1, 0, 1}^2 shift grid (D-29)."""
    N, B = 16, 4
    # Forward correctness (fp32)
    s = torch.randn(N, device="cuda", dtype=torch.float32)
    v = torch.randn(B, N, device="cuda", dtype=torch.float32)
    out = torch_structured._ops.diag_mult(s, v, shift_subdiag, shift_v)
    expected = diag_mult_ref(s, v, shift_subdiag, shift_v)
    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-6), (
        f"forward mismatch shift=({shift_subdiag},{shift_v}), backend={backend}: "
        f"max err = {(out - expected).abs().max()}"
    )

    # Backward grads match autograd through the torch_ref oracle (fp64 for precision)
    s64 = torch.randn(N, dtype=torch.float64, device="cuda", requires_grad=True)
    v64 = torch.randn(B, N, dtype=torch.float64, device="cuda", requires_grad=True)
    grad_out = torch.randn(B, N, dtype=torch.float64, device="cuda")

    out_op = torch_structured._ops.diag_mult(s64, v64, shift_subdiag, shift_v)
    grad_s_op, grad_v_op = torch.autograd.grad(out_op, (s64, v64), grad_outputs=grad_out)

    s64_ref = s64.detach().clone().requires_grad_(True)
    v64_ref = v64.detach().clone().requires_grad_(True)
    out_ref = diag_mult_ref(s64_ref, v64_ref, shift_subdiag, shift_v)
    grad_s_ref, grad_v_ref = torch.autograd.grad(out_ref, (s64_ref, v64_ref), grad_outputs=grad_out)

    assert torch.allclose(grad_s_op, grad_s_ref, rtol=1e-5, atol=1e-6), (
        f"grad_s mismatch shift=({shift_subdiag},{shift_v}), backend={backend}: "
        f"max err = {(grad_s_op - grad_s_ref).abs().max()}"
    )
    assert torch.allclose(grad_v_op, grad_v_ref, rtol=1e-5, atol=1e-6), (
        f"grad_v mismatch shift=({shift_subdiag},{shift_v}), backend={backend}: "
        f"max err = {(grad_v_op - grad_v_ref).abs().max()}"
    )
