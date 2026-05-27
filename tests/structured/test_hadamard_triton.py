"""Cross-backend correctness + fp64 gradcheck + self-inverse + consumer-surface
tests for hadamard_transform (TRI-02; Phase 6).

Tests are parametrized over the ``backend`` fixture from ``tests/conftest.py``
(``['torch', 'triton']`` per D-39); the Triton branch is skipped on hosts
without any Triton kernel installed. Tests use D-05 attribute access through
``torch_structured._ops.hadamard_transform`` so ``set_backend`` rebindings are
visible.

Coverage map:

- ``test_hadamard_eager_fp32`` parametrized over ``log_n in {2..12}`` — ROADMAP SC#1.
- ``test_hadamard_normalize`` — D-35a `2 ** (m / 2)` scale per SC#1 normalize axis.
- ``test_hadamard_gradcheck_fp64`` — D-32 acceptance gate for register_autograd
  plumbing; skipped on the triton backend because the kernel is fp32-only
  (D-31) and the Triton backward already delegates to _torch_ref per D-32
  so the torch-backend gradcheck IS testing the same backward path.
- ``test_hadamard_self_inverse`` parametrized over ``log_n in {8, 10}`` — ROADMAP SC#2.
- ``test_hadamard_module_consumer`` via ``fastfood_multiply`` — ROADMAP SC#3 + D-34.
"""
import pytest
import torch

import torch_structured  # noqa: F401 — triggers extension load + _ops.py resolver
from torch_structured._torch_ref.hadamard import hadamard_transform_torch as hadamard_ref
from torch_structured.structured.fastfood import fastfood_multiply


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="hadamard_transform tests require CUDA"
)


@pytest.mark.parametrize("log_n", list(range(2, 13)))
def test_hadamard_eager_fp32(backend, log_n):
    """Cross-backend forward correctness vs torch_ref oracle, fp32, full log_n grid {2..12} per SC#1."""
    n = 1 << log_n
    u = torch.randn(4, n, device="cuda", dtype=torch.float32)
    out = torch_structured._ops.hadamard_transform(u, normalize=False)
    expected = hadamard_ref(u)
    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-6), (
        f"fp32 mismatch (backend={backend}, log_n={log_n}): "
        f"max err = {(out - expected).abs().max()}"
    )


def test_hadamard_normalize(backend):
    """normalize=True applies the 2**(m/2) divisor verbatim per D-35a."""
    log_n = 10
    n = 1 << log_n
    u = torch.randn(2, n, device="cuda", dtype=torch.float32)
    out_norm = torch_structured._ops.hadamard_transform(u, normalize=True)
    expected = hadamard_ref(u, normalize=True)
    assert torch.allclose(out_norm, expected, rtol=1e-5, atol=1e-6), (
        f"normalize mismatch (backend={backend}): "
        f"max err = {(out_norm - expected).abs().max()}"
    )


def test_hadamard_gradcheck_fp64(backend):
    """fp64 gradcheck — D-32 acceptance gate (backward routes through _torch_ref oracle).

    The Triton kernel is fp32-only per D-31; fp64 gradcheck on the triton
    backend would fire the wrapper's dtype assert. We skip the triton param
    here because per D-32 the Triton backward DELEGATES to _torch_ref exactly,
    so the torch-backend gradcheck IS testing the autograd plumbing for both
    backends (the register_autograd backward callback is identical).
    """
    if backend == "triton":
        pytest.skip(
            "Triton kernel is fp32-only per D-31; fp64 gradcheck covered on "
            "torch backend only — backward path is identical via D-32 oracle"
        )
    n = 8
    u = torch.randn(2, n, dtype=torch.float64, device="cuda", requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a: torch_structured._ops.hadamard_transform(a, normalize=False),
        (u,),
        eps=1e-6,
        atol=1e-5,
    )


@pytest.mark.parametrize("log_n", [8, 10])
def test_hadamard_self_inverse(backend, log_n):
    """H o H = N * I (unnormalized) or I (normalized) per ROADMAP SC#2.

    Acceptance: bit-equivalent within fp32 noise floor accumulated over 2*log_n
    butterfly stages (atol=1e-3 unnormalized, atol=1e-4 normalized).
    """
    n = 1 << log_n
    u = torch.randn(3, n, device="cuda", dtype=torch.float32)
    # Unnormalized: H @ (H @ u) == N * u
    twice = torch_structured._ops.hadamard_transform(
        torch_structured._ops.hadamard_transform(u, normalize=False),
        normalize=False,
    )
    assert torch.allclose(twice, n * u, atol=1e-3), (
        f"self-inverse unnormalized fail (backend={backend}, log_n={log_n}): "
        f"max err = {(twice - n * u).abs().max()}"
    )
    # Normalized: H @ (H @ u) == u (the 1/sqrt(N) scale squared cancels the N factor)
    twice_norm = torch_structured._ops.hadamard_transform(
        torch_structured._ops.hadamard_transform(u, normalize=True),
        normalize=True,
    )
    assert torch.allclose(twice_norm, u, atol=1e-4), (
        f"self-inverse normalized fail (backend={backend}, log_n={log_n}): "
        f"max err = {(twice_norm - u).abs().max()}"
    )


def test_hadamard_module_consumer(backend):
    """fastfood_multiply consumer-surface integration per ROADMAP SC#3 + D-34.

    Verifies that ``structured/fastfood.py`` (rewritten in Task 6) routes
    through ``torch_structured._ops.hadamard_transform`` via D-05 attribute
    access — the call-site re-reads the binding on every call, so
    set_backend rebindings (via the ``backend`` fixture) take effect inside
    fastfood_multiply.
    """
    n = 16
    B = 3
    S = torch.randn(n, device="cuda", dtype=torch.float32)
    G = torch.randn(n, device="cuda", dtype=torch.float32)
    Bdiag = torch.randn(n, device="cuda", dtype=torch.float32)
    P = torch.randperm(n, device="cuda")
    x = torch.randn(B, n, device="cuda", dtype=torch.float32)
    out = fastfood_multiply(S, G, Bdiag, P, x)
    assert out.shape == (B, n), f"shape (backend={backend}): {out.shape}"
    # Cross-backend agreement: compute expected via _torch_ref directly.
    HBx_ref = hadamard_ref(Bdiag * x)
    PHBx_ref = HBx_ref[:, P]
    HGPHBx_ref = hadamard_ref(G * PHBx_ref)
    expected = S * HGPHBx_ref
    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-5), (
        f"consumer mismatch (backend={backend}): "
        f"max err = {(out - expected).abs().max()}"
    )
