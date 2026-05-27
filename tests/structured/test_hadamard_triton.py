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


def test_hadamard_eager_fp32_rank3(backend):
    """Rank-3 forward correctness — closes SC#2 strict-reading gap (06-VERIFICATION.md).

    The Triton wrapper at _triton/hadamard_transform/op.py:104 advertises
    shape (*batch, n); this test exercises a rank-3 input through both
    backends and asserts the result equals the rank-2 reshape baseline
    element-wise.
    """
    n = 16  # log_n = 4 — small enough for tight tolerances, large enough to fire on sign bugs
    u = torch.randn(3, 2, n, device="cuda", dtype=torch.float32)
    out = torch_structured._ops.hadamard_transform(u, normalize=False)
    assert out.shape == (3, 2, n), (
        f"rank-3 shape (backend={backend}): expected (3, 2, {n}), got {tuple(out.shape)}"
    )
    # The rank-N path MUST equal the rank-2 reshape baseline element-wise.
    expected = torch_structured._ops.hadamard_transform(
        u.reshape(-1, n), normalize=False
    ).reshape(3, 2, n)
    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-6), (
        f"rank-3 vs reshape baseline mismatch (backend={backend}): "
        f"max err = {(out - expected).abs().max()}"
    )
    # Also cross-check against the _torch_ref oracle on the rank-3 input directly.
    ref_out = hadamard_ref(u)
    assert torch.allclose(out, ref_out, rtol=1e-5, atol=1e-6), (
        f"rank-3 vs oracle mismatch (backend={backend}): "
        f"max err = {(out - ref_out).abs().max()}"
    )


def test_hadamard_backward_rank3(backend):
    """Rank-3 backward succeeds — closes SC#2 strict-reading gap.

    Before the fix this raised ``ValueError: too many values to unpack
    (expected 2)`` on BOTH backends — on triton because
    _triton/hadamard_transform/op.py:168 delegates backward through
    _torch_ref.hadamard.hadamard_transform_torch; on torch because the
    forward itself was rank-2-only.
    """
    n = 16
    u = torch.randn(3, 2, n, device="cuda", dtype=torch.float32, requires_grad=True)
    out = torch_structured._ops.hadamard_transform(u, normalize=False)
    (grad_u,) = torch.autograd.grad(out.sum(), u)
    assert grad_u.shape == (3, 2, n), (
        f"rank-3 grad shape (backend={backend}): "
        f"expected (3, 2, {n}), got {tuple(grad_u.shape)}"
    )
    # Numerical correctness: cross-check via the rank-2 reshape baseline.
    u2 = u.detach().reshape(-1, n).requires_grad_(True)
    out2 = torch_structured._ops.hadamard_transform(u2, normalize=False)
    (grad_u2,) = torch.autograd.grad(out2.sum(), u2)
    expected_grad = grad_u2.reshape(3, 2, n)
    assert torch.allclose(grad_u, expected_grad, rtol=1e-5, atol=1e-6), (
        f"rank-3 grad vs reshape baseline mismatch (backend={backend}): "
        f"max err = {(grad_u - expected_grad).abs().max()}"
    )


def test_hadamard_self_inverse_rank3(backend):
    """H o H = N * I (unnormalized) or I (normalized) on a rank-3 input.

    Direct ROADMAP SC#2 "any input shape" coverage — the existing
    test_hadamard_self_inverse is rank-2-only; this lifts the contract
    to (*batch, n) for both backends.
    """
    n = 16
    u = torch.randn(3, 2, n, device="cuda", dtype=torch.float32)
    # Unnormalized: H @ (H @ u) == N * u
    twice = torch_structured._ops.hadamard_transform(
        torch_structured._ops.hadamard_transform(u, normalize=False),
        normalize=False,
    )
    assert torch.allclose(twice, n * u, atol=1e-3), (
        f"rank-3 self-inverse unnormalized fail (backend={backend}): "
        f"max err = {(twice - n * u).abs().max()}"
    )
    # Normalized: H @ (H @ u) == u
    twice_norm = torch_structured._ops.hadamard_transform(
        torch_structured._ops.hadamard_transform(u, normalize=True),
        normalize=True,
    )
    assert torch.allclose(twice_norm, u, atol=1e-4), (
        f"rank-3 self-inverse normalized fail (backend={backend}): "
        f"max err = {(twice_norm - u).abs().max()}"
    )
