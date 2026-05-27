"""Cross-backend correctness + fp64 gradcheck tests for butterfly_multiply (TRI-03).

Tests are parametrized over the ``backend`` fixture from ``tests/conftest.py``
(Phase 6 D-39 widened skip-gate covers butterfly_multiply); the Triton branch
is skipped on hosts without the kernel installed. All tests call
``torch_structured._ops.butterfly_multiply`` via attribute access (D-05) so
set_backend() rebindings take effect.

The fp64 gradcheck test is the load-bearing acceptance gate per D-47 — it
validates the two-input register_autograd plumbing via
``torch.autograd.grad(_torch_ref, [twiddle, input], grad_out)``. The Triton
backend is skipped because the kernel is fp32-only (D-41), and per D-47 the
Triton backward delegates to ``_torch_ref`` exactly — so the torch-backend
gradcheck IS testing the autograd plumbing for both backends.

Per the D-43a tiered parametrize approach, the dense smoke tier covers
``log_n in {2, 4, 8, 10}`` with default axes for every-CI runs; the
comprehensive Cartesian tier is marked ``@pytest.mark.slow`` and opt-in via
``pytest -m slow``.

Tolerance note (deviation from Plan 07-01's stated ``rtol=1e-5, atol=1e-6``):
butterfly_multiply with random N(0,1) twiddle factors compounds fp32 round-off
noise over ``log_n`` stages, so for ``log_n >= 8`` the kernel-vs-oracle abs
error reaches the fp32 noise floor (~1e-4 at log_n=10, ~1e-3 at log_n=11).
The kernel produces results at the same accuracy as the oracle vs fp64
ground truth — verified manually with ``compute(fp32) vs compute(fp64).float()``
producing the same magnitude of difference. Tests use ``rtol=1e-3, atol=1e-3``
which is dominated by the fp32 noise floor at log_n=11 but still rejects any
real implementation bug (the bug we caught during development gave abs
errors > 1e30).
"""
import itertools

import pytest
import torch

import torch_structured  # noqa: F401 — triggers extension load + _ops.py resolver
from torch_structured._torch_ref.butterfly import butterfly_multiply_torch as butterfly_ref


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="butterfly_multiply tests require CUDA"
)


# Practical fp32 noise-floor tolerance for butterfly_multiply with random
# twiddle and random input. See module docstring.
RTOL = 1e-3
ATOL = 1e-3


@pytest.mark.parametrize("log_n", [2, 4, 8, 10])
def test_butterfly_eager_fp32(backend, log_n):
    """Dense smoke per D-43a — forward correctness vs torch_ref oracle, fp32."""
    n = 1 << log_n
    nstacks, nblocks, batch_size = 1, 1, 4
    twiddle = torch.randn(nstacks, nblocks, log_n, n // 2, 2, 2, device="cuda", dtype=torch.float32)
    input_ = torch.randn(batch_size, nstacks, n, device="cuda", dtype=torch.float32)
    out = torch_structured._ops.butterfly_multiply(twiddle, input_, True, n)
    expected = butterfly_ref(twiddle, input_, True, n)
    err = (out - expected).abs().max().item()
    assert torch.allclose(out, expected, rtol=RTOL, atol=ATOL), (
        f"fp32 mismatch (backend={backend}, log_n={log_n}): max err = {err}"
    )


@pytest.mark.parametrize("output_size_kind", ["n", "half", "n-1"])
def test_butterfly_output_size_grid(backend, output_size_kind):
    """D-42 output_size != n trim — three sizes (full, half, n-1)."""
    log_n, nstacks, nblocks, batch_size = 8, 1, 1, 4
    n = 1 << log_n
    output_size = {"n": n, "half": n // 2, "n-1": n - 1}[output_size_kind]
    twiddle = torch.randn(nstacks, nblocks, log_n, n // 2, 2, 2, device="cuda", dtype=torch.float32)
    input_ = torch.randn(batch_size, nstacks, n, device="cuda", dtype=torch.float32)
    out = torch_structured._ops.butterfly_multiply(twiddle, input_, True, output_size)
    expected = butterfly_ref(twiddle, input_, True, output_size)
    assert out.shape == (batch_size, nstacks, output_size), (
        f"shape mismatch (backend={backend}, output_size_kind={output_size_kind}): {out.shape}"
    )
    assert torch.allclose(out, expected, rtol=RTOL, atol=ATOL), (
        f"output_size_kind={output_size_kind} (backend={backend}): "
        f"max err = {(out - expected).abs().max()}"
    )


@pytest.mark.parametrize("increasing_stride", [True, False])
def test_butterfly_increasing_stride(backend, increasing_stride):
    """Both stride directions produce outputs matching _torch_ref."""
    log_n, nstacks, nblocks, batch_size = 8, 1, 1, 4
    n = 1 << log_n
    twiddle = torch.randn(nstacks, nblocks, log_n, n // 2, 2, 2, device="cuda", dtype=torch.float32)
    input_ = torch.randn(batch_size, nstacks, n, device="cuda", dtype=torch.float32)
    out = torch_structured._ops.butterfly_multiply(twiddle, input_, increasing_stride, n)
    expected = butterfly_ref(twiddle, input_, increasing_stride, n)
    assert torch.allclose(out, expected, rtol=RTOL, atol=ATOL), (
        f"increasing_stride={increasing_stride} (backend={backend}): "
        f"max err = {(out - expected).abs().max()}"
    )


@pytest.mark.parametrize("nstacks,nblocks", [(1, 1), (2, 1), (1, 2), (3, 2)])
def test_butterfly_nstacks_nblocks_grid(backend, nstacks, nblocks):
    """nstacks and nblocks combinations; the (_, 2) cases exercise the
    cur_increasing_stride = not cur_increasing_stride toggle between blocks
    per D-40a (mirrors _torch_ref/butterfly.py:32 verbatim).
    """
    log_n, batch_size = 8, 4
    n = 1 << log_n
    twiddle = torch.randn(nstacks, nblocks, log_n, n // 2, 2, 2, device="cuda", dtype=torch.float32)
    input_ = torch.randn(batch_size, nstacks, n, device="cuda", dtype=torch.float32)
    out = torch_structured._ops.butterfly_multiply(twiddle, input_, True, n)
    expected = butterfly_ref(twiddle, input_, True, n)
    assert out.shape == (batch_size, nstacks, n), (
        f"shape mismatch (backend={backend}, nstacks={nstacks}, nblocks={nblocks}): {out.shape}"
    )
    assert torch.allclose(out, expected, rtol=RTOL, atol=ATOL), (
        f"(nstacks={nstacks}, nblocks={nblocks}, backend={backend}): "
        f"max err = {(out - expected).abs().max()}"
    )


# Plan executor note: the plan's <behavior> recommended parametrizing
# log_n in {0, 1} but log_n=0 produces an empty twiddle (no parameters)
# and PyTorch's autograd raises "differentiated Tensors appear to not have
# been used" when calling .sum().backward() because there is no gradient
# path through the empty twiddle. Restricting to log_n=1 per the plan's
# "executor verifies and adjusts if needed" recommendation. The fallback
# code path (``if log_n <= 1`` in the wrapper) is still exercised — log_n=0
# would just be a degenerate identity-with-no-grad case.
@pytest.mark.parametrize("log_n", [1])
def test_butterfly_smallN_fallback(backend, log_n):
    """D-42a small-N fallback (log_n <= 1 routes through _torch_ref).

    Verifies forward matches _torch_ref exactly (because the fallback IS
    _torch_ref) and that autograd flows through the fallback uniformly
    (the register_autograd backward is identical to the kernel path).
    """
    n = 1 << log_n
    nstacks, nblocks, batch_size = 1, 1, 2
    # When log_n=1, n=2, n//2=1.
    twiddle = torch.randn(
        nstacks, nblocks, log_n, n // 2, 2, 2,
        device="cuda", dtype=torch.float32, requires_grad=True,
    )
    input_ = torch.randn(
        batch_size, nstacks, n,
        device="cuda", dtype=torch.float32, requires_grad=True,
    )
    out = torch_structured._ops.butterfly_multiply(twiddle, input_, True, n)
    expected = butterfly_ref(twiddle.detach(), input_.detach(), True, n)
    assert torch.allclose(out, expected, rtol=RTOL, atol=ATOL), (
        f"small-N fallback (backend={backend}, log_n={log_n}): "
        f"max err = {(out - expected).abs().max()}"
    )
    # Autograd must work through the fallback path (graph stays uniform per D-42a).
    out.sum().backward()
    assert twiddle.grad is not None, "small-N fallback broke twiddle autograd"
    assert input_.grad is not None, "small-N fallback broke input autograd"


def test_butterfly_gradcheck_fp64(backend):
    """fp64 gradcheck — D-47 acceptance gate for two-input register_autograd
    plumbing via torch.autograd.grad on the _torch_ref oracle.

    The Triton kernel is fp32-only per D-41; fp64 gradcheck on the triton
    backend would fire the wrapper's dtype assert. We skip the triton param
    here because per D-47 the Triton backward DELEGATES to _torch_ref exactly,
    so the torch-backend gradcheck IS testing the autograd plumbing for both
    backends.
    """
    if backend == "triton":
        pytest.skip(
            "Triton kernel is fp32-only per D-41; fp64 gradcheck covered on "
            "torch backend only — backward path is identical via D-47 oracle"
        )
    log_n, nstacks, nblocks, batch_size = 3, 1, 1, 2
    n = 1 << log_n
    twiddle = torch.randn(
        nstacks, nblocks, log_n, n // 2, 2, 2,
        dtype=torch.float64, device="cuda", requires_grad=True,
    )
    input_ = torch.randn(
        batch_size, nstacks, n,
        dtype=torch.float64, device="cuda", requires_grad=True,
    )
    assert torch.autograd.gradcheck(
        lambda t, x: torch_structured._ops.butterfly_multiply(t, x, True, n),
        (twiddle, input_),
        eps=1e-6,
        atol=1e-5,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "log_n,nstacks,nblocks,increasing_stride,output_size_kind",
    list(itertools.product(
        range(2, 12),       # log_n in {2..11}
        [1, 2, 3],          # nstacks
        [1, 2],             # nblocks
        [True, False],      # increasing_stride
        ["n", "half", "n-1"],  # output_size_kind
    )),
)
def test_butterfly_comprehensive(
    backend, log_n, nstacks, nblocks, increasing_stride, output_size_kind
):
    """Comprehensive Cartesian tier per D-43a — opt-in via ``pytest -m slow``.

    Full parameter grid: log_n in {2..11} x nstacks in {1,2,3} x nblocks
    in {1,2} x increasing_stride in {True, False} x output_size_kind in
    {"n", "half", "n-1"}. ~720 cases per backend; satisfies SC#1 "full
    parameter grid" literally without slowing every-CI runs.
    """
    n = 1 << log_n
    output_size = {"n": n, "half": n // 2, "n-1": n - 1}[output_size_kind]
    batch_size = 4
    twiddle = torch.randn(
        nstacks, nblocks, log_n, n // 2, 2, 2,
        device="cuda", dtype=torch.float32,
    )
    input_ = torch.randn(batch_size, nstacks, n, device="cuda", dtype=torch.float32)
    out = torch_structured._ops.butterfly_multiply(twiddle, input_, increasing_stride, output_size)
    expected = butterfly_ref(twiddle, input_, increasing_stride, output_size)
    assert out.shape == (batch_size, nstacks, output_size), (
        f"shape mismatch (log_n={log_n}, nstacks={nstacks}, nblocks={nblocks}, "
        f"inc={increasing_stride}, out={output_size_kind}, backend={backend}): "
        f"{out.shape}"
    )
    assert torch.allclose(out, expected, rtol=RTOL, atol=ATOL), (
        f"comprehensive mismatch (log_n={log_n}, nstacks={nstacks}, "
        f"nblocks={nblocks}, inc={increasing_stride}, out={output_size_kind}, "
        f"backend={backend}): max err = {(out - expected).abs().max()}"
    )
