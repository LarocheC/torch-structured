"""Cross-backend correctness + fp64 gradcheck tests for butterfly_multiply (TRI-03).

Tests are parametrized over the ``backend`` fixture from ``tests/conftest.py``
(Phase 6 D-39 widened skip-gate covers butterfly_multiply); the Triton branch
is skipped on hosts without the kernel installed. All tests call
``torch_structured._ops.butterfly_multiply`` via attribute access (D-05) so
set_backend() rebindings take effect.

The fp64 gradcheck test is the load-bearing acceptance gate per D-47 — it
validates the two-input register_autograd plumbing via
``torch.autograd.grad(_torch_ref, [twiddle, input], grad_out)``. The Triton
backend is skipped because the kernel is fp32/complex64 only (D-41), and per
D-47 the Triton backward delegates to ``_torch_ref`` exactly — so the
torch-backend gradcheck IS testing the autograd plumbing for both backends.

Per the D-43a tiered parametrize approach, the dense smoke tier covers
``log_n in {2, 4, 8, 10}`` with default axes for every-CI runs; the
comprehensive Cartesian tier is marked ``@pytest.mark.slow`` and opt-in via
``pytest -m slow``.

Plan 07-02 extends the file with complex64 forward correctness, Wirtinger
fp64-equivalent (complex128) gradcheck, and the U U^* = I unitary acceptance
gate per PITFALLS §1 (the load-bearing complex-correctness detector that
fails loudly on any 4-FMA sign error in the kernel).

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
from torch_structured.butterfly import Butterfly  # legacy nn.Module surface for the unitary test (D-46 — no consumer refactor in Phase 7)


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


# ============================================================================
# Plan 07-02 additions — complex64 + Wirtinger gradcheck + unitary U U^H = I
# ============================================================================
#
# The four tests below light up the complex64 path of butterfly_multiply per
# ROADMAP Phase 7 SC#1 (complex64) + SC#2 (unitary) + D-47 Wirtinger acceptance.
# The unitary test (test_butterfly_unitary) is the load-bearing PITFALLS §1
# detector — it fails loudly on any 4-FMA sign error in the IS_COMPLEX=True
# kernel branch or any view_as_real round-trip break at the wrapper boundary.
#
# Tolerance for complex64 follows ROADMAP SC#1: ``rtol=1e-4``. fp32 round-off
# noise compounds through log_n stages so the practical tolerance has the
# same scale-awareness pattern documented in the module docstring above —
# we use ``rtol=RTOL, atol=ATOL`` (1e-3, 1e-3) for the comprehensive grid
# at log_n=11 where the noise floor dominates; the smoke tier at log_n <= 10
# uses rtol=1e-4 directly (the SC#1 literal contract).


@pytest.mark.parametrize("log_n", [2, 4, 8, 10])
def test_butterfly_eager_complex64(backend, log_n):
    """Forward correctness vs torch_ref oracle, complex64, dense-smoke parameter set per D-43a.

    Verifies the IS_COMPLEX=True kernel branch (4-FMA per-pair) produces outputs
    matching ``_torch_ref`` within ROADMAP SC#1 complex64 tolerance (rtol=1e-4).
    """
    n = 1 << log_n
    nstacks, nblocks, batch_size = 1, 1, 4
    twiddle = torch.randn(
        nstacks, nblocks, log_n, n // 2, 2, 2,
        device="cuda", dtype=torch.complex64,
    )
    input_ = torch.randn(batch_size, nstacks, n, device="cuda", dtype=torch.complex64)
    out = torch_structured._ops.butterfly_multiply(twiddle, input_, True, n)
    expected = butterfly_ref(twiddle, input_, True, n)
    assert out.dtype == torch.complex64, f"output dtype mismatch: {out.dtype}"
    err = (out - expected).abs().max().item()
    assert torch.allclose(out, expected, rtol=1e-4, atol=1e-4), (
        f"complex64 mismatch (backend={backend}, log_n={log_n}): max err = {err}"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "log_n,nstacks,nblocks,increasing_stride,output_size_kind",
    list(itertools.product(
        range(2, 12),
        [1, 2, 3],
        [1, 2],
        [True, False],
        ["n", "half", "n-1"],
    )),
)
def test_butterfly_eager_complex64_grid(
    backend, log_n, nstacks, nblocks, increasing_stride, output_size_kind
):
    """Full Cartesian complex64 parameter grid (D-43a comprehensive tier — opt-in via ``pytest -m slow``).

    Mirrors test_butterfly_comprehensive but with complex64 dtype. fp32 noise
    compounds through log_n stages of complex multiply; the practical
    tolerance ``RTOL=ATOL=1e-3`` (module-level) accommodates the noise floor
    at log_n=11 while still rejecting any real implementation bug.
    """
    n = 1 << log_n
    output_size = {"n": n, "half": n // 2, "n-1": n - 1}[output_size_kind]
    batch_size = 4
    twiddle = torch.randn(
        nstacks, nblocks, log_n, n // 2, 2, 2,
        device="cuda", dtype=torch.complex64,
    )
    input_ = torch.randn(batch_size, nstacks, n, device="cuda", dtype=torch.complex64)
    out = torch_structured._ops.butterfly_multiply(twiddle, input_, increasing_stride, output_size)
    expected = butterfly_ref(twiddle, input_, increasing_stride, output_size)
    assert out.shape == (batch_size, nstacks, output_size), (
        f"shape mismatch (log_n={log_n}, nstacks={nstacks}, nblocks={nblocks}, "
        f"inc={increasing_stride}, out={output_size_kind}, backend={backend}): "
        f"{out.shape}"
    )
    assert torch.allclose(out, expected, rtol=RTOL, atol=ATOL), (
        f"complex64 comprehensive (log_n={log_n}, nstacks={nstacks}, "
        f"nblocks={nblocks}, inc={increasing_stride}, out={output_size_kind}, "
        f"backend={backend}): max err = {(out - expected).abs().max()}"
    )


def test_butterfly_gradcheck_complex64(backend):
    """fp64-equivalent complex128 gradcheck — D-47 Wirtinger acceptance for complex.

    The triton backend is skipped: the kernel itself is fp32/complex64; gradcheck
    demands fp64/complex128 precision. Per D-47, the backward delegates to
    ``_torch_ref.butterfly_multiply_torch`` via ``torch.autograd.grad(...)`` —
    the torch-backend gradcheck IS testing the register_autograd plumbing both
    backends rely on. ``torch.autograd.grad`` natively handles Wirtinger
    gradients for complex inputs; no manual ``.conj()`` correction is needed
    (Phase 5 ``diag_mult`` used a hand-rolled Wirtinger formula and had to add
    ``.conj()`` manually; Phase 7 delegates the entire gradient to autograd
    inside the oracle's execution).
    """
    if backend == "triton":
        pytest.skip(
            "Triton kernel is fp32/complex64 only; gradcheck on torch backend "
            "exercises the same register_autograd backward (D-47)"
        )
    log_n, nstacks, nblocks, batch_size = 3, 1, 1, 2
    n = 1 << log_n
    twiddle = torch.randn(
        nstacks, nblocks, log_n, n // 2, 2, 2,
        dtype=torch.complex128, device="cuda", requires_grad=True,
    )
    input_ = torch.randn(
        batch_size, nstacks, n,
        dtype=torch.complex128, device="cuda", requires_grad=True,
    )
    assert torch.autograd.gradcheck(
        lambda t, x: torch_structured._ops.butterfly_multiply(t, x, True, n),
        (twiddle, input_),
        eps=1e-6,
        atol=1e-5,
    )


def test_butterfly_unitary(backend):
    """U U^H = I — the load-bearing complex-correctness detector per PITFALLS §1.

    Constructs a ``Butterfly(complex=True, init='ortho')`` module whose
    twiddle is initialized so each 2x2 factor is unitary; the product is
    unitary by composition. Materializes the n x n matrix U by passing the
    n x n identity matrix through the kernel as n parallel inputs (batch=n,
    nstacks=1), then asserts ``U @ U.conj().T ≈ I`` at atol=1e-4.

    Per PITFALLS §1 this is the CHEAPEST correctness gate for the complex
    path — it fails immediately on any 4-FMA sign error in the
    IS_COMPLEX=True kernel branch or any view_as_real round-trip break at
    the wrapper boundary.

    Runs on both backends. The test calls ``torch_structured._ops.butterfly_multiply``
    DIRECTLY (not via ``Butterfly.forward``) because per D-46 the
    ``Butterfly`` nn.Module's forward routes through the legacy C++ op
    (``torch.ops.torch_structured.butterfly_multiply``) — to actually
    exercise the Triton kernel under the ``triton`` backend, we must hit
    ``_ops.butterfly_multiply`` directly.
    """
    log_n = 4
    n = 1 << log_n
    # Construct a unitary butterfly via init='ortho' (butterfly.py:71-90
    # initializes each 2x2 factor as a Haar-unitary 2x2 matrix; the product
    # over log_n stages is unitary by composition).
    b = Butterfly(
        in_size=n, out_size=n, bias=False, complex=True,
        increasing_stride=True, init='ortho',
    ).to("cuda")
    twiddle = b.twiddle  # (nstacks=1, nblocks=1, log_n, n//2, 2, 2) complex64

    # Materialize U column-by-column: feed the n x n identity matrix as a
    # batch of n inputs, where input i is the i-th column of the identity.
    # Butterfly expects (batch, nstacks, in_size). batch=n, nstacks=1,
    # in_size=n. Output is (n, 1, n) where row i is U @ e_i = column i of U.
    identity_batch = torch.eye(n, dtype=torch.complex64, device="cuda").unsqueeze(1)
    assert identity_batch.shape == (n, 1, n)
    with torch.no_grad():
        outputs = torch_structured._ops.butterfly_multiply(twiddle, identity_batch, True, n)
    U = outputs.squeeze(1).T.contiguous()  # (n, n)
    assert U.shape == (n, n), f"materialized U shape mismatch (backend={backend}): {U.shape}"
    assert U.dtype == torch.complex64, f"U dtype mismatch (backend={backend}): {U.dtype}"

    # The load-bearing unitarity gate (PITFALLS §1).
    UUH = U @ U.conj().T
    eye = torch.eye(n, dtype=torch.complex64, device="cuda")
    max_err = (UUH - eye).abs().max().item()
    assert torch.allclose(UUH, eye, atol=1e-4), (
        f"Unitary check FAILED (backend={backend}): max |U U^H - I| = {max_err}"
    )
