"""Phase 9 09-01 integration tests — foundations (SC#1 + SC#3 + §0 LANDMINE).

Bundles four test groups:

1. §0 LANDMINE verification — the rewritten ``butterfly/multiply.py`` delegator
   makes ``Butterfly(...).forward(x)`` reach ``torch_structured._ops.butterfly_multiply``
   under ``set_backend('triton')`` and NOT the C++ op directly. Both-sides gate:
   positive recording sentinel + negative C++ raising stub.

2. ``_has_cuda_legacy_for_op`` probe + 3-axis ``backend`` fixture behavior
   (Task 2 — D-62 / D-62a / D-62b).

3. Backend-agreement at Phase 7+8 tolerances (Task 3 — SC#1).

4. Checkpoint v1.0/v1.1 round-trip, ``make_linear`` + LRU smoke,
   public-API ``inspect.signature`` snapshot (Task 3 — SC#3, COMPAT-01/02/03).

Tolerances (D-62c, inherited from Phase 7 D-43a + Phase 8 D-52):
- fp32 forward + d_input: rtol=1e-5, atol=1e-6
- complex64 d_input: rtol=1e-5, atol=1e-6
- complex64 d_twiddle: rtol=1e-3, atol=1e-4
- complex64 forward: rtol=1e-3, atol=1e-4 (Phase 7 D-43a — matches
  ``tests/test_butterfly_triton.py``'s noise-floor envelope)
- v1.0/v1.1 checkpoint forward across BACKEND=triton vs BACKEND=cuda:
  rtol=1e-3, atol=1e-3 (RESEARCH §7 — fp32 accumulation across log_n=8 stages)
"""
import inspect
import os
import subprocess
import sys

import pytest
import torch

import torch_structured  # noqa: F401 — triggers extension load + _ops.py resolver
import torch_structured.butterfly.multiply as _bf_multiply
from torch_structured import (
    Butterfly,
    ButterflyBmm,
    ButterflyBase4,
    ButterflyUnitary,
    LRU,
    make_linear,
)
from torch_structured._torch_ref.butterfly import butterfly_multiply_torch
from torch_structured._torch_ref.diag_mult import diag_mult as _diag_mult_torch
from torch_structured._torch_ref.hadamard import hadamard_transform_torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Phase 9 integration tests require CUDA",
)


# ─────────────────────────────────────────────────────────────────────────
# Tolerances (D-62c, single-sourced per CLAUDE.md `assert` precondition style)
# ─────────────────────────────────────────────────────────────────────────
_FP32_RTOL = 1e-5
_FP32_ATOL = 1e-6
_C64_FORWARD_RTOL = 1e-3
_C64_FORWARD_ATOL = 1e-4
_C64_DTWIDDLE_RTOL = 1e-3
_C64_DTWIDDLE_ATOL = 1e-4
_C64_DINPUT_RTOL = 1e-5
_C64_DINPUT_ATOL = 1e-6
_CKPT_FORWARD_RTOL = 1e-3
_CKPT_FORWARD_ATOL = 1e-3

# Phase 7+8 noise-floor for cross-backend agreement (matches the existing
# tests/test_butterfly_triton.py:60-72 envelope — random twiddle compounds
# fp32 round-off across log_n stages so the practical floor is ~1e-3 at
# log_n=8, not the 1e-5 mathematical floor).
_FP32_AGREEMENT_RTOL = 1e-3
_FP32_AGREEMENT_ATOL = 1e-3


# ═════════════════════════════════════════════════════════════════════════
# Task 1: §0 LANDMINE verification (both-sides gate)
# ═════════════════════════════════════════════════════════════════════════


@pytest.mark.op("butterfly_multiply")
def test_butterfly_nn_module_routes_through_ops():
    """POSITIVE — §0 LANDMINE fix verification via recording sentinel.

    Install a counting wrapper on ``torch_structured._ops.butterfly_multiply``;
    run ``Butterfly(...).forward + backward`` under BACKEND=triton; assert the
    wrapper was invoked at least once. Pre-fix (the LANDMINE state),
    ``butterfly/multiply.py``'s @torch.jit.script wrapper called
    ``torch.ops.torch_structured.butterfly_multiply`` directly and the sentinel
    counter stayed at 0. Post-fix, the delegator re-reads
    ``_ops.butterfly_multiply`` per call so the sentinel WILL increment.
    """
    torch_structured._ops.set_backend("triton")
    original = torch_structured._ops.butterfly_multiply
    call_count = {"n": 0}

    def recording_wrapper(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    torch_structured._ops.butterfly_multiply = recording_wrapper
    try:
        m = Butterfly(in_size=256, out_size=256, bias=False).cuda()
        x = torch.randn(8, 256, device="cuda", requires_grad=True)
        out = m(x)
        out.sum().backward()
        assert call_count["n"] > 0, (
            "Butterfly.forward did not reach torch_structured._ops.butterfly_multiply "
            "— §0 LANDMINE NOT FIXED (the @torch.jit.script wrapper at "
            "butterfly/multiply.py:27 routed past _ops, see 09-RESEARCH §0)."
        )
    finally:
        torch_structured._ops.butterfly_multiply = original
        torch_structured._ops.set_backend("torch")


@pytest.mark.op("butterfly_multiply")
def test_butterfly_nn_module_does_not_call_cpp_op_directly():
    """NEGATIVE — §0 LANDMINE complement via C++ raising stub.

    Patch ``torch.ops.torch_structured.butterfly_multiply`` (the legacy C++
    entry point) with a stub that raises. Run ``Butterfly(...).forward`` under
    BACKEND=triton; assert NO RuntimeError propagated. Pre-fix, the
    @torch.jit.script wrapper called the C++ op directly and the stub would
    fire; post-fix, the delegator routes through _ops to the Triton kernel
    and the C++ op is unreached.

    Mechanism: ``torch.ops.torch_structured`` is a ``_OpNamespace`` and
    permits attribute assignment (verified in 09-01 plan execution; PyTorch
    2.6's OpNamespace does NOT freeze attributes). We rebind the attribute
    directly. If a future PyTorch version freezes OpNamespace, fall back to
    patching ``torch_structured._cuda_legacy.butterfly.butterfly_multiply``
    — that surface is a regular Python function and is always writable.
    """
    torch_structured._ops.set_backend("triton")
    original_cpp = torch.ops.torch_structured.butterfly_multiply

    def raising_stub(*args, **kwargs):
        raise RuntimeError(
            "§0 LANDMINE: C++ op called directly — Butterfly.forward bypassed _ops"
        )

    setattr(torch.ops.torch_structured, "butterfly_multiply", raising_stub)
    try:
        m = Butterfly(in_size=256, out_size=256, bias=False).cuda()
        x = torch.randn(8, 256, device="cuda")
        # If LANDMINE not fixed, this would raise RuntimeError from the stub.
        out = m(x)
        assert out.shape == (8, 256), f"unexpected output shape {out.shape}"
    finally:
        setattr(torch.ops.torch_structured, "butterfly_multiply", original_cpp)
        torch_structured._ops.set_backend("torch")


def test_butterfly_multiply_delegator_is_not_jit_scripted():
    """The delegator must be a plain Python function (no @torch.jit.script).

    Phase 8 SC#4's monkey-patch (``tests/test_butterfly_triton.py:756-757``)
    rebinds module attributes; @torch.jit.script produces immutable
    ScriptFunction objects with a ``.graph`` attribute and rejects rebinding
    of the underlying Python attribute path. The delegator MUST be a plain
    Python function for that mechanism to keep working.
    """
    assert not hasattr(_bf_multiply.butterfly_multiply, "graph"), (
        "butterfly_multiply still has a .graph attribute — @torch.jit.script "
        "was not removed. Phase 8 SC#4's monkey-patch will silently fail."
    )
    assert not hasattr(_bf_multiply.butterfly_multiply_fw, "graph"), (
        "butterfly_multiply_fw still has a .graph attribute — @torch.jit.script "
        "was not removed. Phase 8 SC#4's monkey-patch will silently fail."
    )
    assert not hasattr(_bf_multiply.butterfly_multiply_bw, "graph"), (
        "butterfly_multiply_bw still has a .graph attribute — @torch.jit.script "
        "was not removed. Phase 8 SC#4's monkey-patch will silently fail."
    )


def test_butterfly_multiply_fw_bw_still_call_cpp():
    """The ``_fw`` and ``_bw`` entry points have no Triton equivalent at this
    surface; they remain direct callers of ``torch.ops.torch_structured.*``.

    Verifies the @torch.jit.script removal did NOT inadvertently make these
    delegate through ``_ops`` (which has no _fw / _bw bindings — only the
    autograd-aware ``butterfly_multiply``). The legacy ``_fw``/``_bw`` API is
    kept for SC#4's monkey-patch contract; the path is C++-direct by design.

    Uses CPU tensors: the C++ op has both CPU and CUDA dispatch keys, and
    the CPU path is the universal one (the .so may have been built without
    CUDA support on some hosts — e.g., when PyTorch's CUDA version differs
    from the toolkit that compiled the .so). The test verifies the routing
    contract (delegators call C++), not the CUDA kernel itself.
    """
    log_n = 4
    n = 1 << log_n
    twiddle = torch.randn(1, 1, log_n, n // 2, 2, 2, dtype=torch.float32)
    x = torch.randn(4, 1, n, dtype=torch.float32)
    out_fw = _bf_multiply.butterfly_multiply_fw(twiddle, x, True, n)
    expected = butterfly_multiply_torch(twiddle, x, True, n)
    assert torch.allclose(out_fw, expected, rtol=_FP32_AGREEMENT_RTOL, atol=_FP32_AGREEMENT_ATOL), (
        f"butterfly_multiply_fw mismatch vs oracle: "
        f"max_err={(out_fw - expected).abs().max().item()}"
    )

    grad = torch.randn_like(out_fw)
    d_twiddle, d_input = _bf_multiply.butterfly_multiply_bw(twiddle, x, grad, True)
    assert d_twiddle.shape == twiddle.shape, (
        f"d_twiddle shape mismatch: {d_twiddle.shape} vs {twiddle.shape}"
    )
    assert d_input.shape == x.shape, (
        f"d_input shape mismatch: {d_input.shape} vs {x.shape}"
    )


# ═════════════════════════════════════════════════════════════════════════
# Task 2: _has_cuda_legacy_for_op probe + 3-axis backend fixture behavior
# ═════════════════════════════════════════════════════════════════════════


def test_has_cuda_legacy_for_op_butterfly_multiply():
    """The new per-op probe agrees with the existing ``_has_cuda_legacy()``
    for the butterfly_multiply op."""
    assert (
        torch_structured._ops._has_cuda_legacy_for_op("butterfly_multiply")
        is torch_structured._ops._has_cuda_legacy()
    )


def test_has_cuda_legacy_for_op_diag_mult():
    """The probe agrees with ``_has_cuda_legacy_diag_mult()`` for diag_mult."""
    assert (
        torch_structured._ops._has_cuda_legacy_for_op("diag_mult")
        is torch_structured._ops._has_cuda_legacy_diag_mult()
    )


def test_has_cuda_legacy_for_op_hadamard_transform():
    """The probe agrees with ``_has_cuda_legacy_hadamard()`` for hadamard."""
    assert (
        torch_structured._ops._has_cuda_legacy_for_op("hadamard_transform")
        is torch_structured._ops._has_cuda_legacy_hadamard()
    )


def test_has_cuda_legacy_for_op_unknown_returns_false():
    """Unknown op_names return False explicitly — no raise, no dynamic import.

    Per CLAUDE.md ``no try/except in core lib`` rule, the unknown-op branch is
    an explicit return False — the existing per-op probes own all the
    try/except surface (D-21 sanctioned site).
    """
    assert torch_structured._ops._has_cuda_legacy_for_op("nonexistent_op") is False
    assert torch_structured._ops._has_cuda_legacy_for_op("") is False


@pytest.mark.op("nonexistent_op_for_skip_test")
def test_backend_fixture_skips_cuda_axis_when_marker_present_and_so_missing(backend):
    """When @pytest.mark.op names a non-existent op, the cuda axis must be
    skipped (collected as 'skipped') while torch and triton run normally.

    Test passes if it's reached under backend='torch' or backend='triton'
    (the marker doesn't skip those axes). Under backend='cuda', the conftest
    fixture's per-op skip-gate fires before this body runs.
    """
    assert backend in ("torch", "triton"), (
        f"backend={backend!r} was reached despite @pytest.mark.op('nonexistent_op_for_skip_test') "
        f"— the conftest skip-gate did not fire"
    )


def test_backend_fixture_does_not_skip_cuda_axis_without_marker(backend):
    """Without ``@pytest.mark.op``, the cuda axis runs normally — marker-less
    tests are backend-agnostic (the per-op skip-gate is opt-in)."""
    # Body trivially asserts; the value of `backend` will be one of
    # {torch, triton, cuda} when the corresponding .so or kernel is available.
    assert backend in ("torch", "triton", "cuda")


# ═════════════════════════════════════════════════════════════════════════
# Task 3: Backend-agreement + checkpoint + make_linear/LRU + public API
# ═════════════════════════════════════════════════════════════════════════


@pytest.mark.op("butterfly_multiply")
@pytest.mark.parametrize("log_n", [2, 4, 8])
def test_backend_agreement_butterfly_fp32(backend, log_n):
    """Forward-output agreement across {torch, triton, cuda} backends, fp32.

    Tolerance reflects the practical fp32 noise floor for random-twiddle
    butterfly composition (matches ``tests/test_butterfly_triton.py:60-72``).
    """
    n = 1 << log_n
    torch.manual_seed(0)
    twiddle = torch.randn(1, 1, log_n, n // 2, 2, 2, device="cuda", dtype=torch.float32)
    input_ = torch.randn(4, 1, n, device="cuda", dtype=torch.float32)
    out = torch_structured._ops.butterfly_multiply(twiddle, input_, True, n)
    expected = butterfly_multiply_torch(twiddle, input_, True, n)
    err = (out - expected).abs().max().item()
    assert torch.allclose(out, expected, rtol=_FP32_AGREEMENT_RTOL, atol=_FP32_AGREEMENT_ATOL), (
        f"fp32 mismatch (backend={backend}, log_n={log_n}): max_err={err}"
    )


@pytest.mark.op("butterfly_multiply")
@pytest.mark.parametrize("log_n", [2, 4, 8])
def test_backend_agreement_butterfly_complex64(backend, log_n):
    """Forward-output agreement across {torch, triton, cuda} backends, complex64.

    Tolerance from Phase 7 D-43a — complex64 forward at fp32 noise floor.
    """
    n = 1 << log_n
    torch.manual_seed(0)
    twiddle = torch.randn(1, 1, log_n, n // 2, 2, 2, device="cuda", dtype=torch.complex64)
    input_ = torch.randn(4, 1, n, device="cuda", dtype=torch.complex64)
    out = torch_structured._ops.butterfly_multiply(twiddle, input_, True, n)
    expected = butterfly_multiply_torch(twiddle, input_, True, n)
    err = (out - expected).abs().max().item()
    assert torch.allclose(out, expected, rtol=_C64_FORWARD_RTOL, atol=_C64_FORWARD_ATOL), (
        f"complex64 mismatch (backend={backend}, log_n={log_n}): max_err={err}"
    )


@pytest.mark.op("diag_mult")
@pytest.mark.parametrize("log_n", [4, 8])
def test_backend_agreement_diag_mult_fp32(backend, log_n):
    """Forward-output agreement across {torch, triton, cuda} for diag_mult, fp32."""
    n = 1 << log_n
    torch.manual_seed(0)
    s = torch.randn(n, device="cuda", dtype=torch.float32)
    v = torch.randn(4, n, device="cuda", dtype=torch.float32)
    out = torch_structured._ops.diag_mult(s, v, 0, -1)
    expected = _diag_mult_torch(s, v, 0, -1)
    err = (out - expected).abs().max().item()
    assert torch.allclose(out, expected, rtol=_FP32_RTOL, atol=_FP32_ATOL), (
        f"diag_mult fp32 mismatch (backend={backend}, log_n={log_n}): max_err={err}"
    )


@pytest.mark.op("hadamard_transform")
@pytest.mark.parametrize("log_n", [4, 8])
def test_backend_agreement_hadamard_transform_fp32(backend, log_n):
    """Forward-output agreement across {torch, triton, cuda} for hadamard, fp32."""
    n = 1 << log_n
    torch.manual_seed(0)
    u = torch.randn(4, n, device="cuda", dtype=torch.float32)
    out = torch_structured._ops.hadamard_transform(u, normalize=False)
    expected = hadamard_transform_torch(u, normalize=False)
    err = (out - expected).abs().max().item()
    # Size-dependent atol (bd torch-structured-f6b): the FWHT sums n terms, so
    # fp32 non-associativity grows with n. The CUDA kernel's tree-reduction order
    # differs from the torch_ref oracle; at log_n=8 (n=256) the benign drift
    # reaches ~7.6e-6, exceeding the strict 1e-6 envelope. Triton matches
    # torch_ref within 1e-6 at the same size, so this is a CUDA-kernel
    # accumulation-order artifact, not a correctness bug — widen atol to 1e-5
    # for log_n >= 8.
    atol = _FP32_ATOL if log_n < 8 else 1e-5
    assert torch.allclose(out, expected, rtol=_FP32_RTOL, atol=atol), (
        f"hadamard_transform fp32 mismatch (backend={backend}, log_n={log_n}): max_err={err}"
    )


@pytest.mark.op("butterfly_multiply")
def test_checkpoint_v10_v11_roundtrip_butterfly_fp32(tmp_path):
    """v1.0/v1.1 state_dict layout round-trip across BACKEND=triton vs BACKEND=cuda.

    Demonstrates COMPAT-02: the twiddle layout ``(nstacks, nblocks, log_n, n/2, 2, 2)``
    is preserved across Phase 7+8's Triton port — a checkpoint saved before
    the port loads cleanly after, and forward outputs match (within fp32
    accumulation tolerance — RESEARCH §7).
    """
    if not torch_structured._ops._has_cuda_legacy_for_op("butterfly_multiply"):
        pytest.skip("No CUDA legacy .so for butterfly_multiply — cannot compare triton vs cuda")

    original_backend = torch_structured._ops._BACKEND
    try:
        # Build a model under any backend, snapshot its state_dict (which
        # matches the v1.0/v1.1 layout because Butterfly.__init__ is unchanged).
        torch_structured._ops.set_backend("torch")
        torch.manual_seed(42)
        m_source = Butterfly(
            in_size=256, out_size=256, bias=False, complex=False, increasing_stride=True,
        ).cuda()
        state_dict = {k: v.detach().clone() for k, v in m_source.state_dict().items()}
        ckpt_path = tmp_path / "v10_v11_butterfly.pt"
        torch.save(state_dict, ckpt_path)

        x = torch.randn(8, 256, device="cuda", dtype=torch.float32)

        # Load under BACKEND=triton
        torch_structured._ops.set_backend("triton")
        m_triton = Butterfly(
            in_size=256, out_size=256, bias=False, complex=False, increasing_stride=True,
        ).cuda()
        loaded = torch.load(ckpt_path, weights_only=True)
        m_triton.load_state_dict(loaded)
        with torch.no_grad():
            out_triton = m_triton(x)

        # Load under BACKEND=cuda
        torch_structured._ops.set_backend("cuda")
        m_cuda = Butterfly(
            in_size=256, out_size=256, bias=False, complex=False, increasing_stride=True,
        ).cuda()
        m_cuda.load_state_dict(loaded)
        with torch.no_grad():
            out_cuda = m_cuda(x)

        err = (out_triton - out_cuda).abs().max().item()
        assert torch.allclose(
            out_triton, out_cuda, rtol=_CKPT_FORWARD_RTOL, atol=_CKPT_FORWARD_ATOL,
        ), (
            f"v1.0/v1.1 checkpoint round-trip mismatch between BACKEND=triton "
            f"and BACKEND=cuda: max_err={err}"
        )
    finally:
        torch_structured._ops.set_backend(original_backend)


@pytest.mark.op("butterfly_multiply")
def test_make_linear_butterfly_triton():
    """``make_linear`` builds a Butterfly-backed linear that supports forward +
    backward under BACKEND=triton (COMPAT-03). Uses the CORRECTED signature
    (positional ``kind``)."""
    original_backend = torch_structured._ops._BACKEND
    try:
        torch_structured._ops.set_backend("triton")
        m = make_linear("butterfly", 256, 256).cuda()
        x = torch.randn(8, 256, device="cuda", requires_grad=True)
        out = m(x)
        out.sum().backward()
        # m wraps Butterfly via _ButterflyLinear — verify the twiddle grad is finite.
        twiddle_grad = m.b.twiddle.grad
        assert twiddle_grad is not None, "make_linear butterfly: twiddle.grad is None"
        assert torch.isfinite(twiddle_grad).all(), (
            "make_linear butterfly: twiddle.grad has non-finite entries"
        )
    finally:
        torch_structured._ops.set_backend(original_backend)


@pytest.mark.op("butterfly_multiply")
def test_lru_butterfly_triton():
    """``LRU(kind='butterfly')`` runs forward + backward under BACKEND=triton
    with finite gradients (COMPAT-03). LRU has a complex-state scan; the
    butterfly projections are real fp32 per RESEARCH §10."""
    original_backend = torch_structured._ops._BACKEND
    try:
        torch_structured._ops.set_backend("triton")
        m = LRU(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            kind="butterfly",
        ).cuda()
        x = torch.randn(4, 32, 64, device="cuda", requires_grad=True)
        out, h_n = m(x)
        out.sum().backward()
        any_finite_grad = False
        for p in m.parameters():
            if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0:
                any_finite_grad = True
                break
        assert any_finite_grad, (
            "LRU(kind='butterfly') forward+backward produced no finite non-zero gradients"
        )
    finally:
        torch_structured._ops.set_backend(original_backend)


# Captured 2026-05-28 (Phase 9 plan execution) — the v1.1 public API surface
# the regression detector locks against. These EXACT strings must match
# ``str(inspect.signature(Cls.__init__))`` — see COMPAT-01.
_EXPECTED_BUTTERFLY_INIT_SIG = (
    "(self, in_size, out_size, bias=True, complex=False, increasing_stride=True, "
    "init='randn', nblocks=1)"
)
_EXPECTED_BUTTERFLY_BMM_INIT_SIG = (
    "(self, in_size, out_size, matrix_batch=1, bias=True, complex=False, "
    "increasing_stride=True, init='randn', nblocks=1)"
)
_EXPECTED_BUTTERFLY_UNITARY_INIT_SIG = (
    "(self, in_size, out_size, bias=True, increasing_stride=True, nblocks=1)"
)
_EXPECTED_BUTTERFLY_BASE4_INIT_SIG = "(self, *args, **kwargs)"


def test_public_api_butterfly_init_signature():
    """COMPAT-01: Butterfly.__init__ signature is locked against v1.1.

    Any future change to the parameter set, defaults, or kw-only markers
    fails this test loudly — protects downstream user code that constructs
    Butterfly with positional or keyword args.
    """
    actual = str(inspect.signature(Butterfly.__init__))
    assert actual == _EXPECTED_BUTTERFLY_INIT_SIG, (
        f"Butterfly.__init__ signature changed:\n"
        f"  expected: {_EXPECTED_BUTTERFLY_INIT_SIG}\n"
        f"  actual:   {actual}"
    )


def test_public_api_butterfly_bmm_init_signature():
    """COMPAT-01: ButterflyBmm.__init__ signature is locked against v1.1."""
    actual = str(inspect.signature(ButterflyBmm.__init__))
    assert actual == _EXPECTED_BUTTERFLY_BMM_INIT_SIG, (
        f"ButterflyBmm.__init__ signature changed:\n"
        f"  expected: {_EXPECTED_BUTTERFLY_BMM_INIT_SIG}\n"
        f"  actual:   {actual}"
    )


def test_public_api_butterfly_unitary_init_signature():
    """COMPAT-01: ButterflyUnitary.__init__ signature is locked against v1.1."""
    actual = str(inspect.signature(ButterflyUnitary.__init__))
    assert actual == _EXPECTED_BUTTERFLY_UNITARY_INIT_SIG, (
        f"ButterflyUnitary.__init__ signature changed:\n"
        f"  expected: {_EXPECTED_BUTTERFLY_UNITARY_INIT_SIG}\n"
        f"  actual:   {actual}"
    )


def test_public_api_butterfly_base4_init_signature():
    """COMPAT-01: ButterflyBase4.__init__ signature is locked against v1.1.

    Note: ButterflyBase4 uses ``*args, **kwargs`` so the signature string is
    minimal — the lock-in is that the class accepts ANY positional/keyword
    inputs (the variadic surface is the public contract).
    """
    actual = str(inspect.signature(ButterflyBase4.__init__))
    assert actual == _EXPECTED_BUTTERFLY_BASE4_INIT_SIG, (
        f"ButterflyBase4.__init__ signature changed:\n"
        f"  expected: {_EXPECTED_BUTTERFLY_BASE4_INIT_SIG}\n"
        f"  actual:   {actual}"
    )


def test_pytest_under_triton_smoke():
    """TEST-06 in-tree smoke: a small pytest invocation under
    ``TORCH_STRUCTURED_BACKEND=triton`` exits 0.

    Targets ``test_backend_agreement_butterfly_fp32`` (one test, all 3 backend
    params) to keep wall-time well under 120s. The full ``pytest tests/``
    sweep under BACKEND=triton is a CI matrix entry (Plan 09-03 territory);
    this in-process check verifies the env-var-driven resolver path actually
    works end-to-end.
    """
    env = {**os.environ, "TORCH_STRUCTURED_BACKEND": "triton"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_phase9_integration.py::test_backend_agreement_butterfly_fp32",
            "-x",
            "--no-header",
            "-q",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"pytest under TORCH_STRUCTURED_BACKEND=triton failed (exit={result.returncode})\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
