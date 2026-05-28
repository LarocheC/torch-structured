"""Phase 9 09-02 deterministic mode tests (D-63 + D-63a + D-63b + D-63c).

Verifies the ``set_deterministic`` opt-in API:

- ``set_deterministic(True)`` mirrors ``set_backend`` save/restore semantics
  (returns the PREVIOUS value, not the new one — D-63).
- The wrapper-level oracle fallback in ``_triton/butterfly/op.py:_backward``
  routes through the pure-PyTorch oracle when EITHER
  ``_ops._DETERMINISTIC`` OR ``torch.are_deterministic_algorithms_enabled()``
  is True (D-63a + D-63b additive OR composition).
- ``set_deterministic`` is exported via ``torch_structured.__all__`` (D-63c).
- The oracle fallback produces bit-identical d_twiddle across two backward
  calls (no atomicAdd reorder noise).
- The non-deterministic path at log_n=11 + batch=4096 DOES differ above
  machine epsilon (sanity: proves the gate actually does something).

Fixture: ``deterministic_env`` sets ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` per
RESEARCH §4 LANDMINE — required so ``torch.use_deterministic_algorithms(True)``
does not raise spurious cublas errors from unrelated PyTorch ops in the test
process.
"""
import pytest
import torch

import torch_structured
from torch_structured import Butterfly  # noqa: F401 — keeps import surface verified


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="deterministic mode tests require CUDA",
)


# Tight tolerances live here as named constants so a future tightening is a
# one-line edit (mirrors tests/test_phase9_integration.py).
_FP32_RTOL = 1e-5
_FP32_ATOL = 1e-6


@pytest.fixture
def deterministic_env(monkeypatch):
    """RESEARCH §4 LANDMINE — set CUBLAS_WORKSPACE_CONFIG so the global
    ``torch.use_deterministic_algorithms(True)`` doesn't raise from unrelated
    PyTorch ops in the test process.
    """
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    yield


# ═════════════════════════════════════════════════════════════════════════
# Test 1: set_deterministic round-trip (D-63 save/restore semantics)
# ═════════════════════════════════════════════════════════════════════════
def test_set_deterministic_round_trip():
    """``set_deterministic(value)`` returns the PREVIOUS value and mirrors
    the new value into the module-level ``_DETERMINISTIC`` flag.
    """
    # Save initial state so the test is order-independent.
    initial = torch_structured._ops._DETERMINISTIC
    try:
        torch_structured.set_deterministic(False)
        assert torch_structured._ops._DETERMINISTIC is False

        prev = torch_structured.set_deterministic(True)
        assert prev is False, f"expected previous value False, got {prev!r}"
        assert torch_structured._ops._DETERMINISTIC is True

        prev2 = torch_structured.set_deterministic(False)
        assert prev2 is True, f"expected previous value True, got {prev2!r}"
        assert torch_structured._ops._DETERMINISTIC is False
    finally:
        torch_structured.set_deterministic(initial)


# ═════════════════════════════════════════════════════════════════════════
# Test 2: top-level export surface (D-63c)
# ═════════════════════════════════════════════════════════════════════════
def test_set_deterministic_exported_from_top_level():
    """``set_deterministic`` is callable AND present in ``__all__``."""
    assert callable(torch_structured.set_deterministic), \
        "set_deterministic must be callable from the top-level package"
    assert "set_deterministic" in torch_structured.__all__, \
        "set_deterministic must be in torch_structured.__all__ (D-63c)"


# ═════════════════════════════════════════════════════════════════════════
# Test 3: _is_deterministic_mode_active OR semantics (D-63b additive OR)
# ═════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("det_flag,torch_global,expected", [
    (False, False, False),
    (True, False, True),
    (False, True, True),
    (True, True, True),
])
def test_is_deterministic_mode_active_or_semantics(
    det_flag, torch_global, expected, deterministic_env,
):
    """Helper returns ``_DETERMINISTIC or torch.are_deterministic_algorithms_enabled()``
    (D-63b composition is additive OR, not AND).
    """
    initial_det = torch_structured._ops._DETERMINISTIC
    initial_global = torch.are_deterministic_algorithms_enabled()
    try:
        torch_structured.set_deterministic(det_flag)
        torch.use_deterministic_algorithms(torch_global, warn_only=True)
        result = torch_structured._ops._is_deterministic_mode_active()
        assert result is expected, (
            f"_is_deterministic_mode_active() returned {result!r} for "
            f"(det_flag={det_flag}, torch_global={torch_global}); "
            f"expected {expected!r} (D-63b additive OR)"
        )
    finally:
        torch_structured.set_deterministic(initial_det)
        torch.use_deterministic_algorithms(initial_global, warn_only=True)


# ═════════════════════════════════════════════════════════════════════════
# Test 4: bit-identical d_twiddle under set_deterministic(True) (D-63a)
# ═════════════════════════════════════════════════════════════════════════
@pytest.mark.op("butterfly_multiply")
def test_deterministic_dtwiddle_bit_identical(backend, deterministic_env):
    """Under ``set_deterministic(True)`` AND ``BACKEND=triton``, two
    consecutive ``loss.backward()`` calls on the SAME (twiddle, x, grad_out)
    produce ``torch.equal(gt1, gt2) AND torch.equal(gi1, gi2)``.

    Parameters per RESEARCH §4 lines 433-447:
    - twiddle (1, 1, 9, 256, 2, 2) fp32 cuda — log_n=9 (kernel path, not the
      small-N fallback)
    - x (4096, 1, 512) fp32 cuda — large batch exercises the atomic-add path

    The oracle fallback is deterministic by construction (pure-PyTorch tensor
    ops; no atomicAdd reduction).
    """
    if backend != "triton":
        pytest.skip("deterministic gate is meaningful only on the triton path")

    initial_det = torch_structured._ops._DETERMINISTIC
    try:
        torch_structured.set_deterministic(True)
        torch.manual_seed(0)
        twiddle = torch.randn(
            1, 1, 9, 256, 2, 2, device="cuda", dtype=torch.float32,
            requires_grad=True,
        )
        x = torch.randn(
            4096, 1, 512, device="cuda", dtype=torch.float32,
            requires_grad=True,
        )
        grad_out = torch.randn(4096, 1, 512, device="cuda", dtype=torch.float32)

        out1 = torch_structured._ops.butterfly_multiply(twiddle, x, True, 512)
        gt1, gi1 = torch.autograd.grad(out1, [twiddle, x], grad_out)

        out2 = torch_structured._ops.butterfly_multiply(twiddle, x, True, 512)
        gt2, gi2 = torch.autograd.grad(out2, [twiddle, x], grad_out)

        assert torch.equal(gt1, gt2), (
            "d_twiddle not bit-identical under set_deterministic(True); "
            "max abs diff = "
            f"{(gt1 - gt2).abs().max().item()}"
        )
        assert torch.equal(gi1, gi2), (
            "d_input not bit-identical under set_deterministic(True); "
            "max abs diff = "
            f"{(gi1 - gi2).abs().max().item()}"
        )
    finally:
        torch_structured.set_deterministic(initial_det)


# ═════════════════════════════════════════════════════════════════════════
# Test 5: torch.use_deterministic_algorithms(True) activates the oracle (D-63b)
# ═════════════════════════════════════════════════════════════════════════
@pytest.mark.op("butterfly_multiply")
def test_deterministic_honors_torch_global_flag(backend, deterministic_env):
    """``set_deterministic(False)`` + ``torch.use_deterministic_algorithms(True)``
    produces bit-identical d_twiddle (D-63b — the global flag activates the
    oracle even when the library flag is False).
    """
    if backend != "triton":
        pytest.skip("deterministic gate is meaningful only on the triton path")

    initial_det = torch_structured._ops._DETERMINISTIC
    initial_global = torch.are_deterministic_algorithms_enabled()
    try:
        torch_structured.set_deterministic(False)
        torch.use_deterministic_algorithms(True, warn_only=True)

        torch.manual_seed(0)
        twiddle = torch.randn(
            1, 1, 9, 256, 2, 2, device="cuda", dtype=torch.float32,
            requires_grad=True,
        )
        x = torch.randn(
            4096, 1, 512, device="cuda", dtype=torch.float32,
            requires_grad=True,
        )
        grad_out = torch.randn(4096, 1, 512, device="cuda", dtype=torch.float32)

        out1 = torch_structured._ops.butterfly_multiply(twiddle, x, True, 512)
        gt1, _ = torch.autograd.grad(out1, [twiddle, x], grad_out)
        out2 = torch_structured._ops.butterfly_multiply(twiddle, x, True, 512)
        gt2, _ = torch.autograd.grad(out2, [twiddle, x], grad_out)

        assert torch.equal(gt1, gt2), (
            "d_twiddle not bit-identical under torch.use_deterministic_algorithms(True); "
            "D-63b composition broken (global flag did NOT activate oracle gate)"
        )
    finally:
        torch.use_deterministic_algorithms(initial_global, warn_only=True)
        torch_structured.set_deterministic(initial_det)


# ═════════════════════════════════════════════════════════════════════════
# Test 6: non-deterministic path DOES differ above epsilon (sanity)
# ═════════════════════════════════════════════════════════════════════════
@pytest.mark.op("butterfly_multiply")
def test_deterministic_non_det_differs_above_epsilon(backend):
    """Under ``set_deterministic(False)`` + ``torch.use_deterministic_algorithms(False)``
    AND log_n=11 + batch=4096 (the highest-atomic-pressure cell per Phase 8
    D-52), two consecutive backward calls produce d_twiddle that differs
    ABOVE machine epsilon — proves the deterministic gate actually does
    something.

    The diff is REAL atomicAdd reorder noise, not nonsense — bounded by a
    sanity ceiling derived from Phase 8 08-01-SUMMARY.md noise model:
    practical fp32 atomicAdd noise at batch=4096 is
    ``sqrt(4096) * eps_fp32 * value_magnitude ≈ 6.4e-3`` relative. With
    randn-magnitudes accumulated over log_n=11 stages (peak values
    ~O(10)), absolute noise is bounded by ~0.1. We pick 0.5 as the sanity
    ceiling — comfortably wider than the empirical noise envelope but
    still tight enough to catch a true regression (e.g., the kernel
    silently produced NaNs or completely-divergent values).
    """
    if backend != "triton":
        pytest.skip("Triton-only atomic-add reorder noise test")

    initial_det = torch_structured._ops._DETERMINISTIC
    initial_global = torch.are_deterministic_algorithms_enabled()
    try:
        torch_structured.set_deterministic(False)
        torch.use_deterministic_algorithms(False)

        torch.manual_seed(0)
        log_n = 11
        n = 1 << log_n  # 2048
        twiddle = torch.randn(
            1, 1, log_n, n // 2, 2, 2, device="cuda", dtype=torch.float32,
            requires_grad=True,
        )
        x = torch.randn(
            4096, 1, n, device="cuda", dtype=torch.float32, requires_grad=True,
        )
        grad_out = torch.randn(4096, 1, n, device="cuda", dtype=torch.float32)

        out1 = torch_structured._ops.butterfly_multiply(twiddle, x, True, n)
        gt1, _ = torch.autograd.grad(out1, [twiddle, x], grad_out)
        out2 = torch_structured._ops.butterfly_multiply(twiddle, x, True, n)
        gt2, _ = torch.autograd.grad(out2, [twiddle, x], grad_out)

        assert not torch.equal(gt1, gt2), (
            "d_twiddle was bit-identical at log_n=11 + batch=4096 with "
            "set_deterministic(False) — the deterministic gate is not doing "
            "anything (or the kernel suddenly became deterministic-by-luck, "
            "which is unexpected at this atomic pressure)"
        )
        max_diff = (gt1 - gt2).abs().max().item()
        assert max_diff < 0.5, (
            f"d_twiddle diff between consecutive backward calls was "
            f"{max_diff} — above the 0.5 sanity ceiling derived from "
            f"Phase 8 noise model (~0.1 expected for log_n=11, batch=4096); "
            f"this is nonsense (NaN/Inf or divergent values), not atomicAdd noise"
        )
    finally:
        torch.use_deterministic_algorithms(initial_global, warn_only=True)
        torch_structured.set_deterministic(initial_det)
