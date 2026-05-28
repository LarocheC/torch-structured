"""Phase 9 D-62 widens the backend fixture to 3-axis with per-op cuda skip-gate.

Phase 5 narrowed the skip-gate to the per-op diag_mult probe; Phase 6 D-39
widened it to ``_has_any_triton_kernel()`` so the fixture skips the ``triton``
param ONLY when NO per-op Triton kernel is installed. Phase 9 D-62 adds a
third axis (``cuda``) with a per-test skip-gate via
``@pytest.mark.op('<op_name>')`` markers (D-62b option 1, recommended in
CONTEXT + RESEARCH — explicit-is-better, markers visible in test source and
``pytest --collect-only``).

The ``cuda`` axis is skipped per-test when:

1. The test bears ``@pytest.mark.op('<op_name>')`` AND
2. ``_ops._has_cuda_legacy_for_op('<op_name>')`` returns False.

Tests without an ``@pytest.mark.op`` marker are backend-agnostic: the
``cuda`` axis is NOT skipped (the test runs against whatever ``cuda`` resolves
to, which falls back to ``torch`` when no .so is built, per Phase 5 D-22).

The fixture captures the current ``_BACKEND``, sets the requested one,
yields the actually-resolved name, then restores the original on teardown so
tests are order-independent (D-22 + the save/restore pattern).
"""
import pytest

import torch_structured  # noqa: F401 — triggers extension load + _ops.py resolver


def pytest_configure(config):
    """Register custom markers to silence PytestUnknownMarkWarning.

    - ``slow`` (Phase 7 D-43a): gates the comprehensive Cartesian tier behind
      ``pytest -m slow``.
    - ``multigpu`` (Phase 9 D-64): requires ≥2 GPUs (opt-in via
      ``torchrun --nproc_per_node=2``). Reserved for Plan 09-02 FSDP test;
      registered here so 09-01 ships a complete marker set.
    - ``op(name)`` (Phase 9 D-62b): binds a test to a specific op for the
      per-op cuda skip-gate (e.g., ``@pytest.mark.op('butterfly_multiply')``).
    """
    config.addinivalue_line(
        "markers",
        "slow: opt-in comprehensive parameter grid (Phase 7 D-43a)",
    )
    config.addinivalue_line(
        "markers",
        "multigpu: requires >=2 GPUs (Phase 9 D-64 — opt-in via "
        "torchrun --nproc_per_node=2)",
    )
    config.addinivalue_line(
        "markers",
        "op(name): bind a test to a specific op for per-op cuda skip-gate "
        "(Phase 9 D-62b; e.g., @pytest.mark.op('butterfly_multiply'))",
    )


@pytest.fixture(params=["torch", "triton", "cuda"])
def backend(request):
    """Switch backend for the duration of a test, restore after.

    Per-axis skip-gate behavior (D-39 + D-62b composed):
    - ``triton`` skipped if no per-op Triton kernel is installed at all
      (``_has_any_triton_kernel()``).
    - ``cuda`` skipped if the test bears ``@pytest.mark.op('<op_name>')``
      AND ``_has_cuda_legacy_for_op('<op_name>')`` returns False. Marker-less
      tests do NOT skip the cuda axis (treated as backend-agnostic).
    - ``torch`` is always available (the pure-PyTorch oracle fallback).
    """
    param = request.param
    if param == "triton" and not torch_structured._ops._has_any_triton_kernel():
        pytest.skip("No Triton kernel installed (no CUDA or CPU-only runner)")
    if param == "cuda":
        op_marker = request.node.get_closest_marker("op")
        if op_marker is not None and op_marker.args:
            op_name = op_marker.args[0]
            if not torch_structured._ops._has_cuda_legacy_for_op(op_name):
                pytest.skip(f"No CUDA legacy .so for {op_name}")
    original = torch_structured._ops._BACKEND
    chosen = torch_structured._ops.set_backend(param)
    yield chosen
    torch_structured._ops.set_backend(original)
