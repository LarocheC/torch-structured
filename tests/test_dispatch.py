"""Cross-cutting dispatch smoke tests.

Covers the dispatch contract that survives across Phase 5/6/7 kernel rollouts:
* ``set_backend`` round-trip + idempotency
* ``ValueError`` on unknown backend (T-04-01 mitigation regression test)
* ``_has_triton_kernel`` honest-probe (CHECKER B3) returns a clean ``bool``

The Phase 4 demonstrator-specific tests were removed per D-28 — kernel
correctness moved to ``test_diag_mult.py`` (Phase 5) and subsequent
per-op test modules in Phase 6/7.
"""
import pytest

import torch_structured  # noqa: F401 — triggers extension load + _ops.py resolver


def test_set_backend_round_trip():
    """set_backend('torch') always succeeds and is idempotent."""
    chosen = torch_structured._ops.set_backend("torch")
    assert chosen == "torch"
    chosen2 = torch_structured._ops.set_backend("torch")
    assert chosen2 == "torch"


def test_unknown_backend_raises_value_error():
    """T-04-01 mitigation: arbitrary env-var values explicitly rejected."""
    with pytest.raises(ValueError, match="Unknown backend"):
        torch_structured._ops.set_backend("nonsense")


def test_has_triton_kernel_probe_returns_bool():
    """CHECKER B3 honest-probe regression: probe returns a clean bool, never raises."""
    assert isinstance(torch_structured._ops._has_triton_kernel("diag_mult"), bool)
    assert isinstance(torch_structured._ops._has_triton_kernel("does_not_exist"), bool)
