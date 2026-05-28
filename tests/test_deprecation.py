"""Phase 10 DEPR-02 verification — DeprecationWarning emission contract.

Three tests gate the D-74 / D-74a / D-74b contract:

- ``test_cuda_backend_emits_deprecation_warning`` (D-75): the verbatim warning
  text fires when ``set_backend('cuda')`` is invoked.
- ``test_cuda_backend_warning_fires_only_once`` (D-75a): the once-per-process
  ``simplefilter("once", DeprecationWarning)`` gate works — back-to-back
  ``set_backend('cuda')`` calls in the same process emit the warning exactly
  once.
- ``test_has_cuda_legacy_probe_does_not_emit_warning`` (D-75b — LOAD-BEARING):
  the Phase 9 ``_has_cuda_legacy_for_op`` probe is silenced by Task 2's
  ``warnings.catch_warnings()`` wrap — the silent probe stays silent so the
  Phase 9 conftest backend fixture doesn't spam users every test run.

Subprocess pattern (D-75 / D-75a): Python's ``simplefilter("once",
DeprecationWarning)`` registry persists across ``warnings.catch_warnings()``
blocks within the same process — once-fired stays once-fired even if the test
resets its filter. Fresh subprocesses get a fresh registry, so each
subprocess-invoked assertion is uncontaminated by other tests in the
collection.

All three tests bear ``@pytest.mark.op('butterfly_multiply')`` per Phase 9
D-62 / D-81 — the cuda axis is skipped when ``_butterfly.so`` is missing.
"""
import subprocess
import sys
import warnings

import pytest

import torch_structured


@pytest.mark.op('butterfly_multiply')
def test_cuda_backend_emits_deprecation_warning():
    """D-75: TORCH_STRUCTURED_BACKEND=cuda emits the verbatim DeprecationWarning.

    Subprocess invocation so the once-per-process gate is uncontaminated by
    other tests. The subprocess gets a fresh ``warnings`` registry; the
    ``simplefilter("once", DeprecationWarning)`` installed at the top of
    ``torch_structured/_cuda_legacy/__init__.py`` then fires exactly once
    on the first import of ``_cuda_legacy``.
    """
    if not torch_structured._ops._has_cuda_legacy_for_op("butterfly_multiply"):
        pytest.skip("No CUDA legacy .so for butterfly_multiply")

    result = subprocess.run(
        [
            sys.executable,
            "-W", "always::DeprecationWarning",
            "-c",
            "import warnings; warnings.simplefilter('always'); "
            "import torch_structured; "
            "torch_structured._ops.set_backend('cuda')",
        ],
        capture_output=True,
        text=True,
    )

    # The verbatim warning text from 04-DEPRECATION-PLAN.md — three load-bearing
    # tokens that uniquely identify the Phase 10 D-74 emission.
    assert "CUDA C++ backend" in result.stderr, (
        f"Expected 'CUDA C++ backend' in stderr; got:\n{result.stderr}"
    )
    assert "default-disabled in v1.3" in result.stderr, (
        f"Expected 'default-disabled in v1.3' in stderr; got:\n{result.stderr}"
    )
    assert "v1.4+" in result.stderr, (
        f"Expected 'v1.4+' in stderr; got:\n{result.stderr}"
    )

    # Exactly one emission per process (D-74 once-gate).
    assert result.stderr.count("CUDA C++ backend") == 1, (
        f"Warning fired {result.stderr.count('CUDA C++ backend')} times; "
        f"expected exactly 1.\nstderr:\n{result.stderr}"
    )


@pytest.mark.op('butterfly_multiply')
def test_cuda_backend_warning_fires_only_once():
    """D-75a: subsequent set_backend('cuda') calls don't re-fire the warning.

    Subprocess does TWO back-to-back ``set_backend('cuda')`` calls. The first
    triggers ``from torch_structured._cuda_legacy import ...`` which executes
    ``_cuda_legacy/__init__.py``'s top-level ``warnings.warn``. The second
    call hits Python's module cache (``sys.modules``) — the module body does
    NOT re-execute — so the warning does NOT re-fire. ``simplefilter("once")``
    is a belt-and-suspenders gate; the primary once-mechanism is module-level
    code running exactly once per process.
    """
    if not torch_structured._ops._has_cuda_legacy_for_op("butterfly_multiply"):
        pytest.skip("No CUDA legacy .so for butterfly_multiply")

    result = subprocess.run(
        [
            sys.executable,
            "-W", "always::DeprecationWarning",
            "-c",
            "import warnings; warnings.simplefilter('always'); "
            "import torch_structured; "
            "torch_structured._ops.set_backend('cuda'); "
            "torch_structured._ops.set_backend('cuda')",
        ],
        capture_output=True,
        text=True,
    )

    assert "CUDA C++ backend" in result.stderr, (
        f"Expected warning to fire at least once; stderr:\n{result.stderr}"
    )
    count = result.stderr.count("CUDA C++ backend")
    assert count == 1, (
        f"Warning fired {count} times across two set_backend('cuda') calls; "
        f"expected exactly 1 (D-74 once-per-process gate).\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.mark.op('butterfly_multiply')
def test_has_cuda_legacy_probe_does_not_emit_warning():
    """D-75b (LOAD-BEARING): the Phase 9 probe stays silent via D-74b wrap.

    Without Task 2's ``warnings.catch_warnings()`` wrap on
    ``_has_cuda_legacy_diag_mult`` / ``_has_cuda_legacy_hadamard``, every call
    to ``_has_cuda_legacy_for_op`` (which the Phase 9 backend fixture invokes
    on every parametrized test) would emit the user-facing
    DeprecationWarning. This test gates the wrap.

    In-process capture is sufficient because the probe is meant to NEVER emit
    the warning — the once-per-process gate is irrelevant when the expected
    count is zero.
    """
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)

        # Exercise all three per-op branches of _has_cuda_legacy_for_op.
        torch_structured._ops._has_cuda_legacy_for_op("butterfly_multiply")
        torch_structured._ops._has_cuda_legacy_for_op("diag_mult")
        torch_structured._ops._has_cuda_legacy_for_op("hadamard_transform")

        user_warnings = [
            w for w in captured
            if issubclass(w.category, DeprecationWarning)
            and "CUDA C++ backend" in str(w.message)
        ]

    assert len(user_warnings) == 0, (
        f"D-74b broken: probe emitted {len(user_warnings)} user-facing "
        f"DeprecationWarning(s); expected 0. Warnings: "
        f"{[str(w.message) for w in user_warnings]}"
    )
