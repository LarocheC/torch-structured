"""Phase 9 09-03 perf gate — TEST-04 (Triton ≥60% of CUDA throughput) + selector unit tests.

This module automates the TEST-04 perf gate by reading the committed baseline
JSON at ``.planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json``
and asserting that every cell with non-null ``reference_cuda_p50`` has
``wall_ms_p50 / reference_cuda_p50 ≤ 1.67`` (= 1/0.60 — Phase 9 D-65b). Cells
where Triton trails CUDA by more than this margin route through the runtime
selector (D-66) and are explicitly allowed (the routing table marks them).

The selector unit tests verify ``_should_route_to_cuda`` and
``set_routing_enabled`` behave correctly on the dev-host bake.

The JSON-driven gate test does NOT touch the GPU; it runs in CPU CI alongside
the rest of ``tests/``. The selector tests that monkey-patch the legacy CUDA
binding skip when ``_has_cuda_legacy()`` is False (matches Phase 9 D-62
3-axis backend fixture convention).
"""
import json
import pathlib

import pytest
import torch

import torch_structured  # noqa: F401 — triggers extension load + _ops.py resolver


BASELINE_PATH = pathlib.Path(
    ".planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json"
)
ROUTING_PATH = pathlib.Path("torch_structured/_routing.json")
THRESHOLD = 1.0 / 0.60  # = 1.67 — Phase 9 D-65b


def _load_baseline_rows() -> list:
    if not BASELINE_PATH.exists():
        pytest.skip(f"baseline file not found: {BASELINE_PATH}")
    return json.loads(BASELINE_PATH.read_text())["rows"]


def test_perf_gate_triton_at_60pct_cuda():
    """TEST-04 (D-65b): every cell with non-null reference_cuda_p50 must have
    Triton wall_ms_p50 / reference_cuda_p50 ≤ 1.67. Cells exceeding the gate
    are routed to CUDA via ``_routing.json`` and explicitly allowed (the
    routing table is the escape valve).

    When ``reference_cuda_p50`` is null across the board (dev-host without
    a working CUDA legacy build — Plan 09-01 SUMMARY documents this for the
    Phase 9 dev environment), the test is a soft pass: nothing to gate, the
    runtime selector falls back to torch-ref weaker gate (5.0×).
    """
    rows = _load_baseline_rows()
    cells_with_cuda = [r for r in rows if r.get("reference_cuda_p50") is not None]

    if not cells_with_cuda:
        pytest.skip(
            "No rows have reference_cuda_p50 — dev-host without working "
            "CUDA legacy build (Plan 09-01 SUMMARY). Soft-passing TEST-04."
        )

    # Load the routing table to know which cells are explicitly allowed to
    # exceed the gate.
    if ROUTING_PATH.exists():
        routing = json.loads(ROUTING_PATH.read_text()).get("rules", {})
    else:
        routing = {}

    failures: list[tuple] = []
    for row in cells_with_cuda:
        ratio = row["wall_ms_p50"] / row["reference_cuda_p50"]
        key = (
            f"{row['kernel']}::{row['log_n']}::{row['dtype']}"
            f"::{row.get('direction', 'forward')}"
        )
        # A failure is a cell exceeding the gate that is NOT in route_to_cuda
        # (i.e., the selector would NOT bypass the Triton kernel here).
        allowed_by_routing = routing.get(key, {}).get("route_to_cuda", False)
        if ratio > THRESHOLD and not allowed_by_routing:
            failures.append((key, ratio, row))

    # Sort failures descending by ratio for actionable reporting.
    failures.sort(key=lambda x: -x[1])
    assert not failures, (
        f"TEST-04 perf gate failures (ratio > {THRESHOLD:.2f}, NOT in "
        f"route_to_cuda routing table):\n" +
        "\n".join(
            f"  {k}: ratio={r:.2f} "
            f"(triton={row['wall_ms_p50']:.4f} ms, cuda={row['reference_cuda_p50']:.4f} ms)"
            for k, r, row in failures
        )
    )


def test_should_route_to_cuda_default_matches_bake():
    """The selector must agree with the committed _routing.json bake.

    For every cell in the table, ``_should_route_to_cuda`` must return the
    same boolean as the cell's ``route_to_cuda`` field. This is the basic
    consistency check between the static bake and the runtime selector.
    """
    rules = torch_structured._ops._ROUTING_TABLE
    if not rules:
        pytest.skip("Empty routing table — nothing to verify")

    # Ensure routing is enabled (the test must run with the default state).
    prev = torch_structured._ops.set_routing_enabled(True)
    try:
        for key, cell in rules.items():
            kernel, log_n_str, dtype_str, direction = key.split("::")
            log_n = int(log_n_str)
            n = 1 << log_n
            shape = (64, 1, n)
            dtype = torch.float32 if dtype_str == "fp32" else torch.complex64

            actual = torch_structured._ops._should_route_to_cuda(
                kernel, shape, dtype, direction
            )
            expected = bool(cell.get("route_to_cuda", False))
            assert actual == expected, (
                f"Selector disagrees with bake for {key}: "
                f"selector={actual}, bake={expected}"
            )
    finally:
        torch_structured._ops.set_routing_enabled(prev)


def test_should_route_to_cuda_unknown_cell_returns_false():
    """A cell that does NOT exist in the routing table returns False
    (no routing). This is the safe default for unknown shapes/dtypes."""
    # log_n=20 is far outside the baked grid (8-11).
    result = torch_structured._ops._should_route_to_cuda(
        "butterfly_multiply",
        (64, 1, 1 << 20),
        torch.float32,
        "forward",
    )
    assert result is False


def test_should_route_to_cuda_disabled_returns_false():
    """When ``_ROUTING_DISABLED`` is True, the selector returns False for
    ANY input, even cells that would otherwise route. D-66c — the test
    override that Phase 8 SC#4 reconciliation (D-61a) relies on."""
    prev = torch_structured._ops.set_routing_enabled(False)
    try:
        # Try every cell in the routing table — none should return True.
        for key, cell in torch_structured._ops._ROUTING_TABLE.items():
            kernel, log_n_str, dtype_str, direction = key.split("::")
            log_n = int(log_n_str)
            n = 1 << log_n
            shape = (64, 1, n)
            dtype = torch.float32 if dtype_str == "fp32" else torch.complex64
            result = torch_structured._ops._should_route_to_cuda(
                kernel, shape, dtype, direction
            )
            assert result is False, (
                f"Routing disabled but {key} returned True (cell={cell})"
            )
    finally:
        torch_structured._ops.set_routing_enabled(prev)


def test_set_routing_enabled_round_trip():
    """The setter returns the PREVIOUS value (save/restore pattern,
    mirrors ``set_backend`` and ``set_deterministic``)."""
    # Capture initial state.
    initial = not torch_structured._ops._ROUTING_DISABLED

    # set to True; previous was 'initial'.
    prev = torch_structured._ops.set_routing_enabled(True)
    assert prev == initial, f"prev should be {initial}, got {prev}"
    assert torch_structured._ops._ROUTING_DISABLED is False

    # set to False; previous was True.
    prev = torch_structured._ops.set_routing_enabled(False)
    assert prev is True
    assert torch_structured._ops._ROUTING_DISABLED is True

    # set to True; previous was False.
    prev = torch_structured._ops.set_routing_enabled(True)
    assert prev is False
    assert torch_structured._ops._ROUTING_DISABLED is False

    # Restore original.
    torch_structured._ops.set_routing_enabled(initial)


def test_routing_log_emitted_once_per_cell(caplog):
    """First-route log fires exactly once per (op, log_n, dtype, direction)
    tuple per process. RESEARCH §8 lines 742-754.

    Skipped when no cell in the dev-host bake is marked route_to_cuda
    (the selector behavior is unverifiable in that configuration).
    """
    routed_cells = [
        k for k, v in torch_structured._ops._ROUTING_TABLE.items()
        if v.get("route_to_cuda", False)
    ]
    if not routed_cells:
        pytest.skip(
            "No route_to_cuda cells in dev-host bake — "
            "selector log behavior unverifiable in this configuration"
        )

    # Pick the first routed cell.
    key = routed_cells[0]
    kernel, log_n_str, dtype_str, direction = key.split("::")
    log_n = int(log_n_str)
    n = 1 << log_n
    shape = (64, 1, n)
    dtype = torch.float32 if dtype_str == "fp32" else torch.complex64

    # Reset the emit set.
    torch_structured._ops._routing_log_emitted.clear()

    prev = torch_structured._ops.set_routing_enabled(True)
    try:
        with caplog.at_level("INFO", logger="torch_structured"):
            # Call twice — should emit only once.
            torch_structured._ops._should_route_to_cuda(kernel, shape, dtype, direction)
            torch_structured._ops._should_route_to_cuda(kernel, shape, dtype, direction)
    finally:
        torch_structured._ops.set_routing_enabled(prev)

    routing_messages = [r for r in caplog.records if "routing" in r.message.lower()]
    assert len(routing_messages) == 1, (
        f"Expected exactly 1 routing log message; got {len(routing_messages)}: "
        f"{[r.message for r in routing_messages]}"
    )


@pytest.mark.op('butterfly_multiply')
def test_runtime_selector_routes_to_cuda(backend):
    """Integration test: under BACKEND=triton AND on a cell marked
    route_to_cuda AND ``_has_cuda_legacy()`` is True, calling
    ``butterfly_multiply`` invokes the CUDA path (not the Triton kernel).

    Monkey-patches ``_cuda_legacy.butterfly.butterfly_multiply`` to a sentinel
    that records invocation; runs Butterfly under a routed shape; asserts the
    sentinel was called.

    Skip when:
    - ``backend != 'triton'`` (the integration is meaningful only for the
      Triton resolver branch).
    - ``_has_cuda_legacy()`` is False (no CUDA legacy backend to route to).
    - No cell in the dev-host bake is marked route_to_cuda (selector
      behavior unverifiable in this configuration).
    """
    if backend != "triton":
        pytest.skip("Routing closure is wired only on the Triton resolver branch")
    if not torch_structured._ops._has_cuda_legacy():
        pytest.skip("No CUDA legacy .so — routing falls back to Triton (D-61b)")
    routed_cells = [
        k for k, v in torch_structured._ops._ROUTING_TABLE.items()
        if v.get("route_to_cuda", False) and "forward" in k
    ]
    if not routed_cells:
        pytest.skip(
            "No forward-direction route_to_cuda cells in dev-host bake — "
            "selector integration unverifiable in this configuration"
        )

    # Pick a forward-direction routed cell.
    key = routed_cells[0]
    kernel, log_n_str, dtype_str, direction = key.split("::")
    log_n = int(log_n_str)
    n = 1 << log_n
    dtype = torch.float32 if dtype_str == "fp32" else torch.complex64

    # Monkey-patch the cuda_legacy butterfly_multiply binding to a sentinel.
    # We need to re-resolve the backend so the routing closure picks up the
    # new (sentinel) binding. We do this by mutating the module attribute
    # the closure imported.
    import torch_structured._cuda_legacy as _cl_pkg
    invoked = {"count": 0}
    original = _cl_pkg.butterfly_multiply

    def sentinel(*args, **kwargs):
        invoked["count"] += 1
        return original(*args, **kwargs)

    # Force re-resolution after monkey-patch so the routed closure binds the
    # sentinel.
    _cl_pkg.butterfly_multiply = sentinel
    prev_backend = torch_structured._ops.set_backend("triton")
    try:
        twiddle = torch.randn(
            1, 1, log_n, n // 2, 2, 2, device="cuda", dtype=dtype
        )
        input_ = torch.randn(64, 1, n, device="cuda", dtype=dtype)
        _ = torch_structured._ops.butterfly_multiply(twiddle, input_, True, n)
        assert invoked["count"] > 0, (
            f"Routing closure did NOT invoke the CUDA sentinel for {key}"
        )
    finally:
        _cl_pkg.butterfly_multiply = original
        torch_structured._ops.set_backend(prev_backend)
