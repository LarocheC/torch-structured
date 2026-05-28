"""Regenerate ``torch_structured/_routing.json`` from the 07-BASELINE.json perf grid.

Phase 9 09-03 (D-66, D-66a, D-66b, D-66c, D-66d) — the static routing table that
the runtime selector at ``torch_structured/_ops.py:_should_route_to_cuda`` consults
to decide whether a given ``(op, log_n, dtype, direction)`` cell should bypass the
Triton kernel and route to the legacy CUDA path. The table is BAKED at packaging
time from the dev-host perf measurements in ``07-BASELINE.json`` and shipped with
the package; users on different hardware may regenerate locally.

LANDMINE (RESEARCH §9): this script must NOT auto-run on every test or CI build.
The bake is intentional — users explicitly opt into a regeneration on their own
hardware. The committed ``_routing.json`` reflects the dev-host bake; users see
that table by default.

Threshold:
- When ``reference_cuda_p50`` is non-null: ``route_to_cuda = (wall_ms_p50 /
  reference_cuda_p50) > 1.67`` (D-65b — the 1/0.60 = 1.6667 perf gate).
- When ``reference_cuda_p50`` is null (CUDA .so missing/unbuilt — see Plan 09-01
  for the dev-host CUDA mismatch): fall back to comparison against
  ``reference_torch_ref_p50`` with the weaker 5.0× threshold per Phase 7's
  documented torch-ref weaker gate (RESEARCH §9).

Output schema (keyed object for O(1) lookup per RESEARCH §9):

    {
      "schema_version": 1,
      "generated_at": "<ISO 8601 Z>",
      "generated_from": ".planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json",
      "threshold_ratio": 1.6666666666666667,
      "rules": {
        "butterfly_multiply::8::fp32::forward":     {"triton_cuda_ratio_p50": 0.85, "route_to_cuda": false},
        "butterfly_multiply::11::complex64::backward": {"triton_cuda_ratio_p50": 1.92, "route_to_cuda": true},
        ...
      }
    }

Key format contract (D-66b):
    f"{kernel}::{log_n}::{dtype}::{direction}"

This format is consumed by ``torch_structured/_ops.py:_should_route_to_cuda``.
If the key format ever changes, BOTH this script AND the selector MUST be
updated together.

Invocation:

    python scripts/regenerate_routing_table.py

Exits 0 on success; prints a one-line summary "Wrote {N} rules; {K} marked
route_to_cuda".
"""
import datetime
import json
import pathlib
import sys


# Path constants — caller is expected to invoke from the repo root.
BASELINE = pathlib.Path(".planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json")
ROUTING = pathlib.Path("torch_structured/_routing.json")

# Phase 9 D-65b perf gate threshold: route to CUDA when Triton/CUDA > 1.67
# (= 1 / 0.60; the "Triton must be at least 60% of CUDA" gate).
THRESHOLD = 1.0 / 0.60

# Fallback when reference_cuda_p50 is null (CUDA .so missing/unbuilt). Phase
# 7's documented weaker gate against the torch-ref oracle (RESEARCH §9 D-65b).
TORCH_REF_THRESHOLD = 5.0


def main() -> int:
    if not BASELINE.exists():
        print(f"ERROR: baseline file not found: {BASELINE}", file=sys.stderr)
        return 1

    baseline = json.loads(BASELINE.read_text())
    rows = baseline.get("rows", [])

    rules: dict = {}
    routed_count = 0
    fallback_count = 0

    for row in rows:
        # Key construction MUST match the resolver's lookup format in
        # torch_structured/_ops.py:_should_route_to_cuda — Phase 9 D-66b
        # contract. If this format changes, update BOTH this script and the
        # selector together.
        kernel = row["kernel"]
        log_n = row["log_n"]
        dtype = row["dtype"]
        direction = row.get("direction", "forward")
        key = f"{kernel}::{log_n}::{dtype}::{direction}"

        # Use do_bench_p50_ms (triton.testing.do_bench — canonical per TEST-04
        # wording) as the authoritative Triton measurement, NOT wall_ms_p50.
        # The custom measure_p50_p95 harness over-reports 47-91% at small kernel
        # sizes (per-iteration CUDA-Event + sync overhead), which would route
        # nearly every cell to CUDA and defeat the Triton migration. Mirrors the
        # TEST-04 gate fix (quick 260528-tv9). Falls back to wall_ms_p50 only if
        # do_bench is unavailable. Amends Phase 9 D-65b / D-66.
        triton_p50 = row.get("do_bench_p50_ms")
        if triton_p50 is None:
            triton_p50 = row.get("wall_ms_p50")
        cuda_p50 = row.get("reference_cuda_p50")
        torch_ref_p50 = row.get("reference_torch_ref_p50")

        if cuda_p50 is not None and cuda_p50 > 0:
            ratio = triton_p50 / cuda_p50
            route = ratio > THRESHOLD
            entry = {
                "triton_cuda_ratio_p50": round(ratio, 4),
                "route_to_cuda": route,
                "comparison_basis": "cuda",
            }
        elif torch_ref_p50 is not None and torch_ref_p50 > 0:
            # Weaker gate when CUDA absent — Phase 7 documented fallback.
            ratio = triton_p50 / torch_ref_p50
            route = ratio > TORCH_REF_THRESHOLD
            entry = {
                "triton_torch_ref_ratio_p50": round(ratio, 4),
                "route_to_cuda": route,
                "comparison_basis": "torch_ref_fallback",
            }
            fallback_count += 1
        else:
            # No reference at all — cannot decide; default to NOT routing.
            entry = {
                "triton_cuda_ratio_p50": None,
                "route_to_cuda": False,
                "comparison_basis": "none",
            }

        if entry["route_to_cuda"]:
            routed_count += 1

        rules[key] = entry

    output = {
        "schema_version": 1,
        "generated_at": datetime.datetime.now(datetime.timezone.utc)
            .isoformat().replace("+00:00", "Z"),
        "generated_from": str(BASELINE),
        "threshold_ratio": THRESHOLD,
        "torch_ref_fallback_threshold": TORCH_REF_THRESHOLD,
        "rules": rules,
    }

    ROUTING.parent.mkdir(parents=True, exist_ok=True)
    ROUTING.write_text(json.dumps(output, indent=2) + "\n")

    print(
        f"Wrote {len(rules)} rules; {routed_count} marked route_to_cuda "
        f"({fallback_count} used torch_ref fallback gate)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
