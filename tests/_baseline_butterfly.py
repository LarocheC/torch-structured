"""Phase 7 Plan 07-02 perf baseline harness for butterfly_multiply (D-43b schema).

Standalone measurement script — NOT a pytest test. The leading underscore in
the filename prevents pytest auto-collection. Invoke manually exactly once at
end of Plan 07-02 execution to populate the JSON baseline that Phase 9
TEST-04 (parity gate) reads verbatim.

Schema (locked at CONTEXT.md D-43b, extended by Phase 8 D-51 + Phase 9 D-65a + W2):

    {
      "rows": [
        {
          "kernel": "butterfly_multiply",
          "dtype": "fp32" | "complex64",
          "log_n": 8 | 9 | 10 | 11,
          "nstacks": 1,
          "nblocks": 1,
          "wall_ms_p50": <float>,
          "wall_ms_p95": <float>,
          "reference_torch_ref_p50": <float>,
          "reference_cuda_p50": <float | null>,  # Phase 9 D-65a
          "do_bench_p50_ms": <float | null>,     # Phase 9 W2 — triton.testing.do_bench sibling
          "measured_at": "<ISO8601 UTC>",
          "gpu": "<torch.cuda.get_device_name(0)>"
        },
        ...
      ]
    }

8 rows total: log_n in {8, 9, 10, 11} x dtype in {fp32, complex64}.

Measurement protocol (07-PATTERNS.md lines 440-457):
- batch_size = 64, nstacks = 1, nblocks = 1, increasing_stride = True, output_size = n
- Warmup 10 iterations + measure 100 iterations
- ``torch.cuda.Event(enable_timing=True)`` start/end pairs around each iteration
- Sync after each iteration so timing is per-iteration wall-clock GPU time
- p50 = times[len/2], p95 = times[int(len * 0.95)] from sorted times

On a CPU-only host this script prints SKIP and exits 0 — the baseline JSON
is then deferred and Plan 07-02 SUMMARY.md records the deferral.

Invocation note: when running from a git worktree where ``torch_structured``
is pip-editable-installed against the main repo, run with
``PYTHONPATH=. python tests/_baseline_butterfly.py`` so the worktree's
``torch_structured`` package is imported (not the main-repo install).
Pytest auto-injects the repo root into ``sys.path`` so its own runs work
without ``PYTHONPATH``; this standalone script does not.
"""
import datetime
import json
import os
import sys

import torch

import torch_structured  # noqa: F401 — triggers extension load + _ops.py resolver
from torch_structured._torch_ref.butterfly import butterfly_multiply_torch as butterfly_ref

# Phase 9 D-65 (W2 — cross-correlation): use triton.testing.do_bench AS A
# SIBLING column for the custom harness. The custom harness drives the gate
# decision; do_bench_p50_ms is recorded alongside for cross-correlation per
# RESEARCH §5. Guard import in case triton is somehow missing.
try:
    import triton.testing as _triton_testing
except ImportError:
    _triton_testing = None


BASELINE_PATH = ".planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json"


def measure_p50_p95(fn, *, warmup: int = 10, n_iter: int = 100) -> tuple[float, float]:
    """Return (p50, p95) per-iteration wall-clock GPU time (milliseconds).

    Uses ``torch.cuda.Event(enable_timing=True)`` start/end pairs surrounding
    ``fn()`` with one sync per iteration; the small sync overhead is
    negligible vs the kernel work at log_n >= 8.
    """
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Measure
    times: list[float] = []
    for _ in range(n_iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # milliseconds

    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    return p50, p95


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA — baseline JSON deferred")
        return 0

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")

    # Stash the original backend so we can restore it at end.
    original_backend = torch_structured._ops._BACKEND

    rows: list[dict] = []

    try:
        for log_n in (8, 9, 10, 11):
            for dtype, dtype_name in ((torch.float32, "fp32"),
                                      (torch.complex64, "complex64")):
                n = 1 << log_n
                batch_size, nstacks, nblocks = 64, 1, 1

                torch.manual_seed(42)
                twiddle = torch.randn(
                    nstacks, nblocks, log_n, n // 2, 2, 2,
                    device="cuda", dtype=dtype,
                )
                input_ = torch.randn(
                    batch_size, nstacks, n,
                    device="cuda", dtype=dtype,
                )

                # Triton backend
                torch_structured._ops.set_backend("triton")
                def triton_call():
                    return torch_structured._ops.butterfly_multiply(
                        twiddle, input_, True, n
                    )
                p50_triton, p95_triton = measure_p50_p95(triton_call)

                # Reference pure-PyTorch oracle
                def ref_call():
                    return butterfly_ref(twiddle, input_, True, n)
                p50_ref, _ = measure_p50_p95(ref_call)

                # Phase 9 D-65a: CUDA p50 measurement when _butterfly.so is built.
                # Falls through to reference_cuda_p50=None when CUDA legacy is
                # missing — the perf gate (D-65b) loosens to "Triton >= 60% of
                # reference_torch_ref_p50" with the 5.0x threshold per Phase 7's
                # documented torch-ref weaker gate.
                p50_cuda = None
                if torch_structured._ops._has_cuda_legacy():
                    torch_structured._ops.set_backend("cuda")
                    def cuda_call():
                        return torch_structured._ops.butterfly_multiply(
                            twiddle, input_, True, n
                        )
                    p50_cuda, _ = measure_p50_p95(cuda_call)
                    # Restore Triton for the do_bench measurement below.
                    torch_structured._ops.set_backend("triton")

                # W2 — Phase 9 D-65 cross-correlation: ALSO measure the Triton
                # path via triton.testing.do_bench. Satisfies the literal
                # TEST-04 wording ("via triton.testing.do_bench") while keeping
                # the custom harness as the gate driver. The do_bench result is
                # the sibling column for cross-checking.
                # RESEARCH §5 LANDMINE: warmup and rep are TIME budgets in
                # MILLISECONDS, NOT iteration counts.
                do_bench_p50 = None
                if _triton_testing is not None:
                    def triton_call_for_do_bench():
                        return torch_structured._ops.butterfly_multiply(
                            twiddle, input_, True, n
                        )
                    do_bench_result = _triton_testing.do_bench(
                        triton_call_for_do_bench,
                        warmup=25,        # ms
                        rep=100,          # ms
                        quantiles=[0.5, 0.95],
                    )
                    # do_bench returns a tuple (p50, p95) when quantiles given.
                    do_bench_p50 = float(do_bench_result[0])

                row = {
                    "kernel": "butterfly_multiply",
                    "dtype": dtype_name,
                    "log_n": log_n,
                    "nstacks": nstacks,
                    "nblocks": nblocks,
                    "wall_ms_p50": round(p50_triton, 6),
                    "wall_ms_p95": round(p95_triton, 6),
                    "reference_torch_ref_p50": round(p50_ref, 6),
                    "reference_cuda_p50": (round(p50_cuda, 6) if p50_cuda is not None else None),
                    "do_bench_p50_ms": (round(do_bench_p50, 6) if do_bench_p50 is not None else None),
                    "measured_at": datetime.datetime.now(datetime.timezone.utc)
                        .isoformat().replace("+00:00", "Z"),
                    "gpu": gpu_name,
                }
                rows.append(row)

                # Cross-correlation diagnostic (W2) — surface any row where
                # the custom harness and triton.testing.do_bench disagree by
                # more than 15% (tolerance per tolerances block; do_bench has
                # different sync semantics — small drift OK).
                if do_bench_p50 is not None and p50_triton > 0:
                    drift = abs(p50_triton - do_bench_p50) / p50_triton
                    if drift > 0.15:
                        print(
                            f"  WARNING: log_n={log_n} dtype={dtype_name} "
                            f"custom-harness p50={p50_triton:.4f} ms vs "
                            f"do_bench p50={do_bench_p50:.4f} ms drift "
                            f"{drift*100:.1f}% (>15%) — surface in 09-03-SUMMARY.md"
                        )

                cuda_str = f"cuda p50={p50_cuda:.4f} ms  " if p50_cuda is not None else "cuda p50=N/A  "
                db_str = f"do_bench p50={do_bench_p50:.4f} ms  " if do_bench_p50 is not None else "do_bench p50=N/A  "
                print(
                    f"  log_n={log_n} dtype={dtype_name:>9s}: "
                    f"triton p50={p50_triton:.4f} ms p95={p95_triton:.4f} ms  "
                    f"ref p50={p50_ref:.4f} ms  "
                    f"{cuda_str}{db_str}"
                    f"speedup={p50_ref / p50_triton:.2f}x"
                )

    finally:
        # Restore the original backend so this script doesn't leave global state changed.
        torch_structured._ops.set_backend(original_backend)

    out_dir = os.path.dirname(BASELINE_PATH)
    os.makedirs(out_dir, exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump({"rows": rows}, f, indent=2)
        f.write("\n")
    print(f"\nWrote {len(rows)} rows to {BASELINE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
