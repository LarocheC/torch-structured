"""Phase 8 Plan 08-02 perf baseline harness for butterfly_multiply BACKWARD.

Standalone measurement script — NOT a pytest test. Leading underscore in the
filename prevents pytest auto-collection. Invoke manually exactly once at end
of Plan 08-02 execution to extend the JSON baseline that Phase 9 TEST-04
(parity gate) reads verbatim.

Per CONTEXT.md D-51 + 08-PATTERNS.md the file is EXTENDED IN-PLACE — does
NOT create a new 08-BASELINE.json. The schema gains a ``direction`` field on
every row (existing 8 forward rows get ``direction='forward'`` added; 8 new
backward rows are appended with ``direction='backward'`` covering the same
grid ``log_n in {8, 9, 10, 11} x dtype in {fp32, complex64}``).

Schema (locked at CONTEXT.md D-43b, extended by D-51 + Phase 9 D-65a + W2):

    {
      "rows": [
        {
          "kernel": "butterfly_multiply",
          "direction": "forward" | "backward",
          "dtype": "fp32" | "complex64",
          "log_n": 8 | 9 | 10 | 11,
          "nstacks": 1,
          "nblocks": 1,
          "wall_ms_p50": <float>,
          "wall_ms_p95": <float>,
          "reference_torch_ref_p50": <float>,
          "reference_cuda_p50": <float | null>,  # Phase 9 D-65a
          "do_bench_p50_ms": <float | null>,     # Phase 9 W2
          "measured_at": "<ISO8601 UTC>",
          "gpu": "<torch.cuda.get_device_name(0)>"
        },
        ...
      ]
    }

After extension: 16 rows total (8 forward + 8 backward).

Measurement protocol (mirrors 07-02 forward harness):
- batch_size = 64, nstacks = 1, nblocks = 1, increasing_stride = True,
  output_size = n.
- Warmup 10 iterations + measure 100 iterations.
- ``torch.cuda.Event(enable_timing=True)`` start/end pairs around each
  iteration; sync after each iteration so timing is per-iteration wall-clock.
- Triton backward measured via ``out.backward(grad_out, retain_graph=True)``
  inside the do_bench loop — measures the FULL backward callback overhead
  including the trail recompute (RESEARCH §"Performance Baseline Strategy"
  option (a)).
- Oracle reference measured via ``torch.autograd.grad(butterfly_ref(twiddle,
  input_, True, n), [twiddle, input_], grad_out)`` — re-runs the forward
  each iteration (no retain_graph available for the oracle path).

On a CPU-only host this script prints SKIP and exits 0 — the extension is
then deferred and Plan 08-02 SUMMARY.md records the deferral.

Invocation: ``python tests/_baseline_butterfly_backward.py`` from the repo
root. The pytest auto-collection guard (leading underscore) ensures this
script is NOT discovered by ``pytest tests/``.
"""
import datetime
import json
import os
import sys

import torch

import torch_structured  # noqa: F401 — triggers extension load + _ops.py resolver
from torch_structured._torch_ref.butterfly import butterfly_multiply_torch as butterfly_ref

# Phase 9 D-65 W2 — cross-correlation: import triton.testing for do_bench.
try:
    import triton.testing as _triton_testing
except ImportError:
    _triton_testing = None


BASELINE_PATH = ".planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json"


def measure_p50_p95(fn, *, warmup: int = 10, n_iter: int = 100) -> tuple[float, float]:
    """Return (p50, p95) per-iteration wall-clock GPU time (milliseconds).

    Mirrors the forward harness in ``_baseline_butterfly.py``.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(n_iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    return p50, p95


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: no CUDA — backward baseline extension deferred")
        return 0

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")

    # ------------------------------------------------------------------
    # Step 1: Read existing baseline file and add ``direction='forward'``
    # to every existing row (per CONTEXT.md D-51).
    # ------------------------------------------------------------------
    with open(BASELINE_PATH) as f:
        data = json.load(f)
    existing_rows = data["rows"]
    print(f"Loaded {len(existing_rows)} existing rows from {BASELINE_PATH}")
    for row in existing_rows:
        row["direction"] = "forward"

    # ------------------------------------------------------------------
    # Step 2: Measure backward p50/p95 at the same grid.
    # ------------------------------------------------------------------
    original_backend = torch_structured._ops._BACKEND
    new_rows: list[dict] = []

    try:
        for log_n in (8, 9, 10, 11):
            for dtype, dtype_name in ((torch.float32, "fp32"),
                                      (torch.complex64, "complex64")):
                n = 1 << log_n
                batch_size, nstacks, nblocks = 64, 1, 1

                torch.manual_seed(42)
                twiddle = torch.randn(
                    nstacks, nblocks, log_n, n // 2, 2, 2,
                    device="cuda", dtype=dtype, requires_grad=True,
                )
                input_ = torch.randn(
                    batch_size, nstacks, n,
                    device="cuda", dtype=dtype, requires_grad=True,
                )
                grad_out = torch.randn(
                    batch_size, nstacks, n, device="cuda", dtype=dtype,
                )

                # Triton backward — retain_graph=True so the forward graph is
                # reused across 100 iterations; measures the FULL backward
                # callback overhead including the trail recompute.
                torch_structured._ops.set_backend("triton")
                out_triton = torch_structured._ops.butterfly_multiply(
                    twiddle, input_, True, n,
                )
                def triton_backward():
                    # Zero existing grads so we measure only the new backward
                    # contribution (the autograd graph accumulates by default).
                    if twiddle.grad is not None:
                        twiddle.grad = None
                    if input_.grad is not None:
                        input_.grad = None
                    out_triton.backward(grad_out, retain_graph=True)
                p50_triton, p95_triton = measure_p50_p95(triton_backward)

                # Oracle reference — re-run forward each iteration (no
                # retain_graph for the oracle path; the recreated graph is
                # discarded after each backward).
                def ref_backward():
                    t_o = twiddle.detach().clone().requires_grad_()
                    x_o = input_.detach().clone().requires_grad_()
                    out_o = butterfly_ref(t_o, x_o, True, n)
                    torch.autograd.grad(out_o, [t_o, x_o], grad_out)
                p50_ref, _ = measure_p50_p95(ref_backward)

                # Phase 9 D-65a: CUDA backward p50 (same gate predicate as
                # forward). W6 — CRITICAL: detach + clone before switching
                # backends to break the cross-backend autograd graph reuse.
                # Without this, the Triton block's requires_grad tensors carry
                # their .grad and graph state into the CUDA block, which can
                # produce double-counted gradients or stale-grad reuse under
                # retain_graph=True. The detach severs the graph; the clone
                # gives a fresh storage; requires_grad_(True) rebuilds a clean
                # leaf node.
                twiddle = twiddle.detach().clone().requires_grad_(True)
                input_ = input_.detach().clone().requires_grad_(True)

                p50_cuda = None
                if torch_structured._ops._has_cuda_legacy():
                    torch_structured._ops.set_backend("cuda")
                    # Build a FRESH forward graph under the CUDA backend; the
                    # detach + clone above gave us fresh leaves.
                    out_cuda = torch_structured._ops.butterfly_multiply(
                        twiddle, input_, True, n,
                    )
                    def cuda_backward():
                        if twiddle.grad is not None:
                            twiddle.grad = None
                        if input_.grad is not None:
                            input_.grad = None
                        out_cuda.backward(grad_out, retain_graph=True)
                    p50_cuda, _ = measure_p50_p95(cuda_backward)

                # W2 — Phase 9 D-65 cross-correlation: ALSO measure Triton
                # backward via do_bench. Detach + clone again before the
                # do_bench block — switching back from CUDA to Triton triggers
                # another fresh autograd graph and avoids state leakage.
                twiddle = twiddle.detach().clone().requires_grad_(True)
                input_ = input_.detach().clone().requires_grad_(True)

                do_bench_p50 = None
                if _triton_testing is not None:
                    torch_structured._ops.set_backend("triton")
                    out_triton_db = torch_structured._ops.butterfly_multiply(
                        twiddle, input_, True, n,
                    )
                    def triton_backward_for_do_bench():
                        if twiddle.grad is not None:
                            twiddle.grad = None
                        if input_.grad is not None:
                            input_.grad = None
                        out_triton_db.backward(grad_out, retain_graph=True)
                    do_bench_result = _triton_testing.do_bench(
                        triton_backward_for_do_bench,
                        warmup=25,        # ms — same params as forward (W2)
                        rep=100,          # ms
                        quantiles=[0.5, 0.95],
                    )
                    do_bench_p50 = float(do_bench_result[0])

                row = {
                    "kernel": "butterfly_multiply",
                    "direction": "backward",
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
                new_rows.append(row)

                # Cross-correlation diagnostic (W2)
                if do_bench_p50 is not None and p50_triton > 0:
                    drift = abs(p50_triton - do_bench_p50) / p50_triton
                    if drift > 0.15:
                        print(
                            f"  WARNING: backward log_n={log_n} dtype={dtype_name} "
                            f"custom-harness p50={p50_triton:.4f} ms vs "
                            f"do_bench p50={do_bench_p50:.4f} ms drift "
                            f"{drift*100:.1f}% (>15%) — surface in 09-03-SUMMARY.md"
                        )

                cuda_str = f"cuda p50={p50_cuda:.4f} ms  " if p50_cuda is not None else "cuda p50=N/A  "
                db_str = f"do_bench p50={do_bench_p50:.4f} ms  " if do_bench_p50 is not None else "do_bench p50=N/A  "
                print(
                    f"  backward log_n={log_n} dtype={dtype_name:>9s}: "
                    f"triton p50={p50_triton:.4f} ms p95={p95_triton:.4f} ms  "
                    f"ref p50={p50_ref:.4f} ms  "
                    f"{cuda_str}{db_str}"
                    f"speedup={p50_ref / p50_triton:.2f}x"
                )

    finally:
        torch_structured._ops.set_backend(original_backend)

    # ------------------------------------------------------------------
    # Step 3: Write extended file back (preserving existing forward rows).
    # ------------------------------------------------------------------
    all_rows = existing_rows + new_rows
    out_dir = os.path.dirname(BASELINE_PATH)
    os.makedirs(out_dir, exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump({"rows": all_rows}, f, indent=2)
        f.write("\n")
    print(
        f"\nExtended {BASELINE_PATH}: "
        f"{len(existing_rows)} forward + {len(new_rows)} backward = "
        f"{len(all_rows)} total rows"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
