"""Benchmark LinearDiagRNN naive (Python for-loop) vs parallel associative scan.

Run: .venv/bin/python experiments/recurrent_poc/bench_lin_rnn.py --quick

If torch.associative_scan (or its _higher_order_ops counterpart) is not
available in this PyTorch build, the scan column reports 'N/A'.
"""

import argparse
import statistics
import sys
import time
import warnings

warnings.filterwarnings("ignore", message=".*different CUDA versions.*")

import torch

from .lin_rnn import LinearDiagRNN, _HAS_SCAN


def time_fn(fn, *, warmup: int, iters: int, device: torch.device) -> float:
    """Median seconds per iteration of fn() with cuda.synchronize on CUDA."""
    is_cuda = device.type == "cuda"
    for _ in range(warmup):
        fn()
    if is_cuda:
        torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if is_cuda:
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def bench_one(kind, H, T, batch, device, warmup, iters):
    try:
        m = LinearDiagRNN(input_size=H, hidden_size=H, kind=kind).to(device)
        x = torch.randn(batch, T, H, device=device)
        t_naive = 1e3 * time_fn(lambda: m.forward_naive(x),
                                warmup=warmup, iters=iters, device=device)
        if _HAS_SCAN:
            t_scan = 1e3 * time_fn(lambda: m.forward_scan(x),
                                   warmup=warmup, iters=iters, device=device)
            speedup = f"{t_naive / t_scan:.2f}x"
            t_scan_s = f"{t_scan:.3f}"
        else:
            t_scan_s, speedup = "N/A", "N/A"
        return {"status": "ok", "kind": kind, "H": H, "T": T,
                "naive_ms": f"{t_naive:.3f}", "scan_ms": t_scan_s, "speedup": speedup}
    except (RuntimeError, AssertionError, ValueError) as e:
        return {"status": "skipped", "kind": kind, "H": H, "T": T,
                "reason": str(e)[:80]}


def _print_table(rows):
    header = ["kind", "H", "T", "naive_ms", "scan_ms", "speedup"]

    def cells(r):
        if r["status"] == "ok":
            return [r["kind"], str(r["H"]), str(r["T"]),
                    r["naive_ms"], r["scan_ms"], r["speedup"]]
        return [r["kind"], str(r["H"]), str(r["T"]),
                f"SKIPPED — {r['reason']}", "", ""]

    all_cells = [header] + [cells(r) for r in rows]
    widths = [max(len(c) for c in col) for col in zip(*all_cells)]
    fmt = lambda cs: " | ".join(c.ljust(widths[i]) for i, c in enumerate(cs))
    print(fmt(header))
    print("-+-".join("-" * w for w in widths))
    for cs in all_cells[1:]:
        print(fmt(cs))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", choices=["cuda", "cpu"],
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--hiddens", default="100,300,1000")
    ap.add_argument("--seq-lens", default="200,500,1000")
    ap.add_argument("--kind", default="dense")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.hiddens, args.seq_lens = "64", "32"
        args.kind = "dense"
        args.warmup, args.iters = 1, 1

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    hiddens = [int(x) for x in args.hiddens.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    rows = []
    total = len(hiddens) * len(seq_lens)
    i = 0
    for H in hiddens:
        for T in seq_lens:
            i += 1
            print(f"[{i}/{total}] kind={args.kind} H={H} T={T}",
                  file=sys.stderr, flush=True)
            rows.append(bench_one(args.kind, H, T, args.batch, device,
                                   args.warmup, args.iters))
    _print_table(rows)


if __name__ == "__main__":
    main()
