"""Benchmark LinearDiagRNN naive (Python for-loop) vs parallel associative scan.

Run: .venv/bin/python experiments/recurrent_poc/bench_lin_rnn.py --quick
     .venv/bin/python experiments/recurrent_poc/bench_lin_rnn.py --kinds dense,butterfly
     .venv/bin/python experiments/recurrent_poc/bench_lin_rnn.py --kind-B butterfly --kind-C dense

If torch.associative_scan (or its _higher_order_ops counterpart) is not
available in this PyTorch build, the scan column reports 'N/A'.
"""

import argparse
import os
import statistics
import sys
import time
import warnings

warnings.filterwarnings("ignore", message=".*different CUDA versions.*")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from experiments.recurrent_poc.lin_rnn import LinearDiagRNN, _HAS_SCAN


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


def bench_one(kind_B, kind_C, H, T, batch, device, warmup, iters):
    # In this POC all dims == H (square), so the only skips we expect are
    # from kind internals (e.g. circulant requires square pow-2; butterfly
    # handles non-pow-2 by internal padding).
    try:
        m = LinearDiagRNN(input_size=H, hidden_size=H,
                          kind_B=kind_B, kind_C=kind_C).to(device)
        x = torch.randn(batch, T, H, device=device)
        params = sum(p.numel() for p in m.parameters())
        t_naive = 1e3 * time_fn(lambda: m.forward_naive(x),
                                warmup=warmup, iters=iters, device=device)
        if _HAS_SCAN:
            t_scan = 1e3 * time_fn(lambda: m.forward_scan(x),
                                   warmup=warmup, iters=iters, device=device)
            speedup = f"{t_naive / t_scan:.2f}x"
            t_scan_s = f"{t_scan:.3f}"
        else:
            t_scan_s, speedup = "N/A", "N/A"
        return {"status": "ok", "kind_B": kind_B, "kind_C": kind_C,
                "H": H, "T": T, "params": str(params),
                "naive_ms": f"{t_naive:.3f}", "scan_ms": t_scan_s, "speedup": speedup}
    except (RuntimeError, AssertionError, ValueError) as e:
        return {"status": "skipped", "kind_B": kind_B, "kind_C": kind_C,
                "H": H, "T": T, "reason": str(e)[:80]}


def _print_table(rows, asymmetric: bool):
    if asymmetric:
        header = ["kind_B", "kind_C", "H", "T", "params", "naive_ms", "scan_ms", "speedup"]
    else:
        header = ["kind", "H", "T", "params", "naive_ms", "scan_ms", "speedup"]

    def cells(r):
        lead = ([r["kind_B"], r["kind_C"]] if asymmetric else [r["kind_B"]])
        if r["status"] == "ok":
            return lead + [str(r["H"]), str(r["T"]), r["params"],
                           r["naive_ms"], r["scan_ms"], r["speedup"]]
        # SKIPPED: fill kind cell(s), H, T, blank params, then reason in
        # naive_ms; scan_ms and speedup blank.
        skipped_cell = f"SKIPPED — {r['reason']}"
        n_trailing = len(header) - len(lead) - 4  # minus lead, H, T, params, naive_ms (4 after lead)
        return lead + [str(r["H"]), str(r["T"]), "", skipped_cell] + [""] * n_trailing

    grid = [header] + [cells(r) for r in rows]
    widths = [max(len(c) for c in col) for col in zip(*grid)]
    fmt = lambda cs: " | ".join(c.ljust(widths[i]) for i, c in enumerate(cs))
    print(fmt(header)); print("-+-".join("-" * w for w in widths))
    for cs in grid[1:]:
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
    ap.add_argument("--kinds", default=None,
                    help="Comma-separated list applied symmetrically to B and C. "
                         "Mutually exclusive with --kind-B/--kind-C.")
    ap.add_argument("--kind-B", default=None,
                    help="Single kind for the input projection B. Requires --kind-C.")
    ap.add_argument("--kind-C", default=None,
                    help="Single kind for the output projection C. Requires --kind-B.")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    symmetric = args.kinds is not None
    asymmetric = args.kind_B is not None or args.kind_C is not None
    if symmetric and asymmetric:
        print("error: --kinds is mutually exclusive with --kind-B/--kind-C",
              file=sys.stderr)
        sys.exit(2)
    if asymmetric and (args.kind_B is None or args.kind_C is None):
        print("error: --kind-B and --kind-C must both be supplied",
              file=sys.stderr)
        sys.exit(2)

    if args.quick:
        args.hiddens, args.seq_lens = "64", "32"
        if not symmetric and not asymmetric:
            args.kinds = "dense,butterfly"
            symmetric = True
        args.warmup, args.iters = 1, 1
    elif not symmetric and not asymmetric:
        args.kinds = "dense"  # default for backwards compat
        symmetric = True

    if symmetric:
        combos = [(k.strip(), k.strip()) for k in args.kinds.split(",")]
    else:
        combos = [(args.kind_B, args.kind_C)]

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    hiddens = [int(x) for x in args.hiddens.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    rows = []
    total = len(combos) * len(hiddens) * len(seq_lens)
    i = 0
    for kb, kc in combos:
        for H in hiddens:
            for T in seq_lens:
                i += 1
                print(f"[{i}/{total}] kind_B={kb} kind_C={kc} H={H} T={T}",
                      file=sys.stderr, flush=True)
                rows.append(bench_one(kb, kc, H, T, args.batch, device,
                                       args.warmup, args.iters))
    _print_table(rows, asymmetric=asymmetric)


if __name__ == "__main__":
    main()
