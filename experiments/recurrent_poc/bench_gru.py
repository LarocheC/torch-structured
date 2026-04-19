"""Benchmark StackedGateGRUCell unrolled over T steps vs cuDNN nn.GRU.

Run: .venv/bin/python experiments/recurrent_poc/bench_gru.py --quick
Full sweep: ... --hiddens 100,300,1000 --seq-lens 200,500,1000 \
            --kinds dense,butterfly,monarch,circulant,cudnn

The benign "...different CUDA versions" warning is suppressed at import.
Output: a single plain-text table on stdout. No JSON, no file writes.
"""

import argparse
import os
import statistics
import sys
import time
import warnings

warnings.filterwarnings("ignore", message=".*different CUDA versions.*")

# Allow running directly as a script (python path/to/bench_gru.py) by
# putting the repo root on sys.path before the package import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch import nn

from experiments.recurrent_poc.gru import StackedGateGRUCell


def time_fn(fn, *, warmup: int, iters: int, device: torch.device) -> float:
    """Run fn() for `warmup` un-timed then `iters` timed iterations.

    Returns median seconds per iteration. Uses cuda.synchronize when the
    target device is CUDA, plain wall-clock otherwise.
    """
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


def run_cell(cell, x_bt, h0):
    """Unroll a (B, T, H) input through `cell`; return final hidden state."""
    h = h0
    for t in range(x_bt.size(1)):
        h = cell(x_bt[:, t], h)
    return h


def bench_one(kind, H, T, batch, device, warmup, iters, include_backward, compile_cell):
    """Benchmark one (kind, H, T) combo. Returns a result dict (status=ok|skipped)."""
    try:
        if kind == "cudnn":
            model = nn.GRU(H, H, batch_first=True).to(device)
            x = torch.randn(batch, T, H, device=device)

            def fwd():
                out, _ = model(x); return out

            def fwd_bwd():
                model.zero_grad(set_to_none=True)
                out, _ = model(x); out.sum().backward()
        else:
            model = StackedGateGRUCell(input_size=H, hidden_size=H, kind=kind).to(device)
            if compile_cell:
                model = torch.compile(model, mode="reduce-overhead")
            x = torch.randn(batch, T, H, device=device)
            h0 = torch.zeros(batch, H, device=device)

            def fwd():
                return run_cell(model, x, h0)

            def fwd_bwd():
                model.zero_grad(set_to_none=True)
                out = run_cell(model, x, h0); out.sum().backward()

        params = sum(p.numel() for p in model.parameters())
        t_fwd = time_fn(fwd, warmup=warmup, iters=iters, device=device)
        t_fwd_bwd = (time_fn(fwd_bwd, warmup=warmup, iters=iters, device=device)
                     if include_backward else float("nan"))
        return {"status": "ok", "kind": kind, "H": H, "T": T, "params": params,
                "t_fwd_ms": 1e3 * t_fwd, "t_fwd_bwd_ms": 1e3 * t_fwd_bwd}
    except (RuntimeError, AssertionError, ValueError) as e:
        return {"status": "skipped", "kind": kind, "H": H, "T": T, "reason": str(e)[:80]}


def _row_cells(r, header):
    if r["status"] == "ok":
        return [r["kind"], str(r["H"]), str(r["T"]), str(r["params"]),
                f"{r['t_fwd_ms']:.3f}", f"{r['t_fwd_bwd_ms']:.3f}"]
    return [r["kind"], str(r["H"]), str(r["T"]), "-",
            f"SKIPPED — {r['reason']}", ""]


def _print_table(rows):
    header = ["kind", "H", "T", "params", "fwd_ms", "fwd_bwd_ms"]
    all_cells = [header] + [_row_cells(r, header) for r in rows]
    widths = [max(len(c) for c in col) for col in zip(*all_cells)]
    fmt = lambda cells: " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells))
    print(fmt(header))
    print("-+-".join("-" * w for w in widths))
    for cells in all_cells[1:]:
        print(fmt(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", choices=["cuda", "cpu"],
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--hiddens", default="100,300,1000")
    ap.add_argument("--seq-lens", default="200,500,1000")
    ap.add_argument("--kinds", default="dense,butterfly,monarch,circulant,cudnn")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-backward", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.hiddens, args.seq_lens = "64", "32"
        args.kinds = "dense,butterfly"
        args.warmup, args.iters = 1, 1

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    hiddens = [int(x) for x in args.hiddens.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    kinds = [x.strip() for x in args.kinds.split(",")]
    include_backward = not args.no_backward

    rows = []
    total = len(kinds) * len(hiddens) * len(seq_lens)
    i = 0
    for kind in kinds:
        for H in hiddens:
            for T in seq_lens:
                i += 1
                print(f"[{i}/{total}] kind={kind} H={H} T={T}", file=sys.stderr, flush=True)
                rows.append(bench_one(kind, H, T, args.batch, device,
                                       args.warmup, args.iters,
                                       include_backward, args.compile))
    _print_table(rows)


if __name__ == "__main__":
    main()
