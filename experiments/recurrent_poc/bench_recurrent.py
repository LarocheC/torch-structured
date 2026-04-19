"""Benchmark cuDNN nn.GRU vs StackedGateGRUCell vs LRU vs Mamba.

Run: .venv/bin/python experiments/recurrent_poc/bench_recurrent.py --quick
Full: ... --hiddens 256,512 --seq-lens 500,1000 \\
          --models cudnn,gru,lru,mamba --kinds dense,butterfly
Other: ... --num-layers 2 --bidirectional

SKIPPED rows print for shape-incompatible combos (e.g. cuDNN with a
non-dense kind, the single-layer-unidirectional gru cell POC with a
num-layers>1 or bidirectional request, circulant with non-pow-2/non-square).
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
from torch import nn

from experiments.recurrent_poc.gru import StackedGateGRUCell, unroll_cell
from experiments.recurrent_poc.lru import LRU
from experiments.recurrent_poc.mamba import Mamba


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


def _skipped(model, kind, H, T, num_layers, bidirectional, reason):
    return {"status": "skipped", "model": model, "kind": kind, "H": H, "T": T,
            "num_layers": num_layers, "bidirectional": bidirectional,
            "reason": str(reason)[:80]}


def bench_one(model, kind, H, T, batch, num_layers, bidirectional, device,
              warmup, iters):
    """Benchmark one (model, kind, H, T) combo. Returns a row dict."""
    try:
        x = torch.randn(batch, T, H, device=device)
        if model == "cudnn":
            if kind != "dense":
                return _skipped(model, kind, H, T, num_layers, bidirectional,
                                "cudnn only supports dense")
            m = nn.GRU(H, H, num_layers=num_layers, batch_first=True,
                       bidirectional=bidirectional).to(device)
            fwd = lambda: m(x)[0]

            def fwd_bwd():
                m.zero_grad(set_to_none=True)
                out, _ = m(x); out.sum().backward()
        elif model == "gru":
            if num_layers != 1 or bidirectional:
                return _skipped(model, kind, H, T, num_layers, bidirectional,
                                "gru cell POC is single-layer, unidirectional only")
            cell = StackedGateGRUCell(H, H, kind=kind).to(device)
            h0 = torch.zeros(batch, H, device=device)
            fwd = lambda: unroll_cell(cell, x, h0)

            def fwd_bwd():
                cell.zero_grad(set_to_none=True)
                unroll_cell(cell, x, h0).sum().backward()
            m = cell
        elif model == "lru":
            m = LRU(H, H, num_layers=num_layers, batch_first=True,
                    bidirectional=bidirectional, kind=kind).to(device)
            fwd = lambda: m(x)[0]

            def fwd_bwd():
                m.zero_grad(set_to_none=True)
                out, _ = m(x); out.sum().backward()
        elif model == "mamba":
            m = Mamba(H, H, num_layers=num_layers, batch_first=True,
                      bidirectional=bidirectional, kind=kind).to(device)
            fwd = lambda: m(x)[0]

            def fwd_bwd():
                m.zero_grad(set_to_none=True)
                out, _ = m(x); out.sum().backward()
        else:
            return _skipped(model, kind, H, T, num_layers, bidirectional,
                            "unknown model")

        t_fwd = time_fn(fwd, warmup=warmup, iters=iters, device=device)
        t_fwd_bwd = time_fn(fwd_bwd, warmup=warmup, iters=iters, device=device)
        return {"status": "ok", "model": model, "kind": kind, "H": H, "T": T,
                "num_layers": num_layers, "bidirectional": bidirectional,
                "params": sum(p.numel() for p in m.parameters()),
                "t_fwd_ms": 1e3 * t_fwd, "t_fwd_bwd_ms": 1e3 * t_fwd_bwd}
    except (RuntimeError, AssertionError, ValueError) as e:
        return _skipped(model, kind, H, T, num_layers, bidirectional, str(e))


def _print_table(rows):
    header = ["model", "kind", "H", "T", "layers", "bi", "params",
              "fwd_ms", "fwd_bwd_ms"]

    def cells(r):
        bi = "yes" if r["bidirectional"] else "no"
        base = [r["model"], r["kind"], str(r["H"]), str(r["T"]),
                str(r["num_layers"]), bi]
        if r["status"] == "ok":
            return base + [str(r["params"]),
                           f"{r['t_fwd_ms']:.3f}",
                           f"{r['t_fwd_bwd_ms']:.3f}"]
        return base + [f"SKIPPED — {r['reason']}", "", ""]

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
    ap.add_argument("--hiddens", default="256,512")
    ap.add_argument("--seq-lens", default="500,1000")
    ap.add_argument("--models", default="cudnn,gru,lru,mamba")
    ap.add_argument("--kinds", default="dense,butterfly")
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.hiddens = "64"
        args.seq_lens = "32"
        args.kinds = "dense"
        args.models = "cudnn,gru,lru,mamba"
        args.warmup = 1
        args.iters = 1
        args.batch = 4
        args.num_layers = 1
        args.bidirectional = False

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    hiddens = [int(x) for x in args.hiddens.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    models = [x.strip() for x in args.models.split(",")]
    kinds = [x.strip() for x in args.kinds.split(",")]

    rows = []
    total = len(models) * len(kinds) * len(hiddens) * len(seq_lens)
    i = 0
    for model in models:
        for kind in kinds:
            for H in hiddens:
                for T in seq_lens:
                    i += 1
                    print(f"[{i}/{total}] model={model} kind={kind} H={H} T={T}",
                          file=sys.stderr, flush=True)
                    rows.append(bench_one(model, kind, H, T, args.batch,
                                           args.num_layers, args.bidirectional,
                                           device, args.warmup, args.iters))
    _print_table(rows)


if __name__ == "__main__":
    main()
