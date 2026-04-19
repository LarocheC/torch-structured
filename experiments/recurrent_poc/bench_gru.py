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

# Allow running directly as a script by putting the repo root on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch._dynamo
from torch import nn

from experiments.recurrent_poc.gru import StackedGateGRUCell, unroll_cell


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


def _compile_cell_str(r):
    st = r.get("compile_status", "not_requested")
    if st == "ok":
        return f"{r['t_compile_fwd_bwd_ms']:.3f}"
    if st == "compile_fail":
        rsn = r.get("compile_reason", "unknown")
        rsn = rsn if len(rsn) <= 50 else rsn[:47] + "..."
        return f"compile_fail: {rsn}"
    return "-"  # not_requested / skipped_cudnn


def _bench_compile(kind, H, T, batch, device, warmup, iters):
    """Fresh cell + compile(unroll_cell). Try reduce-overhead then default."""
    x = torch.randn(batch, T, H, device=device)
    h0 = torch.zeros(batch, H, device=device)
    cw = max(warmup, 5)  # first compile pass is slow
    last_err = None
    for mode in ("reduce-overhead", "default"):
        try:
            cell = StackedGateGRUCell(input_size=H, hidden_size=H, kind=kind).to(device)
            compiled = torch.compile(unroll_cell, mode=mode, fullgraph=False)

            def run():
                cell.zero_grad(set_to_none=True)
                compiled(cell, x, h0).sum().backward()

            t = time_fn(run, warmup=cw, iters=iters, device=device)
            return {"compile_status": "ok", "t_compile_fwd_bwd_ms": 1e3 * t}
        except Exception as e:  # noqa: BLE001 - dynamo errors are varied
            last_err = e
    return {"compile_status": "compile_fail", "t_compile_fwd_bwd_ms": None,
            "compile_reason": f"fallback-failed: {str(last_err)[:80]}"}


def bench_one(kind, H, T, batch, device, warmup, iters, include_backward, compile_mode):
    """Benchmark one (kind, H, T) combo. compile_mode=None skips compile branch."""
    try:
        x = torch.randn(batch, T, H, device=device)
        if kind == "cudnn":
            model = nn.GRU(H, H, batch_first=True).to(device)
            fwd = lambda: model(x)[0]

            def fwd_bwd():
                model.zero_grad(set_to_none=True)
                out, _ = model(x); out.sum().backward()
        else:
            model = StackedGateGRUCell(input_size=H, hidden_size=H, kind=kind).to(device)
            h0 = torch.zeros(batch, H, device=device)
            fwd = lambda: unroll_cell(model, x, h0)

            def fwd_bwd():
                model.zero_grad(set_to_none=True)
                unroll_cell(model, x, h0).sum().backward()

        t_fwd = time_fn(fwd, warmup=warmup, iters=iters, device=device)
        t_fwd_bwd = (time_fn(fwd_bwd, warmup=warmup, iters=iters, device=device)
                     if include_backward else float("nan"))
        row = {"status": "ok", "kind": kind, "H": H, "T": T,
               "params": sum(p.numel() for p in model.parameters()),
               "t_fwd_ms": 1e3 * t_fwd, "t_fwd_bwd_ms": 1e3 * t_fwd_bwd,
               "compile_status": "not_requested", "t_compile_fwd_bwd_ms": None}
        # cuDNN has its own fused kernels; print "-" in compile column for it.
        if compile_mode is not None:
            row.update({"compile_status": "skipped_cudnn"} if kind == "cudnn"
                       else _bench_compile(kind, H, T, batch, device, warmup, iters))
        return row
    except (RuntimeError, AssertionError, ValueError) as e:
        return {"status": "skipped", "kind": kind, "H": H, "T": T, "reason": str(e)[:80]}


def _print_table(rows, show_compile: bool):
    header = ["kind", "H", "T", "params", "fwd_ms", "fwd_bwd_ms"]
    if show_compile:
        header.append("compile_fwd_bwd_ms")

    def cells(r):
        if r["status"] == "ok":
            base = [r["kind"], str(r["H"]), str(r["T"]), str(r["params"]),
                    f"{r['t_fwd_ms']:.3f}", f"{r['t_fwd_bwd_ms']:.3f}"]
        else:
            base = [r["kind"], str(r["H"]), str(r["T"]), "-",
                    f"SKIPPED — {r['reason']}", ""]
        if show_compile:
            base.append(_compile_cell_str(r) if r["status"] == "ok" else "")
        return base

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
    ap.add_argument("--kinds", default="dense,butterfly,monarch,circulant,cudnn")
    ap.add_argument("--compile", action="store_true",
                    help="Additionally bench torch.compile on the whole unroll loop. "
                         "Uses warmup=max(warmup, 5) for the compile measurement "
                         "because the first pass is slow. Per-combo failures are "
                         "recorded as compile_fail and do not abort the sweep.")
    ap.add_argument("--no-backward", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.hiddens, args.seq_lens = "64", "32"
        args.kinds = "dense,butterfly"
        args.warmup, args.iters = 1, 1

    if args.compile:
        torch._dynamo.config.suppress_errors = True
        os.environ.setdefault("TORCH_LOGS", "")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    hiddens = [int(x) for x in args.hiddens.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    kinds = [x.strip() for x in args.kinds.split(",")]
    include_backward = not args.no_backward
    compile_mode = "reduce-overhead" if args.compile else None

    rows = []
    total = len(kinds) * len(hiddens) * len(seq_lens)
    i = 0
    for kind in kinds:
        for H in hiddens:
            for T in seq_lens:
                i += 1
                print(f"[{i}/{total}] kind={kind} H={H} T={T}", file=sys.stderr, flush=True)
                rows.append(bench_one(kind, H, T, args.batch, device, args.warmup,
                                       args.iters, include_backward, compile_mode))
    _print_table(rows, show_compile=args.compile)


if __name__ == "__main__":
    main()
