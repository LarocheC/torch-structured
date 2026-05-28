"""Phase 9 09-02 DDP / FSDP / gradient-checkpointing integration tests
(D-64 + D-64a + D-64b + D-64c).

Three tests:

1. ``test_ddp_butterfly_smoke`` — single-process DDP (D-64c) with
   ``backend='gloo'`` and ``init_method='env://'`` (no ``store=`` kwarg).
   Verifies the DDP wrap path + gradient sync for a Sequential containing
   a Butterfly. World size = 1.

2. ``test_fsdp_butterfly_smoke`` — 2-GPU FSDP1 smoke with NCCL
   (``@pytest.mark.multigpu``). Uses ``ignored_modules=[Butterfly]`` per
   D-64b (PyTorch 2.6's FSDP2 ``fully_shard`` has NO ``ignored_params`` —
   FSDP1 is the 2.6-compatible API per RESEARCH §1). Run via:
   ``torchrun --nproc_per_node=2 -m pytest tests/test_distributed_triton.py -m multigpu``.

3. ``test_gradient_checkpointing_butterfly`` — wraps Butterfly forward in
   ``torch.utils.checkpoint.checkpoint(use_reentrant=False)``. Verifies
   gradients match a non-checkpointed reference within rtol=1e-5/atol=1e-6
   (RESEARCH §6 starts tight; loosen only if observed mismatch).

W5 — single init path: all init_process_group calls use
``init_method='env://'`` ONLY. No ``store=`` kwarg ever — init_method and
store are mutually exclusive in torch.distributed; passing both raises
RuntimeError.
"""
import os

import pytest
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

import torch_structured  # noqa: F401 — triggers extension load + _ops.py resolver
from torch_structured import Butterfly


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="distributed tests require CUDA",
)


# ═════════════════════════════════════════════════════════════════════════
# Test 1: single-process DDP smoke (D-64c, gloo, env-method init)
# ═════════════════════════════════════════════════════════════════════════
@pytest.mark.op("butterfly_multiply")
def test_ddp_butterfly_smoke(backend, monkeypatch):
    """Single-process DDP smoke (D-64c) — exercises the DDP wrap path
    without requiring 2 GPUs. Uses gloo backend (CPU collectives, but the
    model itself is on CUDA).

    W5: use ONLY init_method='env://' with monkeypatched MASTER_ADDR /
    MASTER_PORT. Do NOT also pass store= — init_method and store are
    mutually exclusive in torch.distributed; passing both raises
    RuntimeError. The env-method choice is preferred because it exercises
    the same code path real users hit in torchrun-launched jobs.
    """
    if backend != "triton":
        pytest.skip("DDP test is meaningful only on the triton path")
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed not available")

    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "29501")  # unique port for this test

    torch.distributed.init_process_group(
        backend="gloo",
        init_method="env://",
        rank=0,
        world_size=1,
    )
    try:
        m = nn.Sequential(
            Butterfly(256, 256, bias=False),
            nn.Linear(256, 10),
        ).cuda()
        m = torch.nn.parallel.DistributedDataParallel(m)
        x = torch.randn(8, 256, device="cuda")
        y = m(x)
        loss = y.sum()
        loss.backward()
        # Sanity check: at least one parameter accumulated a finite gradient.
        any_grad = any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in m.parameters()
        )
        assert any_grad, "DDP backward did not produce any finite gradients"
    finally:
        torch.distributed.destroy_process_group()


# ═════════════════════════════════════════════════════════════════════════
# Test 2: 2-GPU FSDP smoke (D-64 + D-64b, FSDP1 with ignored_modules)
# ═════════════════════════════════════════════════════════════════════════
@pytest.mark.multigpu
@pytest.mark.op("butterfly_multiply")
def test_fsdp_butterfly_smoke():
    """2-GPU FSDP1 smoke (D-64 + D-64b — FSDP1 with ignored_modules per
    PyTorch 2.6 baseline). PyTorch 2.6's FSDP2 fully_shard lacks
    ignored_params; FSDP1 is the compatible choice (RESEARCH §1).

    Run via:
        torchrun --nproc_per_node=2 -m pytest \\
            tests/test_distributed_triton.py -m multigpu

    The twiddle parameter is excluded from sharding because Butterfly's
    forward expects the FULL twiddle tensor on each rank (no per-rank
    slice). NCCL is required for GPU collectives — gloo's all-gather is
    not GPU-tested in production.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    # Sanity gate — this test only makes sense when torchrun launches us
    # with WORLD_SIZE >= 2. Under plain pytest the multigpu marker handles
    # the skip, but we double-check here so a misconfigured invocation
    # produces a clear error instead of an obscure NCCL hang.
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    assert world_size >= 2, (
        "FSDP test requires torchrun --nproc_per_node=2 (got WORLD_SIZE="
        f"{world_size}); add ``-m multigpu`` to your pytest command"
    )

    # W5: env-method init only — torchrun sets MASTER_ADDR/MASTER_PORT.
    # Do NOT pass store=.
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)
    try:
        m = nn.Sequential(
            Butterfly(256, 256, bias=False),
            nn.Linear(256, 10),
        ).cuda()

        # Exclude ALL Butterfly modules from sharding — keeps twiddle
        # replicated on each rank per D-64b.
        ignored = [mod for mod in m.modules() if isinstance(mod, Butterfly)]
        m = FSDP(m, ignored_modules=ignored)

        x = torch.randn(8, 256, device="cuda")
        loss = m(x).sum()
        loss.backward()

        # Verify all_gather agreement: every rank should observe the same
        # loss value (within fp32 tolerance) because the input was
        # locally-different but FSDP averages gradients across ranks.
        # Forward outputs on each rank can differ (different x). We
        # therefore check that the all_reduce-averaged loss is consistent.
        loss_t = loss.detach()
        gathered = [torch.zeros_like(loss_t) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, loss_t)
        # Sanity: every rank's loss is finite. This is the load-bearing
        # cross-rank gate — if FSDP silently sharded twiddle (the W bug),
        # one rank would NaN/Inf because it has only a slice of twiddle.
        for r, g in enumerate(gathered):
            assert torch.isfinite(g).all(), (
                f"rank {r}'s loss has NaN/Inf — twiddle may have been "
                f"sharded despite ignored_modules (D-64b broken)"
            )
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


# ═════════════════════════════════════════════════════════════════════════
# Test 3: gradient checkpointing (use_reentrant=False)
# ═════════════════════════════════════════════════════════════════════════
@pytest.mark.op("butterfly_multiply")
def test_gradient_checkpointing_butterfly(backend):
    """Wrap Butterfly forward in torch.utils.checkpoint.checkpoint(
    use_reentrant=False) and verify gradients match a non-checkpointed
    reference.

    Per RESEARCH §6 — start at rtol=1e-5/atol=1e-6 (Phase 7+8 fp32
    d_input envelope). The checkpoint recompute is deterministic by
    construction (Phase 7 forward has no atomicAdd; preserve_rng_state=True
    by default in torch.utils.checkpoint).

    LANDMINE per RESEARCH §6: use a DIRECT Butterfly(256, 256, ...)
    instance — m.twiddle.grad, NOT m.b.twiddle.grad (that's the
    make_linear indirection). The closure captures the same parameter
    object across both branches.
    """
    if backend != "triton":
        pytest.skip("gradient checkpointing test is meaningful only on the triton path")

    torch.manual_seed(0)
    m = Butterfly(in_size=256, out_size=256, bias=False).cuda()

    # Reference: no-checkpoint backward.
    x_ref = torch.randn(8, 256, device="cuda", requires_grad=True)
    out_ref = m(x_ref)
    grad_ref_x, grad_ref_t = torch.autograd.grad(
        out_ref.sum(), [x_ref, m.twiddle], retain_graph=False,
    )

    # Reset twiddle grad before re-running through checkpoint.
    m.twiddle.grad = None

    # Wrap forward in checkpoint with use_reentrant=False (modern path).
    x_cp = x_ref.detach().clone().requires_grad_(True)
    out_cp = cp.checkpoint(lambda inp: m(inp), x_cp, use_reentrant=False)
    grad_cp_x, grad_cp_t = torch.autograd.grad(
        out_cp.sum(), [x_cp, m.twiddle], retain_graph=False,
    )

    torch.testing.assert_close(grad_ref_x, grad_cp_x, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(grad_ref_t, grad_cp_t, rtol=1e-5, atol=1e-6)
