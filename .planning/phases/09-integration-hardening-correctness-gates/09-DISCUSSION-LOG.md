# Phase 9: Integration Hardening & Correctness Gates - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-28
**Phase:** 9-Integration Hardening & Correctness Gates
**Areas discussed:** Plan split, Perf gate fallback, CUDA conftest skip-gate, Deterministic mode API, FSDP venue, Perf grid shape, Runtime selector location

---

## Plan Split

**Question:** How should the 3 plans split the work? Phase 9 has 10 requirements (TEST-01..04, TEST-06, COMPAT-01..04, COMPAT-06) and 5 SCs.

| Option | Description | Selected |
|--------|-------------|----------|
| By SC group — foundations / compose / perf+docs | 09-01: SC#1+SC#3 (conftest axis, end-to-end pytest, checkpoint round-trip). 09-02: SC#2 (torch.compile, DDP, FSDP, grad-checkpoint, deterministic). 09-03: SC#4+SC#5 (perf grid, runtime selector, README). | ✓ |
| By risk — low-risk first, perf last | Reorder by risk: low (test infra + compat), medium (compose), highest (perf gate). | |
| By dependency — unblock test infra first | 09-01 narrow scope (CUDA axis), 09-02 huge testing surface, 09-03 perf+docs. | |

**User's choice:** By SC group — foundations / compose / perf+docs (Recommended)
**Notes:** Captured as D-60. Wave structure D-60a: 09-01 wave 1, 09-02 wave 2 (depends on 09-01 conftest), 09-03 wave 3 (depends on both).

---

## Perf Gate Fallback Strategy

**Question:** What's the fallback strategy if Triton 3-stage kernel is below 60% of CUDA throughput on some perf-grid cells?

| Option | Description | Selected |
|--------|-------------|----------|
| Runtime selector routes below-60% cells to CUDA | Per-shape routing rule in _ops.py. Tension with Phase 8 SC#4 explicitly resolved via D-61a. | ✓ |
| Implement 5-stage tile kernel | Phase 7 deferred 5-stage; would be Phase 9 work. ~1 wave of kernel engineering. Pure-Triton, preserves SC#4. | |
| Document the shortfall, ship as-is | Don't add selector or 5-stage; document in README and let users opt to BACKEND=cuda. | |

**User's choice:** Runtime selector routes below-60% cells to CUDA (Recommended)
**Notes:** Captured as D-61 + D-61a (Phase 8 SC#4 reconciliation) + D-61b (CUDA-missing fallback chain).

---

## CUDA Conftest Skip-Gate

**Question:** How should the conftest `backend` fixture handle the new `cuda` parameter when `_butterfly.so` (or `_diag_mult.so` / `_hadamard.so`) is missing?

| Option | Description | Selected |
|--------|-------------|----------|
| Per-op skip-gate (mirrors Phase 6 D-39) | Granular `_has_cuda_legacy_for_op(op)` probe. Per-test skip based on the op-under-test. | ✓ |
| All-or-nothing global skip-gate | Single `_has_any_cuda_kernel()` probe; skip cuda param if ANY .so is missing. | |
| Hard requirement — fail loudly | Don't skip; raise an error pointing the user to build the .so. | |

**User's choice:** Per-op skip-gate (mirrors Phase 6 D-39) (Recommended)
**Notes:** Captured as D-62. New `_has_cuda_legacy_for_op` per D-62a. Op-name detection via pytest marker recommended (D-62b option 1). Tolerances inherited from Phase 7+8 per D-62c.

---

## Deterministic Mode Opt-In API

**Question:** What's the deterministic-mode opt-in API surface?

| Option | Description | Selected |
|--------|-------------|----------|
| Function call (`torch_structured.set_deterministic(True)`) | Top-level function setting process-level flag. Mirrors `set_backend()`. | ✓ |
| Environment variable (`TORCH_STRUCTURED_DETERMINISTIC=1`) | Read at module import; specializes kernel at JIT time via tl.constexpr. | |
| Context manager (`with torch_structured.deterministic(): ...`) | Block-scoped opt-in. Cleanest test ergonomics. | |
| Honor `torch.are_deterministic_algorithms_enabled()` only | No new API; honor PyTorch's existing flag. | |

**User's choice:** Function call (Recommended)
**Notes:** Captured as D-63. Kernel-level mechanism is wrapper-level oracle fallback per D-63a (composable, simpler). Also honors `torch.are_deterministic_algorithms_enabled()` per D-63b. Exported via `__init__.py` per D-63c.

---

## FSDP Test Venue

**Question:** Where does the 2-GPU FSDP smoke test run?

| Option | Description | Selected |
|--------|-------------|----------|
| Run via torchrun in CI; mark @pytest.mark.multigpu | Custom marker registered in conftest; CI job uses 2-GPU runner. Skips on <2 GPUs. | ✓ |
| Skip in CI, run manually + document | Mark `@pytest.mark.skip` by default; opt-in with `pytest -m multigpu`. | |
| Inline single-process FSDP via spawn | `torch.distributed.spawn(fn, nprocs=2)` within a single pytest worker. | |

**User's choice:** Run via torchrun in CI (Recommended)
**Notes:** Captured as D-64. CI job split per D-64a. FSDP twiddle no-shard hint per D-64b (planner picks the exact PyTorch 2.6+ API). Single-process DDP smoke complement per D-64c.

---

## Perf Grid Shape

**Question:** What's the perf grid shape for TEST-04 (Triton ≥ 60% of CUDA on every cell)?

| Option | Description | Selected |
|--------|-------------|----------|
| Match Phase 7+8 baseline grid: log_n × dtype × direction | 16-row schema at batch=64. Add ≥60% gate against `reference_cuda_p50` column (added in 09-03). | ✓ |
| Wider grid: add batch ∈ {1, 16, 64, 256, 4096} | 80 cells. Catches batch-dependent throughput cliffs. 5× measurement time. | |
| Tighter grid: focus on log_n=11 only | 4 cells. Strict gate where it matters most. Smaller cells ungated. | |

**User's choice:** Match Phase 7+8 baseline grid (Recommended)
**Notes:** Captured as D-65. CUDA p50 column added per D-65a. Gate computation per D-65b: threshold 1.67×. Wider grids deferred per D-65c.

---

## Runtime Selector Location

**Question:** Where does the runtime selector live and what triggers below-60% routing?

| Option | Description | Selected |
|--------|-------------|----------|
| In _ops.py, baked from 07-BASELINE.json measured values | Static routing table in `_routing.json`, committed to repo, regenerated by harness script. | ✓ |
| Dynamic runtime probe — measure on first call per shape | First-call bench, cache winner, route subsequent calls. Adapts to hardware. | |
| No selector — ship slow cells, document in README | Don't add selector; user opts to BACKEND=cuda for slow cells. | |

**User's choice:** In _ops.py, baked from 07-BASELINE.json (Recommended)
**Notes:** Captured as D-66. Function shape per D-66a. Resolver hook per D-66b. Test-time override `_DISABLE_ROUTING` per D-66c. Logging per D-66d.

---

## Claude's Discretion

The user selected all four "Recommended" answers in the first round (Plan split, Perf gate, CUDA skip, Deterministic API) and all three "Recommended" in the second round (FSDP venue, Perf grid, Runtime selector). The following items remain planner-flexible per the CONTEXT.md `### Claude's Discretion` block:

- Exact form of `_detect_op_from_test_name` in conftest — pytest marker vs name-pattern (D-62b)
- Exact PyTorch FSDP `ignored_modules` / `wrap_policy` idiom for PyTorch 2.6+ (D-64b)
- Deterministic path mechanism — oracle fallback (D-63a) vs sequential-atomicAdd kernel path
- Perf-grid harness CUDA measurement vs reusing an existing CUDA bench (D-65a)
- `scripts/regenerate_routing_table.py` location — `scripts/` vs `.planning/`
- Logging level for runtime selector first-route — `info` (D-66d) vs `warning` (D-61b)
- README section structure for Triton backend documentation (D-60 / 09-03)
- CI matrix entry for `BACKEND=cuda` testing — additional job vs reuse existing
- Order of perf-gate failures in `tests/test_perf_grid.py` reporting

## Deferred Ideas

- 5-stage tile kernel (rejected — runtime selector covers below-60% cells)
- Wider perf grid (per-batch axis) — v1.2 anchors on batch=64
- `@triton.autotune` over num_warps / TILE_N (Phase 9 uses fixed schedule from Phase 7 D-40d)
- Dynamic runtime perf probing (rejected — first-call latency, non-determinism)
- bf16 / fp16 (TRI-FUT-01)
- log_n > 11
- ROCm / AMD / Intel XPU validation (PLAT-01, PLAT-02)
- Pre-compiled wheels (DIST-01, DIST-02)
- Deterministic-mode kernel-side specialization (Phase 9 uses wrapper-level gate)
- Stack-trace-based SC#4 verification (Phase 8's dispatch-binding mechanism inherited)
- CHANGELOG migration guide for v1.0/v1.1 users (Phase 10 owns deprecation cadence)
