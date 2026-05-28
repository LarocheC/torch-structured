---
gsd_state_version: 1.0
milestone: v1.2
milestone_name: Triton Migration
status: executing
stopped_at: Phase 9 context gathered
last_updated: "2026-05-28T12:53:35.929Z"
last_activity: 2026-05-28 -- Phase 09 execution started
progress:
  total_phases: 10
  completed_phases: 4
  total_plans: 13
  completed_plans: 9
  percent: 69
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-26)

**Core value:** A single `uv pip install .` that just works -- with CUDA support when available (v1.2 evolves this to wheel-free Triton JIT)
**Current focus:** Phase 09 — Integration Hardening & Correctness Gates

## Current Position

Phase: 09 (Integration Hardening & Correctness Gates) — EXECUTING
Plan: 1 of 3
Status: Executing Phase 09
Last activity: 2026-05-28 -- Phase 09 execution started

## Performance Metrics

**Velocity:**

- Total plans completed: 9 (v1.0 + v1.1)
- Average duration: -
- Total execution time: unknown

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 1 | - | - |
| 2 | 1 | - | - |
| 3 | 2 | 286s | 143s |
| 4 | 1 (of 2) | ~60min | 60min |
| 05 | 1 | - | - |
| 06 | 2 | - | - |
| 07 | 2 | - | - |

**Recent Trend:**

- v1.0 shipped cleanly in 2 phases, 2 plans
- v1.1 shipped in 1 phase, 2 plans (Phase 3)
- v1.2 Phase 4 started 2026-05-27; Plan 04-01 executed (3 tasks, 10 files, dispatch + docs)

| Phase 03 P01 | 46s | 2 tasks | 185 files |
| Phase 03 P02 | 240s | 2 tasks | 2 files |
| Phase 04 P01 | ~60min | 3 tasks | 10 files (7 created, 3 modified) |
| Phase 04 P02 | ~90min | 3 tasks | 6 files (5 created, 1 modified) |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [v1.0]: Two-phase build modernization shipped successfully
- [v1.1]: All 12 removals are independent; single phase sufficient at coarse granularity
- [Phase 03]: Single commit for all 11 legacy removals -- atomic cleanup
- [Phase 03]: Pre-existing test failures (4/45) documented but not fixed -- out of scope for cleanup
- [v1.2 roadmap]: Derived 7 phases (Phase 4-10) from 28 v1.2 requirements; ordering matches all four research streams (infra -> diag_mult -> hadamard -> butterfly fw -> butterfly bw -> integration -> deprecation)
- [v1.2 roadmap]: Phase 4 carries more design weight than its name suggests -- complex64 layout + `triton_op` pattern propagate to every later phase
- [v1.2 roadmap]: Phase 8 (backward) is highest-risk; 3-layer gradcheck is a phase entry gate; allotted 2 plans for fp32 and complex64 separately
- [v1.2 roadmap]: Phase 10 does deprecation only -- csrc/ deletion deferred to a future milestone (TRI-FUT-04) per the 2-release deprecation cadence
- [v1.2 roadmap]: Existing CUDA path stays working through v1.2; parallel paths during migration
- [v1.2 roadmap]: TEST-04 perf gate (>=60% of CUDA) is a Phase 9 entry criterion, not a Phase 8 blocker
- [Phase 04 Plan 01]: CHECKER B3 honest-resolver fix landed — _has_triton_kernel(op_name) per-op probe distinguishes "Triton importable" from "Triton kernel installed"; _BACKEND reflects actual binding, never requested name
- [Phase 04 Plan 01]: D-08 INFO heads-up dormant in Phase 4 (no triton binding possible); first exercised when Phase 5 ships diag_mult on a host that also has the legacy .so
- [Phase 04 Plan 01]: T-04-01 mitigation in place — _resolve() validates against {triton,cuda,torch,auto} and raises ValueError; no dynamic import of env-var value
- [Phase 04 Plan 01]: torch>=2.6 floor (D-11/COMPAT-05) committed in both [build-system].requires and [project].dependencies
- [Phase 04 Plan 02]: Canonical triton_op + wrap_triton + register_autograd + register_fake skeleton landed in _ops.py._demo_identity_op — template for Phase 5+ kernel ports
- [Phase 04 Plan 02]: @triton.heuristics dropped from kernel decorator stack — wrap_triton in PyTorch >=2.6 only accepts plain @triton.jit or @triton.autotune (Rule 1 auto-fix during execution)
- [Phase 04 Plan 02]: Complex64 wrapper-boundary (view_as_real/view_as_complex) implemented unconditionally in the demonstrator — Phase 7 inherits a working reference
- [Phase 04 Plan 02]: 260419-p27 dynamo bug acceptance gate PASS — register_fake on the demonstrator op prevents "data is not allocated yet" under FakeTensorMode tracing
- [Phase 04 Plan 02]: CI workflow shipped with actions/cache@v4 keyed on (os, python, torch, hashFiles('_triton/**/*.py')) — Pitfall 6 (github.sha keying) avoided

### Pending Todos

- Begin Phase 5 (diag_mult Triton kernel) — Plan 5 will delete _demo_identity_op (per D-13) and extend conftest backend params to ["torch", "triton"]

### Blockers/Concerns

None.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260419-olu | Build POC benchmark suite for recurrent models using torch_structured structured linear layers | 2026-04-19 | 3180d2a | [260419-olu-build-poc-benchmark-suite-for-recurrent-](./quick/260419-olu-build-poc-benchmark-suite-for-recurrent-/) |
| 260419-p27 | Extend recurrent_poc: torch.compile Track A + structured projections in Track B | 2026-04-19 | a2d0c86 | [260419-p27-extend-recurrent-poc-torch-compile-track](./quick/260419-p27-extend-recurrent-poc-torch-compile-track/) |
| 260419-pya | Add LRU and Mamba layers with structured B/C as drop-in nn.GRU peers | 2026-04-19 | 0725d2f | [260419-pya-add-lru-and-mamba-layers-with-structured](./quick/260419-pya-add-lru-and-mamba-layers-with-structured/) |
| 260419-v4b | Promote LRU into torch_structured public API (v0.4.0) for cross-repo use | 2026-04-19 | c7929f3 | [260419-v4b-promote-lru-into-torch-structured-packag](./quick/260419-v4b-promote-lru-into-torch-structured-packag/) |
| 260527-flp | narrow CI workflow to test_dispatch.py only (Phase 4 follow-up) | 2026-05-27 | 1108185 | [260527-flp-narrow-ci-workflow-to-test-dispatch-py-o](./quick/260527-flp-narrow-ci-workflow-to-test-dispatch-py-o/) |

## Session Continuity

Last session: 2026-05-28T08:16:46.890Z
Stopped at: Phase 9 context gathered
Resume file: .planning/phases/09-integration-hardening-correctness-gates/09-CONTEXT.md
