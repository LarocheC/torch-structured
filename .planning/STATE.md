---
gsd_state_version: 1.0
milestone: v1.2
milestone_name: Triton Migration
status: executing
stopped_at: Phase 4 Plan 04-01 complete; Plan 04-02 pending
last_updated: "2026-05-27T09:30:00.000Z"
last_activity: 2026-05-27 -- Phase 04 Plan 01 executed (dispatch infrastructure + companion docs)
progress:
  total_phases: 10
  completed_phases: 3
  total_plans: 10
  completed_plans: 8
  percent: 80
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-26)

**Core value:** A single `uv pip install .` that just works -- with CUDA support when available (v1.2 evolves this to wheel-free Triton JIT)
**Current focus:** Phase 4 -- Triton Dispatch Infrastructure & Foundational Decisions

## Current Position

Phase: 4 (Triton Dispatch Infrastructure & Foundational Decisions) -- in progress
Plan: 04-02 (demonstrator op + test_dispatch + CI cache) -- pending
Status: Plan 04-01 complete; Plan 04-02 ready to execute
Last activity: 2026-05-27 -- Plan 04-01 executed (dispatch infrastructure + companion docs)

## Performance Metrics

**Velocity:**

- Total plans completed: 4 (v1.0 + v1.1)
- Average duration: -
- Total execution time: unknown

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 1 | - | - |
| 2 | 1 | - | - |
| 3 | 2 | 286s | 143s |
| 4 | 1 (of 2) | ~60min | 60min |

**Recent Trend:**

- v1.0 shipped cleanly in 2 phases, 2 plans
- v1.1 shipped in 1 phase, 2 plans (Phase 3)
- v1.2 Phase 4 started 2026-05-27; Plan 04-01 executed (3 tasks, 10 files, dispatch + docs)

| Phase 03 P01 | 46s | 2 tasks | 185 files |
| Phase 03 P02 | 240s | 2 tasks | 2 files |
| Phase 04 P01 | ~60min | 3 tasks | 10 files (7 created, 3 modified) |

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

### Pending Todos

- Execute Plan 04-02 (demonstrator op + test_dispatch.py + CI cache)

### Blockers/Concerns

None.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260419-olu | Build POC benchmark suite for recurrent models using torch_structured structured linear layers | 2026-04-19 | 3180d2a | [260419-olu-build-poc-benchmark-suite-for-recurrent-](./quick/260419-olu-build-poc-benchmark-suite-for-recurrent-/) |
| 260419-p27 | Extend recurrent_poc: torch.compile Track A + structured projections in Track B | 2026-04-19 | a2d0c86 | [260419-p27-extend-recurrent-poc-torch-compile-track](./quick/260419-p27-extend-recurrent-poc-torch-compile-track/) |
| 260419-pya | Add LRU and Mamba layers with structured B/C as drop-in nn.GRU peers | 2026-04-19 | 0725d2f | [260419-pya-add-lru-and-mamba-layers-with-structured](./quick/260419-pya-add-lru-and-mamba-layers-with-structured/) |
| 260419-v4b | Promote LRU into torch_structured public API (v0.4.0) for cross-repo use | 2026-04-19 | c7929f3 | [260419-v4b-promote-lru-into-torch-structured-packag](./quick/260419-v4b-promote-lru-into-torch-structured-packag/) |

## Session Continuity

Last session: 2026-05-27T09:30:00.000Z
Stopped at: Phase 4 Plan 04-01 executed; Plan 04-02 pending
Resume file: .planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/04-02-PLAN.md
