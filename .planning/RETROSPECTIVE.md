# Project Retrospective

## Milestone: v1.0 — Build System Modernization

**Shipped:** 2026-04-02
**Phases:** 2 | **Plans:** 4

### What Was Built
- pyproject.toml with PEP 621 metadata, torch>=2.0 build dep, setuptools>=64 backend
- Thin setup.py shim with CUDA arch targeting (7.0 8.0 9.0+PTX via TORCH_CUDA_ARCH_LIST)
- Glob-based __file__-relative extension loading replacing fragile PathFinder
- TORCH_LIBRARY_FRAGMENT migration in version.cpp
- CUDA version check downgraded from RuntimeError to UserWarning
- scipy made lazy import in permutation.py

### What Worked
- Research-first approach: domain research identified the exact patterns (pytorch_scatter, pytorch/extension-cpp) before planning
- Parallel execution of Wave 1 plans saved time
- Incremental verification during execution caught real issues (absolute paths, license format, TORCH_LIBRARY collision) early

### What Was Inefficient
- Plan 01-01 agent ended up doing 01-02's work too (setup.py rewrite) — the two plans could have been one
- The 01-02 agent launched in a worktree but couldn't find its plan file (plans were committed after worktree creation)
- Build artifacts (build/, egg-info/) needed manual cleanup between test iterations

### Patterns Established
- Thin setup.py shim pattern for PyTorch C++ extensions with pyproject.toml
- TORCH_CUDA_ARCH_LIST as the standard env var for CUDA architecture control
- Glob-based .so discovery with platform-aware suffixes (.so, .pyd, .dylib)
- TORCH_LIBRARY_FRAGMENT when multiple .so files register ops in the same namespace

### Key Lessons
- Always clean build/ and egg-info/ between install tests — stale artifacts cause confusing failures
- setuptools version matters for license field format (SPDX string needs >=77, table format works with >=64)
- Worktree isolation + plan files committed separately = race condition. Consider committing plans before spawning worktree agents.

## Milestone: v1.2 — Triton Migration

**Shipped:** 2026-05-29 (PyPI 1.2.0/1.2.1/1.2.2)
**Phases:** 7 (Phases 4–10) | **Plans:** 14 | **Tasks:** 49

### What Was Built
Ported all GPU kernels (`diag_mult`, `hadamard_transform`, `butterfly_multiply` fwd+bwd, fp32 + complex64) from a legacy CUDA C++ extension to a Triton JIT backend made default on Ampere+. Backend dispatch infra (`_ops.py` resolver, `set_backend`, env var), a pure-PyTorch oracle (`_torch_ref/`) as the universal correctness gate, a static `do_bench`-baked routing table that falls genuine Triton-slower shapes back to CUDA, deterministic mode, 3-axis backend-agreement test suite, integration hardening (torch.compile/DDP/FSDP/checkpointing), CUDA-path deprecation (2-release cadence) + `_flashmm` removal. Then made it publishable: gated the CUDA build behind `FORCE_CUDA=1`, shipped a pure-Python `py3-none-any` wheel to PyPI.

### What Worked
- **Oracle-anchored porting:** every Triton kernel gradchecked against `_torch_ref` — caught real numerical issues and gave each phase an objective gate.
- **Parallel CUDA/Triton paths + per-op resolver:** incremental, rollback-able; the migration never broke `master`.
- **The milestone audit's integration check earned its keep:** it caught two real defects (DEPR-02 warning leak; 23 cuda-axis test failures) that single-phase verifications missed because the dev host's CUDA mismatch kept the `cuda` test axis dormant. Lighting up that axis (matched CUDA-13.0 rebuild) was the unlock.
- **Pre-dispatch plan commits** before worktree agents avoided the v1.0-era race condition.

### What Was Inefficient
- **Dev-host CUDA mismatch (PyTorch 13.0 vs `.so` 12.6)** left the entire `cuda` test axis skipped for most of the milestone, so "all green" claims were over-optimistic until the rebuild. The honest-probe skip-gate hid the gap rather than surfacing it loudly.
- **Bookkeeping lag:** traceability table stayed "Pending" and several quick-task SUMMARYs omitted a `status:` field — flagged repeatedly by `audit-open`.
- **Tolerance assumptions:** flat `atol` thresholds didn't match the fp32 accumulation noise floor; the correct model is `sqrt(n)`-scaled (FWHT/butterfly sum n terms).

### Patterns Established
- Triton kernel layer convention: `@triton_op` + `wrap_triton` + `register_autograd` + `register_fake`, fp32-scratch atomic-add for backward, IS_COMPLEX-gated branches with a fp32 wrapper assert lit up in a follow-on plan.
- `sqrt(n)`-scaled fp32 cross-backend test tolerances (not flat atol).
- DeprecationWarning as an explicit idempotent emitter (`warn_cuda_deprecation()`), decoupled from import timing, so probes/routing imports stay silent.

### Key Lessons
- A skipped test axis is a silent coverage hole — surface skip *reasons* loudly at the milestone gate, don't let an environment limitation read as "passed."
- Verify perf/correctness gates on *matched* hardware before claiming them; "infrastructure verified, value pending hardware" is the honest interim state.
- For a research library, a pure-Python Triton wheel removes the single biggest install-friction point (toolchain + per-CUDA wheel matrix).

## Cross-Milestone Trends

- **Cadence:** v1.0 (2 phases) → v1.1 (1 phase) → v1.2 (7 phases, 14 plans, 49 tasks) — first milestone large enough to need a formal audit + integration check.
- **Recurring inefficiency:** worktree/plan-commit races (v1.0) → fixed via pre-dispatch plan commits (v1.2); bookkeeping-status drift appears in every milestone — candidate for automation.
- **Verification maturity:** v1.0 shipped on phase verifications alone; v1.2 added a milestone audit + cross-phase integration check that caught defects no single phase saw.
