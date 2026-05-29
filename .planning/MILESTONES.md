# Milestones

## v1.2 Triton Migration (Shipped: 2026-05-29)

**Phases completed:** 7 phases, 14 plans, 49 tasks

**Key accomplishments:**

- Lays down the dispatch infrastructure foundation (DISP-01..05) every subsequent kernel port phase inherits: peer package layout (`_torch_ref/`, `_cuda_legacy/`), single dispatch point (`_ops.py`) with honest backend resolution, top-level `set_backend` re-export, PyTorch floor bump to >=2.6, and two phase-companion docs that Phase 7 (complex64 routing) and Phase 10 (CUDA deprecation) will implement verbatim.
- Demonstrates the canonical @torch.library.triton_op + wrap_triton + register_autograd + register_fake pipeline via a no-op identity op (`_demo_identity_op`) in `torch_structured/_ops.py`, ships the five-test acceptance suite that gates the 260419-p27 dynamo bug fix (test_demo_identity_compile_fake_tensor_trace), introduces the pytest `backend` fixture pattern for Phase 5+, and lands the first CI workflow with `actions/cache@v4` for `~/.triton/cache` (Pitfall 6 avoided).
- Triton-backed cycle_mult primitive (subdiag, v, shift_subdiag, shift_v) with Wirtinger-correct complex64 backward, replacing the legacy pybind11 _diag_mult_cuda extension and proving the Phase 4 dispatch + register_autograd plumbing end-to-end.
- Single-pass shared-memory Triton Walsh-Hadamard transform replacing the legacy `_hadamard_cuda` C++ extension, with self-inverse register_autograd backward, ``log_n in {2..12}`` correctness via tl.debug_barrier'd shared-memory shuffle, and D-33d back-compat shim preserving the existing import surface.
- New tests — 6 new PASS rows (3 tests × 2 backend params):
- Multi-launch 3-stage out_ptr-as-scratch Triton butterfly_multiply forward kernel (fp32 only) with two-input register_autograd via torch.autograd.grad on the _torch_ref oracle, IS_COMPLEX pre-wiring gated by tl.static_assert + wrapper fp32-assert for Plan 07-02 to light up, small-N fallback with alias-safe clone, and 7-test parametrized cross-backend suite covering ROADMAP SC#1 dense smoke + comprehensive 720-case Cartesian.
- Complex64 path lit up by removing the two Plan-07-01 gates (kernel-entry `tl.static_assert(not IS_COMPLEX, ...)` and wrapper `assert input.dtype == torch.float32`) and implementing the IS_COMPLEX=True branch with the verbatim 4-FMA template adapted for butterfly's four pairwise complex multiplies per stage; kernel signature UNCHANGED (D-41a load-bearing); four new tests including the load-bearing PITFALLS §1 U U^H=I unitary detector; 8-row perf baseline JSON produced for Phase 9 TEST-04 parity gate.
- Replace Phase 7's oracle-delegating `_backward` with a Triton-native backward kernel (fp32 + atomic-add into fp32 scratch) implementing SC#1 three-layer gradcheck + SC#3 fp32 scratch + SC#4 no-csrc-symbol invocation; Plan 08-02 lights up complex64.
- Light up the complex64 backward path of butterfly_multiply by removing the two Plan-08-01 gates (kernel-entry `tl.static_assert(not IS_COMPLEX, ...)` and wrapper fp32-only assert), implementing the IS_COMPLEX=True branch with conjugate-4-FMA for BOTH d_twiddle and d_input per D-50c + RESEARCH correction #3, and extending 07-BASELINE.json with backward p50/p95 entries for Phase 9's TEST-04 perf gate.
- §0 LANDMINE fixed via D-05 delegator with device-aware CPU fallback in butterfly/multiply.py; 3-axis backend fixture + per-op cuda skip-gate + 22-test integration suite + public-API signature lock + v1.0/v1.1 checkpoint round-trip + honest CUDA-legacy probe
- torch.compile(fullgraph=True) traces through Butterfly/ButterflyBmm/make_linear under BACKEND=triton; FakeTensorMode end-to-end 260419-p27 gate passes via the §0-fixed nn.Module surface; DDP + gradient-checkpointing smoke green; FSDP test shipped as @pytest.mark.multigpu; set_deterministic API + wrapper-level oracle fallback delivers bit-identical d_twiddle; CI workflow extended with two opt-in GPU jobs
- 07-BASELINE.json extended with reference_cuda_p50 + do_bench_p50_ms columns; static routing table baked from 16-row perf grid; runtime selector + resolver hook (D-66) routes below-60% cells to CUDA transparently; Phase 8 SC#4 reconciled with closure-aware approach; README "Triton backend" section + CHANGELOG.md v1.2.0 ship
- Verbatim DeprecationWarning on `TORCH_STRUCTURED_BACKEND=cuda` import path (Phase 4 D-15 incantation) + full deletion of the `_flashmm` MathDx kernel + Phase 9-compatible probe-silencing wrap + README/CHANGELOG documentation closing the v1.2 milestone.

**Shipped to PyPI:** 1.2.0 → 1.2.1 (drop maintainer email) → 1.2.2 (fix DEPR-02 warning leak). Tags v1.2.0/v1.2.1/v1.2.2. Pure-Python `py3-none-any` wheel.

**Milestone audit:** passed (`milestones/v1.2-MILESTONE-AUDIT.md`) — the integration check found 2 defects (DEPR-02 warning leak, cuda-axis test gaps); both closed by quick task 260529-bdr before close.

**Known deferred items at close:** 2 (see STATE.md "Deferred Items") — FSDP 2-GPU smoke (env-limited, single-GPU host) + quick-task SUMMARY `status:` bookkeeping.

---

## v1.0 Build System Modernization (Shipped: 2026-04-02)

**Phases completed:** 2 phases, 4 plans, 6 tasks

**Key accomplishments:**

- PEP 621 pyproject.toml with torch build isolation, thin setup.py shim with CUDA 7.0/8.0/9.0+PTX arch targeting, and MANIFEST.in for sdist
- Glob-based .so discovery replacing PathFinder, TORCH_LIBRARY macro in version.cpp, CUDA mismatch downgraded to warning

---
