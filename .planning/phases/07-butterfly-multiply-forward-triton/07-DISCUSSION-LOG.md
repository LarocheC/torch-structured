# Phase 7: butterfly_multiply Forward (Triton) — Discussion Log

**Discussed:** 2026-05-27
**Mode:** discuss (4-area selection, 8 questions across 4 areas + 1 wrap-up)

This log captures the user's selections in chronological order. Not consumed by downstream agents — for audit and retrospective reference only. The locked decisions are in `07-CONTEXT.md`.

## Domain Boundary

Phase 7 ports `butterfly_multiply` forward to Triton with fp32 + complex64; backward temporarily routes through `_torch_ref.butterfly_multiply_torch` via `register_autograd` (Phase 8 lands the heavy backward kernel). 2 plans per ROADMAP. 3-stage tile lands; 5-stage tile deferred to Phase 9.

## Prior-decision survey (carried forward, NOT re-asked)

- Phase 4 D-01..D-03 — complex64 layout via `view_as_real` + `IS_COMPLEX: tl.constexpr` + twiddle layout invariant
- Phase 5 D-21, D-22 — `_cuda_legacy/butterfly.py` try-import + asymmetric fallback `log.warning`
- Phase 5 D-25, D-26 — D-05 attribute access + `_torch_ref` oracle backward (Phase 7 extends to two-input variant)
- Phase 5/6 D-32 — `register_autograd` + `register_fake` + `triton_op` five-component skeleton
- Phase 6 D-39 — conftest `backend` fixture skip-gate widened to `_has_any_triton_kernel()` (already covers butterfly)

## Areas Selected for Discussion

User selected all 4 areas:
1. Kernel structure & 3-stage tile design
2. Plan split (2 plans)
3. output_size != n + small-N edge cases
4. Test surface + perf baseline recording

## Discussion

### Area 1: Kernel structure & 3-stage tile design

**Q1.1 — '3-stage tile' interpretation**
- Options presented: (a) Multi-launch tiled — `ceil(log_n/3)` launches per nblock, register-resident; (b) Single-pass shared-memory (extend Phase 6 pattern to butterfly twiddles); (c) Other
- **User selected:** Multi-launch tiled
- Locked as D-40 in CONTEXT.md.

**Q1.2 — nblocks loop + increasing_stride control**
- Options presented: (a) Python-side nblocks loop + constexpr stride direction; (b) Kernel-side nblocks loop; (c) Mega-launch with @triton.heuristics
- **User selected:** Python-side nblocks + constexpr stride direction
- Locked as D-40a in CONTEXT.md.

**Q1.3 — Tile-internal butterfly body**
- Options presented: (a) Load tile → 3 register-resident stages → store; (b) Load tile → shared memory → 3 stages with `tl.debug_barrier`; (c) Hybrid
- **User selected:** Register-resident
- Locked as D-40b in CONTEXT.md.

**Q1.4 — Grid mapping**
- Options presented: (a) 2-D grid `(n_row_tiles, batch*nstacks)`; (b) 1-D grid with kernel-side decomp; (c) 3-D grid
- **User selected:** 2-D grid
- Locked as D-40c in CONTEXT.md.

### Area 2: Plan split

**Q2.1 — How to split 2 plans**
- Options presented: (a) By dtype — 07-01 fp32 + tests, 07-02 complex64 + unitary; (b) By layer — kernel/autograd then consumer/tests; (c) By stage-group complexity
- **User selected:** By dtype
- Locked as D-41 in CONTEXT.md.

**Q2.2 — Pre-wire IS_COMPLEX in Plan 1**
- Options presented: (a) Pre-wire `IS_COMPLEX: tl.constexpr` flag with `tl.static_assert` gate in Plan 1; (b) Fp32-only kernel, refactor signature in Plan 2
- **User selected:** Pre-wire
- Locked as D-41a in CONTEXT.md.

### Area 3: output_size + small-N edge cases

**Q3.1 — output_size != n handling**
- Options presented: (a) Wrapper-side pad/trim (mirrors `_torch_ref/butterfly.py:18,37`); (b) Kernel-side masked load/store
- **User selected:** Wrapper-side pad/trim
- Locked as D-42 in CONTEXT.md.

**Q3.2 — Small-N (n=1, n=2) handling**
- Options presented: (a) Wrapper fallback to `_torch_ref` for log_n ≤ 1; (b) Kernel handles all log_n ≥ 0; (c) Defer to Phase 9
- **User selected:** Wrapper fallback to `_torch_ref` for log_n ≤ 1
- Locked as D-42a in CONTEXT.md.

### Area 4: Test surface + perf baseline

**Q4.1 — Test file location**
- Options presented: (a) New `tests/test_butterfly_triton.py` (top-level, mirrors `test_diag_mult.py`); (b) Extend existing `tests/test_butterfly.py`; (c) Split kernel-level + module-level
- **User selected:** New `tests/test_butterfly_triton.py`
- Locked as D-43 in CONTEXT.md.

**Q4.2 — Parametrization scope**
- Options presented: (a) Tiered: dense smoke + sparse comprehensive (`@pytest.mark.slow`); (b) Sample-based subset; (c) Full Cartesian on every CI
- **User selected:** Tiered
- Locked as D-43a in CONTEXT.md.

**Q4.3 — Perf baseline storage**
- Options presented: (a) JSON dump `07-BASELINE.json`; (b) Markdown table in SUMMARY.md; (c) Hybrid `07-BASELINE.md` with table + embedded JSON
- **User selected:** JSON dump
- Locked as D-43b in CONTEXT.md.

### Wrap-up

**Q5 — Anything else before writing CONTEXT.md?**
- Options presented: (a) Proceed; (b) Discuss backward formula; (c) Discuss `cur_increasing_stride` block toggling; (d) Discuss nn.Module integration test
- **User selected:** Proceed to CONTEXT.md
- D-44..D-48 documented as inherited from prior phases (no new gray areas).

## Scope Creep / Redirected Items

None — discussion stayed within Phase 7's ROADMAP scope. No suggestions for new capabilities surfaced.

## Claude's Discretion Items (not asked, locked in CONTEXT.md)

- Stage-group boundary handling at non-divisible log_n (recommend degenerate 1-stage final launch for symmetry)
- Exact `num_warps` constants per tile_n band (recommend 4/8/16 schedule, planner verifies)
- `@pytest.fixture(params=...)` vs `@pytest.mark.parametrize` (planner picks for consistency with `test_diag_mult.py`)
- Phase 9 perf-parity threshold (recorded as input, not enforced in Phase 7)
- Whether small-N fallback routes through `register_autograd` (recommend yes — uniform autograd graph)

## Deferred Ideas

- 5-stage tile (Phase 9 perf gate decides)
- Backward Triton kernel (Phase 8 / TRI-04)
- bf16/fp16 forward (TRI-FUT-01)
- log_n > 11 (ROADMAP SC#1 caps at 11)
- CUDA backend in conftest fixture (Phase 9 / TEST-03)
- `@triton.autotune` perf tuning (Phase 9)

---

*Phase: 7-butterfly_multiply Forward (Triton)*
*Discussion completed: 2026-05-27*
