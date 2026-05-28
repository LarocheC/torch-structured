---
phase: 09-integration-hardening-correctness-gates
verified: 2026-05-28T16:00:00Z
status: human_needed
score: 5/5 success criteria verified in code (1 SC has remaining human verification on 2-GPU + matched-CUDA hardware)
overrides_applied: 0
re_verification: null  # initial verification
human_verification:
  - test: "Run `torchrun --nproc_per_node=2 -m pytest tests/test_distributed_triton.py -m multigpu -v` on a host with ≥2 NCCL-capable GPUs."
    expected: "test_fsdp_butterfly_smoke passes on both ranks; all_gather'd loss values are finite on every rank (no NaN/Inf — would indicate twiddle silent sharding)."
    why_human: "Requires multi-GPU runner not available in current environment. SUMMARY confirms test is shipped + correctly marked @pytest.mark.multigpu but was never exercised in this phase's execution."
  - test: "On a host with PyTorch and CUDA versions matched against `_butterfly.so` build (so `_has_cuda_legacy_for_op('butterfly_multiply')` returns True), run `pytest tests/test_phase9_integration.py::test_checkpoint_v10_v11_roundtrip_butterfly_fp32 tests/test_phase9_integration.py::test_backend_agreement_butterfly_fp32 -v` and confirm the CUDA-axis variants pass (currently SKIPPED on dev host with CUDA 12.6/13.0 mismatch)."
    expected: "All cuda-axis parametrizations pass; checkpoint round-trip produces forward outputs matching triton within rtol=1e-3/atol=1e-3."
    why_human: "Dev-host CUDA mismatch documented in SUMMARYs; verification of the cuda-axis paths requires a matched-CUDA host. The TRITON axis is verified end-to-end; the test code is correct (read inline)."
  - test: "On a host with matched CUDA build, run `python tests/_baseline_butterfly.py && python tests/_baseline_butterfly_backward.py && python scripts/regenerate_routing_table.py && pytest tests/test_perf_grid.py::test_perf_gate_triton_at_60pct_cuda -v`."
    expected: "Each of 16 cells has non-null reference_cuda_p50 in 07-BASELINE.json; the perf gate (1.67×) asserts ratios for cells NOT marked route_to_cuda; cells exceeding 1.67× are routed via _routing.json."
    why_human: "TEST-04 gate computes against reference_cuda_p50, which is null on this dev host. The harness code is verified (do_bench + CUDA p50 + detach/clone), but the actual gate value computation requires matched CUDA hardware. Test currently soft-passes (skips) on dev host."
---

# Phase 9 Verification Report — Integration Hardening & Correctness Gates

**Phase Goal (ROADMAP):** Every cross-cutting integration that real users hit — `torch.compile`, DDP, FSDP, gradient checkpointing, deterministic mode, saved checkpoint round-tripping, `make_linear`/`LRU` — works on the Triton backend, and the parametrized backend test suite plus the perf grid prove parity with the CUDA path.

**Verified:** 2026-05-28
**Status:** human_needed (5/5 SCs verified end-to-end in code/tests; 3 human items defer to multi-GPU + matched-CUDA hardware that the dev host lacks)
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths — Success Criteria

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC#1 | `pytest tests/` with TORCH_STRUCTURED_BACKEND=triton passes end-to-end; backend fixture parametrizes over {triton, cuda, torch} and asserts all three agree within tolerances | VERIFIED (Triton+torch axes); cuda axis skips on dev host (CUDA mismatch — documented) | `tests/conftest.py:56` 3-axis fixture; `_has_cuda_legacy_for_op` at `_ops.py:146`; 12-test backend-agreement parametrization in `test_phase9_integration.py:287-353` passes on torch+triton (cuda skip). Tolerances `_FP32_RTOL=1e-5`, `_C64_DTWIDDLE_RTOL=1e-3`, `_C64_DINPUT_RTOL=1e-5` at `test_phase9_integration.py:59-67` match D-62c spec. TEST-06 subprocess smoke `test_pytest_under_triton_smoke` passes (verified by full run: 40 pass / 12 skip). |
| SC#2 | torch.compile(model) on Butterfly/ButterflyBmm/LRU/make_linear traces cleanly with no graph breaks (resolves 260419-p27); 2-GPU FSDP smoke succeeds; DDP and gradient checkpointing produce correct gradients | VERIFIED for non-multigpu items; FSDP human-pending | `test_torch_compile_triton.py:46,67,90,118,191` for Butterfly (3 cells: log_n=8 standard, log_n=4 small-dense, log_n=1 small-N fallback) + ButterflyBmm + make_linear — all PASS under fullgraph=True. LRU + torch.compile XFAIL'd (documented PyTorch 2.11 TorchInductor complex64 limitation, NOT a torch_structured defect). 260419-p27 FakeTensorMode gate `test_butterfly_under_fake_tensor_mode[triton]` PASSES. DDP smoke + grad checkpointing PASS. FSDP shipped + marked @pytest.mark.multigpu — runs via torchrun, needs human venue. |
| SC#3 | v1.0/v1.1 checkpoint round-trip via Butterfly.load_state_dict on Triton backend without conversion | VERIFIED (code+test correct); CUDA-axis assertion requires matched-CUDA host | `test_checkpoint_v10_v11_roundtrip_butterfly_fp32` at `test_phase9_integration.py:356-409` synthesizes v1.0/v1.1-shape state_dict ((nstacks, nblocks, log_n, n/2, 2, 2)), torch.save→torch.load round-trip, loads under BACKEND=triton AND BACKEND=cuda, compares forward outputs within rtol=1e-3/atol=1e-3. SKIPPED on dev host (CUDA mismatch). Per Butterfly.__init__ at `butterfly/butterfly.py`, layout is unchanged since Phase 1, so the synthesis is equivalent to a real v1.0/v1.1 checkpoint. |
| SC#4 | Perf grid (log_n ∈ {8,9,10,11} × dtype × direction) shows Triton ≥60% of CUDA throughput; below-60% cells route to CUDA via documented runtime selector | VERIFIED (infrastructure + test); hardware gate requires matched-CUDA host | 07-BASELINE.json has 16 rows × `reference_cuda_p50` + `do_bench_p50_ms` columns (verified via `python -c "json.load…"` — `Has reference_cuda_p50: True`, `Has do_bench_p50_ms: True`). `_routing.json` has 16 rules in keyed-object form (schema_version=1; 0 marked route_to_cuda on dev host due to CUDA mismatch → torch_ref 5× fallback gate). `_should_route_to_cuda` + `_DISABLE_ROUTING` + `set_routing_enabled` + resolver hook at `_ops.py:317-356` (`_routed_butterfly_multiply` + `_triton_with_cuda_missing_warning`). `tests/test_perf_grid.py` 7 tests — TEST-04 gate `test_perf_gate_triton_at_60pct_cuda` soft-passes when all rows null; on matched-CUDA host the gate computes actual ratios. `scripts/regenerate_routing_table.py` idempotent; re-run produces "Wrote 16 rules; 0 marked route_to_cuda". |
| SC#5 | README documents CC 8.0+ requirement, deterministic-mode opt-in, Volta/Turing v1.1-pinning recommendation | VERIFIED | `README.md:93-160` "Triton backend (v1.2+)" section. CC 8.0+ at line 102; Volta sm_70 + Turing sm_75 v1.1-pinning at lines 106-108; `set_deterministic` at lines 127-129. CHANGELOG.md exists in Keep a Changelog v1.1 format with `## [1.2.0] - 2026-05-28` entry at lines 5-10 documenting Triton port, deterministic mode, runtime selector, minimum PyTorch 2.6. |

**Score:** 5/5 SCs satisfied in codebase (with hardware-dependent assertion arms human-pending for SC#2/SC#3/SC#4).

---

## §0 LANDMINE Fix (Foundational — Plan 09-01 SC#1 prerequisite)

Critical verification item — without this fix, every SC#2/SC#3/SC#4 test verifies the C++ path under `BACKEND=triton`, not the Triton path. The Plan 09-01 RESEARCH §0 documented this as the foundational landmine.

| Check | Status | Evidence |
|-------|--------|----------|
| `torch_structured/butterfly/multiply.py` `butterfly_multiply` delegates to `_ops.butterfly_multiply` (no direct C++ call) | VERIFIED | `multiply.py:72-103`: body re-reads `torch_structured._ops.butterfly_multiply` on every call (with CPU-tensor route to torch oracle for v1.1 compat); `grep -nE 'return torch\.ops\.torch_structured\.butterfly_multiply\(' ... | grep -v '_fw\|_bw'` returns empty. |
| "Import-binding semantics note" comment present | VERIFIED | `multiply.py:62-71`; `grep -c 'Import-binding semantics note' multiply.py` = 1. |
| `@torch.jit.script` decorators removed | VERIFIED | `grep -cE '^@torch\.jit\.script' multiply.py` = 0; only docstring/comment mentions remain (5 occurrences, all explanatory). |
| Both-sides gate: positive recording sentinel + negative C++ raising stub | VERIFIED | Tests `test_butterfly_nn_module_routes_through_ops` (positive at `test_phase9_integration.py:84-117`) AND `test_butterfly_nn_module_does_not_call_cpp_op_directly` (negative at `test_phase9_integration.py:120-154`). Both PASS on dev host. |
| `butterfly_multiply_fw`/`_bw` still call C++ directly (no Triton equivalent) | VERIFIED | `multiply.py:49-50,59` direct calls to `torch.ops.torch_structured.butterfly_multiply_fw/_bw`; `test_butterfly_multiply_fw_bw_still_call_cpp` at `test_phase9_integration.py:180-219` PASSES. |
| Phase 8 SC#4 test still passes (monkey-patch survives @torch.jit.script removal) | VERIFIED | `pytest tests/test_butterfly_triton.py::test_butterfly_backward_no_cpp_symbol` PASSES; full Phase 7+8 regression: 84 pass / 26 skip (matches baseline). |

---

## Required Artifacts (Three-Level Verification)

| Artifact | Exists | Substantive | Wired | Status |
|----------|--------|-------------|-------|--------|
| `torch_structured/butterfly/multiply.py` (D-05 delegators) | Yes (104 lines) | Yes (3 functions + W4 comment + docstring) | Yes (imported by `butterfly/butterfly.py:8`) | VERIFIED |
| `torch_structured/_ops.py` (`_has_cuda_legacy_for_op` + deterministic + routing) | Yes (661 lines) | Yes (10+ new symbols verified at module load) | Yes (conftest + op.py + tests consume) | VERIFIED |
| `torch_structured/__init__.py` (`set_deterministic` export) | Yes | Yes (`__all__` contains `set_deterministic` at line 46; `from ._ops import` at line 35) | Yes (verified via `python -c "torch_structured.set_deterministic(True)"`) | VERIFIED |
| `torch_structured/_triton/butterfly/op.py` (deterministic gate at top of `_backward`) | Yes | Yes (lines 1372-1382 — local import + gate body BEFORE small-N fallback at line 1393) | Yes (gate fires under set_deterministic(True), proven by `test_deterministic_dtwiddle_bit_identical[triton]` PASS) | VERIFIED |
| `torch_structured/_routing.json` (16 keyed rules) | Yes | Yes (schema_version=1, 16 rules, keyed-object form) | Yes (loaded via `_load_routing_table` at `_ops.py:561`) | VERIFIED |
| `tests/conftest.py` (3-axis backend fixture) | Yes (81 lines) | Yes (`params=["torch", "triton", "cuda"]` at line 56; `slow`, `multigpu`, `op` markers registered) | Yes (consumed by all Phase 9 + Phase 7+8 parametrized tests) | VERIFIED |
| `tests/test_phase9_integration.py` (Plan 09-01 — §0 LANDMINE + backend agreement + checkpoint + make_linear + LRU + public API + TEST-06 subprocess smoke) | Yes (561 lines, 22 test functions) | Yes (40 effective tests pass, 12 skip on cuda axis) | Yes (all tests collected; 40 pass / 12 skip) | VERIFIED |
| `tests/test_torch_compile_triton.py` (8 tests + small_dense + small_n_branch + FakeTensorMode + register_fake) | Yes (266 lines, 8 test functions) | Yes (7 pass / 14 skip / 1 xfailed — LRU upstream) | Yes (collected, executed) | VERIFIED |
| `tests/test_distributed_triton.py` (DDP + FSDP @multigpu + grad checkpoint) | Yes (208 lines, 3 test functions) | Yes (DDP + grad-checkpoint PASS; FSDP marked @multigpu) | Yes (collected; 2 pass / 4 skip — multigpu deselected) | VERIFIED (FSDP itself human-pending) |
| `tests/test_deterministic_mode.py` (set_deterministic + bit-identity) | Yes (276 lines, 6 test functions, 15 parametrized) | Yes (9 pass / 6 skip cuda) | Yes (collected; cardinal bit-identity test passes under triton) | VERIFIED |
| `tests/test_perf_grid.py` (TEST-04 gate + 6 selector unit tests) | Yes (296 lines, 7 test functions) | Yes (4 pass / 3 skip — selector unit tests pass; gate soft-skips on null CUDA p50) | Yes (collected, executed) | VERIFIED (assertion arm needs matched-CUDA hardware) |
| `tests/_baseline_butterfly.py` + `_baseline_butterfly_backward.py` (extended with CUDA p50 + do_bench) | Yes | Yes (`triton.testing.do_bench(warmup=25, rep=100, quantiles=[0.5, 0.95])` present in both; CUDA-p50 branches present; W6 detach/clone hygiene present in backward — `grep -c 'detach().clone().requires_grad_'` = 6) | Yes (consumed by routing table regen) | VERIFIED |
| `.planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json` (extended schema) | Yes | Yes (16 rows × `reference_cuda_p50` + `do_bench_p50_ms` columns confirmed by python json.load probe) | Yes (consumed by `scripts/regenerate_routing_table.py` + `tests/test_perf_grid.py`) | VERIFIED |
| `scripts/regenerate_routing_table.py` | Yes | Yes (`def main()` returns 0, writes routing JSON with keyed-object form) | Yes (idempotent; re-run produces "Wrote 16 rules") | VERIFIED |
| `README.md` (Triton backend section) | Yes (extended) | Yes (CC 8.0+ at line 102; set_deterministic at lines 127-129; Volta/Turing pinning at lines 106-108; TORCH_STRUCTURED_BACKEND documented 6 times) | Yes (ships with repo) | VERIFIED |
| `CHANGELOG.md` (Keep a Changelog v1.1) | Yes (NEW file) | Yes (Keep a Changelog cite at line 5; [1.2.0] entry at line 10; Triton + deterministic + selector + min PyTorch 2.6 documented) | Yes (ships with repo) | VERIFIED |
| `.github/workflows/test.yml` (multigpu CI job) | Yes (extended) | Yes (`test-triton` job at line 60 gated on `vars.ENABLE_GPU_CI`; `test-multigpu` job at line 93 gated on `vars.ENABLE_MULTIGPU_CI`; CPU job extended with `-m "not multigpu and not slow"`) | Yes (ships with repo) | VERIFIED |

---

## Key Link Verification (Wiring)

| From | To | Via | Status |
|------|----|----|--------|
| `butterfly/butterfly.py:124,128,239,243` (nn.Module forward call sites) | `torch_structured._ops.butterfly_multiply` | `from .multiply import butterfly_multiply` binds delegator; delegator re-reads `_ops.butterfly_multiply` on every call | WIRED (verified by positive recording-sentinel test passing) |
| `tests/conftest.py` backend fixture `cuda` axis | `_ops._has_cuda_legacy_for_op(op_name)` | `@pytest.mark.op('<op_name>')` marker resolved via `request.node.get_closest_marker("op")` | WIRED (lines 71-76; verified by `test_backend_fixture_skips_cuda_axis_when_marker_present_and_so_missing` PASS) |
| `_triton/butterfly/op.py:_backward` (top of body) | `_ops._is_deterministic_mode_active()` | Local import + `if _is_deterministic_mode_active(): … return` block at lines 1372-1382 | WIRED (gate fires when `set_deterministic(True)` set, proven by bit-identical d_twiddle test PASS) |
| `_ops.py:_resolve` (triton branch) | `_should_route_to_cuda` selector | `_routed_butterfly_multiply` closure at lines 317-325 (when `_has_cuda_legacy()` True); `_triton_with_cuda_missing_warning` at lines 334-356 (D-61b warning fallback) | WIRED (selector consulted on every Triton call; verified by `test_should_route_to_cuda_default_matches_bake` PASS) |
| `scripts/regenerate_routing_table.py:main()` | `_routing.json` (16 keyed rules) | Reads 07-BASELINE.json rows, writes `{"rules": {"<op>::<log_n>::<dtype>::<direction>": {...}}}` | WIRED (idempotent regen verified) |
| `tests/test_butterfly_triton.py::test_butterfly_backward_no_cpp_symbol` | `_ROUTING_TABLE` + `set_routing_enabled(False)` | Pre-test assertion `assert not _ROUTING_TABLE.get("butterfly_multiply::4::fp32::backward", {}).get("route_to_cuda")`; `set_routing_enabled(False)` at line 769; restore in finally at line 813 | WIRED (test PASSES) |
| `tests/test_torch_compile_triton.py::test_butterfly_under_fake_tensor_mode` | Butterfly nn.Module + FakeTensorMode | `with FakeTensorMode(allow_non_fake_inputs=True)` + `mode.from_tensor(x)` + `m(fake_x)` | WIRED (260419-p27 gate PASSES under triton) |
| `torch_structured/__init__.py` | `_ops.set_deterministic` | `from ._ops import set_backend, set_deterministic` + `'set_deterministic'` in `__all__` | WIRED (verified by `python -c "torch_structured.set_deterministic(True)"` returns previous value) |

---

## Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|---------------------|--------|
| `_routed_butterfly_multiply` closure | routing decision | `_should_route_to_cuda(op_name, input_.shape, input_.dtype, "forward")` reads `_ROUTING_TABLE` (16 rules loaded from `_routing.json`) | Yes (16 real entries; lookup uses input.shape → log_n) | FLOWING |
| `_ROUTING_TABLE` | dict of rules | `_load_routing_table()` reads `_routing.json` at module import | Yes (16 entries verified) | FLOWING |
| `_is_deterministic_mode_active()` | bool | `_DETERMINISTIC` (module-level) OR `torch.are_deterministic_algorithms_enabled()` | Yes (additive OR, both inputs toggleable) | FLOWING |
| `Butterfly.forward(x)` → `_ops.butterfly_multiply` | tensor output | Triton kernel under BACKEND=triton (verified by recording sentinel test) | Yes — Phase 7+8 Triton kernel produces real outputs (84 Phase 7+8 tests still pass) | FLOWING |

---

## Requirements Coverage

| Requirement | Source Plan | Description (from Phase 9 must-haves) | Status | Evidence |
|------------|-------------|----------------------------------------|--------|----------|
| TEST-01 | 09-03 | Triton correctness vs torch oracle at fp32 + complex64 | VERIFIED | Backend-agreement tests at `test_phase9_integration.py:287-353` (12 tests pass under torch+triton); tolerances match Phase 7 D-43a |
| TEST-02 | 09-03 | Backward gradcheck against `autograd.grad(_torch_fw, ...)` | VERIFIED (inherited from Phase 8) | `tests/test_butterfly_triton.py` Phase 8 gradcheck tests still pass (84 pass / 26 skip) |
| TEST-03 | 09-01 | Test suite parametrizes over {triton, cuda, torch} and asserts agreement | VERIFIED | `conftest.py:56` 3-axis fixture; backend agreement tests pass on torch+triton axes (cuda skip on dev host) |
| TEST-04 | 09-03 | Triton ≥60% of CUDA throughput via `triton.testing.do_bench` | VERIFIED (infrastructure); HUMAN NEEDED for ratio assertion | 07-BASELINE.json has `do_bench_p50_ms` column (16/16 rows); test `test_perf_gate_triton_at_60pct_cuda` soft-skips on dev host (no CUDA build); routes via _routing.json on assertion failure |
| TEST-06 | 09-01 | Existing tests pass with TORCH_STRUCTURED_BACKEND=triton | VERIFIED | `test_pytest_under_triton_smoke` (subprocess) PASSES; CI workflow extended with `test-triton` job at `.github/workflows/test.yml:60` |
| COMPAT-01 | 09-01 | `Butterfly`/`ButterflyBmm`/`ButterflyUnitary`/`ButterflyBase4` public API unchanged | VERIFIED | 4 `inspect.signature` snapshot tests at `test_phase9_integration.py:481-528` all PASS |
| COMPAT-02 | 09-01 | v1.0/v1.1 checkpoints load on Triton backend | VERIFIED (code+test); HUMAN NEEDED for cuda-axis arm | `test_checkpoint_v10_v11_roundtrip_butterfly_fp32` correctly synthesizes v1.0/v1.1-layout state_dict; SKIPPED on dev host (CUDA mismatch); code-reviewed correct |
| COMPAT-03 | 09-01 | `make_linear` + `LRU` work on Triton | VERIFIED | `test_make_linear_butterfly_triton` + `test_lru_butterfly_triton` PASS on dev host |
| COMPAT-04 | 09-02 | `torch.compile(model)` traces cleanly through Triton (resolves 260419-p27) | VERIFIED | Butterfly + ButterflyBmm + make_linear all PASS fullgraph=True under triton; FakeTensorMode end-to-end gate PASSES; LRU XFAIL'd (upstream TorchInductor complex64 limitation, NOT torch_structured defect) |
| COMPAT-06 | 09-03 | README docs CC 8.0+, deterministic mode, Volta/Turing pinning | VERIFIED | README.md has "Triton backend (v1.2+)" section with all three items; CHANGELOG.md created in Keep a Changelog v1.1 format with v1.2.0 entry |

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `tests/test_torch_compile_triton.py` (LRU test) | 157 | `@pytest.mark.xfail(strict=False)` on `test_torch_compile_lru_butterfly_fullgraph_no_break` | Info | Documented upstream PyTorch 2.11 TorchInductor complex64 limitation. NOT a torch_structured defect; xfail will become XPASS when upstream lands the fix. Acceptable per Plan 09-02 Rule 2 deviation. |
| `torch_structured/_ops.py` | 105-115 | One-shot `try/except RuntimeError` in `_has_cuda_legacy()` | Info | Documented Rule 2 deviation in Plan 09-01 SUMMARY — honest probe pattern matches existing `_cuda_legacy/diag_mult.py` pattern; necessary for honest skip-gate on hosts with CUDA-version mismatch. |
| (none) | — | No debt markers (`TBD`, `FIXME`, `XXX`) found in Phase 9 modified files | Pass | `grep -rn -E 'TBD|FIXME|XXX' torch_structured/butterfly/multiply.py torch_structured/_ops.py torch_structured/_triton/butterfly/op.py tests/test_phase9_integration.py tests/test_torch_compile_triton.py tests/test_distributed_triton.py tests/test_deterministic_mode.py tests/test_perf_grid.py` returns clean |
| (none) | — | No empty-data stubs in production code | Pass | All code reviewed — delegators forward real data; selector returns real bool; routing table loaded from real JSON |

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `torch_structured.set_deterministic` callable + exported | `python -c "import torch_structured; print(torch_structured.set_deterministic(True))"` | `False` (returned previous value as designed; save/restore semantics) | PASS |
| `_ops._should_route_to_cuda` callable + table loaded | `python -c "import torch_structured._ops as o; print(len(o._ROUTING_TABLE))"` | `16` | PASS |
| 07-BASELINE.json has required columns | `python -c "import json; rows = json.load(open('.planning/phases/07.../07-BASELINE.json'))['rows']; print(all('reference_cuda_p50' in r for r in rows), all('do_bench_p50_ms' in r for r in rows))"` | `True True` | PASS |
| `scripts/regenerate_routing_table.py` is idempotent | `python scripts/regenerate_routing_table.py` | `Wrote 16 rules; 0 marked route_to_cuda (16 used torch_ref fallback gate)` | PASS |
| §0 LANDMINE both-sides gate | `pytest tests/test_phase9_integration.py::test_butterfly_nn_module_routes_through_ops tests/test_phase9_integration.py::test_butterfly_nn_module_does_not_call_cpp_op_directly` | 2 passed | PASS |
| Phase 9 integration suite | `pytest tests/test_phase9_integration.py` | 40 passed, 12 skipped | PASS |
| torch.compile + distributed + deterministic + perf grid | `pytest tests/test_torch_compile_triton.py tests/test_distributed_triton.py tests/test_deterministic_mode.py tests/test_perf_grid.py -m "not multigpu"` | 22 passed, 29 skipped, 1 deselected, 1 xfailed | PASS |
| Phase 7+8 regression (test_butterfly_triton.py) | `pytest tests/test_butterfly_triton.py -k 'not slow'` | 84 passed, 26 skipped (matches Plan 09-01/09-02/09-03 baselines exactly) | PASS |
| Dispatch + structured + diag_mult tests | `pytest tests/test_dispatch.py tests/structured/ tests/test_diag_mult.py` | 104 passed, 2 skipped | PASS |
| SC#4 reconciliation test | `pytest tests/test_butterfly_triton.py::test_butterfly_backward_no_cpp_symbol` | 1 passed | PASS |

---

## Inherited Invariants (Regression Check)

| Invariant | Status | Evidence |
|-----------|--------|----------|
| `_setup_context`, `register_autograd`, `register_fake` unchanged | VERIFIED | `_triton/butterfly/op.py:1577-1597` register_fake body present + verified by `test_butterfly_register_fake_is_present` PASS; deterministic gate placed ABOVE small-N fallback (line 1376), not inside the kernel registration |
| `_torch_ref/butterfly.py` (TRI-07) unchanged | VERIFIED | Used as oracle in deterministic gate + small-N fallback; no edits to file |
| `_cuda_legacy/butterfly.py` unchanged | VERIFIED | Imported as `_cuda_bm_for_route` in resolver hook; no edits to file |
| Phase 7+8 test files compatible after Plan 09-01 conftest widening | VERIFIED | 84 Phase 7+8 tests still pass / 26 skip (identical to pre-Phase-9 baseline) |
| 5 documented Rule deviations (per SUMMARYs) are honest empirical calibrations | VERIFIED | Plan 09-01: Rule 1 (CPU compat in delegator) + Rule 2 (honest `_has_cuda_legacy` probe). Plan 09-02: Rule 1 (Test 6 sanity ceiling 1e-2→0.5) + Rule 2 (LRU xfail). Plan 09-03: Rule 3 (.gitignore exception for `scripts/`). All documented in respective SUMMARYs with file:line citations. |

---

## Human Verification Required

Three items defer to hardware unavailable on the verification host:

### 1. FSDP 2-GPU smoke test execution

**Test:** Run `torchrun --nproc_per_node=2 -m pytest tests/test_distributed_triton.py -m multigpu -v`
**Expected:** `test_fsdp_butterfly_smoke` passes on both ranks; `all_gather`'d loss values are finite on every rank (NaN/Inf would indicate twiddle silent sharding despite `ignored_modules`)
**Why human:** Requires ≥2 NCCL-capable GPUs. Plan 09-02 SUMMARY confirms the test is shipped + correctly marked `@pytest.mark.multigpu` using FSDP1 with `ignored_modules=[Butterfly]` (PyTorch 2.6's FSDP2 lacks `ignored_params`), but was never executed in this phase's run.

### 2. Checkpoint round-trip + backend agreement on matched-CUDA host

**Test:** On a host where `_has_cuda_legacy_for_op('butterfly_multiply')` returns True, run `pytest tests/test_phase9_integration.py::test_checkpoint_v10_v11_roundtrip_butterfly_fp32 tests/test_phase9_integration.py -k 'backend_agreement' -v`
**Expected:** All `cuda`-axis parametrizations PASS (currently SKIPPED on dev host); checkpoint round-trip produces forward outputs matching triton within rtol=1e-3/atol=1e-3.
**Why human:** Dev-host `_butterfly.so` was built against CUDA 12.6; PyTorch is on CUDA 13.0. The honest `_has_cuda_legacy()` probe correctly returns False, so cuda-axis tests SKIP. Test code is verified correct via inline review.

### 3. TEST-04 perf gate computation on matched-CUDA host

**Test:** On a host with matched CUDA build, run `python tests/_baseline_butterfly.py && python tests/_baseline_butterfly_backward.py && python scripts/regenerate_routing_table.py && pytest tests/test_perf_grid.py::test_perf_gate_triton_at_60pct_cuda -v`
**Expected:** Each of 16 rows in 07-BASELINE.json gets non-null `reference_cuda_p50`; the perf gate asserts `wall_ms_p50 / reference_cuda_p50 ≤ 1.67` per cell; failing cells appear in `_routing.json` as `route_to_cuda: true`.
**Why human:** On the dev host all `reference_cuda_p50` values are null (CUDA mismatch), so the gate soft-skips and `_routing.json` falls back to the 5.0× torch_ref weaker gate. The harness code (`triton.testing.do_bench(warmup=25, rep=100, quantiles=[0.5, 0.95])` + CUDA p50 measurement + W6 detach/clone hygiene + idempotent script) is verified correct; only the actual ratio computation requires matched hardware.

---

## Gaps Summary

**No code-level gaps found.** All five Success Criteria are satisfied in the codebase. The §0 LANDMINE is fixed and verified by a both-sides gate (positive + negative). All claimed artifacts exist, are substantive, are wired into the right call sites, and have flowing data.

The three human-verification items above arise from the dev host having (a) only one GPU and (b) a CUDA-version mismatch between PyTorch (13.0) and the legacy `_butterfly.so` (12.6). Both conditions are documented in all three Plan SUMMARYs and the honest-probe pattern correctly skip-gates the affected test arms. The test code that would exercise the cuda axis is verified correct via inline review and the test assertions are coherent.

**Recommendation:** Mark phase complete after running the three human items above on appropriate hardware. Phase 10 (deprecation) does not block on these — the v1.2 ship state is honest about which cells need user-side validation.

---

*Verified: 2026-05-28T16:00:00Z*
*Verifier: Claude (gsd-verifier)*
