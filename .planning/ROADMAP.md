# Roadmap: torch_butterfly Build Modernization

## Milestones

- v1.0 Build System Modernization - Phases 1-2 (shipped 2026-04-02)
- v1.1 Repository Cleanup - Phase 3 (shipped 2026-04-03)
- v1.2 Triton Migration - Phases 4-10 (in progress)

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

<details>
<summary>v1.0 Build System Modernization (Phases 1-2) - SHIPPED 2026-04-02</summary>

- [x] **Phase 1: Build System Foundation** - pyproject.toml, PEP 621 metadata, CUDA arch fix, and working non-editable install (completed 2026-04-02)
- [x] **Phase 2: Extension Loading and Editable Installs** - Fix runtime .so discovery, modernize C++ registration, enable editable installs (completed 2026-04-02)

</details>

<details>
<summary>v1.1 Repository Cleanup (Phase 3) - SHIPPED 2026-04-03</summary>

- [x] **Phase 3: Strip and Verify** - Remove all legacy code, experiments, and dead assets; verify tests pass (completed 2026-04-03)

</details>

### v1.2 Triton Migration

- [ ] **Phase 4: Triton Dispatch Infrastructure & Foundational Decisions** - Lock in the `triton_op` wrapper pattern, backend selector, complex64 layout, `torch>=2.6` floor, and CI cache plumbing — no kernels yet
- [ ] **Phase 5: diag_mult Triton Port** - Smallest kernel; validates the end-to-end dispatch + autograd plumbing
- [ ] **Phase 6: hadamard Triton Port** - Self-inverse forward kernel; no atomics, no complex
- [ ] **Phase 7: butterfly_multiply Forward (Triton)** - Forward kernel for fp32 and complex64 (real/imag split), with torch-reference backward as the intermediate state
- [ ] **Phase 8: butterfly_multiply Backward (Triton)** - Full Triton autograd via fp32 scratch accumulator; the highest-risk kernel
- [ ] **Phase 9: Integration Hardening & Correctness Gates** - torch.compile, DDP, FSDP, gradient checkpointing, deterministic mode, full backend-parametrized test suite, perf grid
- [ ] **Phase 10: CUDA Deprecation & flashmm Removal** - Triton becomes the default; CUDA path emits DeprecationWarning; `_flashmm` removed (csrc/ deletion deferred to a future milestone)

## Phase Details

<details>
<summary>v1.0 Build System Modernization (Phases 1-2) - SHIPPED 2026-04-02</summary>

### Phase 1: Build System Foundation
**Goal**: Users can install torch_butterfly from source with `uv pip install .` or `pip install .` and get a working package with CUDA support
**Depends on**: Nothing (first phase)
**Requirements**: BUILD-01, BUILD-02, BUILD-03, BUILD-04, BUILD-05, BUILD-06, CUDA-01, CUDA-02, CUDA-03, CUDA-04, INST-01, INST-02, INST-04, INST-05, INST-06
**Success Criteria** (what must be TRUE):
  1. Running `uv pip install .` in a fresh virtual environment succeeds without errors
  2. Running `pip install .` in a fresh virtual environment succeeds without errors
  3. After install, `import torch_butterfly` works and CUDA extensions load on a CUDA-capable machine
  4. The build succeeds without conda -- only pip/uv and a system CUDA toolkit are needed
  5. Setting `FORCE_CUDA=1` or `FORCE_CPU=1` controls whether CUDA extensions are compiled
**Plans**: 1 plan

Plans:
- [x] 01-01-PLAN.md -- pyproject.toml, setup.py shim, MANIFEST.in, CUDA arch fix

### Phase 2: Extension Loading and Editable Installs
**Goal**: Editable installs work reliably and the C++ extension loading mechanism is robust across install modes
**Depends on**: Phase 1
**Requirements**: EXT-01, EXT-02, EXT-03, INST-03
**Success Criteria** (what must be TRUE):
  1. Running `uv pip install -e .` succeeds and `import torch_butterfly` loads the compiled extensions
  2. After modifying Python source in an editable install, changes are reflected immediately without reinstall
  3. The deprecated RegisterOperators API in version.cpp is replaced with TORCH_LIBRARY macro
**Plans**: 1 plan

Plans:
- [x] 02-01-PLAN.md -- Fix extension loading, modernize version.cpp, soften CUDA check

</details>

<details>
<summary>v1.1 Repository Cleanup (Phase 3) - SHIPPED 2026-04-03</summary>

### Phase 3: Strip and Verify
**Goal**: Repository contains only the core torch_butterfly library, build files, and tests -- all legacy and experiment code removed
**Depends on**: Phase 2
**Requirements**: LEGACY-01, LEGACY-02, LEGACY-03, LEGACY-04, EXP-01, EXP-02, EXP-03, EXP-04, CLEAN-01, CLEAN-02, CLEAN-03, CLEAN-04
**Success Criteria** (what must be TRUE):
  1. The directories `butterfly/`, `tests_old/`, `learning_transforms/`, `cnn/`, `convolution/`, `transformer/`, `gumbel-sinkhorn/`, and `data/` do not exist in the repository
  2. The files `ray_template.sh` and `.gitmodules` do not exist, and the `fairseq/` submodule is fully removed
  3. `build/` and `torch_butterfly.egg-info/` are listed in `.gitignore` and not tracked by git
  4. `pytest tests/` passes with all tests green after all removals
**Plans**: 2 plans

Plans:
- [x] 03-01-PLAN.md -- Remove legacy dirs, experiment dirs, submodule, and dead files
- [x] 03-02-PLAN.md -- Clean .gitignore, modernize README, verify tests pass

</details>

### Phase 4: Triton Dispatch Infrastructure & Foundational Decisions
**Goal**: A `TORCH_STRUCTURED_BACKEND` environment variable selects the kernel backend at import time, the `triton_op` + `register_autograd` + `wrap_triton` wrapper pattern is locked in, and `torch>=2.6` is enforced -- without shipping any Triton kernel yet
**Depends on**: Phase 3
**Requirements**: DISP-01, DISP-02, DISP-03, DISP-04, DISP-05, COMPAT-05, TRI-05, TRI-06, TRI-07, TEST-05
**Success Criteria** (what must be TRUE):
  1. User can set `TORCH_STRUCTURED_BACKEND={triton,cuda,torch,auto}` and the library logs the resolved backend exactly once at import time, with the documented `auto` precedence (triton if available -> cuda .so if loaded -> pure-PyTorch)
  2. User can call `torch_structured.set_backend("torch")` from Python and subsequent kernel calls route to the pure-PyTorch reference -- no per-call branching, dispatch lives in a single `torch_structured/_ops.py` module
  3. A demonstrator op (no real kernel yet, e.g. a no-op identity wrapped via `@torch.library.triton_op` + `register_autograd` + `register_fake`) traces cleanly under `torch.compile` and survives `gradcheck`, proving the wrapper pattern works end-to-end
  4. `pyproject.toml` declares `torch>=2.6`, `uv pip install -e .` succeeds against PyTorch 2.6+, and the recurrent dynamo fake-tensor bug from quick task 260419-p27 no longer reproduces on the demonstrator op
  5. A `complex64` layout decision (real/imag split vs packed `view_as_real`) is documented in `.planning/phases/04/` and a written deprecation plan for `csrc/` exists; CI persists `TRITON_CACHE_DIR` between runs so subsequent first-call JIT cost is amortized
**Plans**: 2 plans

Plans:
- [x] 04-01-PLAN.md -- _torch_ref/ + _cuda_legacy/ + _ops.py dispatch + set_backend + torch>=2.6 + 04-COMPLEX-LAYOUT.md + 04-DEPRECATION-PLAN.md (completed 2026-05-27)
- [x] 04-02-PLAN.md -- demonstrator op + tests/test_dispatch.py + CI cache (completed 2026-05-27)

### Phase 5: diag_mult Triton Port
**Goal**: `diag_mult` runs on Triton for fp32 and complex64 forward+backward, validating that the Phase 4 dispatch and autograd plumbing carry a real kernel end-to-end
**Depends on**: Phase 4
**Requirements**: TRI-01
**Success Criteria** (what must be TRUE):
  1. User running with `TORCH_STRUCTURED_BACKEND=triton` on a CUDA-capable machine gets `diag_mult` from the Triton kernel; correctness vs `_torch_ref` passes at fp32 (rtol=1e-5, atol=1e-6) and complex64 (rtol=1e-4)
  2. `torch.autograd.gradcheck` of the Triton `diag_mult` against `autograd.grad(_torch_ref.diag_mult, ...)` passes in fp64 for both real and complex inputs
  3. `structured/krylov.py` imports `diag_mult` from `torch_structured._ops` (single import point); the CUDA `_diag_mult.so` path remains selectable via `TORCH_STRUCTURED_BACKEND=cuda` and produces the same results
**Plans**: 1 plan

Plans:
- [x] 05-01-PLAN.md -- _torch_ref + _triton/diag_mult + _cuda_legacy/diag_mult + _ops.py per-op resolver + krylov.py refactor + tests/test_diag_mult.py + conftest widening + demonstrator deletion

### Phase 6: hadamard Triton Port
**Goal**: `hadamard` runs on Triton as a forward-only self-inverse kernel, proving the two-pass mixed-radix shared-memory pattern in Triton without atomics or complex
**Depends on**: Phase 5
**Requirements**: TRI-02
**Success Criteria** (what must be TRUE):
  1. With `TORCH_STRUCTURED_BACKEND=triton`, `hadamard` produces fp32 outputs that match `_torch_ref.hadamard_transform_torch` at rtol=1e-5, atol=1e-6 across log_n in {2..12}
  2. Composing `hadamard ∘ hadamard` is bit-equivalent (within fp32 noise) to identity on any input shape, demonstrating self-inverse correctness end-to-end
  3. `structured/hadamard.py` consumers (including `Hadamard` nn.Module and the `hadamard()` factory) route through `torch_structured._ops.hadamard_transform` and pass `pytest tests/structured/` on the Triton backend
**Plans**: 1 plan

Plans:
- [x] 06-01-PLAN.md -- _torch_ref/hadamard + _triton/hadamard_transform + _cuda_legacy/hadamard + _ops.py per-op resolver + structured/hadamard.py back-compat shim + fastfood.py refactor + conftest widening + tests/structured/test_hadamard_triton.py

### Phase 7: butterfly_multiply Forward (Triton)
**Goal**: `butterfly_multiply` forward runs on Triton for fp32 and complex64 across every `(increasing_stride, output_size, nstacks, nblocks)` combination, while backward temporarily routes through the torch-reference via `register_autograd` so the phase ships before the heavy backward kernel
**Depends on**: Phase 6
**Requirements**: TRI-03
**Success Criteria** (what must be TRUE):
  1. With `TORCH_STRUCTURED_BACKEND=triton`, `butterfly_multiply` forward output matches `butterfly_multiply_torch` at rtol=1e-5, atol=1e-6 (fp32) and rtol=1e-4 (complex64) across the full parameter grid, including n=1, n=2, and `output_size != n` edge cases
  2. The unitary butterfly test (`U U^* = I` from `test_butterfly.py`) passes with `complex=True`, proving the real/imag-split complex multiply works under conjugation
  3. `Butterfly`, `ButterflyBmm`, `ButterflyUnitary`, and `ButterflyBase4` nn.Module forward calls produce correct outputs on the Triton path; backward is still functional because `register_autograd` routes to a torch-reference backward wrapped through the same `triton_op` API
  4. The 3-stage tile lands; the 5-stage tile is explicitly deferred (documented in the phase), and the forward perf at log_n in {8,9,10,11} is recorded as the baseline for Phase 9's parity gate
**Plans**: 2 plans

Plans:
- [x] 07-01-PLAN.md -- _triton/butterfly + @triton_op fp32 kernel with multi-launch 3-stage register-resident tile + IS_COMPLEX pre-wiring (gated) + register_autograd two-input backward via _torch_ref + register_fake with load-bearing defaults + tests/test_butterfly_triton.py (smoke + slow comprehensive + fp64 gradcheck + small-N fallback)
- [x] 07-02-PLAN.md -- light up IS_COMPLEX=True 4-FMA branch + remove fp32-only wrapper gate + complex64 + Wirtinger gradcheck + unitary U U* = I test (PITFALLS §1 acceptance gate) + perf baseline JSON to 07-BASELINE.json per D-43b

### Phase 8: butterfly_multiply Backward (Triton)
**Goal**: `butterfly_multiply` backward runs entirely on Triton with a pre-allocated fp32 scratch accumulator for `d_twiddle` atomic adds, replacing the torch-reference backward from Phase 7 and freeing the library from `csrc/butterfly.cpp` at runtime
**Depends on**: Phase 7
**Requirements**: TRI-04
**Success Criteria** (what must be TRUE):
  1. The three-layer gradcheck pattern passes: (a) fp64 `gradcheck` on n=4, batch=1, log_n=2 against `autograd.grad(butterfly_multiply_torch, ...)`; (b) `allclose` for `d_input` at n=256, batch=8; (c) `allclose` for `d_twiddle` at n=512, batch=4096 within the documented atomicAdd noise envelope
  2. Backward correctness holds for complex64 via the same real/imag-split layout as forward; complex-twiddle `d_twiddle` matches the torch-reference autograd within rtol=1e-3, atol=1e-4 at batch=4096
  3. The `d_twiddle` atomic accumulation buffer is allocated in Python as `torch.zeros_like(twiddle, dtype=torch.float32)` and the kernel does a block-level `tl.sum` reduce before its single atomicAdd per block -- never atomicAdd directly into bf16/fp16
  4. With `TORCH_STRUCTURED_BACKEND=triton`, a full training step (`loss.backward()`) on a model containing `Butterfly` no longer invokes any C++ symbol from `csrc/butterfly.cpp`, verified by stack-trace inspection
**Plans**: 2 plans

Plans:
- [x] 08-01-PLAN.md -- _run_forward_stage_groups helper extract (D-49a) + _butterfly_backward_kernel (IS_COMPLEX gated) + _backward body replacement (trail recompute + reverse stage-group walk + per-program tl.sum + tl.atomic_add into fp32 scratch) + three-layer gradcheck (SC#1 a/b/c) + small-case Triton-kernel-exerciser (RESEARCH correction #4) + D-49b small-N fallback test + SC#4 dispatch-binding + monkey-patch shim (NOT sys.modules) + dense smoke + sparse comprehensive fp32 backward tiers
- [x] 08-02-PLAN.md -- light up IS_COMPLEX=True with conjugate-4-FMA for BOTH d_twiddle (D-50c) AND d_input (RESEARCH correction #3) + view_as_real machinery in _backward (D-50b trail_n doubling + scratch trailing-2 axis) + SC#2 complex64 backward allclose at batch=4096 with separate d_twiddle/d_input parity + Wirtinger complex128 gradcheck + unitary landmine detector (PITFALLS §1 analog for backward) + dense smoke + sparse comprehensive complex64 backward tiers + extend 07-BASELINE.json in-place with backward p50/p95 entries (Phase 9 TEST-04 input)

### Phase 9: Integration Hardening & Correctness Gates
**Goal**: Every cross-cutting integration that real users hit -- `torch.compile`, DDP, FSDP, gradient checkpointing, deterministic mode, saved checkpoint round-tripping, `make_linear`/`LRU` -- works on the Triton backend, and the parametrized backend test suite plus the perf grid prove parity with the CUDA path
**Depends on**: Phase 8
**Requirements**: TEST-01, TEST-02, TEST-03, TEST-04, TEST-06, COMPAT-01, COMPAT-02, COMPAT-03, COMPAT-04, COMPAT-06
**Success Criteria** (what must be TRUE):
  1. Running `pytest tests/` with `TORCH_STRUCTURED_BACKEND=triton` passes the existing suite end-to-end; the `backend` fixture parametrizes every shared correctness test over `{triton, cuda, torch}` and asserts all three agree within documented tolerances
  2. `torch.compile(model)` on a model containing `Butterfly`/`ButterflyBmm`/`LRU`/`make_linear` traces cleanly through the Triton kernels with no graph breaks (resolves the dynamo fake-tensor bug from 260419-p27); 2-GPU FSDP smoke test with the `twiddle` no-shard hint succeeds, DDP and `torch.utils.checkpoint.checkpoint` produce correct gradients
  3. Checkpoints saved from v1.0/v1.1 (`twiddle` layout `(nstacks, nblocks, log_n, n/2, 2, 2)`) load via `Butterfly.load_state_dict` on the Triton backend without conversion and produce identical forward outputs to the CUDA path
  4. The published perf grid (log_n in {8,9,10,11} × batch × dtype, measured via `triton.testing.do_bench`) shows the Triton butterfly kernel at >=60% of CUDA throughput on every cell; for any cell below 60%, the runtime selector routes that shape to CUDA and the routing rule is documented
  5. README documents the CC 8.0+ (Ampere+) requirement for the Triton path, the deterministic-mode opt-in flag, and the recommendation that Volta sm_70 / Turing sm_75 users pin to v1.1 or use the CUDA backend
**Plans**: 3 plans

Plans:
- [x] 09-01-PLAN.md -- §0 LANDMINE fix in butterfly/multiply.py (D-05 delegators) + 3-axis conftest backend fixture with per-op cuda skip-gate (D-62) + _has_cuda_legacy_for_op probe + tests/test_phase9_integration.py (backend agreement, v1.0/v1.1 checkpoint round-trip, make_linear/LRU smoke, public API regression detector, BACKEND=triton subprocess smoke)
- [x] 09-02-PLAN.md -- set_deterministic API (D-63) + wrapper-level oracle fallback in _backward (D-63a) + tests/test_torch_compile_triton.py (fullgraph=True + FakeTensorMode 260419-p27 gate) + tests/test_distributed_triton.py (DDP gloo single-process + FSDP1 NCCL @pytest.mark.multigpu + gradient checkpointing use_reentrant=False) + tests/test_deterministic_mode.py + .github/workflows/test.yml multigpu job
- [ ] 09-03-PLAN.md -- _baseline_*.py CUDA p50 measurement + scripts/regenerate_routing_table.py + torch_structured/_routing.json + _should_route_to_cuda + resolver hook + tests/test_perf_grid.py TEST-04 gate + SC#4 reconciliation per D-61a (log_n=4 + _DISABLE_ROUTING) + README Triton-backend section + CHANGELOG.md (Keep a Changelog v1.1)

### Phase 10: CUDA Deprecation & flashmm Removal
**Goal**: Triton ships as the default backend for v1.2; the CUDA path remains available but emits a `DeprecationWarning` pointing at the migration timeline; the `_flashmm` MathDx kernel is removed entirely (not ported); `csrc/`, `setup.py`, and `MANIFEST.in` stay in-tree this release pending the two-release deprecation cadence
**Depends on**: Phase 9
**Requirements**: DEPR-01, DEPR-02, DEPR-03, DEPR-04, DEPR-05
**Success Criteria** (what must be TRUE):
  1. With no env var set, `import torch_structured` selects the Triton backend on a CUDA+Triton-capable machine; selecting `TORCH_STRUCTURED_BACKEND=cuda` still loads the existing `.so` artifacts and emits a single `DeprecationWarning` at import time pointing to the v1.3/v1.4 timeline in the README
  2. The `_flashmm` module is removed from `torch_structured/`, `csrc/flashmm/` is deleted, and any references to it raise `ModuleNotFoundError` with a clear message; `pytest tests/` passes without any test referencing flashmm
  3. `setup.py`, `MANIFEST.in`, and `csrc/{butterfly,hadamard,diag_mult,cpu,cuda}/` remain in-tree and `uv pip install .` still compiles them when `FORCE_CUDA=1` is set, so users on the CUDA path keep a working fallback through v1.2 and v1.3
  4. README and CHANGELOG document the deprecation timeline (v1.2 default-Triton + warning, v1.3 default-disabled CUDA build, future milestone removes `csrc/`) so any user importing the CUDA path knows when they need to migrate

**Plans**: 1 plan

Plans:
- [ ] 10-01-PLAN.md -- TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Build System Foundation | v1.0 | 1/1 | Complete | 2026-04-02 |
| 2. Extension Loading | v1.0 | 1/1 | Complete | 2026-04-02 |
| 3. Strip and Verify | v1.1 | 2/2 | Complete | 2026-04-03 |
| 4. Triton Dispatch Infrastructure | v1.2 | 2/2 | Complete | 2026-05-27 |
| 5. diag_mult Triton Port | v1.2 | 0/1 | Not started | - |
| 6. hadamard Triton Port | v1.2 | 0/1 | Not started | - |
| 7. butterfly Forward (Triton) | v1.2 | 0/2 | Not started | - |
| 8. butterfly Backward (Triton) | v1.2 | 0/2 | Not started | - |
| 9. Integration Hardening | v1.2 | 0/3 | Not started | - |
| 10. CUDA Deprecation & flashmm Removal | v1.2 | 0/1 | Not started | - |
