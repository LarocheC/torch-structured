# Requirements: torch_butterfly

**Defined:** 2026-04-02
**Core Value:** A single `uv pip install .` that just works — with CUDA support when available

## v1.0 Requirements (Complete)

### Build System

- [x] **BUILD-01**: Package uses pyproject.toml with `[build-system]` declaring torch, setuptools, ninja, wheel as build dependencies
- [x] **BUILD-02**: Package metadata follows PEP 621 (`[project]` table with name, version, description, python-requires, etc.)
- [x] **BUILD-03**: Package discovery excludes legacy `butterfly/` directory, only installs `torch_butterfly/`
- [x] **BUILD-04**: MANIFEST.in includes `csrc/` source files so sdist contains C++/CUDA sources
- [x] **BUILD-05**: Optional dependency groups defined for dev and test extras (`[project.optional-dependencies]`)
- [x] **BUILD-06**: setup.py reduced to thin shim (only ext_modules and BuildExtension cmdclass)

### CUDA Compatibility

- [x] **CUDA-01**: Hardcoded sm_35 CUDA architecture flag removed (broken on CUDA 12+)
- [x] **CUDA-02**: Build auto-detects CUDA availability from torch and CUDA_HOME (existing behavior preserved and improved)
- [x] **CUDA-03**: TORCH_CUDA_ARCH_LIST env var supported for user override of CUDA architecture targets
- [x] **CUDA-04**: FORCE_CUDA and FORCE_CPU env vars preserved as manual overrides

### Extension Loading

- [x] **EXT-01**: PathFinder-based extension loading in `__init__.py` replaced with `__file__`-relative .so discovery
- [x] **EXT-02**: Editable installs (`uv pip install -e .`) load extensions correctly
- [x] **EXT-03**: Deprecated RegisterOperators API in `csrc/version.cpp` migrated to TORCH_LIBRARY macro

### Install Experience

- [x] **INST-01**: `uv pip install .` works from source checkout (non-editable)
- [x] **INST-02**: `pip install .` works from source checkout (non-editable)
- [x] **INST-03**: `uv pip install -e .` works for development (editable)
- [x] **INST-04**: Install works without conda — pure pip/uv workflow
- [x] **INST-05**: Python >=3.10, <4 enforced in metadata
- [x] **INST-06**: PyTorch >=2.0 declared as runtime dependency

## v1.1 Requirements

### Legacy Removal

- [x] **LEGACY-01**: Remove `butterfly/` package (old implementation replaced by `torch_butterfly/`)
- [x] **LEGACY-02**: Remove `tests_old/` directory (tests for old `butterfly/` package)
- [x] **LEGACY-03**: Remove `learning_transforms/` directory (Cython experiments)
- [x] **LEGACY-04**: Remove `fairseq/` git submodule and `.gitmodules` reference

### Experiment Removal

- [x] **EXP-01**: Remove `cnn/` directory (CIFAR/ImageNet experiment scripts)
- [x] **EXP-02**: Remove `convolution/` directory (Lightning/Hydra/Ray experiments)
- [x] **EXP-03**: Remove `transformer/` directory (dynamic conv experiments)
- [x] **EXP-04**: Remove `gumbel-sinkhorn/` directory (sorting network experiments)

### Cleanup

- [x] **CLEAN-01**: Remove `data/` directory (dataset files)
- [x] **CLEAN-02**: Remove `ray_template.sh`
- [x] **CLEAN-03**: Remove `build/` and `torch_butterfly.egg-info/` from tracking and add to `.gitignore`
- [x] **CLEAN-04**: All existing tests in `tests/` pass after cleanup

## v1.2 Requirements

### Backend Dispatch Infrastructure (DISP)

- [x] **DISP-01**: User can select backend via `TORCH_STRUCTURED_BACKEND` env var (values: `triton`, `cuda`, `torch`, `auto`)
- [x] **DISP-02**: `auto` mode selects Triton if available → CUDA `.so` if loaded → pure-PyTorch fallback, in that order
- [x] **DISP-03**: Backend is selected once at import time via a single `torch_structured/_ops.py` dispatch module (no per-call branching)
- [x] **DISP-04**: User can call `torch_structured.set_backend("triton"|"cuda"|"torch")` from Python at runtime for tests
- [x] **DISP-05**: Library logs the selected backend at import time so users can verify which path ran

### Triton Kernel Ports (TRI)

- [ ] **TRI-01**: `diag_mult` runs on Triton (forward + backward, fp32 + complex64)
- [ ] **TRI-02**: `hadamard` runs on Triton (self-inverse, forward kernel only, fp32)
- [ ] **TRI-03**: `butterfly_multiply` forward runs on Triton (fp32 + complex64, all `increasing_stride`/`output_size`/`nstacks`/`nblocks` combinations)
- [ ] **TRI-04**: `butterfly_multiply` backward runs on Triton with fp32 scratch accumulator for atomic adds (no direct bf16/fp16 atomicAdd)
- [x] **TRI-05**: All Triton kernels registered via `torch.library.triton_op` + `register_autograd` + `wrap_triton` (not `torch.autograd.Function`)
- [x] **TRI-06**: Complex64 implemented via real/imag-split arithmetic, with the layout decision documented in Phase 1
- [x] **TRI-07**: `butterfly_multiply_torch` remains as runtime fallback for CPU / no-Triton environments — not deleted

### Correctness & Performance Gates (TEST)

- [ ] **TEST-01**: Every Triton kernel passes correctness vs `butterfly_multiply_torch` reference at fp32 (rtol=1e-5, atol=1e-6) and complex64 (rtol=1e-4)
- [ ] **TEST-02**: Backward correctness validated via `gradcheck` against `autograd.grad(_torch_fw, ...)` — not against the CUDA reference
- [ ] **TEST-03**: Test suite parametrizes over `backend ∈ {triton, cuda, torch}` and asserts all three agree (within tolerance)
- [ ] **TEST-04**: Butterfly Triton kernel achieves ≥60% of existing CUDA throughput on log_n ∈ {8, 9, 10, 11} via `triton.testing.do_bench`
- [x] **TEST-05**: CI persists `TRITON_CACHE_DIR` between runs so first-call JIT cost doesn't compound
- [ ] **TEST-06**: Existing test suite (`pytest tests/`) passes with `TORCH_STRUCTURED_BACKEND=triton` set

### Compatibility Constraints (COMPAT)

- [ ] **COMPAT-01**: `Butterfly`, `ButterflyBmm`, `ButterflyUnitary`, `ButterflyBase4` public nn.Module API is byte-identical (no signature/attribute changes)
- [ ] **COMPAT-02**: Twiddle parameter layout `(nstacks, nblocks, log_n, n/2, 2, 2)` unchanged — saved checkpoints from v1.0/v1.1 load without conversion
- [ ] **COMPAT-03**: `make_linear` factory and `LRU` recurrent layer continue to work unchanged on the Triton backend
- [ ] **COMPAT-04**: `torch.compile(model)` traces cleanly through Triton kernels (resolves dynamo fake-tensor bug from quick task 260419-p27)
- [x] **COMPAT-05**: PyTorch minimum bumped from `>=2.0` to `>=2.6` in `pyproject.toml`
- [ ] **COMPAT-06**: README documents that Triton path requires CC 8.0+ (Ampere+); Volta sm_70 and Turing sm_75 users pin to v1.1 or use the CUDA backend with self-built `.so`

### CUDA Deprecation Cadence (DEPR)

- [ ] **DEPR-01**: v1.2 ships with Triton as the default backend, CUDA available via `TORCH_STRUCTURED_BACKEND=cuda`
- [ ] **DEPR-02**: Importing the CUDA path emits `DeprecationWarning` pointing to the migration plan
- [ ] **DEPR-03**: `setup.py`, `MANIFEST.in`, and `csrc/` remain in-tree during v1.2 (deletion deferred to a later milestone gated by 2-release deprecation cadence)
- [ ] **DEPR-04**: `_flashmm` (opt-in MathDx kernel) explicitly removed in v1.2 — not ported, not maintained
- [ ] **DEPR-05**: README and CHANGELOG document the deprecation timeline so users know when to migrate

## Future Requirements

### Distribution

- **DIST-01**: Pre-compiled wheels for common CUDA/Python version combinations (likely obsoleted by v1.2 Triton path)
- **DIST-02**: CI matrix for automated wheel building (likely obsoleted by v1.2)
- **DIST-03**: Conda recipe for conda-forge distribution

### Extended Platform Support

- **PLAT-01**: ROCm (AMD GPU) support — Triton path gives this for free in v1.2 but untested; formal validation deferred
- **PLAT-02**: Intel XPU support

### Triton Path Follow-ups (post-v1.2)

- **TRI-FUT-01**: bf16/fp16 dtype matrix for butterfly kernels (deferred from v1.2 to keep correctness gate manageable)
- **TRI-FUT-02**: 5-stage multi-stage butterfly tile in Triton (v1.2 ships 3-stage only)
- **TRI-FUT-03**: Fused diag×butterfly and perm×butterfly kernels
- **TRI-FUT-04**: Final deletion of `csrc/`, `setup.py`, `MANIFEST.in` after 2-release deprecation cadence

## Out of Scope

| Feature | Reason |
|---------|--------|
| ~~Pre-compiled wheel distribution~~ | Superseded by v1.2 Triton path — no wheels needed |
| Rewriting or updating experiment code | Removing, not fixing |
| ~~C++/CUDA kernel code changes — only build plumbing~~ | Lifted in v1.2: Triton port replaces kernels |
| `_flashmm` Triton port | Triton cannot replicate MathDx tensor-core tuning; drop rather than port |
| Native `tl.complex64` reliance | Triton has none; we use real/imag-split arithmetic |
| Matching hand-tuned CUDA performance | Target is 60–90% of CUDA, not parity |
| Volta sm_70 / Turing sm_75 on Triton path | Triton requires sm_80+; affected users pin to v1.1 |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| BUILD-01 | Phase 1 | Complete |
| BUILD-02 | Phase 1 | Complete |
| BUILD-03 | Phase 1 | Complete |
| BUILD-04 | Phase 1 | Complete |
| BUILD-05 | Phase 1 | Complete |
| BUILD-06 | Phase 1 | Complete |
| CUDA-01 | Phase 1 | Complete |
| CUDA-02 | Phase 1 | Complete |
| CUDA-03 | Phase 1 | Complete |
| CUDA-04 | Phase 1 | Complete |
| EXT-01 | Phase 2 | Complete |
| EXT-02 | Phase 2 | Complete |
| EXT-03 | Phase 2 | Complete |
| INST-01 | Phase 1 | Complete |
| INST-02 | Phase 1 | Complete |
| INST-03 | Phase 2 | Complete |
| INST-04 | Phase 1 | Complete |
| INST-05 | Phase 1 | Complete |
| INST-06 | Phase 1 | Complete |
| LEGACY-01 | Phase 3 | Complete |
| LEGACY-02 | Phase 3 | Complete |
| LEGACY-03 | Phase 3 | Complete |
| LEGACY-04 | Phase 3 | Complete |
| EXP-01 | Phase 3 | Complete |
| EXP-02 | Phase 3 | Complete |
| EXP-03 | Phase 3 | Complete |
| EXP-04 | Phase 3 | Complete |
| CLEAN-01 | Phase 3 | Complete |
| CLEAN-02 | Phase 3 | Complete |
| CLEAN-03 | Phase 3 | Complete |
| CLEAN-04 | Phase 3 | Complete |
| DISP-01 | Phase 4 | Complete |
| DISP-02 | Phase 4 | Complete |
| DISP-03 | Phase 4 | Complete |
| DISP-04 | Phase 4 | Complete |
| DISP-05 | Phase 4 | Complete |
| COMPAT-05 | Phase 4 | Complete |
| TRI-05 | Phase 4 | Complete |
| TRI-06 | Phase 4 | Complete |
| TRI-07 | Phase 4 | Complete |
| TEST-05 | Phase 4 | Complete |
| TRI-01 | Phase 5 | Pending |
| TRI-02 | Phase 6 | Pending |
| TRI-03 | Phase 7 | Pending |
| TRI-04 | Phase 8 | Pending |
| TEST-01 | Phase 9 | Pending |
| TEST-02 | Phase 9 | Pending |
| TEST-03 | Phase 9 | Pending |
| TEST-04 | Phase 9 | Pending |
| TEST-06 | Phase 9 | Pending |
| COMPAT-01 | Phase 9 | Pending |
| COMPAT-02 | Phase 9 | Pending |
| COMPAT-03 | Phase 9 | Pending |
| COMPAT-04 | Phase 9 | Pending |
| COMPAT-06 | Phase 9 | Pending |
| DEPR-01 | Phase 10 | Pending |
| DEPR-02 | Phase 10 | Pending |
| DEPR-03 | Phase 10 | Pending |
| DEPR-04 | Phase 10 | Pending |
| DEPR-05 | Phase 10 | Pending |

**Coverage:**
- v1.0 requirements: 19 total, 19 complete
- v1.1 requirements: 12 total, 12 complete
- v1.2 requirements: 28 total, 0 complete, 28 mapped (Phases 4-10)
- Unmapped: 0

---
*Requirements defined: 2026-04-02*
*Last updated: 2026-05-26 — v1.2 Triton Migration roadmap created; 28 requirements mapped across Phases 4-10*
