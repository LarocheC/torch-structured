# torch_butterfly — Build System Modernization

## What This Is

A PyTorch library implementing butterfly matrices for efficient structured linear transforms (FFT, DCT, Hadamard, circulant, etc.). As of v1.2 the GPU kernels run on a **Triton JIT backend by default** (no compiled `.so` needed); the legacy C++/CUDA path remains opt-in and deprecated. Installable via `uv pip install torch-structured` — published to PyPI as a pure-Python `py3-none-any` wheel — targeting Python 3.10+ and PyTorch 2.6+.

## Core Value

A single `uv pip install torch-structured` that just works on any CUDA-capable machine with **zero compilation step** — Triton JIT-compiles kernels on first use; no conda, no toolchain, no wheel matrix.

## Requirements

### Validated

- ✓ C++/CUDA butterfly multiply kernels (csrc/) — existing
- ✓ Python butterfly module layer (torch_butterfly/) — existing
- ✓ Special transforms (FFT, DCT, Hadamard, etc.) — existing
- ✓ Pure-PyTorch fallback when no CUDA — existing
- ✓ Test suite (tests/) — existing
- ✓ pyproject.toml-based build with PEP 621 metadata — v1.0
- ✓ UV-compatible installation (uv pip install . and -e .) — v1.0
- ✓ pip install . works — v1.0
- ✓ Python 3.10+ enforced in metadata — v1.0
- ✓ PyTorch 2.x compatibility — v1.0
- ✓ CUDA extension compilation with FORCE_CUDA/FORCE_CPU/TORCH_CUDA_ARCH_LIST — v1.0
- ✓ Simplified dependencies (torch + numpy only) — v1.0
- ✓ Glob-based extension loading replacing PathFinder — v1.0
- ✓ RegisterOperators migrated to TORCH_LIBRARY_FRAGMENT — v1.0
- ✓ CUDA version check downgraded to warning — v1.0
- ✓ Legacy butterfly/ package removed — v1.1
- ✓ Experiment directories removed (cnn/, convolution/, transformer/, learning_transforms/, gumbel-sinkhorn/) — v1.1
- ✓ Dead assets removed (fairseq submodule, data/, ray_template.sh) — v1.1
- ✓ Build artifacts cleaned from tracking, .gitignore updated — v1.1
- ✓ README.md modernized for Python 3.10+, PyTorch 2.x, uv/pip — v1.1

- ✓ Triton dispatch infrastructure (`_ops.py` resolver, `set_backend`, `TORCH_STRUCTURED_BACKEND`) — v1.2 (Phase 4)
- ✓ `diag_mult` ported to Triton — v1.2 (Phase 5)
- ✓ `hadamard_transform` ported to Triton — v1.2 (Phase 6)
- ✓ `butterfly_multiply` forward (fp32 + complex64) ported to Triton — v1.2 (Phase 7)
- ✓ `butterfly_multiply` backward + autograd ported to Triton (fp32 scratch accumulator) — v1.2 (Phase 8)
- ✓ Integration hardening: torch.compile, DDP/FSDP, grad checkpointing, deterministic mode, 3-axis backend-agreement suite, perf grid + runtime routing — v1.2 (Phase 9)
- ✓ CUDA backend deprecated (DeprecationWarning, 2-release cadence) + `_flashmm` removed — v1.2 (Phase 10)
- ✓ Triton becomes default backend on Ampere+ (CC 8.0) — v1.2
- ✓ Published to PyPI as pure-Python `py3-none-any` wheel (1.2.0/1.2.1/1.2.2) — v1.2

### Active

(None — v1.2 shipped. Define next milestone via `/gsd-new-milestone`.)

### Out of Scope

- Pre-compiled wheel distribution — library value unproven (superseded by v1.2 Triton path: no wheels needed)
- ~~C++/CUDA kernel code changes — only build plumbing~~ (lifted in v1.2: Triton port replaces kernels)
- Rewriting or updating experiment code — removing, not fixing
- `_flashmm` MathDx kernel port — Triton cannot replicate MathDx tensor-core tuning; drop instead

## Current State

**v1.2 Triton Migration — SHIPPED 2026-05-29** (PyPI: 1.2.0 → 1.2.1 → 1.2.2; tags v1.2.0/v1.2.1/v1.2.2; milestone tag v1.2).

All GPU kernels (`diag_mult`, `hadamard_transform`, `butterfly_multiply` fwd+bwd) run on Triton by default; the legacy CUDA C++ path is opt-in and emits a DeprecationWarning (default-disabled in v1.3, removed v1.4+ per the 2-release cadence). A static routing table transparently falls genuine Triton-slower shapes back to CUDA when a legacy build is present. The package installs as a pure-Python wheel — no compilation, no toolchain.

**Known deferred (env-limited):** FSDP 2-GPU smoke (`@pytest.mark.multigpu`) is shipped and code-correct but never executed (single-GPU dev host); run on ≥2 NCCL GPUs when available.

## Next Milestone Goals

Candidates (not yet scoped — run `/gsd-new-milestone`):
- **v1.3:** default-disable the CUDA backend; ship wheel without `csrc/`.
- **v1.4+:** delete `csrc/`, `setup.py` CUDA extension code, and `_cuda_legacy/` (per the deprecation timeline).
- Possible: autotuning / larger-shape Triton perf work; broaden the published wheel's CI matrix; TestPyPI + trusted-publishing automation.

## Context

Shipped v1.0 (build modernization), v1.1 (repo cleanup), and v1.2 (Triton migration). v1.2 eliminated the install-time compile step: the package is now a wheel-free, source-only pure-Python distribution that JIT-compiles Triton kernels on first use, published to PyPI under the LarocheC fork. Codebase: butterfly/structured/monarch primitives + a Triton kernel layer (`_triton/`), a pure-PyTorch oracle (`_torch_ref/`), and a deprecated legacy CUDA path (`_cuda_legacy/`). Dev host for v1.2: single RTX 2000 Ada (sm_89), CUDA 13.0, PyTorch 2.6+.

## Constraints

- **Build system:** pyproject.toml as single source of truth
- **UV compatibility:** Works with `uv pip install` without conda
- **CUDA support:** Retained via torch.utils.cpp_extension
- **Python:** >=3.10, <4
- **PyTorch:** >=2.0

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Keep torch.utils.cpp_extension | Works with PyTorch 2.x, avoids wheel complexity | ✓ Good |
| Target Python 3.10+ only | Modern packaging, user preference | ✓ Good |
| Target PyTorch 2.x only | Simplifies compat | ✓ Good |
| Skip pre-compiled wheels | Library value unproven | ✓ Good |
| CUDA arch 7.0 8.0 9.0+PTX | Volta through Hopper with forward compat | ✓ Good |
| Remove no_python_abi_suffix | Standard ABI tags with glob discovery | ✓ Good |
| TORCH_LIBRARY_FRAGMENT for version.cpp | Avoids namespace collision with butterfly.cpp | ✓ Good |
| Downgrade CUDA check to warning | Prevents crash on version mismatch | ✓ Good |
| setuptools>=64 (not >=77) | License table format compatible with older setuptools | ✓ Good |
| Triton JIT backend, default on Ampere+ (CC 8.0) | Eliminates compile step; wheel-free install | ✓ Good — v1.2 |
| Parallel CUDA/Triton paths during migration | Per-kernel rollback; ship incrementally | ✓ Good — v1.2 |
| `_torch_ref/` pure-PyTorch oracle as correctness gate | Every Triton kernel gradchecked against it | ✓ Good — v1.2 |
| Static routing table (bake `do_bench` p50) | Falls genuine Triton-slower shapes back to CUDA | ✓ Good — v1.2 |
| 2-release CUDA deprecation cadence (warn v1.2 → off v1.3 → delete v1.4+) | Gives users time to migrate | — Pending (v1.3/v1.4) |
| `DeprecationWarning` decoupled into explicit emitter | Default-backend import stays silent (DEPR-02 fix) | ✓ Good — v1.2.2 |
| Gate CUDA build behind FORCE_CUDA=1; pure-Python wheel | Publishable to PyPI without a wheel matrix | ✓ Good — v1.2 |
| Drop maintainer email from package metadata | Contact via GitHub Issues; reduce exposure | ✓ Good — v1.2.1 |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-29 — after v1.2 Triton Migration milestone (shipped to PyPI 1.2.0/1.2.1/1.2.2)*
