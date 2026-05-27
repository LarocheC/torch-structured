# torch_butterfly — Build System Modernization

## What This Is

A PyTorch library implementing butterfly matrices for efficient structured linear transforms (FFT, DCT, Hadamard, circulant, etc.) with C++/CUDA extensions. Now installable via `uv pip install .` or `uv pip install -e .` with a modern `pyproject.toml`, targeting Python 3.10+ and PyTorch 2.x.

## Core Value

A single `uv pip install .` (or `uv pip install -e .`) that just works — with CUDA support when available, without conda or manual steps.

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

### Active

(None — v1.1 complete)

### Out of Scope

- Pre-compiled wheel distribution — library value unproven (superseded by v1.2 Triton path: no wheels needed)
- ~~C++/CUDA kernel code changes — only build plumbing~~ (lifted in v1.2: Triton port replaces kernels)
- Rewriting or updating experiment code — removing, not fixing
- `_flashmm` MathDx kernel port — Triton cannot replicate MathDx tensor-core tuning; drop instead

## Current Milestone: v1.2 Triton Migration

**Goal:** Port C++/CUDA kernels to Triton so `uv pip install torch-structured` works with zero compilation step on any CUDA-capable machine.

**Target features:**
- ✓ Triton dispatch infrastructure (Phase 4 complete)
- ✓ Port `diag_mult` to Triton (Phase 5 complete — 2026-05-27)
- Port `hadamard` to Triton (Phase 6 next)
- Port butterfly forward kernel to Triton
- Port butterfly backward + autograd to Triton (replaces C++ `torch::autograd::Function`)
- Deprecate and remove `csrc/`, `setup.py` build shim, and MANIFEST.in

**Migration strategy:** Parallel paths — Triton kernels live alongside existing CUDA during migration; a runtime flag selects one. Lets the milestone ship incrementally and roll back per-kernel. `butterfly_multiply_torch` (pure-PyTorch reference) is preserved as the correctness oracle for every ported kernel.

## Context

Shipped v1.0 (build modernization) and v1.1 (repo cleanup). Starting v1.2 (Triton migration) to eliminate the install-time compile step that remains a friction point even with the modern pyproject.toml: users still need a CUDA toolchain and ~5-minute build. End state is a wheel-free, source-only Python package that JIT-compiles Triton kernels on first use.

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
*Last updated: 2026-05-27 — Phase 7 (butterfly_multiply forward, Triton) complete; Phase 8 (butterfly_multiply backward, Triton) next*
