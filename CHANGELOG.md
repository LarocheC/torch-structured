# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.0] - 2026-05-28

This release ports `torch_structured`'s GPU kernels to a Triton-based backend
while preserving full backward compatibility with the legacy CUDA C++ path.

### Added

- **Triton backend** for `butterfly_multiply`, `diag_mult`, and
  `hadamard_transform`. The Triton path is the default when both a CUDA
  device and PyTorch >= 2.6 are available. See
  [README "Triton backend (v1.2+)"](README.md#triton-backend-v12) for
  hardware requirements and usage.
- `torch_structured.set_deterministic(value: bool) -> bool` — opt-in flag
  that routes the `butterfly_multiply` backward through the pure-PyTorch
  oracle for bit-identical gradients. Composes additively (OR) with
  `torch.use_deterministic_algorithms(True)`.
- `torch_structured.set_backend(name)` — runtime backend selector
  (`triton` / `cuda` / `torch` / `auto`). Also configurable at import time
  via the `TORCH_STRUCTURED_BACKEND` env var.
- **Runtime routing selector** — when Triton trails CUDA on a given shape,
  the call transparently routes to the legacy CUDA path. Driven by a static
  routing table (`torch_structured/_routing.json`) baked from
  `triton.testing.do_bench` measurements. Regenerate locally on different
  hardware via `python scripts/regenerate_routing_table.py`.
- `scripts/regenerate_routing_table.py` — CLI utility that reads the perf
  baseline JSON and rebakes the routing table.
- Modern packaging via `pyproject.toml` — `uv pip install .` works without
  conda or manual steps.

### Changed

- **Minimum PyTorch version raised to 2.6** (was 1.8). PyTorch 2.6 ships
  the `triton_op` / `register_autograd` / `register_fake` plumbing the new
  backend depends on.
- **Minimum Python version raised to 3.10** (was 3.6).
- The default GPU backend is now Triton on Ampere+ (CC 8.0+). The legacy
  CUDA path remains available via `TORCH_STRUCTURED_BACKEND=cuda` when
  the `.so` is built.

### Deprecated

- The CUDA C++ backend is still available via `TORCH_STRUCTURED_BACKEND=cuda`
  but is no longer the default. See the v1.3 release notes for the deprecation
  timeline.

### Removed

- Nothing user-visible. The legacy `csrc/butterfly.cpp` symbols remain in
  the build but are no longer invoked by the default code path on
  Ampere+ hardware.

### Fixed

- §0 LANDMINE: `nn.Module` consumers (e.g., `Butterfly`) now correctly
  route through `set_backend()` re-bindings. Pre-1.2 the `@torch.jit.script`
  wrappers in `torch_structured/butterfly/multiply.py` captured the import-time
  C++ binding and bypassed `_ops.py`.
- `_has_cuda_legacy()` is now an honest runtime probe — combines the
  `hasattr` schema check with `torch.cuda.is_available()` and a one-shot
  tiny CUDA dispatch sanity call. Prevents false-positive `cuda` axis
  runs on hosts with CUDA-version mismatch between PyTorch and the
  toolkit that built the `.so`.

### Security

- `torch_structured/_routing.json` is committed to the repo and reviewed
  in PR; users on other hardware regenerate locally. A malicious routing
  entry would cause perf degradation, not security failure (both routed
  paths are vetted internal code).

### Hardware requirements

- **Ampere+ (CC 8.0+):** RTX 30xx/40xx, A100, H100 — full Triton support.
- **Volta (sm_70 — V100, Titan V) and Turing (sm_75 — T4, RTX 20xx):**
  pin to v1.1 or use the CUDA backend.

[Unreleased]: https://github.com/HazyResearch/torch-structured/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/HazyResearch/torch-structured/releases/tag/v1.2.0
