# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.3.0] - 2026-07-09

### Added

- `MonarchLinear` (`torch_structured/monarch/monarch_linear.py`): a genuine
  two-factor Monarch linear layer (block-diagonal x permutation x
  block-diagonal, per Dao et al. ICML 2022), built on the existing
  `blockdiag_butterfly_multiply` primitive. Gives full cross-channel mixing
  by construction, unlike the single block-diagonal factor `BlockdiagLinear`
  (now the `"blockdiag"` kind), which has zero cross-block information flow.
  Uses a variance-matched two-factor init (naively Kaiming-initializing each
  factor independently undershoots the correct composed output variance by
  ~3600x for a 400->1200 shape — regression-guarded by
  `tests/monarch/test_monarch_linear.py::test_variance_matched_init_regression_guard`).
- New `"blockdiag"` kind in `torch_structured/factory.py::make_linear`,
  wrapping the single block-diagonal-factor `BlockdiagLinear` (no permutation,
  zero cross-block mixing) with the small-`H` `nblocks` safety default. This is
  what the `"monarch"` kind used to build before the naming inversion below.
- Full-rank regression guard
  `tests/monarch/test_monarch_linear.py::test_composed_weight_reaches_full_rank`
  — asserts the composed dense-equivalent of a non-square `MonarchLinear`
  (400->1200, nblocks=4) reaches full rank `min(in, out) == 400`, locking out
  the intermediate-width rank bottleneck (a rank near `nblocks**2 == 16`
  indicates regression).
- `tests/monarch/test_blockdiag_butterfly_multiply.py::test_fast_matches_true_dense_ground_truth`
  — the fast `BlockdiagButterflyMultiply` autograd Function previously had no
  dedicated correctness test (only the slow reference implementation was
  tested, and only shape-checked for non-square cases). This checks the fast
  op against a true dense ground truth (explicit `torch.block_diag` +
  explicit permutation matrices, independent of the implementation's own
  `einops.rearrange` logic) for a non-square, non-power-of-2 shape.

### Changed

- **BREAKING — `torch_structured/factory.py` naming inversion.** The
  `"monarch"` kind now builds the genuine two-factor `MonarchLinear`
  (block-diagonal x permutation x block-diagonal, full cross-channel mixing by
  construction). It previously built the single block-diagonal factor
  `BlockdiagLinear`, which is now the new `"blockdiag"` kind. Code that relied
  on `make_linear("monarch", ...)` returning a single-factor block-diagonal
  layer must switch to `make_linear("blockdiag", ...)`. `_SUPPORTED` is now
  `("dense", "butterfly", "monarch", "blockdiag", "circulant")`.

### Fixed

- **`MonarchLinear` rank bottleneck.** The two intermediate block dims are now
  `q = r = in_blksz` (were `nblocks`), so the intermediate width
  `k*q == nblocks * in_blksz == in_features_extended`. Previously the width was
  capped at `nblocks**2` (a fixed 16 for `nblocks=4`) regardless of feature
  sizes, upper-bounding the composed dense-equivalent rank at `nblocks**2`. For
  a 400->1200 layer with `nblocks=4` the composed rank goes from 16 to the full
  400, the parameter count from 6400 to 160000, and the compression `saving`
  from 0.013 to 0.333.
- **`MonarchLinear` variance-matched init** corrected for the new fan-in. The
  init scale depends on the two contraction fan-ins (`p` in `x @ w1^T`, `r` in
  `out1 @ w2^T`), not on `k*q`; the rank fix changed `r` from `nblocks` to
  `in_blksz`, so each factor is now rescaled by
  `sqrt((in_features_extended / in_blksz**2) * v_target)` to restore the
  dense-equivalent composed output variance (reduces to the previous
  `sqrt(v_target)` under the old shapes). Guarded by the existing
  `test_variance_matched_init_regression_guard`.

### Removed

- **`"monarch2"` factory kind.** Its role (the genuine two-factor Monarch) has
  been folded into the now-correct `"monarch"` kind. `make_linear("monarch2",
  ...)` now raises `ValueError`.

## [1.2.5] - 2026-07-09

### Fixed

- Triton backward (`_butterfly_backward_kernel`) no longer fails to compile on
  Triton 3.3.x with `ValueError: Did you forget to add @triton.jit ?`. The
  per-stage `STRIDE` `tl.constexpr` values were computed inside the
  `@triton.jit` body using the Python `max()` builtin and a `... if ... else`
  ternary, which the Triton 3.3.x front-end can no longer evaluate in constexpr
  context. Since every input is host-known, the computation is hoisted to the
  kernel launch site and the strides are passed in as constexpr kernel args
  (`STRIDE_0/1/2`) — matching how `tile_n`/`num_warps` are already
  host-computed. Behavior-preserving: forward + backward parity vs the
  pure-PyTorch oracle holds across fp32/complex64, `nblocks` 1/2,
  `increasing_stride` True/False, `log_n` 2..7 (verified on Triton 3.7.0, where
  the old form still compiled).

## [1.2.4] - 2026-05-31

### Fixed

- `Butterfly` (and `ButterflyUnitary`/`ButterflyBmm`) forward no longer raises
  `AssertionError: input must be contiguous (Pitfall 3)` on CUDA for *expanding*
  layers (`out_size > in_size`, i.e. `nstacks > 1`). `pre_process` produces a
  stride-0 `.expand()` view, which the Triton kernel previously rejected; the
  kernel now coerces inputs to contiguous (forward and the autograd-saved
  tensors in `setup_context`), a no-op when already contiguous. This also
  removes a CPU/CUDA divergence — the same expanding layer worked on CPU but
  asserted on GPU. (torch-structured-7ny)

### Documentation

- README quickstart: corrected the transforms import to
  `from torch_structured.butterfly.special import fft, hadamard`
  (`torch_structured.special` is not a module).

## [1.2.3] - 2026-05-31

### Fixed

- `import torch_structured` no longer crashes when an ABI-incompatible compiled
  extension is present on disk. Extension loading is now restricted to the
  running interpreter's own extension suffixes (so a stale `.so` built for a
  different Python — e.g. a `cpython-313` build under a 3.12 environment — is
  ignored instead of selected), and a `load_library` failure (an
  undefined-symbol `OSError` from a Python/PyTorch-ABI-mismatched `.so`) is now
  caught and downgraded to a warning with a fallback to the Triton / pure-PyTorch
  backend, rather than raising. Previously the graceful fallback only handled a
  *missing* extension; a *present-but-incompatible* one was fatal.

## [1.2.2] - 2026-05-29

### Fixed

- The CUDA-backend `DeprecationWarning` no longer fires on a bare
  `import torch_structured` when the default Triton backend is active and a
  legacy CUDA build happens to be present. It is now emitted only on explicit
  CUDA selection (`set_backend("cuda")` / `TORCH_STRUCTURED_BACKEND=cuda`),
  matching the intended deprecation semantics. The warning was decoupled from
  module-import timing into an idempotent emitter.

### Changed

- Test suite only: the 3-axis backend-agreement gate `{torch, triton, cuda}`
  now passes cleanly on matched-CUDA hardware. fp32 cross-backend tolerances
  track the genuine fp32 accumulation noise floor, and fp64/complex gradchecks
  skip the CUDA axis (the legacy CUDA kernels are fp32-real-only, like Triton).
  No library behavior change.

## [1.2.1] - 2026-05-28

### Changed

- Removed the maintainer email address from package metadata
  (`authors`/`maintainers`). Contact is now via
  [GitHub Issues](https://github.com/LarocheC/torch-structured/issues), added
  as an `Issues` entry in `[project.urls]`. No functional changes.

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
- **DeprecationWarning on `TORCH_STRUCTURED_BACKEND=cuda` import path**
  (fires once per process via `warnings.simplefilter("once", DeprecationWarning)`
  in `torch_structured/_cuda_legacy/__init__.py`). Phase 10.

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
- **`TORCH_STRUCTURED_BACKEND=cuda`** is now soft-deprecated: still available
  via env var or `set_backend('cuda')` but emits a one-time `DeprecationWarning`.
  Will be default-disabled in v1.3; removed in v1.4+. See README
  ["Deprecation timeline"](README.md#deprecation-timeline). Phase 10.

### Removed

- The legacy `csrc/butterfly.cpp`, `csrc/diag_mult/`, and `csrc/hadamard/`
  symbols remain in the build but are no longer invoked by the default code
  path on Ampere+ hardware. (Slated for removal in v1.4+ per the
  "Deprecation timeline" section in README.)
- `_flashmm` MathDx kernel (`csrc/flashmm/`, `torch_structured/monarch/flash_mm.py`,
  `tests/monarch/test_flash_mm.py`) — see README ["Deprecation timeline"](README.md#deprecation-timeline)
  for the broader CUDA-path retirement plan. Phase 10.

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

[Unreleased]: https://github.com/LarocheC/torch-structured/compare/v1.3.0...HEAD
[1.3.0]: https://github.com/LarocheC/torch-structured/releases/tag/v1.3.0
[1.2.5]: https://github.com/LarocheC/torch-structured/releases/tag/v1.2.5
[1.2.4]: https://github.com/LarocheC/torch-structured/releases/tag/v1.2.4
[1.2.3]: https://github.com/LarocheC/torch-structured/releases/tag/v1.2.3
[1.2.2]: https://github.com/LarocheC/torch-structured/releases/tag/v1.2.2
[1.2.1]: https://github.com/LarocheC/torch-structured/releases/tag/v1.2.1
[1.2.0]: https://github.com/LarocheC/torch-structured/releases/tag/v1.2.0
