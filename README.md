# torch-structured

Consolidated PyTorch library of structured-matrix primitives:

- **`torch_structured`** (core) — butterfly matrices for exact fast linear transforms (FFT, iFFT, DCT, DST, Hadamard, circulant, Toeplitz) as learnable `nn.Module` drop-in replacements for `nn.Linear`.
- **`torch_structured.structured`** — low-displacement-rank layers ported from [structured-nets](https://github.com/HazyResearch/structured-nets): Toeplitz-like, Hankel, Vandermonde, Fastfood, Circulant, LDR subdiagonal / tridiagonal, Krylov utilities.
- **`torch_structured.monarch`** — Monarch / block-diagonal-butterfly primitives ported from [m2](https://github.com/HazyResearch/m2): block-diagonal and block-diagonal-butterfly multiplies, structured linear layers, butterfly-factor helper, and Hyena implicit long filter.

See the `NOTICE` file for upstream attributions and citations.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- NumPy, SciPy, einops, opt_einsum
- A C++ compiler supporting C++14 (for building extensions)
- CUDA toolkit (optional, for GPU acceleration)

## Installation

```bash
uv pip install .            # or: pip install .
uv pip install -e ".[dev]"  # development install
```

### CUDA support

CUDA extensions are compiled automatically when a CUDA toolkit is detected. Override with env vars:

```bash
FORCE_CUDA=1 uv pip install .   # force CUDA compilation
FORCE_CPU=1 uv pip install .    # force CPU-only build
```

`TORCH_CUDA_ARCH_LIST` targets specific GPU architectures (default: `"7.0 8.0 9.0+PTX"`).

Built extensions (CUDA builds):
- `torch_structured._butterfly`, `torch_structured._version` — core butterfly ops (torch.ops-style).
- `torch_structured._hadamard_cuda` — fast Walsh-Hadamard transform (pybind module).
- `torch_structured._diag_mult_cuda` — subdiagonal cycle-multiply helper (pybind module).

## Quickstart

### Core butterfly

```python
import torch
from torch_structured import Butterfly
from torch_structured.special import fft, hadamard

layer = Butterfly(in_size=1024, out_size=1024)
fft_layer = fft(1024)
hadamard_layer = hadamard(1024)
```

### Structured (LDR) layers

```python
from torch_structured.structured.layers import ToeplitzLike, LDRSubdiagonal
from torch_structured.structured.hadamard import hadamard_transform_torch

toeplitz = ToeplitzLike(layer_size=256, r=2)
ldr_sd = LDRSubdiagonal(layer_size=256, r=2)
y = hadamard_transform_torch(torch.randn(4, 128))
```

### Monarch primitives

```python
import torch
from torch_structured.monarch.blockdiag_linear import BlockdiagLinear
from torch_structured.monarch.blockdiag_butterfly_multiply import (
    blockdiag_butterfly_multiply,
)

linear = BlockdiagLinear(in_features=512, out_features=512, nblocks=4)
# low-level multiply:
x = torch.randn(8, 64)
w1 = torch.randn(8, 8, 8)
w2 = torch.randn(8, 8, 8)
out = blockdiag_butterfly_multiply(x, w1, w2)
```

## Triton backend (v1.2+)

Starting with v1.2, `torch_structured` ships a Triton-based GPU backend that replaces
the CUDA C++ extensions for the main kernels (`butterfly_multiply`, `diag_mult`,
`hadamard_transform`). The Triton path is the default when both a CUDA device and
PyTorch >= 2.6 are available.

### Hardware requirements

The Triton backend requires NVIDIA CUDA Compute Capability **CC 8.0 or later**
(Ampere generation: RTX 30xx/40xx, A100, H100, etc.). Older GPUs are NOT
supported on the Triton path:

- **Volta (sm_70 — V100, Titan V):** pin to v1.1 (`pip install torch-structured==1.1.*`)
  OR use the CUDA backend with a self-built `.so`.
- **Turing (sm_75 — T4, RTX 20xx):** same recommendation as Volta.

Switch to the CUDA backend (when the legacy `.so` is built) via:

```bash
export TORCH_STRUCTURED_BACKEND=cuda
```

### Deterministic mode

By default, the Triton backward kernel uses atomic-add reductions for
`d_twiddle` accumulation, which can produce slightly different results across
runs (within documented tolerance, but not bit-identical).

For reproducible gradients, opt into deterministic mode:

```python
import torch_structured

torch_structured.set_deterministic(True)
# ... training step ...
torch_structured.set_deterministic(False)
```

Under deterministic mode, the backward routes through the pure-PyTorch oracle
(`butterfly_multiply_torch`) — slower, but deterministic by construction.
Deterministic mode also activates automatically when
`torch.use_deterministic_algorithms(True)` is set globally (additive OR
composition with PyTorch's flag).

### Switching backends

Use `TORCH_STRUCTURED_BACKEND` at import time OR
`torch_structured.set_backend()` at runtime:

```bash
TORCH_STRUCTURED_BACKEND=triton  # default on Ampere+
TORCH_STRUCTURED_BACKEND=cuda    # legacy CUDA C++ path (requires built .so)
TORCH_STRUCTURED_BACKEND=torch   # pure-PyTorch fallback (CPU OK)
TORCH_STRUCTURED_BACKEND=auto    # try triton -> cuda -> torch
```

### Runtime selector

On some shapes the Triton kernel may be slower than the legacy CUDA path. To
avoid forcing users to choose between backends, the library ships a static
routing table (`torch_structured/_routing.json`) baked from
`triton.testing.do_bench`-style measurements at packaging time. When you call
a routed shape with the Triton backend AND the legacy `.so` is available, the
call transparently routes to CUDA. The selector is dormant when no cell is
marked `route_to_cuda` — Triton handles every shape.

To regenerate the routing table on your hardware:

```bash
python tests/_baseline_butterfly.py            # regenerate forward perf grid
python tests/_baseline_butterfly_backward.py   # regenerate backward perf grid
python scripts/regenerate_routing_table.py     # rebake _routing.json
```

## Deprecation timeline

torch_structured ships Triton as the default backend in v1.2. The legacy CUDA
C++ backend (`csrc/`) is being retired over a two-release deprecation cadence:

- **v1.2 (current):** Triton is the default. `TORCH_STRUCTURED_BACKEND=cuda`
  still works for users who built `_butterfly.so` / `_diag_mult.so` /
  `_hadamard.so` locally, but emits a one-time `DeprecationWarning` at import
  time pointing here. The Monarch Mixer MathDx kernel (previously
  vendored under `csrc/`) is removed entirely in v1.2; see the CHANGELOG for
  the full file list.
- **v1.3 (next minor release, ~6 months out):** CUDA build is default-disabled.
  `csrc/` extensions stay in the source tree and can still be compiled via
  `FORCE_CUDA=1`, but the PyPI wheel does NOT include them. The
  `DeprecationWarning` still fires when a locally-built CUDA path is used.
- **v1.4+ (post-milestone):** `csrc/` tree, `setup.py` CUDA extension code,
  and `_cuda_legacy/` are deleted. The standard 2-release deprecation cadence
  gives users two minor releases to migrate.

Migration: most users should set nothing and let the Triton default take over.
If you have a workload that needs the CUDA backend (e.g., Volta sm_70 / Turing
sm_75 hardware that Triton doesn't fully support), see the
["Triton backend"](#triton-backend-v12) section above for hardware
requirements; otherwise pin to v1.1.

## Tests

```bash
pytest tests/
```

CUDA-only tests are automatically skipped when the corresponding extension is not built.

## Citation

See `NOTICE` for full upstream attributions and BibTeX entries for:

- Dao, Gu, Eichhorn, Rudra, Ré, *Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations*, ICML 2019
- Dao et al., *Kaleidoscope*, ICLR 2020
- Thomas, Gu, Dao, Rudra, Ré, *Learning Compressed Transforms with Low Displacement Rank*, NeurIPS 2018
- Dao et al., *Monarch: Expressive Structured Matrices for Efficient and Accurate Training*, ICML 2022
- Fu, Arora, Grogan et al., *Monarch Mixer: A Simple Sub-Quadratic GEMM-Based Architecture*, NeurIPS 2023

## License

Apache-2.0 (see `LICENSE`).
