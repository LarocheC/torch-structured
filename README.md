# torch-structured

Consolidated PyTorch library of structured-matrix primitives:

- **`torch_structured`** (core) — butterfly matrices for exact fast linear transforms (FFT, iFFT, DCT, DST, Hadamard, circulant, Toeplitz) as learnable `nn.Module` drop-in replacements for `nn.Linear`.
- **`torch_structured.structured`** — low-displacement-rank layers ported from [structured-nets](https://github.com/HazyResearch/structured-nets): Toeplitz-like, Hankel, Vandermonde, Fastfood, Circulant, LDR subdiagonal / tridiagonal, Krylov utilities.
- **`torch_structured.monarch`** — Monarch / block-diagonal-butterfly primitives ported from [m2](https://github.com/HazyResearch/m2): block-diagonal and block-diagonal-butterfly multiplies, structured linear layers, butterfly-factor helper, Hyena implicit long filter, and an opt-in fused flashmm CUDA kernel.

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

### Optional: flashmm extension

The Monarch Mixer fused `flashmm` kernel is opt-in because it requires NVIDIA MathDx 22.02 and extra kernel sources not vendored in this repo. See [`csrc/flashmm/README.md`](csrc/flashmm/README.md) for the full procedure:

```bash
python csrc/flashmm/fetch_kernel_sources.py
TORCH_STRUCTURED_BUILD_FLASHMM=1 FORCE_CUDA=1 uv pip install -e .
```

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

## Tests

```bash
pytest tests/
```

CUDA-only and `_flashmm`-only tests are automatically skipped when the corresponding extension is not built.

## Citation

See `NOTICE` for full upstream attributions and BibTeX entries for:

- Dao, Gu, Eichhorn, Rudra, Ré, *Learning Fast Algorithms for Linear Transforms Using Butterfly Factorizations*, ICML 2019
- Dao et al., *Kaleidoscope*, ICLR 2020
- Thomas, Gu, Dao, Rudra, Ré, *Learning Compressed Transforms with Low Displacement Rank*, NeurIPS 2018
- Dao et al., *Monarch: Expressive Structured Matrices for Efficient and Accurate Training*, ICML 2022
- Fu, Arora, Grogan et al., *Monarch Mixer: A Simple Sub-Quadratic GEMM-Based Architecture*, NeurIPS 2023

## License

Apache-2.0 (see `LICENSE`).
