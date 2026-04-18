<!-- GSD:project-start source:PROJECT.md -->
## Project

**torch_structured — Build System Modernization**

Modernize the packaging and build system for `torch_structured`, a PyTorch library implementing butterfly matrices for efficient structured linear transforms (FFT, DCT, Hadamard, circulant, etc.). The library has C++/CUDA extensions and currently uses a legacy `setup.py`-based build. The goal is to make it installable via `uv` with a modern `pyproject.toml`, targeting Python 3.10+ and PyTorch 2.x.

**Core Value:** A single `uv pip install .` (or `uv pip install -e .`) that just works — with CUDA support when available, without conda or manual steps.

### Constraints

- **Build system:** Must use pyproject.toml as the single source of truth for packaging
- **UV compatibility:** Must work with `uv pip install` without conda
- **CUDA support:** Must retain CUDA extension compilation via torch.utils.cpp_extension
- **Python:** >=3.10, <4
- **PyTorch:** >=2.0
- **Backwards compat:** No need to support Python <3.10 or PyTorch <2.0
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Python >=3.6 - All training scripts, model definitions, library code
- C++ (C++14) - Custom PyTorch operator implementations in `csrc/`
- CUDA C++ - GPU kernel implementations in `csrc/cuda/`
- Cython - Legacy matrix multiplication extension in `learning_transforms/ABCD_mult.pyx`
## Runtime
- Python >=3.6 (Dockerfile uses 3.8)
- NVIDIA CUDA 10.1 (per Dockerfile base image `nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04`)
- cuDNN 7
- pip (primary)
- conda (used in Docker and development environment setup)
- Lockfile: not present
## Frameworks
- PyTorch >=1.8 - Deep learning framework, the foundation for the entire project
- PyTorch Lightning >=1.0.3 - Training loop abstraction used in `convolution/` experiments
- Hydra >=1.0.0 - Configuration management for `convolution/` training scripts
- OmegaConf - Config resolution (used via Hydra)
- Ray >=0.6.5 (pinned to 1.0.1 in Docker) - Distributed hyperparameter tuning and training
- Ray Tune - Hyperparameter search with `AsyncHyperBandScheduler`
- setuptools - Package building via `setup.py`
- torch.utils.cpp_extension (CppExtension, CUDAExtension) - Compiles C++/CUDA extensions
- ninja - Build system used by PyTorch's BuildExtension
- Cython - Builds `learning_transforms/ABCD_mult.pyx`
## Key Dependencies
- `torch` (>=1.8) - Core tensor operations, autograd, nn.Module base, JIT scripting, custom C++ operator dispatch
- `numpy` - Numerical operations throughout the codebase
- `scipy` - Used in `gumbel-sinkhorn/my_sinkhorn_ops.py` for `linear_sum_assignment`
- `pytorch-lightning` (>=1.0.3) - Training loop, callbacks, logging in `convolution/`
- `pytorch-lightning-bolts` (>=0.2.5) - Additional utilities for Lightning
- `ray[tune]` (>=1.0.0) - Distributed experiment execution in `cnn/` and `convolution/`
- `hydra-core` (>=1.0.0) - YAML-based configuration in `convolution/`
- `munch` - Dict-to-object utility in `convolution/ray_runner.py`
- `wandb` (==0.9.7 in Docker) - Weights & Biases experiment logging (referenced in `convolution/cfg/config.yaml` project: `butterflynas`)
- `scikit-learn` - Referenced in Docker install, utility functions
- `torchvision` - Image datasets and transforms for CNN experiments
- `matplotlib` (3.0.2) - Visualization in `gumbel-sinkhorn/`
## Configuration
- `FORCE_CUDA=1` / `FORCE_CPU=1` - Override CUDA detection in `setup.py`
- `NVCC_FLAGS` - Custom NVCC compiler flags in `setup.py`
- `BUILD_DOCS=1` - Skip building C++ extensions in `setup.py`
- `PYTHONPATH` - Must include project root and `fairseq/` for Ray workers
- `setup.py` - Main package build (torch_structured with C++/CUDA extensions)
- `butterfly/factor_multiply/setup.py` - Legacy CUDA extension build
- `butterfly/factor_multiply_fast/setup.py` - Legacy fast CUDA extension build
- `learning_transforms/setup.py` - Cython extension build
- `convolution/cfg/config.yaml` - Hydra root config for convolution experiments
- `convolution/cfg/` - Full Hydra config tree (model, optimizer, dataset, runner, lr_scheduler)
## C++/CUDA Extensions
- Source: `csrc/butterfly.cpp` (dispatch), `csrc/cpu/butterfly_cpu.cpp`, `csrc/cuda/butterfly_cuda.cu`
- Registers custom ops via `TORCH_LIBRARY(torch_structured, m)`:
- CUDA architecture: sm_35 minimum
- Loaded at runtime via `torch.ops.load_library()` in `torch_structured/__init__.py`
- `butterfly/factor_multiply/` - Factor multiply CUDA extension (sm_70 for V100)
- `butterfly/factor_multiply_fast/` - Fast butterfly multiply CUDA extension (sm_30 minimum)
## Platform Requirements
- Linux (Ubuntu 18.04 in Docker)
- NVIDIA GPU with CUDA support (optional, CPU fallback exists)
- CUDA toolkit 10.1+ with cuDNN
- C++ compiler supporting C++14
- conda or pip
- Same as development; this is a research library, not a deployed service
- GPU strongly recommended for practical performance
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- Use `snake_case.py` for all Python modules: `butterfly.py`, `complex_utils.py`, `multiply_base4.py`
- Test files follow `test_{module}.py` pattern: `test_butterfly.py`, `test_multiply.py`
- Benchmark files use `benchmark_` prefix or `speed_test.py` style: `benchmark_utils.py`, `benchmark_linear.py`
- C++ source files use `snake_case.cpp` / `snake_case.cu`: `butterfly.cpp`, `butterfly_cpu.cpp`, `butterfly_cuda.cu`
- Use `PascalCase` for all classes: `Butterfly`, `ButterflyUnitary`, `ButterflyBmm`, `ButterflyBase4`
- nn.Module subclasses use descriptive names: `FixedPermutation`, `Diagonal`, `TensorProduct`, `Real2Complex`, `Complex2Real`
- Test classes use `{Feature}Test` suffix: `ButterflyTest`, `ButterflyPermutationTest`, `MultiplyBase4Test`
- Custom autograd functions use `PascalCase`: `ComplexMatmul`, `IndexLastDim`, `Real2ComplexFn`
- Use `snake_case` for all functions: `butterfly_multiply`, `bitreversal_permutation`, `diagonal_butterfly`
- Factory functions that return nn.Module use lowercase: `fft()`, `ifft()`, `hadamard()`, `circulant()`, `toeplitz()`
- Internal/torch JIT functions use `_fw` / `_bw` suffixes: `butterfly_multiply_fw`, `butterfly_multiply_bw`
- Pure-Python reference implementations use `_torch` suffix: `butterfly_multiply_torch`, `butterfly_multiply_base4_torch`
- Conversion utilities use `x2y` or `x_to_y` style: `twiddle_base2_to_base4`, `perm2butterfly`, `real2complex`
- Short mathematical names preferred for tensor variables: `n`, `log_n`, `twiddle`, `phi`, `alpha`, `psi`, `chi`
- Prefix `d_` for gradient tensors: `d_twiddle`, `d_input`
- Dimension sizes use descriptive short names: `batch_size`, `nstacks`, `nblocks`, `in_size`, `out_size`
- Booleans use descriptive names: `increasing_stride`, `br_first`, `diag_first`, `normalized`, `separate_diagonal`
- `complex` (bool) for complex/real flag (shadows Python builtin, but used consistently)
- `dtype` derived from `complex` flag: `torch.float32 if not complex else torch.complex64`
## Code Style
- No automated formatter configured (no `.flake8`, `pyproject.toml`, `.prettierrc`, or similar)
- Line length varies (some lines exceed 100 characters, particularly in test files)
- 4-space indentation (Python standard)
- No linter configured
- No type checking (no `mypy.ini` or `pyproject.toml` with mypy config)
- Standard library first, blank line, then third-party, blank line, then local
- Example from `torch_structured/butterfly.py`:
- `import torch` (always top-level)
- `from torch import nn` (preferred over `import torch.nn as nn`)
- `import torch.nn.functional as F` (standard alias)
- `from torch.nn import functional as F` (also used, inconsistent)
- Local package imports use explicit relative or absolute: `from torch_structured.multiply import butterfly_multiply`
- `# noqa` comments used on re-exports in `__init__.py`
## nn.Module Patterns
- Always call `super().__init__()` first
- Store all config as instance attributes: `self.in_size`, `self.log_n`, `self.n`, etc.
- Use `nn.Parameter()` for learnable tensors
- Use `self.register_parameter('bias', None)` for optional parameters when disabled
- Use `self.register_buffer()` for non-learnable persistent state (e.g., permutations)
- Custom flags on parameters: `self.twiddle._is_structured = True  # Flag to avoid weight decay`
- Separate `reset_parameters()` method for initialization logic
- Always include docstring with Parameters/Return sections
- Input/output shape documented in docstring: `input: (batch, *, in_size)`, `output: (batch, *, out_size)`
- Return formatted string of constructor args:
## Custom Autograd Functions
- Use `ctx.save_for_backward()` for tensors needed in backward
- Check `ctx.needs_input_grad[i]` before computing gradients
- Provide a module-level function wrapping the `.apply` call: `complex_matmul = ComplexMatmul.apply` or `real2complex = Real2ComplexFn.apply`
## Error Handling
- Use `assert` statements for preconditions (not exceptions). This is used pervasively:
- C++ extensions use `TORCH_CHECK` macros for runtime validation:
#define CHECK_DEVICE(x) TORCH_CHECK(x.device().type() == torch::kCPU || x.device().type() == torch::kCUDA, ...)
#define CHECK_DIM(x, y) TORCH_CHECK(x.dim() == y, ...)
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), ...)
- `RuntimeError` raised for CUDA version mismatch in `torch_structured/__init__.py`
- No try/except blocks in the core library code
## Logging
- Commented-out `print` statements appear in test files for debugging (see `tests/test_multiply.py` lines 48-54)
- No production logging
## Comments
- Inline comments explain mathematical reasoning: `# Sampling from the Haar measure on U(2) is a bit subtle.`
- Reference external papers/resources: `# Using the parameterization here: http://home.lu.lv/~sd20008/papers/essays/...`
- Warn about tricky code: `# Warning: All this dimension manipulation (transpose and unsqueeze) is super tricky.`
- Explain workarounds: `# Pytorch 1.7 doesn't support complex reshape backward for non-contiguous tensors`
- Commented-out alternative implementations are common (kept for reference)
- Present on public classes and key functions
- Use plain text format (not Sphinx/NumPy/Google style)
- Parameter sections use `Parameters:` / `Parameter:` and `Return:` / `Returns:` headings
- Shape annotations included: `input: (batch, *, in_size)`
- Example from `torch_structured/combine.py`:
## Function Design
- Type annotations used sparingly: present on some function signatures (`combine.py`), absent on most
- Return type annotations used with `-> nn.Module` or `-> Butterfly` on factory/combination functions
- `torch.jit.script` decorators on performance-critical functions in `torch_structured/multiply.py`
- Factory functions return `nn.Module` or `nn.Sequential`
- Tensor operations return `torch.Tensor`
- In-place operations return `self` (e.g., `__imul__`)
## Module Design
- `torch_structured/__init__.py` defines `__all__` with public API: `Butterfly`, `ButterflyUnitary`, `ButterflyBmm`, `ButterflyBase4`, `butterfly_multiply`
- Submodules imported as namespace: `from . import combine`, `from . import special`
- No barrel files beyond `__init__.py`
- Loaded dynamically in `__init__.py` via `torch.ops.load_library`
- Registered as TorchScript custom ops: `torch.ops.torch_structured.butterfly_multiply_fw`
- CPU/CUDA dispatch handled in C++ with `#ifdef WITH_CUDA` guards
- Consistently used when modifying parameters outside of training:
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- PyTorch C++/CUDA extension pattern: native kernels registered via `torch.ops`, wrapped in Python `torch.jit.script` functions, consumed by `nn.Module` classes
- Two-generation codebase: a legacy `butterfly/` package (old interface) and the current `torch_structured/` package (new interface)
- Experiment directories (`cnn/`, `convolution/`, `transformer/`, `learning_transforms/`, `gumbel-sinkhorn/`) are standalone scripts that import from the core packages
- All transforms operate on power-of-2 sizes; inputs are zero-padded or trimmed internally
## Layers
- Purpose: High-performance butterfly multiply forward and backward passes
- Location: `csrc/`
- Contains: `csrc/butterfly.cpp` (dispatch + autograd), `csrc/cpu/butterfly_cpu.cpp` (CPU impl), `csrc/cuda/butterfly_cuda.cu` (CUDA impl)
- Depends on: PyTorch C++ API (`torch/script.h`, `torch/extension.h`)
- Used by: `torch_structured/multiply.py` via `torch.ops.torch_structured.*`
- Key detail: Registers three ops via `TORCH_LIBRARY(torch_structured, m)`: `butterfly_multiply_fw`, `butterfly_multiply_bw`, `butterfly_multiply` (which wraps autograd)
- Purpose: Python-accessible butterfly multiplication with both native and pure-PyTorch fallback
- Location: `torch_structured/multiply.py`, `torch_structured/multiply_base4.py`
- Contains: `butterfly_multiply()` (calls native op), `butterfly_multiply_torch()` (pure PyTorch reference), `butterfly_multiply_base4_torch()`, `twiddle_base2_to_base4()`
- Depends on: Native kernel layer (for `butterfly_multiply`), PyTorch (for `_torch` variants)
- Used by: `torch_structured/butterfly.py`, `torch_structured/butterfly_base4.py`
- Purpose: User-facing modules compatible with `torch.nn.Linear`
- Location: `torch_structured/butterfly.py`
- Contains: `Butterfly` (main class), `ButterflyUnitary` (unitary-constrained variant), `ButterflyBmm` (batched variant)
- Depends on: Core multiply layer, `torch_structured/complex_utils.py`
- Used by: `torch_structured/special.py`, `torch_structured/combine.py`, experiment code
- Purpose: Combine butterfly matrices with diagonals, permutations, and other butterflies
- Location: `torch_structured/combine.py`, `torch_structured/permutation.py`, `torch_structured/diagonal.py`
- Contains: `diagonal_butterfly()`, `butterfly_product()`, `butterfly_kronecker()`, `TensorProduct`, `FixedPermutation`, `Diagonal`, `perm2butterfly()`
- Depends on: nn.Module layer
- Used by: `torch_structured/special.py`
- Purpose: Factory functions that construct butterfly networks performing exact well-known transforms
- Location: `torch_structured/special.py`
- Contains: `fft()`, `ifft()`, `fft_unitary()`, `ifft_unitary()`, `dct()`, `dst()`, `circulant()`, `toeplitz()`, `hadamard()`, `hadamard_diagonal()`, `conv1d_circular_singlechannel()`, `conv1d_circular_multichannel()`
- Depends on: nn.Module layer, composition layer, permutation layer
- Used by: Tests, experiment code
- Purpose: Original butterfly implementation (being replaced)
- Location: `butterfly/`
- Contains: `butterfly/butterfly.py` (old Butterfly class), `butterfly/butterfly_multiply.py` (old multiply with both C++ and pure-PyTorch), `butterfly/permutation.py`, `butterfly/complex_utils.py`
- Depends on: `butterfly/factor_multiply/` and `butterfly/factor_multiply_fast/` (old C++ extensions, separate `setup.py` each)
- Used by: `learning_transforms/`, `cnn/` experiment code
## Data Flow
- Base-2: `(nstacks, nblocks, log_n, n//2, 2, 2)` - log_n stages of n//2 2x2 butterfly factors
- Base-4: `(nstacks, nblocks, log_n//2, n//4, 4, 4)` - fuses pairs of stages for efficiency
- Unitary: `(nstacks, nblocks, log_n, n//2, 4)` - parameterized by (phi, alpha, psi, chi) angles to enforce unitarity
- All learnable state is in `nn.Parameter` tensors: `twiddle` (or `twiddle4`/`twiddle2` for base-4), `bias`, `diagonal`
- `twiddle._is_structured = True` flag used to exclude from weight decay during training
- Permutations stored as `register_buffer` (not learnable)
## Key Abstractions
- Purpose: Product of log(N) butterfly factors, drop-in replacement for `nn.Linear`
- Examples: `torch_structured/butterfly.py` (lines 15-208)
- Pattern: Parameterized by a single twiddle tensor; forward pass calls into C++/CUDA; supports transpose, conjugate, subtwiddle
- Purpose: Butterfly constrained to be unitary via angle parameterization
- Examples: `torch_structured/butterfly.py` (lines 210-308)
- Pattern: Inherits from Butterfly but overrides twiddle to use 4 angle parameters per 2x2 block, constructs unitary matrix on the fly in forward()
- Purpose: Apply a fixed (non-learnable) permutation to the last dimension
- Examples: `torch_structured/permutation.py` (lines 55-78)
- Pattern: Stores permutation as buffer; uses custom autograd for complex backward compatibility
- Purpose: Element-wise multiplication by a learnable diagonal
- Examples: `torch_structured/diagonal.py`
- Pattern: Simple wrapper around `input * self.diagonal`
- Purpose: Fuse a diagonal matrix into a butterfly's twiddle factors (avoids separate diagonal module)
- Examples: `torch_structured/combine.py` (lines 11-53)
- Pattern: Modifies the first or last twiddle factor in-place to absorb the diagonal
## Entry Points
- Location: `setup.py`
- Triggers: `python setup.py install`
- Responsibilities: Compiles C++/CUDA extensions from `csrc/`, installs `torch_structured` package
- Location: `torch_structured/__init__.py`
- Triggers: `import torch_structured`
- Responsibilities: Loads compiled native libraries (`_version`, `_butterfly`), checks CUDA version compatibility, exports `Butterfly`, `ButterflyUnitary`, `ButterflyBmm`, `ButterflyBase4`, `butterfly_multiply`
- Location: `tests/test_butterfly.py`, `tests/test_special.py`, etc.
- Triggers: pytest
- Responsibilities: Verify correctness of butterfly multiply, special transforms, permutation decomposition
## Error Handling
- C++ layer uses `TORCH_CHECK` macros for shape/device validation in `csrc/butterfly.cpp`
- Python layer uses `assert` statements for shape, size, and parameter validation
- CUDA version mismatch raises `RuntimeError` at import time in `torch_structured/__init__.py`
- Power-of-2 size requirements enforced via assertion: `assert n == 1 << log_n, 'n must be a power of 2'`
## Cross-Cutting Concerns
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
