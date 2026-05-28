# Thin build shim -- all metadata lives in pyproject.toml.
# This file exists only because torch.utils.cpp_extension.BuildExtension
# is a setuptools build_ext subclass that cannot be declared in TOML.
#
# Adapted from https://github.com/pytorch/extension-cpp
#
# DEFAULT BUILD: compiles NOTHING. The runtime backend is Triton (JIT) plus a
# pure-PyTorch fallback, so a stock `uv build` / `pip install .` produces a
# pure-Python `py3-none-any` wheel and never invokes nvcc or the C++ compiler.
# The legacy CUDA C++ extensions are opt-in via FORCE_CUDA=1 (see get_extensions).

import os
from pathlib import Path

from setuptools import setup


def _with_cuda():
    # Reached only on the FORCE_CUDA=1 opt-in path, so torch is guaranteed to be
    # importable here (the caller must provide it in the build environment).
    import torch
    from torch.utils.cpp_extension import CUDA_HOME

    with_cuda = torch.cuda.is_available() and CUDA_HOME is not None
    if os.getenv("FORCE_CUDA", "0") == "1":
        with_cuda = True
    if os.getenv("FORCE_CPU", "0") == "1":
        with_cuda = False
    return with_cuda


def _base_compile_args(with_cuda):
    extra = {"cxx": ["-O3"]}
    if with_cuda:
        nvcc_flags = os.getenv("NVCC_FLAGS", "").split() if os.getenv("NVCC_FLAGS") else []
        nvcc_flags += ["--expt-extended-lambda", "-lineinfo"]
        extra["nvcc"] = nvcc_flags
        if not os.getenv("TORCH_CUDA_ARCH_LIST"):
            os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0 8.0 9.0+PTX"
    return extra


def get_torch_ops_extensions(with_cuda):
    """Core torch.ops-style extensions (butterfly factor ops + CUDA version probe).

    Auto-discovered from csrc/*.cpp; each .cpp may optionally have matching
    csrc/cpu/<name>_cpu.cpp and csrc/cuda/<name>_cuda.cu files.
    """
    from torch.utils.cpp_extension import CppExtension, CUDAExtension

    Extension = CUDAExtension if with_cuda else CppExtension
    define_macros = [("WITH_CUDA", None)] if with_cuda else []
    extra_compile_args = _base_compile_args(with_cuda)

    extensions_dir = Path("csrc")
    extensions = []
    for main in extensions_dir.glob("*.cpp"):
        name = main.stem
        sources = [str(main)]
        cpu_path = extensions_dir / "cpu" / f"{name}_cpu.cpp"
        if cpu_path.exists():
            sources.append(str(cpu_path))
        cuda_path = extensions_dir / "cuda" / f"{name}_cuda.cu"
        if with_cuda and cuda_path.exists():
            sources.append(str(cuda_path))
        extensions.append(
            Extension(
                f"torch_structured._{name}",
                sources,
                include_dirs=[str(extensions_dir)],
                define_macros=define_macros,
                extra_compile_args=extra_compile_args,
            )
        )
    return extensions


def get_pybind_extensions(with_cuda):
    """pybind11-style CUDA extensions ported from structured-nets and m2.

    Loaded via `from torch_structured import _hadamard_cuda`, etc. (not
    through torch.ops). CUDA-only.

    - torch_structured._hadamard_cuda  (from structured-nets hadamard_cuda)
    - torch_structured._diag_mult_cuda (from structured-nets diag_mult_cuda)
    """
    if not with_cuda:
        return []

    from torch.utils.cpp_extension import CUDAExtension

    extra_compile_args = _base_compile_args(with_cuda)
    extensions = []

    hadamard_dir = Path("csrc/hadamard")
    if hadamard_dir.exists():
        extensions.append(
            CUDAExtension(
                name="torch_structured._hadamard_cuda",
                sources=[
                    str(hadamard_dir / "hadamard_cuda.cpp"),
                    str(hadamard_dir / "hadamard_cuda_kernel.cu"),
                ],
                extra_compile_args=extra_compile_args,
            )
        )

    diag_mult_dir = Path("csrc/diag_mult")
    if diag_mult_dir.exists():
        extensions.append(
            CUDAExtension(
                name="torch_structured._diag_mult_cuda",
                sources=[
                    str(diag_mult_dir / "diag_mult_cuda.cpp"),
                    str(diag_mult_dir / "diag_mult_cuda_kernel.cu"),
                ],
                extra_compile_args=extra_compile_args,
            )
        )

    return extensions


def get_extensions():
    if os.getenv("BUILD_DOCS", "0") == "1":
        return []
    # DEFAULT: compile nothing. The legacy CUDA C++ extensions only build when
    # the user explicitly opts in with FORCE_CUDA=1 (torch + ninja + a compiler
    # must already be present in the build environment in that case). This keeps
    # the default wheel pure-Python (py3-none-any) and avoids touching nvcc.
    if os.getenv("FORCE_CUDA", "0") != "1":
        return []
    with_cuda = _with_cuda()
    return [
        *get_torch_ops_extensions(with_cuda),
        *get_pybind_extensions(with_cuda),
    ]


def _build_kwargs():
    ext_modules = get_extensions()
    if not ext_modules:
        # Pure-Python default: no build_ext customization, no torch import.
        return {}
    # Opt-in compiled build: torch is required and BuildExtension drives ninja.
    from torch.utils.cpp_extension import BuildExtension

    return {
        "ext_modules": ext_modules,
        "cmdclass": {"build_ext": BuildExtension.with_options(use_ninja=True)},
    }


setup(**_build_kwargs())
