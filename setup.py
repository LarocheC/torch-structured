# Thin build shim -- all metadata lives in pyproject.toml.
# This file exists only because torch.utils.cpp_extension.BuildExtension
# is a setuptools build_ext subclass that cannot be declared in TOML.
#
# Adapted from https://github.com/pytorch/extension-cpp

import os
from pathlib import Path

from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME


def _with_cuda():
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
    - torch_structured._flashmm        (from m2 csrc/flashmm; opt-in via env var)
    """
    if not with_cuda:
        return []

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

    # flashmm is opt-in: requires MathDx 22.02 headers + two large kernel
    # sources that are not vendored in-tree (see csrc/flashmm/README.md).
    if os.getenv("TORCH_STRUCTURED_BUILD_FLASHMM", "0") == "1":
        flashmm_dir = Path("csrc/flashmm")
        mathdx_include = flashmm_dir / "mathdx" / "22.02" / "include"
        mm_block_src = flashmm_dir / "mm_block_fwd_cuda.cu"
        lut_header = flashmm_dir / "lut.h"

        missing = [p for p in (mm_block_src, lut_header) if not p.exists()]
        if missing:
            names = ", ".join(str(p) for p in missing)
            raise FileNotFoundError(
                f"TORCH_STRUCTURED_BUILD_FLASHMM=1 but the following kernel "
                f"sources are missing: {names}. Run "
                f"`python csrc/flashmm/fetch_kernel_sources.py` first."
            )

        flashmm_compile_args = dict(extra_compile_args)
        flashmm_compile_args["cxx"] = flashmm_compile_args.get("cxx", []) + ["-std=c++17"]
        flashmm_compile_args["nvcc"] = flashmm_compile_args.get("nvcc", []) + [
            "-std=c++17",
            "-arch=compute_80",
            "-gencode=arch=compute_80,code=sm_80",
        ]
        extensions.append(
            CUDAExtension(
                name="torch_structured._flashmm",
                sources=[
                    str(flashmm_dir / "flash_mm.cpp"),
                    str(mm_block_src),
                    str(flashmm_dir / "hyena_filter_cuda.cu"),
                ],
                include_dirs=[
                    str(flashmm_dir),
                    str(mathdx_include),
                ],
                extra_compile_args=flashmm_compile_args,
            )
        )

    return extensions


def get_extensions():
    if os.getenv("BUILD_DOCS", "0") == "1":
        return []
    with_cuda = _with_cuda()
    return [
        *get_torch_ops_extensions(with_cuda),
        *get_pybind_extensions(with_cuda),
    ]


setup(
    ext_modules=get_extensions(),
    cmdclass={
        "build_ext": BuildExtension.with_options(use_ninja=True)
    },
)
