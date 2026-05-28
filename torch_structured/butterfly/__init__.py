"""Butterfly factors and exact fast transforms.

Moved from the old top-level ``torch_structured`` package. Loading this
subpackage triggers compilation-time checks on the two torch.ops-style
extensions (``_butterfly`` and ``_version``) that live in the parent
``torch_structured`` package directory.
"""

import os
import glob
import warnings

import torch

# The compiled .so files live in the *parent* torch_structured/ package
# directory (because setup.py builds them as torch_structured._butterfly and
# torch_structured._version), so we search there rather than in this
# subpackage directory.
_ext_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_extension(name):
    """Load a compiled C++ extension library by name prefix.

    Returns True if the extension was found and loaded, False otherwise.

    A missing extension is the NORMAL state for a pure-Python (py3-none-any)
    wheel: the compiled CUDA C++ backend is optional and only built via
    FORCE_CUDA=1. When it is absent we emit a single low-noise UserWarning and
    fall back to the Triton / pure-PyTorch backend (resolved in _ops.py, whose
    _has_cuda_legacy() probe returns False when the op is not registered). We do
    NOT raise here, so that `import torch_structured` keeps working without a .so.
    """
    for suffix in ['*.so', '*.pyd', '*.dylib']:
        pattern = os.path.join(_ext_dir, f'{name}{suffix}')
        matches = glob.glob(pattern)
        if matches:
            torch.ops.load_library(matches[0])
            return True
    return False


# Load order matters: _version must load before _butterfly so that
# check_cuda_version() can call torch.ops.torch_structured.cuda_version().
_version_loaded = _load_extension('_version')
_butterfly_loaded = _load_extension('_butterfly')

if not (_version_loaded and _butterfly_loaded):
    warnings.warn(
        "torch_structured: compiled CUDA C++ extension(s) not found in "
        f"{_ext_dir}; falling back to the Triton / pure-PyTorch backend. "
        "This is expected for the default pure-Python wheel. To build the "
        "legacy CUDA backend install from source with "
        "`FORCE_CUDA=1 pip install . --no-build-isolation`.",
        UserWarning,
        stacklevel=2,
    )


def check_cuda_version():
    if torch.version.cuda is not None:
        cuda_version = torch.ops.torch_structured.cuda_version()

        if cuda_version == -1:
            major = minor = 0
        elif cuda_version < 10000:
            major, minor = int(str(cuda_version)[0]), int(str(cuda_version)[2])
        else:
            major, minor = int(str(cuda_version)[0:2]), int(str(cuda_version)[3])
        t_major, t_minor = [int(x) for x in torch.version.cuda.split('.')]

        if t_major != major or t_minor != minor:
            warnings.warn(
                f'Detected that PyTorch and torch_structured were compiled '
                f'with different CUDA versions. PyTorch has CUDA version '
                f'{t_major}.{t_minor} and torch_structured has CUDA version '
                f'{major}.{minor}. Please reinstall the torch_structured that '
                f'matches your PyTorch install.',
                UserWarning,
                stacklevel=2,
            )


# check_cuda_version() calls torch.ops.torch_structured.cuda_version(), which is
# only registered when the _version extension loaded. Skip it on the pure-Python
# path where no .so is present.
if _version_loaded:
    check_cuda_version()

from .butterfly import Butterfly, ButterflyUnitary, ButterflyBmm  # noqa: E402
from .butterfly_base4 import ButterflyBase4  # noqa: E402
from .multiply import butterfly_multiply  # noqa: E402
from . import combine  # noqa: F401, E402
from . import complex_utils  # noqa: F401, E402
from . import diagonal  # noqa: F401, E402
from . import permutation  # noqa: F401, E402
from . import special  # noqa: F401, E402
from . import multiply_base4  # noqa: F401, E402

__all__ = [
    'Butterfly',
    'ButterflyUnitary',
    'ButterflyBmm',
    'ButterflyBase4',
    'butterfly_multiply',
]
