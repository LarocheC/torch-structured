"""torch-structured: butterfly, LDR, and Monarch structured-matrix primitives.

Umbrella namespace that re-exports the most commonly used symbols from the
three subpackages:

- ``torch_structured.butterfly`` - butterfly factors and exact transforms
  (FFT, DCT, DST, Hadamard, circulant, Toeplitz).
- ``torch_structured.structured`` - low-displacement-rank layers
  (Toeplitz-like, Hankel, Vandermonde, Fastfood, Circulant, LDR).
- ``torch_structured.monarch`` - Monarch / block-diagonal-butterfly primitives.
- ``torch_structured.recurrent`` - recurrent layers (LRU) that plug into
  the ``make_linear`` factory.

Importing compiled C++/CUDA extensions (``_butterfly``, ``_version``) happens
inside the ``butterfly`` subpackage on first import; the other subpackages
lazily try-import their own CUDA modules (``_hadamard_cuda`` and
``_diag_mult_cuda``) when needed.
"""

__version__ = '1.2.4'

# The butterfly subpackage is imported eagerly because the top-level re-exports
# its most common classes. The other subpackages are left for the user to
# import explicitly to keep startup lightweight (they pull in scipy / einops /
# opt_einsum).
from .butterfly import (
    Butterfly,
    ButterflyBmm,
    ButterflyBase4,
    ButterflyUnitary,
    butterfly_multiply,
)
from .factory import make_linear
from .recurrent import LRU
from ._ops import set_backend, set_deterministic  # noqa: F401

__all__ = [
    'Butterfly',
    'ButterflyBmm',
    'ButterflyBase4',
    'ButterflyUnitary',
    'butterfly_multiply',
    'LRU',
    'make_linear',
    'set_backend',
    'set_deterministic',
    '__version__',
]
