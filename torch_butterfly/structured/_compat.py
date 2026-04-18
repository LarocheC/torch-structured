"""Internal compatibility shims for the structured-nets port.

Bundles three things the legacy LDR code depends on:

1. ``torch.rfft`` / ``torch.irfft`` shims - the old signal_ndim API that
   PyTorch >= 1.7 removed (originally ``pytorch/structure/compat.py``).

2. ``(..., 2)``-pair complex helpers ``complex_mult`` / ``conjugate`` - a
   float-pair representation from before ``torch.complex64`` existed
   (originally ``pytorch/structure/complex_utils.py``). The *top-level*
   ``torch_butterfly.complex_utils`` uses native complex tensors and is
   the canonical API - do not re-export these from here.

3. ``krylov_construct`` - small numpy helper used by ``krylov.py`` slow
   paths (originally ``pytorch/structure/scratch/krylovslow.py``; inlined
   here so we can drop the rest of the research ``scratch/`` tree).
"""

import numpy as np
import torch


# --- old torch.rfft / torch.irfft shim --------------------------------------

def _rfft(input, signal_ndim, normalized=False, onesided=True):
    assert signal_ndim == 1, "Only 1D supported"
    result = torch.fft.rfft(input, dim=-1, norm="ortho" if normalized else None)
    return torch.stack((result.real, result.imag), dim=-1)


def _irfft(input, signal_ndim, normalized=False, onesided=True, signal_sizes=None):
    assert signal_ndim == 1, "Only 1D supported"
    complex_input = torch.complex(input[..., 0], input[..., 1])
    n = signal_sizes[-1] if signal_sizes else None
    return torch.fft.irfft(complex_input, n=n, dim=-1, norm="ortho" if normalized else None)


def patch():
    if not hasattr(torch, 'rfft'):
        torch.rfft = _rfft
    if not hasattr(torch, 'irfft'):
        torch.irfft = _irfft


patch()


# --- legacy real-paired complex helpers -------------------------------------

def conjugate(X):
    assert X.shape[-1] == 2, 'Last dimension must be 2'
    return X * torch.tensor((1, -1), dtype=X.dtype, device=X.device)


def complex_mult(X, Y):
    assert X.shape[-1] == 2 and Y.shape[-1] == 2, 'Last dimension must be 2'
    return torch.stack(
        (X[..., 0] * Y[..., 0] - X[..., 1] * Y[..., 1],
         X[..., 0] * Y[..., 1] + X[..., 1] * Y[..., 0]),
        dim=-1,
    )


# --- krylov_construct (used by krylov.py slow paths) ------------------------

def krylov_construct(A, v, m):
    n = v.shape[0]
    assert A.shape == (n, n)
    d = np.diagonal(A, 0)
    subd = np.diagonal(A, -1)

    K = np.zeros(shape=(m, n))
    K[0, :] = v
    for i in range(1, m):
        K[i, 1:] = subd * K[i - 1, :-1]
    return K
