"""Tests for torch_butterfly.structured.hadamard."""

import numpy as np
import pytest
import torch
from scipy.linalg import hadamard

from torch_butterfly.structured.hadamard import hadamard_transform_torch


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("m", [2, 4, 6])
def test_hadamard_transform_torch_matches_scipy(m):
    n = 1 << m
    x = torch.randn(3, n, device=_device())
    H = torch.tensor(hadamard(n), dtype=x.dtype, device=x.device)
    expected = x @ H.t()
    got = hadamard_transform_torch(x)
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-4)


def test_hadamard_transform_torch_normalize():
    n = 16
    x = torch.randn(2, n, device=_device())
    unnormed = hadamard_transform_torch(x)
    normed = hadamard_transform_torch(x, normalize=True)
    torch.testing.assert_close(normed, unnormed / np.sqrt(n))


def test_hadamard_transform_cuda_if_available():
    cuda_ext = pytest.importorskip("torch_butterfly._hadamard_cuda")
    assert hasattr(cuda_ext, "hadamard_transform")
    n = 32
    x = torch.randn(4, n, device="cuda")
    out = cuda_ext.hadamard_transform(x)
    expected = hadamard_transform_torch(x)
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)
