"""Tests for torch_structured.monarch.blockdiag_butterfly_multiply."""

import torch

from torch_structured.monarch.blockdiag_butterfly_multiply import (
    blockdiag_butterfly_multiply_reference,
)


def test_reference_versions_agree():
    # Pick sizes so that all 3 reference versions are valid
    # (k = q = p = l = s = r = sqrt(n) is required by version 1).
    batch = 2
    root = 4
    n = root * root
    x = torch.randn(batch, n)
    w1_bfly = torch.randn(root, root, root)
    w2_bfly = torch.randn(root, root, root)
    out1 = blockdiag_butterfly_multiply_reference(x, w1_bfly, w2_bfly, version=1)
    out2 = blockdiag_butterfly_multiply_reference(x, w1_bfly, w2_bfly, version=2)
    out3 = blockdiag_butterfly_multiply_reference(x, w1_bfly, w2_bfly, version=3)
    torch.testing.assert_close(out1, out2, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out2, out3, atol=1e-4, rtol=1e-4)


def test_version_2_non_square():
    batch = 3
    k, q, p = 4, 5, 6
    l, s, r = 10, 7, 2  # l * r == k * q = 20
    x = torch.randn(batch, k * p)
    w1_bfly = torch.randn(k, q, p)
    w2_bfly = torch.randn(l, s, r)
    out = blockdiag_butterfly_multiply_reference(x, w1_bfly, w2_bfly, version=2)
    assert out.shape == (batch, s * l)
