"""Tests for torch_structured.monarch.blockdiag_butterfly_multiply."""

import torch
import torch.nn.functional as F

from torch_structured.monarch.blockdiag_butterfly_multiply import (
    blockdiag_butterfly_multiply,
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


def test_fast_matches_true_dense_ground_truth():
    """The fast `BlockdiagButterflyMultiply` autograd Function has no dedicated
    correctness test anywhere (only the slow reference impl is tested, and only
    shape-checked for non-square cases). This builds the TRUE dense matrix via
    explicit torch.block_diag + explicit permutation matrices (not the
    implementation's own einops.rearrange), so the ground truth is independent
    of the code under test, and checks the fast op against it directly for a
    non-square, non-power-of-2 shape.
    """
    torch.manual_seed(0)
    k, q, p = 4, 5, 6
    l, s, r = 10, 7, 2  # l * r == k * q == 20
    n, m, batch = k * p, s * l, 3
    w1 = torch.randn(k, q, p, dtype=torch.float64)
    w2 = torch.randn(l, s, r, dtype=torch.float64)
    x = torch.randn(batch, n, dtype=torch.float64)

    W1_dense = torch.block_diag(*torch.unbind(w1, dim=0))  # (k*q, n)
    W2_dense = torch.block_diag(*torch.unbind(w2, dim=0))  # (l*s, l*r)

    # Explicit permutation matrices for the two intermediate reshapes
    # ('b (k q) -> b (r l)' then 'b (l s) -> b (s l)'), built independently of
    # any rearrange/einsum machinery in the code under test.
    kq = k * q
    P = torch.zeros(kq, kq, dtype=torch.float64)
    for i in range(kq):
        r_idx, l_idx = divmod(i, l)
        P[l_idx * r + r_idx, i] = 1.0

    ls = l * s
    P2 = torch.zeros(ls, ls, dtype=torch.float64)
    for i in range(ls):
        l_idx, s_idx = divmod(i, s)
        P2[s_idx * l + l_idx, i] = 1.0

    dense_full = P2 @ W2_dense @ P @ W1_dense  # (m, n) -- the true dense matrix
    out_dense = F.linear(x, dense_full)
    out_fast = blockdiag_butterfly_multiply(x.float(), w1.float(), w2.float()).double()
    torch.testing.assert_close(out_dense, out_fast, atol=1e-4, rtol=1e-4)
