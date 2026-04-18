"""Test for torch_structured.monarch.butterfly_factor."""

import torch

from torch_structured.monarch.butterfly_factor import butterfly_factor_to_matrix


def test_butterfly_factor_to_matrix_shape():
    b = 2
    log_b_n = 3
    n = b ** log_b_n
    twiddle = torch.arange(1, n * b + 1, dtype=torch.float).reshape(n // b, b, b)
    for factor_index in range(log_b_n):
        m = butterfly_factor_to_matrix(twiddle, factor_index)
        assert m.shape == (n, n)


def test_butterfly_factor_base_3():
    b = 3
    log_b_n = 2
    n = b ** log_b_n
    twiddle = torch.arange(1, n * b + 1, dtype=torch.float).reshape(n // b, b, b)
    m = butterfly_factor_to_matrix(twiddle, 0)
    assert m.shape == (n, n)
