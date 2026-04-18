import torch
from ._compat import complex_mult


def circulant_multiply(c, x):
    """ Multiply circulant matrix with first column c by x
    Parameters:
        c: (n, )
        x: (batch_size, n) or (n, )
    Return:
        prod: (batch_size, n) or (n, )
    """
    return torch.irfft(complex_mult(torch.rfft(c, 1), torch.rfft(x, 1)), 1, signal_sizes=(c.shape[-1], ))
