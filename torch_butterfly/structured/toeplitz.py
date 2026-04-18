'''Functions to multiply by a Toeplitz-like matrix.
'''
import numpy as np
import torch

from ._compat import complex_mult, conjugate
from .krylov import Krylov


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##### Fast multiplication for the Toeplitz-like case

def toeplitz_krylov_transpose_multiply(v, u, f=0.0):
    """Multiply Krylov(Z_f, v_i)^T @ u.
    Parameters:
        v: (rank, n)
        u: (batch_size, n)
        f: real number
    Returns:
        product: (batch, rank, n)
    """
    _, n = u.shape
    _, n_ = v.shape
    assert n == n_, 'u and v must have the same last dimension'
    if f != 0.0:  # cycle version
        # Computing the roots of f
        mod = abs(f) ** (torch.arange(n, dtype=u.dtype, device=u.device) / n)
        if f > 0:
            arg = torch.stack((torch.ones(n, dtype=u.dtype, device=u.device),
                               torch.zeros(n, dtype=u.dtype, device=u.device)), dim=-1)
        else:  # Find primitive roots of -1
            angles = torch.arange(n, dtype=u.dtype, device=u.device) / n * np.pi
            arg = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        eta = mod[:, np.newaxis] * arg
        eta_inverse = (1.0 / mod)[:, np.newaxis] * conjugate(arg)
        u_f = torch.ifft(eta_inverse * u[..., np.newaxis], 1)
        v_f = torch.fft(eta * v[..., np.newaxis], 1)
        uv_f = complex_mult(u_f[:, np.newaxis], v_f[np.newaxis])
        uv = torch.fft(uv_f, 1)
        # We only need the real part of complex_mult(eta, uv)
        return eta[..., 0] * uv[..., 0] - eta[..., 1] * uv[..., 1]
    else:
        u_f = torch.rfft(torch.cat((u.flip(1), torch.zeros_like(u)), dim=-1), 1)
        v_f = torch.rfft(torch.cat((v, torch.zeros_like(v)), dim=-1), 1)
        uv_f = complex_mult(u_f[:, np.newaxis], v_f[np.newaxis])
        return torch.irfft(uv_f, 1, signal_sizes=(2 * n, ))[..., :n].flip(2)


def toeplitz_krylov_multiply_by_autodiff(v, w, f=0.0):
    """Multiply \\sum_i Krylov(Z_f, v_i) @ w_i, using Pytorch's autodiff.
    This function is just to check the result of toeplitz_krylov_multiply.
    """
    batch_size, rank, n = w.shape
    rank_, n_ = v.shape
    assert n == n_, 'w and v must have the same last dimension'
    assert rank == rank_, 'w and v must have the same rank'

    u = torch.zeros((batch_size, n), dtype=v.dtype, device=v.device, requires_grad=True)
    prod = toeplitz_krylov_transpose_multiply(v, u, f)
    result, = torch.autograd.grad(prod, u, grad_outputs=w, create_graph=True)
    return result


def toeplitz_krylov_multiply(v, w, f=0.0):
    """Multiply \\sum_i Krylov(Z_f, v_i) @ w_i.
    """
    _, rank, n = w.shape
    rank_, n_ = v.shape
    assert n == n_, 'w and v must have the same last dimension'
    assert rank == rank_, 'w and v must have the same rank'
    if f != 0.0:  # cycle version
        mod = abs(f) ** (torch.arange(n, dtype=w.dtype, device=w.device) / n)
        if f > 0:
            arg = torch.stack((torch.ones(n, dtype=w.dtype, device=w.device),
                               torch.zeros(n, dtype=w.dtype, device=w.device)), dim=-1)
        else:
            angles = torch.arange(n, dtype=w.dtype, device=w.device) / n * np.pi
            arg = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        eta = mod[:, np.newaxis] * arg
        eta_inverse = (1.0 / mod)[:, np.newaxis] * conjugate(arg)
        w_f = torch.fft(eta * w[..., np.newaxis], 1)
        v_f = torch.fft(eta * v[..., np.newaxis], 1)
        wv_sum_f = complex_mult(w_f, v_f).sum(dim=1)
        wv_sum = torch.ifft(wv_sum_f, 1)
        return eta_inverse[..., 0] * wv_sum[..., 0] - eta_inverse[..., 1] - wv_sum[..., 1]
    else:
        w_f = torch.rfft(torch.cat((w, torch.zeros_like(w)), dim=-1), 1)
        v_f = torch.rfft(torch.cat((v, torch.zeros_like(v)), dim=-1), 1)
        wv_sum_f = complex_mult(w_f, v_f).sum(dim=1)
        return torch.irfft(wv_sum_f, 1, signal_sizes=(2 * n, ))[..., :n]


def toeplitz_mult(G, H, x, cycle=True):
    """Multiply \\sum_i Krylov(Z_f, G_i) @ Krylov(Z_f, H_i) @ x.
    Parameters:
        G: Tensor of shape (rank, n)
        H: Tensor of shape (rank, n)
        x: Tensor of shape (batch_size, n)
        cycle: whether to use f = (1, -1) or f = (0, 0)
    Returns:
        product: Tensor of shape (batch_size, n)
    """
    f = (1, -1) if cycle else (0, 0)
    transpose_out = toeplitz_krylov_transpose_multiply(H, x, f[1])
    return toeplitz_krylov_multiply(G, transpose_out, f[0])


##### Slow multiplication for the Toeplitz-like case

def toeplitz_Z_f_linear_map(f=0.0):
    """The linear map for multiplying by Z_f."""
    return lambda v: torch.cat((f * v[[-1]], v[:-1]))


def krylov_toeplitz_fast(v, f=0.0):
    """Explicit construction of Krylov matrix [v  A @ v  A^2 @ v  ...  A^{n-1} @ v]
    where A = Z_f.
    """
    rank, n = v.shape
    a = torch.arange(n, device=v.device)
    b = -a
    indices = a[:, np.newaxis] + b[np.newaxis]
    K = v[:, indices]
    K[:, indices < 0] *= f
    return K


def toeplitz_mult_slow(G, H, x, cycle=True):
    assert G.shape == H.shape, 'G and H must have the same shape'
    rank, n = G.shape
    f = (1, -1) if cycle else (0, 0)
    krylovs = [(Krylov(toeplitz_Z_f_linear_map(f[0]), G[i]),
                Krylov(toeplitz_Z_f_linear_map(f[1]), H[i]).t()) for i in range(rank)]
    prods = [K[0] @ (K[1] @ x.t()) for K in krylovs]
    return sum(prods).t()


def toeplitz_mult_slow_fast(G, H, x, cycle=True):
    assert G.shape == H.shape
    f_G, f_H = (1, -1) if cycle else (0, 0)
    K_G, K_H = krylov_toeplitz_fast(G, f_G), krylov_toeplitz_fast(H, f_H)
    return ((x @ K_H) @ K_G.transpose(1, 2)).sum(dim=0)
