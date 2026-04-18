'''Functions to multiply by an LDR matrix with subdiagonal and tridiagonal
operator matrices.

We implement the fast multiplication for the subdiagonal case.
This comprises two steps: Krylov(g) @ Krylov(h)^T @ u, which are Krylov
transpose multiply and Krylov multiply.

For tridiagonal case, we implement the slow multiplication algorithm: construct
the Krylov matrix then call regular matrix multiply.
'''

import functools
import numpy as np

import torch
from torch.nn import functional as F

from ._compat import krylov_construct
from ._compat import complex_mult, conjugate

try:
    from torch_structured import _diag_mult_cuda as diag_mult_cuda
except (ImportError, RuntimeError) as e:
    diag_mult_cuda = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##### Fast multiplication for the subdiagonal case


def krylov_transpose_multiply_conv(subdiag, v, u):
    """Multiply Krylov(A, v_i)^T @ u when A is zero except on the subdiagonal.
    Use either Pytorch's conv1d or FFT for polynomial multiplication, depending
    on polynomial degree. This is the fastest implementation.
    Parameters:
        subdiag: Tensor of shape (n - 1, )
        v: Tensor of shape (rank, n)
        u: Tensor of shape (batch_size, n)
    Returns:
        product: Tensor of shape (batch_size, rank, n)
    """
    batch_size, n = u.shape
    rank, n_ = v.shape
    assert n == n_, 'u and v must have the same last dimension'
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'

    result = torch.zeros((batch_size, rank, n), dtype=u.dtype, device=u.device)
    T_00_sum = u @ v.t()
    result[:, :, 0] += T_00_sum
    T_01 = u[..., np.newaxis]
    T_10 = v[..., np.newaxis]
    T_11 = torch.ones(n, device=T_00_sum.device)
    for d in range(m)[::-1]:
        n1, n2 = 1 << d, 1 << (m - d - 1)
        S_00_sum, S_01, S_10, S_11 = T_00_sum, T_01, T_10, T_11
        S0_10_mult_subdiag = S_10[:, ::2] * subdiag[(n2 - 1)::(2 * n2), np.newaxis]
        if n2 <= 128:
            T_00_sum = F.conv1d(S_01[:, 1::2], S0_10_mult_subdiag.flip(2), padding=n2 - 1)
        else:
            S = torch.cat((torch.cat((S0_10_mult_subdiag, S_01[:, 1::2])),
                           torch.zeros((rank + batch_size, n1, n2), dtype=S_10.dtype, device=S_10.device)), dim=-1)
            S_f = torch.rfft(S, 1)
            S0_10_f, S1_01_f = S_f[:rank], S_f[rank:rank + batch_size]
            prod = torch.einsum('bnmo,rnmp->brmop', S1_01_f, S0_10_f)
            T_00_f_sum = torch.stack((prod[..., 0, 0] - prod[..., 1, 1], prod[..., 0, 1] + prod[..., 1, 0]), dim=-1)
            T_00_sum = torch.irfft(T_00_f_sum, 1, signal_sizes=(2 * n2, ))[..., :-1]
        result[:, :, 1:2 * n2] += T_00_sum
        S0_11_mult_subdiag = S_11[::2] * subdiag[(n2 - 1)::(2 * n2)]
        T_01 = torch.cat((S_01[:, ::2], S_01[:, 1::2] * S0_11_mult_subdiag[:, np.newaxis]), dim=-1)
        T_10 = torch.cat((S_10[:, 1::2], S0_10_mult_subdiag * S_11[1::2][:, np.newaxis]), dim=-1)
        T_11 = S0_11_mult_subdiag * S_11[1::2]
    return result


def krylov_transpose_multiply(subdiag, v, u):
    """Multiply Krylov(A, v_i)^T @ u when A is zero except on the subdiagonal.
    Parameters:
        subdiag: Tensor of shape (n - 1, )
        v: Tensor of shape (rank, n)
        u: Tensor of shape (batch_size, n)
    Returns:
        product: Tensor of shape (batch_size, rank, n)
    """
    batch_size, n = u.shape
    rank, n_ = v.shape
    assert n == n_, 'u and v must have the same last dimension'
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'

    result = torch.zeros((batch_size, rank, n), dtype=u.dtype, device=u.device)
    T_00_sum = u @ v.t()
    result[:, :, 0] = T_00_sum
    T_01 = u[..., np.newaxis]
    T_10 = v[..., np.newaxis]
    T_11 = torch.ones(n, device=T_00_sum.device)
    for d in range(m)[::-1]:
        n1, n2 = 1 << d, 1 << (m - d - 1)
        S_01, S_10, S_11 = T_01, T_10, T_11
        S0_10_mult_subdiag = S_10[:, ::2] * subdiag[(n2 - 1)::(2 * n2), np.newaxis]
        S = torch.cat((torch.cat((S0_10_mult_subdiag, S_01[:, 1::2])),
                       torch.zeros((rank + batch_size, n1, n2), dtype=S_10.dtype, device=S_10.device)), dim=-1)

        S_f = torch.rfft(S, 1)
        S0_10_f, S1_01_f = S_f[:rank], S_f[rank:rank + batch_size]
        prod = torch.einsum('bnmo,rnmp->brmop', S1_01_f, S0_10_f)
        T_00_f_sum = torch.stack((prod[..., 0, 0] - prod[..., 1, 1], prod[..., 0, 1] + prod[..., 1, 0]), dim=-1)
        T_00_sum = torch.irfft(T_00_f_sum, 1, signal_sizes=(2 * n2, ))[..., :-1]

        result[:, :, 1:2 * n2] += T_00_sum
        S0_11_mult_subdiag = S_11[::2] * subdiag[(n2 - 1)::(2 * n2)]
        T_01 = torch.cat((S_01[:, ::2], S_01[:, 1::2] * S0_11_mult_subdiag[:, np.newaxis]), dim=-1)
        T_10 = torch.cat((S_10[:, 1::2], S0_10_mult_subdiag * S_11[1::2][:, np.newaxis]), dim=-1)
        T_11 = S0_11_mult_subdiag * S_11[1::2]

    return result


def krylov_multiply_conv(subdiag, v, w):
    """Multiply \\sum_i Krylov(A, v_i) @ w_i when A is zero except on the subdiagonal.
    Since K @ w can be computed by autodiffing K^T @ u, the algorithm is just
    hand-differentiating the code of @krylov_transpose_multiply.
    """
    batch_size, rank, n = w.shape
    rank_, n_ = v.shape
    assert n == n_, 'w and v must have the same last dimension'
    assert rank == rank_, 'w and v must have the same rank'
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'

    save_for_backward = [None] * m
    T_10 = v[..., np.newaxis]
    T_11 = torch.ones((n), device=T_10.device)
    for d in range(m)[::-1]:
        n1, n2 = 1 << d, 1 << (m - d - 1)
        S_10, S_11 = T_10, T_11
        S0_10_mult_subdiag = S_10[:, ::2] * subdiag[(n2 - 1)::(2 * n2), np.newaxis]
        T_10 = torch.cat((S_10[:, 1::2], S0_10_mult_subdiag * S_11[1::2][:, np.newaxis]), dim=-1)
        S0_11_mult_subdiag = S_11[::2] * subdiag[(n2 - 1)::(2 * n2)]
        save_for_backward[d] = S0_10_mult_subdiag, S0_11_mult_subdiag
        T_11 = S0_11_mult_subdiag * S_11[1::2]

    dT_01 = torch.zeros((batch_size, 1, n), dtype=w.dtype, device=w.device)

    for d in range(m):
        n1, n2 = 1 << d, 1 << (m - d - 1)
        S0_10_mult_subdiag, S0_11_mult_subdiag = save_for_backward[d]
        dS_01 = torch.empty((batch_size, 2 * n1, n2), device=w.device)
        dS_01[:, ::2] = dT_01[:, :, :n2]
        if n2 <= 128:
            dS1_01 = F.conv_transpose1d(w[:, :, 1:2 * n2], S0_10_mult_subdiag.flip(2), padding=n2 - 1)
        else:
            dT_00_sum = torch.cat((w[:, :, 1:2 * n2], torch.zeros((batch_size, rank, 1), dtype=w.dtype, device=w.device)), dim=-1)
            dT_00_sum_f = torch.rfft(dT_00_sum, 1)
            S0_10_f = torch.rfft(torch.cat((S0_10_mult_subdiag, torch.zeros_like(S0_10_mult_subdiag)), dim=-1), 1)
            prod = torch.einsum('rnmo,brmp->bnmop', S0_10_f, dT_00_sum_f)
            dS1_01_f = torch.stack((prod[..., 0, 0] + prod[..., 1, 1], prod[..., 0, 1] - prod[..., 1, 0]), dim=-1)
            dS1_01 = torch.irfft(dS1_01_f, 1, signal_sizes=(2 * n2, ))[:, :, :n2]
        dS_01[:, 1::2] = dT_01[:, :, n2:] * S0_11_mult_subdiag[:, np.newaxis] + dS1_01

        dT_01 = dS_01

    du = w[:, :, 0] @ v + dT_01.squeeze(dim=-1)
    return du


def krylov_multiply(subdiag, v, w):
    """Multiply \\sum_i Krylov(A, v_i) @ w_i when A is zero except on the subdiagonal.
    Since K @ w can be computed by autodiffing K^T @ u, the algorithm is just
    hand-differentiating the code of @krylov_transpose_multiply.
    """
    batch_size, rank, n = w.shape
    rank_, n_ = v.shape
    assert n == n_, 'w and v must have the same last dimension'
    assert rank == rank_, 'w and v must have the same rank'
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'

    save_for_backward = [None] * m
    T_10 = v[..., np.newaxis]
    T_11 = torch.ones((n), device=T_10.device)
    for d in range(m)[::-1]:
        n1, n2 = 1 << d, 1 << (m - d - 1)
        S_10, S_11 = T_10, T_11
        S0_10_mult_subdiag = S_10[:, ::2] * subdiag[(n2 - 1)::(2 * n2), np.newaxis]
        T_10 = torch.cat((S_10[:, 1::2], S0_10_mult_subdiag * S_11[1::2][:, np.newaxis]), dim=-1)
        S0_11_mult_subdiag = S_11[::2] * subdiag[(n2 - 1)::(2 * n2)]
        save_for_backward[d] = S0_10_mult_subdiag, S0_11_mult_subdiag
        T_11 = S0_11_mult_subdiag * S_11[1::2]

    dT_01 = torch.zeros((batch_size, 1, n), dtype=w.dtype, device=w.device)

    for d in range(m):
        n1, n2 = 1 << d, 1 << (m - d - 1)
        S0_10_mult_subdiag, S0_11_mult_subdiag = save_for_backward[d]
        dS_01 = torch.empty((batch_size, 2 * n1, n2), device=w.device)
        dS_01[:, ::2] = dT_01[:, :, :n2]
        dT_00_sum = torch.cat((w[:, :, 1:2 * n2], torch.zeros((batch_size, rank, 1), dtype=w.dtype, device=w.device)), dim=-1)

        dT_00_sum_f = torch.rfft(dT_00_sum, 1)
        S0_10_f = torch.rfft(torch.cat((S0_10_mult_subdiag, torch.zeros_like(S0_10_mult_subdiag)), dim=-1), 1)
        prod = torch.einsum('rnmo,brmp->bnmop', S0_10_f, dT_00_sum_f)
        dS1_01_f = torch.stack((prod[..., 0, 0] + prod[..., 1, 1], prod[..., 0, 1] - prod[..., 1, 0]), dim=-1)
        dS1_01 = torch.irfft(dS1_01_f, 1, signal_sizes=(2 * n2, ))[:, :, :n2]
        dS_01[:, 1::2] = dT_01[:, :, n2:] * S0_11_mult_subdiag[:, np.newaxis] + dS1_01

        dT_01 = dS_01

    du = w[:, :, 0] @ v + dT_01.squeeze(dim=-1)
    return du


def krylov_multiply_by_autodiff(subdiag, v, w):
    """Multiply via autodiff of K^T @ u."""
    batch_size, rank, n = w.shape
    rank_, n_ = v.shape
    assert n == n_
    assert rank == rank_
    m = int(np.log2(n))
    assert n == 1 << m

    u = torch.zeros((batch_size, n), dtype=v.dtype, device=v.device, requires_grad=True)
    prod = krylov_transpose_multiply(subdiag, v, u)
    result, = torch.autograd.grad(prod, u, grad_outputs=w, create_graph=True)
    return result


def subdiag_mult_conv(subdiag_A, subdiag_B, G, H, x):
    """Multiply \\sum_i Krylov(A, G_i) @ Krylov(B, H_i) @ x (fast, conv1d path)."""
    rank, n = G.shape
    batch_size = x.shape[0]
    m = int(np.ceil(np.log2(n)))
    n_extended = 1 << m
    if n != n_extended:
        x = torch.cat((x, torch.zeros(batch_size, n_extended - n, dtype=x.dtype, device=x.device)), dim=-1)
        G = torch.cat((G, torch.zeros(rank, n_extended - n, dtype=G.dtype, device=G.device)), dim=-1)
        H = torch.cat((H, torch.zeros(rank, n_extended - n, dtype=H.dtype, device=H.device)), dim=-1)
        subdiag_A = torch.cat((subdiag_A, torch.zeros(n_extended - n, dtype=subdiag_A.dtype, device=subdiag_A.device)))
        subdiag_B = torch.cat((subdiag_B, torch.zeros(n_extended - n, dtype=subdiag_B.dtype, device=subdiag_B.device)))
    KT_out = krylov_transpose_multiply_conv(subdiag_B, H, x)
    K_out = krylov_multiply_conv(subdiag_A, G, KT_out)
    return K_out[:, :n] if n != n_extended else K_out


def subdiag_mult(subdiag_A, subdiag_B, G, H, x):
    """Multiply \\sum_i Krylov(A, G_i) @ Krylov(B, H_i) @ x (fast)."""
    rank, n = G.shape
    batch_size = x.shape[0]
    m = int(np.ceil(np.log2(n)))
    n_extended = 1 << m
    if n != n_extended:
        x = torch.cat((x, torch.zeros(batch_size, n_extended - n, dtype=x.dtype, device=x.device)), dim=-1)
        G = torch.cat((G, torch.zeros(rank, n_extended - n, dtype=G.dtype, device=G.device)), dim=-1)
        H = torch.cat((H, torch.zeros(rank, n_extended - n, dtype=H.dtype, device=H.device)), dim=-1)
        subdiag_A = torch.cat((subdiag_A, torch.zeros(n_extended - n, dtype=subdiag_A.dtype, device=subdiag_A.device)))
        subdiag_B = torch.cat((subdiag_B, torch.zeros(n_extended - n, dtype=subdiag_B.dtype, device=subdiag_B.device)))
    KT_out = krylov_transpose_multiply(subdiag_B, H, x)
    K_out = krylov_multiply(subdiag_A, G, KT_out)
    return K_out[:, :n] if n != n_extended else K_out


##### Slow multiplication for the subdiagonal case

def Krylov(linear_map, v, m=None):
    """Explicit construction of Krylov matrix [v  A @ v  A^2 @ v  ...  A^{m-1} @ v]."""
    if m is None:
        m = v.size(-1)
    cols = [v]
    for _ in range(m - 1):
        v = linear_map(v)
        cols.append(v)
    return torch.stack(cols, dim=-1)


def shift_subdiag(subdiag, v, upper_right_corner=0.0):
    return torch.cat((upper_right_corner * v[[-1]], subdiag * v[:-1]))


def subdiag_linear_map(subdiag, upper_right_corner=0.0):
    n = subdiag.size(0) + 1
    shift_down = torch.arange(-1, n - 1, device=subdiag.device)
    subdiag_extended = torch.cat((torch.tensor([upper_right_corner], dtype=subdiag.dtype, device=subdiag.device), subdiag))
    return lambda v: subdiag_extended * v[..., shift_down]


def krylov_subdiag_fast(subdiag, v, upper_right_corner=0.0):
    rank, n = v.shape
    a = torch.arange(n, dtype=torch.long, device=v.device)
    b = -a
    indices = a[:, np.newaxis] + b[np.newaxis]
    v_circulant = v[:, indices]
    subdiag_extended = torch.cat((torch.tensor([upper_right_corner], dtype=subdiag.dtype, device=subdiag.device), subdiag))
    subdiag_circulant = subdiag_extended[indices]
    subdiag_cumprod = subdiag_circulant.cumprod(dim=1)
    K = v_circulant
    K[:, :, 1:] *= subdiag_cumprod[:, :-1]
    return K


def subdiag_mult_slow_old(subdiag_A, subdiag_B, G, H, x):
    rank, n = G.shape
    linear_map_A = functools.partial(shift_subdiag, subdiag_A)
    linear_map_B = functools.partial(shift_subdiag, subdiag_B)
    krylovs = [(Krylov(linear_map_A, G[i]), Krylov(linear_map_B, H[i]).t()) for i in range(rank)]
    prods = [K[0] @ (K[1] @ x.t()) for K in krylovs]
    return sum(prods).t()


def subdiag_mult_slow(subdiag_A, subdiag_B, G, H, x, corner_A=0.0, corner_B=0.0):
    if G.shape[0] == 1:
        K_G = Krylov(subdiag_linear_map(subdiag_A, corner_A), G[0])
        K_H = Krylov(subdiag_linear_map(subdiag_B, corner_B), H[0])
        return (x @ K_H) @ K_G.t()
    else:
        K_G = Krylov(subdiag_linear_map(subdiag_A, corner_A), G)
        K_H = Krylov(subdiag_linear_map(subdiag_B, corner_B), H)
        return ((x @ K_H) @ K_G.transpose(1, 2)).sum(dim=0)


def subdiag_mult_slow_fast(subdiag_A, subdiag_B, G, H, x):
    K_G, K_H = krylov_subdiag_fast(subdiag_A, G), krylov_subdiag_fast(subdiag_B, H)
    return ((x @ K_H) @ K_G.transpose(1, 2)).sum(dim=0)


class CycleDownMultCuda(torch.autograd.Function):
    '''Cycle v down and do pointwise multiplication with subdiag.
    '''
    @staticmethod
    def forward(ctx, subdiag, v):
        ctx.save_for_backward(subdiag, v)
        return diag_mult_cuda.cycle_mult(subdiag, v, 0, -1)

    @staticmethod
    def backward(ctx, grad):
        subdiag, v = ctx.saved_tensors
        return diag_mult_cuda.cycle_mult(grad, v, 0, -1).sum(dim=0), diag_mult_cuda.cycle_mult(subdiag, grad, 1, 1)


cycle_down_mult = CycleDownMultCuda.apply


def subdiag_linear_map_cuda(subdiag, upper_right_corner=0.0):
    subdiag_extended = torch.cat((torch.tensor([upper_right_corner], dtype=subdiag.dtype, device=subdiag.device), subdiag))
    return lambda v: cycle_down_mult(subdiag_extended, v)


def subdiag_mult_cuda(subdiag_A, subdiag_B, G, H, x, corner_A=0.0, corner_B=0.0):
    K_G = Krylov(subdiag_linear_map_cuda(subdiag_A, corner_A), G)
    K_H = Krylov(subdiag_linear_map_cuda(subdiag_B, corner_B), H)
    return ((x @ K_H) @ K_G.transpose(1, 2)).sum(dim=0)


##### Slow multiplication for the tridiagonal case

def tridiag_linear_map(subdiag, diag, superdiag, upper_right_corner=0.0, lower_left_corner=0.0):
    n = diag.size(0)
    shift_none = torch.arange(n, device=diag.device)
    shift_down = shift_none - 1
    shift_up = (shift_none + 1) % n
    shifts = torch.stack((shift_down, shift_none, shift_up))
    subdiag_extended = torch.cat((torch.tensor([upper_right_corner], dtype=subdiag.dtype, device=subdiag.device), subdiag))
    superdiag_extended = torch.cat((superdiag, torch.tensor([lower_left_corner], dtype=superdiag.dtype, device=superdiag.device)))
    diags = torch.stack((subdiag_extended, diag, superdiag_extended))
    return lambda v: (diags * v[..., shifts]).sum(dim=-2)


def tridiag_linear_map_slow(subdiag, diag, superdiag, upper_right_corner=0.0, lower_left_corner=0.0):
    return lambda v: torch.cat((upper_right_corner * v[..., -1:], subdiag * v[..., :-1]), dim=-1) + diag * v + torch.cat((superdiag * v[..., 1:], lower_left_corner * v[..., :1]), dim=-1)


def tridiag_mult_slow(subdiag_A, diag_A, superdiag_A, subdiag_B, diag_B, superdiag_B, G, H, x,
                     corners_A=(0.0, 0.0), corners_B=(0.0, 0.0)):
    if G.shape[0] == 1:
        K_G = Krylov(tridiag_linear_map(subdiag_A, diag_A, superdiag_A, *corners_A), G[0])
        K_H = Krylov(tridiag_linear_map(subdiag_B, diag_B, superdiag_B, *corners_B), H[0])
        return (x @ K_H) @ K_G.t()
    else:
        K_G = Krylov(tridiag_linear_map(subdiag_A, diag_A, superdiag_A, *corners_A), G)
        K_H = Krylov(tridiag_linear_map(subdiag_B, diag_B, superdiag_B, *corners_B), H)
        return ((x @ K_H) @ K_G.transpose(1, 2)).sum(dim=0)
