import math
from typing import List, Tuple, Union

import numpy as np

import torch
from torch import nn

from .butterfly import Butterfly
from .complex_utils import index_last_dim, real2complex


def bitreversal_permutation(n, pytorch_format=False):
    """Return the bit reversal permutation used in FFT.
    """
    log_n = int(math.log2(n))
    assert n == 1 << log_n, 'n must be a power of 2'
    perm = np.arange(n).reshape(n, 1)
    for i in range(log_n):
        n1 = perm.shape[0] // 2
        perm = np.hstack((perm[:n1], perm[n1:]))
    perm = perm.squeeze(0)
    return perm if not pytorch_format else torch.tensor(perm)


def wavelet_permutation(n, pytorch_format=False):
    """Return the bit reversal permutation used in discrete wavelet transform.
    Example: [0, 1, ..., 7] -> [0, 4, 2, 6, 1, 3, 5, 7]
    """
    log_n = int(math.log2(n))
    assert n == 1 << log_n, 'n must be a power of 2'
    perm = np.arange(n)
    head, tail = perm[:], perm[:0]
    for i in range(log_n):
        even, odd = head[::2], head[1::2]
        head = even
        tail = np.hstack((odd, tail))
    perm = np.hstack((head, tail))
    return perm if not pytorch_format else torch.tensor(perm)


class FixedPermutation(nn.Module):

    def __init__(self, permutation: torch.Tensor) -> None:
        """Fixed permutation."""
        super().__init__()
        self.register_buffer('permutation', permutation)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return index_last_dim(input, self.permutation)

    def to_butterfly(self, complex=False, increasing_stride=False):
        return perm2butterfly(self.permutation, complex, increasing_stride)


def invert(perm: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Get the inverse of a given permutation vector."""
    assert isinstance(perm, (np.ndarray, torch.Tensor))
    n = perm.shape[-1]
    if isinstance(perm, np.ndarray):
        result = np.empty(n, dtype=int)
        result[perm] = np.arange(n, dtype=int)
    else:
        result = torch.empty(n, dtype=int, device=perm.device)
        result[perm] = torch.arange(n, dtype=int)
    return result


def perm_vec_to_mat(p: np.ndarray, left: bool = False) -> np.ndarray:
    """Convert a permutation vector to a permutation matrix."""
    n = len(p)
    matrix = np.zeros((n, n), dtype=int)
    matrix[p, np.arange(n, dtype=int)] = 1
    return matrix if not left else matrix.T


def perm_mat_to_vec(m, left=False):
    """Convert a permutation matrix to a permutation vector."""
    input = np.arange(m.shape[0])
    return m @ input if left else m.T @ input


def is_2x2_block_diag(mat: np.ndarray) -> bool:
    """Check that each of the 4 blocks of a matrix is diagonal."""
    nh = mat.shape[0] // 2
    for block in [mat[:nh, :nh], mat[:nh, nh:], mat[nh:, :nh], mat[nh:, nh:]]:
        if np.count_nonzero(block - np.diag(np.diagonal(block))):
            return False
    return True


def is_butterfly_factor(mat: np.ndarray, k: int) -> bool:
    """Checks whether "mat" is in B_k."""
    assert k > 1 and k == 1 << int(math.log2(k))
    n = mat.shape[0]
    assert n >= k and n == 1 << int(math.log2(n))
    z = np.zeros(mat.shape)
    for i in range(n // k):
        block = mat[i * k:(i + 1) * k, i * k:(i + 1) * k]
        if not is_2x2_block_diag(block):
            return False
        z[i * k:(i + 1) * k, i * k:(i + 1) * k] = block
    return np.count_nonzero(mat - z) == 0


def matrix_to_butterfly_factor(mat, log_k, pytorch_format=False, check_input=False):
    """Converts a matrix to a butterfly factor B_k."""
    k = 1 << log_k
    if check_input:
        assert is_butterfly_factor(mat, k)
    n = mat.shape[0]
    out = np.zeros((n // 2, 2, 2))
    for block in range(n // 2):
        base = (2 * block // k) * k + (block % (k // 2))
        for i, j in np.ndindex((2, 2)):
            out[block, i, j] = mat[base + i * k // 2, base + j * k // 2]
    if pytorch_format:
        out = torch.tensor(out, dtype=torch.float32)
    return out


class Node:
    def __init__(self, value):
        self.value = value
        self.in_edges = []
        self.out_edges = []


def half_balance(
    v: np.ndarray, return_swap_locations: bool = False
) -> Tuple[Union[np.ndarray, torch.Tensor], np.ndarray]:
    """Return the permutation vector that makes the permutation vector v n//2-balanced."""
    n = len(v)
    assert n % 2 == 0
    nh = n // 2
    nodes = [Node(i) for i in range(nh)]
    for i in range(nh):
        s, t = nodes[v[i] % nh], nodes[v[i + nh] % nh]
        s.out_edges.append((t, i))
        t.in_edges.append((s, i + nh))
    assert all(len(node.in_edges) + len(node.out_edges) == 2 for node in nodes)
    swap_low_locs = []
    swap_high_locs = []
    while len(nodes):
        start_node, start_loc = nodes[-1], n - 1
        next_node = None
        while next_node != start_node:
            if next_node is None:
                next_node, next_loc = start_node, start_loc
            old_node, old_loc = next_node, next_loc
            if old_node.out_edges:
                next_node, old_loc = old_node.out_edges.pop()
                next_loc = old_loc + nh
                next_node.in_edges.remove((old_node, next_loc))
            else:
                next_node, old_loc = old_node.in_edges.pop()
                next_loc = old_loc - nh
                next_node.out_edges.remove((old_node, next_loc))
                swap_low_locs.append(next_loc)
                swap_high_locs.append(old_loc)
            nodes.remove(old_node)
    perm = np.arange(n, dtype=int)
    perm[swap_low_locs], perm[swap_high_locs] = swap_high_locs, swap_low_locs
    if not return_swap_locations:
        return perm, v[perm]
    else:
        return swap_low_locs, v[perm]


def modular_balance(v: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """Returns the sequence of permutations to transform v into a modular-balanced matrix."""
    t = n = len(v)
    perms = []
    while t >= 2:
        chunks = np.split(v, n // t)
        swap_perm = np.hstack([half_balance(chunk)[0] + chunk_idx * t
                               for chunk_idx, chunk in enumerate(chunks)])
        v = v[swap_perm]
        perms.append(swap_perm)
        t //= 2
    return perms, v


def is_modular_balanced(perm):
    """Corresponds to Definition G.1 in the paper."""
    if isinstance(perm, np.ndarray) and len(perm.shape) > 1:
        perm = perm_mat_to_vec(perm)
    n = len(perm)
    log_n = int(math.log2(n))
    assert n == 1 << log_n
    for j in (1 << k for k in range(1, log_n + 1)):
        for chunk in range(n // j):
            mod_vals = set(perm[i] % j for i in range(chunk * j, (chunk + 1) * j))
            if len(mod_vals) != j:
                return False
    return True


def modular_balanced_to_butterfly_factor(L: np.ndarray) -> List[np.ndarray]:
    """Returns a sequence of butterfly factors that, when multiplied together, create L."""
    if isinstance(L, list) or len(L.shape) == 1:
        L = perm_vec_to_mat(L)
    n = L.shape[0]
    import scipy.linalg
    if n == 2:
        return [L.copy()]
    nh = n // 2
    L1 = L[:nh, :nh] + L[nh:, :nh]
    L2 = L[:nh, nh:] + L[nh:, nh:]
    Lp = scipy.linalg.block_diag(L1, L2)
    Bn = L @ Lp.T
    perms1 = modular_balanced_to_butterfly_factor(L1)
    perms2 = modular_balanced_to_butterfly_factor(L2)
    return [Bn] + [scipy.linalg.block_diag(p1, p2) for p1, p2 in zip(perms1, perms2)]


def perm2butterfly_slow(v: Union[np.ndarray, torch.Tensor],
                        complex: bool = False,
                        increasing_stride: bool = False) -> Butterfly:
    """Convert a permutation to a Butterfly that performs the same permutation.
    Slower but follows the proofs in Appendix G more closely.
    """
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    n = len(v)
    log_n = int(math.ceil(math.log2(n)))
    if n < 1 << log_n:
        v = np.concatenate([v, np.arange(n, 1 << log_n)])
    if increasing_stride:
        br = bitreversal_permutation(1 << log_n)
        b = perm2butterfly_slow(br[v[br]], complex=complex, increasing_stride=False)
        b.increasing_stride = True
        br_half = bitreversal_permutation((1 << log_n) // 2, pytorch_format=True)
        with torch.no_grad():
            b.twiddle.copy_(b.twiddle[:, :, :, br_half])
        b.in_size = b.out_size = n
        return b
    Rinv_perms, L_vec = modular_balance(invert(v))
    L_perms = list(reversed(modular_balanced_to_butterfly_factor(L_vec)))
    R_perms = [perm_vec_to_mat(invert(p), left=True) for p in reversed(Rinv_perms)]
    L_twiddle = torch.stack([matrix_to_butterfly_factor(l.T, log_k=i + 1, pytorch_format=True)
                             for i, l in enumerate(L_perms)])
    R_twiddle = torch.stack([matrix_to_butterfly_factor(r, log_k=i + 1, pytorch_format=True)
                             for i, r in enumerate(R_perms)]).flip([0])
    twiddle = torch.stack([R_twiddle, L_twiddle]).unsqueeze(0)
    b = Butterfly(n, n, bias=False, complex=complex, increasing_stride=False,
                  init=twiddle if not complex else real2complex(twiddle), nblocks=2)
    return b


def swap_locations_to_twiddle_factor(n: int, swap_locations: np.ndarray) -> torch.Tensor:
    twiddle = torch.eye(2).expand(n // 2, 2, 2).contiguous()
    swap_matrix = torch.tensor([[0, 1], [1, 0]], dtype=torch.float)
    twiddle[swap_locations] = swap_matrix.unsqueeze(0)
    return twiddle


def outer_twiddle_factors(v: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Decompose the permutations v to get twiddle factors."""
    batch_size, size = v.shape
    assert size >= 2
    v_right = np.vstack([invert(chunk) for chunk in v])
    half_balance_results = [half_balance(chunk, return_swap_locations=True)
                            for chunk in v_right]
    twiddle_right_factor = torch.cat(
        [swap_locations_to_twiddle_factor(size, swap_low_locs)
            for swap_low_locs, _ in half_balance_results]
    )
    v_right = np.vstack([v_permuted for _, v_permuted in half_balance_results])
    v_left = np.vstack([invert(perm) for perm in v_right])
    size_half = size // 2
    swap_low_x, swap_low_y = np.nonzero(v_left[:, :size_half] // size_half == 1)
    swap_low_locs_flat = swap_low_y + swap_low_x * size // 2
    twiddle_left_factor = swap_locations_to_twiddle_factor(batch_size * size, swap_low_locs_flat)
    v_left[swap_low_x, swap_low_y], v_left[swap_low_x, swap_low_y + size_half] = (
        v_left[swap_low_x, swap_low_y + size // 2], v_left[swap_low_x, swap_low_y]
    )
    new_v = (v_left % size_half).reshape(batch_size * 2, size // 2)
    assert np.allclose(np.sort(new_v), np.arange(size // 2))
    return twiddle_right_factor, twiddle_left_factor, new_v


def perm2butterfly(v: Union[np.ndarray, torch.Tensor],
                   complex: bool = False,
                   increasing_stride: bool = False) -> Butterfly:
    """Convert a permutation to a Butterfly that performs the same permutation."""
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    n = len(v)
    log_n = int(math.ceil(math.log2(n)))
    if n < 1 << log_n:
        v = np.concatenate([v, np.arange(n, 1 << log_n)])
    if increasing_stride:
        br = bitreversal_permutation(1 << log_n)
        b = perm2butterfly(br[v[br]], complex=complex, increasing_stride=False)
        b.increasing_stride = True
        br_half = bitreversal_permutation((1 << log_n) // 2, pytorch_format=True)
        with torch.no_grad():
            b.twiddle.copy_(b.twiddle[:, :, :, br_half])
        b.in_size = b.out_size = n
        return b
    v = v[None]
    twiddle_right_factors, twiddle_left_factors = [], []
    for _ in range(log_n):
        right_factor, left_factor, v = outer_twiddle_factors(v)
        twiddle_right_factors.append(right_factor)
        twiddle_left_factors.append(left_factor)
    twiddle = torch.stack([torch.stack(twiddle_right_factors),
                           torch.stack(twiddle_left_factors).flip([0])]).unsqueeze(0)
    b = Butterfly(n, n, bias=False, complex=complex, increasing_stride=False,
                  init=twiddle if not complex else real2complex(twiddle), nblocks=2)
    return b
