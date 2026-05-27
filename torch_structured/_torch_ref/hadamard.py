"""Pure-PyTorch reference implementation of the Walsh-Hadamard transform (D-32, D-35c).

Moved verbatim from ``torch_structured/structured/hadamard.py:15-30`` (per Phase 6 D-33d).
The function name, signature, and body are preserved exactly so the back-compat shim
at ``structured/hadamard.py`` (D-33d) and the existing
``tests/structured/test_hadamard.py`` import surface continue to work unchanged.

Three roles per the Phase 6 plan:

1. **Gradcheck oracle.** ``torch.autograd.gradcheck`` runs against this fp64-capable
   pure-PyTorch implementation; the Triton kernel's ``register_autograd`` backward
   callback also invokes this function (per D-32).
2. **Runtime fallback** for the D-22 asymmetric path: when
   ``TORCH_STRUCTURED_BACKEND=cuda`` is requested but ``_hadamard_cuda.so`` is not
   built, the resolver binds ``_ops.hadamard_transform`` to this function and emits
   a ``log.warning``.
3. **Backend = ``torch``** path: this is the implementation used when
   ``set_backend("torch")`` is in effect (no GPU required).

Self-inverse property (per ROADMAP SC#2): ``H @ (H @ u) = N * u`` (unnormalized) or
``u`` (normalized). The kernel and this oracle share this invariant; the
``test_hadamard_self_inverse`` test fires on any sign error in either path.

Note: ``np.log2`` (not ``n.bit_length() - 1``) preserves numerical parity with the
existing ``structured/hadamard.py`` path — for odd ``m``, ``2 ** (m / 2)`` is a float
expression ``sqrt(2) * 2 ** ((m - 1) / 2)`` and the dtype of ``m`` matters.
"""
import numpy as np
import torch


def hadamard_transform_torch(u, normalize=False):
    """Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
    n must be a power of 2.
    Parameters:
        u: Tensor of shape (..., n) where n is a power of 2
        normalize: if True, divide the result by 2^{m/2} where m = log_2(n).
    Returns:
        product: Tensor of shape (..., n) — same shape as input
    """
    # Rank-N handling per 06-VERIFICATION.md gap closure (SC#2 strict reading):
    # the wrapper at _triton/hadamard_transform/op.py:104 advertises (*batch, n).
    # The interleaved-butterfly loop body uses `...` indexing and is rank-N-correct
    # already; reshape to rank-2 minimizes diff vs the verbatim-relocation lineage
    # from structured/hadamard.py:15-30 (preserves np.log2 + torch.cat semantics).
    original_shape = u.shape
    n = u.shape[-1]
    u = u.reshape(-1, n)
    batch_size = u.shape[0]
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    x = u[..., np.newaxis]
    for d in range(m)[::-1]:
        x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
    out = x.squeeze(-2) / 2**(m / 2) if normalize else x.squeeze(-2)
    return out.reshape(original_shape)
