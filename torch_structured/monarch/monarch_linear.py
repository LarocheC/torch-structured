"""Genuine two-factor Monarch linear layer (block-diagonal x permutation x
block-diagonal), as in Dao et al., "Monarch: Expressive Structured Matrices
for Efficient and Accurate Training" (ICML 2022, arxiv:2204.00595).

Distinct from `BlockdiagLinear` (a single block-diagonal factor, no
permutation, zero cross-block information flow by construction -- that is
what this library's `"monarch"` kind has always meant). `MonarchLinear` wires
up the two-factor construction that already exists as a raw primitive in
`blockdiag_butterfly_multiply.py` (ported from the paper's own reference
implementation) but was never exposed as a usable nn.Module: full
cross-channel mixing by construction, regardless of any downstream
chunking/reshaping applied to its output.
"""

import math

import torch
import torch.nn as nn

from .structured_linear import StructuredLinear
from .blockdiag_butterfly_multiply import blockdiag_butterfly_multiply


class MonarchLinear(StructuredLinear):

    def __init__(self, *args, nblocks=4, **kwargs):
        """Two factors w1 (k=nblocks, q=in_blksz, p=in_blksz) and
        w2 (l=nblocks, s=out_blksz, r=in_blksz), i.e. q = r = in_blksz.

        The primitive's only two shape constraints are k*p == n (input) and
        l*r == k*q (intermediate width matches). With k == l == nblocks, the
        second constraint forces r == q. Choosing q = r = in_blksz makes the
        intermediate width k*q == nblocks * in_blksz == in_features_extended,
        i.e. as wide as the (padded) input itself.

        Setting q = r = nblocks instead (the previous formulation) capped the
        intermediate width at nblocks**2 (a fixed 16 for nblocks=4) regardless
        of feature sizes, which upper-bounds the composed dense-equivalent rank
        at nblocks**2 -- a severe rank bottleneck for any layer wider than
        nblocks**2. Using in_blksz for the intermediate dim removes that
        bottleneck: the composed map can reach full rank min(in, out) for any
        shape, at the cost of the larger (and correct) two-factor parameter
        count.
        """
        super().__init__(*args, **kwargs)
        b = nblocks
        self.nblocks = b
        in_blksz = int(math.ceil(self.in_features / b))
        out_blksz = int(math.ceil(self.out_features / b))
        self.in_blksz, self.out_blksz = in_blksz, out_blksz
        self.in_features_extended = b * in_blksz
        self.out_features_extended = b * out_blksz
        self.w1 = nn.Parameter(torch.empty(b, in_blksz, in_blksz))
        self.w2 = nn.Parameter(torch.empty(b, out_blksz, in_blksz))
        self.reset_parameters()

    def set_weights_from_dense_init(self, dense_init_fn_):
        """Variance-matched two-factor init.

        Naively Kaiming-initializing w1 and w2 independently (each "correctly"
        scaled for its own fan-in) compounds the variance-preservation twice,
        undershooting the correct composed variance by ~3600x for a
        400->1200 shape (measured) -- since w1/w2 are two halves of ONE linear
        map with no nonlinearity between them, not a 2-layer MLP. Instead,
        measure the empirical per-element variance dense_init_fn_ would
        produce on a plain (out_features_extended, in_features_extended)
        tensor, then rescale both factors (after their own dense_init_fn_
        draw, which only sets their *shape* of randomness, not scale) so the
        COMPOSED output variance matches that target:
        Var(out) = nblocks * in_blksz * Var(w1) * Var(w2) * Var(x)
        (nblocks * in_blksz == in_features_extended exactly, by construction),
        so setting Var(w1) = Var(w2) = sqrt(v_target) makes
        Var(out) = in_features_extended * v_target * Var(x), matching a plain
        dense/BlockdiagLinear-equivalent layer.
        """
        device, dtype = self.w1.device, self.w1.dtype
        dense_ref = torch.empty(self.out_features_extended, self.in_features_extended,
                                device=device, dtype=dtype)
        dense_init_fn_(dense_ref)
        v_target = dense_ref.pow(2).mean()
        with torch.no_grad():
            dense_init_fn_(self.w1)
            dense_init_fn_(self.w2)
            v1 = self.w1.pow(2).mean().clamp_min(1e-12)
            v2 = self.w2.pow(2).mean().clamp_min(1e-12)
            target = v_target.sqrt()
            self.w1.mul_((target / v1).sqrt())
            self.w2.mul_((target / v2).sqrt())

    @property
    def saving(self):
        return (self.w1.numel() + self.w2.numel()) / (self.in_features * self.out_features)

    def convert_to_dense_weight(self):
        # Override: the base class's default assumes a single self.weight,
        # which doesn't apply here (two factors, no unified weight tensor).
        factory_kwargs = {'device': self.w1.device, 'dtype': self.w1.dtype}
        dense_weight = self.forward_matmul(torch.eye(self.in_features, **factory_kwargs)).T
        return dense_weight

    def forward_matmul(self, x):
        x = self.preprocess(x)
        output = blockdiag_butterfly_multiply(x, self.w1, self.w2)
        return self.postprocess(output)
