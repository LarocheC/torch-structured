"""Tests for torch_structured.monarch.monarch_linear.MonarchLinear (genuine
two-factor Monarch: block-diagonal x permutation x block-diagonal, full
cross-channel mixing by construction -- distinct from the single-factor
BlockdiagLinear, which has zero cross-block mixing).
"""

import math

import pytest
import torch

from torch_structured.monarch.monarch_linear import MonarchLinear

# The 5 real NsNet2 layer shapes this was built for (257/400/600 fc/gru
# projections, including the two that need input or output padding since
# 257 isn't divisible by nblocks=4), plus a couple of exact-fit shapes.
SHAPES = [
    (257, 400),
    (400, 1200),
    (400, 600),
    (600, 600),
    (600, 257),
    (64, 96),
    (128, 128),
]


@pytest.mark.parametrize("in_f,out_f", SHAPES)
def test_forward_shape_and_finite(in_f, out_f):
    torch.manual_seed(0)
    lin = MonarchLinear(in_f, out_f, nblocks=4, bias=True)
    x = torch.randn(5, in_f)
    y = lin(x)
    assert y.shape == (5, out_f)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("in_f,out_f", SHAPES)
def test_backward_gradients_flow(in_f, out_f):
    torch.manual_seed(0)
    lin = MonarchLinear(in_f, out_f, nblocks=4, bias=True)
    x = torch.randn(5, in_f, requires_grad=True)
    lin(x).sum().backward()
    for t in (lin.w1.grad, lin.w2.grad, lin.bias.grad, x.grad):
        assert t is not None
        assert torch.isfinite(t).all()
        assert t.abs().sum() > 0


@pytest.mark.parametrize("in_f,out_f", SHAPES)
def test_convert_to_dense_weight_matches_forward(in_f, out_f):
    """convert_to_dense_weight() must reproduce the structured forward pass
    (minus bias) exactly, since it's built by identity-probing the same
    forward_matmul -- proves the padding/truncation bookkeeping is correct,
    not just shape-compatible."""
    torch.manual_seed(0)
    lin = MonarchLinear(in_f, out_f, nblocks=4, bias=True)
    x = torch.randn(8, in_f)
    with torch.no_grad():
        dense = lin.convert_to_dense_weight()
        y_structured = lin(x)
        y_dense = x @ dense.T + lin.bias
    assert dense.shape == (out_f, in_f)
    torch.testing.assert_close(y_structured, y_dense, atol=1e-5, rtol=1e-5)


def test_full_mixing_every_output_depends_on_every_input_block():
    """The whole point of the two-factor construction: unlike BlockdiagLinear
    (zero cross-block mixing by construction), every output element must
    depend on every structural input block."""
    torch.manual_seed(0)
    in_f, out_f, nblocks = 400, 1200, 4
    lin = MonarchLinear(in_f, out_f, nblocks=nblocks, bias=True)
    blk = in_f // nblocks
    x0 = torch.randn(1, in_f)
    with torch.no_grad():
        base = lin(x0)
        for b in range(nblocks):
            x = x0.clone()
            x[:, b * blk:(b + 1) * blk] += torch.randn(blk)
            y = lin(x)
            frac_affected = ((y - base).abs() > 1e-6).float().mean().item()
            assert frac_affected > 0.99, (
                f"block {b}: only {frac_affected:.2%} of output affected -- "
                "expected ~100% (full mixing)"
            )


def test_composed_weight_reaches_full_rank():
    """Regression guard for the intermediate-width rank bottleneck: for a
    non-square shape the composed dense-equivalent weight must reach full rank
    min(in, out). Before the fix the two intermediate block dims were set to
    nblocks (q = r = nblocks), capping the intermediate width -- and hence the
    composed rank -- at nblocks**2 (a fixed 16 for nblocks=4) regardless of the
    feature sizes. Setting q = r = in_blksz removes that cap."""
    torch.manual_seed(0)
    in_f, out_f, nblocks = 400, 1200, 4
    lin = MonarchLinear(in_f, out_f, nblocks=nblocks, bias=True)
    W = lin.convert_to_dense_weight()  # (out, in) == (1200, 400)
    assert W.shape == (out_f, in_f)
    # Count genuinely-nonzero singular values with an explicit relative
    # tolerance. The composed map is full-rank by construction, but a random
    # draw can leave one singular value a few 1e-5 below sigma_max, which the
    # DEFAULT matrix_rank tolerance rounds off (giving 399). rtol=1e-6
    # distinguishes those genuinely-nonzero values from the ~1e-16 structural
    # zeros a rank bottleneck would produce (which would drop rank to
    # nblocks**2 == 16, far below any tolerance choice).
    rank = torch.linalg.matrix_rank(W, rtol=1e-6).item()
    assert rank == min(in_f, out_f), (
        f"composed dense-equivalent rank {rank} != min(in, out) "
        f"{min(in_f, out_f)}; a rank near nblocks**2 ({nblocks ** 2}) means the "
        "intermediate-width rank bottleneck has regressed (q=r must be in_blksz, "
        "not nblocks)"
    )


def test_variance_matched_init_regression_guard():
    """Regression guard for a real bug caught during development: naively
    Kaiming-initializing w1 and w2 independently compounds the
    variance-preservation twice, undershooting the correct composed output
    variance by ~3600x for a 400->1200 shape. This asserts the composed
    output variance stays within a wide (20%) tolerance of what a plain dense
    layer of the same shape would produce -- loose enough to not be flaky,
    tight enough to catch a reintroduced compounding bug (which shrinks
    variance by orders of magnitude, not percents)."""
    torch.manual_seed(0)
    in_f, out_f, nblocks = 400, 1200, 4
    lin = MonarchLinear(in_f, out_f, nblocks=nblocks, bias=False)
    x = torch.randn(20000, in_f)
    with torch.no_grad():
        composed_var = lin(x).var().item()

    dense_ref = torch.empty(lin.out_features_extended, lin.in_features_extended)
    torch.nn.init.kaiming_uniform_(dense_ref, a=math.sqrt(5))
    target_var = dense_ref.pow(2).mean().item() * in_f

    ratio = composed_var / target_var
    assert 0.8 < ratio < 1.2, (
        f"composed output variance {composed_var:.4g} vs dense-equivalent "
        f"target {target_var:.4g} (ratio {ratio:.4g}) -- expected ~1.0; "
        "a ratio near 1/nblocks^2 or smaller indicates the two-factor "
        "variance-compounding bug has regressed"
    )
