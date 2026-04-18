"""Smoke test for torch_butterfly.monarch imports and reference paths."""

import torch

from torch_butterfly.monarch.blockdiag_multiply import (
    blockdiag_multiply_reference,
    blockdiag_weight_to_dense_weight,
)


def test_import_monarch():
    import torch_butterfly.monarch  # noqa: F401
    from torch_butterfly.monarch.blockdiag_multiply import blockdiag_multiply  # noqa: F401
    from torch_butterfly.monarch.blockdiag_butterfly_multiply import (
        blockdiag_butterfly_multiply,
        blockdiag_butterfly_multiply_reference,
    )  # noqa: F401
    from torch_butterfly.monarch.blockdiag_linear import BlockdiagLinear  # noqa: F401
    from torch_butterfly.monarch.butterfly_factor import butterfly_factor_to_matrix  # noqa: F401
    from torch_butterfly.monarch.low_rank import low_rank_project  # noqa: F401
    from torch_butterfly.monarch.structured_linear import StructuredLinear  # noqa: F401


def test_blockdiag_reference_matches_dense():
    nblocks, q, p = 4, 3, 5
    batch = 7
    x = torch.randn(batch, nblocks * p)
    weight = torch.randn(nblocks, q, p)
    out_ref = blockdiag_multiply_reference(x, weight)
    dense = blockdiag_weight_to_dense_weight(weight)
    out_dense = x @ dense.t()
    torch.testing.assert_close(out_ref, out_dense, atol=1e-4, rtol=1e-4)
