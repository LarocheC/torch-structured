"""Wrapper around m2/csrc/flashmm/test_flash_mm.py.

Gated behind `pytest.importorskip('torch_butterfly._flashmm')` so this file is
skipped when the flashmm extension is not built. See csrc/flashmm/README.md
for how to build it.
"""

import pytest

pytest.importorskip("torch_butterfly._flashmm")


def test_flashmm_symbols_available():
    from torch_butterfly import _flashmm
    for name in ("mm_block_fwd", "hyena_filter_fwd", "exp_mod_in_place_fwd"):
        assert hasattr(_flashmm, name), name
