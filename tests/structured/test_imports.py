"""Smoke test for torch_structured.structured imports."""


def test_import_structured():
    import torch_structured.structured  # noqa: F401
    from torch_structured.structured.hadamard import (
        hadamard_transform, hadamard_transform_torch,
    )
    from torch_structured.structured.circulant import circulant_multiply
    from torch_structured.structured.fastfood import fastfood_multiply
    from torch_structured.structured.krylov import Krylov, subdiag_linear_map
    assert callable(hadamard_transform)
    assert callable(hadamard_transform_torch)
    assert callable(circulant_multiply)
    assert callable(fastfood_multiply)
    assert callable(Krylov)
    assert callable(subdiag_linear_map)


def test_import_structured_layers():
    from torch_structured.structured.layers import (
        Layer, Circulant, FastFood, LowRank, ToeplitzLike, HankelLike,
        VandermondeLike, LDRSubdiagonal, LDRTridiagonal, class_map,
        StructuredLinear,
    )
    # class_map is populated from class __subclasses__ at import time.
    assert "toeplitz" in class_map
    assert "circulant" in class_map
    assert "fastfood" in class_map
    assert callable(StructuredLinear)
