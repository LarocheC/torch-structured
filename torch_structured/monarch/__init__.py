"""Monarch / block-diagonal-butterfly primitives.

Ported from https://github.com/HazyResearch/m2 (bert/src/{mm,ops}/), which
accompanies:
- Dao et al., *Monarch: Expressive Structured Matrices for Efficient and
  Accurate Training*, ICML 2022
- Fu, Arora, Grogan et al., *Monarch Mixer: A Simple Sub-Quadratic
  GEMM-Based Architecture*, NeurIPS 2023

This subpackage contains only the structured-matrix primitives (block-diagonal
and block-diagonal-butterfly multiply, structured linear layers, a butterfly
factor helper, and the HyenaFilter implicit long-filter). BERT-specific
training code, fused dense / dropout / flash-attention, and the end-to-end
Monarch Mixer sequence mixer are out of scope.
"""
