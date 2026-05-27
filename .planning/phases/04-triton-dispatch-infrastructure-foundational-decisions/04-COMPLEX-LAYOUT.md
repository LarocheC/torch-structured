# Phase 4: Complex64 Layout — Wrapper Boundary Decision

**Written:** 2026-05-27
**Consumers:** Phase 7 (TRI-03 butterfly forward kernel) — reads this verbatim
**Locks:** D-01, D-02, D-03 — and the call-site contract for TRI-06 implementation

## Decision (D-01)

Complex64 inputs are reinterpreted via `torch.view_as_real()` at the
`torch_structured/_ops.py` wrapper boundary (zero-copy). Triton kernels receive
trailing-2 real tensors and an `IS_COMPLEX: tl.constexpr` compile-time flag so
the same kernel source compiles to both the real-only and complex (4-FMA)
arithmetic. The wrapper restores `complex64` to the caller via
`torch.view_as_complex(...)`. Public API and `nn.Module` call sites
(`Butterfly.forward`, etc.) continue to accept and return `complex64` exactly
as today (COMPAT-01). No native `tl.complex*` is used — Triton does not provide
a complex dtype, and milestone research (PITFALLS.md §1) rejected hand-rolled
complex shims.

## Twiddle Layout Invariant (D-03)

The twiddle layout `(nstacks, nblocks, log_n, n/2, 2, 2)` is **NOT** touched
(COMPAT-02). Complex64 twiddles use this layout via `c10::complex<float>`
storage; the same memory aliases to `(nstacks, nblocks, log_n, n/2, 2, 2, 2)`
real (final 2 = re/im) under `view_as_real`. This satisfies the
`view_as_complex` stride contract verbatim — the innermost 2×2 block is stored
contiguously with last-dim stride 1, so reinterpreting it as a trailing-(2)
real dimension is the canonical zero-copy case. Phase 7 must NOT reshape or
permute the twiddle to introduce a new complex axis; the existing storage is
already in the right shape.

## Wrapper Boundary Code Template (D-02)

The canonical wrapper pattern Phase 7 (and any other phase that ports a
complex-accepting op) MUST follow:

```python
def wrapper(x: torch.Tensor) -> torch.Tensor:
    is_complex = x.is_complex()
    if is_complex:
        # Pitfall 3 (see below): non-contiguous complex (e.g. after .transpose)
        # would inherit transposed stride pattern; the kernel would read garbage.
        assert x.is_contiguous(), \
            "complex input must be contiguous before view_as_real (Pitfall 3)"
        x_real = torch.view_as_real(x)        # trailing-2 view, stride-1 last dim
    else:
        x_real = x
    out_real = _kernel_invoke(x_real, IS_COMPLEX=is_complex)
    return torch.view_as_complex(out_real.contiguous()) if is_complex else out_real
```

## Kernel-Side `IS_COMPLEX` Template

The same `@triton.jit` kernel source compiles to real-only or complex-aware
code via the `IS_COMPLEX: tl.constexpr` flag — Triton specializes per-constexpr
at JIT time, so there is no runtime branch overhead.

```python
@triton.jit
def butterfly_kernel(in_ptr, twiddle_ptr, out_ptr, ...,
                     IS_COMPLEX: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # ... load layout for real-or-complex inputs ...
    if IS_COMPLEX:
        # Load (re, im) pairs for input and twiddle
        a_re, a_im = tl.load(in_ptr + off_re), tl.load(in_ptr + off_im)
        c_re, c_im = tl.load(twiddle_ptr + t_re), tl.load(twiddle_ptr + t_im)
        # Complex multiply (a + bi)(c + di) = (ac - bd) + (ad + bc)i — 4 FMAs
        out_re = a_re * c_re - a_im * c_im
        out_im = a_re * c_im + a_im * c_re
        tl.store(out_ptr + off_re, out_re)
        tl.store(out_ptr + off_im, out_im)
    else:
        a = tl.load(in_ptr + off)
        c = tl.load(twiddle_ptr + t)
        tl.store(out_ptr + off, a * c)
```

## Contiguity Gotcha (Pitfall 3)

When a consumer passes a complex tensor obtained from a `.transpose(-1, -2)`
(this happens at `torch_structured/butterfly/butterfly.py:126` on the transpose
path), the trailing-2 view inherits the transposed stride pattern — the kernel
will read garbage because the kernel expects packed `(re, im)` with stride 1 on
the last dim. The wrapper MUST call `.contiguous()` before `view_as_real`:

```python
# Right
x_packed = torch.view_as_real(xt.contiguous())   # may copy; correct strides

# Wrong (silent incorrectness on transposed complex inputs)
x_packed = torch.view_as_real(xt)                # inherits transpose strides
```

Warning sign: complex tests pass for non-transposed cases but fail for
`Butterfly.forward(input, transpose=True, complex=True)`.

## Autograd Preservation

Both `view_as_real` and `view_as_complex` are differentiable views with
explicit autograd wiring in PyTorch's `derivatives.yaml`:

- `view_as_real` backward: `at::view_as_complex(grad.contiguous())`
- `view_as_complex` backward: `at::view_as_real(grad.contiguous().resolve_conj())`

The wrapper preserves gradient flow without manual handling. Phase 7 does NOT
need a custom backward for the complex routing — `register_autograd` on the
underlying Triton op composes with the `view_as_*` views automatically.

## Why Not `tl.complex64`

Triton has no native complex type (PITFALLS.md §1; "Don't Hand-Roll" table in
RESEARCH.md). Real/imag-split via `view_as_real` is the only viable path. A
hand-rolled `tl.complex64` shim would require packing/unpacking on every load
and store and would not compose with Triton's autotune / heuristics —
non-starter.

## Phase 7 Acceptance Reference

Phase 7 (TRI-03) implements `butterfly_multiply` forward consuming this
layout. The `ButterflyUnitary` complex test (`U U^* = I`, see
`tests/test_butterfly.py`) is the gradcheck-equivalent acceptance gate: it
verifies that the round-trip through `view_as_real → kernel → view_as_complex`
preserves unitarity to within fp32 tolerance and that gradients flow correctly
back through both view ops.
