# flashmm kernel sources

The flashmm CUDA extension is opt-in. To build it:

1. Install NVIDIA MathDx 22.02 and drop the headers under
   `csrc/flashmm/mathdx/22.02/include/`.

2. Fetch the two large kernel sources that are not vendored in this repo:

   ```
   python csrc/flashmm/fetch_kernel_sources.py
   ```

   This downloads `mm_block_fwd_cuda.cu` (34 KB) and `lut.h` (59 KB) from
   the upstream m2 repository. The `lut.h` file can alternatively be
   regenerated locally with `lut_code_gen.py` once MathDx is installed.

3. Build with the env var set:

   ```
   TORCH_BUTTERFLY_BUILD_FLASHMM=1 FORCE_CUDA=1 uv pip install -e .
   ```

The Python side (`torch_butterfly.monarch.flash_mm`) imports the compiled
extension lazily, so other subpackages remain usable even if flashmm is not
built.
