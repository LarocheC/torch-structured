# Phase 9: Integration Hardening & Correctness Gates - Context

**Gathered:** 2026-05-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Close out v1.2 correctness + perf parity before Phase 10's deprecation cleanup. **Three plans by SC group** (D-60):
- **09-01 — Foundations (SC#1 + SC#3):** Widen `tests/conftest.py` `backend` fixture to `params=["torch", "triton", "cuda"]` with **per-op cuda skip-gate** (D-62, mirrors Phase 6 D-39 — granular `_has_cuda_legacy_for_op(op)` probe). End-to-end `pytest tests/` under `TORCH_STRUCTURED_BACKEND=triton` passes existing suite (TEST-06). v1.0/v1.1 checkpoint round-trip via `Butterfly.load_state_dict` on Triton backend (COMPAT-02), producing identical forward outputs to CUDA path. Verifies `make_linear` + `LRU` continue to work unchanged (COMPAT-01, COMPAT-03). Covers: TEST-03, TEST-06, COMPAT-01, COMPAT-02, COMPAT-03.
- **09-02 — Compose (SC#2):** `torch.compile(model)` traces cleanly through Triton kernels with no graph breaks for `Butterfly` / `ButterflyBmm` / `LRU` / `make_linear` — resolves the 260419-p27 fake-tensor bug end-to-end. 2-GPU FSDP smoke test with `twiddle` no-shard hint via `torchrun --nproc_per_node=2`, marked `@pytest.mark.multigpu` (D-63). DDP + `torch.utils.checkpoint.checkpoint` produce correct gradients. Deterministic mode opt-in via `torch_structured.set_deterministic(True)` top-level function (D-64). Covers: COMPAT-04.
- **09-03 — Perf + docs (SC#4 + SC#5):** Perf grid + runtime selector + README. Reuse the Phase 7+8 `07-BASELINE.json` 16-row schema (4 log_n × 2 dtype × 2 direction at batch_size=64, nstacks=1, nblocks=1) per D-65. Add automated ≥60% gate computing `wall_ms_p50_triton / reference_torch_ref_p50_cuda ≤ 1.0/0.60 = 1.67` per cell. **Runtime selector in `_ops.py`** baked from baseline JSON (D-66 — `_should_route_to_cuda(op, shape, dtype, direction)` returns True for cells where Triton's measured p50 is > 1.67× CUDA's). README documents CC 8.0+ (Ampere+) requirement + `set_deterministic()` opt-in + Volta sm_70 / Turing sm_75 pinning recommendation (COMPAT-06). Covers: TEST-01, TEST-02, TEST-04, COMPAT-06.

**In scope:**
- **09-01:**
  - Extend `tests/conftest.py` `backend` fixture to `params=["torch", "triton", "cuda"]`. Add `_has_cuda_legacy_for_op(op_name)` probe in `torch_structured/_ops.py` mirroring `_has_any_triton_kernel()`. Per-op skip — tests for `diag_mult` skip the cuda param if `_diag_mult.so` is missing; tests for `butterfly_multiply` skip if `_butterfly.so` is missing; `hadamard_transform` if `_hadamard.so` is missing. Composable with Phase 8 SC#4 (per-test, not global).
  - New `tests/test_phase9_integration.py` (or extend existing files) with:
    - `test_pytest_under_triton` — runs as a smoke meta-test that re-invokes `pytest tests/` with `TORCH_STRUCTURED_BACKEND=triton` and asserts exit=0 (TEST-06). Or: gate this via a CI-level workflow that runs `TORCH_STRUCTURED_BACKEND=triton pytest tests/` and `tests/conftest.py` ensures each test that doesn't already parametrize via `backend` is at least covered by the env-var sweep.
    - `test_backend_agreement_butterfly` — for each (twiddle, input) shape in the Phase 7 dense smoke tier, assert `torch_structured._ops.butterfly_multiply(...)` produces equal outputs under all three backends within Phase 7 D-43a tolerances (fp32: rtol=1e-5/atol=1e-6; complex64: rtol=1e-3/atol=1e-4). Parametrized via the widened `backend` fixture.
    - `test_backend_agreement_diag_mult` and `test_backend_agreement_hadamard` — same pattern for the other two ported ops.
    - `test_checkpoint_v10_v11_roundtrip_butterfly` — synthesize a `Butterfly` state_dict in the v1.0/v1.1 layout `(nstacks, nblocks, log_n, n/2, 2, 2)`, save to a temp file, load via `Butterfly.load_state_dict` under `BACKEND=triton`, compare forward output to `BACKEND=cuda` (skip if `_butterfly.so` missing) within rtol=1e-5/atol=1e-6 for fp32. Covers COMPAT-02.
    - `test_make_linear_triton` — uses `make_linear(in_size=256, out_size=256, structure='butterfly')` and verifies forward + backward work under BACKEND=triton (COMPAT-03).
    - `test_lru_triton` — instantiate `LRU(hidden_size=64, ...)`, run a forward + backward step under BACKEND=triton, verify no errors and correct gradient shapes (COMPAT-03).
    - `test_public_api_unchanged` — assert `dir(torch_structured.Butterfly)`, `dir(torch_structured.ButterflyBmm)`, etc. include the v1.1 public attributes (signature regression detector for COMPAT-01).
- **09-02:**
  - `tests/test_torch_compile_triton.py` — new file. Tests:
    - `test_torch_compile_butterfly` — `torch.compile(Butterfly(in_size=256, ...))(x).sum().backward()` traces with `torch._dynamo.config.suppress_errors=False` and asserts no graph breaks via `torch._dynamo.utils.compile_metrics()` or `torch._dynamo.explain(model, x)` (D-67 specifics — D-67a planner chooses the exact API). Test under both `BACKEND=triton` and `BACKEND=torch`.
    - `test_torch_compile_butterfly_bmm` — same for `ButterflyBmm`.
    - `test_torch_compile_lru` — same for `LRU(hidden_size=64)`.
    - `test_torch_compile_make_linear` — same for `make_linear(in_size=256, out_size=256, structure='butterfly')`.
    - `test_torch_compile_no_fake_tensor_bug` — the 260419-p27 acceptance gate: invokes a Butterfly under FakeTensorMode tracing + asserts no "tensor has non-zero number of elements but data not allocated yet" error.
  - `tests/test_distributed_triton.py` — new file. Tests:
    - `test_ddp_butterfly` — single-process DDP smoke (uses `torch.distributed.init_process_group('gloo')` + DDP-wrap a model containing Butterfly + run loss.backward() + verify gradient sync). Skip if `torch.distributed.is_available() is False`.
    - `test_fsdp_butterfly_smoke` — marked `@pytest.mark.multigpu`. Requires `torchrun --nproc_per_node=2`. Wraps `Butterfly` with the FSDP `twiddle` no-shard hint (D-67b — exact API is `FullyShardedDataParallel(model, ignored_modules=[m for m in model.modules() if hasattr(m, 'twiddle')])` or equivalent; planner picks the right idiom for PyTorch 2.6+). Verifies forward + backward succeed on both ranks with identical loss values. Documents the `pytest -m multigpu` opt-in in conftest.
    - `test_gradient_checkpointing_butterfly` — wraps a Butterfly forward in `torch.utils.checkpoint.checkpoint(use_reentrant=False)` and verifies gradients match a non-checkpointed reference within rtol=1e-4/atol=1e-5 (looser than D-43a fp32 because checkpoint adds one recompute layer on top of Phase 8's trail recompute).
  - `tests/test_deterministic_mode.py` — new file. Tests:
    - `test_set_deterministic_api` — `torch_structured.set_deterministic(True); ...; torch_structured.set_deterministic(False)` toggles a process-level flag.
    - `test_deterministic_dtwiddle` — under `set_deterministic(True)`, two consecutive `loss.backward()` calls produce bit-identical d_twiddle (no atomicAdd reorder noise). Under `set_deterministic(False)` (default), d_twiddle differs across calls beyond machine epsilon at batch=4096 (proves the deterministic flag actually does something).
  - `torch_structured/__init__.py` exports `set_deterministic`. `torch_structured/_ops.py` adds module-level `_DETERMINISTIC` flag + `set_deterministic(value: bool) -> bool` setter (mirrors `set_backend()` shape).
  - Kernel-side deterministic path (in `torch_structured/_triton/butterfly/op.py` `_butterfly_backward_kernel` body): when `_DETERMINISTIC=True`, the kernel is JIT-specialized with a constexpr `DETERMINISTIC: tl.constexpr` that selects a single-warp sequential atomicAdd ordering (or a host-side reduction path). D-67c — planner picks the exact mechanism; recommend: gate at the wrapper level and route to a host-side `torch.zeros + per-program scatter-add via a separate kernel pass` when DETERMINISTIC=True. Keeps the kernel body unchanged; adds a Python-level reduce path for the deterministic flag.
- **09-03:**
  - `.planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json` already has 16 rows (Phases 7+8). Phase 9 does NOT regenerate; it CONSUMES.
  - New `tests/test_perf_grid.py` — automated TEST-04 gate. Reads `07-BASELINE.json`, for each row computes `triton_p50 / cuda_p50` (using `reference_torch_ref_p50` as a proxy when `_butterfly.so` is missing — but the gate is against `cuda_p50` from a separate column the executor adds). Asserts ratio ≤ 1.67 (= 1/0.60). Marks any failing cell for the runtime selector. `@pytest.mark.gpu_required`.
  - `torch_structured/_ops.py` — new function `_should_route_to_cuda(op_name: str, shape: Tuple, dtype: torch.dtype, direction: str) -> bool` (D-66). Reads a static routing table at module-load time from a JSON file at `torch_structured/_routing.json` (the runtime-accessible bake of the baseline gate result). When `_BACKEND='triton'` AND `_should_route_to_cuda(...) is True` AND `_has_cuda_legacy_for_op(op_name) is True`: route the call to the CUDA legacy path (with a one-time `logging.info` "torch_structured: routing {op}({shape}) to CUDA for perf parity (cell below 60% Triton/CUDA ratio)"). When CUDA legacy is missing, run on Triton anyway (no fallback to CUDA when CUDA isn't available — TRI-07 says torch oracle, not CUDA, is the runtime fallback).
  - New helper script `scripts/regenerate_routing_table.py` — reads `07-BASELINE.json`, writes `torch_structured/_routing.json` with shape: `{ "<op_name>": [{"log_n": int, "dtype": str, "direction": str, "route_to_cuda": bool, "triton_cuda_ratio_p50": float}, ...] }`. Run by the planner once at the end of 09-03; committed to the repo.
  - **Phase 8 SC#4 reconciliation (CRITICAL):** Phase 8's `test_butterfly_backward_no_cpp_symbol` asserts that under `BACKEND=triton`, no csrc/butterfly.cpp symbol is invoked. **The runtime selector breaks this assertion for shapes where it routes to CUDA.** Resolution: in 09-03, update Phase 8's SC#4 test to set `_DETERMINISTIC=False` and skip routing (e.g., add a `_DISABLE_ROUTING` flag for the test, OR exclude the test from running on log_n cells that the selector marks for CUDA routing). Document this in the test's docstring + a comment in `_should_route_to_cuda`.
  - `README.md` — add a "Triton backend" section documenting:
    - CC 8.0+ (Ampere+) is required for the default Triton path
    - `set_deterministic(True)` for reproducible d_twiddle
    - Volta sm_70 / Turing sm_75 users: pin to v1.1 (`pip install torch-structured==1.1.*`) OR use `TORCH_STRUCTURED_BACKEND=cuda` with a self-built `.so`
    - Brief table of which (log_n, dtype, direction) cells route to CUDA via the runtime selector (regenerated from `_routing.json`)
  - `CHANGELOG.md` — note v1.2 release with: Triton port complete; CUDA backend opt-in; minimum PyTorch bumped to 2.6.

**Out of scope:**
- **Phase 10 deprecation work** (DEPR-01..05) — Phase 10 owns `DeprecationWarning` on CUDA backend, `_flashmm` removal, csrc/ deletion gating.
- **bf16 / fp16** (TRI-FUT-01). Phase 9 is fp32 + complex64.
- **`log_n > 11`** — kernel works in principle; perf grid + correctness gate only exercise up to 11.
- **5-stage tile kernel** (deferred from Phase 7 D-40). Phase 9 uses **runtime selector** to route to CUDA for below-60% cells instead of building 5-stage (D-61).
- **`@triton.autotune`** — fixed `num_warps` schedule from Phase 7 D-40d carries.
- **ROCm / AMD / Intel XPU validation** — out of v1.2 scope (PLAT-01, PLAT-02).
- **Pre-compiled wheels** — out of v1.2 (DIST-01, DIST-02 deferred).
- **`@triton.heuristics`** — rejected by `wrap_triton` per Phase 4.
- **Touching `csrc/butterfly.cpp` / `butterfly_cuda.cu`** — Phase 10 deletion candidates. Phase 9 leaves them alone.
- **Editing `_setup_context`, `register_autograd` registration, `register_fake`** in `_triton/butterfly/op.py` — Phase 8 D-57 invariants preserved.
- **Modifying `_torch_ref/` files** — oracle stays untouched (TRI-07).
- **Adding new `_triton/` ops** — Phase 9 hardens the 3 existing ones (diag_mult, hadamard_transform, butterfly_multiply).
- **Dynamic per-call perf probing** — runtime selector is STATIC, baked from `07-BASELINE.json` at gate time (D-66). Dynamic probing rejected to keep first-call latency bounded.

</domain>

<decisions>
## Implementation Decisions

### Plan split — 3 plans by SC group (User choice, locked)

- **D-60:** **Three plans, sliced by SC group:**
  - **09-01 (Foundations — SC#1 + SC#3):** CUDA conftest axis + per-op skip-gate + end-to-end pytest under BACKEND=triton + checkpoint round-trip + make_linear/LRU smoke tests + public API regression detector. Covers TEST-03, TEST-06, COMPAT-01, COMPAT-02, COMPAT-03.
  - **09-02 (Compose — SC#2):** torch.compile + DDP + FSDP + gradient checkpointing + deterministic mode + `set_deterministic()` API. Covers COMPAT-04.
  - **09-03 (Perf + docs — SC#4 + SC#5):** Perf grid gate + runtime selector in `_ops.py` + README + CHANGELOG. Covers TEST-01, TEST-02, TEST-04, COMPAT-06.
- **D-60a:** Plans execute as `wave: 1` (09-01), `wave: 2` (09-02 depends on 09-01 for the cuda axis), `wave: 3` (09-03 depends on 09-01 + 09-02). The wave structure mirrors the natural data flow: foundations → compose → perf+docs. Plans within a wave are not parallelizable (each plan touches the conftest or _ops.py in ways the next plan needs to see).

### Perf gate fallback — runtime selector routes below-60% cells to CUDA (User choice, locked)

- **D-61:** **Below-60% perf cells route to CUDA via static runtime selector** (NOT 5-stage tile kernel work, NOT documented-shortfall-only). The selector logic is baked into `torch_structured/_ops.py` and consults a static routing table at `torch_structured/_routing.json` generated from `07-BASELINE.json` measured values.
- **D-61a:** **Tension with Phase 8 SC#4 explicitly acknowledged.** Phase 8 SC#4 asserts that under `BACKEND=triton`, no csrc/butterfly.cpp symbol is invoked. With the runtime selector active, BACKEND=triton CAN invoke csrc for shapes where the selector routes to CUDA. **Resolution (Phase 9 09-03 task):** the Phase 8 SC#4 test (`test_butterfly_backward_no_cpp_symbol`) is updated to:
  1. Use a small log_n cell (e.g., log_n=4 — well above the small-N fallback at log_n≤1, but below any cell where Triton trails CUDA in practice).
  2. Add a comment + docstring noting that the test asserts the **default routing** path is pure-Triton — shapes where the runtime selector intentionally routes to CUDA are an exception documented at `_routing.json`.
  3. The Phase 9 perf grid harness regenerates `_routing.json` after measurements; the SC#4 test reads `_routing.json` at test setup and asserts the small log_n cell it uses is NOT in the route-to-cuda list.
- **D-61b:** **Fallback chain when CUDA is unavailable.** `_should_route_to_cuda(op, shape, dtype, direction)` returns True for marked cells. When `_has_cuda_legacy_for_op(op)` is False (no `.so` built), the call FALLS BACK to Triton (does NOT fall back to `_torch_ref` — that would be a third hop with different perf characteristics). Documented in the `_should_route_to_cuda` docstring + a `logging.warning("torch_structured: selector wanted CUDA route for {op}({shape}) but _butterfly.so is missing; running on Triton (perf may be below 60% of CUDA target)")` on first miss.

### CUDA conftest skip-gate — per-op probe (User choice, locked)

- **D-62:** **Per-op cuda skip-gate**, mirroring Phase 6 D-39's per-op Triton skip pattern. Extend `tests/conftest.py` `backend` fixture to `params=["torch", "triton", "cuda"]`. The 'cuda' param skips per-test based on which op the test exercises:
  ```python
  @pytest.fixture(params=["torch", "triton", "cuda"])
  def backend(request):
      param = request.param
      if param == "triton" and not _ops._has_any_triton_kernel():
          pytest.skip("No Triton kernel installed")
      if param == "cuda":
          # Per-op skip — check via test's op_name marker or implicit detection
          op_name = _detect_op_from_test_name(request.node.name)  # or via marker
          if not _ops._has_cuda_legacy_for_op(op_name):
              pytest.skip(f"No CUDA legacy .so for {op_name}")
      original = _ops._BACKEND
      chosen = _ops.set_backend(param)
      yield chosen
      _ops.set_backend(original)
  ```
- **D-62a:** **New `_has_cuda_legacy_for_op(op_name: str) -> bool`** in `torch_structured/_ops.py`. Returns True iff the specific `_butterfly.so` / `_diag_mult.so` / `_hadamard.so` is importable. Granular probe (mirrors the per-op `_has_triton_kernel(op_name)` from Phase 5). Composable with the existing `_has_any_triton_kernel()` for downstream uses.
- **D-62b:** **Op-name detection mechanism in conftest:** Two options for connecting a test to an op name — (1) custom `@pytest.mark.op('butterfly_multiply')` markers per test, (2) parsing the test function name (e.g., `test_butterfly_*` → `'butterfly_multiply'`). Recommend (1) for explicit-is-better — markers are visible in test source and `pytest --collect-only`. Planner picks the exact mechanism; if (2) is simpler and unambiguous for the existing test names, use that.
- **D-62c:** **Test-time tolerances for backend agreement** (TEST-01 → TEST-03 verbatim): fp32 `rtol=1e-5, atol=1e-6`; complex64 `rtol=1e-3, atol=1e-4` for `d_twiddle` (matches Phase 8 D-52) and `rtol=1e-5, atol=1e-6` for `d_input` (matches Phase 8 layer (b)). All three backends must agree within these tolerances when the per-op skip-gate doesn't fire.

### Deterministic mode opt-in — `set_deterministic()` function (User choice, locked)

- **D-63:** **Top-level function API** — `torch_structured.set_deterministic(value: bool) -> bool`. Returns the previously-set value (for save/restore pattern, mirrors `set_backend()`). Stored as `_ops._DETERMINISTIC` module-level flag. Default `False`.
- **D-63a:** **Kernel-level deterministic specialization.** Phase 9 09-02 picks the mechanism. Recommended (executor's discretion): rather than threading a constexpr through `_butterfly_backward_kernel`, gate at the **wrapper level** in `_triton/butterfly/op.py` `_backward`. When `_DETERMINISTIC=True`, route through a sequential-atomicAdd path (e.g., issue per-program kernels and host-side reduce in deterministic order) OR route through `torch.autograd.grad(_butterfly_multiply_torch, ...)` — the torch oracle's autograd is deterministic by virtue of pure-PyTorch tensor ops. The latter is simpler but slower; the former is faster but more complex. Pick the simpler path (oracle fallback) unless perf testing shows it's untenable.
- **D-63b:** **Composition with `torch.use_deterministic_algorithms(True)`:** Honor BOTH `torch_structured._DETERMINISTIC` AND `torch.are_deterministic_algorithms_enabled()`. If either is True, deterministic path is active. The `set_deterministic()` function affects only torch_structured's flag; users opt into PyTorch's global deterministic mode separately.
- **D-63c:** **Export point.** `set_deterministic` exported via `torch_structured/__init__.py` `__all__`. Visible as `torch_structured.set_deterministic(...)`.

### FSDP test execution — torchrun in CI with @pytest.mark.multigpu (User choice, locked)

- **D-64:** **FSDP test runs via `torchrun --nproc_per_node=2`** inside a CI workflow that explicitly targets 2-GPU runners. Test marked `@pytest.mark.multigpu`. Skip in CI when GPU count < 2; the marker is documented in `tests/conftest.py` `pytest_configure` (mirroring the `slow` marker per Phase 7).
- **D-64a:** **CI workflow split.** Add a separate CI job (or matrix entry) that runs `torchrun --nproc_per_node=2 -m pytest tests/test_distributed_triton.py::test_fsdp_butterfly_smoke -v`. The standard test job runs `pytest tests/` (skips multigpu tests). The multigpu job uses the same `TRITON_CACHE_DIR` cache (TEST-05) so JIT cost doesn't compound.
- **D-64b:** **FSDP twiddle no-shard hint.** The exact API for telling FSDP not to shard the `twiddle` parameter is PyTorch-version-specific. Planner picks the right idiom for PyTorch 2.6+:
  - Recommended: `FullyShardedDataParallel(model, ignored_modules=[m for m in model.modules() if hasattr(m, 'twiddle')])`
  - Alternative: `wrap_policy` that excludes Butterfly modules
  - Document the choice in the test's docstring + reference the PyTorch FSDP docs.
- **D-64c:** **Single-process DDP smoke test as a complement.** Run a smaller `test_ddp_butterfly` that uses `torch.distributed.init_process_group(backend='gloo', init_method='env://', rank=0, world_size=1)` — exercises the DDP wrap path without needing multi-process. Verifies the autograd graph survives DDP's hooks without graph breaks. Composes naturally with `torch.compile`.

### Perf grid shape — match Phase 7+8 baseline (User choice, locked)

- **D-65:** **Reuse the 16-row `07-BASELINE.json` grid** as the perf gate axis. No new measurements in 09-03 beyond automating the ≥60% gate computation against the existing rows. Schema (from Phase 7 D-43b + Phase 8 D-50 extension):
  - log_n ∈ {8, 9, 10, 11} × dtype ∈ {fp32, complex64} × direction ∈ {forward, backward} = 16 rows
  - batch_size=64, nstacks=1, nblocks=1, increasing_stride=True, output_size=n
  - Each row has `kernel`, `dtype`, `direction`, `log_n`, `nstacks`, `nblocks`, `wall_ms_p50`, `wall_ms_p95`, `reference_torch_ref_p50`, `measured_at`, `gpu`
- **D-65a:** **CUDA p50 column.** The existing rows have `reference_torch_ref_p50` (the torch-oracle reference). For TEST-04 gate ≥60% of CUDA, the harness needs a **CUDA p50 measurement** — this is NOT yet in the JSON. **09-03 Task 1: extend `_baseline_butterfly_backward.py` and the forward baseline harness** to ALSO measure CUDA p50 when `_butterfly.so` is built. Add `reference_cuda_p50` column to all 16 rows. When CUDA is unbuilt, leave `reference_cuda_p50: null` and the gate becomes "Triton ≥ 60% of `reference_torch_ref_p50`" — a weaker gate but still measurable. Document this in the harness docstring.
- **D-65b:** **Gate computation:** `route_to_cuda = (wall_ms_p50 / reference_cuda_p50) > (1.0 / 0.60)`. Threshold 1.67× (= 1/0.6). Pass: ratio ≤ 1.67 (Triton is at least 60% of CUDA speed). Fail: ratio > 1.67 (Triton is below 60% — route to CUDA). When `reference_cuda_p50 is None`, the gate uses `reference_torch_ref_p50` and the threshold is loosened to 5.0× (Phase 7's documented torch-ref threshold).
- **D-65c:** **Wider grids deferred.** Per-batch grid (D-65 alternative) is out of scope — accepted that small-batch cells (batch=1, 16) are below 60% by design (launch overhead dominates) and the v1.2 user contract is "match CUDA at typical training batch sizes" (batch=64 is the canonical anchor).

### Runtime selector — `_ops.py` baked from baseline JSON (User choice, locked)

- **D-66:** **Static routing table baked from `07-BASELINE.json`** by `scripts/regenerate_routing_table.py`. Output: `torch_structured/_routing.json` (committed to the repo, version-controlled, regenerated when the baseline JSON changes).
- **D-66a:** **Selector function shape:**
  ```python
  def _should_route_to_cuda(op_name: str, shape: Tuple[int, ...], dtype: torch.dtype, direction: str) -> bool:
      """Consult the static routing table to decide if this call should route to CUDA.

      Returns True iff:
      - The (op_name, log_n, dtype, direction) cell has `route_to_cuda: true` in _routing.json
      - The selector is not disabled (e.g., for Phase 8 SC#4 test isolation)
      """
      if _ROUTING_DISABLED:
          return False
      key = (op_name, _shape_to_log_n(shape), str(dtype), direction)
      return _ROUTING_TABLE.get(key, False)
  ```
- **D-66b:** **Hook point in resolver.** The selector is consulted in `torch_structured/_ops.py` after the standard backend resolve: when `_BACKEND='triton'` and the per-op Triton path is bound, the resolver wraps the bound function with a thin shim that calls `_should_route_to_cuda(...)` per-call. If True, the call routes to the CUDA path (via `_has_cuda_legacy_for_op` probe). Adds one function-call overhead per op invocation — negligible.
- **D-66c:** **Test-time override.** Add `_DISABLE_ROUTING` module-level flag with a `set_routing_enabled(value: bool) -> bool` API (NOT exported as a top-level user API — only `torch_structured._ops.set_routing_enabled`). Phase 8 SC#4 test uses this to disable routing for the duration of the test, then restores it.
- **D-66d:** **Logging on first route.** First time a shape routes to CUDA in a process, emit `logging.info("torch_structured: routing %s(log_n=%d, dtype=%s, direction=%s) to CUDA (Triton/CUDA ratio %.2fx > 1.67x threshold)")` once. Tracked via a `_routing_log_emitted` set keyed on `(op_name, log_n, dtype, direction)`.

### Inherited from prior phases (NOT re-discussed — locked upstream)

- **D-67 (inherits Phase 4 D-04..D-08, D-11..D-16):** `TORCH_STRUCTURED_BACKEND` env var; `set_backend()` API; `triton_op` + `register_autograd` + `register_fake` + `wrap_triton` registration contract; the `_torch_ref/` peer package; the demonstrator-op pattern from Phase 4 (test_dispatch.py). Phase 9 doesn't touch these.
- **D-68 (inherits Phase 5 D-21..D-30 / Phase 6 D-39):** `_cuda_legacy/*.py` try-import + sentinel pattern; per-op asymmetric fallback; the `backend` fixture skip-gate widening pattern. Phase 9 extends the fixture pattern to a third axis (cuda) per D-62.
- **D-69 (inherits Phase 7 D-43a / Phase 8 D-43a):** Tiered test parametrization — dense smoke (every CI) + sparse comprehensive (`@pytest.mark.slow`). Phase 9 adds a third tier marker: `@pytest.mark.multigpu` for FSDP tests (D-64).
- **D-70 (inherits Phase 8 D-49..D-59):** Triton backward kernel + fp32 d_twiddle scratch + per-program reduce + atomic-add + small-N fallback + complex64 conjugate 4-FMA. Phase 9 doesn't modify the kernel; it adds the deterministic-mode WRAPPER-LEVEL flag (D-63a — gate at the `_backward` callback, not inside the kernel).
- **D-71 (inherits Phase 8 SC#4 / D-53):** The Phase 8 SC#4 dispatch-binding + monkey-patch shim test. Phase 9 updates this test for the runtime selector (D-61a) — adds `_DISABLE_ROUTING` flag during the test, asserts the chosen test cell (small log_n) is not in the route-to-cuda list.
- **D-72:** TRI-07 — `butterfly_multiply_torch` stays as runtime fallback for CPU / no-Triton environments. Phase 9 inherits and verifies via `test_pytest_under_triton` (TEST-06).

### Claude's Discretion

Areas where Claude (planner / executor) has flexibility:
- Exact form of `_detect_op_from_test_name(request.node.name)` in conftest — pytest marker (D-62b option 1) vs name-pattern (option 2). Recommend marker.
- Exact PyTorch FSDP ignored_modules / wrap_policy idiom (D-64b) — planner picks the right PyTorch 2.6+ API.
- Whether the deterministic path uses the oracle (D-63a alternative) or a sequential-atomicAdd path inside the Triton kernel. Recommend oracle (simpler, slower; perf is a secondary concern for the deterministic opt-in).
- Whether the perf-grid harness measures CUDA p50 in 09-03 OR reuses an existing CUDA bench from elsewhere. Recommend in-house CUDA measurement using the existing baseline scripts.
- Whether `scripts/regenerate_routing_table.py` is committed to repo or kept under `.planning/` (out of distribution). Recommend committed under `scripts/` since users may want to regenerate for their hardware.
- Whether the runtime selector logs on first miss to `logging.info` (D-66d) or `logging.warning` (D-61b alternative). Recommend `info` for the routing log (it's expected behavior); `warning` for the "wanted CUDA but `.so` is missing" log (it's a degraded perf state).
- Exact README section structure for the Triton backend documentation (D-60 — 09-03 task). Planner picks; suggested headings: "Triton backend (default)" / "Hardware requirements" / "Deterministic mode" / "Switching backends".
- Whether to add a CI matrix job for `BACKEND=cuda` testing OR rely on the existing `BACKEND=triton` job. Recommend a single new matrix entry that runs `TORCH_STRUCTURED_BACKEND=cuda pytest tests/test_*_triton.py -v` (tests the cuda path through the same parametrized backend fixture).
- Order of perf-gate failures in `tests/test_perf_grid.py` reporting (sorted by ratio? grouped by op?). Recommend sorted by descending ratio (worst-perf cells first).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 9 charter
- `.planning/ROADMAP.md` §"Phase 9" — phase goal, depends on Phase 8, 5 success criteria, 3 plan slots
- `.planning/REQUIREMENTS.md` §"v1.2 Requirements" → TEST-01, TEST-02, TEST-03, TEST-04, TEST-06, COMPAT-01, COMPAT-02, COMPAT-03, COMPAT-04, COMPAT-06 (all 10 REQs this phase covers)
- `.planning/REQUIREMENTS.md` §"Traceability" — confirms TEST-01..04, TEST-06, COMPAT-01..04, COMPAT-06 all mapped to Phase 9

### Phase 4 hand-off (LOCKED — dispatch + register_fake + demonstrator-op pattern)
- `.planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/04-CONTEXT.md` — D-04..D-08 (dispatch + `set_backend`); D-11..D-12 (torch>=2.6, triton_op + register_autograd + register_fake + wrap_triton); D-13..D-14 (demonstrator op + test_dispatch.py); D-15 (deprecation plan); D-16 (CI cache)
- `.planning/phases/04-triton-dispatch-infrastructure-foundational-decisions/04-DEPRECATION-PLAN.md` — Phase 10 implements; Phase 9 cuda-legacy fallback uses `log.warning` not `DeprecationWarning`
- `tests/test_dispatch.py` — Phase 4's demonstrator-op test. The 260419-p27 fake-tensor bug acceptance gate. Phase 9 09-02 tests the same bug end-to-end through Butterfly nn.Modules.

### Phase 5/6 hand-off (LOCKED — backend fixture pattern + per-op skip-gate)
- `.planning/phases/05-diag-mult-triton-port/05-CONTEXT.md` — D-21 (try-import + sentinel), D-22 (per-op asymmetric fallback), D-27/D-28 (test surface pattern)
- `.planning/phases/06-hadamard-triton-port/06-CONTEXT.md` — D-31..D-39: D-39 widened the skip-gate to `_has_any_triton_kernel()`. **Phase 9 extends this pattern with `_has_cuda_legacy_for_op(op_name)` per D-62a.**

### Phase 7 hand-off (LOCKED — tiered test surface + perf baseline schema)
- `.planning/phases/07-butterfly-multiply-forward-triton/07-CONTEXT.md` — D-40..D-48: D-43a tiered parametrization (dense smoke + sparse comprehensive). Phase 9 extends with `@pytest.mark.multigpu` tier per D-64.
- `.planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json` — **CRITICAL.** 16-row schema. Phase 9 09-03 extends with `reference_cuda_p50` column per D-65a. The perf gate (D-65b) reads this file.

### Phase 8 hand-off (LOCKED — most recent op port; SC#4 reconciliation)
- `.planning/phases/08-butterfly-multiply-backward-triton/08-CONTEXT.md` — D-49..D-59: Triton backward kernel + fp32 d_twiddle scratch + per-program reduce + atomic-add + small-N fallback + complex64 conjugate 4-FMA. Phase 9 doesn't modify the kernel; it adds the deterministic-mode wrapper-level flag per D-63a.
- `.planning/phases/08-butterfly-multiply-backward-triton/08-RESEARCH.md` §"SC#4 Verification Mechanism" — the dispatch-binding `is`-check + monkey-patch shim. **Phase 9 D-61a reconciles SC#4 with the new runtime selector** — small log_n cell, `_DISABLE_ROUTING` flag during test.
- `.planning/phases/08-butterfly-multiply-backward-triton/08-VERIFICATION.md` — Phase 8 verification report. Phase 9 builds on top of Phase 8's clean SUMMARY + verification baseline.
- `.planning/phases/08-butterfly-multiply-backward-triton/08-01-PLAN.md` and `08-02-PLAN.md` — Phase 8's plan structure (deep_work_rules, must_haves with truths, file budget). Phase 9 plans mirror this shape with the SC-group split per D-60.
- `.planning/phases/08-butterfly-multiply-backward-triton/08-01-SUMMARY.md` and `08-02-SUMMARY.md` — Phase 8 deliverable summaries. Phase 9's deterministic-mode wrapper (D-63a) hooks into the `_backward` body that Plan 08-01 wrote.

### Research outputs (milestone-wide — load-bearing for Phase 9)
- `.planning/research/PITFALLS.md` — Phase 9 09-02's `torch.compile` tests should account for the §1 complex64 layout pitfall (`view_as_real` boundary). The fake-tensor bug (260419-p27) acceptance gate is end-to-end through Butterfly nn.Modules.
- `.planning/research/STACK.md` — `@triton.jit` + `wrap_triton` + `register_autograd` + `register_fake` API contract. Phase 9 inherits.
- `.planning/research/ARCHITECTURE.md` — `_triton/<op>/op.py` + `_torch_ref/` + `_cuda_legacy/` layout pattern. Phase 9 09-03 adds `_routing.json` + `_should_route_to_cuda` to this stack.

### Project-level constraints
- `.planning/PROJECT.md` §"Current Milestone: v1.2" — `butterfly_multiply_torch` preserved as oracle + runtime fallback (TRI-07); v1.2 default backend = Triton (DEPR-01); minimum PyTorch = 2.6 (COMPAT-05). Phase 9 inherits.
- `./CLAUDE.md` (project root) — `assert` preconditions, no try/except in core lib (one exception: `_cuda_legacy/*.py` try-imports — documented honest-probe pattern from Phase 5 D-21). The runtime selector adds NO try/except — `_has_cuda_legacy_for_op` uses the existing try-import probe pattern.
- `/home/claroche/CLAUDE.md` (user-level) — `bd` for task tracking, NOT TaskCreate/TodoWrite.

### Code-level references (read before editing)
- `tests/conftest.py:1-50` — current `backend` fixture with `params=["torch", "triton"]`. Phase 9 09-01 widens to `["torch", "triton", "cuda"]` per D-62.
- `torch_structured/_ops.py:1-340` — current resolver. Phase 9 09-01 adds `_has_cuda_legacy_for_op` per D-62a; 09-02 adds `_DETERMINISTIC` flag + `set_deterministic` per D-63; 09-03 adds `_should_route_to_cuda` + `_ROUTING_TABLE` per D-66.
- `torch_structured/_ops.py:204-228` — `butterfly_multiply` resolver block. Phase 9 09-03 wraps this with the routing shim per D-66b. The existing resolve flow (Phase 5 D-22 per-op fallback) stays intact.
- `torch_structured/_torch_ref/butterfly.py:1-33` — the oracle. Phase 9 09-02 deterministic path may route through this when `_DETERMINISTIC=True` per D-63a.
- `torch_structured/_triton/butterfly/op.py` — Phase 8 deliverable. Phase 9 doesn't modify the kernel. The deterministic-mode wrapper-level gate (D-63a) lives in `_backward` callback body (Phase 8's deliverable), NOT in the kernel.
- `torch_structured/_cuda_legacy/butterfly.py` — already exists. Phase 9 doesn't modify; consumed by the runtime selector + per-op skip-gate.
- `torch_structured/__init__.py:1-46` — public API exports. Phase 9 09-02 adds `set_deterministic` to `__all__`.
- `torch_structured/factory.py:102+` — `make_linear`. Phase 9 09-01 verifies forward + backward work under BACKEND=triton.
- `torch_structured/recurrent/lru.py:1+` — `LRU`. Same.
- `tests/test_butterfly_triton.py` — Phase 7+8 test file. Phase 9 09-01 adds backend-agreement tests; 09-02 adds torch.compile + distributed tests in NEW files; 09-03 adds perf-grid test in a NEW file. The existing test file stays as-is.
- `tests/_baseline_butterfly.py` and `tests/_baseline_butterfly_backward.py` — Phase 7+8 perf harnesses. Phase 9 09-03 extends with `reference_cuda_p50` measurement per D-65a.
- `.planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json` — the perf baseline. Phase 9 09-03 extends with the CUDA column; the routing table generator reads from this.
- `README.md` — Phase 9 09-03 adds the "Triton backend" section per D-60 (COMPAT-06).
- `CHANGELOG.md` — Phase 9 09-03 documents v1.2 release.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`tests/conftest.py` `backend` fixture** — Phase 6 D-39's widened skip-gate pattern. Phase 9 D-62 extends to a third axis (cuda) using the same pattern.
- **`torch_structured/_ops.py` `_has_any_triton_kernel()` + per-op probes** — Phase 5 D-21 / Phase 6 D-39's try-import + sentinel pattern. Phase 9 D-62a clones the pattern for `_has_cuda_legacy_for_op(op_name)`.
- **`torch_structured/_ops.py` resolver** — already routes `BACKEND=triton` per Phase 5 D-22 asymmetric fallback. Phase 9 D-66b wraps the bound function with a thin selector shim — minimal surface change.
- **`07-BASELINE.json` (16 rows)** — Phase 7+8 deliverable. Phase 9 D-65 reuses verbatim, D-65a extends with CUDA p50 column.
- **`tests/_baseline_butterfly.py` + `tests/_baseline_butterfly_backward.py`** — Phase 7+8 perf harnesses. Phase 9 D-65a extends to also measure CUDA p50.
- **`set_backend()` API** — Phase 4 D-04. Phase 9 D-63 mirrors the shape for `set_deterministic()`.
- **`@pytest.mark.slow` marker** — Phase 7 D-43a. Phase 9 D-64 adds `@pytest.mark.multigpu` following the same registration pattern in `pytest_configure`.

### Established Patterns
- **`assert` preconditions, no try/except in core lib** (CLAUDE.md). Exception: `_cuda_legacy/*.py` try-imports (D-21 honest-probe pattern). Phase 9 uses the same pattern for the runtime selector and `_has_cuda_legacy_for_op`.
- **Per-op skip-gate widening** (Phase 6 D-39) — Phase 9 D-62 inherits and extends with cuda axis.
- **D-05 attribute access** — consumer modules call `torch_structured._ops.<op>` (not `from torch_structured._ops import <op>`) so `set_backend()` rebinds. Phase 9 D-66b's selector hooks into the same resolver — consumers don't change.
- **`tl.atomic_add(sem='relaxed')` is fp32-only into the scratch buffer** (Phase 8 SC#3). Phase 9 doesn't touch the kernel's atomic-add path. The deterministic-mode wrapper-level gate (D-63a) bypasses the kernel entirely when active.
- **`register_fake` is load-bearing for `torch.compile`** (Phase 4 D-12 + 260419-p27). Phase 9 09-02 tests this assumption end-to-end through `Butterfly` nn.Modules.

### Integration Points
- **`torch_structured.Butterfly`, `ButterflyBmm`, `ButterflyUnitary`, `ButterflyBase4` nn.Modules** — call `torch_structured._ops.butterfly_multiply` via D-05 attribute access. Phase 9 09-02's `torch.compile` tests trace through these surfaces. Phase 9 09-03's runtime selector intercepts calls at the `_ops.py` resolver level — consumers don't see the selector.
- **`torch_structured.make_linear` + `torch_structured.LRU`** — composite factories that build models containing `Butterfly`. Phase 9 09-01 + 09-02 verify they continue to work unchanged (COMPAT-01, COMPAT-03).
- **`tests/test_butterfly_triton.py`** — Phase 7+8's main test file. Phase 9 ADDS new test files (`test_phase9_integration.py`, `test_torch_compile_triton.py`, `test_distributed_triton.py`, `test_deterministic_mode.py`, `test_perf_grid.py`) — does NOT modify the existing file (except updating the Phase 8 SC#4 test for runtime selector reconciliation per D-61a).
- **`07-BASELINE.json`** — Phase 9 09-03 extends with `reference_cuda_p50` column. The routing table generator + the perf gate test both read this file. The file stays at its Phase 7 path (not moved).
- **`README.md` + `CHANGELOG.md`** — Phase 9 09-03 adds the Triton-backend section + v1.2 release notes. Verify nothing breaks rendering on GitHub.

</code_context>

<specifics>
## Specific Ideas

- **The runtime selector's table generation is a one-time bake.** `scripts/regenerate_routing_table.py` reads `07-BASELINE.json`, computes `triton_p50 / cuda_p50` per row, and writes `_routing.json` with the `route_to_cuda` field per cell. The script is committed; users can re-run on their hardware to get a hardware-specific routing table. Phase 9 09-03 commits the dev-host bake; users who care about parity on different hardware re-run the script.

- **The conftest backend fixture connects test → op name via marker (D-62b).** Concrete pattern (illustrative — planner picks the form):
  ```python
  # tests/test_butterfly_triton.py
  @pytest.mark.op('butterfly_multiply')
  def test_butterfly_eager_fp32(backend, ...):
      ...
  # tests/conftest.py
  def backend(request):
      param = request.param
      op_marker = request.node.get_closest_marker('op')
      op_name = op_marker.args[0] if op_marker else None
      if param == 'cuda' and op_name and not _ops._has_cuda_legacy_for_op(op_name):
          pytest.skip(f"No CUDA legacy .so for {op_name}")
      ...
  ```

- **`set_deterministic()` deterministic path via oracle fallback (D-63a recommended).** Concrete pattern (illustrative):
  ```python
  # _triton/butterfly/op.py _backward
  def _backward(ctx, grad_out):
      if torch_structured._ops._DETERMINISTIC or torch.are_deterministic_algorithms_enabled():
          twiddle, input_ = ctx.saved_tensors
          twiddle_d = twiddle.detach().requires_grad_(True)
          input_d = input_.detach().requires_grad_(True)
          with torch.enable_grad():
              out = _butterfly_multiply_torch(twiddle_d, input_d, ctx.increasing_stride, ctx.output_size)
          gt, gi = torch.autograd.grad(out, [twiddle_d, input_d], grad_out)
          return gt, gi, None, None
      # ... existing Phase 8 trail-recompute + atomicAdd path ...
  ```
  Pro: simple, exercises the well-tested oracle; deterministic by virtue of pure-PyTorch tensor ops. Con: slow at log_n=11 (the trail-recompute path is faster). The user opts in knowing this trade-off; documented in README per D-60.

- **The 260419-p27 acceptance gate end-to-end.** Phase 4's `test_dispatch.py` already covers the demonstrator op. Phase 9 09-02 adds:
  ```python
  def test_torch_compile_no_fake_tensor_bug(backend):
      """End-to-end 260419-p27 gate: Butterfly nn.Module traces through torch.compile without the fake-tensor bug."""
      m = Butterfly(in_size=16, out_size=16, bias=False).cuda()
      m_compiled = torch.compile(m, fullgraph=True)  # fullgraph=True asserts no graph breaks
      x = torch.randn(8, 16, device='cuda', requires_grad=True)
      y = m_compiled(x)
      loss = y.sum()
      loss.backward()
      assert x.grad is not None
      # 260419-p27 bug would raise: "tensor has non-zero number of elements but data not allocated yet"
  ```

- **FSDP no-shard for twiddle (D-64b).** Concrete pattern (illustrative — planner verifies the right PyTorch 2.6+ API):
  ```python
  # tests/test_distributed_triton.py
  @pytest.mark.multigpu
  def test_fsdp_butterfly_smoke():
      """2-GPU FSDP smoke. Run via: torchrun --nproc_per_node=2 -m pytest -m multigpu ...

      The twiddle parameter is excluded from sharding because Butterfly's per-rank twiddle
      access pattern requires the full tensor (not a sharded slice). The ignored_modules
      hint tells FSDP to keep Butterfly params replicated on each rank.
      """
      from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
      torch.distributed.init_process_group('nccl')
      m = nn.Sequential(Butterfly(256, 256, bias=False), nn.Linear(256, 10)).cuda()
      m = FSDP(m, ignored_modules=[mod for mod in m.modules() if isinstance(mod, Butterfly)])
      x = torch.randn(8, 256, device='cuda', requires_grad=True)
      loss = m(x).sum()
      loss.backward()
      # Verify gradient sync — all ranks see the same gradient norm
      grad_norm = sum(p.grad.norm() for p in m.parameters() if p.grad is not None)
      ...
  ```

- **Perf gate test pattern (illustrative):**
  ```python
  # tests/test_perf_grid.py
  @pytest.mark.gpu_required
  def test_perf_gate_triton_at_60pct_cuda():
      """TEST-04: Triton >=60% of CUDA on every (log_n, dtype, direction) cell at batch=64."""
      baseline = json.loads(Path('.planning/phases/07-butterfly-multiply-forward-triton/07-BASELINE.json').read_text())
      failures = []
      for row in baseline['rows']:
          if row.get('reference_cuda_p50') is None:
              continue  # Skip cells without CUDA measurement
          ratio = row['wall_ms_p50'] / row['reference_cuda_p50']
          if ratio > 1.67:  # = 1/0.60
              failures.append((row['op'], row['log_n'], row['dtype'], row['direction'], ratio))
      assert not failures, f"Below 60% of CUDA on cells: {failures}"
  ```

- **Plan structure transcription from Phase 8.** Plan 09-01 transcribes Phase 8 08-01's task structure shape (conftest + new test file + new helper function). Plan 09-02 transcribes 08-02's shape (gate removal + branch fill-in + extension). Plan 09-03 transcribes 08-02's perf-baseline extension shape, plus README + CHANGELOG tasks.

- **Phase 8 SC#4 reconciliation in 09-03.** The Phase 8 test `test_butterfly_backward_no_cpp_symbol` currently runs at `log_n=4` (small enough that Triton is faster than CUDA, so the selector won't route). Phase 9 09-03:
  1. Verifies the chosen test cell (log_n=4) is NOT in `_routing.json`'s route-to-cuda list.
  2. Adds a comment to the Phase 8 test referencing `_routing.json` + the `_DISABLE_ROUTING` flag for tests that need to assert pure-Triton execution.
  3. Adds a NEW test `test_runtime_selector_routes_below_60_pct` that explicitly invokes a routed shape and asserts the CUDA path was taken (positive case for the selector).

</specifics>

<deferred>
## Deferred Ideas

- **5-stage tile kernel** (Phase 7 D-40 explicit, Phase 9 D-61 explicit). Phase 9 uses runtime selector to route below-60% cells to CUDA instead. If a future milestone wants to ELIMINATE the CUDA dependency entirely, the 5-stage variant becomes that milestone's scope.
- **Wider perf grid (per-batch axis)** (D-65c). Accepted that small-batch cells (batch=1, 16) are below 60% by design; v1.2 anchors on batch=64.
- **`@triton.autotune`** (Phase 7 D-40d). Phase 9 uses the fixed num_warps schedule from Phase 7.
- **Dynamic runtime perf probing** (D-66 alternative). Phase 9 uses static routing table; dynamic probe adds first-call latency + non-determinism.
- **bf16 / fp16** (TRI-FUT-01). Phase 9 is fp32 + complex64.
- **`log_n > 11`** — kernel works in principle; test surface and perf grid only exercise up to 11.
- **ROCm / AMD / Intel XPU validation** (PLAT-01, PLAT-02). Out of v1.2 scope.
- **Pre-compiled wheels** (DIST-01, DIST-02). Out of v1.2.
- **Deterministic-mode kernel-side specialization** (D-63a alternative). Phase 9 uses wrapper-level gate + oracle fallback for simplicity; kernel-side `DETERMINISTIC: tl.constexpr` deferred to a future perf-optimization milestone.
- **Stack-trace-based SC#4 verification** (Phase 8 alternative). Phase 9 inherits Phase 8's dispatch-binding + monkey-patch shim mechanism per D-71.
- **CHANGELOG migration guide for users on v1.0/v1.1.** Phase 10 owns the deprecation cadence; Phase 9 only documents the v1.2 NEW state, not the migration steps from older versions.

### Reviewed Todos (not folded)
None — no pending todos surfaced for Phase 9.

</deferred>

---

*Phase: 9-Integration Hardening & Correctness Gates*
*Context gathered: 2026-05-28*
