---
phase: 06-hadamard-triton-port
plan: 02
subsystem: kernels
tags: [triton, hadamard, autograd, pytorch, rank-n, shape-handling]

# Dependency graph
requires:
  - phase: 06-hadamard-triton-port
    provides: "Triton hadamard kernel + _torch_ref oracle + register_autograd backward delegate (Plan 06-01); the wrapper docstring at _triton/hadamard_transform/op.py:104 advertising `(*batch, n)` shape — this gap closure makes that contract truthful end-to-end."
provides:
  - "Rank-N-correct `_torch_ref.hadamard.hadamard_transform_torch(u, normalize=False)` accepting `u.shape == (..., n)` for any rank `>= 1`."
  - "Both backends (`torch` and `triton`) now honor the `(*batch, n)` shape contract end-to-end for forward AND backward (the Triton backward at `_triton/hadamard_transform/op.py:168` delegates through `_torch_ref`, so this fix lights up both backends simultaneously)."
  - "Three new rank-3 regression tests parametrized over the `backend` fixture (`test_hadamard_eager_fp32_rank3`, `test_hadamard_backward_rank3`, `test_hadamard_self_inverse_rank3`) — 6 new PASS rows."
affects: [phase-07, phase-09, phase-10]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Reshape-and-restore for rank-N support in pure-PyTorch reference kernels: capture `original_shape`, reshape to `(-1, n)` for the inner loop, restore the original shape on return. Zero-allocation on contiguous inputs (the reshape is a view); preserves verbatim-relocation lineage of the inner loop body."

key-files:
  created: []
  modified:
    - torch_structured/_torch_ref/hadamard.py
    - tests/structured/test_hadamard_triton.py

key-decisions:
  - "Approach (a) reshape-and-restore (per verifier recommendation in 06-VERIFICATION.md). 4-line patch around the existing rank-2 loop body; preserves the D-35a `np.log2` invariant and the verbatim-relocation lineage from `structured/hadamard.py:15-30` documented in 06-01-SUMMARY.md key-decisions. Approach (b) — rewriting the inner loop to slice along the last dim directly — was rejected as more diff and more accidental-bug surface for no behavioral gain."
  - "Kept the unused local `batch_size = u.shape[0]` after the reshape to document the rank-2-after-reshape contract and minimize diff churn. Removing it would not change behavior but would obscure the post-reshape invariant."
  - "Did NOT touch `_triton/hadamard_transform/op.py` — its forward path at line 127 (`n_batch = u.numel() // n`) was already rank-N-correct; its backward at line 168 delegates to `_torch_ref` so the single-file fix at `_torch_ref/hadamard.py:41` lights up both backends. Docstring at line 104 becomes truthful end-to-end with no edit."
  - "Did NOT touch `_cuda_legacy/hadamard.py` (CR-02) per VERIFICATION.md DEFERRED — the .so is not built on this host; Phase 9 extends the `backend` fixture to {triton, cuda, torch} and Phase 10 covers cuda deprecation."

patterns-established:
  - "Rank-N pure-PyTorch reference oracle pattern: when a reference function lives at a backward-delegate seam (per D-32), its shape contract MUST be at least as wide as the wrapper's advertised contract. Narrowing at the oracle silently narrows the backward path for every backend that delegates through it."

requirements-completed: [TRI-02]

# Metrics
duration: 4min
completed: 2026-05-27
---

# Phase 6 Plan 02: Rank-N gap closure for hadamard_transform Summary

**Single-line shape-unpack widening in `_torch_ref/hadamard.py:41` (approach (a) reshape-and-restore) plus three rank-3 regression tests close the SC#2 strict-reading gap from 06-VERIFICATION.md; both backends now honor `(*batch, n)` forward + backward.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-05-27T17:16:15Z
- **Completed:** 2026-05-27T17:20:20Z
- **Tasks:** 1 (TDD cycle: RED + GREEN)
- **Files modified:** 2 (`torch_structured/_torch_ref/hadamard.py`, `tests/structured/test_hadamard_triton.py`)

## Accomplishments

- **Closed CR-01 + SC#2 strict-reading gap.** The line `batch_size, n = u.shape` at `_torch_ref/hadamard.py:41` is gone; replaced by 4 lines that capture `original_shape`, extract `n = u.shape[-1]`, reshape to `(-1, n)` for the existing rank-2 loop body, and restore the original shape on return.
- **Both VERIFICATION.md FAIL behavioral spot-check rows now PASS.** `torch_structured._ops.hadamard_transform(torch.randn(2,3,8, requires_grad=True)).sum().backward()` succeeds with `grad.shape == (2, 3, 8)` on the triton backend; `set_backend('torch')` + rank-3 forward succeeds.
- **Three new regression tests added** to `tests/structured/test_hadamard_triton.py`, each parametrized over the `backend` fixture (torch + triton); 6 new PASS rows. `test_hadamard_eager_fp32_rank3` cross-checks the rank-3 forward vs both the rank-2 reshape baseline and the `_torch_ref` oracle. `test_hadamard_backward_rank3` exercises `torch.autograd.grad` on the rank-3 input (the literal failing command from VERIFICATION.md). `test_hadamard_self_inverse_rank3` verifies `H ∘ H = N * I` (unnormalized) and `= I` (normalized) on a `(3, 2, 16)` input.
- **Wrapper docstring at `_triton/hadamard_transform/op.py:104` becomes truthful with no wrapper edit.** Approach (a) is end-to-end correct: forward path at op.py:127 was already rank-N-correct via `n_batch = u.numel() // n`; backward path at op.py:168 lights up the moment its `_torch_ref` delegate is widened.
- **Zero regression.** All existing Phase 6 / back-compat / Phase 5 test profiles preserved or improved.

## Task Commits

TDD cycle for Task 1:

1. **Task 1 RED — failing rank-3 regression tests** — `900838e` (test)
2. **Task 1 GREEN — widen `_torch_ref/hadamard.py` to rank-N** — `960857c` (fix)

_(No REFACTOR commit — the 4-line patch is already minimal.)_

## Files Created/Modified

- `torch_structured/_torch_ref/hadamard.py` — Widened `hadamard_transform_torch` to accept any rank `>= 1` whose last dim is a power of 2. The interleaved-butterfly loop body and the `np.log2` math are preserved verbatim (D-35a / 06-01-SUMMARY.md key-decision lineage intact); only the shape-unpack widens via approach (a) reshape-and-restore. Docstring at lines 36, 39 updated from `(batch_size, n)` to `(..., n)`. Inline comment documents the gap-closure rationale + the link to `_triton/hadamard_transform/op.py:104`.
- `tests/structured/test_hadamard_triton.py` — Appended three new test functions (`test_hadamard_eager_fp32_rank3`, `test_hadamard_backward_rank3`, `test_hadamard_self_inverse_rank3`) at the end of the file. The existing 5 test functions are byte-identical so the regression evidence in 06-VERIFICATION.md remains directly comparable.

### Exact diff: `torch_structured/_torch_ref/hadamard.py`

```diff
 def hadamard_transform_torch(u, normalize=False):
     """Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
     n must be a power of 2.
     Parameters:
-        u: Tensor of shape (batch_size, n)
+        u: Tensor of shape (..., n) where n is a power of 2
         normalize: if True, divide the result by 2^{m/2} where m = log_2(n).
     Returns:
-        product: Tensor of shape (batch_size, n)
+        product: Tensor of shape (..., n) — same shape as input
     """
-    batch_size, n = u.shape
+    # Rank-N handling per 06-VERIFICATION.md gap closure (SC#2 strict reading):
+    # the wrapper at _triton/hadamard_transform/op.py:104 advertises (*batch, n).
+    # The interleaved-butterfly loop body uses `...` indexing and is rank-N-correct
+    # already; reshape to rank-2 minimizes diff vs the verbatim-relocation lineage
+    # from structured/hadamard.py:15-30 (preserves np.log2 + torch.cat semantics).
+    original_shape = u.shape
+    n = u.shape[-1]
+    u = u.reshape(-1, n)
+    batch_size = u.shape[0]
     m = int(np.log2(n))
     assert n == 1 << m, 'n must be a power of 2'
     x = u[..., np.newaxis]
     for d in range(m)[::-1]:
         x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
-    return x.squeeze(-2) / 2**(m / 2) if normalize else x.squeeze(-2)
+    out = x.squeeze(-2) / 2**(m / 2) if normalize else x.squeeze(-2)
+    return out.reshape(original_shape)
```

13 insertions + 4 deletions; the inner butterfly loop body is untouched (verbatim-relocation lineage from `structured/hadamard.py:15-30` preserved).

### Pytest evidence

**New tests — 6 new PASS rows (3 tests × 2 backend params):**

```
tests/structured/test_hadamard_triton.py::test_hadamard_eager_fp32_rank3[torch] PASSED
tests/structured/test_hadamard_triton.py::test_hadamard_eager_fp32_rank3[triton] PASSED
tests/structured/test_hadamard_triton.py::test_hadamard_backward_rank3[torch] PASSED
tests/structured/test_hadamard_triton.py::test_hadamard_backward_rank3[triton] PASSED
tests/structured/test_hadamard_triton.py::test_hadamard_self_inverse_rank3[torch] PASSED
tests/structured/test_hadamard_triton.py::test_hadamard_self_inverse_rank3[triton] PASSED
======================== 6 passed, 4 warnings in 0.93s =========================
```

**Phase 6 dedicated suite — improved profile (was 31 PASS + 1 SKIP, now 37 PASS + 1 SKIP):**

```
pytest tests/structured/test_hadamard_triton.py -v
================== 37 passed, 1 skipped, 4 warnings in 1.17s ===================
```

**Back-compat regression — unchanged baseline (6 PASS + 1 SKIP):**

```
pytest tests/structured/test_hadamard.py tests/structured/test_imports.py -v
=================== 6 passed, 1 skipped, 4 warnings in 0.79s ===================
```

**Phase 5 regression — unchanged baseline (29 PASS):**

```
pytest tests/test_dispatch.py tests/test_diag_mult.py -v
======================== 29 passed, 4 warnings in 1.59s ========================
```

**Cross-suite gate — improved profile (was 37 PASS + 2 SKIP, now 43 PASS + 2 SKIP, 0 FAIL):**

```
TORCH_STRUCTURED_BACKEND=triton pytest tests/structured/ -v
================== 43 passed, 2 skipped, 4 warnings in 2.10s ===================
```

**Direct VERIFICATION.md behavioral spot-check repro — both previously-FAIL rows now PASS:**

```
$ python -c "import torch; import torch_structured; u = torch.randn(2,3,8,device='cuda',requires_grad=True,dtype=torch.float32); torch_structured._ops.hadamard_transform(u).sum().backward(); assert u.grad.shape == (2,3,8); print('PASS', u.grad.shape)"
PASS torch.Size([2, 3, 8])

$ python -c "import torch; import torch_structured; torch_structured._ops.set_backend('torch'); out = torch_structured._ops.hadamard_transform(torch.randn(2,3,8)); assert out.shape == (2,3,8); print('PASS', out.shape)"
PASS torch.Size([2, 3, 8])
```

## Gap Closure Status

| Item | Before (06-VERIFICATION.md) | After (this plan) |
|------|----------------------------|-------------------|
| Truth #2 (SC#2 self-inverse on any input shape) | **PARTIAL** — rank-2 only; rank-3+ FAILs backward on both backends | **VERIFIED** — rank-N forward + backward on both backends; numerical parity with rank-2 reshape baseline |
| Truth #13 (Plan SC#2 restated) | **PARTIAL** — same as Truth #2 | **VERIFIED** |
| CR-01 (`_torch_ref/hadamard.py:41` rank-2-only) | **OPEN** — blocker under strict reading | **CLOSED** — line widened to `n = u.shape[-1]` + reshape-and-restore |
| CR-02 (`_cuda_legacy/hadamard.py` signature mismatch with `normalize`) | **DEFERRED** — not exercised on this host (.so absent) | **DEFERRED** — Phase 9 / Phase 10 (per VERIFICATION.md frontmatter) |
| Wrapper docstring `(*batch, n)` at `_triton/.../op.py:104` | Load-bearing but misleading — forward correct on triton, backward crashed on both | Truthful end-to-end with NO change to wrapper file |
| Behavioral spot-check: "Rank-3 backward succeeds" | **FAIL** (`ValueError: too many values to unpack (expected 2)`) | **PASS** |
| Behavioral spot-check: "Rank-3 forward on torch backend" | **FAIL** (same ValueError) | **PASS** |
| TRI-02 requirement | PARTIALLY SATISFIED | SATISFIED (gap-closure confirmation; originally completed by 06-01) |

## Decisions Made

- **Approach (a) reshape-and-restore over approach (b) loop-body rewrite.** Approach (a) minimizes diff (13 inserted + 4 deleted lines vs a rewrite of the inner butterfly loop), preserves the verbatim-relocation lineage from `structured/hadamard.py:15-30` (the 06-01-SUMMARY.md key-decision "uses `np.log2` (not `bit_length`) to preserve numerical parity" stays load-bearing — only the shape-unpack widens; the math is untouched), and has zero accidental-bug surface. The existing loop body's `x[..., ::2, :]` indexing already uses `...` so it is rank-N-correct as-is.
- **Kept the unused `batch_size = u.shape[0]` local after the reshape.** Removing it would not affect behavior but would obscure the post-reshape rank-2 invariant; keeping it documents the contract and minimizes diff churn.
- **Did not modify `_triton/hadamard_transform/op.py`.** Verifier confirmed the forward path at line 127 (`n_batch = u.numel() // n`) was already rank-N-correct; the backward at line 168 delegates through `_torch_ref` so the single-file fix at `_torch_ref/hadamard.py:41` lights up both backends simultaneously. The wrapper docstring at line 104 becomes truthful with no edit.
- **Did not modify `_cuda_legacy/hadamard.py`.** CR-02 stays DEFERRED per VERIFICATION.md frontmatter. The cuda branch is unreachable on this host (`_hadamard_cuda.so` not built). Phase 9 extends the `backend` fixture to `{triton, cuda, torch}` and Phase 10 covers cuda deprecation.

## Deviations from Plan

None — plan executed exactly as written. The TDD cycle followed the plan's `tdd="true"` directive (RED + GREEN commits). All 14 verify commands in the plan's `<verify>` block produced the expected output. All 5 source-level grep gates from the plan's `acceptance_criteria` passed:

- `grep -c "n = u.shape\[-1\]" torch_structured/_torch_ref/hadamard.py` → 1 (>= 1 required)
- `grep -v '^#' torch_structured/_torch_ref/hadamard.py | grep -c "batch_size, n = u.shape"` → 0 (0 required)
- `grep -c "u.reshape(-1, n)" torch_structured/_torch_ref/hadamard.py` → 1 (>= 1 required)
- `grep -c "out.reshape(original_shape)" torch_structured/_torch_ref/hadamard.py` → 1 (>= 1 required)
- `grep -cE "test_hadamard_eager_fp32_rank3|test_hadamard_backward_rank3|test_hadamard_self_inverse_rank3" tests/structured/test_hadamard_triton.py` → 3 (>= 3 required)

## TDD Gate Compliance

The plan has `tdd="true"` on the single task. The git log shows the canonical TDD cycle:

1. **RED gate:** `900838e test(06-02): RED — rank-3+ regression tests reproduce VERIFICATION gap` — confirmed 5 of 6 new test cases FAIL with the expected `ValueError: too many values to unpack (expected 2)` originating at `_torch_ref/hadamard.py:41` (one case — `test_hadamard_self_inverse_rank3[triton]` — passed RED because Triton forward already handled rank-N via `u.numel() // n` and that test does no autograd).
2. **GREEN gate:** `960857c fix(06-02): GREEN — widen _torch_ref/hadamard.py to rank-N` — all 6 new test cases now PASS on both backends; all regression suites unchanged.

REFACTOR gate skipped — the 4-line patch is already minimal and the inner loop body is untouched (verbatim-relocation invariant preserved per 06-01-SUMMARY.md).

## Issues Encountered

- **Editable install pointed at the main repo, not the worktree.** The pip-installed `torch_structured` (editable mode) hardcodes the main repo path via `__editable___torch_structured_0_4_0_finder.py`, so pytest from the worktree loaded code from `/home/claroche/torch-structured/torch_structured/_torch_ref/hadamard.py` instead of the worktree's copy. Resolved by exporting `PYTHONPATH=/home/claroche/torch-structured/.claude/worktrees/agent-a2c09cba89987810b:$PYTHONPATH` before invoking pytest — `PYTHONPATH` takes precedence over `.pth`-based editable finders, so the worktree's code became authoritative. This is a worktree-mechanics issue, not a plan-content issue; the fix did not require changing any code under test. The pre-built `_butterfly.so` / `_version.so` were copied from the main repo into the worktree to satisfy the extension-load step in `torch_structured/__init__.py`.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Phase 6 closure is complete under the strict SC#2 reading. Truth #2 and Truth #13 in `06-VERIFICATION.md` upgrade from PARTIAL to VERIFIED; the wrapper docstring at `_triton/hadamard_transform/op.py:104` is now truthful end-to-end.
- TRI-02 is SATISFIED. Phase 6 is the milestone owner.
- Phase 7+ consumers reading the `(*batch, n)` wording in the wrapper docstring can write rank-N code with confidence; the resolver-bound `_ops.hadamard_transform` honors the contract on both backends.
- **Deferred to Phase 9 / Phase 10:** CR-02 (`_cuda_legacy/hadamard.py` signature mismatch with `normalize`). The cuda branch is unreachable on this host; Phase 9 extends the `backend` fixture to `{triton, cuda, torch}` and will surface it then.
- **Deferred to Phase 7 cleanup or `/gsd-code-review --fix`:** the 4 WARNING and 3 INFO findings from `06-REVIEW.md` (WR-01..WR-04, IN-01..IN-03). WR-01 specifically is closed by this plan (the wrapper docstring promise of `(*batch, n)` is now truthful).

## Self-Check: PASSED

- `torch_structured/_torch_ref/hadamard.py` — FOUND (modified, 13 insertions + 4 deletions)
- `tests/structured/test_hadamard_triton.py` — FOUND (modified, 87 insertions)
- `.planning/phases/06-hadamard-triton-port/06-02-SUMMARY.md` — FOUND (created)
- Commit `900838e` (RED — test) — FOUND in git log
- Commit `960857c` (GREEN — fix) — FOUND in git log

---
*Phase: 06-hadamard-triton-port*
*Plan: 02*
*Completed: 2026-05-27*
