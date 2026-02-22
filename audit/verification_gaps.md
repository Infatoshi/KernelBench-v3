# Verification Gaps Audit

## Current Verification Implemented

From `modal_eval.py`:
- Deterministic correctness seeds: `[42, 123, 456, 789, 1337]`.
- Benchmark seed: `2026`.
- Correctness uses fixed tolerance: `atol=0.05`, `rtol=0.02`.
- Correctness check is `max_abs_diff <= atol + rtol * max_abs(ref)`.
- Shape mismatch is explicitly rejected.
- Runtime timing uses CUDA events with warmup/timed iterations and median reporting.
- Kernel-count comparison is collected via `torch.profiler`.
- CUDA synchronization exists around profiling and timing paths.

## Metadata Coverage Gaps

- Total problems audited: `59`.
- Full metadata (`OP_TYPE`, `SUPPORTED_PRECISIONS`, `HARDWARE_REQUIRED`) present: `59/59`.
- Missing `OP_TYPE`: `0`.
- Missing `SUPPORTED_PRECISIONS`: `0`.
- Missing `HARDWARE_REQUIRED`: `0`.
- Missing `SYNC_REQUIRED`: `59`.
- Missing `VERIFIER`: `59`.

Implication:
- Required metadata coverage is complete.
- Per-problem synchronization and custom verifier hooks still do not exist.

## Input/Output Handling Observations

- `get_inputs()` present: `59/59`.
- `get_init_inputs()` present: `59/59`.
- Input generation mostly stochastic:
  - uses `torch.randn`: `55/59`
  - uses `torch.randint`: `8/59`
  - uses `torch.rand`: `1/59`
- Per-problem internal seeding: `0/59` (harness-level seeding controls determinism).
- Explicit dtype declaration in problem file code: `20/59`.

## Functional Verification Gaps

1. Tolerance policy is global, not operation/dtype/hardware specific.
- Same `atol/rtol` is applied to softmax, GEMM, attention, quantized paths, etc.

2. No explicit NaN/Inf fail-fast.
- Current max-diff path can miss pathology if NaN propagates in a non-comparable way.

3. No determinism/race checks on generated kernels.
- No repeated-run consistency check on the same input.

4. No memory-safety checks.
- No OOB write/read detection, no sanitizer integration, no guard-region checks.

5. Single-shape verification per problem invocation.
- Correctness is multi-seed but not multi-shape; shape-generalization bugs can pass.

6. No custom per-problem verifier logic.
- Complex outputs (e.g., structured/stateful outputs) cannot define richer equivalence criteria.

7. Backend-specific verifier is still CUDA-centric in the base benchmark template.
- Good for CUDA/Triton/CUTLASS/CuTe/CuTile, but requires dedicated harness behavior for Metal/Graphics.

## Platform and Portability Findings

- Portable problems (no obvious CUDA markers): `56/59`.
- Needs adaptation (runtime/device-specific markers): `3/59` (all CuTile files).
- CUDA low-level primitive usage in problem references: `0/59`.

Metal (Level 1-4 only):
- Runs unchanged (heuristic): `39`.
- Needs adaptation: `0`.
- Impossible on current Metal stack: `2` (`FP8`, `INT4` level-4 variants).

## Precision Suitability Gaps

- Most files do not declare precision policy metadata directly.
- Precision suitability is currently inferred rather than asserted by problem owners.
- No per-problem verifier tolerance override for low-precision modes.
