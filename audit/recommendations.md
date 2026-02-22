# Benchmark Audit Recommendations

## Priority 0 (Blocking Accuracy)

1. Add explicit metadata to every problem file.
- Required fields: `OP_TYPE`, `SUPPORTED_PRECISIONS`, `HARDWARE_REQUIRED`.
- Optional but strongly recommended: `SYNC_REQUIRED`, `VERIFIER`.

2. Introduce per-op/per-precision tolerance policy.
- Example: stricter tolerances for FP32/BF16, relaxed and explicitly bounded tolerances for FP8/FP4.
- Avoid one-size-fits-all `atol/rtol`.

3. Add NaN/Inf explicit rejection in correctness check.
- Fail immediately if either reference or solution output contains NaN/Inf unless problem-specific verifier allows it.

## Priority 1 (Verification Quality)

4. Add custom verifier hook support.
- If `VERIFIER` exists in problem module, call it before default allclose/max-diff fallback.
- Enables operation-specific correctness criteria.

5. Add deterministic repeatability check.
- Re-run same inputs at least 2-3 times and assert stable outputs for deterministic kernels.
- Catch race-condition style failures.

6. Add shape-fuzz pass for Level 1/2.
- Add 2-3 alternate valid shape sets per problem.
- Avoid overfitting to one canonical shape.

## Priority 2 (Platform Robustness)

7. Maintain explicit backend-compatibility registry.
- Keep `audit/metal_compatibility.csv`-style data in source control and regenerate in CI.
- Mark unsupported precision/backend pairs at scheduling time.

8. Keep CuTile hardware gating strict.
- Current CUDA 13.1 `tileiras` is Blackwell-only in this environment.
- Keep CuTile restricted to B200 until toolchain support expands.

## Priority 3 (Operational Hygiene)

9. Add CI lint for metadata completeness.
- Fail CI if new problem files omit required metadata keys.

10. Emit structured audit artifact on each benchmark run.
- Include verifier used, tolerance profile, precision profile, and backend compatibility decisions.

## Suggested Implementation Order

1. Metadata completion + CI checks.
2. Per-op/per-precision tolerance matrix + NaN/Inf checks.
3. `VERIFIER` integration + repeatability checks.
4. Shape-fuzz support.
5. Compatibility registry maintenance.

