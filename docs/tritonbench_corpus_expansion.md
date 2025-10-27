# TritonBench Corpus Expansion Plan

## Objectives
- Expand the local TritonBench dataset so the agentic benchmark covers the full kernel suite (≈4 k problems) instead of the single `context_attn_nopad.py` task.
- Preserve the existing evaluation workflow (batch runner, agentic/raw modes, visualization) while scaling dataset size.
- Document validation and release steps so future updates follow the same structure.

## Prerequisites
- Storage for the complete TritonBench assets (instructions JSON, reference kernels, perf harnesses, golden metrics). Expect several hundred MB.
- Network access to download the official TritonBench release (e.g., NGC, GitHub, or internal artifact registry).
- Ability to re-run long LLM benchmarks (hours), including quota for all configured providers.

## Step-by-Step

### 1. Source the Corpus
1. Locate the latest TritonBench release package (tarball/zip).
2. Verify contents include:
   - Instruction JSON (e.g., `TritonBench_G_comp_alpac_v1_fixed_with_difficulty.json`, `train_crawl.json`).
   - Reference implementations under `TritonBench_G_v1/` (one `*.py` per kernel with test harness appended).
   - Performance harness and golden metrics under `performance_metrics/perf_G/`.
3. Record checksums for each artifact for reproducibility.

### 2. Stage Files Locally
1. Create a staging directory outside the repo (e.g., `/tmp/tritonbench_full`).
2. Extract the release).
3. Inspect file structure; ensure filenames match loader expectations (e.g., `context_attn_nopad.py`, `layer_norm.py`).

### 3. Normalize for Loader Compatibility
1. Compare staged files against `data/TritonBench/data/` in the repo.
2. Adjust if necessary:
   - Ensure instruction JSON has `instruction`, `output`/`label`, `filename` fields.
   - Confirm the test harness delimiter (`#` * 146) exists so loader splits prompt from tests.
   - Update any hard-coded absolute paths in perf harnesses to use placeholders the loader rewrites.
3. If the release introduces new schema fields, extend `TritonBench.load_ps` to handle them (e.g., fallback file lookup, new metadata fields).

### 4. Integrate into the Repo
1. Copy staged contents into `data/TritonBench/data/`:
   - Replace `train_crawl.json`, `TritonBench_G_v1/`, and related performance folders.
   - Keep backups of the original files in case regression comparison is needed.
2. Decide whether to commit the assets (large diff) or add a fetch step:
   - **Committed**: add `.gitattributes` if large files require Git LFS.
   - **Fetched**: create a bootstrap script (e.g., `scripts/fetch_tritonbench_full.sh`) and document in README.

### 5. Update Configuration
1. Create a new benchmark config (e.g., `configs/all_providers_full.yaml`) with:
   - `problems.max_problems: null` (or omitted) so agentic & raw traversals cover all problems.
   - Explicit `visualization.enabled: true` and `plots_dir` for large-run charts.
2. Optionally add smaller configs for smoke/regression runs.

### 6. Validate Incrementally
1. Run a targeted smoke test (few problems) to verify the loader sees >1 problem:
   ```
   uv run python eval.py --config configs/all_providers_full.yaml --num-runs 5 --mode agentic --language cuda
   ```
2. Confirm logs show `Using full TritonBench dataset: N problems` with N >> 1.
3. Inspect a sample run directory to ensure manifests include compiled/correct stats for multiple kernels.

### 7. Execute the Full Benchmark
1. Prepare quota & scheduling: full run across all providers can take many GPU hours.
2. Use the provided orchestration script (see `scripts/run_full_tritonbench_eval.sh`) to kick off the job.
3. Monitor logs for rate limiting; consider staggering providers or reducing concurrency if needed.

### 8. Post-Run Artifacts
1. Collect `json/batch_summary_*.json`, per-run `manifest.yaml`, and `plots/benchmark_*.png`.
2. Archive results with timestamp (e.g., move to `results/tritonbench_full/YYYYMMDD`).
3. Update README and PROVE.md with findings (success rates, notable issues).

### 9. Maintenance
- Track upstream TritonBench releases; repeat normalization when schema changes.
- Add regression checks to CI (e.g., ensure loader still finds expected problem count).
- Consider snapshotting instructions/tests in a dedicated submodule to simplify updates.

## Risk & Mitigation
- **Large data footprint**: Use Git LFS or document external download.
- **Provider quotas**: Implement rate-limit backoff (already added) and schedule bulk runs off-peak.
- **Schema drift**: Keep a fixture-based smoke test that loads a handful of kernels to detect loader breakage early.

## Open Questions
- Should we parameterize problem subsets (difficulty tiers) for targeted benchmarking?
- Do we want automated reporting (HTML dashboards) once plots are generated?
- How will we version datasets to ensure reproducibility across runs?

