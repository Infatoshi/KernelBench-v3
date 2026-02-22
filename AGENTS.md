# AGENTS.md - KernelBench-v3 Handoff

Last updated: 2026-02-22

## Snapshot
- Repository: `/home/infatoshi/cuda/KernelBench-v3`
- Branch: `master`
- Latest pushed commit: `278ca66`
- Remote: `origin https://github.com/Infatoshi/KernelBench-v3.git`
- Local run artifacts are not in git (`outputs/` is ignored)

## Backup Locations
- Local artifacts: `/home/infatoshi/cuda/KernelBench-v3/outputs`
- Mac backup mirror: `~/KernelBench-v3-backup/outputs`
- Verified backup parity at handoff:
  - run directories: `126`
  - files: `9046`

## Non-Negotiable Project Rules
- Use UV only for Python commands.
  - `uv run ...`
  - `uv add ...` / `uv pip install ...`
- Do not use bare `python` or `pip` from this machine workflow.
- Before closing implementation work, run:
  - `uv run ruff check . --fix`
  - `uv run pytest`

## Implemented Harnesses
- `batch_eval.py` (CUDABench)
- `triton_batch_eval.py` (TritonBench)
- `cutlass_batch_eval.py` (CUTLASSBench)
- `cute_batch_eval.py` (CuTeBench)
- `cutile_batch_eval.py` (CuTileBench)
- `metal_batch_eval.py` (MetalBench)
- `graphics_batch_eval.py` (GraphicsBench)

Supporting eval modules and prompts are present under:
- `*_eval.py`
- `src/prompts/*.py`
- `src/config/*.py`
- `src/agent/*.py`

## Current GPU Policy (Enforced)
| Benchmark | Allowed GPUs |
|---|---|
| CUDA | `RTX3090`, `H100`, `B200` |
| Triton | `RTX3090`, `H100`, `B200` |
| CUTLASS | `H100`, `B200` |
| CuTe | `H100`, `B200` |
| CuTile | `B200` |
| Metal | `M4MAX` |
| Graphics | `RTX3090` |

Validation helpers are centralized in:
- `src/config/runtime_validation.py`

## Problem Inventory
Current curated problem counts are in `problem_inventory.md`:
- Level 1: 15
- Level 2: 15
- Level 3: 3
- Level 4: 8
- Graphics: 2
- Tile specialized: 13
- CuTile: 3
- Total: 59

## Model Registry Status
`modal_eval.py` includes frontier model entries used in this session, including:
- `anthropic/claude-opus-4.6`
- `openai/gpt-5.2-codex`
- `google/gemini-3-flash-preview`
- `google/gemini-3-pro-preview`
- `minimax/minimax-m2.5`
- `z-ai/glm-5`
- `deepseek/deepseek-v3.2`
- `x-ai/grok-4.1-fast`
- `moonshotai/kimi-k2.5`

## What Is Verified Recently
1. GraphicsBench on RTX3090 with Claude Opus 4.6 (`run_20260221_230535`)
- `bloom.py`: compiled=true, correct=true, speedup=1.6673x
- `particles.py`: compiled=true, correct=true, speedup=0.9361x

2. Minimax Triton path success reference (`run_20260213_201602`)
- `1_Square_matrix_multiplication_.py` on H100: compiled=true, correct=true, speedup=1.3838x

3. Baseline metric fields are wired into results for newer runs
- `solution_path`, `solution_hash`
- `correctness_seeds`, `benchmark_seed`
- `precision_*` fields
- `achieved_tflops`, `ref_tflops`, `% peak` where applicable

## Known Hard Problems / Open Gaps
1. CuTe frontier difficulty remains unresolved for correctness
- Best known compile from Claude Sonnet 4 is still numerically wrong (`max_diff` around `243`).
- Treat as capability benchmark, not a harness bug, unless a concrete harness error appears.

2. CuTile is implemented but model reliability is low
- Harness exists and is B200-restricted.
- Typical current failure mode in recent runs: no valid submission within turn budget.
- Keep prompt/examples aligned with actual `cuda.tile` APIs to reduce misuse.

3. MetalBench is operational as separate mac workflow but still needs broader model sweeps
- Remote workspace exists at `~/MetalBench` on `macbook`.
- `.venv` exists there.

## Key Infra Changes Already Landed
- Per-problem wall clock timeout in `modal_eval.py`
- Auto-submit behavior when compilation/import checks pass
- Per-turn artifacts under `outputs/.../turns/`
- Precision matrix at `src/config/precision_matrix.py`
- Precision-aware tolerance, NaN/Inf checks, determinism checks
- Aggregation script: `aggregate_results.py`
- Baseline validator script: `validate_baselines.py`

## Fresh Session Resume Plan
1. Verify environment quickly
```bash
uv run python batch_eval.py --list-models
uv run python batch_eval.py --list-problems
```

2. Fast smoke test per target benchmark (dry-run)
```bash
uv run python batch_eval.py --models minimax/minimax-m2.5 --gpus H100 --levels 1 --problems-per-level 1 --dry-run
uv run python triton_batch_eval.py --models minimax/minimax-m2.5 --gpus H100 --levels 1 --problems-per-level 1 --dry-run
uv run python cutlass_batch_eval.py --models minimax/minimax-m2.5 --gpus H100 --levels 1 --problems-per-level 1 --dry-run
uv run python cute_batch_eval.py --models minimax/minimax-m2.5 --gpus H100 --levels 1 --problems-per-level 1 --dry-run
uv run python cutile_batch_eval.py --models minimax/minimax-m2.5 --gpus B200 --levels 1 --problems-per-level 1 --dry-run
uv run python graphics_batch_eval.py --models minimax/minimax-m2.5 --gpus RTX3090 --levels 1 --problems-per-level 1 --dry-run
```

3. Metal-only check from macbook
```bash
ssh macbook
cd ~/MetalBench
uv run python metal_batch_eval.py --models minimax/minimax-m2.5 --gpus M4MAX --levels 1 --problems-per-level 1 --dry-run
```

4. Run targeted real evals after dry-runs pass
- Keep `--problems-per-level 1` for incremental validation.
- Use higher `--max-turns` only when needed.

## Release Readiness Notes
- `run_all_benchmarks.py` exists for orchestration.
- `README.md` and `problem_inventory.md` were updated for current benchmark layout.
- Audit outputs exist in `audit/` and tests were added under `tests/`.

## Cautions for Next Agent
- `outputs/` is large and intentionally gitignored. Do not expect git to preserve run artifacts.
- A file named `;` and `excalidraw.log` are currently tracked from historical workspace state.
- Do not delete or reset unrelated files unless explicitly requested.
