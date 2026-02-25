# KernelBench v3 — LLM Agent Guide

Read this file first. It contains everything needed to run evaluations autonomously.

## What This Is

GPU kernel optimization benchmark. LLM agents write CUDA/Triton/MLX kernels in sandboxed environments, competing against `torch.compile` baselines. Measures compilation rate, correctness, and speedup.

## Single Entry Point

```bash
uv run python bench.py <command> [args]
```

Commands:
- `batch <backend>` — run all problems for a model
- `eval <backend>` — run single problem
- `list-models` — show registered models
- `list-backends` — show backends with allowed GPUs
- `list-problems --backend <name>` — show problems for a backend
- `summary <run_dir>` — print results for a completed run

## Running Evaluations

```bash
# CUDA on RTX3090 (local GPU, cheapest)
uv run python bench.py batch cuda --models minimax/minimax-m2.5 --gpus RTX3090 --levels 1,2,3,4 --workers 4

# Triton on RTX3090
uv run python bench.py batch triton --models deepseek/deepseek-v3.2 --gpus RTX3090 --levels 1,2,3,4 --workers 4

# Metal on MacBook (run from macbook)
cd ~/MetalBench && uv run python bench.py batch metal --models minimax/minimax-m2.5 --gpus M4MAX --levels 1,2,3,4 --workers 4

# Dry-run (shows tasks, no execution, no cost)
uv run python bench.py batch cuda --models minimax/minimax-m2.5 --gpus H100 --levels 1 --problems-per-level 1 --dry-run
```

## Backends and GPU Restrictions (non-negotiable)

| Backend | Allowed GPUs | Notes |
|---|---|---|
| cuda | RTX3090, H100, B200 | CUDA C++ via `load_inline` |
| triton | RTX3090, H100, B200 | `@triton.jit` kernels |
| cutlass | H100, B200 | CUTLASS 3.x tile APIs |
| cute | H100, B200 | CuTe abstractions |
| cutile | B200 | `cuda.tile` Python DSL |
| metal | M4MAX | MLX kernels, macOS only |
| graphics | RTX3090 | Triton for graphics workloads |

## Model Registry

Models are in `src/models.py`. Current set:
- `anthropic/claude-opus-4.6`, `anthropic/claude-sonnet-4.6`
- `openai/gpt-5.2-codex`, `openai/gpt-5.3-codex`
- `google/gemini-3-flash-preview`, `google/gemini-3-pro-preview`, `google/gemini-3.1-pro-preview`
- `deepseek/deepseek-v3.2`, `z-ai/glm-5`, `minimax/minimax-m2.5`
- `moonshotai/kimi-k2.5`, `qwen/qwen3-coder-next`, `qwen/qwen3.5-397b-a17b`

Any valid OpenRouter model ID also works dynamically.

## Architecture

```
bench.py                    CLI entry point
src/
  eval/agent.py             Agent loop (Gemini/reasoning/standard)
  eval/context.py           Workspace context building
  eval/results.py           EvalResult dataclass
  eval/guardrails.py        Solution validation per backend
  batch.py                  Batch orchestration
  models.py                 Model registry, pricing
  api.py                    API communication, token tracking
  tools.py                  Tool schemas + dispatch
  parsing.py                XML/code parsing
  backends/                 Backend classes (cuda, triton, metal, etc.)
  prompts/                  Per-backend system prompts
  config/                   Problem config, precision matrix, GPU validation
  agent/                    Sandbox implementations (Modal, Metal, Local)
problems/                   Problem definitions
  level1/..level4/          Shared DL problems (41)
  metal_level1/..4/         Metal-specific (26)
  graphics/                 Graphics problems (2)
  tile_specialized/         CUTLASS/CuTe problems (13)
  cutile/                   CuTile problems (3)
tests/                      32 tests
outputs/                    Run artifacts (gitignored)
```

## Rules

- Use `uv run` for all Python execution. Never bare `python` or `pip`.
- Before completing work: `uv run --with ruff ruff check . --fix && uv run pytest`
- Do not relax GPU restrictions.
- Do not refactor unrelated modules.
- Kernel solutions are saved in `outputs/batch_eval/run_*/kernels/` for website export.

## Problem Levels

- L1 (15): Simple ops — matmul, softmax, conv, norms
- L2 (15): Fused ops — matmul+activation chains
- L3 (3): Architecture blocks — attention, transformer block
- L4 (8): Novel layers — MLA, MoE, GQA, FP8, INT4, GatedDeltaNet
- Metal M1-M4 (26): Image processing, physics, rendering, scientific compute
- Graphics (2): Bloom, particles
- Tile (13): GEMM variants for CUTLASS/CuTe
- CuTile (3): Persistent/stream-K/warp-specialized GEMM

## MetalBench Contract

- Evaluates real problems from `reference.py`, not synthetic matmul.
- Solution interface: `solution(*inputs)` with dynamic arity.
- Inputs from `reference.get_inputs()` converted to MLX types.
- Guardrails: forbid `torch`, `triton`, CUDA extensions; require `import mlx.core as mx`.

## Fresh Session Checklist

```bash
uv run python bench.py list-models
uv run python bench.py list-backends
uv run python bench.py batch cuda --models minimax/minimax-m2.5 --gpus RTX3090 --levels 1 --problems-per-level 1 --dry-run
```
