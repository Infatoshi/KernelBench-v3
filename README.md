# KernelBench-v3

GPU kernel generation benchmark suite for LLMs.

## Benchmarks

| Benchmark | DSL | Hardware | Problems |
|-----------|-----|----------|----------|
| CUDABench | Raw CUDA | RTX3090, H100, B200 | 41 |
| TritonBench | Triton | RTX3090, H100, B200 | 41 |
| CUTLASSBench | CUTLASS | H100, B200 | 13 |
| CuTeBench | CuTe | H100, B200 | 13 |
| CuTileBench | CuTile Python DSL | B200 | 3 |
| MetalBench | Metal/MLX | M4MAX | 41 |
| GraphicsBench | CUDA/Triton | RTX3090 | 2 |

## Quick Start

```bash
# Install dependencies
uv sync

# Run one benchmark family
uv run python batch_eval.py --models anthropic/claude-opus-4.6 --gpus H100 --levels 1,2,3,4 --max-turns 10 --sequential

# Run all benchmark families with automatic platform filtering
uv run python run_all_benchmarks.py anthropic/claude-opus-4.6

# Aggregate results
uv run python aggregate_results.py --output results.csv
```

## GPU Restrictions

Each batch entrypoint enforces benchmark-specific GPU constraints and fails fast with explicit errors.

- CUDABench: `RTX3090`, `H100`, `B200`
- TritonBench: `RTX3090`, `H100`, `B200`
- CUTLASSBench: `H100`, `B200`
- CuTeBench: `H100`, `B200`
- CuTileBench: `B200`
- MetalBench: `M4MAX`
- GraphicsBench: `RTX3090`

Platform checks:
- `M4MAX` is macOS-only (`Darwin`).
- NVIDIA benchmark paths are Linux-only in this harness.

## Hardware Requirements

- NVIDIA (Linux/Modal): `RTX3090`, `H100`, `B200`
- Apple Silicon (macOS): `M4MAX` (MetalBench only)

## Supported Models

Current model registry (`modal_eval.py:MODELS`):

- `claude-opus-4.5`
- `claude-sonnet-4.5`
- `gpt-5.2`
- `gemini-3-flash`
- `gemini-3-pro`
- `grok-4.1`
- `glm-4.7`
- `deepseek-v3.2`
- `kimi-k2-thinking`
- `minimax-m2.1`
- `z-ai/glm-5`
- `openrouter/aurora-alpha`
- `anthropic/claude-opus-4.6`
- `openai/gpt-5.2-codex`
- `google/gemini-3-flash-preview`
- `google/gemini-3-pro-preview`
- `minimax/minimax-m2.5`
- `deepseek/deepseek-v3.2`
- `x-ai/grok-4.1-fast`
- `moonshotai/kimi-k2.5`
- `kimi-k2.5`

To list models directly:

```bash
uv run python batch_eval.py --list-models
```

## Validation Commands

```bash
# Wiring check (no expensive execution)
uv run python batch_eval.py --models minimax/minimax-m2.5 --gpus H100 --levels 1 --problems-per-level 1 --dry-run

# Wrong GPU examples (expected graceful failures)
uv run python cutile_batch_eval.py --gpus H100 --dry-run
uv run python metal_batch_eval.py --gpus H100 --dry-run
uv run python cutlass_batch_eval.py --gpus RTX3090 --dry-run
uv run python graphics_batch_eval.py --gpus H100 --dry-run
```

## Repository Layout

```text
KernelBench-v3/
├── batch_eval.py
├── triton_batch_eval.py
├── cutlass_batch_eval.py
├── cute_batch_eval.py
├── cutile_batch_eval.py
├── metal_batch_eval.py
├── graphics_batch_eval.py
├── run_all_benchmarks.py
├── modal_eval.py
├── aggregate_results.py
├── KernelBench/
│   ├── level1/ level2/ level3/ level4/
│   ├── tile_specialized/
│   ├── cutile/
│   └── graphics/
└── outputs/
```

## License

MIT
