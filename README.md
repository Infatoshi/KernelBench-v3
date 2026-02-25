# KernelBench v3

GPU kernel optimization benchmark for frontier LLMs. Tests whether AI models can write kernels that beat `torch.compile` on real workloads.

**7 backends**: CUDA, Triton, CUTLASS, CuTe, CuTile, Metal/MLX, Graphics
**3 GPU families**: NVIDIA (RTX3090, H100, B200), Apple Silicon (M4 Max)
**85 problems**: Deep learning ops, physics sims, rendering, scientific compute

## Quick Start

```bash
uv sync

uv run python bench.py list-models
uv run python bench.py list-backends

# Run CUDA benchmark on RTX3090
uv run python bench.py batch cuda --models deepseek/deepseek-v3.2 --gpus RTX3090 --levels 1,2,3,4 --workers 4

# Run Metal benchmark on MacBook
cd ~/MetalBench && uv run python bench.py batch metal --models minimax/minimax-m2.5 --gpus M4MAX --levels 1,2,3,4

# View results
uv run python bench.py summary outputs/batch_eval/run_XXXXXXXX
```

## How It Works

Each problem has a `reference.py` with a PyTorch `Model` class. The LLM agent gets N turns in a sandboxed environment to write an optimized `solution.py` using the target backend (CUDA C++, Triton, MLX, etc.). The solution is benchmarked against the reference — speedup > 1.0x means the model beat the baseline.

The agent has tools: `read_file`, `write_file`, `edit_file`, `bash`, `submit`. It reads the reference, writes a solution, tests compilation, and submits for benchmarking. Guardrails prevent reward hacking (e.g., falling back to PyTorch ops instead of writing actual kernels).

## Benchmarks

| Backend | Language | GPUs | Problems |
|---|---|---|---|
| `cuda` | CUDA C++ | RTX3090, H100, B200 | 41 |
| `triton` | Triton | RTX3090, H100, B200 | 41 |
| `cutlass` | CUTLASS 3.x | H100, B200 | 13 |
| `cute` | CuTe | H100, B200 | 13 |
| `cutile` | CuTile Python | B200 | 3 |
| `metal` | MLX (Metal) | M4MAX | 63 |
| `graphics` | Triton | RTX3090 | 2 |

## Problem Levels

- **L1** (15 problems): Simple operators — matmul, softmax, conv, LayerNorm, GELU
- **L2** (15 problems): Fused operations — matmul + activation + norm chains
- **L3** (3 problems): Architecture blocks — attention, transformer block
- **L4** (8 problems): Novel layers — DeepSeek MLA/MoE, GQA, FP8/INT4, GatedDeltaNet
- **Metal M1-M4** (26 problems): Image processing, physics, rendering, scientific compute
- **Graphics** (2 problems): Bloom effect, particle simulation
- **Tile** (13 problems): GEMM variants for CUTLASS/CuTe
- **CuTile** (3 problems): Persistent/stream-K/warp-specialized GEMM

## Results

Results are saved to `outputs/batch_eval/run_YYYYMMDD_HHMMSS/` with:
- `results.jsonl` — per-problem results (compiled, correct, speedup, tokens, cost)
- `kernels/` — submitted solution code (viewable on website)
- `summary.json` — aggregated statistics

## For LLMs

See [docs/LLMs.md](docs/LLMs.md) for the full autonomous agent guide — architecture, commands, rules, and session checklist. Everything needed to run evals without human intervention.

## Docs

- [Problem Inventory](docs/problem_inventory.md) — complete listing of all problems
- [Development Log](docs/devlog.md) — design decisions and incremental progress

## License

MIT
