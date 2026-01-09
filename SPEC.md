# KernelBench v3

## Overview

Benchmark evaluating LLM ability to write optimized GPU kernels. Models receive PyTorch reference implementations and must produce faster CUDA kernels that pass correctness checks.

## Current State: READY TO RUN

All infrastructure is complete. Ready to execute evaluations.

### Next Action
Run the 5 cheapest models first (total ~$12-22):
```bash
cd /home/infatoshi/cuda/KernelBench-v3
uv run python batch_eval.py --models gemini-3-flash --gpus H100,B200 --levels 1,2,3,4 --workers 4
uv run python batch_eval.py --models deepseek-v3.2 --gpus H100,B200 --levels 1,2,3,4 --workers 4
uv run python batch_eval.py --models glm-4.7 --gpus H100,B200 --levels 1,2,3,4 --workers 4
uv run python batch_eval.py --models minimax-m2.1 --gpus H100,B200 --levels 1,2,3,4 --workers 4
uv run python batch_eval.py --models kimi-k2-thinking --gpus H100,B200 --levels 1,2,3,4 --workers 4
```

## Problem Set (41 total)

| Level | Count | Turns | Description |
|-------|-------|-------|-------------|
| L1 | 15 | 10 | Simple ops: matmul, softmax, conv, norms |
| L2 | 15 | 12 | Fused ops: matmul+activation chains |
| L3 | 3 | 15 | Single blocks: attention, transformer block |
| L4 | 8 | 15 | Novel layers: MLA, MoE, GQA, FP8, INT4, DeltaNet |

## Models (10 total)

### Frontier
- Claude Opus 4.5 ($15/$75 per M) - ~$80-120 per run
- Claude Sonnet 4.5 ($3/$15 per M) - ~$25-40 per run
- GPT-5.2 ($10/$30 per M) - ~$30-50 per run
- Gemini 3 Flash ($0.10/$0.40 per M) - ~$1-2 per run
- Gemini 3 Pro ($1.25/$5 per M) - ~$8-12 per run
- Grok 4.1 ($3/$15 per M) - ~$15-25 per run

### Open/Chinese (via OpenRouter)
- GLM-4.7 ($0.50/$2 per M) - ~$3-5 per run
- DeepSeek V3.2 ($0.30/$1.20 per M) - ~$2-4 per run
- Kimi K2 Thinking ($0.40/$1.75 per M) - ~$3-6 per run
- MiniMax M2.1 ($0.50/$2 per M) - ~$3-5 per run

**Total cost for all 10 models: $170-270**

## Infrastructure

### GPUs (via Modal)
- H100 (Hopper, sm_90) - FP8, TMA, wgmma
- B200 (Blackwell, sm_100) - cuTile DSL, FP8/FP4

### Environment
- CUDA 13.1 (cuTile DSL, full Blackwell support)
- PyTorch with torch._scaled_mm for FP8
- CUTLASS/CuTe DSL available
- git, cmake, ninja for building custom kernels

### Correctness
- 5 random seeds (42, 123, 456, 789, 1337)
- atol=1e-02, rtol=1e-02

### Entry Points
- `batch_eval.py` - Batch evaluation across models/GPUs/problems
- `modal_eval.py` - Single problem evaluation

## Output

Results in `outputs/batch_eval/run_YYYYMMDD_HHMMSS/`:
- `results.jsonl` - Per-problem: model, gpu, problem, speedup, tokens, cost
- `summary.json` - Aggregated statistics

## API Keys Required

Set in environment:
- ANTHROPIC_API_KEY
- OPENAI_API_KEY
- XAI_API_KEY
- GEMINI_API_KEY
- OPENROUTER_API_KEY
- Modal: ~/.modal.toml (profile: elliot-2)
