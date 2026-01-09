# KernelBench v3

A benchmark for evaluating LLM ability to write optimized GPU kernels.

Models are given a PyTorch reference implementation and must produce a faster CUDA kernel that passes correctness checks across multiple random seeds. Evaluation runs on H100 (Hopper) and B200 (Blackwell) GPUs via Modal.

## Quick Start

```bash
# Install dependencies
uv sync

# Run evaluation for a single model
uv run python batch_eval.py --models gemini-3-flash --gpus H100,B200 --levels 1,2,3,4 --workers 4

# Run all models
uv run python batch_eval.py --models all --gpus H100,B200 --levels 1,2,3,4 --workers 4

# Resume interrupted run
uv run python batch_eval.py --resume outputs/batch_eval/run_XXXXXXXX

# View results
uv run python batch_eval.py --summary-only outputs/batch_eval/run_XXXXXXXX
```

## Problem Set

41 problems across 4 difficulty levels:

### Level 1: Simple Operators (15 problems, 10 turns)

Single GPU kernels for basic operations.

| Problem | Description |
|---------|-------------|
| 1_Square_matrix_multiplication_.py | Square matrix multiplication |
| 2_Standard_matrix_multiplication_.py | Standard matrix multiplication |
| 3_Batched_matrix_multiplication.py | Batched matrix multiplication |
| 4_Matrix_vector_multiplication_.py | Matrix-vector multiplication |
| 8_Matmul_with_irregular_shapes_.py | Matmul with irregular shapes |
| 9_Tall_skinny_matrix_multiplication_.py | Tall-skinny matrix multiplication |
| 23_Softmax.py | Softmax activation |
| 26_GELU_.py | GELU activation |
| 36_RMSNorm_.py | RMSNorm |
| 40_LayerNorm.py | LayerNorm |
| 42_Max_Pooling_2D.py | Max pooling 2D |
| 47_Sum_reduction_over_a_dimension.py | Sum reduction |
| 63_conv_standard_2D__square_input__square_kernel.py | 2D convolution |
| 82_conv_depthwise_2D_square_input_square_kernel.py | Depthwise 2D convolution |
| 95_CrossEntropyLoss.py | Cross-entropy loss |

### Level 2: Fused Operations (15 problems, 12 turns)

Multi-op fusion patterns (matmul + activation + norm).

| Problem | Description |
|---------|-------------|
| 6_Conv3d_Softmax_MaxPool_MaxPool.py | Conv3D with pooling chain |
| 17_Conv2d_InstanceNorm_Divide.py | Conv2D + InstanceNorm fusion |
| 37_Matmul_Swish_Sum_GroupNorm.py | Matmul + Swish + GroupNorm |
| 40_Matmul_Scaling_ResidualAdd.py | Matmul + scaling + residual |
| 46_Conv2d_Subtract_Tanh_Subtract_AvgPool.py | Conv2D + activation chain |
| 52_Conv2d_Activation_BatchNorm.py | Conv2D + BatchNorm fusion |
| 55_Matmul_MaxPool_Sum_Scale.py | Matmul + pooling chain |
| 59_Matmul_Swish_Scaling.py | Matmul + Swish + scaling |
| 66_Matmul_Dropout_Mean_Softmax.py | Matmul + dropout + softmax |
| 73_Conv2d_BatchNorm_Scaling.py | Conv2D + BatchNorm + scaling |
| 82_Conv2d_Tanh_Scaling_BiasAdd_Max.py | Conv2D + activation chain |
| 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp.py | Conv2D + GroupNorm fusion |
| 86_Matmul_Divide_GELU.py | Matmul + GELU fusion |
| 98_Matmul_AvgPool_GELU_Scale_Max.py | Matmul + pooling + GELU |
| 99_Matmul_GELU_Softmax.py | Matmul + GELU + softmax |

### Level 3: Single Blocks (3 problems, 15 turns)

Individual architecture blocks.

| Problem | Description |
|---------|-------------|
| 31_VisionAttention.py | Vision attention mechanism |
| 43_MinGPTCausalAttention.py | Causal attention (MinGPT style) |
| 44_MiniGPTBlock.py | Full transformer block |

### Level 4: Novel Layers (8 problems, 15 turns)

Modern inference kernels and custom layers - tests true kernel engineering ability on architectures not in training data.

| Problem | Description |
|---------|-------------|
| 1_DeepSeek_MLA.py | Multi-head Latent Attention with LoRA KV compression |
| 2_DeepSeek_MoE.py | MoE with grouped expert routing |
| 3_GroupedQueryAttention.py | GQA with KV head expansion (Llama 3 style) |
| 4_FP8_Matmul.py | FP8 E4M3 quantized matmul with tensor cores |
| 5_MoE_GatedGEMM.py | MoE with fused gated dual GEMM (SwiGLU FFN) |
| 6_INT4_Quantized_GEMM.py | INT4 weight-only quant with group-wise dequant fusion |
| 7_GatedDeltaNet.py | Gated delta rule linear attention (ICLR 2025) |
| 8_KimiDeltaAttention.py | Channel-wise gated delta attention |

## Models

10 models evaluated:

| Model | Provider | Pricing (input/output per M tokens) |
|-------|----------|-------------------------------------|
| Claude Opus 4.5 | Anthropic | $15 / $75 |
| Claude Sonnet 4.5 | Anthropic | $3 / $15 |
| GPT-5.2 | OpenAI | $10 / $30 |
| Gemini 3 Flash | Google | $0.10 / $0.40 |
| Gemini 3 Pro | Google | $1.25 / $5 |
| Grok 4.1 | xAI | $3 / $15 |
| GLM-4.7 | Zhipu AI | $0.50 / $2 |
| DeepSeek V3.2 | DeepSeek | $0.30 / $1.20 |
| Kimi K2 Thinking | Moonshot | $0.40 / $1.75 |
| MiniMax M2.1 | MiniMax | $0.50 / $2 |

### Cost Estimates (82 runs per model)

| Model | Estimated Cost |
|-------|----------------|
| Gemini 3 Flash | $1-2 |
| DeepSeek V3.2 | $2-4 |
| GLM-4.7 | $3-5 |
| MiniMax M2.1 | $3-5 |
| Kimi K2 Thinking | $3-6 |
| Gemini 3 Pro | $8-12 |
| Grok 4.1 | $15-25 |
| Claude Sonnet 4.5 | $25-40 |
| GPT-5.2 | $30-50 |
| Claude Opus 4.5 | $80-120 |
| **Total (all models)** | **$170-270** |

## Architecture

```
KernelBench-v3/
├── batch_eval.py           # Batch evaluation across models/GPUs/problems
├── modal_eval.py           # Single problem evaluation with Modal GPU
├── modal_benchmark.py      # GPU benchmarking infrastructure
├── KernelBench/            # Problem definitions
│   ├── level1/             # Simple operators (15)
│   ├── level2/             # Fused operations (15)
│   ├── level3/             # Single blocks (3)
│   └── level4/             # Novel layers (8)
├── src/
│   ├── agent/              # Agentic evaluation
│   │   ├── modal_sandbox.py    # Modal sandbox for GPU execution
│   │   └── ...
│   └── prompts/            # System prompts for models
└── outputs/
    └── batch_eval/         # Evaluation results (gitignored)
        └── run_YYYYMMDD_HHMMSS/
            ├── results.jsonl   # Per-problem results
            └── summary.json    # Aggregated statistics
```

### Evaluation Flow

1. **Agent Loop**: Model receives problem + GPU info, writes CUDA kernel iteratively
2. **Modal Sandbox**: Code executes on remote GPU (H100 or B200) with CUDA 13.1
3. **Correctness Check**: Tested across 5 random seeds (42, 123, 456, 789, 1337)
4. **Performance Benchmark**: Speedup measured over PyTorch baseline (10 trials)

### GPU Support

| GPU | Architecture | SM Version | Notes |
|-----|--------------|------------|-------|
| H100 | Hopper | sm_90 | FP8, TMA, wgmma |
| B200 | Blackwell | sm_100 | cuTile DSL, FP8/FP4 |

Both GPUs run CUDA 13.1 with full CUTLASS and CuTe DSL support.

## Output Format

Each evaluation produces:

```json
{
  "model": "gemini-3-flash",
  "gpu": "H100",
  "problem": "level1/23_Softmax.py",
  "speedup": 2.34,
  "baseline_ms": 0.45,
  "solution_ms": 0.19,
  "correctness": true,
  "input_tokens": 12500,
  "output_tokens": 8300,
  "estimated_cost_usd": 0.0046
}
```

## Requirements

- Python 3.11+
- Modal account with GPU access
- API keys for model providers (set in environment)

Required environment variables:
```
ANTHROPIC_API_KEY
OPENAI_API_KEY
XAI_API_KEY
GEMINI_API_KEY
OPENROUTER_API_KEY
```

## Citation

```bibtex
@misc{ouyang2025kernelbenchllmswriteefficient,
  title={KernelBench: Can LLMs Write Efficient GPU Kernels?},
  author={Anne Ouyang and Simon Guo and Simran Arora and Alex L. Zhang and William Hu and Christopher Re and Azalia Mirhoseini},
  year={2025},
  eprint={2502.10517},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2502.10517},
}
```

## License

MIT
