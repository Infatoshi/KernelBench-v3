<?xml version="1.0" encoding="UTF-8"?>
<claude-instructions>

<!-- ============================================================================
     CURRENT STATUS (2026-01-09)
     ============================================================================ -->

<status>
  <state>READY TO RUN EVALUATIONS</state>
  <next-action>
    Run the 5 cheapest models first (~$12-22 total):
    1. gemini-3-flash (~$1-2)
    2. deepseek-v3.2 (~$2-4)
    3. glm-4.7 (~$3-5)
    4. minimax-m2.1 (~$3-5)
    5. kimi-k2-thinking (~$3-6)
  </next-action>
  <completed>
    - 41 problems finalized (L1:15, L2:15, L3:3, L4:8)
    - 10 models configured with pricing
    - Modal sandbox with CUDA 13.1, git, cmake for CUTLASS/CuTe
    - Multi-seed correctness (5 seeds)
    - FP8 baseline fixed (torch._scaled_mm)
    - README.md updated (Karpathy style)
    - .gitignore updated (no API keys)
  </completed>
</status>

<!-- ============================================================================
     QUICK REFERENCE (READ THIS AFTER CONTEXT COMPACTION)
     ============================================================================ -->

<quick-reference>
  <summary>
    KernelBench v3: GPU kernel optimization benchmark for 10 LLMs on 41 problems.
    Evaluates on H100 + B200 GPUs via Modal. Tracks tokens and costs.
  </summary>

  <current-config>
    <problems total="41">
      <level id="1" count="15" max_turns="10">Simple ops: matmul, softmax, conv, norms</level>
      <level id="2" count="15" max_turns="12">Fused ops: matmul+activation chains</level>
      <level id="3" count="3" max_turns="15">Single blocks: attention, transformer block</level>
      <level id="4" count="8" max_turns="15">Novel layers: MLA, MoE, GQA, FP8, GatedGEMM, INT4, GatedDeltaNet, KDA</level>
    </problems>
    <gpus>H100, B200 (2 GPUs via Modal)</gpus>
    <total-evals-per-model>41 problems x 2 GPUs = 82 runs</total-evals-per-model>
  </current-config>

  <models count="10">
    Frontier: Claude Opus 4.5, Claude Sonnet 4.5, GPT-5.2, Gemini 3 Flash, Gemini 3 Pro, Grok 4.1
    Open: GLM-4.7, DeepSeek V3.2, Kimi K2 Thinking, MiniMax M2.1
  </models>

  <key-commands>
    <cmd name="run-model">cd /home/infatoshi/cuda/KernelBench-v3 && uv run python batch_eval.py --models MODEL --gpus H100,B200 --levels 1,2,3,4 --workers 4</cmd>
    <cmd name="check-results">uv run python batch_eval.py --summary-only outputs/batch_eval/RUN_DIR</cmd>
    <cmd name="monitor">tail -f logs/MODEL.log | grep -E "Token Usage|Cost|\[OK\]|\[FAIL\]"</cmd>
  </key-commands>

  <cost-estimates per-82-runs="estimated">
    <model name="gemini-3-flash">$1-2</model>
    <model name="deepseek-v3.2">$2-4</model>
    <model name="glm-4.7">$3-5</model>
    <model name="minimax-m2.1">$3-5</model>
    <model name="kimi-k2-thinking">$3-6</model>
    <model name="gemini-3-pro">$8-12</model>
    <model name="grok-4.1">$15-25</model>
    <model name="claude-sonnet-4.5">$25-40</model>
    <model name="gpt-5.2">$30-50</model>
    <model name="claude-opus-4.5">$80-120</model>
  </cost-estimates>
</quick-reference>

<!-- ============================================================================
     PROBLEM SET (41 total)
     ============================================================================ -->

<problem-set updated="2026-01-09" total="41">
  <level id="1" name="Simple Operators" count="15" max_turns="10">
    <description>Single GPU kernels for basic operations</description>
    <problems>
      1_Square_matrix_multiplication_.py
      2_Standard_matrix_multiplication_.py
      3_Batched_matrix_multiplication.py
      4_Matrix_vector_multiplication_.py
      8_Matmul_with_irregular_shapes_.py
      9_Tall_skinny_matrix_multiplication_.py
      23_Softmax.py
      26_GELU_.py
      36_RMSNorm_.py
      40_LayerNorm.py
      42_Max_Pooling_2D.py
      47_Sum_reduction_over_a_dimension.py
      63_conv_standard_2D__square_input__square_kernel.py
      82_conv_depthwise_2D_square_input_square_kernel.py
      95_CrossEntropyLoss.py
    </problems>
  </level>

  <level id="2" name="Fused Operations" count="15" max_turns="12">
    <description>Multi-op fusion patterns (matmul + activation + norm)</description>
    <problems>
      6_Conv3d_Softmax_MaxPool_MaxPool.py
      17_Conv2d_InstanceNorm_Divide.py
      37_Matmul_Swish_Sum_GroupNorm.py
      40_Matmul_Scaling_ResidualAdd.py
      46_Conv2d_Subtract_Tanh_Subtract_AvgPool.py
      52_Conv2d_Activation_BatchNorm.py
      55_Matmul_MaxPool_Sum_Scale.py
      59_Matmul_Swish_Scaling.py
      66_Matmul_Dropout_Mean_Softmax.py
      73_Conv2d_BatchNorm_Scaling.py
      82_Conv2d_Tanh_Scaling_BiasAdd_Max.py
      85_Conv2d_GroupNorm_Scale_MaxPool_Clamp.py
      86_Matmul_Divide_GELU.py
      98_Matmul_AvgPool_GELU_Scale_Max.py
      99_Matmul_GELU_Softmax.py
    </problems>
  </level>

  <level id="3" name="Single Blocks" count="3" max_turns="15">
    <description>Individual architecture blocks (attention, transformer block) - no full models</description>
    <problems>
      31_VisionAttention.py
      43_MinGPTCausalAttention.py
      44_MiniGPTBlock.py
    </problems>
  </level>

  <level id="4" name="Novel Layers" count="8" max_turns="15">
    <description>Modern inference kernels and custom layers (not in training data) - tests true kernel engineering ability</description>
    <problems>
      1_DeepSeek_MLA.py       # Multi-head Latent Attention with LoRA KV compression
      2_DeepSeek_MoE.py       # MoE with grouped expert routing (batched computation)
      3_GroupedQueryAttention.py  # GQA with KV head expansion (Llama 3 style)
      4_FP8_Matmul.py         # FP8 E4M3 quantized matmul with tensor cores (torch._scaled_mm)
      5_MoE_GatedGEMM.py      # MoE with fused gated dual GEMM (SwiGLU FFN)
      6_INT4_Quantized_GEMM.py   # INT4 weight-only quant with group-wise dequant fusion
      7_GatedDeltaNet.py      # Gated delta rule linear attention (Qwen3-Next, ICLR 2025)
      8_KimiDeltaAttention.py # Channel-wise gated delta attention (Kimi Linear 48B)
    </problems>
    <rationale>
      L4 tests modern inference optimization patterns:
      - DeepSeek MLA/MoE: novel architectures not in training data
      - GQA: memory-efficient attention with KV sharing (implicit vs explicit repeat)
      - FP8: tensor core utilization at reduced precision via torch._scaled_mm
      - Gated GEMM: dual GEMM fusion for SwiGLU (CUTLASS pattern)
      - INT4 GEMM: weight-only quantization with fused unpack+dequant+matmul
      - Gated DeltaNet: linear attention with delta rule (arxiv 2412.06464, ICLR 2025, Qwen3-Next)
      - Kimi Delta Attention: channel-wise gating extension (arxiv 2510.26692, Kimi Linear 48B)
      All have clear optimization targets and naive PyTorch baselines.
    </rationale>
  </level>

  <architecture-policy>
    All L1-L4 problems are architecture-agnostic:
    - Same problems run on both H100 (Hopper/SM90) and B200 (Blackwell/SM100)
    - Models receive GPU info in prompt and can look up arch-specific instructions
    - FP8 (E4M3/E5M2) is the lowest precision - supported on both architectures
    - No FP4/FP6 problems (Blackwell-only, would exclude H100)
    - Results reported separately per GPU for direct comparison
  </architecture-policy>

  <future-levels>
    <level id="5">Multi-GPU: tensor parallelism, pipeline parallelism</level>
    <level id="6">Multi-node: distributed training/inference patterns</level>
  </future-levels>
</problem-set>

<!-- ============================================================================
     MODELS
     ============================================================================ -->

<models-to-evaluate count="10">
  <tier name="Frontier">
    <model key="claude-opus-4.5" id="claude-opus-4-5-20251101" provider="anthropic" pricing="$15/$75 per M"/>
    <model key="claude-sonnet-4.5" id="claude-sonnet-4-5-20250929" provider="anthropic" pricing="$3/$15 per M"/>
    <model key="gpt-5.2" id="gpt-5.2" provider="openai" pricing="$10/$30 per M"/>
    <model key="gemini-3-flash" id="gemini-3-flash-preview" provider="gemini" pricing="$0.10/$0.40 per M"/>
    <model key="gemini-3-pro" id="gemini-3-pro-preview" provider="gemini" pricing="$1.25/$5 per M"/>
    <model key="grok-4.1" id="grok-4-1-fast-reasoning" provider="xai" pricing="$3/$15 per M"/>
  </tier>
  <tier name="Open/Chinese">
    <model key="glm-4.7" id="z-ai/glm-4.7" provider="openrouter" pricing="$0.50/$2 per M"/>
    <model key="deepseek-v3.2" id="deepseek/deepseek-chat" provider="openrouter" pricing="$0.30/$1.20 per M"/>
    <model key="kimi-k2-thinking" id="moonshotai/kimi-k2-thinking" provider="openrouter" pricing="$0.40/$1.75 per M"/>
    <model key="minimax-m2.1" id="minimax/minimax-m2.1" provider="openrouter" pricing="$0.50/$2 per M"/>
  </tier>
</models-to-evaluate>

<!-- ============================================================================
     EVALUATION INFRASTRUCTURE
     ============================================================================ -->

<evaluation-infrastructure>
  <entry-points>
    <file path="modal_eval.py">Single problem evaluation with Modal GPU</file>
    <file path="batch_eval.py">Batch evaluation across models/GPUs/problems</file>
  </entry-points>

  <batch-eval-usage>
    # Run single model on all problems
    uv run python batch_eval.py --models gemini-3-flash --gpus H100,B200 --levels 1,2,3,4 --workers 4

    # Run all models
    uv run python batch_eval.py --models all --gpus H100,B200 --levels 1,2,3,4 --workers 4

    # Resume interrupted run
    uv run python batch_eval.py --resume outputs/batch_eval/run_XXXXXXXX

    # View summary only
    uv run python batch_eval.py --summary-only outputs/batch_eval/run_XXXXXXXX
  </batch-eval-usage>

  <output-structure>
    outputs/batch_eval/run_YYYYMMDD_HHMMSS/
      results.jsonl    # One JSON per eval: model, gpu, problem, speedup, tokens, cost
      summary.json     # Aggregated stats
    logs/
      MODEL.log        # Per-model evaluation logs
  </output-structure>

  <token-tracking>
    Each result includes: input_tokens, output_tokens, total_tokens, estimated_cost_usd
    Aggregated in summary: total cost per model, per level, overall
  </token-tracking>
</evaluation-infrastructure>

<!-- ============================================================================
     API KEYS (all verified working)
     ============================================================================ -->

<api-keys status="loaded">
  ANTHROPIC_API_KEY, OPENAI_API_KEY, XAI_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY
  MODAL_TOKEN in ~/.modal.toml (profile: elliot-2)
  DO NOT repeatedly verify - assume they exist and work.
</api-keys>

<!-- ============================================================================
     RULES
     ============================================================================ -->

<rules>
  <rule>Use uv run for all Python execution</rule>
  <rule>Run from /home/infatoshi/cuda/KernelBench-v3</rule>
  <rule>Default GPUs: H100, B200 (skip L40S, A100 to save cost)</rule>
  <rule>Per-level turn limits: L1=10, L2=12, L3=15, L4=15</rule>
  <rule>Track token usage and costs for all runs</rule>
  <rule>Assume API keys exist - don't repeatedly check</rule>
</rules>

</claude-instructions>
