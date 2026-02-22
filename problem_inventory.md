# KernelBench-v3 Problem Inventory

## Summary

| Category | Count |
|----------|-------|
| Level 1 | 15 |
| Level 2 | 15 |
| Level 3 | 3 |
| Level 4 | 8 |
| Graphics | 2 |
| Tile Specialized | 13 |
| CuTile | 3 |
| **Total** | **59** |

## Benchmark Mapping

- CUDA/Triton: KernelBench/level1..level4
- CUTLASS/CuTe: KernelBench/tile_specialized
- CuTile: KernelBench/cutile
- Metal: KernelBench/level1..level4
- Graphics: KernelBench/graphics

## CuTileBench Status

- Harness scripts implemented: `cutile_eval.py`, `cutile_batch_eval.py`, `src/prompts/cutile_system.py`
- Modal CUDA image: `nvidia/cuda:13.1.0-devel-ubuntu24.04`
- Runtime dependency:
  - installs `cuda-tile` (`cutile-python`) in Modal image
  - import path is `import cuda.tile as ct`
- Hardware support (current runtime):
  - B200 only (`tileiras` in CUDA 13.1 rejects `sm_90`)
- Validation:
  - Dry-run passed: `uv run python cutile_batch_eval.py --models minimax/minimax-m2.5 --gpus B200 --levels 1 --problems-per-level 1 --dry-run`
  - Real run status depends on model capability and CUDA/driver support for Tile IR on target GPU.
- Current state: CuTileBench is implemented as a separate harness and problem set using the official Python API.

## Per-Model Run Counts

| Benchmark | Problems | Runs (9 models) |
|-----------|----------|-----------------|
| CUDABench | 41 | 369 |
| TritonBench | 41 | 369 |
| CUTLASSBench | 13 | 117 |
| CuTeBench | 13 | 117 |
| CuTileBench | 3 | 27 |
| MetalBench | 41 | 369 |
| GraphicsBench | 2 | 18 |
| **Total** |  | **1386** |

## Detailed Listings

### Level 1
1. KernelBench/level1/1_Square_matrix_multiplication_.py
2. KernelBench/level1/23_Softmax.py
3. KernelBench/level1/26_GELU_.py
4. KernelBench/level1/2_Standard_matrix_multiplication_.py
5. KernelBench/level1/36_RMSNorm_.py
6. KernelBench/level1/3_Batched_matrix_multiplication.py
7. KernelBench/level1/40_LayerNorm.py
8. KernelBench/level1/42_Max_Pooling_2D.py
9. KernelBench/level1/47_Sum_reduction_over_a_dimension.py
10. KernelBench/level1/4_Matrix_vector_multiplication_.py
11. KernelBench/level1/63_conv_standard_2D__square_input__square_kernel.py
12. KernelBench/level1/82_conv_depthwise_2D_square_input_square_kernel.py
13. KernelBench/level1/8_Matmul_with_irregular_shapes_.py
14. KernelBench/level1/95_CrossEntropyLoss.py
15. KernelBench/level1/9_Tall_skinny_matrix_multiplication_.py

### Level 2
1. KernelBench/level2/17_Conv2d_InstanceNorm_Divide.py
2. KernelBench/level2/37_Matmul_Swish_Sum_GroupNorm.py
3. KernelBench/level2/40_Matmul_Scaling_ResidualAdd.py
4. KernelBench/level2/46_Conv2d_Subtract_Tanh_Subtract_AvgPool.py
5. KernelBench/level2/52_Conv2d_Activation_BatchNorm.py
6. KernelBench/level2/55_Matmul_MaxPool_Sum_Scale.py
7. KernelBench/level2/59_Matmul_Swish_Scaling.py
8. KernelBench/level2/66_Matmul_Dropout_Mean_Softmax.py
9. KernelBench/level2/6_Conv3d_Softmax_MaxPool_MaxPool.py
10. KernelBench/level2/73_Conv2d_BatchNorm_Scaling.py
11. KernelBench/level2/82_Conv2d_Tanh_Scaling_BiasAdd_Max.py
12. KernelBench/level2/85_Conv2d_GroupNorm_Scale_MaxPool_Clamp.py
13. KernelBench/level2/86_Matmul_Divide_GELU.py
14. KernelBench/level2/98_Matmul_AvgPool_GELU_Scale_Max.py
15. KernelBench/level2/99_Matmul_GELU_Softmax.py

### Level 3
1. KernelBench/level3/31_VisionAttention.py
2. KernelBench/level3/43_MinGPTCausalAttention.py
3. KernelBench/level3/44_MiniGPTBlock.py

### Level 4
1. KernelBench/level4/1_DeepSeek_MLA.py
2. KernelBench/level4/2_DeepSeek_MoE.py
3. KernelBench/level4/3_GroupedQueryAttention.py
4. KernelBench/level4/4_FP8_Matmul.py
5. KernelBench/level4/5_MoE_GatedGEMM.py
6. KernelBench/level4/6_INT4_Quantized_GEMM.py
7. KernelBench/level4/7_GatedDeltaNet.py
8. KernelBench/level4/8_KimiDeltaAttention.py

### Graphics
1. KernelBench/graphics/bloom.py
2. KernelBench/graphics/particles.py

### Tile Specialized
1. KernelBench/tile_specialized/gemm_bf16.py
2. KernelBench/tile_specialized/gemm_bias_gelu.py
3. KernelBench/tile_specialized/gemm_bias_relu.py
4. KernelBench/tile_specialized/gemm_bias_silu.py
5. KernelBench/tile_specialized/gemm_fp4.py
6. KernelBench/tile_specialized/gemm_fp8.py
7. KernelBench/tile_specialized/gemm_mixed_fp8_fp16.py
8. KernelBench/tile_specialized/gemm_residual_add.py
9. KernelBench/tile_specialized/gemv_bf16.py
10. KernelBench/tile_specialized/gemv_fp16.py
11. KernelBench/tile_specialized/gemv_fp4.py
12. KernelBench/tile_specialized/gemv_fp8.py
13. KernelBench/tile_specialized/moe_grouped_gemm.py

### CuTile
1. KernelBench/cutile/persistent_gemm.py
2. KernelBench/cutile/stream_k_gemm.py
3. KernelBench/cutile/warp_specialized_gemm.py
