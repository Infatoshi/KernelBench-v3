# CuTeBench Analysis (2026-02-13)

## Scope
This analysis covers:
1. Claude's CuTe submission that compiled but failed correctness (`max_diff=243.18`).
2. Current CuTe tutorial patterns extracted from CUTLASS in Modal.
3. Prompt-vs-docs gap analysis.
4. Prompt updates applied in `src/prompts/cute_system.py`.
5. Post-update validation run on H100.

## 1) Claude Submission That Failed Correctness
Source: `outputs/batch_eval/run_20260213_001820/kernels/Anthropic:_Claude_Sonnet_4_H100_1_Square_matrix_multiplication_/solution.py`

### Full submitted code
```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel code using CuTe abstractions
cuda_source = """
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>

using namespace cute;

template<typename T>
__global__ void gemm_cute_kernel(
    T* A, T* B, T* C,
    int M, int N, int K
) {
    // Block tile sizes
    constexpr int BM = 128;
    constexpr int BN = 128; 
    constexpr int BK = 32;
    
    // Thread block coordinates
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Shared memory allocation
    __shared__ T sA[BM * BK];
    __shared__ T sB[BK * BN];
    
    // Create global tensor views with row-major layout
    auto gA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, Int<1>{}));
    auto gB = make_tensor(make_gmem_ptr(B), make_shape(K, N), make_stride(N, Int<1>{}));
    auto gC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, Int<1>{}));
    
    // Create shared memory tensor views
    auto smemA = make_tensor(make_smem_ptr(sA), make_shape(Int<BM>{}, Int<BK>{}));
    auto smemB = make_tensor(make_smem_ptr(sB), make_shape(Int<BK>{}, Int<BN>{}));
    
    // Tile the global tensors to block level
    auto tAgA = local_tile(gA, make_tile(Int<BM>{}, Int<BK>{}), make_coord(by, _));
    auto tBgB = local_tile(gB, make_tile(Int<BK>{}, Int<BN>{}), make_coord(_, bx));
    auto tCgC = local_tile(gC, make_tile(Int<BM>{}, Int<BN>{}), make_coord(by, bx));
    
    // Accumulator for this thread block
    auto acc = make_fragment_like(tCgC);
    clear(acc);
    
    // Thread-level tiling
    auto thr_layout = make_layout(make_shape(Int<16>{}, Int<8>{}));
    auto thr_idx = threadIdx.x + threadIdx.y * blockDim.x;
    
    // Main loop over K dimension  
    for (int k = 0; k < K; k += BK) {
        // Copy global tile to shared memory
        copy(tAgA(_, _, k/BK), smemA);
        copy(tBgB(_, _, k/BK), smemB);
        
        __syncthreads();
        
        // Perform GEMM on shared memory tiles
        // Simple implementation - each thread computes one element
        if (tx < BM && ty < BN) {
            T sum = T(0);
            for (int kk = 0; kk < BK; ++kk) {
                sum += smemA(tx, kk) * smemB(kk, ty);
            }
            acc(tx, ty) += sum;
        }
        
        __syncthreads();
    }
    
    // Store result back to global memory
    if (tx < BM && ty < BN) {
        tCgC(tx, ty) = acc(tx, ty);
    }
}

torch::Tensor gemm_cute(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be on CUDA");
    TORCH_CHECK(B.is_cuda(), "B must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Only float32 supported");
    
    const int M = A.size(0);
    const int N = B.size(1); 
    const int K = A.size(1);
    
    TORCH_CHECK(K == B.size(0), "Inner dimensions must match");
    
    auto C = torch::zeros({M, N}, A.options());
    
    // Launch configuration
    dim3 block(16, 16);
    dim3 grid((N + 127) / 128, (M + 127) / 128);
    
    gemm_cute_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(), 
        C.data_ptr<float>(),
        M, N, K
    );
    
    cudaDeviceSynchronize();
    return C;
}
"""

cpp_source = """
#include <torch/extension.h>

torch::Tensor gemm_cute(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_cute", &gemm_cute, "CuTe GEMM");
}
"""

# Load the CUDA extension
gemm_module = load_inline(
    name='gemm_cute_module',
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    extra_include_paths=['/opt/cutlass/include'],
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_90a'],
    verbose=True
)

def solution(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Optimized matrix multiplication using CuTe abstractions.
    
    Args:
        A: Input tensor of shape (M, K)
        B: Input tensor of shape (K, N)
        
    Returns:
        C: Output tensor of shape (M, N) where C = A @ B
    """
    # Ensure tensors are on GPU and contiguous
    A = A.cuda().contiguous()
    B = B.cuda().contiguous()
    
    return gemm_module.gemm_cute(A, B)

class Model(nn.Module):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return solution(A, B)

N = 2048

def get_inputs():
    A = torch.randn(N, N)
    B = torch.randn(N, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed```

### Why this code is wrong (root cause)
- The kernel defines CTA tile sizes `BM=128`, `BN=128`, `BK=32` but launches `dim3 block(16, 16)`.
- Compute path uses:
  - `if (tx < BM && ty < BN)` (always true for 16x16 block)
  - `sum += smemA(tx, kk) * smemB(kk, ty)`
  - `tCgC(tx, ty) = acc(tx, ty)`
- That means each block only computes a `16x16` corner of an intended `128x128` tile. Most C tile elements are never produced.
- Result: large numerical mismatch (`max_diff=243.18019104003906`).

## 2) Current CuTe Tutorial Code Extracted From CUTLASS
Extraction script: `extract_cute_docs.py`
Output directory: `outputs/cute_docs/`

### Tutorial files present
From `outputs/cute_docs/tutorial_listing.txt`:
```text
CMakeLists.txt
blackwell
hopper
sgemm_1.cu
sgemm_2.cu
sgemm_sm70.cu
sgemm_sm80.cu
tiled_copy.cu
tiled_copy_if.cu
```

### CuTe include hits from current examples
From `outputs/cute_docs/include_hits.txt`:
```text
/opt/cutlass/examples/111_hopper_ssd/111_hopper_ssd.cu
/opt/cutlass/examples/112_blackwell_ssd/112_blackwell_ssd.cu
/opt/cutlass/examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm.cu
/opt/cutlass/examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu
/opt/cutlass/examples/50_hopper_gemm_with_epilogue_swizzle/50_hopper_gemm_with_epilogue_swizzle.cu
/opt/cutlass/examples/54_hopper_fp8_warp_specialized_gemm/54_hopper_fp8_warp_specialized_gemm.cu
/opt/cutlass/examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_bf16_gemm.cu
/opt/cutlass/examples/55_hopper_mixed_dtype_gemm/55_hopper_int4_fp8_gemm.cu
/opt/cutlass/examples/55_hopper_mixed_dtype_gemm/55_hopper_mixed_dtype_gemm.cu
/opt/cutlass/examples/56_hopper_ptr_array_batched_gemm/56_hopper_ptr_array_batched_gemm.cu
```

### Key `sgemm_1.cu` excerpt (current baseline partitioning style)
Source: `outputs/cute_docs/tutorial/sgemm_1.cu`
```cpp

  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  // Shared memory buffers
  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K)

  //
  // Partition the copying of A and B tiles across the threads
  //

  // TUTORIAL: Example of simple raked partitioning of ThreadLayouts tA|tB over data A|B tiles

  Tensor tAgA = local_partition(gA, tA, threadIdx.x);                  // (THR_M,THR_K,k)
  Tensor tAsA = local_partition(sA, tA, threadIdx.x);                  // (THR_M,THR_K)

  Tensor tBgB = local_partition(gB, tB, threadIdx.x);                  // (THR_N,THR_K,k)
  Tensor tBsB = local_partition(sB, tB, threadIdx.x);                  // (THR_N,THR_K)

  CUTE_STATIC_ASSERT_V(size<0>(tAgA) == size<0>(tAsA));                // THR_M
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // THR_K
  CUTE_STATIC_ASSERT_V(size<0>(tBgB) == size<0>(tBsB));                // THR_N
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // THR_K

  //
  // Define A/B partitioning and C accumulators
  //

  // TUTORIAL: Example of partitioning via projections of a ThreadLayout tC

  // Partition sA (BLK_M, BLK_K) by the rows of tC
  Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});   // (THR_M,BLK_K)
  // Partition sB (BLK_N, BLK_K) by the cols of tC
  Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step< X,_1>{});   // (THR_N,BLK_K)
  // Partition gC (M,N) by the tile of tC
  Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{});   // (THR_M,THR_N)

  // Allocate the accumulators -- same shape/layout as the partitioned data
  Tensor tCrC = make_tensor_like(tCgC);                                // (THR_M,THR_N)

  CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCgC));                // THR_M
  CUTE_STATIC_ASSERT_V(size<0>(tCrC) == size<0>(tCsA));                // THR_M
  CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<1>(tCgC));                // THR_N
  CUTE_STATIC_ASSERT_V(size<1>(tCrC) == size<0>(tCsB));                // THR_N
  CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCsB));                // BLK_K

  // Clear the accumulators
  clear(tCrC);

#if 0
  if(thread0()) {
    print("  mA : "); print(  mA); print("\n");
    print("  gA : "); print(  gA); print("\n");
    print("  sA : "); print(  sA); print("\n");
    print("tAgA : "); print(tAgA); print("\n");
    print("tAsA : "); print(tAsA); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mB : "); print(  mB); print("\n");
    print("  gB : "); print(  gB); print("\n");
    print("  sB : "); print(  sB); print("\n");
    print("tBgB : "); print(tBgB); print("\n");
    print("tBsB : "); print(tBsB); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mC : "); print(  mC); print("\n");
    print("  gC : "); print(  gC); print("\n");
    print("tCsA : "); print(tCsA); print("\n");
    print("tCsB : "); print(tCsB); print("\n");
    print("tCgC : "); print(tCgC); print("\n");
    print("tCrC : "); print(tCrC); print("\n");
  }
#endif

#if 1

  // TUTORIAL: Example of a simple mainloop that read tiles of data into shared memory,
  //           and then computes on those tiles.
  //   copy(.) operates on the global and shared memory via the tA|tB partitioning
  //   gemm(.) operates on the shared and register memory via the tC partitioning

  auto K_TILE_MAX = size<2>(tAgA);

  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
    // Copy gmem to smem with tA|tB thread-partitioned tensors
    copy(tAgA(_,_,k_tile), tAsA);      // A   (THR_M,THR_K) -> (THR_M,THR_K)
    copy(tBgB(_,_,k_tile), tBsB);      // B   (THR_N,THR_K) -> (THR_N,THR_K)

    // TUTORIAL: The above call to copy(tAgA(_,_,k_tile), tAsA) is equivalent to
    //   Tensor tAgAk = tAgA(_,_,k_tile);
    //   CUTE_UNROLL
    //   for (int i = 0; i < size(tAsA); ++i) {
    //     tAsA(i) = tAgAk(i);
    //   }

    cp_async_fence();        // Label the end of (potential) cp.async instructions
    cp_async_wait<0>();      // Sync on all (potential) cp.async instructions
    __syncthreads();         // Wait for all threads to write to smem

    // Compute gemm on tC thread-partitioned smem
    gemm(tCsA, tCsB, tCrC);            // (THR_M,THR_N) += (THR_M,BLK_K) * (THR_N,BLK_K)

    // TUTORIAL: The above call to gemm(tCsA, tCsB, tCrC) is equivalent to
    //   CUTE_UNROLL
    //   for (int k = 0; k < size<1>(tCsA); ++k) {
    //     CUTE_UNROLL
    //     for (int m = 0; m < size<0>(tCrC); ++m) {
    //       CUTE_UNROLL
    //       for (int n = 0; n < size<1>(tCrC); ++n) {
    //         tCrC(m,n) += tCsA(m,k) * tCsB(n,k);
    //       }
    //     }
    //   }

    __syncthreads();         // Wait for all threads to read from smem
  }

#endif

  //
  // Epilogue
  //
```

### Key `sgemm_2.cu` excerpt (current TiledCopy/TiledMMA idiom)
Source: `outputs/cute_docs/tutorial/sgemm_2.cu`
```cpp

  // Shared memory buffers
  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K)

  //
  // Partition the copying of A and B tiles across the threads
  //

  // TUTORIAL: Example of partitioning via a TiledCopy

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K)
  // Allocate registers same shape/layout as partitioned data
  Tensor tArA = make_fragment_like(tAsA);                              // (CPY,CPY_M,CPY_K)

  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K)
  // Allocate registers same shape/layout as partitioned data
  Tensor tBrB = make_fragment_like(tBsB);                              // (CPY,CPY_N,CPY_K)

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tArA));                // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tArA));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBrB));                // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBrB));                // CPY_K

  // Copy gmem to rmem for k_tile=0
  copy(copy_a, tAgA(_,_,_,0), tArA);
  copy(copy_b, tBgB(_,_,_,0), tBrB);
  //
  // Define A/B partitioning and C accumulators
  //

  // TUTORIAL: Example of partitioning via a TiledMMA

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K)
  Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

  CUTE_STATIC_ASSERT_V(  shape(tCrC) ==   shape(tCgC));                // (MMA,MMA_M,MMA_N)
  CUTE_STATIC_ASSERT_V(size<1>(tCgC) == size<1>(tCsA));                // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(tCgC) == size<1>(tCsB));                // MMA_N
  CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                // MMA_K

  // Clear the accumulators
  clear(tCrC);

#if 0
  if(thread0()) {
    print("  mA : "); print(  mA); print("\n");
    print("  gA : "); print(  gA); print("\n");
    print("  sA : "); print(  sA); print("\n");
    print("tAgA : "); print(tAgA); print("\n");
    print("tAsA : "); print(tAsA); print("\n");
    print("tArA : "); print(tArA); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mB : "); print(  mB); print("\n");
    print("  gB : "); print(  gB); print("\n");
    print("  sB : "); print(  sB); print("\n");
    print("tBgB : "); print(tBgB); print("\n");
    print("tBsB : "); print(tBsB); print("\n");
    print("tArA : "); print(tArA); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mC : "); print(  mC); print("\n");
    print("  gC : "); print(  gC); print("\n");
    print("tCsA : "); print(tCsA); print("\n");
    print("tCsB : "); print(tCsB); print("\n");
    print("tCgC : "); print(tCgC); print("\n");
    print("tCrC : "); print(tCrC); print("\n");
  }
#endif

#if 1

  // TUTORIAL: Example of an inner loop that pipelines compute with reads
  //           from global memory by staging through register and shared memory.
  //   Data is read from global to registers, then to shared via the TiledCopy partitions
  //   gemm(.) operates on the shared memory directly via the TiledMMA partitions

  auto K_TILE_MAX = size<3>(tAgA);

  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
    // Copy rmem to smem with tA|tB thread-partitioned tensors
    __syncthreads();         // Wait for all threads to consume smem
    copy(tArA, tAsA);
    copy(tBrB, tBsB);
    __syncthreads();         // Wait for all threads to consume smem

    // Copy gmem to rmem for k_tile+1 with tA|tB thread-partitioned tensors
    int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
    copy(copy_a, tAgA(_,_,_,k_tile_next), tArA);
    copy(copy_b, tBgB(_,_,_,k_tile_next), tBrB);
    // TUTORIAL: The above call to copy(copy_a, tAgA(_,_,_,k_tile_next), tArA) is equivalent to
    //   CUTE_UNROLL
    //   for (int k = 0; k < size<1>(tCsA); ++k) {
    //     CUTE_UNROLL
    //     for (int m = 0; m < size<0>(tCrC); ++m) {
    //       copy_a.call(tAgA(_,m,k), tArA(_,m,k);
    //     }
    //   }

    // Compute gemm on mma-partitioned smem
    gemm(mma, tCsA, tCsB, tCrC);
    // TUTORIAL: The above call to gemm(tCsA, tCsB, tCrC) is equivalent to
    //   CUTE_UNROLL
    //   for (int k = 0; k < size<1>(tCsA); ++k) {
    //     CUTE_UNROLL
    //     for (int m = 0; m < size<0>(tCrC); ++m) {
    //       CUTE_UNROLL
    //       for (int n = 0; n < size<1>(tCrC); ++n) {
    //         mma.call(tCsA(_,m,k), tCsB(_,n,k), tCrC(_,m,n);
    //       }
    //     }
    //   }
  }

#endif

  //
  // Epilogue
  //

  axpby(alpha, tCrC, beta, tCgC);
}

// Setup params for a NT GEMM
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm_nt(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        cudaStream_t stream = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define NT strides (mixed)
  auto dA = make_stride(Int<1>{}, ldA);                      // (dM, dK)
  auto dB = make_stride(Int<1>{}, ldB);                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<  8>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM, bK));                 // (m,k) -> smem_idx; m-major
  auto sB = make_layout(make_shape(bN, bK));                 // (n,k) -> smem_idx; n-major
  auto sC = make_layout(make_shape(bM, bN));                 // (m,n) -> smem_idx; m-major

  // Define the thread layouts (static)

  // TUTORIAL: Construct TiledCopy with a particular Copy_Atom to use and
  //           define the partitioning pattern to apply.
  // Each thread will (try to) copy 4x1 elements of type TA using 128-bit copy.
  // Use 32x8 of these threads.

  TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TA>{},
                                    Layout<Shape<_32,_8>>{},  // Thr layout 32x8 m-major
                                    Layout<Shape< _4,_1>>{}); // Val layout  4x1 m-major
  TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TB>{},
                                    Layout<Shape<_32,_8>>{},  // Thr layout 32x8 n-major
                                    Layout<Shape< _4,_1>>{}); // Val layout  4x1 n-major

  // TUTORIAL: Construct TiledMMA with a particular MMA_Atom to use and
  //           define the partitioning pattern to apply.
  // Use a 1x1x1 FMA on the types TC += TA * TB. Each atom requires a single thread.
  // Reproduce that atom 16x16x1 times (m-major) across threads so that we use 256 threads.

  TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC,TA,TB>{},
                                 Layout<Shape<_16,_16,_1>>{});  // 16x16x1 UniversalFMA

#if 0
  print(copyA);
  print(copyB);
  print(mmaC);
#endif

#if 0
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, copyA,
       B, dB, sB, copyB,
       C, dC, sC, mmaC,
       alpha, beta);
}

// Setup params for a TN GEMM
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm_tn(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
```

### Notes on README lookup
- Attempted path: `/opt/cutlass/include/cute/README.md`
- Extracted file `outputs/cute_docs/cute_README.md` is empty in this environment.
- The actionable, up-to-date CuTe guidance is in tutorial sources under `examples/cute/tutorial/`.

## 3) Prompt vs Current Docs Gap Analysis

### What old prompt taught
Old `src/prompts/cute_system.py` used a simplified template centered on:
- `local_tile` + manual loop over K
- no explicit `ThrCopy = copy_a.get_slice(threadIdx.x)`
- no explicit `ThrMMA = mma.get_slice(threadIdx.x)`
- no explicit `partition_S/D`, `partition_A/B/C` sequence
- no hard launch contract tying block size to `size(mmaC)`

### What current tutorials require (`sgemm_2.cu`)
- Cooperative copy path:
  - `ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x)`
  - `partition_S` / `partition_D`
  - staged register fragments (`tArA`, `tBrB`)
- Cooperative MMA path:
  - `ThrMMA thr_mma = mma.get_slice(threadIdx.x)`
  - `partition_A`, `partition_B`, `partition_C`
  - accumulator via `thr_mma.make_fragment_C(...)`
- Launch dimensions derived from tiler object:
  - `dim3 dimBlock(size(mmaC))`
  - not ad hoc `16x16` when using larger CTA tiles

### API gap conclusion
Yes, the gap is primarily prompt-level:
- Missing thread-partition and mma-partition idioms caused models to generate tile/thread mismatches.
- Missing launch contract enabled incorrect partial-tile compute.
- Include path confusion (`/opt/cutlass/include/include`) also caused compile failures in earlier runs.

## 4) Updated Prompt (`src/prompts/cute_system.py`)

### Full current prompt file (no truncation)
```python
"""System prompts for CuTe kernel generation."""


def get_cute_system_prompt(gpu_name: str, vram_gb: int, use_xml_tools: bool = False) -> str:
    """Generate tool-use system prompt for CuTe-based solutions."""

    header = (
        f"You are an expert GPU kernel engineer. You have SSH access to an NVIDIA {gpu_name} GPU "
        f"({vram_gb}GB VRAM)."
    )

    body = """

YOUR TASK: Optimize the PyTorch model in reference.py using CuTe (CUTLASS 3.x) abstractions.

ENVIRONMENT:
- CUTLASS/CuTe headers are installed at: `/opt/cutlass/include`
- Use exactly: `extra_include_paths=['/opt/cutlass/include']`
- Do NOT use `/opt/cutlass/include/include` (wrong path)
- CUDA toolkit is at: `/usr/local/cuda`

REQUIREMENTS:
1. Use CuTe headers:
   - `cute/tensor.hpp`
   - `cute/layout.hpp`
2. Use CuTe concepts:
   - Tensors via `make_tensor`
   - CTA tiling via `local_tile`
   - Thread partitioning via `TiledCopy` (`get_slice`, `partition_S`, `partition_D`)
   - MMA partitioning via `TiledMMA` (`partition_A`, `partition_B`, `partition_C`)
3. Build the extension with `torch.utils.cpp_extension.load_inline`.
4. Keep evaluator compatibility with:
   - class `Model(nn.Module)`
   - `get_inputs()` and `get_init_inputs()`

CRITICAL SIGNATURE REQUIREMENT:
Your solution function MUST have this exact signature:

```python
def solution(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A is shape (M, K)
    # B is shape (K, N)
    # Return C of shape (M, N)
    ...
    return C
```

Do NOT use signatures like:
- `gemm_cute(A, B, C)`  # Wrong: C should be created inside
- `solution(A, B, C)`   # Wrong: only 2 inputs allowed

Model contract:
- `Model.forward(self, A, B)` must call `solution(A, B)` and return only `C`.

DO NOT USE:
- Raw CUDA-only final loops as the main implementation
- CUTLASS 2.x-only APIs
- Triton
- Python imports like `import cute` (usually unavailable at runtime)

REQUIRED WORKFLOW:
1. `cat /workspace/reference.py`
2. Write `/workspace/solution.py`
3. Compile check:
   `python -c "from solution import Model; print('OK')"`
4. Submit with `solution.py`

IMPORTANT:
- Correct and compilable code first.
- After compile check prints `OK`, submit immediately.
- DO NOT implement fallback paths. If CuTe headers fail to include, your solution should fail to compile.
- DO NOT use `#ifdef CUTE_AVAILABLE` or any basic-CUDA fallback branch.

CURRENT CuTe PATTERN (from CUTLASS examples/cute/tutorial/sgemm_2.cu):
```cpp
#include <cute/tensor.hpp>
using namespace cute;

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class CSmemLayout, class TiledMma>
__global__ void gemm_device(
    ProblemShape shape_MNK, CtaTiler cta_tiler,
    TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
    TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
    TC* C, CStride dC, CSmemLayout, TiledMma mma
) {
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});

  __shared__ TA smemA[/* cosize_v<ASmemLayout> */];
  __shared__ TB smemB[/* cosize_v<BSmemLayout> */];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tAsA = thr_copy_a.partition_D(sA);
  Tensor tArA = make_fragment_like(tAsA);

  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);
  Tensor tBsB = thr_copy_b.partition_D(sB);
  Tensor tBrB = make_fragment_like(tBsB);

  copy(copy_a, tAgA(_,_,_,0), tArA);
  copy(copy_b, tBgB(_,_,_,0), tBrB);

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCgC = thr_mma.partition_C(gC);
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);
  clear(tCrC);

  int K_TILE_MAX = size<3>(tAgA);
  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
    __syncthreads();
    copy(tArA, tAsA);
    copy(tBrB, tBsB);
    __syncthreads();

    int k_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
    copy(copy_a, tAgA(_,_,_,k_next), tArA);
    copy(copy_b, tBgB(_,_,_,k_next), tBrB);

    gemm(mma, tCsA, tCsB, tCrC);
  }

  axpby(1.0f, tCrC, 0.0f, tCgC);
}
```

LAUNCH CONTRACT (CRITICAL):
```cpp
// Use CuTe object sizes for launch dimensions
dim3 dimBlock(size(mmaC));  // e.g. 256 threads, not arbitrary 16x16
dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bN)));
```

DO NOT hardcode a tiny thread block (e.g. 16x16) while indexing a 128x128 tile with `tx/ty`.
That computes only a corner tile and causes large numerical errors.

PYTHON SHAPE/SIGNATURE TEMPLATE:
```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r'''
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <torch/extension.h>

using namespace cute;

// Your CuTe kernel and torch::Tensor gemm_cute(torch::Tensor A, torch::Tensor B)
'''

ext = load_inline(
    name='cute_gemm',
    cpp_sources=[''],
    cuda_sources=[cuda_source],
    extra_include_paths=['/opt/cutlass/include'],  # EXACT PATH - DO NOT MODIFY
    extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr', '-arch=sm_90'],
    verbose=False
)

def solution(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return ext.gemm_cute(A.contiguous(), B.contiguous())

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return solution(A, B)
```
"""

    if use_xml_tools:
        xml_tools = """

TOOLS - Use XML format to call tools:

1. bash:
<tool_call><bash><command>YOUR_COMMAND_HERE</command></bash></tool_call>

2. submit:
<tool_call><submit><solution_path>solution.py</solution_path></submit></tool_call>
"""
        return header + body + xml_tools

    native_tools = """

TOOLS:
- bash: Execute shell commands
- submit: Submit your solution path
"""
    return header + body + native_tools


def get_cute_reasoning_system_prompt(gpu_name: str, vram_gb: int) -> str:
    """Generate reasoning-only system prompt for CuTe-based solutions."""

    header = (
        "You are an expert GPU kernel engineer. "
        f"TARGET GPU: NVIDIA {gpu_name} ({vram_gb}GB VRAM)."
    )

    body = """

Write a complete `solution.py` using CuTe abstractions from CUTLASS 3.x.

Hard requirements:
- Include and use `cute/tensor.hpp`, `cute/layout.hpp` in C++/CUDA extension code.
- Use `torch.utils.cpp_extension.load_inline` with:
  - `extra_include_paths=['/opt/cutlass/include']`
  - do not use `/opt/cutlass/include/include`
- Do not import Python `cute` modules.
- Keep `Model`, `get_inputs`, and `get_init_inputs` compatible with reference.
- Do not implement fallback paths. If CuTe include fails, fail compilation.
- Do not use `#ifdef CUTE_AVAILABLE` fallback branches.
- Use current CuTe partitioning idioms:
  - `copy_a.get_slice(threadIdx.x)` / `partition_S` / `partition_D`
  - `mma.get_slice(threadIdx.x)` / `partition_A` / `partition_B` / `partition_C`
  - launch with `dim3 dimBlock(size(mmaC))`, not arbitrary `16x16`.

Mandatory signature in Python:
```python
def solution(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    ...
    return C
```

Avoid:
- `solution(A, B, C)`
- `gemm_cute(A, B, C)` as the public API

Return only complete Python code in a markdown code block.
Start from a compilable baseline and only then tune.
"""

    return header + body
```

### What changed
- Added explicit current CuTe pattern based on `sgemm_2.cu`:
  - `TiledCopy` with `get_slice`, `partition_S`, `partition_D`
  - `TiledMMA` with `get_slice`, `partition_A/B/C`
  - pipelined loop with register/shared-memory staging
- Added strict launch contract:
  - `dim3 dimBlock(size(mmaC))`
  - explicit warning against tiny hardcoded blocks for large tiles
- Kept exact include path requirement:
  - `extra_include_paths=['/opt/cutlass/include']`
- Kept no-fallback requirement.

## 5) Modal Extraction Script
Created: `extract_cute_docs.py`

Behavior:
- Spins up Modal function using CUDA 13.1 base image.
- Clones CUTLASS into `/opt/cutlass`.
- Extracts tutorial files and include-hit list.
- Saves locally under `outputs/cute_docs/`.

Run output:
- `Wrote CuTe docs to outputs/cute_docs`
- `Tutorial files: 6`
- `Include hits: 10`

## 6) Post-Update Validation Run
Command:
```bash
uv run python cute_batch_eval.py --models anthropic/claude-sonnet-4 --gpus H100 --levels 1 --problems-per-level 1 --max-turns 5 --sequential
```
Run directory:
- `outputs/batch_eval/run_20260213_160148`

Result (`results.jsonl`):
```json
{"model": "Anthropic: Claude Sonnet 4", "gpu": "H100", "problem": "1_Square_matrix_multiplication_.py", "level": 1, "compiled": false, "correct": false, "speedup": null, "ref_ms": null, "sol_ms": null, "turns": 1, "submitted": true, "error": "module 'torch.utils' has no attribute 'cpp_extension'", "elapsed_seconds": 76.94439435005188, "input_tokens": 722, "output_tokens": 4554, "total_tokens": 5276, "cache_creation_tokens": 0, "cache_read_tokens": 0, "estimated_cost_usd": 0.070476, "ref_kernels": null, "sol_kernels": null}
```

Interpretation:
- Model now attempted more CuTe-specific code, but failed at runtime due wrapper issue:
  - `module 'torch.utils' has no attribute 'cpp_extension'`
- This is a generated-code bug (using `torch.utils.cpp_extension.load_inline` without importing `load_inline` path correctly for this runtime), not a CUTLASS include-path issue.

## 7) Current Status
- CuTe prompt has been updated to current CUTLASS tutorial idioms.
- Docs extraction pipeline exists and is reproducible (`extract_cute_docs.py`).
- Claude still does not pass end-to-end correctness/benchmarking after the prompt refresh.
- Next likely bottleneck is generated wrapper robustness (`load_inline` import/usage and extension wiring), then numerical/kernel correctness.
