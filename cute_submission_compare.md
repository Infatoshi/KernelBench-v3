# CuTe Claude Submission Comparison (Old vs New)

## Artifacts Compared
- Old submission: `outputs/batch_eval/run_20260213_001820/kernels/Anthropic:_Claude_Sonnet_4_H100_1_Square_matrix_multiplication_/solution.py`
- New submission: `outputs/batch_eval/run_20260213_174806/kernels/Anthropic:_Claude_Sonnet_4_H100_1_Square_matrix_multiplication_/solution.py`
- Reference pattern: `outputs/cute_docs/tutorial/sgemm_2.cu`

Both runs failed with the same benchmark error:
- `seed=42, max_diff=243.18019104003906`

## Side-by-Side Comparison

| Aspect | Old Submission | New Submission | sgemm_2.cu Reference |
|---|---|---|---|
| Uses `TiledMMA` | No | No | Yes (`make_tiled_mma`, `mma.get_slice`) |
| Uses `TiledCopy` | No | Superficially only (`make_tiled_copy` created, not used via partition APIs) | Yes (`get_slice` + `partition_S`/`partition_D`) |
| Thread partitioning | None (`local_tile` + manual `tx/ty`) | None for CuTe MMA/copy path | Required (`partition_S`, `partition_D`, `partition_A/B/C`) |
| Block config | `dim3 block(16,16)` (256 threads) with 128x128 tile | `dim3 dimBlock(64,64)` (4096 threads, invalid launch) | `dim3 dimBlock(size(mmaC))` (typically 256 threads) |
| Accumulator init | `make_fragment_like(tCgC); clear(acc)` | `float acc = 0.0f` per thread | `tCrC = thr_mma.make_fragment_C(...); clear(tCrC)` |
| Writeback method | `tCgC(tx, ty) = acc(tx, ty)` | `gC(global_m, global_n) = acc` | `axpby(alpha, tCrC, beta, tCgC)` |
| Uses `gemm(mma, ...)` | No | No | Yes |

## Root Cause Analysis

### 1) Old submission root cause (partial tile computation)
Key lines (old file):
- Tile sizes: `BM=128, BN=128` (`line 20-22`)
- Launch: `dim3 block(16, 16)` (`line 98`)
- Compute only `acc(tx,ty)` and write `tCgC(tx,ty)` (`line 66-72`, `line 78-79`)

This computes only a 16x16 corner of each 128x128 tile. Most outputs remain at initialization (zeros), producing large error.

### 2) New submission root cause (invalid launch + non-CuTe kernel)
Key lines (new file):
- Launch: `dim3 dimBlock(TILE_N, TILE_M)` with `TILE_N=TILE_M=64` (`line 132`, constants at `line 11-13`)
- That is 4096 threads/block, exceeding CUDA's max 1024 threads/block.
- Kernel launch likely fails; code calls `cudaDeviceSynchronize()` but does not check return code (`line 143`), so output tensor `C` (initialized zeros) is returned.

Even ignoring launch failure, the kernel is still not true CuTe GEMM:
- `make_tiled_copy` and `get_slice` are created (`line 45-50`) but never used via `partition_S/partition_D`.
- No `TiledMMA`, no `partition_A/B/C`, no `gemm(mma, ...)` path.

## Why Same `max_diff` Appears in Both Runs
Same numeric value is consistent with mostly-zero output tensors against a large random GEMM reference where `max(abs(ref))` is around 243 on this seed.
- Old: partially written output (small corner of each tile) -> mostly zeros.
- New: likely invalid launch -> effectively no writes -> all zeros.

## Reference Pattern Required (from sgemm_2.cu)
Required elements missing from both submissions:
- `ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);`
- `tAgA = thr_copy_a.partition_S(gA); tAsA = thr_copy_a.partition_D(sA);`
- `ThrMMA thr_mma = mma.get_slice(threadIdx.x);`
- `tCsA = thr_mma.partition_A(sA); tCsB = thr_mma.partition_B(sB); tCgC = thr_mma.partition_C(gC);`
- `gemm(mma, tCsA, tCsB, tCrC);`
- launch with `dim3 dimBlock(size(mmaC));`

## Full Old Submission Code
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
    return []  # No special initialization inputs needed
```

## Full New Submission Code
```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <cute/tensor.hpp>
#include <cute/layout.hpp>

using namespace cute;

constexpr int TILE_M = 64;
constexpr int TILE_N = 64; 
constexpr int TILE_K = 16;

__global__ void gemm_cute_kernel(
    const float* A_ptr, const float* B_ptr, float* C_ptr,
    int M, int N, int K) {
    
    __shared__ float sA[TILE_M * TILE_K];
    __shared__ float sB[TILE_K * TILE_N];
    
    // Create global tensors with row-major layout
    auto gA = make_tensor(make_gmem_ptr(A_ptr), 
                         make_layout(make_shape(M, K), make_stride(K, 1)));
    auto gB = make_tensor(make_gmem_ptr(B_ptr), 
                         make_layout(make_shape(K, N), make_stride(N, 1)));  
    auto gC = make_tensor(make_gmem_ptr(C_ptr), 
                         make_layout(make_shape(M, N), make_stride(N, 1)));
    
    // Create shared memory tensors
    auto sA_tensor = make_tensor(make_smem_ptr(sA), 
                                make_layout(make_shape(TILE_M, TILE_K)));
    auto sB_tensor = make_tensor(make_smem_ptr(sB), 
                                make_layout(make_shape(TILE_K, TILE_N)));
    
    // Block coordinates
    int block_m = blockIdx.y;
    int block_n = blockIdx.x;
    
    // Create copy operations
    using CopyOp = Copy_Atom<UniversalCopy<float>, float>;
    auto copy_op = CopyOp{};
    
    // Simple tiled copy with 1D thread layout
    auto tiled_copy = make_tiled_copy(copy_op,
                                     make_layout(make_shape(64)), // 64 threads 
                                     make_layout(make_shape(1)));
    
    auto copy_a = tiled_copy.get_slice(threadIdx.x);
    auto copy_b = tiled_copy.get_slice(threadIdx.x);
    
    // Initialize accumulator
    float acc = 0.0f;
    
    // Main computation loop over K dimension
    for (int k_tile = 0; k_tile < (K + TILE_K - 1) / TILE_K; k_tile++) {
        
        // Calculate global offsets
        int gA_m_offset = block_m * TILE_M;
        int gA_k_offset = k_tile * TILE_K;
        int gB_k_offset = k_tile * TILE_K; 
        int gB_n_offset = block_n * TILE_N;
        
        // Copy A tile to shared memory using simple indexing
        int tid = threadIdx.x;
        int total_threads = blockDim.x;
        
        for (int i = tid; i < TILE_M * TILE_K; i += total_threads) {
            int sm_m = i / TILE_K;
            int sm_k = i % TILE_K;
            int gm_m = gA_m_offset + sm_m;
            int gm_k = gA_k_offset + sm_k;
            
            if (gm_m < M && gm_k < K) {
                sA_tensor(sm_m, sm_k) = gA(gm_m, gm_k);
            } else {
                sA_tensor(sm_m, sm_k) = 0.0f;
            }
        }
        
        // Copy B tile to shared memory
        for (int i = tid; i < TILE_K * TILE_N; i += total_threads) {
            int sm_k = i / TILE_N;
            int sm_n = i % TILE_N;
            int gm_k = gB_k_offset + sm_k;
            int gm_n = gB_n_offset + sm_n;
            
            if (gm_k < K && gm_n < N) {
                sB_tensor(sm_k, sm_n) = gB(gm_k, gm_n);
            } else {
                sB_tensor(sm_k, sm_n) = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute using shared memory
        int t_m = threadIdx.y;
        int t_n = threadIdx.x;
        
        if (t_m < TILE_M && t_n < TILE_N) {
            for (int k = 0; k < TILE_K; k++) {
                acc += sA_tensor(t_m, k) * sB_tensor(k, t_n);
            }
        }
        
        __syncthreads();
    }
    
    // Write results back
    int global_m = block_m * TILE_M + threadIdx.y;
    int global_n = block_n * TILE_N + threadIdx.x;
    
    if (global_m < M && global_n < N && threadIdx.y < TILE_M && threadIdx.x < TILE_N) {
        gC(global_m, global_n) = acc;
    }
}

torch::Tensor gemm_cute(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA tensor");
    
    int M = A.size(0);
    int K = A.size(1);  
    int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    // Launch configuration
    dim3 dimBlock(TILE_N, TILE_M);
    dim3 dimGrid((N + TILE_N - 1) / TILE_N, 
                 (M + TILE_M - 1) / TILE_M);
    
    gemm_cute_kernel<<<dimGrid, dimBlock>>>(
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
torch::Tensor gemm_cute(torch::Tensor A, torch::Tensor B);
"""

# Load the extension with proper include paths
gemm_extension = load_inline(
    name='gemm_cute',
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=['gemm_cute'],
    extra_include_paths=['/opt/cutlass/include'],
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_80', '-std=c++17']
)

class Model(nn.Module):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication using CuTe-optimized kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return solution(A, B)

def solution(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Ensure tensors are on GPU and float32
    A = A.cuda().float().contiguous()
    B = B.cuda().float().contiguous()
    
    return gemm_extension.gemm_cute(A, B)

N = 2048

def get_inputs():
    A = torch.randn(N, N)
    B = torch.randn(N, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
```

## Diagnostic Recommendation

### Add a tiny deterministic precheck before full correctness loop
In benchmark harness, before seed-based full-size checks:
1. Construct a tiny case on GPU (e.g., 4x4 or 8x8):
   - `A = I`, `B = I`, expected `C = I`
2. Run solution once and assert close to identity.
3. If failed, return explicit error:
   - `"diagnostic_failed: identity_case"`

This separates:
- fundamental indexing/launch/writeback bugs (fail tiny identity), from
- larger-tile or boundary issues (pass tiny case, fail full size).

### Optional second precheck (launch sanity)
After solution call, immediately check runtime error in extension code:
- `auto err = cudaGetLastError();`
- if `err != cudaSuccess`, raise with `cudaGetErrorString(err)`.

This would have made the new submission fail with an explicit invalid-launch error instead of a confusing numeric mismatch.

## Recommendation
1. Keep current prompt import fixes.
2. Add harness diagnostics (identity + launch error check) for fast triage.
3. For CuTe viability, either:
   - enforce a near-complete `sgemm_2`-style skeleton (model fills only tile/layout params), or
   - introduce a simpler CuTe Level-0 problem first (vector add / elementwise) before GEMM.
4. Treat current CuTe GEMM generation as **not yet reliable** across models.
