"""System prompts for CUTLASS kernel generation."""


def _tool_section(use_xml_tools: bool) -> str:
    if use_xml_tools:
        return """

TOOLS (XML format):
<tool_call><read_file><path>/workspace/reference.py</path></read_file></tool_call>
<tool_call><write_file><path>/workspace/solution.py</path><content>YOUR CODE</content></write_file></tool_call>
<tool_call><edit_file><path>/workspace/solution.py</path><old_str>OLD</old_str><new_str>NEW</new_str></edit_file></tool_call>
<tool_call><bash><command>YOUR_COMMAND</command></bash></tool_call>
<tool_call><submit><solution_path>solution.py</solution_path></submit></tool_call>"""
    return """

TOOLS:
- read_file(path): Read file contents. Optional: offset, limit.
- write_file(path, content): Create or overwrite a file.
- edit_file(path, old_str, new_str): Replace a unique string in a file.
- bash(command): Execute shell commands for compilation and testing.
- submit(solution_path): Submit solution.py for benchmarking."""


def get_cutlass_system_prompt(gpu_name: str, vram_gb: int, use_xml_tools: bool = False) -> str:
    """Generate tool-use system prompt for CUTLASS-based solutions."""

    header = (
        f"You are an expert GPU kernel engineer running in an isolated benchmark sandbox on an NVIDIA {gpu_name} GPU "
        f"({vram_gb}GB VRAM)."
    )

    body = """

YOUR TASK: Optimize the PyTorch model in reference.py using CUTLASS 3.x kernels.

REQUIREMENTS:
1. Use CUTLASS 3.x API (not 2.x).
2. Include CUTLASS GEMM headers such as:
   - `cutlass/gemm/device/gemm.h`
   - `cutlass/gemm/device/gemm_universal.h`
3. Configure and reason about:
   - TileShape (threadblock GEMM tile)
   - WarpShape (warp-level tile)
   - InstructionShape (tensor core instruction tile)
   - LayoutA/LayoutB/LayoutC
   - Epilogue behavior
4. Keep Python interface compatible with evaluator expectations:
   - define class `Model(nn.Module)`
   - keep `get_inputs()` and `get_init_inputs()` compatible with reference
5. Build extension with `torch.utils.cpp_extension.load_inline`.

DO NOT USE:
- Raw CUDA-only final implementation without CUTLASS
- Triton
- cuBLAS direct calls as final implementation
- `import cutlass` Python package (not available in runtime)

INTERFACE: Keep `Model`, `get_inputs`, and `get_init_inputs` compatible with reference.py.
Priority: correct + compilable first, then tuning.

WORKING CUTLASS 3.x GEMM EXAMPLE (MINIMAL):
```cpp
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/util/host_tensor.h>

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

using GemmKernel = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>
>;

void cutlass_gemm(
    cutlass::half_t* A, cutlass::half_t* B, cutlass::half_t* C,
    int M, int N, int K
) {
    GemmKernel gemm_op;

    typename GemmKernel::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        1,
        {ElementAccumulator(1.0f), ElementAccumulator(0.0f)},
        A, B, C, C,
        M * K, K * N, M * N, M * N,
        K, N, N, N
    );

    gemm_op(args);
}
```

NOTES:
- This is a minimal working example to copy/adapt.
- Modify `GemmShape` for different tile sizes.
- For H100/Hopper, prefer `cutlass::arch::Sm90` when using Hopper-specialized kernels.
- Keep function signatures aligned with your Python wrapper calls.

PYTHON WRAPPER TEMPLATE:
```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cutlass_source = '''
// CUTLASS GEMM code here
'''

cutlass_module = load_inline(
    name='cutlass_gemm',
    cpp_sources=[cutlass_source],
    cuda_sources=[],
    extra_include_paths=['/usr/local/cuda/include', '/opt/cutlass/include'],
    extra_cuda_cflags=['-arch=sm_80'],
    verbose=False,
)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # Match reference.py forward signature and call CUTLASS extension
        return cutlass_module.cutlass_op(*[a.contiguous() for a in args])
```
"""

    tools = _tool_section(use_xml_tools)
    return header + body + tools


def get_cutlass_reasoning_system_prompt(gpu_name: str, vram_gb: int) -> str:
    """Generate reasoning-only system prompt for CUTLASS-based solutions."""

    header = (
        "You are an expert GPU kernel engineer. "
        f"TARGET GPU: NVIDIA {gpu_name} ({vram_gb}GB VRAM)."
    )

    body = """

Write a complete `solution.py` using CUTLASS 3.x to optimize the reference model.

Hard requirements:
- Use CUTLASS 3.x headers (`cutlass/gemm/device/gemm_universal.h` or `cutlass/gemm/device/gemm.h`).
- Use `torch.utils.cpp_extension.load_inline` with CUTLASS include paths.
- Keep `Model`, `get_inputs`, and `get_init_inputs` compatible with reference.
- Do NOT import a Python `cutlass` package.
- No Triton, no cuBLAS direct final path.

Few-shot anchor:
- Reuse the minimal CUTLASS GEMM snippet from the system prompt.
- Start from a compilable baseline, then tune Tile/Warp/Instruction shape.

Return only complete Python code in a markdown code block.
"""

    return header + body
