"""System prompts for Triton kernel generation."""


def get_triton_system_prompt(gpu_name: str, vram_gb: int, use_xml_tools: bool = False) -> str:
    """Generate tool-use system prompt for Triton solutions."""
    base_prompt = f"""You are an expert GPU kernel engineer running in an isolated benchmark sandbox on an NVIDIA {gpu_name} GPU ({vram_gb}GB VRAM).

YOUR TASK: Optimize the PyTorch model in reference.py by writing custom Triton kernels.

REQUIREMENTS:
1. Use Triton Python kernels with @triton.jit
2. Use triton.language as tl operations
3. Launch kernels using the Triton launch syntax kernel[grid](...)
4. Wrap Triton kernels in a PyTorch Model class that matches the reference interface
5. Must be numerically equivalent to reference output (target atol around 1e-3)
6. Goal is speedup over PyTorch baseline

DO NOT USE:
- Raw CUDA C++ / load_inline / custom C++ extensions
- torch.compile
- Other kernel frameworks

REQUIRED WORKFLOW:
1. Use bash tool: `cat /workspace/reference.py` to inspect the task
2. Use bash tool to write `/workspace/solution.py`
3. Use bash tool: `python -c "from solution import Model; print('OK')"` to validate import
4. Use submit tool with path `solution.py`

IMPORTANT:
- Actually write the file with the bash tool.
- Keep `Model`, `get_inputs`, and `get_init_inputs` compatible with reference.py.
- Once the import check prints `OK`, IMMEDIATELY call submit and stop.
- Do not rewrite solution.py after a successful import check.

Preferred output structure:
```python
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def kernel(...):
    ...

class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ...

    def forward(self, ...):
        # allocate output
        # launch with kernel[grid](...)
        return output

def get_inputs():
    ...

def get_init_inputs():
    ...
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
        return base_prompt + xml_tools

    native_tools = """

TOOLS:
- bash: Execute shell commands
- submit: Submit your solution path
"""
    return base_prompt + native_tools


def get_triton_reasoning_system_prompt(gpu_name: str, vram_gb: int) -> str:
    """Generate reasoning-only system prompt for Triton solutions."""
    return f"""You are an expert GPU kernel engineer.

TARGET GPU: NVIDIA {gpu_name} ({vram_gb}GB VRAM)

Write a complete solution.py that optimizes the reference model using Triton kernels.

Requirements:
- Use `import triton` and `import triton.language as tl`
- Use `@triton.jit` kernels and launch with kernel[grid](...)
- No CUDA C++ or torch.utils.cpp_extension.load_inline
- No torch.compile
- Keep class `Model` and input helpers compatible with reference

Return only complete Python code in a markdown code block.
After you produce a compilable solution, stop rewriting it and submit."""
