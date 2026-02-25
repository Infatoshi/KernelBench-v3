"""System prompts for Triton kernel generation."""


def _tool_section(use_xml_tools: bool) -> str:
    """Return the standard tool description block for all backend prompts."""
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

INTERFACE: Keep `Model`, `get_inputs`, and `get_init_inputs` compatible with reference.py.

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

    tools = _tool_section(use_xml_tools)
    return base_prompt + tools


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
