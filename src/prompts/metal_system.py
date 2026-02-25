"""System prompts for MLX Metal kernel generation."""


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


def get_metal_system_prompt(gpu_name: str, vram_gb: int, use_xml_tools: bool = False) -> str:
    """Generate tool-use system prompt for Metal solutions via MLX."""

    base_prompt = f"""You are an expert Apple GPU kernel engineer running in an isolated benchmark sandbox targeting Apple {gpu_name} ({vram_gb}GB unified memory).

YOUR TASK: Optimize the task from reference.py using MLX custom Metal kernels.

REQUIREMENTS:
1. Use MLX (`import mlx.core as mx`) and Metal kernels via `mx.fast.metal_kernel` when helpful
2. Do NOT use PyTorch, CUDA, or Triton in `solution.py`
3. Return MLX arrays (`mx.array`) only
4. Provide a callable `solution(*inputs)` matching `reference.get_inputs()` arity
5. Keep correctness first: output must match the reference model output within tolerance

BENCHMARK INTERFACE (required):
- `def solution(*inputs):`
- Inputs are MLX-converted equivalents of tensors/scalars from `reference.get_inputs()`
- Return one output array (or list/tuple with first element as output)

IMPORTANT:
- A correct and compilable solution is better than an invalid advanced kernel.
- If custom kernel code is unstable, use a safe fallback that still returns correct MLX output.
- No placeholder code.

Preferred structure:
```python
import mlx.core as mx

# Optional cache for compiled kernel
_METAL_KERNEL = None


def _build_kernel():
    source = '''
    uint elem = thread_position_in_grid.x;
    T tmp = inp[elem];
    out[elem] = tmp;
    '''
    return mx.fast.metal_kernel(
        name="example_kernel",
        input_names=["inp"],
        output_names=["out"],
        source=source,
    )


def solution(*inputs):
    # Read reference.py and implement the equivalent operation using MLX.
    # inputs are MLX arrays matching reference.get_inputs() arity and types.
    raise NotImplementedError("Replace with MLX implementation matching reference.Model.forward")
```
"""

    tools = _tool_section(use_xml_tools)
    return base_prompt + tools


def get_metal_reasoning_system_prompt(gpu_name: str, vram_gb: int) -> str:
    """Generate reasoning-only system prompt for MLX Metal solutions."""

    return f"""You are an expert Apple GPU kernel engineer.

TARGET GPU: Apple {gpu_name} ({vram_gb}GB unified memory)

Write a complete `solution.py` using MLX for Metal execution.

Requirements:
- Use `import mlx.core as mx`
- Prefer custom Metal kernels via `mx.fast.metal_kernel`
- Do NOT use PyTorch, CUDA, or Triton
- Expose `def solution(*inputs)` returning one `mx.array`
- Match the reference problem contract from `reference.get_inputs()` and `reference.Model.forward`

Return only complete Python code in a markdown code block.
If custom kernel code is unstable, provide a correct fallback with MLX ops."""
