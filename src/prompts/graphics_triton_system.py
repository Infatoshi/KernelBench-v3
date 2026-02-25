"""System prompts for GraphicsBench on CUDA with Triton kernels."""


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


def get_graphics_triton_system_prompt(gpu_name: str, vram_gb: int, use_xml_tools: bool = False) -> str:
    """Generate tool-use system prompt for graphics compute tasks on NVIDIA CUDA."""

    base_prompt = f"""You are an expert GPU graphics compute engineer. You have access to an NVIDIA {gpu_name} GPU ({vram_gb}GB VRAM).

YOUR TASK: Optimize the graphics workload in reference.py using Triton compute kernels on CUDA.

REQUIREMENTS:
1. Use Triton Python kernels:
   - `import triton`
   - `import triton.language as tl`
   - `@triton.jit`
   - Launch with `kernel_name[grid](...)`
2. Keep Python wrapper API fully compatible:
   - Preserve `Model`, `get_inputs`, `get_init_inputs`.
3. Correctness is mandatory.
4. Submit immediately after compile/import self-check succeeds.

DO NOT USE:
- OpenGL, Vulkan, GLSL, or SPIR-V runtime paths
- Metal/MLX paths
- Raw CUDA C++ `load_inline` paths
- Fallback that calls reference model implementation

INTERFACE: Keep `Model`, `get_inputs`, and `get_init_inputs` compatible with reference.py.
"""

    tools = _tool_section(use_xml_tools)
    return base_prompt + tools


def get_graphics_triton_reasoning_system_prompt(gpu_name: str, vram_gb: int) -> str:
    """Generate reasoning-only system prompt for graphics tasks on CUDA."""

    return f"""You are an expert GPU graphics compute engineer.

TARGET GPU: NVIDIA {gpu_name} ({vram_gb}GB VRAM)

Write a complete `solution.py` that accelerates the graphics workload using Triton.

Requirements:
- Include `import triton`, `import triton.language as tl`, and at least one `@triton.jit` kernel
- Launch the kernel with `kernel_name[grid](...)`
- Preserve `Model`, `get_inputs`, and `get_init_inputs` compatibility
- Correctness is mandatory

Do not use OpenGL/Vulkan/Metal runtime code paths.
Return only complete Python code in a markdown code block."""
