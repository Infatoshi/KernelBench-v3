"""System prompts for Vulkan compute-shader generation."""


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


def get_vulkan_system_prompt(gpu_name: str, vram_gb: int, use_xml_tools: bool = False) -> str:
    """Generate tool-use system prompt for Vulkan graphics tasks."""

    base_prompt = f"""You are an expert graphics compute engineer. You have access to an NVIDIA {gpu_name} GPU ({vram_gb}GB VRAM).

YOUR TASK: Optimize the graphics workload in reference.py using Vulkan compute style code.

REQUIREMENTS:
1. Use Vulkan-style compute shader concepts:
   - Descriptor sets and bindings
   - Push constants for dynamic parameters
   - Compute pipeline dispatch configuration
2. Shader code should reflect Vulkan-compatible GLSL/SPIR-V style organization.
3. Keep Python wrapper interface compatible with `Model`, `get_inputs`, `get_init_inputs`.
4. Correctness is mandatory.

DO NOT USE:
- OpenGL-only APIs as final approach for Vulkan levels
- CUDA-only kernels as final approach
- Triton

FALLBACK RULE:
- If Vulkan runtime setup is unavailable in this environment, include Vulkan-style shader/pipeline code and preserve correctness with a safe fallback execution path.

INTERFACE: Keep `Model`, `get_inputs`, and `get_init_inputs` compatible with reference.py.
"""

    tools = _tool_section(use_xml_tools)
    return base_prompt + tools


def get_vulkan_reasoning_system_prompt(gpu_name: str, vram_gb: int) -> str:
    """Generate reasoning-only system prompt for Vulkan graphics tasks."""

    return f"""You are an expert graphics compute engineer.

TARGET GPU: NVIDIA {gpu_name} ({vram_gb}GB VRAM)

Write a complete `solution.py` using Vulkan compute style architecture.

Requirements:
- Include descriptor set / push constant / compute dispatch structure
- Keep `Model`, `get_inputs`, and `get_init_inputs` compatible
- Preserve correctness
- Provide fallback path if Vulkan runtime setup is unavailable

Return only complete Python code in a markdown code block."""
