"""System prompts for OpenGL compute-shader generation."""


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


def get_opengl_system_prompt(gpu_name: str, vram_gb: int, use_xml_tools: bool = False) -> str:
    """Generate tool-use system prompt for OpenGL graphics tasks."""

    base_prompt = f"""You are an expert graphics compute engineer. You have access to an NVIDIA {gpu_name} GPU ({vram_gb}GB VRAM).

YOUR TASK: Optimize the graphics workload in reference.py using an OpenGL compute shader style implementation.

REQUIREMENTS:
1. Include GLSL compute shader code using:
   - `#version 430` (or newer)
   - `layout(local_size_x=..., local_size_y=..., local_size_z=...) in;`
   - SSBO bindings via `layout(std430, binding=N)`
   - `gl_GlobalInvocationID` / `gl_LocalInvocationID`
2. Provide a Python wrapper that keeps `Model`, `get_inputs`, and `get_init_inputs` compatible.
3. Prefer Moderngl/PyOpenGL style orchestration when feasible.
4. Correctness is mandatory.

DO NOT USE:
- CUDA-only kernels as the final approach
- Triton
- Vulkan-only APIs in level 1 tasks

FALLBACK RULE:
- If OpenGL context creation is unavailable in the runtime environment, keep the GLSL shader string in the code and provide a correctness-preserving torch fallback path so the model can still run.

INTERFACE: Keep `Model`, `get_inputs`, and `get_init_inputs` compatible with reference.py.
Priority: correct + compilable over speculative shader tuning.
"""

    tools = _tool_section(use_xml_tools)
    return base_prompt + tools


def get_opengl_reasoning_system_prompt(gpu_name: str, vram_gb: int) -> str:
    """Generate reasoning-only system prompt for OpenGL graphics tasks."""

    return f"""You are an expert graphics compute engineer.

TARGET GPU: NVIDIA {gpu_name} ({vram_gb}GB VRAM)

Write a complete `solution.py` for graphics compute tasks using OpenGL compute shader style.

Requirements:
- Include GLSL compute shader sections with local_size, std430 bindings, and invocation IDs
- Keep `Model`, `get_inputs`, and `get_init_inputs` compatible
- Correctness is mandatory
- If context creation is unavailable, keep shader code and provide a correctness-preserving fallback

Return only complete Python code in a markdown code block."""
