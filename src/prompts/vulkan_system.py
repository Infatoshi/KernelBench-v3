"""System prompts for Vulkan compute-shader generation."""


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

REQUIRED WORKFLOW:
1. `cat /workspace/reference.py`
2. Write `/workspace/solution.py`
3. `python -c "from solution import Model; print('OK')"`
4. Submit `solution.py`
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
