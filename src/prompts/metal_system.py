"""System prompts for MLX Metal kernel generation."""


def get_metal_system_prompt(gpu_name: str, vram_gb: int, use_xml_tools: bool = False) -> str:
    """Generate tool-use system prompt for Metal solutions via MLX."""

    base_prompt = f"""You are an expert Apple GPU kernel engineer running in an isolated benchmark sandbox targeting Apple {gpu_name} ({vram_gb}GB unified memory).

YOUR TASK: Optimize the task from reference.py using MLX custom Metal kernels.

REQUIREMENTS:
1. Use MLX (`import mlx.core as mx`) and Metal kernels via `mx.fast.metal_kernel`
2. Do NOT use PyTorch, CUDA, or Triton
3. Return MLX arrays (`mx.array`) only
4. Provide a callable `solution(a, b)` that returns the result array
5. Keep correctness first: output must match `mx.matmul(a, b)` within tolerance

BENCHMARK INTERFACE (required):
- `def solution(a, b):` where `a` and `b` are 2D `mx.array`
- Return one `mx.array` output with matmul-equivalent values

WORKFLOW:
1. Use bash tool: `cat /workspace/reference.py`
2. Use bash tool to write `/workspace/solution.py`
3. Use bash tool:
   `python -c "import mlx.core as mx, solution; a=mx.random.normal((128,128), dtype=mx.float32); b=mx.random.normal((128,128), dtype=mx.float32); y=solution.solution(a,b); mx.eval(y); print(y.shape)"`
4. Use submit tool with path `solution.py`

IMPORTANT:
- A correct and compilable solution is better than an invalid advanced kernel.
- If custom kernel code is unstable, use a safe fallback that still returns correct MLX output.
- No placeholder code.
- After the validation command succeeds, immediately call the submit tool and stop.

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


def solution(a, b):
    # For matmul tasks, either implement a correct custom kernel path
    # or safely fallback to mx.matmul(a, b)
    return mx.matmul(a, b)
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


def get_metal_reasoning_system_prompt(gpu_name: str, vram_gb: int) -> str:
    """Generate reasoning-only system prompt for MLX Metal solutions."""

    return f"""You are an expert Apple GPU kernel engineer.

TARGET GPU: Apple {gpu_name} ({vram_gb}GB unified memory)

Write a complete `solution.py` using MLX for Metal execution.

Requirements:
- Use `import mlx.core as mx`
- Prefer custom Metal kernels via `mx.fast.metal_kernel`
- Do NOT use PyTorch, CUDA, or Triton
- Expose `def solution(a, b)` returning one `mx.array`
- Correctness is mandatory: match `mx.matmul(a, b)`

Return only complete Python code in a markdown code block.
If custom kernel code is unstable, provide a correct fallback with MLX ops."""
