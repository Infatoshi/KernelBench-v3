#!/usr/bin/env python3
"""
Modal-based KernelBench Evaluation

Run LLM agents on Modal GPUs to optimize kernel benchmarks.
Supports all providers: Anthropic, OpenAI, Gemini, xAI, OpenRouter.

Usage:
    # Single evaluation
    uv run python modal_eval.py --model claude-opus-4-5-20251101 --gpu H100 --problem level4/1_Qwen3-0p6B_bs32_seq256.py

    # Batch evaluation
    uv run python modal_eval.py --batch --models all --gpus H100,A100 --levels 1,2,3,4
"""

import argparse
import ast
import hashlib
import json
import os
import re
import signal
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Literal, Dict, Any, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.local_sandbox import LocalSandbox, LocalSandboxConfig  # noqa: E402
from src.agent.modal_sandbox import ModalSandbox, ModalSandboxConfig  # noqa: E402
from src.config.precision_matrix import (  # noqa: E402
    HARDWARE_PEAK_TFLOPS,
    HARDWARE_PRECISIONS,
    OP_PRECISION_VALIDITY,
)


# =============================================================================
# Dynamic Pricing (fetched from APIs, cached in memory)
# =============================================================================

# Cache for OpenRouter model pricing: {model_id: (input_per_million, output_per_million)}
_openrouter_pricing_cache: Dict[str, Tuple[float, float]] = {}
_openrouter_models_cache: Optional[Dict[str, Any]] = None

def _fetch_openrouter_models() -> Dict[str, Any]:
    """Fetch all OpenRouter models and cache them."""
    global _openrouter_models_cache
    if _openrouter_models_cache is not None:
        return _openrouter_models_cache

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return {}

    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            _openrouter_models_cache = {m["id"]: m for m in data.get("data", [])}
            return _openrouter_models_cache
    except Exception as e:
        print(f"Warning: Failed to fetch OpenRouter models: {e}")
        return {}

def get_openrouter_pricing(model_id: str) -> Optional[Tuple[float, float]]:
    """Get pricing for an OpenRouter model (input, output per million tokens)."""
    if model_id in _openrouter_pricing_cache:
        return _openrouter_pricing_cache[model_id]

    models = _fetch_openrouter_models()
    if model_id not in models:
        return None

    pricing = models[model_id].get("pricing", {})
    # OpenRouter returns per-token pricing, convert to per-million
    input_per_token = float(pricing.get("prompt", 0))
    output_per_token = float(pricing.get("completion", 0))

    result = (input_per_token * 1_000_000, output_per_token * 1_000_000)
    _openrouter_pricing_cache[model_id] = result
    return result

def is_valid_openrouter_model(model_id: str) -> bool:
    """Check if a model ID is valid on OpenRouter."""
    models = _fetch_openrouter_models()
    return model_id in models


# Mapping from our internal model IDs to OpenRouter model IDs for pricing lookup
# OpenRouter has pricing for ALL providers, so we use it as single source of truth
# Note: OpenRouter models (with /) are already in correct format - no mapping needed
MODEL_TO_OPENROUTER = {
    # Anthropic (we use dated IDs, OpenRouter uses canonical names)
    "claude-opus-4-5-20251101": "anthropic/claude-opus-4.5",
    "claude-sonnet-4-5-20250929": "anthropic/claude-sonnet-4.5",
    # OpenAI (direct API uses short names, OpenRouter uses openai/ prefix)
    "gpt-5.2": "openai/gpt-5.2",
    # Gemini (direct API uses short names, OpenRouter uses google/ prefix)
    "gemini-3-flash-preview": "google/gemini-3-flash-preview",
    "gemini-3-pro-preview": "google/gemini-3-pro-preview",
    # xAI (our internal name differs from OpenRouter)
    "grok-4-1-fast-reasoning": "x-ai/grok-4.1-fast",
}


# =============================================================================
# XML Tool Parsing (for models that use XML tool calling)
# =============================================================================

def unescape_html(text: str) -> str:
    """Unescape HTML entities that models sometimes output."""
    import html
    return html.unescape(text)


def parse_xml_tool_calls(content: str) -> List[Dict[str, Any]]:
    """Parse XML-formatted tool calls from model response."""
    tool_calls = []

    # Pattern 1: <tool_call><bash><command>...</command></bash></tool_call>
    tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(tool_call_pattern, content, re.DOTALL)

    for match in matches:
        # Try bash
        bash_match = re.search(r'<bash[^>]*>\s*<command>(.*?)</command>\s*</bash[^>]*>', match, re.DOTALL)
        if bash_match:
            tool_calls.append({
                "id": f"xml_bash_{len(tool_calls)}",
                "name": "bash",
                "input": {"command": unescape_html(bash_match.group(1).strip())}
            })
            continue

        # Try submit
        submit_match = re.search(r'<submit[^>]*>\s*<solution_path>(.*?)</solution_path>\s*</submit[^>]*>', match, re.DOTALL)
        if submit_match:
            tool_calls.append({
                "id": f"xml_submit_{len(tool_calls)}",
                "name": "submit",
                "input": {"solution_path": unescape_html(submit_match.group(1).strip())}
            })

    # Pattern 2: Direct tool calls without wrapper
    if not tool_calls:
        # Direct bash
        for bash_match in re.finditer(r'<bash[^>]*>\s*<command>(.*?)</command>\s*</bash[^>]*>', content, re.DOTALL):
            tool_calls.append({
                "id": f"xml_bash_{len(tool_calls)}",
                "name": "bash",
                "input": {"command": unescape_html(bash_match.group(1).strip())}
            })

        # Direct submit
        for submit_match in re.finditer(r'<submit[^>]*>\s*<solution_path>(.*?)</solution_path>\s*</submit[^>]*>', content, re.DOTALL):
            tool_calls.append({
                "id": f"xml_submit_{len(tool_calls)}",
                "name": "submit",
                "input": {"solution_path": unescape_html(submit_match.group(1).strip())}
            })

    return tool_calls


# =============================================================================
# Code Extraction (for reasoning models without tool use)
# =============================================================================

def extract_python_code(text: str) -> Optional[str]:
    """Extract Python code from model response.

    Tries multiple patterns:
    1. Markdown code blocks with python/py tag
    2. Generic markdown code blocks
    3. Code between specific markers

    Returns the last (most complete) code block found, or None.
    """
    # Pattern 1: ```python or ```py code blocks
    python_blocks = re.findall(r'```(?:python|py)\s*\n(.*?)```', text, re.DOTALL)
    if python_blocks:
        # Return the last one (usually the most complete/final version)
        return python_blocks[-1].strip()

    # Pattern 2: Generic ``` code blocks (if they look like Python)
    generic_blocks = re.findall(r'```\s*\n(.*?)```', text, re.DOTALL)
    for block in reversed(generic_blocks):
        # Check if it looks like Python (has imports, def, class, etc.)
        if any(marker in block for marker in ['import ', 'def ', 'class ', 'torch.', 'cuda_source']):
            return block.strip()

    # Pattern 3: Look for solution.py content markers
    solution_match = re.search(r'# solution\.py\s*\n(.*?)(?=\n#\s*\w+\.py|\Z)', text, re.DOTALL)
    if solution_match:
        return solution_match.group(1).strip()

    return None


def get_reasoning_system_prompt(gpu_name: str, vram_gb: int) -> str:
    """Generate system prompt for reasoning models (no tool use)."""
    return f"""You are a GPU kernel optimization expert running inside an isolated benchmark sandbox. Your task is to write optimized CUDA kernels.

**TARGET GPU**: NVIDIA {gpu_name} ({vram_gb}GB VRAM)

**YOUR TASK**: Write a custom CUDA kernel to optimize the PyTorch model shown in the reference code.

**CRITICAL REQUIREMENTS**:
1. You MUST write actual CUDA C++ code using `torch.utils.cpp_extension.load_inline`
2. Do NOT use torch.compile, Triton, or flash_attn
3. Your __global__ kernels MUST be called in the C++ wrapper functions
4. Do NOT fall back to PyTorch/cuBLAS in the wrapper (no torch::mm, torch::matmul, torch::conv2d, etc.)

**PERFORMANCE REQUIREMENTS BY OP TYPE**:
- For matrix operations (GEMM, matmul, linear layers, attention), for best performance consider Tensor Cores via WMMA (`<mma.h>`, `nvcuda::wmma::*`) or inline PTX (`mma.sync.aligned`).
- Tensor Core tile alignment is often helpful: 16x16x16 for FP16, 8x8x4 for TF32.
- For non-matrix ops (reductions, activations, norms), standard CUDA optimization is sufficient: memory coalescing, shared memory, warp-level primitives.
- Priority order: (1) correct and compilable kernel, (2) performance optimization. If a tensor-core path does not compile reliably, submit a correct standard CUDA kernel first.
- Default strategy: implement a standard tiled CUDA kernel first. Only use WMMA/PTX when you are fully confident it will compile and run correctly.

**OUTPUT FORMAT**: Provide your complete solution.py in a markdown code block:

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

// Your custom CUDA kernel
__global__ void my_kernel(float* out, const float* in, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx];
}}

// Wrapper that MUST call your kernel (not torch::mm or other PyTorch ops)
torch::Tensor my_op(torch::Tensor input) {{
    auto output = torch::empty_like(input);
    int n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    my_kernel<<<blocks, threads>>>(output.data_ptr<float>(), input.data_ptr<float>(), n);
    return output;
}}
\"\"\"

my_module = load_inline(
    name='my_module',
    cpp_sources=['torch::Tensor my_op(torch::Tensor);'],
    cuda_sources=[cuda_source],
    functions=['my_op'],
    verbose=False
)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return my_module.my_op(x)

def get_inputs():
    return [torch.randn(1024, 1024, device='cuda')]

def get_init_inputs():
    return []
```

**FORBIDDEN** (will result in failure):
- torch::mm, torch::matmul, torch::conv2d, torch::linear in C++ wrapper
- Defining a __global__ kernel but not calling it
- Using cuBLAS/cuDNN directly instead of your own kernel
- torch.compile or @torch.jit.script

**RULES**:
- Keep the same class name `Model` and same `get_inputs`/`get_init_inputs` as reference
- Write actual __global__ CUDA kernels and CALL them
- Optimize for {gpu_name}: use shared memory, tiling, warp-level primitives
- Provide COMPLETE, COMPILABLE code - no placeholders or TODOs

If your code has errors, I will show you the error message and you should provide a corrected version."""


# =============================================================================
# Model Configurations
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for an LLM model."""
    name: str
    model_id: str
    provider: Literal["anthropic", "openai", "gemini", "xai", "openrouter"]
    use_xml_tools: bool = False
    provider_order: Optional[List[str]] = None  # For OpenRouter
    reasoning_mode: bool = False  # For reasoning models without tool use (e.g., kimi-k2.5)


MODELS = {
    # Tier 1: Frontier models
    "claude-opus-4.5": ModelConfig(
        name="Claude Opus 4.5",
        model_id="anthropic/claude-opus-4",
        provider="openrouter"
    ),
    "claude-sonnet-4.5": ModelConfig(
        name="Claude Sonnet 4.5",
        model_id="anthropic/claude-sonnet-4",
        provider="openrouter"
    ),
    "gpt-5.2": ModelConfig(
        name="GPT-5.2",
        model_id="gpt-5.2",
        provider="openai"
    ),
    "gemini-3-flash": ModelConfig(
        name="Gemini 3 Flash",
        model_id="google/gemini-2.0-flash-exp",
        provider="openrouter",
        use_xml_tools=False
    ),
    "gemini-3-pro": ModelConfig(
        name="Gemini 3 Pro",
        model_id="gemini-3-pro-preview",
        provider="gemini",
        use_xml_tools=False  # Native function calling works
    ),
    "grok-4.1": ModelConfig(
        name="Grok 4.1 Fast Reasoning",
        model_id="grok-4-1-fast-reasoning",
        provider="xai"
    ),
    # Tier 2: Strong open/Chinese models via OpenRouter (native function calling works)
    "glm-4.7": ModelConfig(
        name="GLM-4.7",
        model_id="z-ai/glm-4.7",
        provider="openrouter"
    ),
    "deepseek-v3.2": ModelConfig(
        name="DeepSeek V3.2",
        model_id="deepseek/deepseek-chat",
        provider="openrouter"
    ),
    "kimi-k2-thinking": ModelConfig(
        name="Kimi K2 Thinking",
        model_id="moonshotai/kimi-k2-thinking",
        provider="openrouter"
    ),
    "minimax-m2.1": ModelConfig(
        name="MiniMax M2.1",
        model_id="minimax/minimax-m2.1",
        provider="openrouter"
    ),
    # OpenRouter frontier models
    "z-ai/glm-5": ModelConfig(
        name="GLM-5",
        model_id="z-ai/glm-5",
        provider="openrouter"
    ),
    "openrouter/aurora-alpha": ModelConfig(
        name="OpenRouter Aurora Alpha",
        model_id="openrouter/aurora-alpha",
        provider="openrouter"
    ),
    # Phase 1 compatibility matrix model IDs (exact keys for harness dry-run checks)
    "anthropic/claude-opus-4.6": ModelConfig(
        name="Claude Opus 4.6",
        model_id="anthropic/claude-opus-4.6",
        provider="openrouter"
    ),
    "openai/gpt-5.2-codex": ModelConfig(
        name="GPT-5.2 Codex",
        model_id="openai/gpt-5.2-codex",
        provider="openrouter"
    ),
    "google/gemini-3-flash-preview": ModelConfig(
        name="Gemini 3 Flash Preview",
        model_id="google/gemini-3-flash-preview",
        provider="openrouter"
    ),
    "google/gemini-3-pro-preview": ModelConfig(
        name="Gemini 3 Pro Preview",
        model_id="google/gemini-3-pro-preview",
        provider="openrouter"
    ),
    "minimax/minimax-m2.5": ModelConfig(
        name="MiniMax M2.5",
        model_id="minimax/minimax-m2.5",
        provider="openrouter"
    ),
    "deepseek/deepseek-v3.2": ModelConfig(
        name="DeepSeek V3.2",
        model_id="deepseek/deepseek-v3.2",
        provider="openrouter"
    ),
    "x-ai/grok-4.1-fast": ModelConfig(
        name="Grok 4.1 Fast",
        model_id="x-ai/grok-4.1-fast",
        provider="openrouter"
    ),
    "moonshotai/kimi-k2.5": ModelConfig(
        name="Kimi K2.5",
        model_id="moonshotai/kimi-k2.5",
        provider="openrouter",
        reasoning_mode=True
    ),
    # Reasoning models (no tool use, code extracted from text output)
    "kimi-k2.5": ModelConfig(
        name="Kimi K2.5",
        model_id="moonshotai/kimi-k2.5",
        provider="openrouter",
        reasoning_mode=True
    ),
}


def get_model_config(model_key: str) -> Optional[ModelConfig]:
    """Get model config by key, supporting both predefined and dynamic OpenRouter models.

    For predefined models, returns the config from MODELS dict.
    For OpenRouter models not in MODELS, validates against OpenRouter API
    and creates a dynamic config.

    Args:
        model_key: Either a predefined key (e.g., "claude-opus-4.5") or
                   an OpenRouter model ID (e.g., "anthropic/claude-3-opus")

    Returns:
        ModelConfig if valid, None otherwise
    """
    # Check predefined models first
    if model_key in MODELS:
        return MODELS[model_key]

    # Check if it looks like an OpenRouter model ID (contains /)
    if "/" in model_key:
        if is_valid_openrouter_model(model_key):
            # Get model info from OpenRouter
            models = _fetch_openrouter_models()
            model_info = models.get(model_key, {})
            name = model_info.get("name", model_key)

            # Check if model supports tool use
            supported_params = model_info.get("supported_parameters", [])
            has_tools = "tools" in supported_params

            return ModelConfig(
                name=name,
                model_id=model_key,
                provider="openrouter",
                reasoning_mode=not has_tools  # Use reasoning mode if no tool support
            )
        else:
            return None

    return None


# =============================================================================
# Provider Clients
# =============================================================================

def get_provider_client(provider: str):
    """Get API client for the specified provider."""
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic()

    elif provider == "openai":
        from openai import OpenAI
        return OpenAI()

    elif provider == "gemini":
        # Use native Google AI SDK for Gemini 3 (handles thought signatures)
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        return genai  # Return the module itself

    elif provider == "xai":
        from openai import OpenAI
        return OpenAI(
            api_key=os.environ.get("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )

    elif provider == "openrouter":
        from openai import OpenAI
        return OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


# =============================================================================
# Agent Execution
# =============================================================================

def get_system_prompt(gpu_name: str, vram_gb: int, use_xml_tools: bool = False) -> str:
    """Generate system prompt with GPU info."""
    base_prompt = f"""You are a GPU kernel optimization expert running in an isolated sandbox on an NVIDIA {gpu_name} GPU ({vram_gb}GB VRAM).

**YOUR TASK**: Write a custom CUDA kernel to optimize the PyTorch model in reference.py.

**CRITICAL REQUIREMENTS**:
1. You MUST write actual CUDA C++ code using `torch.utils.cpp_extension.load_inline`
2. Do NOT use torch.compile, Triton, or flash_attn
3. Your __global__ kernels MUST be called in the C++ wrapper functions
4. Do NOT fall back to PyTorch/cuBLAS in the wrapper (no torch::mm, torch::matmul, torch::conv2d, etc.)

**PERFORMANCE REQUIREMENTS BY OP TYPE**:
- For matrix operations (GEMM, matmul, linear layers, attention), for best performance consider Tensor Cores via WMMA (`<mma.h>`, `nvcuda::wmma::*`) or inline PTX (`mma.sync.aligned`).
- Tensor Core tile alignment is often helpful: 16x16x16 for FP16, 8x8x4 for TF32.
- For non-matrix ops (reductions, activations, norms), standard CUDA optimization is sufficient: memory coalescing, shared memory, warp-level primitives.
- Priority order: (1) correct and compilable kernel, (2) performance optimization. If a tensor-core path does not compile reliably, submit a correct standard CUDA kernel first.
- Default strategy: implement a standard tiled CUDA kernel first. Only use WMMA/PTX when you are fully confident it will compile and run correctly.

**REQUIRED WORKFLOW** (follow exactly - USE THE TOOLS):
1. Use bash tool: `cat /workspace/reference.py` - read the reference model
2. Use bash tool: `cat > /workspace/solution.py << 'EOF'\n<your code>\nEOF` - write your solution
3. Use bash tool: `python -c "import reference, solution, torch; m=solution.Model(*reference.get_init_inputs()).cuda().eval(); inputs=[x.cuda() if isinstance(x, torch.Tensor) else x for x in reference.get_inputs()]; _=m(*inputs); print('OK')"` - test compile + forward pass
4. Use submit tool with path "solution.py" - submit for benchmarking

IMPORTANT: You MUST use the bash tool to write files. Do NOT just describe code - actually write it using bash.

**SOLUTION FORMAT** - Your solution.py MUST have this structure:
```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = \"\"\"
#include <torch/extension.h>
#include <cuda_runtime.h>

// Your custom CUDA kernel
__global__ void my_kernel(float* out, const float* in, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx];
}}

// Wrapper that MUST call your kernel (not torch::mm or other PyTorch ops)
torch::Tensor my_op(torch::Tensor input) {{
    auto output = torch::empty_like(input);
    int n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    my_kernel<<<blocks, threads>>>(output.data_ptr<float>(), input.data_ptr<float>(), n);
    return output;
}}
\"\"\"

my_module = load_inline(
    name='my_module',
    cpp_sources=['torch::Tensor my_op(torch::Tensor);'],
    cuda_sources=[cuda_source],
    functions=['my_op'],
    verbose=False
)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return my_module.my_op(x)

def get_inputs():
    return [torch.randn(1024, 1024, device='cuda')]

def get_init_inputs():
    return []
```

**FORBIDDEN** (will result in 1.0x speedup = failure):
- torch::mm, torch::matmul, torch::conv2d, torch::linear in C++ wrapper
- Defining a __global__ kernel but not calling it
- Using cuBLAS/cuDNN directly instead of your own kernel
- torch.compile or @torch.jit.script

**RULES**:
- Keep the same class name `Model` and same `get_inputs`/`get_init_inputs` as reference
- Write actual __global__ CUDA kernels and CALL them
- Optimize for {gpu_name}: use shared memory, tiling, warp-level primitives
- Submit quickly - you have limited turns

The reference code is in /workspace/reference.py."""

    if use_xml_tools:
        xml_tools = """

TOOLS - Use XML format to call tools:

1. bash - Execute shell commands:
<tool_call><bash><command>YOUR_COMMAND_HERE</command></bash></tool_call>

2. submit - Submit your solution:
<tool_call><submit><solution_path>solution.py</solution_path></submit></tool_call>

IMPORTANT: Always wrap tool calls in <tool_call> tags. You can use multiple tool calls in one response."""
        return base_prompt + xml_tools
    else:
        native_tools = """

TOOLS:
- bash: Execute any shell command
- submit: Call when done with the path to your optimized solution.py"""
        return base_prompt + native_tools


def _safe_literal_eval(node: ast.AST) -> Any:
    """Best-effort literal evaluation for static metadata extraction."""
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _extract_reference_metadata(reference_code: str) -> Dict[str, Any]:
    """Extract static metadata from reference.py without executing it."""
    metadata: Dict[str, Any] = {
        "op_type": "unknown",
        "supported_precisions": [],
        "hardware_required": [],
        "has_model_class": False,
        "has_get_inputs": False,
        "has_get_init_inputs": False,
    }
    try:
        tree = ast.parse(reference_code)
    except SyntaxError:
        return metadata

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            metadata["has_model_class"] = True
        if isinstance(node, ast.FunctionDef) and node.name == "get_inputs":
            metadata["has_get_inputs"] = True
        if isinstance(node, ast.FunctionDef) and node.name == "get_init_inputs":
            metadata["has_get_init_inputs"] = True
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id == "OP_TYPE":
                value = _safe_literal_eval(node.value)
                if isinstance(value, str):
                    metadata["op_type"] = value
            elif target.id == "SUPPORTED_PRECISIONS":
                value = _safe_literal_eval(node.value)
                if isinstance(value, (list, tuple)):
                    metadata["supported_precisions"] = [str(x) for x in value]
            elif target.id == "HARDWARE_REQUIRED":
                value = _safe_literal_eval(node.value)
                if isinstance(value, (list, tuple)):
                    metadata["hardware_required"] = [str(x) for x in value]
    return metadata


def _backend_self_check_command(backend: str) -> str:
    """Command models should run before submit to verify solution behavior."""
    if backend == "metal":
        return (
            "python -c \"import mlx.core as mx, solution; "
            "a=mx.random.normal((128,128), dtype=mx.float32); "
            "b=mx.random.normal((128,128), dtype=mx.float32); "
            "y=solution.solution(a,b); mx.eval(y); print('OK')\""
        )
    if backend == "cutile":
        return (
            "python -c \"import reference, solution, torch; "
            "m=solution.Model(*reference.get_init_inputs()).cuda().eval(); "
            "inputs=[x.cuda() if isinstance(x, torch.Tensor) else x for x in reference.get_inputs()]; "
            "_=m(*inputs); print('OK')\""
        )
    return (
        "python -c \"import reference, solution, torch; "
        "m=solution.Model(*reference.get_init_inputs()).cuda().eval(); "
        "inputs=[x.cuda() if isinstance(x, torch.Tensor) else x for x in reference.get_inputs()]; "
        "_=m(*inputs); print('OK')\""
    )


def _augment_system_prompt(system_prompt: str, backend: str) -> str:
    """Append environment contract and anti-probing policy to any prompt."""
    self_check = _backend_self_check_command(backend)
    return (
        system_prompt
        + f"""

EXECUTION ENVIRONMENT CONTRACT:
- You are in an isolated benchmark sandbox shell, not a general SSH host session.
- Working directory is `/workspace`.
- Preloaded files: `/workspace/reference.py`, `/workspace/ENVIRONMENT.md`, `/workspace/BACKEND_API.md`, `/workspace/TEMPLATE_solution.py`, `/workspace/TASK_CONTEXT.md`.
- Internet access and package installation are not part of this task. Do not run package managers.

ANTI-PROBING POLICY:
- At most one lightweight environment probe command in turn 1.
- Do not spend turns on `pip list`, `which python`, `sys.path`, or filesystem crawling unless a compile/runtime error explicitly requires it.
- Move to implementing `solution.py` immediately after reading context files.

REQUIRED PRE-SUBMIT SELF-CHECK:
- Run exactly:
`{self_check}`
- If this prints `OK`, call submit immediately and stop editing.
"""
    )


def _inject_workspace_context(system_prompt: str, context_bundle: Dict[str, str]) -> str:
    """Inline workspace context content directly into the prompt."""
    environment_md = context_bundle.get("environment_md", "").strip()
    backend_api_md = context_bundle.get("backend_api_md", "").strip()
    task_context_md = context_bundle.get("task_context_md", "").strip()
    template_solution_py = context_bundle.get("template_solution_py", "").strip()

    return (
        system_prompt
        + "\n\nINLINE WORKSPACE CONTEXT (authoritative; do not re-discover):\n"
        + "\n\n[ENVIRONMENT.md]\n"
        + environment_md
        + "\n\n[BACKEND_API.md]\n"
        + backend_api_md
        + "\n\n[TASK_CONTEXT.md]\n"
        + task_context_md
        + "\n\n[TEMPLATE_solution.py]\n```python\n"
        + template_solution_py
        + "\n```\n"
    )


def _build_backend_api_reference(backend: str) -> str:
    """Generate concise backend-specific API notes for models."""
    backend_key = backend.lower()
    if backend_key == "triton":
        return """# Backend API Quick Reference: Triton

- Required imports:
  - `import triton`
  - `import triton.language as tl`
- Required kernel pattern:
  - `@triton.jit`
  - `kernel_name[grid](...)`
- Keep `Model`, `get_inputs`, `get_init_inputs` compatible with `reference.py`.
- Do NOT use `torch.utils.cpp_extension.load_inline` in Triton backend.
"""
    if backend_key == "cutlass":
        return """# Backend API Quick Reference: CUTLASS

- Required includes from `/opt/cutlass/include`:
  - `cutlass/gemm/device/gemm_universal.h`
- Use `from torch.utils.cpp_extension import load_inline`.
- Keep `Model`, `get_inputs`, `get_init_inputs` compatible with `reference.py`.
- Do NOT fallback to raw PyTorch ops in the wrapper path.
"""
    if backend_key == "cute":
        return """# Backend API Quick Reference: CuTe

- CuTe headers path: `/opt/cutlass/include`.
- Required includes:
  - `cute/tensor.hpp`
  - `cute/layout.hpp`
- Use `from torch.utils.cpp_extension import load_inline`.
- Keep `Model`, `get_inputs`, `get_init_inputs` compatible with `reference.py`.
- Use exact include path `/opt/cutlass/include` (not `/opt/cutlass/include/include`).
"""
    if backend_key == "cutile":
        return """# Backend API Quick Reference: CuTile Python

- Required import:
  - `import cuda.tile as ct`
- Use CuTile Python kernels with `@ct.kernel`.
- Launch signature:
  - `ct.launch(stream, grid, kernel, kernel_args_tuple)`
- Use compile-time constants for tile shapes.
- Do NOT use `load_inline` or C++ extension code in CuTile backend.
"""
    if backend_key == "metal":
        return """# Backend API Quick Reference: Metal (MLX)

- Required import:
  - `import mlx.core as mx`
- Implement `solution(a, b)` for MLX arrays.
- Use MLX operations / kernel APIs, not PyTorch CUDA extensions.
- Do NOT use `torch.utils.cpp_extension.load_inline`.
"""
    if backend_key == "graphics":
        return """# Backend API Quick Reference: Graphics (CUDA Triton)

- Required imports:
  - `import triton`
  - `import triton.language as tl`
- Required kernel pattern:
  - `@triton.jit`
  - `kernel_name[grid](...)`
- Keep `Model`, `get_inputs`, `get_init_inputs` compatible with `reference.py`.
- Do NOT use OpenGL/Vulkan/Metal runtime code paths.
"""
    return """# Backend API Quick Reference: CUDA

- Use `from torch.utils.cpp_extension import load_inline`.
- Write actual `__global__` CUDA kernels and call them from wrappers.
- Keep `Model`, `get_inputs`, `get_init_inputs` compatible with `reference.py`.
- Do NOT fallback to raw PyTorch/cuBLAS calls in wrapper path.
"""


def _build_template_solution(backend: str) -> str:
    """Generate a minimal backend-specific template solution file."""
    backend_key = backend.lower()
    if backend_key == "triton":
        return """import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, x, mask=mask)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.empty_like(x)
        n = x.numel()
        grid = (triton.cdiv(n, 256),)
        _kernel[grid](x, y, n, BLOCK=256)
        return y
"""
    if backend_key == "graphics":
        return """import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _graphics_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, x, mask=mask)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.empty_like(x)
        n = x.numel()
        grid = (triton.cdiv(n, 256),)
        _graphics_kernel[grid](x, y, n, BLOCK=256)
        return y
"""
    if backend_key in {"cutlass", "cute", "cuda"}:
        return """import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void identity_kernel(const float* x, float* y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) y[idx] = x[idx];
}

torch::Tensor launch_identity(torch::Tensor x) {
  auto y = torch::empty_like(x);
  int n = x.numel();
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  identity_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
  return y;
}
'''

ext = load_inline(
    name='kb_template_ext',
    cpp_sources='torch::Tensor launch_identity(torch::Tensor x);',
    cuda_sources=cuda_source,
    functions=['launch_identity'],
    verbose=False,
)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ext.launch_identity(x)
"""
    if backend_key == "cutile":
        return """import torch
import torch.nn as nn
import cuda.tile as ct

@ct.kernel
def _kernel(x, y, n):
    # Fill in CuTile tile-level ops here using compile-time constants.
    # Keep launch signature: ct.launch(stream, grid, kernel, kernel_args_tuple)
    pass

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = torch.empty_like(x)
        n = x.numel()
        grid = (ct.cdiv(n, 256),)
        ct.launch(torch.cuda.current_stream(), grid, _kernel, (x, y, n))
        return y
"""
    if backend_key == "metal":
        return """import mlx.core as mx

def solution(a, b):
    return mx.matmul(a, b)
"""
    return """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
"""


def _collect_runtime_environment(sandbox, backend: str, gpu_name: str, vram_gb: int, level: int) -> str:
    """Collect runtime environment facts and return markdown."""
    probe_cmd = """python - <<'PY'
import importlib
import json
import platform
import sys

modules = {}
for name in ("torch", "triton", "mlx.core", "cuda.tile", "flash_linear_attention"):
    try:
        mod = importlib.import_module(name)
        modules[name] = getattr(mod, "__version__", "unknown")
    except Exception:
        modules[name] = None

payload = {
    "python_version": sys.version.split()[0],
    "platform": platform.platform(),
    "modules": modules,
}
print(json.dumps(payload))
PY"""
    probe_result = sandbox.run_command(probe_cmd, timeout=45)
    payload: Dict[str, Any] = {}
    if probe_result.get("returncode") == 0:
        for line in reversed((probe_result.get("stdout") or "").splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue

    modules = payload.get("modules", {})
    lines = [
        "# Environment Snapshot",
        "",
        f"- backend: `{backend}`",
        f"- target_gpu: `{gpu_name}`",
        f"- target_vram_gb: `{vram_gb}`",
        f"- level: `{level}`",
        "- working_directory: `/workspace`",
        f"- python_version: `{payload.get('python_version', 'unknown')}`",
        f"- platform: `{payload.get('platform', 'unknown')}`",
        "",
        "## Module Availability",
    ]
    for name in ("torch", "triton", "mlx.core", "cuda.tile", "flash_linear_attention"):
        version = modules.get(name)
        lines.append(f"- {name}: `{version if version else 'unavailable'}`")
    lines.extend([
        "",
        "## Constraints",
        "- Do not run package installers during solving.",
        "- Use preinstalled toolchain and files in `/workspace`.",
    ])
    return "\n".join(lines)


def _build_task_context(
    problem_name: str,
    level: int,
    gpu_name: str,
    backend: str,
    metadata: Dict[str, Any],
) -> str:
    """Generate per-problem task context file content."""
    precisions = ", ".join(metadata.get("supported_precisions", [])) or "unknown"
    hardware_required = ", ".join(metadata.get("hardware_required", [])) or "unknown"
    return f"""# Task Context

- problem: `{problem_name}`
- level: `{level}`
- backend: `{backend}`
- gpu: `{gpu_name}`
- op_type: `{metadata.get("op_type", "unknown")}`
- supported_precisions: `{precisions}`
- hardware_required: `{hardware_required}`
- has_model_class: `{metadata.get("has_model_class", False)}`
- has_get_inputs: `{metadata.get("has_get_inputs", False)}`
- has_get_init_inputs: `{metadata.get("has_get_init_inputs", False)}`

Read `/workspace/reference.py` and preserve API compatibility.
"""


def _prepare_workspace_context(
    backend: str,
    gpu_name: str,
    vram_gb: int,
    level: int,
    problem_name: str,
    metadata: Dict[str, Any],
    sandbox,
) -> Dict[str, str]:
    """Build helper context bundle used for files and inline prompt injection."""
    return {
        "environment_md": _collect_runtime_environment(
            sandbox=sandbox,
            backend=backend,
            gpu_name=gpu_name,
            vram_gb=vram_gb,
            level=level,
        ),
        "backend_api_md": _build_backend_api_reference(backend),
        "template_solution_py": _build_template_solution(backend),
        "task_context_md": _build_task_context(problem_name, level, gpu_name, backend, metadata),
    }


def _seed_workspace_context(sandbox, context_bundle: Dict[str, str]) -> None:
    """Write helper context files into /workspace for reproducibility/debugging."""
    try:
        sandbox.write_file("ENVIRONMENT.md", context_bundle.get("environment_md", ""))
        sandbox.write_file("BACKEND_API.md", context_bundle.get("backend_api_md", ""))
        sandbox.write_file("TEMPLATE_solution.py", context_bundle.get("template_solution_py", ""))
        sandbox.write_file("TASK_CONTEXT.md", context_bundle.get("task_context_md", ""))
    except Exception as exc:
        # Context files are helpful guidance, not a hard requirement.
        print(f"Warning: failed to seed workspace context files: {exc}", flush=True)


def _build_initial_user_message(
    backend: str,
    problem_name: str,
    level: int,
    gpu_name: str,
    max_turns: int,
    reference_code: str,
    metadata: Dict[str, Any],
) -> str:
    """Build rich initial task message with metadata and full reference code."""
    precisions = ", ".join(metadata.get("supported_precisions", [])) or "unknown"
    hardware_required = ", ".join(metadata.get("hardware_required", [])) or "unknown"
    return f"""Optimize the benchmark task and produce `/workspace/solution.py`.

Task summary:
- benchmark backend: `{backend}`
- problem: `{problem_name}`
- level: `{level}`
- target GPU: `{gpu_name}`
- OP_TYPE: `{metadata.get("op_type", "unknown")}`
- SUPPORTED_PRECISIONS: `{precisions}`
- HARDWARE_REQUIRED: `{hardware_required}`

Mirrored workspace files (same content already provided inline in system context):
- `/workspace/reference.py`
- `/workspace/ENVIRONMENT.md`
- `/workspace/BACKEND_API.md`
- `/workspace/TEMPLATE_solution.py`
- `/workspace/TASK_CONTEXT.md`

Rules:
- Preserve `Model`, `get_inputs`, and `get_init_inputs` compatibility with `reference.py`.
- Run one compile/import/forward self-check, then submit immediately.
- Avoid environment probing beyond one lightweight check.

Turn budget (hard cap):
- You have exactly `{max_turns}` turns.
- Use turn 1 to inspect/reference and write a complete draft.
- Use intermediate turns only to fix concrete compile/runtime errors.
- On the final turn, if a `submit` tool is available, submit your best current `solution.py` even if still imperfect.
- Do not finish without either submitting or clearly reporting a blocking compile error.

Reference code:
```python
{reference_code}
```
"""


GPU_SPECS = {
    "L40S": ("L40S", 48),
    "A100": ("A100", 40),
    "H100": ("H100", 80),
    "B200": ("B200", 192),
    "RTX3090": ("RTX 3090", 24),
    "LOCAL": ("Local CUDA GPU", 24),
}

LOCAL_GPUS = {"RTX3090", "LOCAL"}

# Wall-clock timeout fallback per problem (seconds).
MAX_PROBLEM_TIME_SECONDS = {
    1: 300,   # L1: 5 minutes
    2: 600,   # L2: 10 minutes
    3: 900,   # L3: 15 minutes
    4: 1200,  # L4: 20 minutes
}


TOOLS_ANTHROPIC = [
    {
        "name": "bash",
        "description": "Execute a shell command",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "Shell command"}},
            "required": ["command"]
        }
    },
    {
        "name": "submit",
        "description": "Submit solution for benchmarking",
        "input_schema": {
            "type": "object",
            "properties": {"solution_path": {"type": "string", "description": "Path to solution"}},
            "required": ["solution_path"]
        }
    }
]

TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "Shell command"}},
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit solution for benchmarking",
            "parameters": {
                "type": "object",
                "properties": {"solution_path": {"type": "string", "description": "Path to solution"}},
                "required": ["solution_path"]
            }
        }
    }
]


@dataclass
class EvalResult:
    """Result of a single evaluation."""
    model: str
    gpu: str
    problem: str
    level: int
    compiled: bool = False
    correct: bool = False
    speedup: Optional[float] = None
    ref_ms: Optional[float] = None
    sol_ms: Optional[float] = None
    ref_mean_ms: Optional[float] = None
    sol_mean_ms: Optional[float] = None
    ref_std_ms: Optional[float] = None
    sol_std_ms: Optional[float] = None
    ref_p10_ms: Optional[float] = None
    ref_p90_ms: Optional[float] = None
    sol_p10_ms: Optional[float] = None
    sol_p90_ms: Optional[float] = None
    turns: int = 0
    submitted: bool = False
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    # Token usage tracking
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # Cache token tracking (for cost estimation)
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    # Cost tracking (in USD, estimated)
    estimated_cost_usd: Optional[float] = None
    # Solution code (the submitted kernel)
    solution_code: Optional[str] = None
    solution_path: Optional[str] = None
    solution_hash: Optional[str] = None
    # Kernel count tracking (for megakernel verification)
    ref_kernels: Optional[int] = None
    sol_kernels: Optional[int] = None
    # Benchmark metadata
    correctness_seeds: Optional[List[int]] = None
    benchmark_seed: Optional[int] = None
    baseline_type: Optional[str] = None
    precision: Optional[str] = None
    precision_used: Optional[str] = None
    valid_precisions: Optional[List[str]] = None
    precision_supported: Optional[bool] = None
    tolerance_atol: float = 0.05
    tolerance_rtol: float = 0.02
    has_nan: bool = False
    has_inf: bool = False
    is_deterministic: bool = True
    # Performance metadata
    achieved_tflops: Optional[float] = None
    ref_tflops: Optional[float] = None
    pct_of_peak: Optional[float] = None
    ref_pct_of_peak: Optional[float] = None


def _get_turn_artifact_dir() -> Optional[Path]:
    """Get per-turn artifact directory from environment, if configured."""
    raw = os.environ.get("KB_TURN_ARTIFACT_DIR")
    if not raw:
        return None
    artifact_dir = Path(raw)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _write_turn_artifact(turn: int, suffix: str, content: str) -> None:
    """Persist per-turn artifacts for debugging (best-effort)."""
    artifact_dir = _get_turn_artifact_dir()
    if artifact_dir is None:
        return

    try:
        path = artifact_dir / f"turn_{turn}_{suffix}"
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception:
        # Artifact persistence is diagnostics-only and must never break eval.
        pass


def _format_command_result(cmd: str, cmd_result: dict) -> str:
    """Format command execution result for logs/artifacts."""
    return (
        f"command: {cmd}\n"
        f"return_code: {cmd_result.get('returncode')}\n"
        f"stdout:\n{cmd_result.get('stdout', '')}\n"
        f"stderr:\n{cmd_result.get('stderr', '')}\n"
    )


def _auto_submit_if_compilable(sandbox, submitted: bool, solution_path: Optional[str]) -> tuple[bool, Optional[str]]:
    """Auto-submit solution.py when model forgot submit tool but code imports."""
    if submitted:
        return submitted, solution_path

    if not sandbox.file_exists("solution.py"):
        return submitted, solution_path

    compile_checks = [
        ('python -c "from solution import Model; m = Model(); print(\'OK\')"', "Model import check OK"),
        ('python -c "import solution; print(\'OK\')"', "module import check OK"),
    ]
    compile_logs: List[str] = []

    for compile_cmd, success_label in compile_checks:
        compile_result = sandbox.run_command(compile_cmd, timeout=120)
        compile_logs.append(_format_command_result(compile_cmd, compile_result))
        if compile_result["returncode"] == 0 and "OK" in compile_result["stdout"]:
            _write_turn_artifact(999, "compile.log", "\n\n".join(compile_logs))
            print(f"  AUTO-SUBMITTED: solution.py ({success_label})", flush=True)
            return True, "solution.py"

    _write_turn_artifact(999, "compile.log", "\n\n".join(compile_logs))

    return submitted, solution_path


def _begin_problem_alarm(level: int) -> Optional[Any]:
    """Enable SIGALRM timeout for a problem, returning previous handler."""
    timeout_seconds = MAX_PROBLEM_TIME_SECONDS.get(level)
    if timeout_seconds is None or timeout_seconds <= 0:
        return None
    if not hasattr(signal, "SIGALRM"):
        return None

    def _timeout_handler(_signum, _frame):
        raise TimeoutError("Problem time limit exceeded")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    return previous_handler


def _clear_problem_alarm(previous_handler: Optional[Any]) -> None:
    """Disable SIGALRM timeout and restore previous handler."""
    if previous_handler is None:
        return
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous_handler)


def _attach_solution_metadata(result: EvalResult, solution_path: Optional[str], sandbox) -> None:
    """Load solution code and attach hash/path metadata to result."""
    if not solution_path:
        return
    sol_path = solution_path if solution_path.startswith("/") else f"/workspace/{solution_path}"
    solution_code = sandbox.read_file(sol_path.replace("/workspace/", ""))
    result.solution_code = solution_code
    result.solution_path = solution_path
    if solution_code is not None:
        result.solution_hash = hashlib.sha256(solution_code.encode("utf-8")).hexdigest()[:16]


def _apply_benchmark_metrics(result: EvalResult, benchmark_result: Dict[str, Any]) -> None:
    """Populate EvalResult fields from benchmark output."""
    result.compiled = benchmark_result.get("compiled", False)
    result.correct = benchmark_result.get("correct", False)
    result.speedup = benchmark_result.get("speedup")
    result.ref_ms = benchmark_result.get("ref_ms")
    result.sol_ms = benchmark_result.get("sol_ms")
    result.ref_mean_ms = benchmark_result.get("ref_mean_ms")
    result.sol_mean_ms = benchmark_result.get("sol_mean_ms")
    result.ref_std_ms = benchmark_result.get("ref_std_ms")
    result.sol_std_ms = benchmark_result.get("sol_std_ms")
    result.ref_p10_ms = benchmark_result.get("ref_p10_ms")
    result.ref_p90_ms = benchmark_result.get("ref_p90_ms")
    result.sol_p10_ms = benchmark_result.get("sol_p10_ms")
    result.sol_p90_ms = benchmark_result.get("sol_p90_ms")
    result.ref_kernels = benchmark_result.get("ref_kernels")
    result.sol_kernels = benchmark_result.get("sol_kernels")

    result.correctness_seeds = benchmark_result.get("correctness_seeds")
    result.benchmark_seed = benchmark_result.get("benchmark_seed")
    result.baseline_type = benchmark_result.get("baseline_type")
    result.precision = benchmark_result.get("precision") or benchmark_result.get("precision_used")
    result.precision_used = benchmark_result.get("precision_used") or benchmark_result.get("precision")
    result.valid_precisions = benchmark_result.get("valid_precisions")
    result.precision_supported = benchmark_result.get("precision_supported")
    result.tolerance_atol = benchmark_result.get("tolerance_atol", result.tolerance_atol)
    result.tolerance_rtol = benchmark_result.get("tolerance_rtol", result.tolerance_rtol)
    result.has_nan = benchmark_result.get("has_nan", False)
    result.has_inf = benchmark_result.get("has_inf", False)
    result.is_deterministic = benchmark_result.get("is_deterministic", True)
    result.achieved_tflops = benchmark_result.get("achieved_tflops")
    result.ref_tflops = benchmark_result.get("ref_tflops")
    result.pct_of_peak = benchmark_result.get("pct_of_peak")
    result.ref_pct_of_peak = benchmark_result.get("ref_pct_of_peak")

    if benchmark_result.get("error"):
        result.error = benchmark_result["error"]


def _run_gemini_agent(
    model_config: ModelConfig,
    sandbox,
    system_prompt: str,
    initial_user_message: str,
    max_turns: int
) -> tuple:
    """Run Gemini 3 agent using native SDK with automatic function calling.

    Returns: (submitted, solution_path, turns_used, input_tokens, output_tokens)
    """
    import google.generativeai as genai
    from google.generativeai.types import FunctionDeclaration, Tool

    # Define tools for Gemini
    bash_func = FunctionDeclaration(
        name="bash",
        description="Execute a shell command in the sandbox",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute"}
            },
            "required": ["command"]
        }
    )

    submit_func = FunctionDeclaration(
        name="submit",
        description="Submit your optimized solution for benchmarking",
        parameters={
            "type": "object",
            "properties": {
                "solution_path": {"type": "string", "description": "Path to the solution file"}
            },
            "required": ["solution_path"]
        }
    )

    tools = Tool(function_declarations=[bash_func, submit_func])

    # Create model and chat
    model = genai.GenerativeModel(
        model_config.model_id,
        system_instruction=system_prompt,
        tools=[tools]
    )
    chat = model.start_chat()

    submitted = False
    solution_path = None
    turns_used = 0
    total_input_tokens = 0
    total_output_tokens = 0

    # Initial message
    response = chat.send_message(initial_user_message)

    # Track tokens from initial response
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        total_input_tokens += getattr(response.usage_metadata, 'prompt_token_count', 0)
        total_output_tokens += getattr(response.usage_metadata, 'candidates_token_count', 0)

    for turn in range(max_turns):
        turns_used = turn + 1
        print(f"\n[Turn {turn + 1}/{max_turns}]", flush=True)

        # Check for function calls
        function_calls = []
        text_parts = []

        for part in response.candidates[0].content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                function_calls.append(part.function_call)
            elif hasattr(part, 'text') and part.text:
                text_parts.append(part.text)

        # Print any text
        if text_parts:
            text = " ".join(text_parts)
            text = text[:200] + "..." if len(text) > 200 else text
            print(f"Assistant: {text}", flush=True)
        _write_turn_artifact(turn + 1, "response.txt", "\n".join(text_parts))

        if not function_calls:
            print("No tool calls - agent finished", flush=True)
            break

        # Execute function calls and collect results
        function_responses = []
        for fc in function_calls:
            tool_name = fc.name
            tool_args = dict(fc.args)
            print(f"  Tool: {tool_name}", flush=True)

            if tool_name == "bash":
                cmd = tool_args.get("command", "")
                print(f"    $ {cmd[:80]}..." if len(cmd) > 80 else f"    $ {cmd}", flush=True)
                cmd_result = sandbox.run_command(cmd)
                output = f"stdout:\n{cmd_result['stdout']}\nstderr:\n{cmd_result['stderr']}\nreturn_code: {cmd_result['returncode']}"
                _write_turn_artifact(turn + 1, "compile.log", _format_command_result(cmd, cmd_result))
                if cmd_result["stdout"]:
                    out = cmd_result["stdout"][:150] + "..." if len(cmd_result["stdout"]) > 150 else cmd_result["stdout"]
                    print(f"    -> {out}", flush=True)
                function_responses.append(
                    genai.protos.Part(function_response=genai.protos.FunctionResponse(
                        name=tool_name,
                        response={"result": output}
                    ))
                )

            elif tool_name == "submit":
                solution_path = tool_args.get("solution_path", "solution.py")
                submitted = True
                print(f"  SUBMITTED: {solution_path}", flush=True)
                function_responses.append(
                    genai.protos.Part(function_response=genai.protos.FunctionResponse(
                        name=tool_name,
                        response={"result": f"Submitted: {solution_path}"}
                    ))
                )

        if submitted:
            break

        # Send function responses back
        response = chat.send_message(function_responses)

        # Track tokens from this response
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            total_input_tokens += getattr(response.usage_metadata, 'prompt_token_count', 0)
            total_output_tokens += getattr(response.usage_metadata, 'candidates_token_count', 0)

    submitted, solution_path = _auto_submit_if_compilable(sandbox, submitted, solution_path)
    return submitted, solution_path, turns_used, total_input_tokens, total_output_tokens


def _run_reasoning_agent(
    model_config: ModelConfig,
    sandbox,
    system_prompt: str,
    initial_user_message: str,
    max_turns: int
) -> tuple:
    """Run reasoning model agent that extracts code from text output.

    For models like kimi-k2.5 that use reasoning mode instead of tool calls.
    The model outputs code directly in markdown blocks, which we extract and test.

    Returns: (submitted, solution_path, turns_used, input_tokens, output_tokens)
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )

    submitted = False
    solution_path = "solution.py"
    turns_used = 0
    total_input_tokens = 0
    total_output_tokens = 0

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_user_message}
    ]

    for turn in range(max_turns):
        turns_used = turn + 1
        print(f"\n[Turn {turn + 1}/{max_turns}]", flush=True)

        try:
            # Call API with reasoning enabled
            response = client.chat.completions.create(
                model=model_config.model_id,
                messages=messages,
                max_tokens=16384,
                extra_body={"reasoning": {"enabled": True}}
            )
        except Exception as e:
            print(f"API error: {e}", flush=True)
            break

        # Track tokens
        if hasattr(response, 'usage') and response.usage:
            input_toks = getattr(response.usage, 'prompt_tokens', 0)
            output_toks = getattr(response.usage, 'completion_tokens', 0)
            total_input_tokens += input_toks
            total_output_tokens += output_toks

        # Extract response content
        content = response.choices[0].message.content or ""
        text_preview = content[:300] + "..." if len(content) > 300 else content
        print(f"Assistant: {text_preview}", flush=True)
        _write_turn_artifact(turn + 1, "response.txt", content)

        # Add assistant message to history
        messages.append({"role": "assistant", "content": content})

        # Extract Python code from response
        code = extract_python_code(content)
        if not code:
            print("  No Python code found in response", flush=True)
            messages.append({
                "role": "user",
                "content": "I couldn't find a Python code block in your response. Please provide the complete solution.py in a ```python code block."
            })
            continue

        print(f"  Extracted {len(code)} chars of Python code", flush=True)

        # Write code to sandbox
        sandbox.write_file("solution.py", code)
        _write_turn_artifact(turn + 1, "solution.py", code)

        # Test if it compiles
        print("  Testing compilation...", flush=True)
        compile_result = sandbox.run_command(
            'python -c "from solution import Model; m = Model(); print(\'OK\')"',
            timeout=120
        )
        _write_turn_artifact(
            turn + 1,
            "compile.log",
            _format_command_result('python -c "from solution import Model; m = Model(); print(\'OK\')"', compile_result),
        )

        if compile_result["returncode"] == 0 and "OK" in compile_result["stdout"]:
            print("  Compilation: OK", flush=True)
            submitted = True
            break
        else:
            # Compilation failed - feed error back to model
            error_msg = compile_result["stderr"] or compile_result["stdout"] or "Unknown error"
            # Truncate very long errors
            if len(error_msg) > 2000:
                error_msg = error_msg[:2000] + "\n... (truncated)"
            print(f"  Compilation FAILED: {error_msg[:200]}...", flush=True)

            messages.append({
                "role": "user",
                "content": f"Your code failed to compile/import with this error:\n\n```\n{error_msg}\n```\n\nPlease fix the error and provide the corrected solution.py in a ```python code block."
            })

    submitted, solution_path = _auto_submit_if_compilable(sandbox, submitted, solution_path)
    return submitted, solution_path, turns_used, total_input_tokens, total_output_tokens


def run_agent_on_modal(
    model_config: ModelConfig,
    gpu: str,
    problem_code: str,
    problem_name: str,
    level: int,
    max_turns: int = 20,
    backend: str = "cuda",
) -> EvalResult:
    """Run an LLM agent on Modal or local sandbox."""
    result = EvalResult(
        model=model_config.name,
        gpu=gpu,
        problem=problem_name,
        level=level
    )

    start_time = time.time()
    gpu_name, vram = GPU_SPECS.get(gpu, ("Unknown", 80))
    system_prompt = get_system_prompt(gpu_name, vram, use_xml_tools=model_config.use_xml_tools)
    system_prompt = _augment_system_prompt(system_prompt, backend=backend)
    reference_metadata = _extract_reference_metadata(problem_code)

    # Create sandbox (Modal for cloud GPUs, local for RTX3090/LOCAL)
    if gpu in LOCAL_GPUS:
        sandbox = LocalSandbox(problem_code, LocalSandboxConfig(timeout=300))
    else:
        sandbox = ModalSandbox(problem_code, ModalSandboxConfig(gpu=gpu, timeout=300, sandbox_timeout=3600))

    alarm_handler = _begin_problem_alarm(level)
    try:
        sandbox.start()
        print(f"Sandbox started: {sandbox.get_gpu_info()}", flush=True)
        context_bundle = _prepare_workspace_context(
            backend=backend,
            gpu_name=gpu_name,
            vram_gb=vram,
            level=level,
            problem_name=problem_name,
            metadata=reference_metadata,
            sandbox=sandbox,
        )
        _seed_workspace_context(sandbox=sandbox, context_bundle=context_bundle)
        system_prompt = _inject_workspace_context(system_prompt, context_bundle)
        initial_user_message = _build_initial_user_message(
            backend=backend,
            problem_name=problem_name,
            level=level,
            gpu_name=gpu_name,
            max_turns=max_turns,
            reference_code=problem_code,
            metadata=reference_metadata,
        )

        # Use dedicated Gemini handler for native SDK
        if model_config.provider == "gemini":
            submitted, solution_path, turns_used, input_tokens, output_tokens = _run_gemini_agent(
                model_config,
                sandbox,
                system_prompt,
                initial_user_message,
                max_turns,
            )
            result.turns = turns_used
            result.submitted = submitted
            result.input_tokens = input_tokens
            result.output_tokens = output_tokens
            result.total_tokens = input_tokens + output_tokens
            result.estimated_cost_usd = _estimate_cost(model_config.model_id, model_config.provider, input_tokens, output_tokens)

            print(f"\n[Token Usage] Input: {input_tokens:,} | Output: {output_tokens:,} | Total: {input_tokens + output_tokens:,}", flush=True)
            if result.estimated_cost_usd:
                print(f"[Est. Cost] ${result.estimated_cost_usd:.4f}", flush=True)

            if submitted and solution_path:
                print("\n" + "=" * 60, flush=True)
                print("RUNNING BENCHMARK", flush=True)
                print("=" * 60, flush=True)

                _attach_solution_metadata(result, solution_path, sandbox)
                benchmark_result = _run_benchmark(
                    sandbox,
                    solution_path,
                    hardware=gpu,
                    level=level,
                    backend=backend,
                )
                _apply_benchmark_metrics(result, benchmark_result)
            else:
                result.error = "No solution submitted"

            result.elapsed_seconds = time.time() - start_time
            return result

        # Use dedicated reasoning handler for models without tool use
        if model_config.reasoning_mode:
            reasoning_prompt = get_reasoning_system_prompt(gpu_name, vram)
            reasoning_prompt = _augment_system_prompt(reasoning_prompt, backend=backend)
            reasoning_prompt = _inject_workspace_context(reasoning_prompt, context_bundle)
            submitted, solution_path, turns_used, input_tokens, output_tokens = _run_reasoning_agent(
                model_config,
                sandbox,
                reasoning_prompt,
                initial_user_message,
                max_turns,
            )
            result.turns = turns_used
            result.submitted = submitted
            result.input_tokens = input_tokens
            result.output_tokens = output_tokens
            result.total_tokens = input_tokens + output_tokens
            result.estimated_cost_usd = _estimate_cost(model_config.model_id, model_config.provider, input_tokens, output_tokens)

            print(f"\n[Token Usage] Input: {input_tokens:,} | Output: {output_tokens:,} | Total: {input_tokens + output_tokens:,}", flush=True)
            if result.estimated_cost_usd:
                print(f"[Est. Cost] ${result.estimated_cost_usd:.4f}", flush=True)

            if submitted and solution_path:
                print("\n" + "=" * 60, flush=True)
                print("RUNNING BENCHMARK", flush=True)
                print("=" * 60, flush=True)

                _attach_solution_metadata(result, solution_path, sandbox)
                benchmark_result = _run_benchmark(
                    sandbox,
                    solution_path,
                    hardware=gpu,
                    level=level,
                    backend=backend,
                )
                _apply_benchmark_metrics(result, benchmark_result)
            else:
                result.error = "No solution submitted"

            result.elapsed_seconds = time.time() - start_time
            return result

        # Standard flow for other providers
        client = get_provider_client(model_config.provider)
        messages = []

        if model_config.provider == "anthropic":
            messages = [{"role": "user", "content": initial_user_message}]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": initial_user_message},
            ]

        submitted = False
        solution_path = None
        total_input_tokens = 0
        total_output_tokens = 0
        total_cache_creation_tokens = 0
        total_cache_read_tokens = 0

        for turn in range(max_turns):
            result.turns = turn + 1
            print(f"\n[Turn {turn + 1}/{max_turns}]", flush=True)

            # Get response from model
            try:
                response = _get_model_response(
                    client, model_config, system_prompt, messages
                )
            except Exception as e:
                print(f"API error: {e}", flush=True)
                result.error = f"API error: {e}"
                break

            # Track token usage from response (including cache tokens)
            input_toks, output_toks, cache_create, cache_read = _extract_token_usage(response, model_config)
            total_input_tokens += input_toks
            total_output_tokens += output_toks
            total_cache_creation_tokens += cache_create
            total_cache_read_tokens += cache_read

            # Process response
            assistant_content, tool_calls = _parse_response(response, model_config)

            # Log response
            if isinstance(assistant_content, str) and assistant_content:
                text = assistant_content[:200] + "..." if len(assistant_content) > 200 else assistant_content
                print(f"Assistant: {text}", flush=True)
            turn_payload = {
                "assistant_content": assistant_content,
                "tool_calls": tool_calls,
            }
            _write_turn_artifact(turn + 1, "response.txt", json.dumps(turn_payload, indent=2, default=str))

            # Add assistant message
            messages.append(_format_assistant_message(assistant_content, tool_calls, model_config))

            if not tool_calls:
                print("No tool calls - agent finished", flush=True)
                break

            # Execute tools
            tool_results = []
            turn_tool_logs: List[str] = []
            for tc in tool_calls:
                tool_name = tc["name"]
                tool_input = tc["input"]
                tool_id = tc.get("id", f"tool_{turn}")

                print(f"  Tool: {tool_name}", flush=True)

                if tool_name == "bash":
                    cmd = tool_input.get("command", "")
                    print(f"    $ {cmd[:80]}..." if len(cmd) > 80 else f"    $ {cmd}", flush=True)
                    cmd_result = sandbox.run_command(cmd)
                    output = f"stdout:\n{cmd_result['stdout']}\nstderr:\n{cmd_result['stderr']}\nreturn_code: {cmd_result['returncode']}"
                    turn_tool_logs.append(_format_command_result(cmd, cmd_result))
                    if cmd_result["stdout"]:
                        out = cmd_result["stdout"][:150] + "..." if len(cmd_result["stdout"]) > 150 else cmd_result["stdout"]
                        print(f"    -> {out}", flush=True)
                    tool_results.append({"id": tool_id, "name": tool_name, "content": output})

                elif tool_name == "submit":
                    solution_path = tool_input.get("solution_path", "solution.py")
                    submitted = True
                    result.submitted = True
                    print(f"  SUBMITTED: {solution_path}", flush=True)
                    tool_results.append({"id": tool_id, "name": tool_name, "content": f"Submitted: {solution_path}"})

            _write_turn_artifact(turn + 1, "compile.log", "\n\n".join(turn_tool_logs))
            if sandbox.file_exists("solution.py"):
                sol_snapshot = sandbox.read_file("solution.py") or ""
                _write_turn_artifact(turn + 1, "solution.py", sol_snapshot)

            # Add tool results to messages
            messages.extend(_format_tool_results(tool_results, model_config))

            if submitted:
                break

        submitted, solution_path = _auto_submit_if_compilable(sandbox, submitted, solution_path)
        result.submitted = submitted

        # Store token usage (including cache stats)
        result.input_tokens = total_input_tokens
        result.output_tokens = total_output_tokens
        result.total_tokens = total_input_tokens + total_output_tokens
        result.cache_creation_tokens = total_cache_creation_tokens
        result.cache_read_tokens = total_cache_read_tokens
        result.estimated_cost_usd = _estimate_cost(
            model_config.model_id, model_config.provider,
            total_input_tokens, total_output_tokens,
            total_cache_creation_tokens, total_cache_read_tokens
        )

        # Log token usage with cache info
        cache_info = ""
        if total_cache_creation_tokens or total_cache_read_tokens:
            cache_info = f" | Cache Create: {total_cache_creation_tokens:,} | Cache Read: {total_cache_read_tokens:,}"
        print(f"\n[Token Usage] Input: {total_input_tokens:,} | Output: {total_output_tokens:,} | Total: {total_input_tokens + total_output_tokens:,}{cache_info}", flush=True)
        if result.estimated_cost_usd:
            print(f"[Est. Cost] ${result.estimated_cost_usd:.4f}", flush=True)

        # Run benchmark if submitted
        if submitted and solution_path:
            print("\n" + "=" * 60, flush=True)
            print("RUNNING BENCHMARK", flush=True)
            print("=" * 60, flush=True)

            _attach_solution_metadata(result, solution_path, sandbox)
            benchmark_result = _run_benchmark(
                sandbox,
                solution_path,
                hardware=gpu,
                level=level,
                backend=backend,
            )
            _apply_benchmark_metrics(result, benchmark_result)

        else:
            result.error = "No solution submitted"

    except TimeoutError:
        result.error = "timeout_exceeded"
    except Exception as e:
        import traceback
        traceback.print_exc()
        result.error = str(e)

    finally:
        _clear_problem_alarm(alarm_handler)
        sandbox.stop()

    result.elapsed_seconds = time.time() - start_time
    return result


def _extract_token_usage(response, model_config: ModelConfig) -> tuple:
    """Extract input and output token counts from API response.

    Returns: (input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens)
    Cache tokens are 0 if not available/applicable.
    """
    input_tokens = 0
    output_tokens = 0
    cache_creation_tokens = 0
    cache_read_tokens = 0

    if model_config.provider == "anthropic":
        # Anthropic response.usage includes cache token counts
        if hasattr(response, 'usage') and response.usage:
            input_tokens = getattr(response.usage, 'input_tokens', 0)
            output_tokens = getattr(response.usage, 'output_tokens', 0)
            cache_creation_tokens = getattr(response.usage, 'cache_creation_input_tokens', 0)
            cache_read_tokens = getattr(response.usage, 'cache_read_input_tokens', 0)
    else:
        # OpenAI-compatible (openai, xai, openrouter)
        if hasattr(response, 'usage') and response.usage:
            input_tokens = getattr(response.usage, 'prompt_tokens', 0)
            output_tokens = getattr(response.usage, 'completion_tokens', 0)
            # OpenRouter returns cache info in prompt_tokens_details
            details = getattr(response.usage, 'prompt_tokens_details', None)
            if details:
                cache_read_tokens = getattr(details, 'cached_tokens', 0)

    return input_tokens, output_tokens, cache_creation_tokens, cache_read_tokens


def _get_pricing(model_id: str, provider: str) -> Optional[Tuple[float, float]]:
    """Get pricing for a model (input, output per million tokens).

    All pricing is fetched dynamically from OpenRouter API.
    OpenRouter has pricing for all providers (Anthropic, OpenAI, Google, xAI, etc).
    """
    # Map our internal model ID to OpenRouter model ID
    openrouter_id = MODEL_TO_OPENROUTER.get(model_id, model_id)

    # Fetch pricing from OpenRouter
    return get_openrouter_pricing(openrouter_id)


def _estimate_cost(
    model_id: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0
) -> Optional[float]:
    """Estimate cost in USD based on model pricing.

    Cache pricing (Anthropic):
    - Cache creation: 1.25x input price (25% premium)
    - Cache reads: 0.10x input price (90% savings)

    Note: input_tokens from API already includes non-cached input.
    Cache read tokens get the discounted rate.
    """
    pricing = _get_pricing(model_id, provider)
    if pricing is None:
        return None

    input_price, output_price = pricing

    # Base cost for non-cached input and output
    # Note: For Anthropic, input_tokens is total input minus cache reads
    # For providers without caching breakdown, this is just normal pricing
    base_input_cost = input_tokens * input_price / 1_000_000
    output_cost = output_tokens * output_price / 1_000_000

    # Cache costs (Anthropic pricing model)
    # Cache creation: 1.25x input price
    # Cache reads: 0.10x input price (90% savings)
    cache_creation_cost = cache_creation_tokens * (input_price * 1.25) / 1_000_000
    cache_read_cost = cache_read_tokens * (input_price * 0.10) / 1_000_000

    cost = base_input_cost + output_cost + cache_creation_cost + cache_read_cost
    return round(cost, 6)


def _get_model_response(client, model_config: ModelConfig, system_prompt: str, messages: list):
    """Get response from model."""
    if model_config.provider == "anthropic":
        # Use prompt caching for Anthropic - format system as content block with cache_control
        # Cache reads are 0.25x input token cost (75% savings)
        system_with_cache = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ]
        kwargs = {
            "model": model_config.model_id,
            "max_tokens": 8192,
            "system": system_with_cache,
            "messages": messages
        }
        if not model_config.use_xml_tools:
            kwargs["tools"] = TOOLS_ANTHROPIC
        return client.messages.create(**kwargs)

    elif model_config.provider == "openai":
        kwargs = {
            "model": model_config.model_id,
            "max_completion_tokens": 8192,
            "messages": messages
        }
        if not model_config.use_xml_tools:
            kwargs["tools"] = TOOLS_OPENAI
        return client.chat.completions.create(**kwargs)

    else:
        # OpenAI-compatible (gemini, xai, openrouter)
        # For OpenRouter, use prompt caching with cache_control on system message
        # This works for Anthropic models via OpenRouter (75% savings on cache reads)
        # For other providers (DeepSeek, GLM, etc.), automatic caching applies
        if model_config.provider == "openrouter":
            # Format system message with cache_control for OpenRouter
            # OpenRouter supports Anthropic-style caching for all providers
            cached_messages = []
            for msg in messages:
                if msg.get("role") == "system":
                    cached_messages.append({
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": msg["content"],
                                "cache_control": {"type": "ephemeral"}
                            }
                        ]
                    })
                else:
                    cached_messages.append(msg)
            kwargs = {
                "model": model_config.model_id,
                "max_tokens": 8192,
                "messages": cached_messages
            }
        else:
            # Gemini, xAI - automatic caching, no special format needed
            kwargs = {
                "model": model_config.model_id,
                "max_tokens": 8192,
                "messages": messages
            }
        if not model_config.use_xml_tools:
            kwargs["tools"] = TOOLS_OPENAI
        return client.chat.completions.create(**kwargs)


def _parse_response(response, model_config: ModelConfig) -> tuple:
    """Parse response into content and tool calls."""
    if model_config.provider == "anthropic":
        content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })
        # Also check for XML tool calls in content
        if model_config.use_xml_tools and not tool_calls:
            tool_calls = parse_xml_tool_calls(content)
        return content, tool_calls

    else:
        # OpenAI-compatible
        message = response.choices[0].message
        content = message.content or ""
        tool_calls = []

        # First try native function calling
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments)
                })

        # For XML mode or if no native tool calls, parse XML from content
        if model_config.use_xml_tools and not tool_calls and content:
            tool_calls = parse_xml_tool_calls(content)

        return content, tool_calls


def _format_assistant_message(content, tool_calls, model_config: ModelConfig) -> dict:
    """Format assistant message for conversation history."""
    # For XML tool calling, just return the content as-is (tool calls are in the text)
    if model_config.use_xml_tools:
        return {"role": "assistant", "content": content or ""}

    if model_config.provider == "anthropic":
        blocks = []
        if content:
            blocks.append({"type": "text", "text": content})
        for tc in tool_calls:
            blocks.append({
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["name"],
                "input": tc["input"]
            })
        return {"role": "assistant", "content": blocks}

    else:
        msg = {"role": "assistant"}
        if content:
            msg["content"] = content
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["input"])
                    }
                }
                for tc in tool_calls
            ]
        return msg


def _format_tool_results(tool_results: list, model_config: ModelConfig) -> list:
    """Format tool results for conversation history."""
    # For XML tool calling, return results as a regular user message
    if model_config.use_xml_tools:
        result_text = ""
        for tr in tool_results:
            result_text += f"<tool_result name=\"{tr['name']}\">\n{tr['content']}\n</tool_result>\n\n"
        return [{"role": "user", "content": result_text.strip()}]

    if model_config.provider == "anthropic":
        return [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tr["id"], "content": tr["content"]}
                for tr in tool_results
            ]
        }]

    else:
        return [
            {"role": "tool", "tool_call_id": tr["id"], "content": tr["content"]}
            for tr in tool_results
        ]


def _run_benchmark(
    sandbox: ModalSandbox,
    solution_path: str,
    hardware: Optional[str] = None,
    level: Optional[int] = None,
    backend: str = "cuda",
) -> dict:
    """Run benchmark on the submitted solution."""
    if not solution_path.startswith("/"):
        solution_path = f"/workspace/{solution_path}"

    if not sandbox.file_exists(solution_path.replace("/workspace/", "")):
        return {"compiled": False, "error": f"Solution not found: {solution_path}"}

    solution_code = sandbox.read_file(solution_path.replace("/workspace/", ""))
    guardrail_error = validate_solution_guardrails(solution_code, backend=backend)
    if guardrail_error:
        return {
            "compiled": False,
            "correct": False,
            "speedup": None,
            "error": guardrail_error,
        }

    benchmark_template = '''
import json
import importlib.util
import statistics
import sys
import traceback

import torch

device = torch.device("cuda:0")
HARDWARE = __HARDWARE__
HARDWARE_PRECISIONS = __HARDWARE_PRECISIONS__
OP_PRECISION_VALIDITY = __OP_PRECISION_VALIDITY__
HARDWARE_PEAK_TFLOPS = __HARDWARE_PEAK_TFLOPS__

def dtype_to_precision(dtype):
    text = str(dtype)
    if "float8" in text:
        return "fp8"
    if "bfloat16" in text:
        return "bf16"
    if "float16" in text:
        return "fp16"
    if "float32" in text:
        return "fp32"
    if "float64" in text:
        return "fp64"
    return text.replace("torch.", "")

def get_valid_precisions(hardware, op_type):
    hw_precs = set(HARDWARE_PRECISIONS.get(hardware, ["fp32"]))
    op_precs = set(OP_PRECISION_VALIDITY.get(op_type, ["fp32"]))
    return sorted(hw_precs & op_precs)

def infer_op_type(inputs):
    if len(inputs) >= 2 and isinstance(inputs[0], torch.Tensor) and isinstance(inputs[1], torch.Tensor):
        a, b = inputs[0], inputs[1]
        if a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]:
            return "gemm"
    return "unknown"

def infer_problem_size(op_type, inputs):
    if op_type == "gemm" and len(inputs) >= 2 and isinstance(inputs[0], torch.Tensor) and isinstance(inputs[1], torch.Tensor):
        a, b = inputs[0], inputs[1]
        if a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]:
            m = int(a.shape[0])
            n = int(b.shape[1])
            k = int(a.shape[1])
            return [m, n, k]
    return None

def compute_tflops(op_type, problem_size, time_ms):
    if not problem_size or not time_ms or time_ms <= 0:
        return None
    if op_type == "gemm":
        m, n, k = problem_size
        flops = 2 * m * n * k
    elif op_type == "attention":
        b, h, s, d = problem_size
        flops = 4 * b * h * s * s * d
    else:
        return None
    return (flops / 1e12) / (time_ms / 1000.0)

def compute_percent_of_peak(achieved_tflops, hardware, precision):
    if achieved_tflops is None:
        return None
    peak = HARDWARE_PEAK_TFLOPS.get(hardware, {}).get(precision)
    if peak is None or peak <= 0:
        return None
    return (achieved_tflops / peak) * 100.0

PRECISION_TOLERANCES = {
    "fp4": {"atol": 0.5, "rtol": 0.1},
    "fp8": {"atol": 0.1, "rtol": 0.05},
    "fp16": {"atol": 0.01, "rtol": 0.01},
    "bf16": {"atol": 0.01, "rtol": 0.01},
    "fp32": {"atol": 0.001, "rtol": 0.001},
}

REPEATABILITY_CHECK = True
REPEATABILITY_RUNS = 2

def get_tolerance(precision):
    return PRECISION_TOLERANCES.get(precision, {"atol": 0.05, "rtol": 0.02})

def check_valid_output(tensor, name="output"):
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan:
        return False, f"{name} contains NaN", True, bool(has_inf)
    if has_inf:
        return False, f"{name} contains Inf", bool(has_nan), True
    return True, "", False, False

try:
    def load_module(module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load module from {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    reference_module = load_module("kb_reference", "reference.py")
    solution_module = load_module("kb_solution", "solution.py")
    RefModel, SolModel = reference_module.Model, solution_module.Model
    get_inputs = reference_module.get_inputs
    get_init_inputs = reference_module.get_init_inputs

    op_type = str(getattr(reference_module, "OP_TYPE", "unknown")).lower()
    declared_supported_precisions = getattr(reference_module, "SUPPORTED_PRECISIONS", [])
    if not isinstance(declared_supported_precisions, (list, tuple)):
        declared_supported_precisions = []
    declared_supported_precisions = [str(p).lower() for p in declared_supported_precisions]

    print("Loading models...", flush=True)
    ref_model = RefModel(*get_init_inputs()).to(device).eval()
    sol_model = SolModel(*get_init_inputs()).to(device).eval()
    if not torch.cuda.is_available():
        print(json.dumps({"compiled": False, "correct": False, "speedup": None, "error": "CUDA unavailable in benchmark runtime"}))
        sys.exit(0)

    CORRECTNESS_SEEDS = [42, 123, 456, 789, 1337]
    worst_max_diff, worst_tolerance, worst_seed = 0.0, 0.0, CORRECTNESS_SEEDS[0]
    precision = "fp32"
    tol = get_tolerance(precision)
    has_nan = False
    has_inf = False
    is_deterministic = True

    print("Checking correctness across seeds...", flush=True)
    for seed in CORRECTNESS_SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in get_inputs()]
        for value in inputs:
            if isinstance(value, torch.Tensor):
                precision = dtype_to_precision(value.dtype)
                break
        tol = get_tolerance(precision)
        with torch.no_grad():
            ref_out, sol_out = ref_model(*inputs), sol_model(*inputs)

        if not isinstance(ref_out, torch.Tensor) or not isinstance(sol_out, torch.Tensor):
            print(json.dumps({
                "compiled": False,
                "correct": False,
                "speedup": None,
                "error": "Only tensor outputs are supported",
                "precision_used": precision,
                "tolerance_atol": tol["atol"],
                "tolerance_rtol": tol["rtol"],
                "has_nan": has_nan,
                "has_inf": has_inf,
                "is_deterministic": is_deterministic,
            }))
            sys.exit(0)
        if ref_out.shape != sol_out.shape:
            print(json.dumps({
                "compiled": True,
                "correct": False,
                "speedup": None,
                "error": f"shape_mismatch_seed={seed}: {tuple(ref_out.shape)} vs {tuple(sol_out.shape)}",
                "precision_used": precision,
                "tolerance_atol": tol["atol"],
                "tolerance_rtol": tol["rtol"],
                "has_nan": has_nan,
                "has_inf": has_inf,
                "is_deterministic": is_deterministic,
            }))
            sys.exit(0)

        ref_valid, ref_error, ref_has_nan, ref_has_inf = check_valid_output(ref_out, "reference output")
        sol_valid, sol_error, sol_has_nan, sol_has_inf = check_valid_output(sol_out, "solution output")
        has_nan = has_nan or ref_has_nan or sol_has_nan
        has_inf = has_inf or ref_has_inf or sol_has_inf
        if not ref_valid:
            print(json.dumps({
                "compiled": True,
                "correct": False,
                "speedup": None,
                "error": ref_error,
                "precision_used": precision,
                "tolerance_atol": tol["atol"],
                "tolerance_rtol": tol["rtol"],
                "has_nan": has_nan,
                "has_inf": has_inf,
                "is_deterministic": is_deterministic,
            }))
            sys.exit(0)
        if not sol_valid:
            print(json.dumps({
                "compiled": True,
                "correct": False,
                "speedup": None,
                "error": sol_error,
                "precision_used": precision,
                "tolerance_atol": tol["atol"],
                "tolerance_rtol": tol["rtol"],
                "has_nan": has_nan,
                "has_inf": has_inf,
                "is_deterministic": is_deterministic,
            }))
            sys.exit(0)

        ref_f, sol_f = ref_out.float(), sol_out.float()
        max_diff = (ref_f - sol_f).abs().max().item()
        max_ref = ref_f.abs().max().item()
        tolerance = tol["atol"] + tol["rtol"] * max_ref
        if max_diff > worst_max_diff:
            worst_max_diff = max_diff
            worst_tolerance = tolerance
            worst_seed = seed
        if max_diff >= tolerance:
            print(json.dumps({
                "compiled": True,
                "correct": False,
                "speedup": None,
                "error": f"seed={seed}, max_diff={max_diff}",
                "precision_used": precision,
                "tolerance_atol": tol["atol"],
                "tolerance_rtol": tol["rtol"],
                "has_nan": has_nan,
                "has_inf": has_inf,
                "is_deterministic": is_deterministic,
            }))
            sys.exit(0)

    print(f"worst_seed: {worst_seed}, max_diff: {worst_max_diff:.6f}, tolerance: {worst_tolerance:.6f}", flush=True)

    benchmark_seed = 2026
    torch.manual_seed(benchmark_seed)
    torch.cuda.manual_seed_all(benchmark_seed)
    bench_inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in get_inputs()]

    if op_type == "unknown":
        op_type = infer_op_type(bench_inputs)

    precision = "fp32"
    for value in bench_inputs:
        if isinstance(value, torch.Tensor):
            precision = dtype_to_precision(value.dtype)
            break
    tol = get_tolerance(precision)

    valid_precisions = get_valid_precisions(HARDWARE, op_type)
    if declared_supported_precisions:
        valid_precisions = sorted(set(valid_precisions) & set(declared_supported_precisions))
    precision_supported = precision in valid_precisions if valid_precisions else None
    baseline_type = "cutlass" if precision == "fp4" and HARDWARE == "B200" else "pytorch"

    problem_size = infer_problem_size(op_type, bench_inputs)

    if REPEATABILITY_CHECK:
        repeat_outputs = []
        for _ in range(REPEATABILITY_RUNS):
            with torch.no_grad():
                repeat_out = sol_model(*bench_inputs)
            torch.cuda.synchronize()
            repeat_outputs.append(repeat_out.clone())

        for idx in range(1, len(repeat_outputs)):
            if not torch.equal(repeat_outputs[0], repeat_outputs[idx]):
                is_deterministic = False
                print(json.dumps({
                    "compiled": True,
                    "correct": False,
                    "speedup": None,
                    "error": "Non-deterministic output (possible race condition)",
                    "precision_used": precision,
                    "tolerance_atol": tol["atol"],
                    "tolerance_rtol": tol["rtol"],
                    "has_nan": has_nan,
                    "has_inf": has_inf,
                    "is_deterministic": is_deterministic,
                }))
                sys.exit(0)

    def count_kernels(model, model_inputs):
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.no_grad():
                model(*model_inputs)
        torch.cuda.synchronize()
        return sum(1 for e in prof.key_averages() if e.device_type == torch.profiler.DeviceType.CUDA)

    ref_kernels = count_kernels(ref_model, bench_inputs)
    sol_kernels = count_kernels(sol_model, bench_inputs)
    print(f"Kernel count: ref={ref_kernels}, sol={sol_kernels}", flush=True)

    print("Benchmarking...", flush=True)
    WARMUP_ITERS = 5
    TIMED_ITERS = 30

    def summarize_runtime_ms(model, model_inputs):
        for _ in range(WARMUP_ITERS):
            with torch.no_grad():
                model(*model_inputs)
        torch.cuda.synchronize()

        times_ms = []
        for _ in range(TIMED_ITERS):
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            with torch.no_grad():
                model(*model_inputs)
            end_evt.record()
            torch.cuda.synchronize()
            times_ms.append(start_evt.elapsed_time(end_evt))

        ordered = sorted(times_ms)
        n = len(ordered)
        p10_idx = int(0.10 * (n - 1))
        p90_idx = int(0.90 * (n - 1))
        return {
            "median": statistics.median(ordered),
            "mean": statistics.mean(ordered),
            "std": statistics.pstdev(ordered),
            "p10": ordered[p10_idx],
            "p90": ordered[p90_idx],
        }

    ref_stats = summarize_runtime_ms(ref_model, bench_inputs)
    sol_stats = summarize_runtime_ms(sol_model, bench_inputs)
    ref_ms, sol_ms = ref_stats["median"], sol_stats["median"]
    ref_mean_ms, sol_mean_ms = ref_stats["mean"], sol_stats["mean"]
    ref_std_ms, sol_std_ms = ref_stats["std"], sol_stats["std"]
    ref_p10_ms, ref_p90_ms = ref_stats["p10"], ref_stats["p90"]
    sol_p10_ms, sol_p90_ms = sol_stats["p10"], sol_stats["p90"]

    ref_tflops = compute_tflops(op_type, problem_size, ref_ms)
    achieved_tflops = compute_tflops(op_type, problem_size, sol_ms)
    ref_pct_of_peak = compute_percent_of_peak(ref_tflops, HARDWARE, precision)
    pct_of_peak = compute_percent_of_peak(achieved_tflops, HARDWARE, precision)

    print(json.dumps({
        "compiled": True,
        "correct": True,
        "speedup": ref_ms / sol_ms,
        "ref_ms": ref_ms,
        "sol_ms": sol_ms,
        "ref_mean_ms": ref_mean_ms,
        "sol_mean_ms": sol_mean_ms,
        "ref_std_ms": ref_std_ms,
        "sol_std_ms": sol_std_ms,
        "ref_p10_ms": ref_p10_ms,
        "ref_p90_ms": ref_p90_ms,
        "sol_p10_ms": sol_p10_ms,
        "sol_p90_ms": sol_p90_ms,
        "ref_kernels": ref_kernels,
        "sol_kernels": sol_kernels,
        "seeds_tested": len(CORRECTNESS_SEEDS),
        "correctness_seeds": CORRECTNESS_SEEDS,
        "benchmark_seed": benchmark_seed,
        "baseline_type": baseline_type,
        "precision": precision,
        "precision_used": precision,
        "valid_precisions": valid_precisions,
        "precision_supported": precision_supported,
        "tolerance_atol": tol["atol"],
        "tolerance_rtol": tol["rtol"],
        "has_nan": has_nan,
        "has_inf": has_inf,
        "is_deterministic": is_deterministic,
        "op_type": op_type,
        "problem_size": problem_size,
        "achieved_tflops": achieved_tflops,
        "ref_tflops": ref_tflops,
        "pct_of_peak": pct_of_peak,
        "ref_pct_of_peak": ref_pct_of_peak,
    }))
except Exception as e:
    traceback.print_exc()
    print(json.dumps({
        "compiled": False,
        "correct": False,
        "speedup": None,
        "error": str(e),
        "precision_used": None,
        "tolerance_atol": None,
        "tolerance_rtol": None,
        "has_nan": False,
        "has_inf": False,
        "is_deterministic": True,
    }))
'''

    benchmark_script = (
        benchmark_template
        .replace("__HARDWARE__", json.dumps(hardware or "UNKNOWN"))
        .replace("__HARDWARE_PRECISIONS__", json.dumps(HARDWARE_PRECISIONS))
        .replace("__OP_PRECISION_VALIDITY__", json.dumps(OP_PRECISION_VALIDITY))
        .replace("__HARDWARE_PEAK_TFLOPS__", json.dumps(HARDWARE_PEAK_TFLOPS))
    )

    sandbox.write_file("_benchmark.py", benchmark_script)
    benchmark_timeout = MAX_PROBLEM_TIME_SECONDS.get(level or 1, 600) + 120
    result = sandbox.run_command("python _benchmark.py", timeout=benchmark_timeout)

    print(f"Benchmark output:\n{result['stdout']}", flush=True)
    if result["stderr"]:
        print(f"Errors:\n{result['stderr']}", flush=True)

    for line in result["stdout"].split("\n"):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                continue

    return {"compiled": False, "error": "Failed to parse benchmark output"}


FORBIDDEN_SOLUTION_PATTERNS = [
    (
        re.compile(r"torch::\s*(?:mm|matmul|conv1d|conv2d|conv3d|linear)\s*\("),
        "Forbidden C++ wrapper fallback to PyTorch operator",
    ),
    (
        re.compile(r"(?:^|[^\w])torch\.(?:mm|matmul|conv1d|conv2d|conv3d|linear)\s*\("),
        "Forbidden Python fallback to PyTorch operator",
    ),
    (
        re.compile(r"(?:^|[^\w])F\.(?:linear|conv1d|conv2d|conv3d)\s*\("),
        "Forbidden Python fallback via torch.nn.functional",
    ),
    (
        re.compile(r"(?:^|[^\w])torch\.compile\s*\("),
        "Forbidden use of torch.compile",
    ),
    (
        re.compile(r"@torch\.jit\.script"),
        "Forbidden use of torch.jit.script",
    ),
]

GRAPHICS_REQUIRED_PATTERNS = [
    (
        re.compile(r"(?:^|\n)\s*(?:import\s+triton\b|from\s+triton\b)"),
        "Missing Triton import for GraphicsBench",
    ),
    (
        re.compile(r"@triton\.jit"),
        "Missing `@triton.jit` kernel for GraphicsBench",
    ),
    (
        re.compile(r"[A-Za-z_]\w*\s*\[[^\]]+\]\s*\("),
        "Missing Triton kernel launch `kernel[grid](...)` for GraphicsBench",
    ),
]

GRAPHICS_FORBIDDEN_PATTERNS = [
    (
        re.compile(r"(?:^|[^\w])reference\.Model(?:\W|$)"),
        "Forbidden direct fallback to reference model in GraphicsBench",
    ),
    (
        re.compile(r"(?:^|\n)\s*(?:import\s+OpenGL\b|from\s+OpenGL\b)"),
        "Forbidden OpenGL runtime path in GraphicsBench",
    ),
    (
        re.compile(r"(?:^|\n)\s*(?:import\s+moderngl\b|from\s+moderngl\b)"),
        "Forbidden Moderngl runtime path in GraphicsBench",
    ),
]

def validate_solution_guardrails(solution_code: str, backend: str = "cuda") -> Optional[str]:
    """Reject common reward-hacking shortcuts before expensive benchmarking."""
    for pattern, message in FORBIDDEN_SOLUTION_PATTERNS:
        match = pattern.search(solution_code)
        if match:
            snippet = match.group(0).strip()
            return f"{message}: `{snippet}`"

    if backend == "graphics":
        for pattern, message in GRAPHICS_FORBIDDEN_PATTERNS:
            match = pattern.search(solution_code)
            if match:
                snippet = match.group(0).strip()
                return f"{message}: `{snippet}`"

        for pattern, message in GRAPHICS_REQUIRED_PATTERNS:
            if not pattern.search(solution_code):
                return message
    return None


# =============================================================================
# Main
# =============================================================================

def find_problems(levels: List[int]) -> List[tuple]:
    """Find all problems for specified levels."""
    problems = []
    kernelbench_dir = PROJECT_ROOT / "KernelBench"

    for level in levels:
        level_dir = kernelbench_dir / f"level{level}"
        if level_dir.exists():
            for problem_file in sorted(level_dir.glob("*.py")):
                if problem_file.name.startswith("_"):
                    continue
                problems.append((level, problem_file))

    return problems


def main():
    parser = argparse.ArgumentParser(description="Modal-based KernelBench Evaluation")
    parser.add_argument("--model", type=str, help="Model key (e.g., claude-opus-4.5)")
    parser.add_argument("--gpu", type=str, default="H100", help="GPU type: L40S, A100, H100, B200, RTX3090, LOCAL")
    parser.add_argument("--problem", type=str, help="Problem path (e.g., level4/1_Qwen3-0p6B_bs32_seq256.py)")
    parser.add_argument("--max-turns", type=int, default=20, help="Maximum turns per problem")
    parser.add_argument("--output-dir", type=str, default="outputs/modal_eval", help="Output directory")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list_models:
        print("Predefined models:")
        for key, cfg in MODELS.items():
            print(f"  {key}: {cfg.name} ({cfg.provider})")
        print("\nOpenRouter models:")
        print("  Any valid OpenRouter model ID can be used (e.g., 'anthropic/claude-3-opus')")
        print("  Pricing is fetched dynamically from OpenRouter API")
        return

    if not args.model or not args.problem:
        parser.print_help()
        return

    # Load model config (supports predefined keys and dynamic OpenRouter models)
    model_config = get_model_config(args.model)
    if model_config is None:
        print(f"Unknown model: {args.model}")
        print(f"Predefined models: {list(MODELS.keys())}")
        print("Or use any valid OpenRouter model ID (e.g., 'anthropic/claude-3-opus')")
        return

    # Load problem
    problem_path = PROJECT_ROOT / "KernelBench" / args.problem
    if not problem_path.exists():
        print(f"Problem not found: {problem_path}")
        return

    with open(problem_path) as f:
        problem_code = f.read()

    # Determine level from path
    level = int(args.problem.split("/")[0].replace("level", ""))

    print("=" * 60)
    print(f"MODEL: {model_config.name}")
    print(f"GPU: {args.gpu}")
    print(f"PROBLEM: {args.problem}")
    print(f"LEVEL: {level}")
    print("=" * 60)

    # Run evaluation
    result = run_agent_on_modal(
        model_config=model_config,
        gpu=args.gpu,
        problem_code=problem_code,
        problem_name=problem_path.name,
        level=level,
        max_turns=args.max_turns
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Compiled: {result.compiled}")
    print(f"Correct: {result.correct}")
    if result.speedup:
        print(f"Speedup: {result.speedup:.2f}x")
    print(f"Turns: {result.turns}")
    print(f"Time: {result.elapsed_seconds:.1f}s")
    print(f"Tokens: {result.input_tokens:,} in / {result.output_tokens:,} out / {result.total_tokens:,} total")
    if result.estimated_cost_usd:
        print(f"Est. Cost: ${result.estimated_cost_usd:.4f}")
    if result.error:
        print(f"Error: {result.error}")

    # Save result
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"{args.model}_{args.gpu}_{problem_path.stem}_{timestamp}.json"
    with open(result_path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"\nResult saved: {result_path}")


if __name__ == "__main__":
    main()
