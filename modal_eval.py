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
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Literal, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.modal_sandbox import ModalSandbox, ModalSandboxConfig, GPUType


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


MODELS = {
    # Tier 1: Frontier models
    "claude-opus-4.5": ModelConfig(
        name="Claude Opus 4.5",
        model_id="claude-opus-4-5-20251101",
        provider="anthropic"
    ),
    "claude-sonnet-4.5": ModelConfig(
        name="Claude Sonnet 4.5",
        model_id="claude-sonnet-4-5-20250929",
        provider="anthropic"
    ),
    "gpt-5.2": ModelConfig(
        name="GPT-5.2",
        model_id="gpt-5.2",
        provider="openai"
    ),
    "gemini-3-flash": ModelConfig(
        name="Gemini 3 Flash",
        model_id="gemini-3-flash-preview",
        provider="gemini",
        use_xml_tools=False  # Native function calling works
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
    # Tier 2: Strong open/Chinese models via OpenRouter
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
}


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
    base_prompt = f"""You are a GPU kernel optimization expert. You have SSH access to an NVIDIA {gpu_name} GPU ({vram_gb}GB VRAM).

**YOUR TASK**: Write a custom CUDA kernel to optimize the PyTorch model in reference.py.

**CRITICAL REQUIREMENTS**:
1. You MUST write actual CUDA C++ code using `torch.utils.cpp_extension.load_inline`
2. Do NOT use torch.compile, Triton, or flash_attn
3. Your __global__ kernels MUST be called in the C++ wrapper functions
4. Do NOT fall back to PyTorch/cuBLAS in the wrapper (no torch::mm, torch::matmul, torch::conv2d, etc.)

**REQUIRED WORKFLOW** (follow exactly):
1. `cat /workspace/reference.py` - read the reference model
2. Write solution.py with your CUDA kernel implementation
3. Test: `python -c "from solution import Model; print('OK')"`
4. Submit: call the submit tool with solution.py

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


GPU_SPECS = {
    "L40S": ("L40S", 48),
    "A100": ("A100", 40),
    "H100": ("H100", 80),
    "B200": ("B200", 192),
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
    turns: int = 0
    submitted: bool = False
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    # Token usage tracking
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # Cost tracking (in USD, estimated)
    estimated_cost_usd: Optional[float] = None
    # Solution code (the submitted kernel)
    solution_code: Optional[str] = None
    # Kernel count tracking (for megakernel verification)
    ref_kernels: Optional[int] = None
    sol_kernels: Optional[int] = None


def _run_gemini_agent(
    model_config: ModelConfig,
    sandbox,
    system_prompt: str,
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
    response = chat.send_message("Optimize the model in reference.py.")

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

    return submitted, solution_path, turns_used, total_input_tokens, total_output_tokens


def run_agent_on_modal(
    model_config: ModelConfig,
    gpu: GPUType,
    problem_code: str,
    problem_name: str,
    level: int,
    max_turns: int = 20
) -> EvalResult:
    """Run an LLM agent on Modal sandbox."""
    result = EvalResult(
        model=model_config.name,
        gpu=gpu,
        problem=problem_name,
        level=level
    )

    start_time = time.time()
    gpu_name, vram = GPU_SPECS.get(gpu, ("Unknown", 80))
    system_prompt = get_system_prompt(gpu_name, vram, use_xml_tools=model_config.use_xml_tools)

    # Create Modal sandbox
    sandbox = ModalSandbox(problem_code, ModalSandboxConfig(gpu=gpu, timeout=300, sandbox_timeout=3600))

    try:
        sandbox.start()
        print(f"Sandbox started: {sandbox.get_gpu_info()}", flush=True)

        # Use dedicated Gemini handler for native SDK
        if model_config.provider == "gemini":
            submitted, solution_path, turns_used, input_tokens, output_tokens = _run_gemini_agent(
                model_config, sandbox, system_prompt, max_turns
            )
            result.turns = turns_used
            result.submitted = submitted
            result.input_tokens = input_tokens
            result.output_tokens = output_tokens
            result.total_tokens = input_tokens + output_tokens
            result.estimated_cost_usd = _estimate_cost(model_config.model_id, input_tokens, output_tokens)

            print(f"\n[Token Usage] Input: {input_tokens:,} | Output: {output_tokens:,} | Total: {input_tokens + output_tokens:,}", flush=True)
            if result.estimated_cost_usd:
                print(f"[Est. Cost] ${result.estimated_cost_usd:.4f}", flush=True)

            if submitted and solution_path:
                print("\n" + "=" * 60, flush=True)
                print("RUNNING BENCHMARK", flush=True)
                print("=" * 60, flush=True)

                # Read solution code before benchmark
                sol_path = solution_path if solution_path.startswith("/") else f"/workspace/{solution_path}"
                result.solution_code = sandbox.read_file(sol_path.replace("/workspace/", ""))

                benchmark_result = _run_benchmark(sandbox, solution_path)
                result.compiled = benchmark_result.get("compiled", False)
                result.correct = benchmark_result.get("correct", False)
                result.speedup = benchmark_result.get("speedup")
                result.ref_ms = benchmark_result.get("ref_ms")
                result.sol_ms = benchmark_result.get("sol_ms")
                result.ref_kernels = benchmark_result.get("ref_kernels")
                result.sol_kernels = benchmark_result.get("sol_kernels")
                if benchmark_result.get("error"):
                    result.error = benchmark_result["error"]
            else:
                result.error = "No solution submitted"

            result.elapsed_seconds = time.time() - start_time
            return result

        # Standard flow for other providers
        client = get_provider_client(model_config.provider)
        messages = []

        if model_config.provider == "anthropic":
            messages = [{"role": "user", "content": "Optimize the model in reference.py."}]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Optimize the model in reference.py."}
            ]

        submitted = False
        solution_path = None
        total_input_tokens = 0
        total_output_tokens = 0

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

            # Track token usage from response
            input_toks, output_toks = _extract_token_usage(response, model_config)
            total_input_tokens += input_toks
            total_output_tokens += output_toks

            # Process response
            assistant_content, tool_calls = _parse_response(response, model_config)

            # Log response
            if isinstance(assistant_content, str) and assistant_content:
                text = assistant_content[:200] + "..." if len(assistant_content) > 200 else assistant_content
                print(f"Assistant: {text}", flush=True)

            # Add assistant message
            messages.append(_format_assistant_message(assistant_content, tool_calls, model_config))

            if not tool_calls:
                print("No tool calls - agent finished", flush=True)
                break

            # Execute tools
            tool_results = []
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

            # Add tool results to messages
            messages.extend(_format_tool_results(tool_results, model_config))

            if submitted:
                break

        # Store token usage
        result.input_tokens = total_input_tokens
        result.output_tokens = total_output_tokens
        result.total_tokens = total_input_tokens + total_output_tokens
        result.estimated_cost_usd = _estimate_cost(model_config.model_id, total_input_tokens, total_output_tokens)

        print(f"\n[Token Usage] Input: {total_input_tokens:,} | Output: {total_output_tokens:,} | Total: {total_input_tokens + total_output_tokens:,}", flush=True)
        if result.estimated_cost_usd:
            print(f"[Est. Cost] ${result.estimated_cost_usd:.4f}", flush=True)

        # Run benchmark if submitted
        if submitted and solution_path:
            print("\n" + "=" * 60, flush=True)
            print("RUNNING BENCHMARK", flush=True)
            print("=" * 60, flush=True)

            # Read solution code before benchmark
            sol_path = solution_path if solution_path.startswith("/") else f"/workspace/{solution_path}"
            result.solution_code = sandbox.read_file(sol_path.replace("/workspace/", ""))

            benchmark_result = _run_benchmark(sandbox, solution_path)
            result.compiled = benchmark_result.get("compiled", False)
            result.correct = benchmark_result.get("correct", False)
            result.speedup = benchmark_result.get("speedup")
            result.ref_ms = benchmark_result.get("ref_ms")
            result.sol_ms = benchmark_result.get("sol_ms")
            result.ref_kernels = benchmark_result.get("ref_kernels")
            result.sol_kernels = benchmark_result.get("sol_kernels")
            if benchmark_result.get("error"):
                result.error = benchmark_result["error"]

        else:
            result.error = "No solution submitted"

    except Exception as e:
        import traceback
        traceback.print_exc()
        result.error = str(e)

    finally:
        sandbox.stop()

    result.elapsed_seconds = time.time() - start_time
    return result


def _extract_token_usage(response, model_config: ModelConfig) -> tuple:
    """Extract input and output token counts from API response.

    Returns: (input_tokens, output_tokens)
    """
    input_tokens = 0
    output_tokens = 0

    if model_config.provider == "anthropic":
        # Anthropic response.usage
        if hasattr(response, 'usage') and response.usage:
            input_tokens = getattr(response.usage, 'input_tokens', 0)
            output_tokens = getattr(response.usage, 'output_tokens', 0)
    else:
        # OpenAI-compatible (openai, xai, openrouter)
        if hasattr(response, 'usage') and response.usage:
            input_tokens = getattr(response.usage, 'prompt_tokens', 0)
            output_tokens = getattr(response.usage, 'completion_tokens', 0)

    return input_tokens, output_tokens


# Pricing per million tokens (input, output) in USD
# Updated 2026-01-09
MODEL_PRICING = {
    # Anthropic
    "claude-opus-4-5-20251101": (15.0, 75.0),
    "claude-sonnet-4-5-20250929": (3.0, 15.0),
    # OpenAI
    "gpt-5.2": (10.0, 30.0),
    # Gemini
    "gemini-3-flash-preview": (0.10, 0.40),
    "gemini-3-pro-preview": (1.25, 5.0),
    # xAI
    "grok-4-1-fast-reasoning": (3.0, 15.0),
    # OpenRouter models
    "z-ai/glm-4.7": (0.50, 2.0),
    "deepseek/deepseek-chat": (0.30, 1.20),
    "moonshotai/kimi-k2-thinking": (0.40, 1.75),
    "minimax/minimax-m2.1": (0.50, 2.0),
}


def _estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """Estimate cost in USD based on model pricing."""
    if model_id not in MODEL_PRICING:
        return None

    input_price, output_price = MODEL_PRICING[model_id]
    cost = (input_tokens * input_price / 1_000_000) + (output_tokens * output_price / 1_000_000)
    return round(cost, 6)


def _get_model_response(client, model_config: ModelConfig, system_prompt: str, messages: list):
    """Get response from model."""
    if model_config.provider == "anthropic":
        kwargs = {
            "model": model_config.model_id,
            "max_tokens": 8192,
            "system": system_prompt,
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


def _run_benchmark(sandbox: ModalSandbox, solution_path: str) -> dict:
    """Run benchmark on the solution."""
    # Normalize path
    if not solution_path.startswith("/"):
        solution_path = f"/workspace/{solution_path}"

    # Check solution exists
    if not sandbox.file_exists(solution_path.replace("/workspace/", "")):
        return {"compiled": False, "error": f"Solution not found: {solution_path}"}

    benchmark_script = '''
import torch, time, json, sys, traceback
device = torch.device("cuda:0")
try:
    ref_ns, sol_ns = {}, {}
    exec(open("reference.py").read(), ref_ns)
    exec(open("solution.py").read(), sol_ns)
    RefModel, SolModel = ref_ns["Model"], sol_ns["Model"]
    get_inputs, get_init_inputs = ref_ns["get_inputs"], ref_ns["get_init_inputs"]

    print("Loading models...", flush=True)
    ref_model = RefModel(*get_init_inputs()).to(device).eval()
    sol_model = SolModel(*get_init_inputs()).to(device).eval()
    inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in get_inputs()]

    print("Checking correctness...", flush=True)
    with torch.no_grad():
        ref_out, sol_out = ref_model(*inputs), sol_model(*inputs)
    max_diff = (ref_out.float() - sol_out.float()).abs().max().item() if isinstance(ref_out, torch.Tensor) else 0
    correct = max_diff < 5.0  # Relaxed for bf16
    print(f"max_diff: {max_diff}, correct: {correct}", flush=True)

    if not correct:
        print(json.dumps({"compiled": True, "correct": False, "speedup": None, "error": f"max_diff={max_diff}"}))
        sys.exit(0)

    # Count kernel launches for reference and solution
    def count_kernels(model, inputs):
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.no_grad():
                model(*inputs)
        torch.cuda.synchronize()
        # Count CUDA kernel events
        kernel_count = sum(1 for e in prof.key_averages() if e.device_type == torch.profiler.DeviceType.CUDA)
        return kernel_count

    ref_kernels = count_kernels(ref_model, inputs)
    sol_kernels = count_kernels(sol_model, inputs)
    print(f"Kernel count: ref={ref_kernels}, sol={sol_kernels}", flush=True)

    print("Benchmarking...", flush=True)
    torch.cuda.synchronize()
    for _ in range(3):
        with torch.no_grad(): ref_model(*inputs); sol_model(*inputs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(10):
        with torch.no_grad(): ref_model(*inputs)
    torch.cuda.synchronize()
    ref_time = (time.perf_counter() - start) / 10

    start = time.perf_counter()
    for _ in range(10):
        with torch.no_grad(): sol_model(*inputs)
    torch.cuda.synchronize()
    sol_time = (time.perf_counter() - start) / 10

    print(json.dumps({"compiled": True, "correct": True, "speedup": ref_time/sol_time, "ref_ms": ref_time*1000, "sol_ms": sol_time*1000, "ref_kernels": ref_kernels, "sol_kernels": sol_kernels}))
except Exception as e:
    traceback.print_exc()
    print(json.dumps({"compiled": False, "correct": False, "speedup": None, "error": str(e)}))
'''

    sandbox.write_file("_benchmark.py", benchmark_script)
    result = sandbox.run_command("python _benchmark.py", timeout=600)

    print(f"Benchmark output:\n{result['stdout']}", flush=True)
    if result["stderr"]:
        print(f"Errors:\n{result['stderr']}", flush=True)

    # Parse result
    for line in result["stdout"].split("\n"):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except:
                pass

    return {"compiled": False, "error": "Failed to parse benchmark output"}


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
    parser.add_argument("--gpu", type=str, default="H100", help="GPU type: L40S, A100, H100, B200")
    parser.add_argument("--problem", type=str, help="Problem path (e.g., level4/1_Qwen3-0p6B_bs32_seq256.py)")
    parser.add_argument("--max-turns", type=int, default=20, help="Maximum turns per problem")
    parser.add_argument("--output-dir", type=str, default="outputs/modal_eval", help="Output directory")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for key, cfg in MODELS.items():
            print(f"  {key}: {cfg.name} ({cfg.provider})")
        return

    if not args.model or not args.problem:
        parser.print_help()
        return

    # Load model config
    if args.model not in MODELS:
        print(f"Unknown model: {args.model}")
        print(f"Available: {list(MODELS.keys())}")
        return

    model_config = MODELS[args.model]

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
