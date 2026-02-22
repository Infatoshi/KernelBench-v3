#!/usr/bin/env python3
"""
Metal-based KernelBench Evaluation.

Reuses the existing evaluation pipeline from modal_eval, but swaps prompts and
execution behavior for Metal/MLX on Apple Silicon.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from typing import Iterator

import modal_eval as cuda_eval
from src.agent.metal_sandbox import MetalSandbox, MetalSandboxConfig
from src.prompts.metal_system import (
    get_metal_reasoning_system_prompt,
    get_metal_system_prompt,
)

# Re-export shared data structures and helpers used by batch runners.
ModelConfig = cuda_eval.ModelConfig
EvalResult = cuda_eval.EvalResult
MODELS = cuda_eval.MODELS
find_problems = cuda_eval.find_problems
get_model_config = cuda_eval.get_model_config

GPU_SPECS = dict(cuda_eval.GPU_SPECS)
GPU_SPECS["M4MAX"] = ("M4 Max", 36)


METAL_FORBIDDEN_PATTERNS = [
    (
        re.compile(r"(?:^|[^\w])import\s+torch\b"),
        "Forbidden import: torch is not allowed in Metal backend",
    ),
    (
        re.compile(r"(?:^|[^\w])torch\."),
        "Forbidden PyTorch usage in Metal backend",
    ),
    (
        re.compile(r"(?:^|[^\w])import\s+triton\b"),
        "Forbidden import: triton is not allowed in Metal backend",
    ),
    (
        re.compile(r"(?:^|[^\w])triton\."),
        "Forbidden Triton usage in Metal backend",
    ),
    (
        re.compile(r"torch\.utils\.cpp_extension|(?:^|[^\w])load_inline\s*\("),
        "Forbidden CUDA extension usage in Metal backend",
    ),
]


def validate_metal_solution_guardrails(solution_code: str) -> str | None:
    """Reject reward-hacking shortcuts for Metal submissions."""
    for pattern, message in METAL_FORBIDDEN_PATTERNS:
        match = pattern.search(solution_code)
        if match:
            snippet = match.group(0).strip()
            return f"{message}: `{snippet}`"

    if "import mlx.core as mx" not in solution_code:
        return "Missing MLX import: expected `import mlx.core as mx`"
    if "def solution(" not in solution_code:
        return "Missing required benchmark interface: `def solution(a, b)`"
    return None


def _run_benchmark_metal(sandbox, solution_path: str, **_kwargs) -> dict:
    """Run benchmark on MLX/Metal device."""
    if not solution_path.startswith("/"):
        solution_path = f"/workspace/{solution_path}"

    if not sandbox.file_exists(solution_path.replace("/workspace/", "")):
        return {"compiled": False, "error": f"Solution not found: {solution_path}"}

    solution_code = sandbox.read_file(solution_path.replace("/workspace/", ""))
    guardrail_error = validate_metal_solution_guardrails(solution_code)
    if guardrail_error:
        return {
            "compiled": False,
            "correct": False,
            "speedup": None,
            "error": guardrail_error,
        }

    benchmark_script = '''
import importlib.util
import json
import os
import statistics
import sys
import time
import traceback

import mlx.core as mx
import numpy as np


def _one_output(value):
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("solution() returned an empty list/tuple")
        return value[0]
    return value


def _timed_stats_ms(fn, iters):
    times_ms = []
    for _ in range(iters):
        start = time.perf_counter()
        out = _one_output(fn())
        mx.eval(out)
        times_ms.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times_ms), statistics.mean(times_ms)


def _seeded_input_pair(n, seed):
    rng = np.random.default_rng(seed)
    a = mx.array(rng.standard_normal((n, n)).astype(np.float32))
    b = mx.array(rng.standard_normal((n, n)).astype(np.float32))
    return a, b


try:
    sol_spec = importlib.util.spec_from_file_location("solution_mod", os.path.abspath("solution.py"))
    sol_mod = importlib.util.module_from_spec(sol_spec)
    sol_spec.loader.exec_module(sol_mod)

    if not hasattr(sol_mod, "solution"):
        print(json.dumps({"compiled": False, "correct": False, "speedup": None, "error": "solution.py must define solution(a, b)"}))
        sys.exit(0)

    candidate = sol_mod.solution

    print("Preparing MLX inputs...", flush=True)
    n = 1024
    correctness_seeds = [42, 123, 456, 789, 1337]
    atol, rtol = 0.05, 0.02
    worst_max_diff = 0.0
    worst_tolerance = 0.0
    worst_seed = correctness_seeds[0]

    print("Checking correctness across seeds...", flush=True)
    for seed in correctness_seeds:
        a, b = _seeded_input_pair(n, seed)
        ref_out = mx.matmul(a, b)
        sol_out = _one_output(candidate(a, b))
        mx.eval(ref_out, sol_out)

        if not hasattr(sol_out, "shape"):
            print(json.dumps({"compiled": False, "correct": False, "speedup": None, "error": "solution() must return an array-like object with shape"}))
            sys.exit(0)
        if ref_out.shape != sol_out.shape:
            print(json.dumps({"compiled": True, "correct": False, "speedup": None, "error": f"shape_mismatch_seed={seed}: {ref_out.shape} vs {sol_out.shape}"}))
            sys.exit(0)

        max_diff = float(mx.max(mx.abs(ref_out - sol_out)).item())
        max_ref = float(mx.max(mx.abs(ref_out)).item())
        tolerance = atol + rtol * max_ref
        if max_diff > worst_max_diff:
            worst_max_diff = max_diff
            worst_tolerance = tolerance
            worst_seed = seed
        if max_diff >= tolerance:
            print(json.dumps({"compiled": True, "correct": False, "speedup": None, "error": f"seed={seed}, max_diff={max_diff}"}))
            sys.exit(0)

    print(f"worst_seed: {worst_seed}, max_diff: {worst_max_diff:.6f}, tolerance: {worst_tolerance:.6f}", flush=True)

    a_bench, b_bench = _seeded_input_pair(n, 2026)

    print("Benchmarking...", flush=True)
    WARMUP_ITERS = 5
    TIMED_ITERS = 30
    for _ in range(WARMUP_ITERS):
        mx.eval(mx.matmul(a_bench, b_bench))
        mx.eval(_one_output(candidate(a_bench, b_bench)))

    ref_ms, ref_mean_ms = _timed_stats_ms(lambda: mx.matmul(a_bench, b_bench), iters=TIMED_ITERS)
    sol_ms, sol_mean_ms = _timed_stats_ms(lambda: candidate(a_bench, b_bench), iters=TIMED_ITERS)

    print(json.dumps({
        "compiled": True,
        "correct": True,
        "speedup": ref_ms / sol_ms,
        "ref_ms": ref_ms,
        "sol_ms": sol_ms,
        "ref_mean_ms": ref_mean_ms,
        "sol_mean_ms": sol_mean_ms,
        "seeds_tested": len(correctness_seeds),
        "ref_kernels": None,
        "sol_kernels": None,
    }))
except Exception as e:
    traceback.print_exc()
    print(json.dumps({"compiled": False, "correct": False, "speedup": None, "error": str(e)}))
'''

    sandbox.write_file("_benchmark.py", benchmark_script)
    result = sandbox.run_command("python _benchmark.py", timeout=900)

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


@contextmanager
def _metal_eval_overrides() -> Iterator[None]:
    """Enable Metal prompts, sandbox routing, and benchmark behavior."""
    original_system_prompt = cuda_eval.get_system_prompt
    original_reasoning_prompt = cuda_eval.get_reasoning_system_prompt
    original_benchmark = cuda_eval._run_benchmark
    original_gpu_specs = cuda_eval.GPU_SPECS
    original_local_gpus = cuda_eval.LOCAL_GPUS
    original_local_sandbox = cuda_eval.LocalSandbox
    original_local_config = cuda_eval.LocalSandboxConfig

    cuda_eval.get_system_prompt = get_metal_system_prompt
    cuda_eval.get_reasoning_system_prompt = get_metal_reasoning_system_prompt
    cuda_eval._run_benchmark = _run_benchmark_metal
    cuda_eval.GPU_SPECS = GPU_SPECS
    cuda_eval.LOCAL_GPUS = {"M4MAX"}
    cuda_eval.LocalSandbox = MetalSandbox
    cuda_eval.LocalSandboxConfig = MetalSandboxConfig
    try:
        yield
    finally:
        cuda_eval.get_system_prompt = original_system_prompt
        cuda_eval.get_reasoning_system_prompt = original_reasoning_prompt
        cuda_eval._run_benchmark = original_benchmark
        cuda_eval.GPU_SPECS = original_gpu_specs
        cuda_eval.LOCAL_GPUS = original_local_gpus
        cuda_eval.LocalSandbox = original_local_sandbox
        cuda_eval.LocalSandboxConfig = original_local_config


def run_agent_on_modal(
    model_config: ModelConfig,
    gpu: str,
    problem_code: str,
    problem_name: str,
    level: int,
    max_turns: int = 20,
) -> EvalResult:
    """Run an LLM agent using Metal prompts and Metal sandbox for M4MAX."""
    with _metal_eval_overrides():
        return cuda_eval.run_agent_on_modal(
            model_config=model_config,
            gpu=gpu,
            problem_code=problem_code,
            problem_name=problem_name,
            level=level,
            max_turns=max_turns,
            backend="metal",
        )


def main() -> None:
    """Run modal_eval CLI with Metal overrides enabled."""
    with _metal_eval_overrides():
        cuda_eval.main()


if __name__ == "__main__":
    main()
