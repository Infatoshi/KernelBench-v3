#!/usr/bin/env python3
"""
Triton-based KernelBench Evaluation.

Reuses the existing evaluation pipeline from modal_eval, but swaps system prompts
to request Triton kernels instead of CUDA C++ extensions.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from typing import Iterator

import modal_eval as cuda_eval
from src.config.precision_matrix import (
    HARDWARE_PEAK_TFLOPS,
    HARDWARE_PRECISIONS,
    OP_PRECISION_VALIDITY,
)
from src.prompts.triton_system import (
    get_triton_reasoning_system_prompt,
    get_triton_system_prompt,
)

# Re-export shared data structures and helpers used by batch runners.
ModelConfig = cuda_eval.ModelConfig
EvalResult = cuda_eval.EvalResult
MODELS = cuda_eval.MODELS
GPU_SPECS = cuda_eval.GPU_SPECS
find_problems = cuda_eval.find_problems
get_model_config = cuda_eval.get_model_config


TRITON_FORBIDDEN_PATTERNS = [
    (
        re.compile(r"(?:^|[^\w])torch\.compile\s*\("),
        "Forbidden use of torch.compile in Triton backend",
    ),
    (
        re.compile(r"@torch\.jit\.script"),
        "Forbidden use of torch.jit.script in Triton backend",
    ),
    (
        re.compile(r"(?:^|[^\w])torch\.(?:mm|matmul|conv1d|conv2d|conv3d|linear)\s*\("),
        "Forbidden Python fallback to PyTorch operator in Triton backend",
    ),
    (
        re.compile(r"(?:^|[^\w])F\.(?:linear|conv1d|conv2d|conv3d)\s*\("),
        "Forbidden Python fallback via torch.nn.functional in Triton backend",
    ),
    (
        re.compile(r"(?:^|[^\w])load_inline\s*\("),
        "Forbidden CUDA C++ extension fallback in Triton backend",
    ),
    (
        re.compile(r"torch\.utils\.cpp_extension"),
        "Forbidden CUDA C++ extension import in Triton backend",
    ),
]


def validate_triton_solution_guardrails(solution_code: str) -> str | None:
    """Reject reward-hacking shortcuts for Triton submissions."""
    for pattern, message in TRITON_FORBIDDEN_PATTERNS:
        match = pattern.search(solution_code)
        if match:
            snippet = match.group(0).strip()
            return f"{message}: `{snippet}`"

    if "@triton.jit" not in solution_code:
        return "Missing Triton kernel definition: expected `@triton.jit`"
    if "import triton" not in solution_code:
        return "Missing Triton import: expected `import triton`"
    return None


@contextmanager
def _triton_prompts_enabled() -> Iterator[None]:
    """Temporarily replace modal_eval prompts with Triton prompts."""
    original_system_prompt = cuda_eval.get_system_prompt
    original_reasoning_prompt = cuda_eval.get_reasoning_system_prompt
    cuda_eval.get_system_prompt = get_triton_system_prompt
    cuda_eval.get_reasoning_system_prompt = get_triton_reasoning_system_prompt
    try:
        yield
    finally:
        cuda_eval.get_system_prompt = original_system_prompt
        cuda_eval.get_reasoning_system_prompt = original_reasoning_prompt


def _run_benchmark_triton(sandbox, solution_path: str, **kwargs) -> dict:
    """Run benchmark on Triton solution via module import (not exec)."""
    hardware = kwargs.get("hardware", "UNKNOWN")
    level = kwargs.get("level")

    if not solution_path.startswith("/"):
        solution_path = f"/workspace/{solution_path}"

    if not sandbox.file_exists(solution_path.replace("/workspace/", "")):
        return {"compiled": False, "error": f"Solution not found: {solution_path}"}

    solution_code = sandbox.read_file(solution_path.replace("/workspace/", ""))
    guardrail_error = validate_triton_solution_guardrails(solution_code)
    if guardrail_error:
        return {
            "compiled": False,
            "correct": False,
            "speedup": None,
            "error": guardrail_error,
        }

    benchmark_template = '''
import importlib.util, json, os, statistics, sys, traceback
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
            return [int(a.shape[0]), int(b.shape[1]), int(a.shape[1])]
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

try:
    ref_spec = importlib.util.spec_from_file_location("reference_mod", os.path.abspath("reference.py"))
    ref_mod = importlib.util.module_from_spec(ref_spec)
    ref_spec.loader.exec_module(ref_mod)

    sol_spec = importlib.util.spec_from_file_location("solution_mod", os.path.abspath("solution.py"))
    sol_mod = importlib.util.module_from_spec(sol_spec)
    sol_spec.loader.exec_module(sol_mod)

    RefModel, SolModel = ref_mod.Model, sol_mod.Model
    get_inputs, get_init_inputs = ref_mod.get_inputs, ref_mod.get_init_inputs
    op_type = str(getattr(ref_mod, "OP_TYPE", "unknown")).lower()
    declared_supported_precisions = getattr(ref_mod, "SUPPORTED_PRECISIONS", [])
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
    atol, rtol = 0.05, 0.02
    worst_max_diff, worst_tolerance, worst_seed = 0.0, 0.0, CORRECTNESS_SEEDS[0]

    print("Checking correctness across seeds...", flush=True)
    for seed in CORRECTNESS_SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in get_inputs()]

        with torch.no_grad():
            ref_out, sol_out = ref_model(*inputs), sol_model(*inputs)

        if not isinstance(ref_out, torch.Tensor) or not isinstance(sol_out, torch.Tensor):
            print(json.dumps({"compiled": False, "correct": False, "speedup": None, "error": "Only tensor outputs are supported"}))
            sys.exit(0)
        if ref_out.shape != sol_out.shape:
            print(json.dumps({"compiled": True, "correct": False, "speedup": None, "error": f"shape_mismatch_seed={seed}: {tuple(ref_out.shape)} vs {tuple(sol_out.shape)}"}))
            sys.exit(0)

        ref_f, sol_f = ref_out.float(), sol_out.float()
        max_diff = (ref_f - sol_f).abs().max().item()
        max_ref = ref_f.abs().max().item()
        tolerance = atol + rtol * max_ref
        if max_diff > worst_max_diff:
            worst_max_diff = max_diff
            worst_tolerance = tolerance
            worst_seed = seed
        if max_diff >= tolerance:
            print(json.dumps({"compiled": True, "correct": False, "speedup": None, "error": f"seed={seed}, max_diff={max_diff}"}))
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

    valid_precisions = get_valid_precisions(HARDWARE, op_type)
    if declared_supported_precisions:
        valid_precisions = sorted(set(valid_precisions) & set(declared_supported_precisions))
    precision_supported = precision in valid_precisions if valid_precisions else None
    baseline_type = "cutlass" if precision == "fp4" and HARDWARE == "B200" else "pytorch"
    problem_size = infer_problem_size(op_type, bench_inputs)

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

    def median_runtime_ms(model, model_inputs):
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
        return statistics.median(times_ms), statistics.mean(times_ms)

    ref_ms, ref_mean_ms = median_runtime_ms(ref_model, bench_inputs)
    sol_ms, sol_mean_ms = median_runtime_ms(sol_model, bench_inputs)
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
        "ref_kernels": ref_kernels,
        "sol_kernels": sol_kernels,
        "seeds_tested": len(CORRECTNESS_SEEDS),
        "correctness_seeds": CORRECTNESS_SEEDS,
        "benchmark_seed": benchmark_seed,
        "baseline_type": baseline_type,
        "precision": precision,
        "valid_precisions": valid_precisions,
        "precision_supported": precision_supported,
        "op_type": op_type,
        "problem_size": problem_size,
        "achieved_tflops": achieved_tflops,
        "ref_tflops": ref_tflops,
        "pct_of_peak": pct_of_peak,
        "ref_pct_of_peak": ref_pct_of_peak,
    }))
except Exception as e:
    traceback.print_exc()
    print(json.dumps({"compiled": False, "correct": False, "speedup": None, "error": str(e)}))
'''

    benchmark_script = (
        benchmark_template
        .replace("__HARDWARE__", json.dumps(hardware))
        .replace("__HARDWARE_PRECISIONS__", json.dumps(HARDWARE_PRECISIONS))
        .replace("__OP_PRECISION_VALIDITY__", json.dumps(OP_PRECISION_VALIDITY))
        .replace("__HARDWARE_PEAK_TFLOPS__", json.dumps(HARDWARE_PEAK_TFLOPS))
    )

    sandbox.write_file("_benchmark.py", benchmark_script)
    benchmark_timeout = cuda_eval.MAX_PROBLEM_TIME_SECONDS.get(level or 1, 600) + 120
    result = sandbox.run_command("python _benchmark.py", timeout=benchmark_timeout)

    print(f"Benchmark output:\n{result['stdout']}", flush=True)
    if result["stderr"]:
        print(f"Errors:\n{result['stderr']}", flush=True)

    for line in result["stdout"].split("\n"):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                pass

    return {"compiled": False, "error": "Failed to parse benchmark output"}


@contextmanager
def _triton_eval_overrides() -> Iterator[None]:
    """Enable Triton prompts and Triton benchmark behavior."""
    original_system_prompt = cuda_eval.get_system_prompt
    original_reasoning_prompt = cuda_eval.get_reasoning_system_prompt
    original_benchmark = cuda_eval._run_benchmark

    cuda_eval.get_system_prompt = get_triton_system_prompt
    cuda_eval.get_reasoning_system_prompt = get_triton_reasoning_system_prompt
    cuda_eval._run_benchmark = _run_benchmark_triton
    try:
        yield
    finally:
        cuda_eval.get_system_prompt = original_system_prompt
        cuda_eval.get_reasoning_system_prompt = original_reasoning_prompt
        cuda_eval._run_benchmark = original_benchmark


def run_agent_on_modal(
    model_config: ModelConfig,
    gpu: str,
    problem_code: str,
    problem_name: str,
    level: int,
    max_turns: int = 20,
) -> EvalResult:
    """Run an LLM agent using Triton prompts on existing sandbox paths."""
    with _triton_eval_overrides():
        return cuda_eval.run_agent_on_modal(
            model_config=model_config,
            gpu=gpu,
            problem_code=problem_code,
            problem_name=problem_name,
            level=level,
            max_turns=max_turns,
            backend="triton",
        )


def main() -> None:
    """Run modal_eval CLI with Triton prompts enabled."""
    with _triton_eval_overrides():
        cuda_eval.main()


if __name__ == "__main__":
    main()
