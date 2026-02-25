"""Metal backend -- MLX custom kernels for Apple Silicon."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from src.backends import Backend, register
from src.config.precision_matrix import (
    HARDWARE_PEAK_TFLOPS,
    HARDWARE_PRECISIONS,
    OP_PRECISION_VALIDITY,
)
from src.eval.context import _build_backend_api_reference, _build_template_solution
from src.eval.guardrails import validate_metal
from src.prompts.metal_system import get_metal_reasoning_system_prompt, get_metal_system_prompt

_METAL_EXCLUDE = {
    "4_FP8_Matmul.py",
    "6_INT4_Quantized_GEMM.py",
    "7_GatedDeltaNet.py",
    "8_KimiDeltaAttention.py",
}


@register("metal")
class MetalBackend(Backend):
    name = "metal"
    benchmark_name = "MetalBench"
    allowed_gpus = ["M4MAX"]
    gpu_reason = "MetalBench targets Apple Silicon via MLX on M4 Max."

    @property
    def gpu_specs(self) -> dict:
        specs = super().gpu_specs
        specs["M4MAX"] = ("M4 Max", 36)
        return specs

    @property
    def local_gpus(self) -> set:
        return {"M4MAX"}

    def get_system_prompt(self, gpu_name, vram_gb, use_xml_tools=False):
        return get_metal_system_prompt(gpu_name, vram_gb, use_xml_tools)

    def get_reasoning_prompt(self, gpu_name, vram_gb):
        return get_metal_reasoning_system_prompt(gpu_name, vram_gb)

    def find_problems(self, project_root, levels):
        problems: List[Tuple[int, Path]] = []
        kernelbench_dir = project_root / "problems"
        for level in levels:
            level_dir = kernelbench_dir / f"level{level}"
            if level_dir.exists():
                for problem_file in sorted(level_dir.glob("*.py")):
                    if problem_file.name.startswith("_"):
                        continue
                    problems.append((level, problem_file))
            metal_dir = kernelbench_dir / f"metal_level{level}"
            if metal_dir.exists():
                for problem_file in sorted(metal_dir.glob("*.py")):
                    if problem_file.name.startswith("_"):
                        continue
                    problems.append((level, problem_file))
        problems = [(lvl, p) for lvl, p in problems if p.name not in _METAL_EXCLUDE]
        return problems

    def validate_solution(self, solution_code):
        return validate_metal(solution_code)

    def build_api_reference(self):
        return _build_backend_api_reference("metal")

    def build_template_solution(self):
        return _build_template_solution("metal")

    def create_sandbox(self, gpu: str, problem_code: str):
        from src.agent.metal_sandbox import MetalSandbox, MetalSandboxConfig
        return MetalSandbox(problem_code, MetalSandboxConfig())

    def run_benchmark(self, sandbox, solution_path, **kwargs):
        return _run_metal_benchmark(sandbox, solution_path, **kwargs)


METAL_BENCHMARK_TEMPLATE = '''
import importlib.util
import json
import statistics
import sys
import time
import traceback

import mlx.core as mx
import numpy as np
import torch

HARDWARE = __HARDWARE__
HARDWARE_PRECISIONS = __HARDWARE_PRECISIONS__
OP_PRECISION_VALIDITY = __OP_PRECISION_VALIDITY__
HARDWARE_PEAK_TFLOPS = __HARDWARE_PEAK_TFLOPS__


def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def one_output(value):
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("solution() returned an empty list/tuple")
        return value[0]
    return value


def to_torch_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, (list, tuple)):
        out = [to_torch_device(v, device) for v in value]
        return type(value)(out)
    return value


def to_mx(value):
    if isinstance(value, torch.Tensor):
        return mx.array(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return mx.array(value)
    if isinstance(value, (list, tuple)):
        out = [to_mx(v) for v in value]
        return type(value)(out)
    return value


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.array(value)


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


def torch_sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


PRECISION_TOLERANCES = {
    "fp4": {"atol": 0.5, "rtol": 0.1},
    "fp8": {"atol": 0.1, "rtol": 0.05},
    "fp16": {"atol": 0.01, "rtol": 0.01},
    "bf16": {"atol": 0.01, "rtol": 0.01},
    "fp32": {"atol": 0.001, "rtol": 0.001},
}


def get_tolerance(precision):
    return PRECISION_TOLERANCES.get(precision, {"atol": 0.05, "rtol": 0.02})


def check_valid_output(array_np, name="output"):
    has_nan = bool(np.isnan(array_np).any())
    has_inf = bool(np.isinf(array_np).any())
    if has_nan:
        return False, f"{name} contains NaN", has_nan, has_inf
    if has_inf:
        return False, f"{name} contains Inf", has_nan, has_inf
    return True, "", has_nan, has_inf


try:
    reference_module = load_module("kb_reference", "reference.py")
    solution_module = load_module("kb_solution", "solution.py")

    if not hasattr(solution_module, "solution"):
        print(json.dumps({"compiled": False, "correct": False, "speedup": None, "error": "solution.py must define solution(*inputs)"}))
        sys.exit(0)

    RefModel = reference_module.Model
    get_inputs = reference_module.get_inputs
    get_init_inputs = reference_module.get_init_inputs
    candidate = solution_module.solution

    op_type = str(getattr(reference_module, "OP_TYPE", "unknown")).lower()
    declared_supported_precisions = getattr(reference_module, "SUPPORTED_PRECISIONS", [])
    if not isinstance(declared_supported_precisions, (list, tuple)):
        declared_supported_precisions = []
    declared_supported_precisions = [str(p).lower() for p in declared_supported_precisions]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ref_model = RefModel(*get_init_inputs()).to(device).eval()

    CORRECTNESS_SEEDS = [42, 123, 456, 789, 1337]
    benchmark_seed = 2026
    worst_max_diff = 0.0
    worst_tolerance = 0.0
    worst_seed = CORRECTNESS_SEEDS[0]
    precision = "fp32"
    tol = get_tolerance(precision)
    has_nan = False
    has_inf = False
    is_deterministic = True

    for seed in CORRECTNESS_SEEDS:
        torch.manual_seed(seed)
        raw_inputs = get_inputs()
        torch_inputs = [to_torch_device(x, device) for x in raw_inputs]
        mx_inputs = [to_mx(x) for x in raw_inputs]

        for value in torch_inputs:
            if isinstance(value, torch.Tensor):
                precision = dtype_to_precision(value.dtype)
                break
        tol = get_tolerance(precision)

        with torch.no_grad():
            ref_out = one_output(ref_model(*torch_inputs))
        torch_sync(device)

        try:
            sol_out = one_output(candidate(*mx_inputs))
        except TypeError as exc:
            print(json.dumps({
                "compiled": False,
                "correct": False,
                "speedup": None,
                "error": f"signature_mismatch_expected_{len(mx_inputs)}_inputs: {exc}",
                "precision_used": precision,
                "tolerance_atol": tol["atol"],
                "tolerance_rtol": tol["rtol"],
                "has_nan": has_nan,
                "has_inf": has_inf,
                "is_deterministic": is_deterministic,
            }))
            sys.exit(0)

        mx.eval(sol_out)

        ref_np = to_numpy(ref_out)
        sol_np = to_numpy(sol_out)

        if ref_np.shape != sol_np.shape:
            print(json.dumps({
                "compiled": True,
                "correct": False,
                "speedup": None,
                "error": f"shape_mismatch_seed={seed}: {tuple(ref_np.shape)} vs {tuple(sol_np.shape)}",
                "precision_used": precision,
                "tolerance_atol": tol["atol"],
                "tolerance_rtol": tol["rtol"],
                "has_nan": has_nan,
                "has_inf": has_inf,
                "is_deterministic": is_deterministic,
            }))
            sys.exit(0)

        ref_valid, ref_error, ref_nan, ref_inf = check_valid_output(ref_np, "reference output")
        sol_valid, sol_error, sol_nan, sol_inf = check_valid_output(sol_np, "solution output")
        has_nan = has_nan or ref_nan or sol_nan
        has_inf = has_inf or ref_inf or sol_inf

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

        ref_f = ref_np.astype(np.float32)
        sol_f = sol_np.astype(np.float32)
        max_diff = float(np.max(np.abs(ref_f - sol_f)))
        max_ref = float(np.max(np.abs(ref_f)))
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

        if sol_f.size > 1:
            sol_std = float(np.std(sol_f))
            ref_std_val = float(np.std(ref_f))
            if ref_std_val > 1e-6 and sol_std < 1e-6:
                print(json.dumps({
                    "compiled": True,
                    "correct": False,
                    "speedup": None,
                    "error": f"constant_output_hack_seed={seed}: sol_std={sol_std:.2e} vs ref_std={ref_std_val:.2e}",
                    "precision_used": precision,
                    "tolerance_atol": tol["atol"],
                    "tolerance_rtol": tol["rtol"],
                    "has_nan": has_nan,
                    "has_inf": has_inf,
                    "is_deterministic": is_deterministic,
                }))
                sys.exit(0)
            if ref_std_val > 1e-6:
                ref_flat = ref_f.flatten()
                sol_flat = sol_f.flatten()
                dot = float(np.sum(ref_flat * sol_flat))
                ref_norm = float(np.sqrt(np.sum(ref_flat ** 2)))
                sol_norm = float(np.sqrt(np.sum(sol_flat ** 2)))
                cos = dot / (ref_norm * sol_norm + 1e-10)
                if cos < 0.95:
                    print(json.dumps({
                        "compiled": True,
                        "correct": False,
                        "speedup": None,
                        "error": f"low_cosine_similarity_seed={seed}: cos={cos:.4f}",
                        "precision_used": precision,
                        "tolerance_atol": tol["atol"],
                        "tolerance_rtol": tol["rtol"],
                        "has_nan": has_nan,
                        "has_inf": has_inf,
                        "is_deterministic": is_deterministic,
                    }))
                    sys.exit(0)

    torch.manual_seed(benchmark_seed)
    raw_inputs = get_inputs()
    torch_inputs = [to_torch_device(x, device) for x in raw_inputs]
    mx_inputs = [to_mx(x) for x in raw_inputs]

    if REPEATABILITY_CHECK := True:
        REPEATABILITY_RUNS = 2
        repeat_outputs = []
        for _ in range(REPEATABILITY_RUNS):
            out = one_output(candidate(*mx_inputs))
            mx.eval(out)
            repeat_outputs.append(to_numpy(out).copy())
        for idx in range(1, len(repeat_outputs)):
            if not np.array_equal(repeat_outputs[0], repeat_outputs[idx]):
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

    WARMUP_ITERS = 5
    TIMED_ITERS = 30

    def timed_ref_ms(iters):
        times = []
        for _ in range(iters):
            start = time.perf_counter()
            with torch.no_grad():
                _ = one_output(ref_model(*torch_inputs))
            torch_sync(device)
            times.append((time.perf_counter() - start) * 1000.0)
        return times

    def timed_sol_ms(iters):
        times = []
        for _ in range(iters):
            start = time.perf_counter()
            out = one_output(candidate(*mx_inputs))
            mx.eval(out)
            times.append((time.perf_counter() - start) * 1000.0)
        return times

    for _ in range(WARMUP_ITERS):
        with torch.no_grad():
            _ = one_output(ref_model(*torch_inputs))
        torch_sync(device)
        out = one_output(candidate(*mx_inputs))
        mx.eval(out)

    ref_times = sorted(timed_ref_ms(TIMED_ITERS))
    sol_times = sorted(timed_sol_ms(TIMED_ITERS))

    n = len(ref_times)
    p10_idx = int(0.10 * (n - 1))
    p90_idx = int(0.90 * (n - 1))

    ref_ms = float(statistics.median(ref_times))
    sol_ms = float(statistics.median(sol_times))
    ref_mean_ms = float(statistics.mean(ref_times))
    sol_mean_ms = float(statistics.mean(sol_times))
    ref_std_ms = float(statistics.pstdev(ref_times))
    sol_std_ms = float(statistics.pstdev(sol_times))
    ref_p10_ms = float(ref_times[p10_idx])
    ref_p90_ms = float(ref_times[p90_idx])
    sol_p10_ms = float(sol_times[p10_idx])
    sol_p90_ms = float(sol_times[p90_idx])

    valid_precisions = get_valid_precisions(HARDWARE, op_type)
    if declared_supported_precisions:
        valid_precisions = sorted(set(valid_precisions) & set(declared_supported_precisions))
    precision_supported = precision in valid_precisions if valid_precisions else None
    baseline_type = "pytorch"

    problem_size = infer_problem_size(op_type, torch_inputs)
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
        "ref_kernels": None,
        "sol_kernels": None,
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
except Exception as exc:
    traceback.print_exc()
    print(json.dumps({
        "compiled": False,
        "correct": False,
        "speedup": None,
        "error": str(exc),
        "precision_used": None,
        "tolerance_atol": None,
        "tolerance_rtol": None,
        "has_nan": False,
        "has_inf": False,
        "is_deterministic": True,
    }))
'''

MAX_PROBLEM_TIME_SECONDS = {1: 300, 2: 600, 3: 900, 4: 1200}


def _run_metal_benchmark(sandbox, solution_path: str, **kwargs) -> dict:
    hardware = kwargs.get("hardware", "UNKNOWN")
    level = kwargs.get("level")

    if not solution_path.startswith("/"):
        solution_path = f"/workspace/{solution_path}"

    if not sandbox.file_exists(solution_path.replace("/workspace/", "")):
        return {"compiled": False, "error": f"Solution not found: {solution_path}"}

    solution_code = sandbox.read_file(solution_path.replace("/workspace/", ""))
    guardrail_error = validate_metal(solution_code)
    if guardrail_error:
        return {"compiled": False, "correct": False, "speedup": None, "error": guardrail_error}

    benchmark_script = (
        METAL_BENCHMARK_TEMPLATE
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
