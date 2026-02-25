"""CUDA backend -- default, uses load_inline custom kernels."""

from __future__ import annotations

import json

from src.backends import Backend, register
from src.config.precision_matrix import (
    HARDWARE_PEAK_TFLOPS,
    HARDWARE_PRECISIONS,
    OP_PRECISION_VALIDITY,
)
from src.eval.context import _build_backend_api_reference, _build_template_solution
from src.eval.guardrails import validate_cuda
from src.prompts.cuda_system import get_cuda_reasoning_system_prompt, get_cuda_system_prompt


@register("cuda")
class CudaBackend(Backend):
    name = "cuda"
    benchmark_name = "CUDABench"
    allowed_gpus = ["RTX3090", "H100", "B200"]
    gpu_reason = "CUDABench targets NVIDIA CUDA paths validated for RTX3090, H100, and B200."

    def get_system_prompt(self, gpu_name, vram_gb, use_xml_tools=False):
        return get_cuda_system_prompt(gpu_name, vram_gb, use_xml_tools)

    def get_reasoning_prompt(self, gpu_name, vram_gb):
        return get_cuda_reasoning_system_prompt(gpu_name, vram_gb)

    def find_problems(self, project_root, levels):
        problems = []
        kernelbench_dir = project_root / "problems"
        for level in levels:
            level_dir = kernelbench_dir / f"level{level}"
            if level_dir.exists():
                for problem_file in sorted(level_dir.glob("*.py")):
                    if problem_file.name.startswith("_"):
                        continue
                    problems.append((level, problem_file))
        return problems

    def validate_solution(self, solution_code):
        return validate_cuda(solution_code)

    def build_api_reference(self):
        return _build_backend_api_reference("cuda")

    def build_template_solution(self):
        return _build_template_solution("cuda")

    def run_benchmark(self, sandbox, solution_path, **kwargs):
        return _run_cuda_benchmark(sandbox, solution_path, **kwargs)


CUDA_BENCHMARK_TEMPLATE = '''
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
    if "float8" in text: return "fp8"
    if "bfloat16" in text: return "bf16"
    if "float16" in text: return "fp16"
    if "float32" in text: return "fp32"
    if "float64" in text: return "fp64"
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
    if not problem_size or not time_ms or time_ms <= 0: return None
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
    if achieved_tflops is None: return None
    peak = HARDWARE_PEAK_TFLOPS.get(hardware, {}).get(precision)
    if peak is None or peak <= 0: return None
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
    if has_nan: return False, f"{name} contains NaN", True, bool(has_inf)
    if has_inf: return False, f"{name} contains Inf", bool(has_nan), True
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
            print(json.dumps({"compiled": False, "correct": False, "speedup": None, "error": "Only tensor outputs are supported", "precision_used": precision, "tolerance_atol": tol["atol"], "tolerance_rtol": tol["rtol"], "has_nan": has_nan, "has_inf": has_inf, "is_deterministic": is_deterministic}))
            sys.exit(0)
        if ref_out.shape != sol_out.shape:
            print(json.dumps({"compiled": True, "correct": False, "speedup": None, "error": f"shape_mismatch_seed={seed}: {tuple(ref_out.shape)} vs {tuple(sol_out.shape)}", "precision_used": precision, "tolerance_atol": tol["atol"], "tolerance_rtol": tol["rtol"], "has_nan": has_nan, "has_inf": has_inf, "is_deterministic": is_deterministic}))
            sys.exit(0)

        ref_valid, ref_error, ref_has_nan, ref_has_inf = check_valid_output(ref_out, "reference output")
        sol_valid, sol_error, sol_has_nan, sol_has_inf = check_valid_output(sol_out, "solution output")
        has_nan = has_nan or ref_has_nan or sol_has_nan
        has_inf = has_inf or ref_has_inf or sol_has_inf
        if not ref_valid:
            print(json.dumps({"compiled": True, "correct": False, "speedup": None, "error": ref_error, "precision_used": precision, "tolerance_atol": tol["atol"], "tolerance_rtol": tol["rtol"], "has_nan": has_nan, "has_inf": has_inf, "is_deterministic": is_deterministic}))
            sys.exit(0)
        if not sol_valid:
            print(json.dumps({"compiled": True, "correct": False, "speedup": None, "error": sol_error, "precision_used": precision, "tolerance_atol": tol["atol"], "tolerance_rtol": tol["rtol"], "has_nan": has_nan, "has_inf": has_inf, "is_deterministic": is_deterministic}))
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
            print(json.dumps({"compiled": True, "correct": False, "speedup": None, "error": f"seed={seed}, max_diff={max_diff}", "precision_used": precision, "tolerance_atol": tol["atol"], "tolerance_rtol": tol["rtol"], "has_nan": has_nan, "has_inf": has_inf, "is_deterministic": is_deterministic}))
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
                print(json.dumps({"compiled": True, "correct": False, "speedup": None, "error": "Non-deterministic output (possible race condition)", "precision_used": precision, "tolerance_atol": tol["atol"], "tolerance_rtol": tol["rtol"], "has_nan": has_nan, "has_inf": has_inf, "is_deterministic": is_deterministic}))
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
        return {"median": statistics.median(ordered), "mean": statistics.mean(ordered), "std": statistics.pstdev(ordered), "p10": ordered[p10_idx], "p90": ordered[p90_idx]}

    ref_stats = summarize_runtime_ms(ref_model, bench_inputs)
    sol_stats = summarize_runtime_ms(sol_model, bench_inputs)
    ref_ms, sol_ms = ref_stats["median"], sol_stats["median"]

    ref_tflops = compute_tflops(op_type, problem_size, ref_ms)
    achieved_tflops = compute_tflops(op_type, problem_size, sol_ms)
    ref_pct_of_peak = compute_percent_of_peak(ref_tflops, HARDWARE, precision)
    pct_of_peak = compute_percent_of_peak(achieved_tflops, HARDWARE, precision)

    print(json.dumps({
        "compiled": True, "correct": True, "speedup": ref_ms / sol_ms,
        "ref_ms": ref_ms, "sol_ms": sol_ms,
        "ref_mean_ms": ref_stats["mean"], "sol_mean_ms": sol_stats["mean"],
        "ref_std_ms": ref_stats["std"], "sol_std_ms": sol_stats["std"],
        "ref_p10_ms": ref_stats["p10"], "ref_p90_ms": ref_stats["p90"],
        "sol_p10_ms": sol_stats["p10"], "sol_p90_ms": sol_stats["p90"],
        "ref_kernels": ref_kernels, "sol_kernels": sol_kernels,
        "seeds_tested": len(CORRECTNESS_SEEDS), "correctness_seeds": CORRECTNESS_SEEDS,
        "benchmark_seed": benchmark_seed, "baseline_type": baseline_type,
        "precision": precision, "precision_used": precision,
        "valid_precisions": valid_precisions, "precision_supported": precision_supported,
        "tolerance_atol": tol["atol"], "tolerance_rtol": tol["rtol"],
        "has_nan": has_nan, "has_inf": has_inf, "is_deterministic": is_deterministic,
        "op_type": op_type, "problem_size": problem_size,
        "achieved_tflops": achieved_tflops, "ref_tflops": ref_tflops,
        "pct_of_peak": pct_of_peak, "ref_pct_of_peak": ref_pct_of_peak,
    }))
except Exception as e:
    traceback.print_exc()
    print(json.dumps({"compiled": False, "correct": False, "speedup": None, "error": str(e), "precision_used": None, "tolerance_atol": None, "tolerance_rtol": None, "has_nan": False, "has_inf": False, "is_deterministic": True}))
'''

MAX_PROBLEM_TIME_SECONDS = {1: 300, 2: 600, 3: 900, 4: 1200}


def _run_cuda_benchmark(sandbox, solution_path: str, **kwargs) -> dict:
    hardware = kwargs.get("hardware", "UNKNOWN")
    level = kwargs.get("level")

    if not solution_path.startswith("/"):
        solution_path = f"/workspace/{solution_path}"
    if not sandbox.file_exists(solution_path.replace("/workspace/", "")):
        return {"compiled": False, "error": f"Solution not found: {solution_path}"}

    solution_code = sandbox.read_file(solution_path.replace("/workspace/", ""))
    guardrail_error = validate_cuda(solution_code)
    if guardrail_error:
        return {"compiled": False, "correct": False, "speedup": None, "error": guardrail_error}

    benchmark_script = (
        CUDA_BENCHMARK_TEMPLATE
        .replace("__HARDWARE__", json.dumps(hardware))
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
