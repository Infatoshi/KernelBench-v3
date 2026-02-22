#!/usr/bin/env python3
"""
Modal-based GPU benchmarking for KernelBench.

Runs benchmarks on remote GPUs (A100, H100, L40S, B200) via Modal.
The agent evaluation runs locally, but final benchmarking happens on Modal.

Usage:
    # Import and use in evaluate_agent.py
    from modal_benchmark import benchmark_on_gpu
    result = benchmark_on_gpu.remote("A100", solution_code, reference_code)
"""

import modal

# Define the Modal app
app = modal.App("kernelbench-benchmark")

# Base image with CUDA 13.1, PyTorch, and dependencies
# CUDA 13.1 provides full Blackwell support and cuTile DSL
base_image = (
    modal.Image.from_registry("nvidia/cuda:13.1.0-devel-ubuntu24.04", add_python="3.11")
    .apt_install("ninja-build", "build-essential")
    .pip_install(
        "torch",
        "numpy",
        "pydantic",
        "transformers",
        "accelerate",
        "triton",
    )
    .env({"TORCH_USE_CUDA_DSA": "1"})
)

# GPU-specific functions
# We create separate functions for each GPU type to enable parallel execution


@app.function(gpu="A100", image=base_image, timeout=600)
def benchmark_a100(solution_code: str, reference_code: str, num_perf_trials: int = 10) -> dict:
    """Benchmark on A100 GPU."""
    return _run_benchmark(solution_code, reference_code, num_perf_trials)


@app.function(gpu="H100", image=base_image, timeout=600)
def benchmark_h100(solution_code: str, reference_code: str, num_perf_trials: int = 10) -> dict:
    """Benchmark on H100 GPU."""
    return _run_benchmark(solution_code, reference_code, num_perf_trials)


@app.function(gpu="L40S", image=base_image, timeout=600)
def benchmark_l40s(solution_code: str, reference_code: str, num_perf_trials: int = 10) -> dict:
    """Benchmark on L40S GPU."""
    return _run_benchmark(solution_code, reference_code, num_perf_trials)


@app.function(gpu="B200", image=base_image, timeout=600)
def benchmark_b200(solution_code: str, reference_code: str, num_perf_trials: int = 10) -> dict:
    """Benchmark on B200 GPU."""
    return _run_benchmark(solution_code, reference_code, num_perf_trials)


def _run_benchmark(solution_code: str, reference_code: str, num_perf_trials: int = 10) -> dict:
    """
    Run benchmark on current GPU.

    This is the core benchmark logic that runs inside the Modal container.
    It's essentially a simplified version of eval_kernel_against_ref.

    Correctness is tested across multiple random seeds to prevent caching exploits.
    """
    import re
    import statistics
    import numpy as np
    import torch
    import traceback

    # Multiple seeds to prevent hardcoding/caching for a single seed
    CORRECTNESS_SEEDS = [42, 123, 456, 789, 1337]
    # Stabilize timing across runs while preserving caller override for larger sweeps.
    num_perf_trials = max(num_perf_trials, 30)

    forbidden_solution_patterns = [
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

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    result = {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "cuda_version": torch.version.cuda,
        "compiled": False,
        "correctness": False,
        "speedup": None,
        "baseline_ms": None,
        "solution_ms": None,
        "error": None,
    }

    context = {}

    try:
        # Guardrail check before executing any submitted code.
        for pattern, message in forbidden_solution_patterns:
            match = pattern.search(solution_code)
            if match:
                result["error"] = f"{message}: `{match.group(0).strip()}`"
                return result

        # Load reference model
        exec(reference_code, context)
        Model = context.get("Model")
        get_init_inputs = context.get("get_init_inputs")
        get_inputs = context.get("get_inputs")

        if not all([Model, get_init_inputs, get_inputs]):
            result["error"] = "Reference code missing Model, get_init_inputs, or get_inputs"
            return result

        # Initialize reference model
        torch.manual_seed(42)
        init_inputs = get_init_inputs()
        init_inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in init_inputs]

        with torch.no_grad():
            torch.manual_seed(42)
            original_model = Model(*init_inputs).cuda()

        # Load solution model
        solution_context = {}
        exec(solution_code, solution_context)
        ModelNew = solution_context.get("ModelNew")

        if ModelNew is None:
            result["error"] = "Solution code missing ModelNew class"
            return result

        result["compiled"] = True

        # Initialize solution model
        with torch.no_grad():
            torch.manual_seed(42)
            custom_model = ModelNew(*init_inputs).cuda()

        # Check correctness across multiple seeds to prevent caching exploits
        atol, rtol = 0.05, 0.02
        for seed in CORRECTNESS_SEEDS:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            inputs = get_inputs()
            inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]

            with torch.no_grad():
                output_ref = original_model(*inputs)
                torch.cuda.synchronize()

                output_new = custom_model(*inputs)
                torch.cuda.synchronize()

            if output_ref.shape != output_new.shape:
                result["error"] = f"Shape mismatch at seed={seed}: {output_ref.shape} vs {output_new.shape}"
                return result

            output_ref_f = output_ref.float()
            output_new_f = output_new.float()
            max_diff = torch.max(torch.abs(output_ref_f - output_new_f)).item()
            tolerance = atol + rtol * output_ref_f.abs().max().item()
            if max_diff >= tolerance:
                max_diff = torch.max(torch.abs(output_ref - output_new)).item()
                result["error"] = f"Output mismatch at seed={seed}: max_diff={max_diff:.6f}"
                return result

        result["correctness"] = True
        result["seeds_tested"] = len(CORRECTNESS_SEEDS)

        # Deterministic benchmark inputs after correctness passes.
        torch.manual_seed(2026)
        torch.cuda.manual_seed_all(2026)
        inputs = get_inputs()
        inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]

        # Benchmark performance
        num_warmup = 5

        # Warmup reference
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = original_model(*inputs)
        torch.cuda.synchronize()

        # Time reference
        ref_times = []
        for _ in range(num_perf_trials):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                _ = original_model(*inputs)
            end.record()
            torch.cuda.synchronize()
            ref_times.append(start.elapsed_time(end))

        # Warmup solution
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = custom_model(*inputs)
        torch.cuda.synchronize()

        # Time solution
        sol_times = []
        for _ in range(num_perf_trials):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                _ = custom_model(*inputs)
            end.record()
            torch.cuda.synchronize()
            sol_times.append(start.elapsed_time(end))

        result["baseline_ms"] = float(statistics.median(ref_times))
        result["solution_ms"] = float(statistics.median(sol_times))
        result["baseline_mean_ms"] = float(np.mean(ref_times))
        result["solution_mean_ms"] = float(np.mean(sol_times))
        result["baseline_std_ms"] = float(np.std(ref_times))
        result["solution_std_ms"] = float(np.std(sol_times))

        if result["solution_ms"] > 0:
            result["speedup"] = round(result["baseline_ms"] / result["solution_ms"], 3)

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:500]}"
        result["traceback"] = traceback.format_exc()[:1000]

    return result


# Dispatcher function to call the right GPU benchmark
GPU_FUNCTIONS = {
    "A100": benchmark_a100,
    "H100": benchmark_h100,
    "L40S": benchmark_l40s,
    "B200": benchmark_b200,
}


def benchmark_on_gpus(
    solution_code: str,
    reference_code: str,
    gpus: list[str] = None,
    num_perf_trials: int = 10,
) -> dict[str, dict]:
    """
    Benchmark solution on multiple GPUs in parallel.

    Args:
        solution_code: The optimized solution code
        reference_code: The reference implementation
        gpus: List of GPU types to benchmark on (default: all available)
        num_perf_trials: Number of performance trials per GPU

    Returns:
        Dict mapping GPU name to benchmark results
    """
    if gpus is None:
        gpus = ["A100", "H100", "L40S", "B200"]

    results = {}

    # Launch all benchmarks in parallel using Modal's .spawn()
    handles = {}
    for gpu in gpus:
        if gpu in GPU_FUNCTIONS:
            fn = GPU_FUNCTIONS[gpu]
            handle = fn.spawn(solution_code, reference_code, num_perf_trials)
            handles[gpu] = handle

    # Collect results
    for gpu, handle in handles.items():
        try:
            results[gpu] = handle.get()
        except Exception as e:
            results[gpu] = {
                "error": f"Modal execution failed: {str(e)}",
                "gpu_name": gpu,
            }

    return results


@app.local_entrypoint()
def main():
    """Test the benchmark system with a simple example."""

    reference_code = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear = nn.Linear(size, size)

    def forward(self, x):
        return self.linear(x)

def get_init_inputs():
    return [512]

def get_inputs():
    return [torch.randn(32, 512)]
'''

    solution_code = '''
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear = nn.Linear(size, size)

    def forward(self, x):
        return self.linear(x)
'''

    print("Testing benchmark on A100...")
    result = benchmark_a100.remote(solution_code, reference_code, num_perf_trials=5)
    print(f"A100 Result: {result}")

    print("\nTesting parallel benchmark on all GPUs...")
    results = benchmark_on_gpus(solution_code, reference_code, num_perf_trials=5)
    for gpu, res in results.items():
        print(f"\n{gpu}:")
        print(f"  GPU: {res.get('gpu_name', 'N/A')}")
        print(f"  Compiled: {res.get('compiled', False)}")
        print(f"  Correct: {res.get('correctness', False)}")
        print(f"  Speedup: {res.get('speedup', 'N/A')}x")
        if res.get('error'):
            print(f"  Error: {res['error']}")
