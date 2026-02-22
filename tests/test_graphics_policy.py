from pathlib import Path

import graphics_batch_eval
from run_all_benchmarks import BENCHMARK_CONFIG
from src.config.benchmark_problems import BENCHMARK_PROBLEMS, get_problem_hardware_required


def test_graphics_gpu_policy_is_rtx3090_only() -> None:
    assert graphics_batch_eval.ALLOWED_GPUS == ["RTX3090"]
    assert BENCHMARK_PROBLEMS["graphics"]["hardware"] == ["RTX3090"]
    assert BENCHMARK_CONFIG["graphics"]["gpus"] == ["RTX3090"]


def test_graphics_problem_hardware_metadata_is_rtx3090_only() -> None:
    bloom = Path("KernelBench/graphics/bloom.py")
    particles = Path("KernelBench/graphics/particles.py")
    assert get_problem_hardware_required(bloom) == ["RTX3090"]
    assert get_problem_hardware_required(particles) == ["RTX3090"]
