from pathlib import Path

from src.backends import get_backend
from src.config.benchmark_problems import BENCHMARK_PROBLEMS, get_problem_hardware_required
from tools.run_all import BENCHMARK_CONFIG


def test_graphics_gpu_policy_is_rtx3090_only() -> None:
    backend = get_backend("graphics")
    assert backend.allowed_gpus == ["RTX3090"]
    assert BENCHMARK_PROBLEMS["graphics"]["hardware"] == ["RTX3090"]
    assert BENCHMARK_CONFIG["graphics"]["gpus"] == ["RTX3090"]


def test_graphics_problem_hardware_metadata_is_rtx3090_only() -> None:
    bloom = Path("problems/graphics/bloom.py")
    particles = Path("problems/graphics/particles.py")
    assert get_problem_hardware_required(bloom) == ["RTX3090"]
    assert get_problem_hardware_required(particles) == ["RTX3090"]
