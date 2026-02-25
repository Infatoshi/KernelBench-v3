from src.backends import get_backend
from src.prompts.triton_system import (
    get_triton_reasoning_system_prompt,
    get_triton_system_prompt,
)


def test_triton_backend_exists_and_has_expected_attributes() -> None:
    backend = get_backend("triton")
    assert backend.name == "triton"
    assert backend.benchmark_name == "TritonBench"
    assert "H100" in backend.allowed_gpus
    assert callable(backend.find_problems)
    assert callable(backend.validate_solution)


def test_triton_prompt_enforces_triton_constraints() -> None:
    prompt = get_triton_system_prompt("H100", 80)
    assert "@triton.jit" in prompt
    assert "triton.language as tl" in prompt
    assert "load_inline" in prompt
    assert "submit" in prompt

    reasoning_prompt = get_triton_reasoning_system_prompt("H100", 80)
    assert "No CUDA C++" in reasoning_prompt
    assert "Return only complete Python code" in reasoning_prompt


def test_triton_backend_gpu_policy() -> None:
    backend = get_backend("triton")
    assert "RTX3090" in backend.allowed_gpus
    assert "H100" in backend.allowed_gpus
    assert "B200" in backend.allowed_gpus
