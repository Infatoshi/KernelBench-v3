from src.agent.metal_sandbox import MetalSandbox, MetalSandboxConfig
from src.backends import get_backend
from src.prompts.metal_system import (
    get_metal_reasoning_system_prompt,
    get_metal_system_prompt,
)


def test_metal_backend_exists_and_has_expected_attributes() -> None:
    backend = get_backend("metal")
    assert backend.name == "metal"
    assert backend.benchmark_name == "MetalBench"
    assert "M4MAX" in backend.allowed_gpus
    assert "M4MAX" in backend.gpu_specs
    assert callable(backend.find_problems)
    assert callable(backend.validate_solution)


def test_metal_prompt_enforces_metal_constraints() -> None:
    prompt = get_metal_system_prompt("M4 Max", 36)
    assert "import mlx.core as mx" in prompt
    assert "mx.fast.metal_kernel" in prompt
    assert "Do NOT use PyTorch" in prompt
    assert "submit" in prompt
    assert "def solution(*inputs):" in prompt

    reasoning_prompt = get_metal_reasoning_system_prompt("M4 Max", 36)
    assert "Do NOT use PyTorch" in reasoning_prompt
    assert "solution(*inputs)" in reasoning_prompt


def test_metal_backend_gpu_policy() -> None:
    backend = get_backend("metal")
    assert backend.allowed_gpus == ["M4MAX"]


def test_metal_sandbox_config_defaults() -> None:
    config = MetalSandboxConfig()
    sandbox = MetalSandbox("print('x')", config)
    assert config.ssh_host == "macbook"
    assert sandbox.problem_code
