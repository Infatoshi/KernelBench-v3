from src.agent.metal_sandbox import MetalSandbox, MetalSandboxConfig
from src.prompts.metal_system import (
    get_metal_reasoning_system_prompt,
    get_metal_system_prompt,
)
import metal_batch_eval
import metal_eval


def test_metal_eval_exports_expected_entrypoints() -> None:
    assert callable(metal_eval.run_agent_on_modal)
    assert callable(metal_eval.main)
    assert "M4MAX" in metal_eval.GPU_SPECS


def test_metal_prompt_enforces_metal_constraints() -> None:
    prompt = get_metal_system_prompt("M4 Max", 36)
    assert "import mlx.core as mx" in prompt
    assert "mx.fast.metal_kernel" in prompt
    assert "Do NOT use PyTorch" in prompt
    assert 'python -c "import mlx.core as mx, solution;' in prompt

    reasoning_prompt = get_metal_reasoning_system_prompt("M4 Max", 36)
    assert "Do NOT use PyTorch" in reasoning_prompt
    assert "solution(a, b)" in reasoning_prompt


def test_metal_batch_eval_has_main_entrypoint() -> None:
    assert callable(metal_batch_eval.main)


def test_metal_sandbox_config_defaults() -> None:
    config = MetalSandboxConfig()
    sandbox = MetalSandbox("print('x')", config)
    assert config.ssh_host == "macbook"
    assert sandbox.problem_code
