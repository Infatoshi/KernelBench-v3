from src.prompts.triton_system import (
    get_triton_reasoning_system_prompt,
    get_triton_system_prompt,
)
import triton_batch_eval
import triton_eval


def test_triton_eval_exports_expected_entrypoints() -> None:
    assert callable(triton_eval.run_agent_on_modal)
    assert callable(triton_eval.main)
    assert triton_eval.MODELS


def test_triton_prompt_enforces_triton_constraints() -> None:
    prompt = get_triton_system_prompt("H100", 80)
    assert "@triton.jit" in prompt
    assert "triton.language as tl" in prompt
    assert "load_inline" in prompt
    assert "submit tool" in prompt

    reasoning_prompt = get_triton_reasoning_system_prompt("H100", 80)
    assert "No CUDA C++" in reasoning_prompt
    assert "Return only complete Python code" in reasoning_prompt


def test_triton_batch_eval_has_main_entrypoint() -> None:
    assert callable(triton_batch_eval.main)
