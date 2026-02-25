from src.models import get_model_config


def test_registered_models_are_resolvable() -> None:
    glm5 = get_model_config("z-ai/glm-5")
    assert glm5 is not None
    assert glm5.provider == "openrouter"
    assert glm5.model_id == "z-ai/glm-5"

    sonnet = get_model_config("anthropic/claude-sonnet-4.6")
    assert sonnet is not None
    assert sonnet.provider == "openrouter"

    qwen_coder = get_model_config("qwen/qwen3-coder-next")
    assert qwen_coder is not None
    assert qwen_coder.provider == "openrouter"
