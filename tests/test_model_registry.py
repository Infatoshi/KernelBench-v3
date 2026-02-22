import modal_eval


def test_openrouter_alpha_models_are_registered() -> None:
    glm5 = modal_eval.get_model_config("z-ai/glm-5")
    aurora = modal_eval.get_model_config("openrouter/aurora-alpha")

    assert glm5 is not None
    assert glm5.provider == "openrouter"
    assert glm5.model_id == "z-ai/glm-5"

    assert aurora is not None
    assert aurora.provider == "openrouter"
    assert aurora.model_id == "openrouter/aurora-alpha"
