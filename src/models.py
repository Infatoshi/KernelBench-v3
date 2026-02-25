"""Model configurations, pricing, and provider client management."""

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Optional, List, Literal, Dict, Any, Tuple


# =============================================================================
# Dynamic Pricing (fetched from APIs, cached in memory)
# =============================================================================

_openrouter_pricing_cache: Dict[str, Tuple[float, float]] = {}
_openrouter_models_cache: Optional[Dict[str, Any]] = None


def _fetch_openrouter_models() -> Dict[str, Any]:
    """Fetch all OpenRouter models and cache them."""
    global _openrouter_models_cache
    if _openrouter_models_cache is not None:
        return _openrouter_models_cache

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return {}

    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            _openrouter_models_cache = {m["id"]: m for m in data.get("data", [])}
            return _openrouter_models_cache
    except Exception as e:
        print(f"Warning: Failed to fetch OpenRouter models: {e}")
        return {}


def get_openrouter_pricing(model_id: str) -> Optional[Tuple[float, float]]:
    """Get pricing for an OpenRouter model (input, output per million tokens)."""
    if model_id in _openrouter_pricing_cache:
        return _openrouter_pricing_cache[model_id]

    models = _fetch_openrouter_models()
    if model_id not in models:
        return None

    pricing = models[model_id].get("pricing", {})
    input_per_token = float(pricing.get("prompt", 0))
    output_per_token = float(pricing.get("completion", 0))

    result = (input_per_token * 1_000_000, output_per_token * 1_000_000)
    _openrouter_pricing_cache[model_id] = result
    return result


def is_valid_openrouter_model(model_id: str) -> bool:
    """Check if a model ID is valid on OpenRouter."""
    models = _fetch_openrouter_models()
    return model_id in models


MODEL_TO_OPENROUTER = {
    "claude-opus-4-5-20251101": "anthropic/claude-opus-4.5",
    "claude-sonnet-4-5-20250929": "anthropic/claude-sonnet-4.5",
    "gpt-5.2": "openai/gpt-5.2",
    "gemini-3-flash-preview": "google/gemini-3-flash-preview",
    "gemini-3-pro-preview": "google/gemini-3-pro-preview",
    "grok-4-1-fast-reasoning": "x-ai/grok-4.1-fast",
}


@dataclass
class ModelConfig:
    """Configuration for an LLM model."""
    name: str
    model_id: str
    provider: Literal["anthropic", "openai", "gemini", "xai", "openrouter"]
    use_xml_tools: bool = False
    provider_order: Optional[List[str]] = None
    reasoning_mode: bool = False


MODELS = {
    "anthropic/claude-opus-4.6": ModelConfig(
        name="Claude Opus 4.6", model_id="anthropic/claude-opus-4.6", provider="openrouter"
    ),
    "anthropic/claude-sonnet-4.6": ModelConfig(
        name="Claude Sonnet 4.6", model_id="anthropic/claude-sonnet-4.6", provider="openrouter"
    ),
    "openai/gpt-5.2-codex": ModelConfig(
        name="GPT-5.2 Codex", model_id="openai/gpt-5.2-codex", provider="openrouter"
    ),
    "openai/gpt-5.3-codex": ModelConfig(
        name="GPT-5.3 Codex", model_id="openai/gpt-5.3-codex", provider="openrouter"
    ),
    "google/gemini-3-flash-preview": ModelConfig(
        name="Gemini 3 Flash Preview", model_id="google/gemini-3-flash-preview", provider="openrouter"
    ),
    "google/gemini-3-pro-preview": ModelConfig(
        name="Gemini 3 Pro Preview", model_id="google/gemini-3-pro-preview", provider="openrouter"
    ),
    "google/gemini-3.1-pro-preview": ModelConfig(
        name="Gemini 3.1 Pro Preview", model_id="google/gemini-3.1-pro-preview", provider="openrouter"
    ),
    "deepseek/deepseek-v3.2": ModelConfig(
        name="DeepSeek V3.2", model_id="deepseek/deepseek-v3.2", provider="openrouter"
    ),
    "z-ai/glm-5": ModelConfig(
        name="GLM-5", model_id="z-ai/glm-5", provider="openrouter"
    ),
    "minimax/minimax-m2.5": ModelConfig(
        name="MiniMax M2.5", model_id="minimax/minimax-m2.5", provider="openrouter"
    ),
    "moonshotai/kimi-k2.5": ModelConfig(
        name="Kimi K2.5", model_id="moonshotai/kimi-k2.5", provider="openrouter", reasoning_mode=True
    ),
    "qwen/qwen3-coder-next": ModelConfig(
        name="Qwen3 Coder Next", model_id="qwen/qwen3-coder-next", provider="openrouter"
    ),
    "qwen/qwen3.5-397b-a17b": ModelConfig(
        name="Qwen3.5 397B A17B", model_id="qwen/qwen3.5-397b-a17b", provider="openrouter"
    ),
}


def get_model_config(model_key: str) -> Optional[ModelConfig]:
    """Get model config by key, supporting both predefined and dynamic OpenRouter models."""
    if model_key in MODELS:
        return MODELS[model_key]

    if "/" in model_key:
        if is_valid_openrouter_model(model_key):
            models = _fetch_openrouter_models()
            model_info = models.get(model_key, {})
            name = model_info.get("name", model_key)
            supported_params = model_info.get("supported_parameters", [])
            has_tools = "tools" in supported_params

            return ModelConfig(
                name=name,
                model_id=model_key,
                provider="openrouter",
                reasoning_mode=not has_tools
            )
        else:
            return None

    return None


def get_provider_client(provider: str):
    """Get API client for the specified provider."""
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic()

    elif provider == "openai":
        from openai import OpenAI
        return OpenAI()

    elif provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        return genai

    elif provider == "xai":
        from openai import OpenAI
        return OpenAI(
            api_key=os.environ.get("XAI_API_KEY"),
            base_url="https://api.x.ai/v1"
        )

    elif provider == "openrouter":
        from openai import OpenAI
        return OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")
