from __future__ import annotations

import os
from typing import Dict, Iterable, Type

from .base import BaseProvider, ProviderConfig
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .groq_provider import GroqProvider
from .xai_provider import XAIProvider
from .gemini_provider import GeminiProvider
from .cerebras_provider import CerebrasProvider
from .vllm_provider import VLLMProvider
from .sglang_provider import SGLangProvider
from .ollama_provider import OllamaProvider
from .openrouter_provider import OpenRouterProvider


PROVIDER_REGISTRY: Dict[str, Type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "groq": GroqProvider,
    "xai": XAIProvider,
    "gemini": GeminiProvider,
    "cerebras": CerebrasProvider,
    "vllm": VLLMProvider,
    "sglang": SGLangProvider,
    "ollama": OllamaProvider,
    "openrouter": OpenRouterProvider,
}


_PROVIDER_API_KEY_ALIASES: Dict[str, Iterable[str]] = {
    "openai": ("OPENAI_API_KEY",),
    "groq": ("GROQ_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "xai": ("XAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GENAI_API_KEY"),
    "cerebras": ("CEREBRAS_API_KEY",),
    "vllm": ("VLLM_API_KEY",),
    "sglang": ("SGLANG_API_KEY",),
    "ollama": ("OLLAMA_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY", "OR_API_KEY"),
}


def resolve_provider_api_key(provider: str) -> str | None:
    """Resolve provider API keys from common environment variable names."""
    provider_slug = provider.lower()
    candidates = list(_PROVIDER_API_KEY_ALIASES.get(provider_slug, ()))

    for env_name in candidates:
        value = os.getenv(env_name)
        if value:
            return value
    return None


def create_provider(config: ProviderConfig) -> BaseProvider:
    provider_cls = PROVIDER_REGISTRY.get(config.provider.lower())
    if provider_cls is None:
        raise ValueError(f"Unsupported provider: {config.provider}")
    instance = provider_cls(config)
    instance.preflight()
    return instance


def verify_model_responds_hello(provider: str, model: str, base_url: str | None = None) -> None:
    """Ensure a provider/model pair can generate a simple deterministic response."""
    provider_config = ProviderConfig(
        provider=provider,
        model=model,
        api_key=resolve_provider_api_key(provider),
        base_url=base_url,
    )
    instance = create_provider(provider_config)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Reply with exactly the word Hello."},
    ]
    handshake_tokens = 16
    preflight_cap = getattr(instance, "preflight_max_output_tokens", None)
    if isinstance(preflight_cap, int) and preflight_cap > 0:
        handshake_tokens = max(handshake_tokens, preflight_cap)
    response = instance.generate(messages, max_tokens=handshake_tokens)
    if response is None:
        raise RuntimeError("No response returned by provider.")
    normalized = response.strip()
    if normalized not in {"Hello.", "Hello"}:
        raise RuntimeError(
            f"Unexpected response from provider '{provider}' model '{model}': {normalized!r}"
        )
