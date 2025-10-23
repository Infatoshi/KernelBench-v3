from __future__ import annotations

from typing import Any, Dict, List

try:
    from cerebras.cloud.sdk import Cerebras
except ImportError:  # pragma: no cover - fallback to OpenAI-compatible interface
    from openai import OpenAI as Cerebras

from .base import BaseProvider, ProviderConfig


class CerebrasProvider(BaseProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        kwargs = {"api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        if config.timeout:
            kwargs["timeout"] = config.timeout
        self.client = Cerebras(**kwargs)

    def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        # Cerebras SDK exposes a chat interface similar to OpenAI
        if hasattr(self.client, "chat"):
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return response.choices[0].message.content

        # Fallback if the SDK is not available and OpenAI client is used
        return ""


