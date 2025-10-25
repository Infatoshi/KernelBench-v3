from __future__ import annotations

from typing import Any, Dict, List

from openai import OpenAI

from .base import BaseProvider, ProviderConfig


class OllamaProvider(BaseProvider):
    requires_api_key = False

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        base_url = config.base_url or "http://localhost:11434/v1"
        self.client = OpenAI(api_key=config.api_key or "ollama", base_url=base_url, timeout=config.timeout)

    def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.choices[0].message.content

