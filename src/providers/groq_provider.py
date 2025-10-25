from __future__ import annotations

from typing import Any, Dict, List

from groq import Groq

from .base import BaseProvider, ProviderConfig


class GroqProvider(BaseProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.client = Groq(api_key=config.api_key, timeout=config.timeout)

    def _perform_preflight_request(self) -> None:
        self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0.0,
        )

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
