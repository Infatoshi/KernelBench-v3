from __future__ import annotations

from typing import Any, Dict, List

from anthropic import Anthropic

from .base import BaseProvider, ProviderConfig


class AnthropicProvider(BaseProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.client = Anthropic(api_key=config.api_key, timeout=config.timeout)

    def _perform_preflight_request(self) -> None:
        self.client.messages.create(
            model=self.config.model,
            system="",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )

    def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        system_prompt = ""
        formatted_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                system_prompt = content
            elif role in {"user", "assistant"}:
                formatted_messages.append({"role": role, "content": content})

        response = self.client.messages.create(
            model=self.config.model,
            system=system_prompt,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.content[0].text
