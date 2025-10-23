from __future__ import annotations

from typing import Any, Dict, List

from openai import OpenAI

from .base import BaseProvider, ProviderConfig


class OpenAIProvider(BaseProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )
        model_lower = (config.model or "").lower()
        uses_gpt5_family = model_lower.startswith("gpt-5")
        self._uses_max_completion_tokens = uses_gpt5_family
        self._requires_default_temperature = uses_gpt5_family
        if uses_gpt5_family:
            # GPT-5 responses often exceed a single token, so widen the preflight budget.
            if getattr(self, "preflight_max_output_tokens", None) is not None:
                self.preflight_max_output_tokens = max(
                    32, self.preflight_max_output_tokens
                )

    def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        temperature = kwargs.pop("temperature", temperature)

        request: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }

        request.update(kwargs)

        if not self._requires_default_temperature and temperature is not None:
            request["temperature"] = temperature

        if max_tokens is not None:
            if (
                "max_tokens" not in request
                and "max_completion_tokens" not in request
            ):
                if self._uses_max_completion_tokens:
                    request["max_completion_tokens"] = max_tokens
                else:
                    request["max_tokens"] = max_tokens
        elif self._uses_max_completion_tokens and "max_completion_tokens" not in request:
            # Default to a small completion budget for new models when none supplied.
            request["max_completion_tokens"] = 256

        response = self.client.chat.completions.create(**request)
        return response.choices[0].message.content
