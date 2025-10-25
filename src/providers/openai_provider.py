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
        self._use_responses_api = uses_gpt5_family
        if uses_gpt5_family:
            # GPT-5 responses stream reasoning tokens before final text; budget generously.
            current_budget = self.preflight_max_output_tokens or 0
            self.preflight_max_output_tokens = max(256, current_budget)

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
                    request["max_completion_tokens"] = max(max_tokens, 16)
                else:
                    request["max_tokens"] = max_tokens
        elif self._uses_max_completion_tokens and "max_completion_tokens" not in request:
            # Default to a small completion budget for new models when none supplied.
            request["max_completion_tokens"] = 256

        if self._use_responses_api:
            responses_kwargs: Dict[str, Any] = {}
            if kwargs:
                responses_kwargs.update(kwargs)

            if max_tokens is not None:
                target_tokens = max(max_tokens, 256)
            elif self.preflight_max_output_tokens is not None:
                target_tokens = max(self.preflight_max_output_tokens, 256)
            else:
                target_tokens = 256
            responses_kwargs["max_output_tokens"] = target_tokens

            if not self._requires_default_temperature and temperature is not None:
                responses_kwargs["temperature"] = temperature

            normalized_messages: List[Dict[str, Any]] = []
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                if isinstance(content, list):
                    # Join multi-part content into a single text segment.
                    text_parts: List[str] = []
                    for item in content:
                        text_value = getattr(item, "text", None)
                        if text_value:
                            text_parts.append(text_value)
                        elif isinstance(item, dict):
                            val = item.get("text")
                            if val:
                                text_parts.append(val)
                    content = "".join(text_parts)
                normalized_messages.append({"role": role, "content": content})

            response = self.client.responses.create(
                model=self.config.model,
                input=normalized_messages,
                **responses_kwargs,
            )

            content_text = getattr(response, "output_text", None)
            if content_text:
                return content_text

            outputs = getattr(response, "output", []) or []
            segments: List[str] = []
            for item in outputs:
                content_items = getattr(item, "content", []) or []
                for part in content_items:
                    text_value = getattr(part, "text", None)
                    if text_value:
                        segments.append(text_value)
                    elif isinstance(part, dict):
                        val = part.get("text")
                        if val:
                            segments.append(val)

            return "".join(segments)

        response = self.client.chat.completions.create(**request)
        message = response.choices[0].message
        content = getattr(message, "content", "")

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)
                elif isinstance(item, dict):
                    part_text = item.get("text")
                    if part_text:
                        parts.append(part_text)
            content = "".join(parts)

        if content is None:
            content = ""

        return content
