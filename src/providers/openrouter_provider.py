from __future__ import annotations

import os
from typing import Any, Dict, List

from openai import OpenAI

from .base import BaseProvider, ProviderConfig


class OpenRouterProvider(BaseProvider):
    """Provider adapter for OpenRouter's OpenAI-compatible API."""

    default_base_url = "https://openrouter.ai/api/v1"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        base_url = config.base_url or self.default_base_url

        self.client = OpenAI(
            api_key=config.api_key,
            base_url=base_url,
            timeout=config.timeout,
        )

        extra = config.extra or {}
        referer = extra.get("http_referer") or os.getenv("OR_SITE_URL")
        title = extra.get("x_title") or os.getenv("OR_APP_NAME")

        headers: Dict[str, str] = {}
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title

        self._default_headers = headers if headers else None

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
            max_tokens=max(max_tokens or 0, 2048) if max_tokens is not None else None,
            extra_headers=self._default_headers,
            **kwargs,
        )
        message = response.choices[0].message
        content = getattr(message, "content", "")

        if isinstance(content, list):
            fragments: List[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if text:
                    fragments.append(text)
                elif isinstance(item, dict):
                    value = item.get("text")
                    if value:
                        fragments.append(value)
            content = "".join(fragments)

        if content is None:
            content = ""

        return content
