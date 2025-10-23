from __future__ import annotations

from typing import Any, Dict, List

import google.generativeai as genai

from .base import BaseProvider, ProviderConfig


class GeminiProvider(BaseProvider):
    preflight_max_output_tokens = None

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        genai.configure(api_key=config.api_key)
        self.model = genai.GenerativeModel(model_name=config.model)

    def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        prompt_parts: List[str] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n")
            else:
                prompt_parts.append(f"Assistant: {content}\n")

        prompt = "".join(prompt_parts)
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            **kwargs,
        )
        if max_tokens is not None:
            generation_config.max_output_tokens = max_tokens

        response = self.model.generate_content(
            prompt,
            generation_config=generation_config,
        )
        try:
            return response.text
        except ValueError:
            # Fallback: aggregate textual parts manually if quick accessor fails.
            text_segments: List[str] = []
            finish_reasons: List[Any] = []
            safety_blocks: List[Any] = []

            for candidate in getattr(response, "candidates", []) or []:
                finish_reasons.append(getattr(candidate, "finish_reason", None))
                safety_blocks.append(getattr(candidate, "safety_ratings", None))

                content = getattr(candidate, "content", None)
                if not content:
                    continue
                for part in getattr(content, "parts", []) or []:
                    text_value = getattr(part, "text", None)
                    if text_value:
                        text_segments.append(text_value)

            if text_segments:
                return "".join(text_segments)

            raise RuntimeError(
                "Gemini response contained no textual parts; "
                f"finish_reasons={finish_reasons}, safety_ratings={safety_blocks}"
            )
