from __future__ import annotations

from typing import Any, Dict, List

import google.generativeai as genai

from .base import BaseProvider, ProviderConfig


class GeminiProvider(BaseProvider):
    preflight_max_output_tokens = 2048

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
        def normalize_content(value: Any) -> str:
            if isinstance(value, list):
                parts: List[str] = []
                for item in value:
                    text_value = getattr(item, "text", None)
                    if text_value:
                        parts.append(text_value)
                    elif isinstance(item, dict):
                        val = item.get("text")
                        if val:
                            parts.append(val)
                return "".join(parts)
            return str(value)

        normalized_messages: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = normalize_content(msg.get("content", ""))
            if role == "assistant":
                gemini_role = "model"
            else:
                gemini_role = "user"
            normalized_messages.append({
                "role": gemini_role,
                "parts": [{"text": content}],
            })

        if not normalized_messages:
            normalized_messages.append({
                "role": "user",
                "parts": [{"text": ""}],
            })
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            **kwargs,
        )
        if max_tokens is not None:
            generation_config.max_output_tokens = max(max_tokens, 512)
        elif self.preflight_max_output_tokens is not None:
            generation_config.max_output_tokens = self.preflight_max_output_tokens

        history = normalized_messages[:-1]
        last_message = normalized_messages[-1]
        last_text = "".join(part.get("text", "") for part in last_message["parts"])

        chat_session = self.model.start_chat(history=history)
        response = chat_session.send_message(
            last_text,
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
