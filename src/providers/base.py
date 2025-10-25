from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


Message = Dict[str, Any]


@dataclass
class ProviderConfig:
    """Configuration required to construct a provider wrapper."""

    provider: str
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseProvider:
    """Base class for LLM providers."""

    #: Whether this provider requires an API key to operate.
    requires_api_key: bool = True
    #: Default message used for the lightweight preflight ping.
    preflight_message: str = "KernelBench preflight ping"
    #: Maximum tokens requested during preflight checks.
    preflight_max_output_tokens: int | None = 1

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    def preflight(self) -> None:
        """Light-weight credential/model check before heavy benchmarking."""
        if self.requires_api_key and not self.config.api_key:
            raise RuntimeError(
                f"Provider '{self.config.provider}' requires an API key but none was supplied."
            )

        try:
            self._perform_preflight_request()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Preflight request for provider '{self.config.provider}' failed: {exc}"
            ) from exc

    def generate(
        self,
        messages: List[Message],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError

    def _perform_preflight_request(self) -> None:
        """Execute the default preflight chat completion."""
        kwargs: Dict[str, Any] = {"temperature": 0.0}
        if self.preflight_max_output_tokens is not None:
            kwargs["max_tokens"] = self.preflight_max_output_tokens
        self.generate(
            [{"role": "user", "content": self.preflight_message}],
            **kwargs,
        )
