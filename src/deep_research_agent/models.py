"""Provider-neutral model adapter skeleton for the foundation story.

This module deliberately defines the contract without performing live provider
calls yet. Later graph nodes can depend on these stable request/response and
fallback metadata shapes while concrete clients are filled in behind the same
interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from .configuration import AgentConfig, ModelProvider, load_config


@dataclass(frozen=True)
class ModelRequest:
    """A model invocation request independent of provider transport."""

    node: str
    prompt: str
    response_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FallbackEvent:
    """Auditable metadata captured whenever model fallback is selected."""

    provider: ModelProvider
    model: str
    trigger: str
    node: str
    retry_count: int
    timestamp: str
    error_class: str = ""
    error_message: str = ""

    @classmethod
    def record(
        cls,
        *,
        provider: ModelProvider,
        model: str,
        trigger: str,
        node: str,
        retry_count: int,
        error: Exception | None = None,
    ) -> FallbackEvent:
        return cls(
            provider=provider,
            model=model,
            trigger=trigger,
            node=node,
            retry_count=retry_count,
            timestamp=datetime.now(UTC).isoformat(),
            error_class=type(error).__name__ if error else "",
            error_message=str(error) if error else "",
        )


@dataclass(frozen=True)
class ModelResponse:
    """A provider-neutral model response with fallback audit trail."""

    provider: ModelProvider
    model: str
    content: str
    structured: dict[str, Any] | None = None
    fallback_events: tuple[FallbackEvent, ...] = ()


class ModelClient(Protocol):
    """Minimal interface expected by future graph nodes."""

    async def invoke(self, request: ModelRequest) -> ModelResponse:
        """Invoke a model and return normalized output."""


@dataclass(frozen=True)
class ConfiguredModelClient:
    """Import-safe placeholder client that exposes selected provider metadata."""

    config: AgentConfig

    @property
    def primary_provider(self) -> ModelProvider:
        return self.config.primary_provider

    @property
    def primary_model(self) -> str:
        return self.config.provider_model()

    async def invoke(self, request: ModelRequest) -> ModelResponse:
        """Return a deterministic skeleton response until live clients are implemented."""

        return ModelResponse(
            provider=self.primary_provider,
            model=self.primary_model,
            content="",
            structured={
                "node": request.node,
                "status": "not_implemented",
                "message": "Model transport is intentionally stubbed in G001 foundation.",
            },
        )


def build_model_client(config: AgentConfig | None = None) -> ConfiguredModelClient:
    """Build an import-safe model client from explicit configuration."""

    return ConfiguredModelClient(config=load_config() if config is None else config)
