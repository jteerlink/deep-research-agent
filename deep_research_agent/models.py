"""Provider-neutral model metadata and fallback contracts for graph execution.

The G003 workflow needs auditable provider/model metadata without requiring live
hosted model calls. This module keeps that boundary import-safe: it records why
fallbacks were selected and returns deterministic placeholder responses that
future live transports can replace behind the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from .config import AppConfig, ModelProvider, load_config


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
        """Build a timestamped fallback event."""

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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable event representation."""

        return {
            "provider": self.provider.value,
            "model": self.model,
            "trigger": self.trigger,
            "node": self.node,
            "retry_count": self.retry_count,
            "timestamp": self.timestamp,
            "error_class": self.error_class,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class ModelResponse:
    """A provider-neutral model response with fallback audit trail."""

    provider: ModelProvider
    model: str
    content: str
    structured: dict[str, Any] | None = None
    fallback_events: tuple[FallbackEvent, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable response representation."""

        return {
            "provider": self.provider.value,
            "model": self.model,
            "content": self.content,
            "structured": self.structured,
            "fallback_events": [event.to_dict() for event in self.fallback_events],
        }


class ModelClient(Protocol):
    """Minimal interface expected by graph nodes."""

    async def invoke(self, request: ModelRequest) -> ModelResponse:
        """Invoke a model and return normalized output."""


@dataclass(frozen=True)
class ConfiguredModelClient:
    """Import-safe placeholder client that exposes selected provider metadata."""

    config: AppConfig

    @property
    def primary_provider(self) -> ModelProvider:
        return self.config.primary_provider

    @property
    def primary_model(self) -> str:
        return self.config.primary_model

    def provider_model(self, provider: ModelProvider) -> str:
        """Return the configured model for a provider."""

        if provider is ModelProvider.OLLAMA_NATIVE:
            return self.config.ollama_native.model
        if provider is ModelProvider.OLLAMA_OPENAI:
            return self.config.ollama_openai.model
        if provider is ModelProvider.CODEX:
            return self.config.codex.model
        return self.config.openai.model

    def provider_available(self, provider: ModelProvider) -> bool:
        """Return whether the placeholder can select provider metadata locally."""

        if provider is ModelProvider.OPENAI:
            return bool(self.config.openai.api_key)
        if provider is ModelProvider.CODEX:
            return bool(self.config.codex.api_key)
        return True

    def provider_order(self) -> tuple[ModelProvider, ...]:
        """Return primary provider plus deterministic hosted fallbacks."""

        ordered = [self.primary_provider, ModelProvider.CODEX, ModelProvider.OPENAI]
        deduped: list[ModelProvider] = []
        for provider in ordered:
            if provider not in deduped:
                deduped.append(provider)
        return tuple(deduped)

    async def invoke(self, request: ModelRequest) -> ModelResponse:
        """Return deterministic metadata and fallback events without live calls."""

        fallback_events: list[FallbackEvent] = []
        for retry_count, provider in enumerate(self.provider_order()):
            model = self.provider_model(provider)
            if not self.provider_available(provider):
                fallback_events.append(
                    FallbackEvent.record(
                        provider=provider,
                        model=model,
                        trigger="missing_api_key",
                        node=request.node,
                        retry_count=retry_count,
                    )
                )
                continue
            return ModelResponse(
                provider=provider,
                model=model,
                content="",
                structured={
                    "node": request.node,
                    "status": "metadata_only",
                    "message": "Live model transport is not enabled in the local G003 workflow.",
                    "request_metadata": dict(request.metadata),
                },
                fallback_events=tuple(fallback_events),
            )

        # All configured hosted fallbacks were unavailable; preserve metadata and
        # let the graph continue deterministically for offline smoke tests.
        return ModelResponse(
            provider=self.primary_provider,
            model=self.primary_model,
            content="",
            structured={
                "node": request.node,
                "status": "no_available_model",
                "message": "No local model metadata provider was selectable.",
                "request_metadata": dict(request.metadata),
            },
            fallback_events=tuple(fallback_events),
        )


def build_model_client(config: AppConfig | None = None) -> ConfiguredModelClient:
    """Build an import-safe model client from explicit configuration."""

    return ConfiguredModelClient(config=load_config() if config is None else config)
