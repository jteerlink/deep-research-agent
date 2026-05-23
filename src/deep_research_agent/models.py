"""Compatibility wrapper for packaged model metadata contracts."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deep_research_agent.models import (  # noqa: E402,F401
    ConfiguredModelClient,
    FallbackEvent,
    ModelClient,
    ModelRequest,
    ModelResponse,
    build_model_client,
)

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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly audit representation."""

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


class ModelClient(Protocol):
    """Minimal interface expected by future graph nodes."""

    async def invoke(self, request: ModelRequest) -> ModelResponse:
        """Invoke a model and return normalized output."""


@dataclass(frozen=True)
class ConfiguredModelClient:
    """Import-safe placeholder client that exposes selected provider metadata.

    The G003 graph can depend on deterministic provider selection and fallback
    audit events before live transports exist. A provider with no configured
    model is treated as unavailable and the client selects the next configured
    provider from ``fallback_order`` while recording why the fallback occurred.
    """

    config: AgentConfig

    @property
    def primary_provider(self) -> ModelProvider:
        return self.config.primary_provider

    @property
    def primary_model(self) -> str:
        return self.config.provider_model()

    @property
    def provider_sequence(self) -> tuple[ModelProvider, ...]:
        """Return primary provider followed by unique configured fallbacks."""

        return tuple(_unique_providers((self.primary_provider, *self.config.fallback_order)))

    def _select_provider(
        self, request: ModelRequest
    ) -> tuple[ModelProvider, str, tuple[FallbackEvent, ...]]:
        events: list[FallbackEvent] = []
        candidates = self.provider_sequence
        if not candidates:
            candidates = (self.primary_provider,)

        for candidate in candidates:
            model = self.config.provider_model(candidate)
            if model:
                return candidate, model, tuple(events)
            events.append(
                FallbackEvent.record(
                    provider=candidate,
                    model=model,
                    trigger="model_not_configured",
                    node=request.node,
                    retry_count=0,
                )
            )

        provider = candidates[0]
        return provider, self.config.provider_model(provider), tuple(events)

    async def invoke(self, request: ModelRequest) -> ModelResponse:
        """Return a deterministic skeleton response with fallback audit metadata."""

        provider, model, fallback_events = self._select_provider(request)
        return ModelResponse(
            provider=provider,
            model=model,
            content="",
            structured={
                "node": request.node,
                "status": "not_implemented",
                "message": "Model transport is intentionally stubbed in G003 foundation.",
                "selected_provider": provider.value,
                "selected_model": model,
                "fallback_used": bool(fallback_events),
                "fallback_events": [event.to_dict() for event in fallback_events],
                "request_metadata": dict(request.metadata),
            },
            fallback_events=fallback_events,
        )


def _unique_providers(providers: Iterable[ModelProvider]) -> Iterable[ModelProvider]:
    seen: set[ModelProvider] = set()
    for provider in providers:
        if provider in seen:
            continue
        seen.add(provider)
        yield provider


def build_model_client(config: AgentConfig | None = None) -> ConfiguredModelClient:
    """Build an import-safe model client from explicit configuration."""

    return ConfiguredModelClient(config=load_config() if config is None else config)
