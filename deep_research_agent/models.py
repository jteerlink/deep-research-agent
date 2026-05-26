"""Provider-neutral model metadata and fallback contracts for graph execution.

The G003 workflow needs auditable provider/model metadata without requiring live
hosted model calls. This module keeps that boundary import-safe: it records why
fallbacks were selected and returns deterministic placeholder responses that
future live transports can replace behind the same interface.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

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

    def provider_base_url(self, provider: ModelProvider) -> str:
        """Return the configured transport base URL for a provider."""

        if provider is ModelProvider.OLLAMA_NATIVE:
            return self.config.ollama_native.base_url
        if provider is ModelProvider.OLLAMA_OPENAI:
            return self.config.ollama_openai.base_url
        if provider is ModelProvider.CODEX:
            return self.config.codex.base_url
        return self.config.openai.base_url

    def provider_api_key(self, provider: ModelProvider) -> str:
        """Return the configured API key for a provider when one exists."""

        if provider is ModelProvider.OLLAMA_NATIVE:
            return self.config.ollama_native.api_key
        if provider is ModelProvider.OLLAMA_OPENAI:
            return self.config.ollama_openai.api_key
        if provider is ModelProvider.CODEX:
            return self.config.codex.api_key
        return self.config.openai.api_key

    def provider_available(self, provider: ModelProvider) -> bool:
        """Return whether the placeholder can select provider metadata locally."""

        model = self.provider_model(provider)
        base_url = self.provider_base_url(provider)
        if not model:
            return False
        if provider is ModelProvider.OLLAMA_NATIVE:
            host = urlparse(base_url).hostname or ""
            return bool(self.provider_api_key(provider)) and not _is_local_host(host)
        if provider is ModelProvider.OLLAMA_OPENAI:
            host = urlparse(base_url).hostname or ""
            return bool(base_url and self.provider_api_key(provider)) and not _is_local_host(host)
        if provider is ModelProvider.OPENAI:
            return bool(base_url and self.config.openai.api_key)
        if provider is ModelProvider.CODEX:
            return bool(base_url and self.config.codex.api_key)
        return bool(base_url)

    def provider_order(self) -> tuple[ModelProvider, ...]:
        """Return primary provider plus deterministic hosted fallbacks."""

        ordered = [self.primary_provider, *self.config.fallback_order]
        deduped: list[ModelProvider] = []
        for provider in ordered:
            if provider not in deduped:
                deduped.append(provider)
        return tuple(deduped)

    async def invoke(self, request: ModelRequest) -> ModelResponse:
        """Invoke structured model requests while keeping metadata-only calls offline."""

        if request.response_schema is not None:
            return await self._invoke_structured(request)

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

    async def _invoke_structured(self, request: ModelRequest) -> ModelResponse:
        fallback_events: list[FallbackEvent] = []
        for retry_count, provider in enumerate(self.provider_order()):
            model = self.provider_model(provider)
            if not self.provider_available(provider):
                fallback_events.append(
                    FallbackEvent.record(
                        provider=provider,
                        model=model,
                        trigger="provider_unavailable",
                        node=request.node,
                        retry_count=retry_count,
                    )
                )
                continue
            try:
                content = await self._transport(provider, model, request)
                return ModelResponse(
                    provider=provider,
                    model=model,
                    content=content,
                    structured=_parse_json_content(content),
                    fallback_events=tuple(fallback_events),
                )
            except Exception as exc:  # pragma: no cover - transport errors vary by provider
                fallback_events.append(
                    FallbackEvent.record(
                        provider=provider,
                        model=model,
                        trigger="transport_error",
                        node=request.node,
                        retry_count=retry_count,
                        error=exc,
                    )
                )
        return ModelResponse(
            provider=self.primary_provider,
            model=self.primary_model,
            content="",
            structured={
                "node": request.node,
                "status": "model_unavailable",
                "message": "No configured model transport returned structured output.",
                "request_metadata": dict(request.metadata),
            },
            fallback_events=tuple(fallback_events),
        )

    async def _transport(
        self, provider: ModelProvider, model: str, request: ModelRequest
    ) -> str:
        if provider is ModelProvider.OLLAMA_NATIVE:
            return await self._invoke_ollama_native(provider, model, request)
        return await self._invoke_openai_compatible(provider, model, request)

    async def _invoke_openai_compatible(
        self, provider: ModelProvider, model: str, request: ModelRequest
    ) -> str:
        base_url = self.provider_base_url(provider).rstrip("/")
        headers = {"Content-Type": "application/json"}
        api_key = self.provider_api_key(provider)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only valid JSON that matches the requested schema.",
                },
                {"role": "user", "content": request.prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        async with httpx.AsyncClient(timeout=self.config.search.timeout_seconds) as client:
            response = await client.post(
                f"{base_url}/chat/completions", headers=headers, json=payload
            )
            response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"])

    async def _invoke_ollama_native(
        self, provider: ModelProvider, model: str, request: ModelRequest
    ) -> str:
        base_url = self.provider_base_url(provider).rstrip("/")
        endpoint = (
            f"{base_url}/generate" if base_url.endswith("/api") else f"{base_url}/api/generate"
        )
        headers = {"Content-Type": "application/json"}
        api_key = self.provider_api_key(provider)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {"model": model, "prompt": request.prompt, "stream": False, "format": "json"}
        async with httpx.AsyncClient(timeout=self.config.search.timeout_seconds) as client:
            response = await client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
        return str(response.json().get("response", ""))


def _parse_json_content(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("structured model output must be a JSON object")
    return parsed


def _is_local_host(host: str) -> bool:
    return host in {"", "localhost", "127.0.0.1", "::1", "0.0.0.0"}


def build_model_client(config: AppConfig | None = None) -> ConfiguredModelClient:
    """Build an import-safe model client from explicit configuration."""

    return ConfiguredModelClient(config=load_config() if config is None else config)
