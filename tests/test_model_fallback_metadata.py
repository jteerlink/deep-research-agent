from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, replace

from deep_research_agent.config import (
    AppConfig,
    CodexConfig,
    ModelProvider,
    OllamaNativeConfig,
    OllamaOpenAIConfig,
    OpenAIConfig,
    SearchConfig,
)
from deep_research_agent.models import (
    ConfiguredModelClient,
    FallbackEvent,
    ModelRequest,
    ModelResponse,
    build_model_client,
)


def _config(
    *,
    primary_provider: ModelProvider = ModelProvider.OLLAMA_NATIVE,
    fallback_order: tuple[ModelProvider, ...] = (),
    openai: OpenAIConfig | None = None,
    codex: CodexConfig | None = None,
) -> AppConfig:
    return AppConfig(
        primary_provider=primary_provider,
        fallback_order=fallback_order,
        ollama_native=OllamaNativeConfig(),
        ollama_openai=OllamaOpenAIConfig(),
        openai=openai or OpenAIConfig(),
        codex=codex or CodexConfig(),
        search=SearchConfig(),
    )


def test_fallback_event_record_captures_serializable_error_metadata() -> None:
    error = RuntimeError("primary transport unavailable")

    event = FallbackEvent.record(
        provider=ModelProvider.OLLAMA_NATIVE,
        model="gpt-oss:120b",
        trigger="transport_error",
        node="supervisor",
        retry_count=2,
        error=error,
    )

    assert event.provider is ModelProvider.OLLAMA_NATIVE
    assert event.model == "gpt-oss:120b"
    assert event.trigger == "transport_error"
    assert event.node == "supervisor"
    assert event.retry_count == 2
    assert event.error_class == "RuntimeError"
    assert event.error_message == "primary transport unavailable"
    assert event.timestamp.endswith("+00:00")
    json.dumps(event.to_dict())
    json.dumps(asdict(event))


def test_model_response_defaults_to_empty_fallback_events() -> None:
    response = ModelResponse(provider=ModelProvider.OPENAI, model="gpt-4.1-mini", content="")

    assert response.fallback_events == ()


def test_configured_model_client_selects_primary_without_fallback_events() -> None:
    config = _config(
        primary_provider=ModelProvider.OPENAI,
        openai=OpenAIConfig(api_key="sk-test-openai"),
    )
    client = build_model_client(config)

    response = asyncio.run(client.invoke(ModelRequest(node="researcher", prompt="research acme")))

    assert response.provider is ModelProvider.OPENAI
    assert response.model == "gpt-4.1-mini"
    assert response.fallback_events == ()
    assert response.structured is not None
    assert response.structured["fallback_used"] is False
    assert response.structured["selected_provider"] == "openai"


def test_configured_model_client_records_fallback_when_primary_model_is_missing() -> None:
    config = _config(
        primary_provider=ModelProvider.CODEX,
        fallback_order=(ModelProvider.OPENAI,),
        openai=OpenAIConfig(model="gpt-fallback", api_key="sk-test-openai"),
    )
    request = ModelRequest(
        node="supervisor",
        prompt="decide next step",
        metadata={"thread_id": "thread-123", "iteration": 1},
    )

    response = asyncio.run(ConfiguredModelClient(config).invoke(request))

    assert response.provider is ModelProvider.OPENAI
    assert response.model == "gpt-fallback"
    assert len(response.fallback_events) == 1
    event = response.fallback_events[0]
    assert event.provider is ModelProvider.CODEX
    assert event.model == ""
    assert event.trigger == "missing_model"
    assert event.node == "supervisor"
    assert event.retry_count == 0
    assert response.structured is not None
    assert response.structured["fallback_used"] is True
    assert response.structured["request_metadata"] == {"thread_id": "thread-123", "iteration": 1}
    assert response.structured["fallback_events"] == [event.to_dict()]


def test_provider_sequence_deduplicates_primary_from_fallback_order() -> None:
    config = replace(
        _config(primary_provider=ModelProvider.OPENAI),
        fallback_order=(ModelProvider.OPENAI, ModelProvider.CODEX),
    )

    assert ConfiguredModelClient(config).provider_order() == (
        ModelProvider.OPENAI,
        ModelProvider.CODEX,
    )
