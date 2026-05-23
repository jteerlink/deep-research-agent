"""Environment-backed foundation configuration.

The first story intentionally keeps provider configuration explicit instead of
inferring endpoint semantics from a single base URL. Ollama's native API and
Ollama's OpenAI-compatible API have different paths and payload shapes, so the
settings are split at the type level and in environment variable names.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping


class ModelProvider(StrEnum):
    """Supported primary/fallback model provider modes."""

    OLLAMA_NATIVE = "ollama_native"
    OLLAMA_OPENAI_COMPATIBLE = "ollama_openai_compatible"
    OPENAI = "openai"
    CODEX_OPENAI_COMPATIBLE = "codex_openai_compatible"


@dataclass(frozen=True)
class OllamaNativeConfig:
    """Configuration for Ollama's native `/api` transport."""

    base_url: str = "https://ollama.com/api"
    model: str = "deepseek-v4-pro:cloud"
    api_key: str = ""


@dataclass(frozen=True)
class OllamaOpenAICompatibleConfig:
    """Configuration for Ollama's OpenAI-compatible `/v1` transport."""

    base_url: str = "http://localhost:11434/v1"
    model: str = "deepseek-v4-pro:cloud"
    api_key: str = "ollama"


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    """Configuration for hosted OpenAI-compatible fallback models."""

    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4.1-mini"
    api_key: str = ""


@dataclass(frozen=True)
class CodexOpenAICompatibleConfig:
    """Configuration for Codex/OpenAI-compatible fallback experiments."""

    base_url: str = ""
    model: str = ""
    api_key: str = ""


@dataclass(frozen=True)
class SearchConfig:
    """Search defaults shared by future package wrappers around async_multi_search.py."""

    max_results: int = 5
    timeout_seconds: int = 10


@dataclass(frozen=True)
class AgentConfig:
    """Top-level foundation configuration for G001."""

    primary_provider: ModelProvider = ModelProvider.OLLAMA_NATIVE
    fallback_order: tuple[ModelProvider, ...] = field(
        default_factory=lambda: (
            ModelProvider.CODEX_OPENAI_COMPATIBLE,
            ModelProvider.OPENAI,
        )
    )
    ollama_native: OllamaNativeConfig = field(default_factory=OllamaNativeConfig)
    ollama_openai_compatible: OllamaOpenAICompatibleConfig = field(
        default_factory=OllamaOpenAICompatibleConfig
    )
    openai: OpenAICompatibleConfig = field(default_factory=OpenAICompatibleConfig)
    codex_openai_compatible: CodexOpenAICompatibleConfig = field(
        default_factory=CodexOpenAICompatibleConfig
    )
    search: SearchConfig = field(default_factory=SearchConfig)
    retry_attempts: int = 1

    def provider_model(self, provider: ModelProvider | None = None) -> str:
        """Return the configured model for a provider."""

        selected = self.primary_provider if provider is None else provider
        if selected is ModelProvider.OLLAMA_NATIVE:
            return self.ollama_native.model
        if selected is ModelProvider.OLLAMA_OPENAI_COMPATIBLE:
            return self.ollama_openai_compatible.model
        if selected is ModelProvider.CODEX_OPENAI_COMPATIBLE:
            return self.codex_openai_compatible.model
        return self.openai.model


def _get(env: Mapping[str, str], key: str, default: str) -> str:
    value = env.get(key, "")
    return value if value else default


def _get_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc


def _provider(raw: str, *, key: str) -> ModelProvider:
    try:
        return ModelProvider(raw)
    except ValueError as exc:
        allowed = ", ".join(provider.value for provider in ModelProvider)
        raise ValueError(f"{key} must be one of: {allowed}") from exc


def _fallback_order(env: Mapping[str, str]) -> tuple[ModelProvider, ...]:
    raw = env.get("DEEP_RESEARCH_FALLBACK_ORDER", "")
    if not raw:
        return (ModelProvider.CODEX_OPENAI_COMPATIBLE, ModelProvider.OPENAI)
    return tuple(
        _provider(part.strip(), key="DEEP_RESEARCH_FALLBACK_ORDER")
        for part in raw.split(",")
        if part.strip()
    )


def load_config(env: Mapping[str, str] | None = None) -> AgentConfig:
    """Load configuration from an environment-like mapping.

    Passing an explicit mapping keeps tests and CLI help import-safe; production
    callers default to ``os.environ``. This function performs no network or file
    system access.
    """

    source = os.environ if env is None else env
    return AgentConfig(
        primary_provider=_provider(
            _get(source, "DEEP_RESEARCH_MODEL_PROVIDER", ModelProvider.OLLAMA_NATIVE.value),
            key="DEEP_RESEARCH_MODEL_PROVIDER",
        ),
        fallback_order=_fallback_order(source),
        ollama_native=OllamaNativeConfig(
            base_url=_get(source, "OLLAMA_NATIVE_BASE_URL", "https://ollama.com/api"),
            model=_get(source, "OLLAMA_NATIVE_MODEL", "deepseek-v4-pro:cloud"),
            api_key=_get(source, "OLLAMA_API_KEY", ""),
        ),
        ollama_openai_compatible=OllamaOpenAICompatibleConfig(
            base_url=_get(source, "OLLAMA_OPENAI_BASE_URL", "http://localhost:11434/v1"),
            model=_get(source, "OLLAMA_OPENAI_MODEL", "deepseek-v4-pro:cloud"),
            api_key=_get(source, "OLLAMA_OPENAI_API_KEY", "ollama"),
        ),
        openai=OpenAICompatibleConfig(
            base_url=_get(source, "OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=_get(source, "OPENAI_MODEL", "gpt-4.1-mini"),
            api_key=_get(source, "OPENAI_API_KEY", ""),
        ),
        codex_openai_compatible=CodexOpenAICompatibleConfig(
            base_url=_get(source, "CODEX_OPENAI_BASE_URL", ""),
            model=_get(source, "CODEX_OPENAI_MODEL", ""),
            api_key=_get(source, "CODEX_OPENAI_API_KEY", ""),
        ),
        search=SearchConfig(
            max_results=_get_int(source, "DEEP_RESEARCH_MAX_SEARCH_RESULTS", 5),
            timeout_seconds=_get_int(source, "DEEP_RESEARCH_SEARCH_TIMEOUT_SECONDS", 10),
        ),
        retry_attempts=_get_int(source, "DEEP_RESEARCH_MODEL_RETRY_ATTEMPTS", 1),
    )
