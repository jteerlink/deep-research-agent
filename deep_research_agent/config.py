"""Environment-driven configuration for the deep research agent foundation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping


class ModelProvider(StrEnum):
    """Supported primary model provider modes."""

    OLLAMA_NATIVE = "ollama_native"
    OLLAMA_OPENAI = "ollama_openai"
    OPENAI = "openai"
    CODEX = "codex"


@dataclass(frozen=True)
class OllamaNativeConfig:
    """Ollama native API configuration.

    This is intentionally distinct from Ollama's OpenAI-compatible `/v1` API.
    """

    base_url: str = "http://localhost:11434"
    model: str = "llama3.1"


@dataclass(frozen=True)
class OllamaOpenAIConfig:
    """Ollama OpenAI-compatible API configuration."""

    base_url: str = "http://localhost:11434/v1"
    model: str = "llama3.1"
    api_key: str = "ollama"


@dataclass(frozen=True)
class OpenAIConfig:
    """OpenAI fallback configuration."""

    api_key: str = ""
    model: str = "gpt-4.1-mini"
    base_url: str = "https://api.openai.com/v1"


@dataclass(frozen=True)
class CodexConfig:
    """Codex/OpenAI-compatible fallback configuration."""

    api_key: str = ""
    model: str = "gpt-5.4-mini"
    base_url: str = "https://api.openai.com/v1"


@dataclass(frozen=True)
class SearchConfig:
    """Search runtime configuration shared with async_multi_search.py defaults."""

    max_results: int = 5
    timeout_seconds: int = 10


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    primary_provider: ModelProvider
    ollama_native: OllamaNativeConfig
    ollama_openai: OllamaOpenAIConfig
    openai: OpenAIConfig
    codex: CodexConfig
    search: SearchConfig

    @property
    def primary_model(self) -> str:
        """Return the configured model for the selected primary provider."""

        if self.primary_provider is ModelProvider.OLLAMA_NATIVE:
            return self.ollama_native.model
        if self.primary_provider is ModelProvider.OLLAMA_OPENAI:
            return self.ollama_openai.model
        if self.primary_provider is ModelProvider.CODEX:
            return self.codex.model
        return self.openai.model


def _get(env: Mapping[str, str], key: str, default: str) -> str:
    value = env.get(key, "")
    return value if value != "" else default


def _get_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key, "")
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc


def _provider(value: str) -> ModelProvider:
    try:
        return ModelProvider(value)
    except ValueError as exc:
        allowed = ", ".join(provider.value for provider in ModelProvider)
        raise ValueError(f"DRA_PRIMARY_PROVIDER must be one of: {allowed}") from exc


def load_config(env: Mapping[str, str] | None = None) -> AppConfig:
    """Load configuration from environment-like mapping.

    Passing a mapping makes tests deterministic while production callers default
    to `os.environ`.
    """

    source = os.environ if env is None else env
    return AppConfig(
        primary_provider=_provider(_get(source, "DRA_PRIMARY_PROVIDER", "ollama_native")),
        ollama_native=OllamaNativeConfig(
            base_url=_get(source, "DRA_OLLAMA_BASE_URL", "http://localhost:11434"),
            model=_get(source, "DRA_OLLAMA_MODEL", "llama3.1"),
        ),
        ollama_openai=OllamaOpenAIConfig(
            base_url=_get(source, "DRA_OLLAMA_OPENAI_BASE_URL", "http://localhost:11434/v1"),
            model=_get(source, "DRA_OLLAMA_OPENAI_MODEL", "llama3.1"),
            api_key=_get(source, "DRA_OLLAMA_OPENAI_API_KEY", "ollama"),
        ),
        openai=OpenAIConfig(
            api_key=_get(source, "OPENAI_API_KEY", ""),
            model=_get(source, "OPENAI_MODEL", "gpt-4.1-mini"),
            base_url=_get(source, "OPENAI_BASE_URL", "https://api.openai.com/v1"),
        ),
        codex=CodexConfig(
            api_key=_get(source, "CODEX_API_KEY", ""),
            model=_get(source, "CODEX_MODEL", "gpt-5.4-mini"),
            base_url=_get(source, "CODEX_BASE_URL", "https://api.openai.com/v1"),
        ),
        search=SearchConfig(
            max_results=_get_int(source, "DRA_MAX_SEARCH_RESULTS", 5),
            timeout_seconds=_get_int(source, "DRA_SEARCH_TIMEOUT_SECONDS", 10),
        ),
    )
