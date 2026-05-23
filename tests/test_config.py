from __future__ import annotations

import pytest

from deep_research_agent.config import ModelProvider, load_config


def test_default_config_is_ollama_native() -> None:
    config = load_config({})

    assert config.primary_provider is ModelProvider.OLLAMA_NATIVE
    assert config.ollama_native.base_url == "http://localhost:11434"
    assert config.ollama_openai.base_url == "http://localhost:11434/v1"
    assert config.primary_model == "llama3.1"


def test_openai_and_codex_fallbacks_are_distinct() -> None:
    config = load_config(
        {
            "DRA_PRIMARY_PROVIDER": "codex",
            "OPENAI_API_KEY": "openai-key",
            "OPENAI_MODEL": "gpt-openai",
            "CODEX_API_KEY": "codex-key",
            "CODEX_MODEL": "gpt-codex",
        }
    )

    assert config.primary_provider is ModelProvider.CODEX
    assert config.openai.api_key == "openai-key"
    assert config.codex.api_key == "codex-key"
    assert config.primary_model == "gpt-codex"


def test_invalid_provider_reports_allowed_values() -> None:
    with pytest.raises(ValueError, match="ollama_native"):
        load_config({"DRA_PRIMARY_PROVIDER": "hosted-ui"})
