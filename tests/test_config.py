from __future__ import annotations

import pytest

from deep_research_agent.config import ModelProvider, load_config


def test_default_config_is_ollama_native() -> None:
    config = load_config({})

    assert config.primary_provider is ModelProvider.OLLAMA_NATIVE
    assert config.ollama_native.base_url == "https://ollama.com/api"
    assert config.ollama_openai.base_url == "http://localhost:11434/v1"
    assert config.primary_model == "deepseek-v4-pro:cloud"


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


def test_deep_research_provider_aliases_map_to_root_config() -> None:
    config = load_config({"DEEP_RESEARCH_MODEL_PROVIDER": "codex_openai_compatible"})

    assert config.primary_provider is ModelProvider.CODEX


def test_dotenv_values_are_loaded_without_overriding_explicit_env(tmp_path, monkeypatch) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("DRA_OLLAMA_MODEL=from-dotenv\nOPENAI_API_KEY=secret-from-dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DRA_OLLAMA_MODEL", "from-shell")

    config = load_config()

    assert config.ollama_native.model == "from-shell"
    assert config.openai.api_key == "secret-from-dotenv"


def test_deep_research_provider_takes_precedence_over_legacy_alias() -> None:
    config = load_config(
        {
            "DEEP_RESEARCH_MODEL_PROVIDER": "openai",
            "DRA_PRIMARY_PROVIDER": "codex",
            "OLLAMA_NATIVE_MODEL": "canonical-model",
            "DRA_OLLAMA_MODEL": "legacy-model",
        }
    )

    assert config.primary_provider is ModelProvider.OPENAI
    assert config.ollama_native.model == "canonical-model"


def test_deep_research_search_knobs_take_precedence_over_legacy_aliases() -> None:
    config = load_config(
        {
            "DEEP_RESEARCH_MAX_SEARCH_RESULTS": "9",
            "DRA_MAX_SEARCH_RESULTS": "2",
            "DEEP_RESEARCH_SEARCH_TIMEOUT_SECONDS": "30",
            "DRA_SEARCH_TIMEOUT_SECONDS": "5",
        }
    )

    assert config.search.max_results == 9
    assert config.search.timeout_seconds == 30
