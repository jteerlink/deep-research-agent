"""Regression checks for the local-first env/docs foundation skeleton."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_env_example_documents_provider_split_and_search_keys() -> None:
    env_example = (ROOT / ".env.example").read_text()

    required_keys = {
        "DEEP_RESEARCH_MODEL_PROVIDER",
        "OLLAMA_NATIVE_BASE_URL",
        "OLLAMA_NATIVE_MODEL",
        "OLLAMA_OPENAI_BASE_URL",
        "OLLAMA_OPENAI_MODEL",
        "OLLAMA_OPENAI_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "CODEX_OPENAI_BASE_URL",
        "CODEX_OPENAI_API_KEY",
        "CODEX_OPENAI_MODEL",
        "TAVILY_API_KEY",
        "EXA_API_KEY",
        "SERPER_API_KEY",
        "FIRECRAWL_API_KEY",
        "YDC_API_KEY",
    }

    for key in required_keys:
        assert f"{key}=" in env_example

    assert "Native Ollama API" in env_example
    assert "Ollama OpenAI-compatible API" in env_example
    assert "Codex/OpenAI-compatible fallback" in env_example


def test_legacy_langgraph_docs_and_entrypoint_are_absent() -> None:
    assert not (ROOT / "langgraph.json").exists()
    assert "LangGraph" not in (ROOT / "docs" / "foundation-configuration.md").read_text()
    assert "langgraph dev" not in (ROOT / "docs" / "development.md").read_text()


def test_foundation_docs_preserve_scope_and_import_compatibility() -> None:
    foundation_doc = (ROOT / "docs" / "foundation-configuration.md").read_text()
    development_doc = (ROOT / "docs" / "development.md").read_text()

    assert "from async_multi_search import web_search" in foundation_doc
    assert "hosted ui" in foundation_doc.lower()
    assert "no model zoo" in foundation_doc.lower()
    assert "no database" in development_doc.lower()
    assert ".omx/ultragoal" not in foundation_doc
    assert ".omx/ultragoal" not in development_doc


def test_ui_extra_includes_keyless_search_fallback_dependency() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())

    assert "ddgs>=9.0" in pyproject["project"]["optional-dependencies"]["ui"]
