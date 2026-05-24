"""Regression checks for the G001 env/LangGraph/docs foundation skeleton."""

from __future__ import annotations

import json
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
        "LANGGRAPH_HOST",
        "LANGGRAPH_PORT",
    }

    for key in required_keys:
        assert f"{key}=" in env_example

    assert "Native Ollama API" in env_example
    assert "Ollama OpenAI-compatible API" in env_example
    assert "Codex/OpenAI-compatible fallback" in env_example


def test_langgraph_config_points_to_package_graph_entrypoint() -> None:
    config = json.loads((ROOT / "langgraph.json").read_text())

    assert config["dependencies"] == ["."]
    assert config["env"] == ".env"
    assert config["graphs"] == {"deep_research_agent": "./src/deep_research_agent/graph.py:graph"}


def test_foundation_docs_preserve_scope_and_import_compatibility() -> None:
    foundation_doc = (ROOT / "docs" / "foundation-configuration.md").read_text()
    development_doc = (ROOT / "docs" / "development.md").read_text()

    assert "from async_multi_search import web_search" in foundation_doc
    assert "hosted ui" in foundation_doc.lower()
    assert "no model zoo" in foundation_doc.lower()
    assert "no database" in development_doc.lower()
    assert ".omx/ultragoal" not in foundation_doc
    assert ".omx/ultragoal" not in development_doc
