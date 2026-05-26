from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from deep_research_agent.ui import (
    build_prospect_directive,
    env_file_overlay,
    provider_env_from_env_file,
    provider_env_overlay,
    result_preview,
    temporary_env,
)


def test_provider_env_overlay_keeps_only_non_empty_supported_keys() -> None:
    overlay = provider_env_overlay(
        {
            "FIRECRAWL_API_KEY": " fc-test ",
            "SERPER_API_KEY": "",
            "OPENAI_API_KEY": "not-a-search-key",
        }
    )

    assert overlay == {"FIRECRAWL_API_KEY": "fc-test"}


def test_provider_env_from_env_file_preserves_shell_precedence(tmp_path, monkeypatch) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "TAVILY_API_KEY=from-dotenv\nFIRECRAWL_API_KEY=from-dotenv-firecrawl\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TAVILY_API_KEY", "from-shell")

    overlay = provider_env_from_env_file(dotenv)

    assert overlay == {
        "TAVILY_API_KEY": "from-shell",
        "FIRECRAWL_API_KEY": "from-dotenv-firecrawl",
    }


def test_env_file_overlay_loads_non_exported_runtime_values(tmp_path, monkeypatch) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "TAVILY_API_KEY=from-dotenv\n"
        "SSL_CERT_FILE=/tmp/corp-ca.pem\n"
        "REQUESTS_CA_BUNDLE=/tmp/requests-ca.pem\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TAVILY_API_KEY", "from-shell")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)

    overlay = env_file_overlay(dotenv)

    assert overlay == {
        "SSL_CERT_FILE": "/tmp/corp-ca.pem",
        "REQUESTS_CA_BUNDLE": "/tmp/requests-ca.pem",
    }


def test_build_prospect_directive_includes_business_target_context() -> None:
    assert build_prospect_directive(
        "dental practices",
        "Dallas-Fort Worth",
        "patient reactivation opportunity",
    ) == (
        "industry: dental practices\n"
        "geography: Dallas-Fort Worth\n"
        "criteria: patient reactivation opportunity"
    )


def test_temporary_env_restores_secret_values(monkeypatch) -> None:
    monkeypatch.setenv("FIRECRAWL_API_KEY", "original")

    with temporary_env({"FIRECRAWL_API_KEY": "session-only", "SERPER_API_KEY": "serper"}):
        assert os.environ["FIRECRAWL_API_KEY"] == "session-only"
        assert os.environ["SERPER_API_KEY"] == "serper"

    assert os.environ["FIRECRAWL_API_KEY"] == "original"
    assert "SERPER_API_KEY" not in os.environ


def test_result_preview_reads_markdown_and_limits_lists(tmp_path) -> None:
    markdown_path = tmp_path / "artifact.md"
    markdown_path.write_text("# Artifact\n\nPreview text", encoding="utf-8")
    state = {
        "thread_id": "ui-thread",
        "status": "completed",
        "warnings": ["warn"],
        "events": [{"event": f"e{i}"} for i in range(30)],
        "evidence": [{"id": str(i)} for i in range(12)],
        "prospect_targets": [{"name": str(i)} for i in range(12)],
        "artifact_paths": {"markdown": str(markdown_path)},
    }

    preview = result_preview(state)

    assert preview["thread_id"] == "ui-thread"
    assert preview["event_count"] == 30
    assert len(preview["events"]) == 25
    assert len(preview["evidence"]) == 10
    assert len(preview["prospect_targets"]) == 10
    assert preview["markdown_preview"].startswith("# Artifact")


def test_ui_imports_when_loaded_as_streamlit_script() -> None:
    path = Path(__file__).resolve().parents[1] / "deep_research_agent" / "ui.py"
    spec = importlib.util.spec_from_file_location("streamlit_script_ui", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    assert module.DEFAULT_CHECKPOINT_DIR == ".deep_research_agent/checkpoints"
