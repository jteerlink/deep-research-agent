from __future__ import annotations

import subprocess
import sys


def test_module_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "deep_research_agent", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Local-first deep research agent foundation" in result.stdout
    assert "config" in result.stdout
    assert "run" in result.stdout
    assert "resume" in result.stdout
    assert "inspect" in result.stdout


def test_cli_config_json() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "deep_research_agent", "config", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"primary_provider"' in result.stdout
    assert '"ollama_native"' in result.stdout


def test_legacy_module_cli_help_does_not_run_demo() -> None:
    result = subprocess.run(
        [sys.executable, "async_multi_search.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Run the async multi-provider search fallback chain" in result.stdout
    assert "--max-results" in result.stdout
    assert "AllProvidersFailedError" not in result.stderr


def test_cli_config_json_redacts_api_keys() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "deep_research_agent", "config", "--json"],
        check=True,
        capture_output=True,
        text=True,
        env={"OPENAI_API_KEY": "secret-openai", "CODEX_API_KEY": "secret-codex"},
    )

    assert "secret-openai" not in result.stdout
    assert "secret-codex" not in result.stdout
    assert "<redacted>" in result.stdout
