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


def test_cli_config_json() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "deep_research_agent", "config", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"primary_provider"' in result.stdout
    assert '"ollama_native"' in result.stdout
