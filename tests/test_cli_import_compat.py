"""Regression coverage for the legacy import and CLI compatibility surface."""

from __future__ import annotations

import subprocess
import sys


def test_legacy_async_multi_search_imports() -> None:
    from async_multi_search import SearchResult, web_search

    assert SearchResult.__name__ == "SearchResult"
    assert callable(web_search)


def test_package_search_reexports_legacy_symbols() -> None:
    import async_multi_search
    import deep_research_agent
    from deep_research_agent.search import SearchResult, web_search

    assert web_search is async_multi_search.web_search
    assert SearchResult is async_multi_search.SearchResult
    assert deep_research_agent.web_search is async_multi_search.web_search


def test_package_cli_help_does_not_import_provider_runtime() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "deep_research_agent.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "usage: deep-research-agent" in completed.stdout
    assert "search" in completed.stdout
    assert completed.stderr == ""


def test_legacy_module_help_does_not_run_search() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "async_multi_search", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "usage: async_multi_search" in completed.stdout
    assert "--max-results" in completed.stdout
    assert completed.stderr == ""
