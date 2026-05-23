from __future__ import annotations


def test_legacy_async_multi_search_import_compatibility() -> None:
    import async_multi_search

    assert async_multi_search.web_search
    assert async_multi_search.SearchResult(title="t", url="u", content="c").provider == ""


def test_package_imports() -> None:
    import deep_research_agent
    from deep_research_agent.search import web_search

    assert deep_research_agent.load_config
    assert web_search
