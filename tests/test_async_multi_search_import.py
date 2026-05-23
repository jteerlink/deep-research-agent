import importlib


def test_legacy_async_multi_search_import_exports_public_api():
    module = importlib.import_module("async_multi_search")

    assert module.SearchResult("title", "url", "body").provider == ""
    assert callable(module.web_search)
    assert module.AllProvidersFailedError.__name__ == "AllProvidersFailedError"
    assert [provider.name for provider in module.AsyncMultiProviderSearch().providers] == [
        "tavily",
        "brave",
        "exa",
        "serper",
        "duckduckgo",
    ]


def test_search_provider_configured_uses_only_its_declared_env_key(monkeypatch):
    module = importlib.import_module("async_multi_search")
    providers = module.AsyncMultiProviderSearch().providers

    for key in ("TAVILY_API_KEY", "BRAVE_API_KEY", "EXA_API_KEY", "SERPER_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    assert {provider.name: provider.configured for provider in providers} == {
        "tavily": False,
        "brave": False,
        "exa": False,
        "serper": False,
        "duckduckgo": True,
    }

    monkeypatch.setenv("BRAVE_API_KEY", "test-token")
    assert {provider.name: provider.configured for provider in providers} == {
        "tavily": False,
        "brave": True,
        "exa": False,
        "serper": False,
        "duckduckgo": True,
    }
