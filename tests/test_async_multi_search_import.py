import asyncio
import importlib

import httpx


def test_legacy_async_multi_search_import_exports_public_api():
    module = importlib.import_module("async_multi_search")

    assert module.SearchResult("title", "url", "body").provider == ""
    assert callable(module.web_search)
    assert module.AllProvidersFailedError.__name__ == "AllProvidersFailedError"
    assert [provider.name for provider in module.AsyncMultiProviderSearch().providers] == [
        "tavily",
        "exa",
        "serper",
        "firecrawl",
        "ydc",
        "duckduckgo",
    ]


def test_search_provider_configured_uses_only_its_declared_env_key(monkeypatch):
    module = importlib.import_module("async_multi_search")
    providers = module.AsyncMultiProviderSearch().providers

    for key in (
        "TAVILY_API_KEY",
        "EXA_API_KEY",
        "SERPER_API_KEY",
        "FIRECRAWL_API_KEY",
        "YDC_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    assert {provider.name: provider.configured for provider in providers} == {
        "tavily": False,
        "exa": False,
        "serper": False,
        "firecrawl": False,
        "ydc": False,
        "duckduckgo": True,
    }

    monkeypatch.setenv("YDC_API_KEY", "test-token")
    assert {provider.name: provider.configured for provider in providers} == {
        "tavily": False,
        "exa": False,
        "serper": False,
        "firecrawl": False,
        "ydc": True,
        "duckduckgo": True,
    }


def test_firecrawl_provider_maps_v2_web_results_without_scraped_content(monkeypatch):
    module = importlib.import_module("async_multi_search")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.firecrawl.dev/v2/search"
        assert request.headers["authorization"] == "Bearer fc-test"
        assert request.read() == b'{"query":"find prospects","limit":2,"sources":["web"]}'
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "web": [
                        {
                            "title": "Acme",
                            "url": "https://example.com/acme",
                            "description": "Search description",
                            "markdown": "# Should not be used",
                        },
                        {
                            "title": "Beta",
                            "url": "https://example.com/beta",
                            "snippet": "Fallback snippet",
                        },
                    ]
                },
            },
            request=request,
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await module.FirecrawlProvider().search(client, "find prospects", 2)

    results = asyncio.run(run())

    assert [(result.provider, result.title, result.content) for result in results] == [
        ("firecrawl", "Acme", "Search description"),
        ("firecrawl", "Beta", "Fallback snippet"),
    ]


def test_ydc_provider_maps_web_results(monkeypatch):
    module = importlib.import_module("async_multi_search")
    monkeypatch.setenv("YDC_API_KEY", "ydc-test")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://ydc-index.io/v1/search"
        assert request.headers["x-api-key"] == "ydc-test"
        assert request.read() == b'{"query":"find prospects","count":2}'
        return httpx.Response(
            200,
            json={
                "results": {
                    "web": [
                        {
                            "title": "Acme",
                            "url": "https://example.com/acme",
                            "description": "YDC description",
                        }
                    ]
                }
            },
            request=request,
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await module.YdcProvider().search(client, "find prospects", 2)

    results = asyncio.run(run())

    assert [(result.provider, result.title, result.content) for result in results] == [
        ("ydc", "Acme", "YDC description")
    ]
