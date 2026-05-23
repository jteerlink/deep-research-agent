"""
async_multi_search.py — async sequential-fallback web search for agents.

Same priority-chain behavior as the sync version, but awaitable and with
one retry on timeout before falling through to the next provider:

  - any provider that throws OR returns nothing -> move to the next
  - a timeout specifically -> retry that provider ONCE, then move on
  - providers with no API key set -> skipped automatically

Plug-in usage:
    from async_multi_search import web_search
    results = await web_search("mixture-of-experts routing tricks")

Keys (set whichever you have):
    TAVILY_API_KEY, EXA_API_KEY, BRAVE_API_KEY, SERPER_API_KEY
DuckDuckGo needs no key and sits last as the always-available fallback
(pip install httpx ddgs).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10  # seconds per attempt
DEFAULT_MAX_RESULTS = 5


@dataclass
class SearchResult:
    """Normalized shape every provider maps into."""

    title: str
    url: str
    content: str  # snippet OR full text, depending on provider
    score: float | None = None
    provider: str = ""


class AllProvidersFailedError(Exception):
    """Raised when every configured provider in the chain failed."""


# --------------------------------------------------------------------------- #
# Provider base class
# --------------------------------------------------------------------------- #
class SearchProvider(ABC):
    name: str = "base"
    env_key: str | None = None  # required env var; None = no key needed

    @property
    def configured(self) -> bool:
        return self.env_key is None or bool(os.getenv(self.env_key))

    @abstractmethod
    async def search(
        self, client: httpx.AsyncClient, query: str, max_results: int
    ) -> list[SearchResult]: ...


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #
class TavilyProvider(SearchProvider):
    name = "tavily"
    env_key = "TAVILY_API_KEY"

    async def search(self, client, query, max_results):
        resp = await client.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {os.environ[self.env_key]}"},
            json={"query": query, "max_results": max_results},
        )
        resp.raise_for_status()
        return [
            SearchResult(
                r.get("title", ""),
                r.get("url", ""),
                r.get("content", ""),
                r.get("score"),
                self.name,
            )
            for r in resp.json().get("results", [])
        ]


class ExaProvider(SearchProvider):
    name = "exa"
    env_key = "EXA_API_KEY"

    async def search(self, client, query, max_results):
        resp = await client.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": os.environ[self.env_key]},
            json={"query": query, "numResults": max_results, "contents": {"text": True}},
        )
        resp.raise_for_status()
        return [
            SearchResult(
                r.get("title", ""),
                r.get("url", ""),
                r.get("text", "") or "",
                r.get("score"),
                self.name,
            )
            for r in resp.json().get("results", [])
        ]


class BraveProvider(SearchProvider):
    name = "brave"
    env_key = "BRAVE_API_KEY"

    async def search(self, client, query, max_results):
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": os.environ[self.env_key],
            },
            params={"q": query, "count": max_results},
        )
        resp.raise_for_status()
        hits = resp.json().get("web", {}).get("results", [])
        return [
            SearchResult(
                h.get("title", ""), h.get("url", ""), h.get("description", ""), None, self.name
            )
            for h in hits
        ]


class SerperProvider(SearchProvider):
    name = "serper"
    env_key = "SERPER_API_KEY"

    async def search(self, client, query, max_results):
        resp = await client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": os.environ[self.env_key], "Content-Type": "application/json"},
            json={"q": query, "num": max_results},
        )
        resp.raise_for_status()
        return [
            SearchResult(
                h.get("title", ""), h.get("link", ""), h.get("snippet", ""), None, self.name
            )
            for h in resp.json().get("organic", [])
        ]


class DuckDuckGoProvider(SearchProvider):
    """No key, no quota — guaranteed-available last resort. The ddgs lib is
    sync, so we run it in a thread to avoid blocking the event loop."""

    name = "duckduckgo"
    env_key = None

    def _sync_search(self, query, max_results):
        try:
            from ddgs import DDGS  # current package
        except ImportError:
            from duckduckgo_search import DDGS  # legacy name
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    async def search(self, client, query, max_results):
        hits = await asyncio.to_thread(self._sync_search, query, max_results)
        return [
            SearchResult(h.get("title", ""), h.get("href", ""), h.get("body", ""), None, self.name)
            for h in hits
        ]


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
class AsyncMultiProviderSearch:
    def __init__(self, providers=None, treat_empty_as_failure=True, timeout=DEFAULT_TIMEOUT):
        # Order = priority. Keyless DuckDuckGo sits last as a safety net.
        self.providers = providers or [
            TavilyProvider(),
            BraveProvider(),
            ExaProvider(),
            SerperProvider(),
            DuckDuckGoProvider(),
        ]
        self.treat_empty_as_failure = treat_empty_as_failure
        self.timeout = timeout

    async def _attempt(self, client, provider, query, max_results):
        """Run one provider, retrying exactly once on timeout."""
        try:
            return await provider.search(client, query, max_results)
        except httpx.TimeoutException:
            logger.info("%s timed out — retrying once", provider.name)
            return await provider.search(client, query, max_results)

    async def search(self, query, max_results=DEFAULT_MAX_RESULTS):
        errors = []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for p in self.providers:
                if not p.configured:
                    logger.info("skip %s (no key)", p.name)
                    continue
                try:
                    results = await self._attempt(client, p, query, max_results)
                except Exception as e:  # incl. a 2nd consecutive timeout
                    logger.warning("%s failed: %s", p.name, e)
                    errors.append((p.name, str(e)))
                    continue
                if self.treat_empty_as_failure and not results:
                    logger.warning("%s returned no results", p.name)
                    errors.append((p.name, "empty"))
                    continue
                logger.info("%s ok (%d results)", p.name, len(results))
                return results
        raise AllProvidersFailedError(f"all providers failed: {errors}")


# Module-level default so the plug-in function is one import away.
_default = AsyncMultiProviderSearch()


async def web_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[SearchResult]:
    """Drop-in awaitable entry point for your agent/tool."""
    return await _default.search(query, max_results)


def build_parser():
    """Build a small compatibility CLI for the legacy module path."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="async_multi_search",
        description="Run the async multi-provider web search fallback chain.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="what is retrieval augmented generation",
        help="search query to run; defaults to the original demo query",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_MAX_RESULTS,
        help="maximum number of results to return from the first successful provider",
    )
    return parser


async def _run_cli(query: str, max_results: int) -> int:
    """Execute a search for the compatibility CLI."""
    for r in await web_search(query, max_results=max_results):
        print(f"[{r.provider}] {r.title}\n  {r.url}\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the legacy module CLI without breaking ``--help`` behavior."""
    logging.basicConfig(level=logging.INFO)
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run_cli(args.query, args.max_results))


if __name__ == "__main__":
    raise SystemExit(main())
