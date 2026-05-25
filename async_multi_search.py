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
    TAVILY_API_KEY, EXA_API_KEY, SERPER_API_KEY, FIRECRAWL_API_KEY,
    YDC_API_KEY
DuckDuckGo needs no key and sits last as the always-available fallback
(pip install httpx ddgs).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 10  # seconds per attempt
DEFAULT_MAX_RESULTS = 5
CA_BUNDLE_ENV_KEYS = ("DEEP_RESEARCH_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")
VerifySetting: TypeAlias = bool | str


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


def _explicit_ca_bundle(environ: Mapping[str, str] = os.environ) -> str | None:
    for key in CA_BUNDLE_ENV_KEYS:
        value = environ.get(key, "").strip()
        if value:
            return value
    return None


def _macos_keychain_ca_bundle(
    environ: Mapping[str, str] = os.environ,
) -> str | None:
    if sys.platform != "darwin":
        return None
    if environ.get("DEEP_RESEARCH_DISABLE_MACOS_KEYCHAIN_CA", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        return None

    cache_dir = Path(environ.get("DEEP_RESEARCH_CACHE_DIR", ".deep_research_agent/cache"))
    bundle_path = cache_dir / "macos-system-ca-bundle.pem"
    if bundle_path.exists() and bundle_path.stat().st_size > 0:
        return str(bundle_path)

    keychains = [
        "/System/Library/Keychains/SystemRootCertificates.keychain",
        "/Library/Keychains/System.keychain",
    ]
    try:
        result = subprocess.run(
            ["security", "find-certificate", "-a", "-p", *keychains],
            check=True,
            capture_output=True,
            text=True,
        )
        if "BEGIN CERTIFICATE" not in result.stdout:
            return None
        cache_dir.mkdir(parents=True, exist_ok=True)
        bundle_path.write_text(result.stdout, encoding="utf-8")
    except Exception as exc:
        logger.debug("macOS keychain CA export failed: %s", exc)
        return None
    return str(bundle_path)


def _default_verify_setting() -> VerifySetting:
    return _explicit_ca_bundle() or _macos_keychain_ca_bundle() or True


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


class FirecrawlProvider(SearchProvider):
    """Firecrawl v2 search adapter.

    The first-pass integration intentionally requests SERP-style web results
    only. It does not opt into Firecrawl scrape formats, so normalized content
    remains snippet/description evidence rather than a page read.
    """

    name = "firecrawl"
    env_key = "FIRECRAWL_API_KEY"

    async def search(self, client, query, max_results):
        resp = await client.post(
            "https://api.firecrawl.dev/v2/search",
            headers={
                "Authorization": f"Bearer {os.environ[self.env_key]}",
                "Content-Type": "application/json",
            },
            json={"query": query, "limit": max_results, "sources": ["web"]},
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("success") is False:
            raise RuntimeError(f"firecrawl search failed: {payload}")
        hits = payload.get("data", {}).get("web", [])
        return [
            SearchResult(
                h.get("title", ""),
                h.get("url", ""),
                h.get("description", h.get("snippet", "")) or "",
                None,
                self.name,
            )
            for h in hits
        ]


class YdcProvider(SearchProvider):
    """You.com Developer Cloud Search API adapter."""

    name = "ydc"
    env_key = "YDC_API_KEY"

    async def search(self, client, query, max_results):
        resp = await client.post(
            "https://ydc-index.io/v1/search",
            headers={
                "X-API-Key": os.environ[self.env_key],
                "Content-Type": "application/json",
            },
            json={"query": query, "count": max_results},
        )
        resp.raise_for_status()
        hits = resp.json().get("results", {}).get("web", [])
        return [
            SearchResult(
                h.get("title", ""),
                h.get("url", ""),
                h.get("description", "") or "",
                None,
                self.name,
            )
            for h in hits
        ]


class DuckDuckGoProvider(SearchProvider):
    """No key, no quota — guaranteed-available last resort. The ddgs lib is
    sync, so we run it in a thread to avoid blocking the event loop."""

    name = "duckduckgo"
    env_key = None

    def _sync_search(self, query, max_results):
        try:
            from ddgs import DDGS  # noqa: I001
        except ImportError:
            from duckduckgo_search import DDGS  # type: ignore[import-not-found]  # noqa: I001
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
    def __init__(
        self,
        providers=None,
        treat_empty_as_failure=True,
        timeout=DEFAULT_TIMEOUT,
        verify: VerifySetting | None = None,
    ):
        # Order = priority. Keyless DuckDuckGo sits last as a safety net.
        self.providers = providers or [
            TavilyProvider(),
            ExaProvider(),
            SerperProvider(),
            FirecrawlProvider(),
            YdcProvider(),
            DuckDuckGoProvider(),
        ]
        self.treat_empty_as_failure = treat_empty_as_failure
        self.timeout = timeout
        self.verify = verify

    async def _attempt(self, client, provider, query, max_results):
        """Run one provider, retrying exactly once on timeout."""
        try:
            return await provider.search(client, query, max_results)
        except httpx.TimeoutException:
            logger.info("%s timed out — retrying once", provider.name)
            return await provider.search(client, query, max_results)

    async def search(self, query, max_results=DEFAULT_MAX_RESULTS):
        errors = []
        verify = _default_verify_setting() if self.verify is None else self.verify
        async with httpx.AsyncClient(timeout=self.timeout, verify=verify) as client:
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


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the async multi-provider search fallback chain.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="what is retrieval augmented generation",
        help="Search query to run when executing this module directly.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_MAX_RESULTS,
        help="Maximum results requested from the first successful provider.",
    )
    return parser


async def _main() -> int:
    logging.basicConfig(level=logging.INFO)
    args = _build_cli_parser().parse_args()
    for r in await web_search(args.query, max_results=args.max_results):
        print(f"[{r.provider}] {r.title}\n  {r.url}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
