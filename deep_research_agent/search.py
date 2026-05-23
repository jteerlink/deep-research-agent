"""Package-level accessors for the legacy async_multi_search module."""

from __future__ import annotations

from async_multi_search import (  # noqa: F401
    AllProvidersFailedError,
    AsyncMultiProviderSearch,
    BraveProvider,
    DuckDuckGoProvider,
    ExaProvider,
    SearchProvider,
    SearchResult,
    SerperProvider,
    TavilyProvider,
    web_search,
)

__all__ = [
    "AllProvidersFailedError",
    "AsyncMultiProviderSearch",
    "BraveProvider",
    "DuckDuckGoProvider",
    "ExaProvider",
    "SearchProvider",
    "SearchResult",
    "SerperProvider",
    "TavilyProvider",
    "web_search",
]
