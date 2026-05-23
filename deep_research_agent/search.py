"""Package-native access to the legacy async multi-provider search helpers.

``async_multi_search.py`` remains importable for existing users. New package code
can import from ``deep_research_agent.search`` while sharing the same
implementation and behavior.
"""

from __future__ import annotations

from async_multi_search import (
    AllProvidersFailedError,
    AsyncMultiProviderSearch,
    SearchProvider,
    SearchResult,
    web_search,
)

__all__ = [
    "AllProvidersFailedError",
    "AsyncMultiProviderSearch",
    "SearchProvider",
    "SearchResult",
    "web_search",
]
