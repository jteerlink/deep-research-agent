"""Deep Research Agent foundation package.

The initial package surface keeps the historical ``async_multi_search`` module
available while adding package-native imports for future LangGraph agents.
"""

from __future__ import annotations

__version__ = "0.1.0"

_SEARCH_EXPORTS = {
    "AllProvidersFailedError",
    "AsyncMultiProviderSearch",
    "SearchProvider",
    "SearchResult",
    "web_search",
}

__all__ = ["__version__", *_SEARCH_EXPORTS]


def __getattr__(name: str):
    """Lazily expose search helpers without making package import heavy."""
    if name in _SEARCH_EXPORTS:
        from . import search as _search

        return getattr(_search, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
