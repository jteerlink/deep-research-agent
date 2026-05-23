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
from .evidence import (  # noqa: F401
    Citation,
    CitationValidationError,
    EvidenceKind,
    EvidenceRecord,
    Prospect,
    SearchEvidenceArtifact,
    SearchFailure,
    collect_search_evidence,
    evidence_schema,
    prospect_schema,
    validate_citations,
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
    "Citation",
    "CitationValidationError",
    "EvidenceKind",
    "EvidenceRecord",
    "Prospect",
    "SearchEvidenceArtifact",
    "SearchFailure",
    "collect_search_evidence",
    "evidence_schema",
    "prospect_schema",
    "validate_citations",
    "web_search",
]
