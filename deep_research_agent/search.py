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
    EvidenceRecord,
    SearchEvidenceBatch,
    SearchFailure,
    collect_search_evidence,
    evidence_id_for,
    evidence_source_type,
    normalize_search_results,
)
from .prospects import (  # noqa: F401
    CitationValidationError,
    EvidenceReference,
    Prospect,
    ProspectCitation,
    ProspectRecord,
    citation_schema,
    evidence_reference_schema,
    normalize_evidence_catalog,
    prospect_schema,
    validate_citation,
    validate_prospect,
    validate_prospect_citations,
    validate_prospects,
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
    "CitationValidationError",
    "EvidenceRecord",
    "EvidenceReference",
    "Prospect",
    "ProspectCitation",
    "ProspectRecord",
    "SearchEvidenceBatch",
    "SearchFailure",
    "citation_schema",
    "collect_search_evidence",
    "evidence_id_for",
    "evidence_reference_schema",
    "evidence_source_type",
    "normalize_evidence_catalog",
    "normalize_search_results",
    "prospect_schema",
    "validate_citation",
    "validate_prospect",
    "validate_prospect_citations",
    "validate_prospects",
    "web_search",
]
