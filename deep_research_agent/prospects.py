"""Prospect schemas and citation validation for research artifacts.

G002 introduces citation-aware prospect records without adding a database or a
hosted service dependency.  The validators in this module intentionally accept
plain mappings as well as dataclass-like objects so they can sit between the
legacy ``async_multi_search`` results, future evidence wrappers, and artifact
writers.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

EvidenceType = Literal["snippet", "page_read"]
CitationPurpose = Literal["discovery", "fact"]

SUPPORTED_EVIDENCE_TYPES: frozenset[str] = frozenset({"snippet", "page_read"})
SUPPORTED_CITATION_PURPOSES: frozenset[str] = frozenset({"discovery", "fact"})


class CitationValidationError(ValueError):
    """Raised when a prospect citation cannot support the requested claim."""


@dataclass(frozen=True)
class EvidenceReference:
    """Minimal evidence shape required for prospect citation validation.

    ``snippet`` evidence represents search-result text that can support discovery
    or candidate-source claims. ``page_read`` evidence represents retrieved page
    content that can support factual prospect claims when the cited quote appears
    in the captured content.
    """

    evidence_id: str
    url: str
    evidence_type: EvidenceType
    content: str
    title: str = ""
    provider: str = ""
    query: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable evidence reference."""

        return {
            "evidence_id": self.evidence_id,
            "url": self.url,
            "evidence_type": self.evidence_type,
            "content": self.content,
            "title": self.title,
            "provider": self.provider,
            "query": self.query,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ProspectCitation:
    """Citation linking one prospect claim to one evidence record."""

    evidence_id: str
    claim: str
    purpose: CitationPurpose = "fact"
    quote: str = ""
    field: str = ""

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable citation."""

        return {
            "evidence_id": self.evidence_id,
            "claim": self.claim,
            "purpose": self.purpose,
            "quote": self.quote,
            "field": self.field,
        }


@dataclass(frozen=True)
class ProspectRecord:
    """Citation-backed prospect record emitted by research flows."""

    organization: str
    website: str = ""
    summary: str = ""
    relevance_reason: str = ""
    confidence: float = 0.0
    citations: tuple[ProspectCitation, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable prospect record."""

        return {
            "organization": self.organization,
            "website": self.website,
            "summary": self.summary,
            "relevance_reason": self.relevance_reason,
            "confidence": self.confidence,
            "citations": [citation.to_dict() for citation in self.citations],
            "metadata": dict(self.metadata),
        }


def citation_schema() -> dict[str, Any]:
    """Return the JSON-schema-compatible shape for a prospect citation."""

    return {
        "type": "object",
        "required": ["evidence_id", "claim", "purpose"],
        "additionalProperties": False,
        "properties": {
            "evidence_id": {"type": "string", "minLength": 1},
            "claim": {"type": "string", "minLength": 1},
            "purpose": {"type": "string", "enum": sorted(SUPPORTED_CITATION_PURPOSES)},
            "quote": {"type": "string"},
            "field": {"type": "string"},
        },
    }


def prospect_schema() -> dict[str, Any]:
    """Return the JSON-schema-compatible shape for a prospect artifact row."""

    return {
        "type": "object",
        "required": ["organization", "confidence", "citations"],
        "additionalProperties": False,
        "properties": {
            "organization": {"type": "string", "minLength": 1},
            "website": {"type": "string"},
            "summary": {"type": "string"},
            "relevance_reason": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "citations": {"type": "array", "items": citation_schema(), "minItems": 1},
            "metadata": {"type": "object"},
        },
    }


def evidence_reference_schema() -> dict[str, Any]:
    """Return the JSON-schema-compatible shape for cited evidence metadata."""

    return {
        "type": "object",
        "required": ["evidence_id", "url", "evidence_type", "content"],
        "additionalProperties": False,
        "properties": {
            "evidence_id": {"type": "string", "minLength": 1},
            "url": {"type": "string", "minLength": 1},
            "evidence_type": {"type": "string", "enum": sorted(SUPPORTED_EVIDENCE_TYPES)},
            "content": {"type": "string"},
            "title": {"type": "string"},
            "provider": {"type": "string"},
            "query": {"type": "string"},
            "metadata": {"type": "object"},
        },
    }


def normalize_evidence_catalog(evidence: Iterable[EvidenceReference | Mapping[str, Any] | Any]) -> dict[str, EvidenceReference]:
    """Normalize mappings or object-like evidence records by ``evidence_id``.

    The function accepts aliases used by search wrappers (``id``/``source_url`` /
    ``source_type``) so artifact validators do not couple to one concrete
    evidence dataclass.
    """

    catalog: dict[str, EvidenceReference] = {}
    for item in evidence:
        reference = _coerce_evidence_reference(item)
        if reference.evidence_id in catalog:
            raise CitationValidationError(f"duplicate evidence_id: {reference.evidence_id}")
        catalog[reference.evidence_id] = reference
    return catalog


def validate_citation(
    citation: ProspectCitation | Mapping[str, Any] | Any,
    evidence_catalog: Mapping[str, EvidenceReference | Mapping[str, Any] | Any],
) -> ProspectCitation:
    """Validate and return a citation against normalized evidence semantics."""

    normalized = _coerce_citation(citation)
    if not normalized.evidence_id:
        raise CitationValidationError("citation evidence_id is required")
    if not normalized.claim.strip():
        raise CitationValidationError("citation claim is required")
    if normalized.purpose not in SUPPORTED_CITATION_PURPOSES:
        raise CitationValidationError(f"unsupported citation purpose: {normalized.purpose}")

    try:
        raw_evidence = evidence_catalog[normalized.evidence_id]
    except KeyError as exc:
        raise CitationValidationError(f"unknown evidence_id: {normalized.evidence_id}") from exc

    evidence = _coerce_evidence_reference(raw_evidence)
    if evidence.evidence_type not in SUPPORTED_EVIDENCE_TYPES:
        raise CitationValidationError(f"unsupported evidence_type: {evidence.evidence_type}")

    if evidence.evidence_type == "snippet" and normalized.purpose != "discovery":
        raise CitationValidationError(
            "snippet evidence can only support discovery citations; use page_read for fact claims"
        )

    if evidence.evidence_type == "page_read" and normalized.purpose == "fact":
        if not normalized.quote.strip():
            raise CitationValidationError("page_read fact citations require a quote")
        if _normalize_text(normalized.quote) not in _normalize_text(evidence.content):
            raise CitationValidationError("citation quote is not present in page_read evidence")

    if normalized.quote and _normalize_text(normalized.quote) not in _normalize_text(evidence.content):
        raise CitationValidationError("citation quote is not present in cited evidence")

    return normalized


def validate_prospect(
    prospect: ProspectRecord | Mapping[str, Any] | Any,
    evidence: Mapping[str, EvidenceReference | Mapping[str, Any] | Any]
    | Iterable[EvidenceReference | Mapping[str, Any] | Any],
) -> ProspectRecord:
    """Validate a prospect record and all of its citations."""

    normalized = _coerce_prospect(prospect)
    if not normalized.organization.strip():
        raise CitationValidationError("prospect organization is required")
    if not 0.0 <= normalized.confidence <= 1.0:
        raise CitationValidationError("prospect confidence must be between 0.0 and 1.0")
    if not normalized.citations:
        raise CitationValidationError("prospect requires at least one citation")

    catalog = evidence if isinstance(evidence, Mapping) else normalize_evidence_catalog(evidence)
    validated_citations = tuple(validate_citation(citation, catalog) for citation in normalized.citations)
    return ProspectRecord(
        organization=normalized.organization,
        website=normalized.website,
        summary=normalized.summary,
        relevance_reason=normalized.relevance_reason,
        confidence=normalized.confidence,
        citations=validated_citations,
        metadata=normalized.metadata,
    )


def validate_prospects(
    prospects: Iterable[ProspectRecord | Mapping[str, Any] | Any],
    evidence: Mapping[str, EvidenceReference | Mapping[str, Any] | Any]
    | Iterable[EvidenceReference | Mapping[str, Any] | Any],
) -> list[ProspectRecord]:
    """Validate multiple prospect records against one evidence catalog."""

    catalog = evidence if isinstance(evidence, Mapping) else normalize_evidence_catalog(evidence)
    return [validate_prospect(prospect, catalog) for prospect in prospects]


def _coerce_evidence_reference(item: EvidenceReference | Mapping[str, Any] | Any) -> EvidenceReference:
    if isinstance(item, EvidenceReference):
        return item

    evidence_id = str(_read_value(item, "evidence_id", "id", default=""))
    url = str(_read_value(item, "url", "source_url", default=""))
    raw_type = str(_read_value(item, "evidence_type", "source_type", default="snippet"))
    evidence_type = _coerce_evidence_type(raw_type)
    content = str(_read_value(item, "content", "text", "snippet", default=""))
    return EvidenceReference(
        evidence_id=evidence_id,
        url=url,
        evidence_type=evidence_type,
        content=content,
        title=str(_read_value(item, "title", default="")),
        provider=str(_read_value(item, "provider", default="")),
        query=str(_read_value(item, "query", default="")),
        metadata=_read_mapping(item, "metadata"),
    )


def _coerce_citation(item: ProspectCitation | Mapping[str, Any] | Any) -> ProspectCitation:
    if isinstance(item, ProspectCitation):
        return item

    raw_purpose = str(_read_value(item, "purpose", default="fact"))
    purpose = _coerce_citation_purpose(raw_purpose)
    return ProspectCitation(
        evidence_id=str(_read_value(item, "evidence_id", "id", default="")),
        claim=str(_read_value(item, "claim", default="")),
        purpose=purpose,
        quote=str(_read_value(item, "quote", default="")),
        field=str(_read_value(item, "field", default="")),
    )


def _coerce_prospect(item: ProspectRecord | Mapping[str, Any] | Any) -> ProspectRecord:
    if isinstance(item, ProspectRecord):
        return item

    citations = tuple(_coerce_citation(citation) for citation in _read_sequence(item, "citations"))
    return ProspectRecord(
        organization=str(_read_value(item, "organization", "company", "company_name", default="")),
        website=str(_read_value(item, "website", "url", default="")),
        summary=str(_read_value(item, "summary", default="")),
        relevance_reason=str(_read_value(item, "relevance_reason", "reason", default="")),
        confidence=float(_read_value(item, "confidence", default=0.0)),
        citations=citations,
        metadata=_read_mapping(item, "metadata"),
    )


def _coerce_evidence_type(value: str) -> EvidenceType:
    if value not in SUPPORTED_EVIDENCE_TYPES:
        raise CitationValidationError(f"unsupported evidence_type: {value}")
    return value  # type: ignore[return-value]


def _coerce_citation_purpose(value: str) -> CitationPurpose:
    if value not in SUPPORTED_CITATION_PURPOSES:
        raise CitationValidationError(f"unsupported citation purpose: {value}")
    return value  # type: ignore[return-value]


def _read_value(item: Mapping[str, Any] | Any, *names: str, default: Any) -> Any:
    for name in names:
        if isinstance(item, Mapping) and name in item:
            return item[name]
        if not isinstance(item, Mapping) and hasattr(item, name):
            return getattr(item, name)
    return default


def _read_mapping(item: Mapping[str, Any] | Any, name: str) -> Mapping[str, Any]:
    value = _read_value(item, name, default={})
    return value if isinstance(value, Mapping) else {}


def _read_sequence(item: Mapping[str, Any] | Any, name: str) -> Iterable[Any]:
    value = _read_value(item, name, default=())
    if isinstance(value, str):
        return ()
    return value if isinstance(value, Iterable) else ()


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


__all__ = [
    "CitationPurpose",
    "CitationValidationError",
    "EvidenceReference",
    "EvidenceType",
    "ProspectCitation",
    "ProspectRecord",
    "citation_schema",
    "evidence_reference_schema",
    "normalize_evidence_catalog",
    "prospect_schema",
    "validate_citation",
    "validate_prospect",
    "validate_prospects",
]
