"""Prospect schemas and citation validation for G002 artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .evidence import EvidenceRecord

CitationPurpose = Literal["discovery", "fact"]
EvidenceType = Literal["page_read", "snippet"]


class CitationValidationError(ValueError):
    """Raised when a prospect citation does not reference available evidence."""


@dataclass(frozen=True)
class EvidenceReference:
    """Normalized evidence shape used by prospect validation."""

    evidence_id: str
    url: str
    evidence_type: EvidenceType
    content: str
    provider: str = ""
    title: str = ""

    @classmethod
    def from_evidence_record(cls, record: EvidenceRecord) -> EvidenceReference:
        return cls(
            evidence_id=record.id,
            url=record.url,
            evidence_type=record.source_type,
            content=record.text,
            provider=record.provider,
            title=record.title,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProspectCitation:
    """A citation from a prospect claim to normalized evidence."""

    evidence_id: str
    claim: str
    quote: str = ""
    purpose: CitationPurpose = "fact"
    field: str = ""

    def __post_init__(self) -> None:
        if not self.evidence_id:
            raise ValueError("ProspectCitation.evidence_id is required")
        if not self.claim:
            raise ValueError("ProspectCitation.claim is required")
        if self.purpose not in {"discovery", "fact"}:
            raise ValueError("ProspectCitation.purpose must be discovery or fact")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ProspectRecord:
    """Structured prospect-target record for exports."""

    organization: str
    confidence: float
    citations: tuple[ProspectCitation, ...] = ()
    website: str = ""
    summary: str = ""
    decision_maker_leads: tuple[str, ...] = ()
    fit_rationale: str = ""
    personalized_angles: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.organization:
            raise ValueError("ProspectRecord.organization is required")
        object.__setattr__(self, "citations", tuple(self.citations))
        object.__setattr__(self, "decision_maker_leads", tuple(self.decision_maker_leads))
        object.__setattr__(self, "personalized_angles", tuple(self.personalized_angles))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "organization": self.organization,
            "website": self.website,
            "summary": self.summary,
            "confidence": self.confidence,
            "decision_maker_leads": list(self.decision_maker_leads),
            "fit_rationale": self.fit_rationale,
            "personalized_angles": list(self.personalized_angles),
            "citations": [citation.to_dict() for citation in self.citations],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Prospect:
    """Backward-compatible simple prospect used by early G002 artifact tests."""

    name: str
    summary: str
    citations: tuple[ProspectCitation, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Prospect.name is required")
        object.__setattr__(self, "citations", tuple(self.citations))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_record(self) -> ProspectRecord:
        return ProspectRecord(
            organization=self.name,
            summary=self.summary,
            confidence=float(self.metadata.get("confidence", 1.0)),
            citations=self.citations,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "summary": self.summary,
            "citations": [citation.to_dict() for citation in self.citations],
            "metadata": dict(self.metadata),
        }


def citation_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["evidence_id", "claim", "purpose"],
        "properties": {
            "evidence_id": {"type": "string"},
            "claim": {"type": "string"},
            "quote": {"type": "string"},
            "purpose": {"type": "string", "enum": ["discovery", "fact"]},
            "field": {"type": "string"},
        },
    }


def evidence_reference_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["evidence_id", "url", "evidence_type", "content"],
        "properties": {
            "evidence_id": {"type": "string"},
            "url": {"type": "string"},
            "evidence_type": {"type": "string", "enum": ["page_read", "snippet"]},
            "content": {"type": "string"},
        },
    }


def prospect_schema() -> dict[str, Any]:
    citation = citation_schema()
    return {
        "type": "object",
        "required": ["organization", "confidence", "citations"],
        "properties": {
            "organization": {"type": "string"},
            "website": {"type": "string"},
            "summary": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "decision_maker_leads": {"type": "array", "items": {"type": "string"}},
            "fit_rationale": {"type": "string"},
            "personalized_angles": {"type": "array", "items": {"type": "string"}},
            "citations": {"type": "array", "items": citation},
        },
    }


def normalize_evidence_catalog(
    evidence_records: Iterable[EvidenceReference | EvidenceRecord | Mapping[str, Any]]
    | Mapping[str, EvidenceReference | EvidenceRecord | Mapping[str, Any]],
) -> dict[str, EvidenceReference]:
    evidence: dict[str, EvidenceReference] = {}
    items = evidence_records.values() if isinstance(evidence_records, Mapping) else evidence_records
    for item in items:
        if isinstance(item, EvidenceReference):
            ref = item
        elif isinstance(item, EvidenceRecord):
            ref = EvidenceReference.from_evidence_record(item)
        else:
            evidence_id = str(item.get("evidence_id") or item.get("id") or "")
            ref = EvidenceReference(
                evidence_id=evidence_id,
                url=str(item.get("url") or item.get("source_url") or ""),
                evidence_type=str(
                    item.get("evidence_type") or item.get("source_type") or "snippet"
                ),  # type: ignore[arg-type]
                content=str(item.get("content") or item.get("text") or ""),
                provider=str(item.get("provider") or ""),
                title=str(item.get("title") or ""),
            )
        evidence[ref.evidence_id] = ref
    return evidence


def _coerce_citation(value: ProspectCitation | Mapping[str, Any]) -> ProspectCitation:
    if isinstance(value, ProspectCitation):
        return value
    return ProspectCitation(
        evidence_id=str(value.get("evidence_id", "")),
        claim=str(value.get("claim", "")),
        quote=str(value.get("quote", "")),
        purpose=str(value.get("purpose", "fact")),  # type: ignore[arg-type]
        field=str(value.get("field", "")),
    )


def validate_citation(
    citation: ProspectCitation | Mapping[str, Any],
    evidence: Mapping[str, EvidenceReference],
) -> ProspectCitation:
    citation_obj = _coerce_citation(citation)
    ref = evidence.get(citation_obj.evidence_id)
    if ref is None:
        raise CitationValidationError(f"unknown evidence_id {citation_obj.evidence_id!r}")
    if ref.evidence_type == "snippet" and citation_obj.purpose != "discovery":
        raise CitationValidationError("snippet evidence can only support discovery citations")
    if ref.evidence_type == "page_read" and citation_obj.purpose == "fact":
        if not citation_obj.quote:
            raise CitationValidationError("page_read fact citations require a quote")
        if citation_obj.quote not in ref.content:
            raise CitationValidationError("quote is not present in evidence content")
    return citation_obj


def _coerce_prospect(value: ProspectRecord | Prospect | Mapping[str, Any]) -> ProspectRecord:
    if isinstance(value, ProspectRecord):
        return value
    if isinstance(value, Prospect):
        return value.to_record()
    citations = tuple(_coerce_citation(c) for c in value.get("citations", ()))
    return ProspectRecord(
        organization=str(
            value.get("organization") or value.get("company_name") or value.get("name") or ""
        ),
        website=str(value.get("website", "")),
        summary=str(value.get("summary", "")),
        confidence=float(value.get("confidence", 1.0)),
        decision_maker_leads=tuple(str(item) for item in value.get("decision_maker_leads", ())),
        fit_rationale=str(value.get("fit_rationale", "")),
        personalized_angles=tuple(str(item) for item in value.get("personalized_angles", ())),
        citations=citations,
        metadata=dict(value.get("metadata", {})),
    )


def validate_prospect(
    prospect: ProspectRecord | Prospect | Mapping[str, Any],
    evidence_records: Iterable[EvidenceReference | EvidenceRecord | Mapping[str, Any]]
    | Mapping[str, EvidenceReference | EvidenceRecord | Mapping[str, Any]],
) -> ProspectRecord:
    evidence = normalize_evidence_catalog(evidence_records)
    record = _coerce_prospect(prospect)
    if not 0 <= record.confidence <= 1:
        raise CitationValidationError("confidence must be between 0 and 1")
    if not record.citations:
        raise CitationValidationError("prospect requires at least one citation")
    for citation in record.citations:
        validate_citation(citation, evidence)
    return record


def validate_prospects(
    prospects: Sequence[ProspectRecord | Prospect | Mapping[str, Any]],
    evidence_records: Iterable[EvidenceReference | EvidenceRecord | Mapping[str, Any]]
    | Mapping[str, EvidenceReference | EvidenceRecord | Mapping[str, Any]],
) -> tuple[ProspectRecord, ...]:
    evidence = normalize_evidence_catalog(evidence_records)
    return tuple(validate_prospect(prospect, evidence) for prospect in prospects)


def validate_prospect_citations(
    prospect: Prospect,
    evidence_records: Iterable[EvidenceRecord],
) -> None:
    """Validate citations for the simple Prospect compatibility model."""

    evidence = normalize_evidence_catalog(evidence_records)
    for citation in prospect.citations:
        ref = evidence.get(citation.evidence_id)
        if ref is None:
            raise CitationValidationError(
                f"Prospect {prospect.name!r} cites unknown evidence id {citation.evidence_id!r}"
            )
        if citation.quote and citation.quote not in ref.content:
            raise CitationValidationError(
                f"Prospect {prospect.name!r} quote for {citation.evidence_id!r} is not present"
            )


def validate_prospects_citations(
    prospects: Sequence[Prospect],
    evidence_records: Iterable[EvidenceRecord],
) -> None:
    evidence = tuple(evidence_records)
    for prospect in prospects:
        validate_prospect_citations(prospect, evidence)
