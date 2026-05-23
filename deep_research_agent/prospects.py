"""Prospect schemas and citation validation for G002 artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .evidence import EvidenceRecord


class CitationValidationError(ValueError):
    """Raised when a prospect citation does not reference available evidence."""


@dataclass(frozen=True)
class ProspectCitation:
    """A citation from a prospect claim to a normalized evidence record."""

    evidence_id: str
    claim: str
    quote: str = ""

    def __post_init__(self) -> None:
        if not self.evidence_id:
            raise ValueError("ProspectCitation.evidence_id is required")
        if not self.claim:
            raise ValueError("ProspectCitation.claim is required")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class Prospect:
    """A research prospect with explicit evidence-backed citations."""

    name: str
    summary: str
    citations: tuple[ProspectCitation, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Prospect.name is required")
        object.__setattr__(self, "citations", tuple(self.citations))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "summary": self.summary,
            "citations": [citation.to_dict() for citation in self.citations],
            "metadata": dict(self.metadata),
        }


def _evidence_by_id(evidence_records: Iterable[EvidenceRecord]) -> dict[str, EvidenceRecord]:
    return {record.id: record for record in evidence_records}


def validate_prospect_citations(
    prospect: Prospect,
    evidence_records: Iterable[EvidenceRecord],
) -> None:
    """Validate that every citation points to known evidence.

    If a citation includes a quote, the quote must appear in the evidence text.
    Empty quotes are allowed for citation-only references.
    """

    evidence = _evidence_by_id(evidence_records)
    for citation in prospect.citations:
        record = evidence.get(citation.evidence_id)
        if record is None:
            raise CitationValidationError(
                f"Prospect {prospect.name!r} cites unknown evidence id {citation.evidence_id!r}"
            )
        if citation.quote and citation.quote not in record.text:
            raise CitationValidationError(
                f"Prospect {prospect.name!r} quote for {citation.evidence_id!r} "
                "is not present in evidence text"
            )


def validate_prospects_citations(
    prospects: Sequence[Prospect],
    evidence_records: Iterable[EvidenceRecord],
) -> None:
    """Validate citations for a collection of prospects."""

    evidence = tuple(evidence_records)
    for prospect in prospects:
        validate_prospect_citations(prospect, evidence)
