"""Offline-testable data contracts for tiered prospect research.

The tiered workflow separates company qualification, contact discovery, and
person-level personalization. These dataclasses intentionally keep provider and
browser transports outside the model layer so graph nodes can validate and
serialize contract-shaped data without live network calls.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

EvidenceDepth = Literal["fast", "standard", "deep"]
BrowserCaptureMode = Literal["none", "screenshots_only", "text_extraction", "full_page_audit"]
SourceConfidence = Literal["low", "medium", "high"]
RoleCategory = Literal["owner", "executive", "operator", "marketing", "unknown"]
ContactKind = Literal["person", "company_email", "company_phone", "company_contact_page"]
SignalConfidence = Literal["low", "medium", "high"]
TieredWorkflowStatus = Literal[
    "running",
    "review_required",
    "approval_recorded",
    "final_enrichment_blocked",
    "final_enrichment_complete",
    "completed",
    "partial_completed",
    "failed",
]


class TieredModelValidationError(ValueError):
    """Raised when tiered prospect research contracts are invalid."""


@dataclass(frozen=True)
class SearchDirective:
    """Operator input that starts a tiered prospect research run."""

    industry: str
    target_prospect_count: int
    geographic_area: str
    research_criteria: str
    negative_criteria: str = ""
    preferred_contact_roles: tuple[str, ...] = ()
    source_preferences: tuple[str, ...] = ()
    evidence_depth: EvidenceDepth = "standard"
    browser_capture: BrowserCaptureMode = "none"

    def __post_init__(self) -> None:
        _require_text(self.industry, "SearchDirective.industry")
        _require_text(self.geographic_area, "SearchDirective.geographic_area")
        _require_text(self.research_criteria, "SearchDirective.research_criteria")
        if self.target_prospect_count <= 0:
            raise TieredModelValidationError("target_prospect_count must be greater than 0")
        if self.evidence_depth not in {"fast", "standard", "deep"}:
            raise TieredModelValidationError("evidence_depth must be fast, standard, or deep")
        if self.browser_capture not in {
            "none",
            "screenshots_only",
            "text_extraction",
            "full_page_audit",
        }:
            raise TieredModelValidationError(
                "browser_capture must be none, screenshots_only, text_extraction, "
                "or full_page_audit"
            )
        object.__setattr__(
            self, "preferred_contact_roles", _text_tuple(self.preferred_contact_roles)
        )
        object.__setattr__(self, "source_preferences", _text_tuple(self.source_preferences))

    def to_dict(self) -> dict[str, Any]:
        return {
            "industry": self.industry,
            "target_prospect_count": self.target_prospect_count,
            "geographic_area": self.geographic_area,
            "research_criteria": self.research_criteria,
            "negative_criteria": self.negative_criteria,
            "preferred_contact_roles": list(self.preferred_contact_roles),
            "source_preferences": list(self.source_preferences),
            "evidence_depth": self.evidence_depth,
            "browser_capture": self.browser_capture,
        }


@dataclass(frozen=True)
class CompanyProspect:
    """Tier 1 company discovery and qualification output."""

    company_id: str
    name: str
    evidence_ids: tuple[str, ...]
    website: str = ""
    industry: str = ""
    geographic_area: str = ""
    locations: tuple[str, ...] = ()
    fit_score: float = 0.0
    fit_rationale: str = ""
    disqualification_flags: tuple[str, ...] = ()
    source_confidence: SourceConfidence = "medium"

    def __post_init__(self) -> None:
        _require_text(self.company_id, "CompanyProspect.company_id")
        _require_text(self.name, "CompanyProspect.name")
        if not 0 <= self.fit_score <= 1:
            raise TieredModelValidationError("fit_score must be between 0 and 1")
        if self.source_confidence not in {"low", "medium", "high"}:
            raise TieredModelValidationError("source_confidence must be low, medium, or high")
        object.__setattr__(
            self, "evidence_ids", _required_text_tuple(self.evidence_ids, "evidence_ids")
        )
        object.__setattr__(self, "locations", _text_tuple(self.locations))
        object.__setattr__(self, "disqualification_flags", _text_tuple(self.disqualification_flags))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("evidence_ids", "locations", "disqualification_flags"):
            payload[key] = list(payload[key])
        return payload


@dataclass(frozen=True)
class ContactCandidate:
    """Tier 2 contact discovered inside a qualified company."""

    contact_id: str
    company_id: str
    name: str
    evidence_ids: tuple[str, ...]
    title: str = ""
    role_category: RoleCategory = "unknown"
    profile_urls: tuple[str, ...] = ()
    email: str | None = None
    phone: str | None = None
    contact_confidence: float = 0.0
    notes: str = ""
    contact_kind: ContactKind = "person"
    label: str = ""
    url: str = ""
    contact_url: str = ""
    source_url: str = ""
    source_confidence: SourceConfidence = "medium"

    def __post_init__(self) -> None:
        _require_text(self.contact_id, "ContactCandidate.contact_id")
        _require_text(self.company_id, "ContactCandidate.company_id")
        _require_text(self.name, "ContactCandidate.name")
        if not 0 <= self.contact_confidence <= 1:
            raise TieredModelValidationError("contact_confidence must be between 0 and 1")
        if self.role_category not in {"owner", "executive", "operator", "marketing", "unknown"}:
            raise TieredModelValidationError(
                "role_category must be owner, executive, operator, marketing, or unknown"
            )
        if self.contact_kind not in {
            "person",
            "company_email",
            "company_phone",
            "company_contact_page",
        }:
            raise TieredModelValidationError(
                "contact_kind must be person, company_email, company_phone, or company_contact_page"
            )
        if self.source_confidence not in {"low", "medium", "high"}:
            raise TieredModelValidationError("source_confidence must be low, medium, or high")
        if self.contact_kind == "company_email" and not self.email:
            raise TieredModelValidationError("company_email contact requires email")
        if self.contact_kind == "company_phone" and not self.phone:
            raise TieredModelValidationError("company_phone contact requires phone")
        if self.contact_kind == "company_contact_page" and not (self.contact_url or self.url):
            raise TieredModelValidationError(
                "company_contact_page contact requires contact_url or url"
            )
        object.__setattr__(self, "label", self.label.strip() or self.name)
        object.__setattr__(
            self, "evidence_ids", _required_text_tuple(self.evidence_ids, "evidence_ids")
        )
        object.__setattr__(self, "profile_urls", _text_tuple(self.profile_urls))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence_ids"] = list(payload["evidence_ids"])
        payload["profile_urls"] = list(payload["profile_urls"])
        return payload


@dataclass(frozen=True)
class PersonalizationSignal:
    """Tier 3 evidence-backed person-level outreach context."""

    contact_id: str
    signal: str
    message_angle: str
    evidence_ids: tuple[str, ...]
    confidence: SignalConfidence = "medium"

    def __post_init__(self) -> None:
        _require_text(self.contact_id, "PersonalizationSignal.contact_id")
        _require_text(self.signal, "PersonalizationSignal.signal")
        _require_text(self.message_angle, "PersonalizationSignal.message_angle")
        if self.confidence not in {"low", "medium", "high"}:
            raise TieredModelValidationError("confidence must be low, medium, or high")
        object.__setattr__(
            self, "evidence_ids", _required_text_tuple(self.evidence_ids, "evidence_ids")
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence_ids"] = list(payload["evidence_ids"])
        return payload


@dataclass(frozen=True)
class ContactPersonalization:
    """Personalization bundle for one selected contact."""

    contact_id: str
    personalization_signals: tuple[PersonalizationSignal, ...] = ()
    suggested_opening_line: str = ""
    do_not_claim: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.contact_id, "ContactPersonalization.contact_id")
        signals = tuple(self.personalization_signals)
        for signal in signals:
            if signal.contact_id != self.contact_id:
                raise TieredModelValidationError(
                    "personalization signal contact_id must match ContactPersonalization.contact_id"
                )
        object.__setattr__(self, "personalization_signals", signals)
        object.__setattr__(self, "do_not_claim", _text_tuple(self.do_not_claim))

    def to_dict(self) -> dict[str, Any]:
        return {
            "contact_id": self.contact_id,
            "personalization_signals": [
                signal.to_dict() for signal in self.personalization_signals
            ],
            "suggested_opening_line": self.suggested_opening_line,
            "do_not_claim": list(self.do_not_claim),
        }


@dataclass(frozen=True)
class BrowserCapture:
    """Optional rendered-page evidence captured for audit."""

    browser_capture_id: str
    url: str
    captured_at: str
    evidence_ids: tuple[str, ...]
    company_id: str = ""
    contact_id: str = ""
    text_excerpt: str = ""
    screenshot_path: str = ""
    dom_snapshot_path: str = ""

    def __post_init__(self) -> None:
        _require_text(self.browser_capture_id, "BrowserCapture.browser_capture_id")
        _require_text(self.url, "BrowserCapture.url")
        _require_text(self.captured_at, "BrowserCapture.captured_at")
        object.__setattr__(
            self, "evidence_ids", _required_text_tuple(self.evidence_ids, "evidence_ids")
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence_ids"] = list(payload["evidence_ids"])
        return payload


@dataclass(frozen=True)
class ApprovedProspectSelection:
    """Human-approved company/contact IDs allowed to proceed past review."""

    approved_company_ids: tuple[str, ...] = ()
    approved_contact_ids: tuple[str, ...] = ()
    reviewer: str = ""
    approved_at: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "approved_company_ids",
            _required_text_tuple(self.approved_company_ids, "approved_company_ids"),
        )
        object.__setattr__(self, "approved_contact_ids", _text_tuple(self.approved_contact_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved_company_ids": list(self.approved_company_ids),
            "approved_contact_ids": list(self.approved_contact_ids),
            "reviewer": self.reviewer,
            "approved_at": self.approved_at,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class FinalEnrichmentRecord:
    """Approval-gated enrichment that must not mutate company qualification."""

    enrichment_id: str
    company_id: str
    contact_id: str = ""
    summary: str = ""
    evidence_ids: tuple[str, ...] = ()
    provider: str = "mock"
    warnings: tuple[str, ...] = ()
    contact_snapshot: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _require_text(self.enrichment_id, "FinalEnrichmentRecord.enrichment_id")
        _require_text(self.company_id, "FinalEnrichmentRecord.company_id")
        object.__setattr__(
            self, "evidence_ids", _required_text_tuple(self.evidence_ids, "evidence_ids")
        )
        object.__setattr__(self, "warnings", _text_tuple(self.warnings))
        if self.contact_snapshot is not None:
            object.__setattr__(self, "contact_snapshot", dict(self.contact_snapshot))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence_ids"] = list(payload["evidence_ids"])
        payload["warnings"] = list(payload["warnings"])
        return payload


@dataclass(frozen=True)
class TieredResearchRun:
    """Full normalized tiered prospect package for artifacts and review."""

    run_id: str
    directive: SearchDirective
    companies: tuple[CompanyProspect, ...] = ()
    contacts: tuple[ContactCandidate, ...] = ()
    personalizations: tuple[ContactPersonalization, ...] = ()
    browser_captures: tuple[BrowserCapture, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.run_id, "TieredResearchRun.run_id")
        object.__setattr__(self, "companies", tuple(self.companies))
        object.__setattr__(self, "contacts", tuple(self.contacts))
        object.__setattr__(self, "personalizations", tuple(self.personalizations))
        object.__setattr__(self, "browser_captures", tuple(self.browser_captures))
        object.__setattr__(self, "warnings", _text_tuple(self.warnings))
        validate_tiered_research_run(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "directive": self.directive.to_dict(),
            "companies": [company.to_dict() for company in self.companies],
            "contacts": [contact.to_dict() for contact in self.contacts],
            "personalizations": [item.to_dict() for item in self.personalizations],
            "browser_captures": [capture.to_dict() for capture in self.browser_captures],
            "warnings": list(self.warnings),
        }


def tiered_research_schema() -> dict[str, Any]:
    """Return a JSON-schema-like contract for serialized tiered research artifacts."""

    return {
        "type": "object",
        "required": ["run_id", "directive", "companies", "contacts", "personalizations"],
        "properties": {
            "run_id": {"type": "string"},
            "directive": {
                "type": "object",
                "required": [
                    "industry",
                    "target_prospect_count",
                    "geographic_area",
                    "research_criteria",
                ],
                "properties": {
                    "industry": {"type": "string"},
                    "target_prospect_count": {"type": "integer", "minimum": 1},
                    "geographic_area": {"type": "string"},
                    "research_criteria": {"type": "string"},
                    "negative_criteria": {"type": "string"},
                    "preferred_contact_roles": {"type": "array", "items": {"type": "string"}},
                    "source_preferences": {"type": "array", "items": {"type": "string"}},
                    "evidence_depth": {"type": "string", "enum": ["fast", "standard", "deep"]},
                    "browser_capture": {
                        "type": "string",
                        "enum": ["none", "screenshots_only", "text_extraction", "full_page_audit"],
                    },
                },
            },
            "companies": {"type": "array", "items": {"$ref": "#/definitions/company"}},
            "contacts": {"type": "array", "items": {"$ref": "#/definitions/contact"}},
            "personalizations": {
                "type": "array",
                "items": {"$ref": "#/definitions/contact_personalization"},
            },
            "browser_captures": {
                "type": "array",
                "items": {"$ref": "#/definitions/browser_capture"},
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "definitions": {
            "company": {
                "type": "object",
                "required": ["company_id", "name", "fit_score", "evidence_ids"],
                "properties": {
                    "company_id": {"type": "string"},
                    "name": {"type": "string"},
                    "website": {"type": "string"},
                    "industry": {"type": "string"},
                    "geographic_area": {"type": "string"},
                    "locations": {"type": "array", "items": {"type": "string"}},
                    "fit_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "fit_rationale": {"type": "string"},
                    "disqualification_flags": {"type": "array", "items": {"type": "string"}},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    "source_confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                },
            },
            "contact": {
                "type": "object",
                "required": ["contact_id", "company_id", "name", "evidence_ids"],
                "properties": {
                    "contact_id": {"type": "string"},
                    "company_id": {"type": "string"},
                    "contact_kind": {
                        "type": "string",
                        "enum": [
                            "person",
                            "company_email",
                            "company_phone",
                            "company_contact_page",
                        ],
                    },
                    "label": {"type": "string"},
                    "name": {"type": "string"},
                    "title": {"type": "string"},
                    "role_category": {
                        "type": "string",
                        "enum": ["owner", "executive", "operator", "marketing", "unknown"],
                    },
                    "profile_urls": {"type": "array", "items": {"type": "string"}},
                    "email": {"type": ["string", "null"]},
                    "phone": {"type": ["string", "null"]},
                    "url": {"type": "string"},
                    "contact_url": {"type": "string"},
                    "source_url": {"type": "string"},
                    "source_confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    "contact_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                },
            },
            "personalization_signal": {
                "type": "object",
                "required": ["contact_id", "signal", "message_angle", "evidence_ids"],
                "properties": {
                    "contact_id": {"type": "string"},
                    "signal": {"type": "string"},
                    "message_angle": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                },
            },
            "contact_personalization": {
                "type": "object",
                "required": ["contact_id", "personalization_signals"],
                "properties": {
                    "contact_id": {"type": "string"},
                    "personalization_signals": {
                        "type": "array",
                        "items": {"$ref": "#/definitions/personalization_signal"},
                    },
                    "suggested_opening_line": {"type": "string"},
                    "do_not_claim": {"type": "array", "items": {"type": "string"}},
                },
            },
            "browser_capture": {
                "type": "object",
                "required": ["browser_capture_id", "url", "captured_at", "evidence_ids"],
                "properties": {
                    "browser_capture_id": {"type": "string"},
                    "url": {"type": "string"},
                    "company_id": {"type": "string"},
                    "contact_id": {"type": "string"},
                    "captured_at": {"type": "string"},
                    "text_excerpt": {"type": "string"},
                    "screenshot_path": {"type": "string"},
                    "dom_snapshot_path": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    }


def validate_tiered_research_run(run: TieredResearchRun) -> TieredResearchRun:
    """Validate cross-record relationships in a tiered research package."""

    company_ids = _unique_ids((company.company_id for company in run.companies), "company_id")
    contact_ids = _unique_ids((contact.contact_id for contact in run.contacts), "contact_id")
    for contact in run.contacts:
        if contact.company_id not in company_ids:
            raise TieredModelValidationError(
                "contact "
                f"{contact.contact_id!r} references unknown company_id {contact.company_id!r}"
            )
    seen_personalizations: set[str] = set()
    for personalization in run.personalizations:
        if personalization.contact_id not in contact_ids:
            raise TieredModelValidationError(
                f"personalization references unknown contact_id {personalization.contact_id!r}"
            )
        if personalization.contact_id in seen_personalizations:
            raise TieredModelValidationError(
                f"duplicate personalization for contact_id {personalization.contact_id!r}"
            )
        seen_personalizations.add(personalization.contact_id)
    for capture in run.browser_captures:
        if capture.company_id and capture.company_id not in company_ids:
            raise TieredModelValidationError(
                f"browser capture references unknown company_id {capture.company_id!r}"
            )
        if capture.contact_id and capture.contact_id not in contact_ids:
            raise TieredModelValidationError(
                f"browser capture references unknown contact_id {capture.contact_id!r}"
            )
    return run


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TieredModelValidationError(f"{field_name} is required")


def _text_tuple(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(value).strip() for value in values if str(value).strip())


def _required_text_tuple(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    normalized = _text_tuple(values)
    if not normalized:
        raise TieredModelValidationError(f"{field_name} requires at least one value")
    return normalized


def _unique_ids(values: Iterable[str], field_name: str) -> set[str]:
    ids: set[str] = set()
    for value in values:
        if value in ids:
            raise TieredModelValidationError(f"duplicate {field_name} {value!r}")
        ids.add(value)
    return ids


def coerce_search_directive(value: SearchDirective | Mapping[str, Any]) -> SearchDirective:
    """Coerce a mapping or directive instance into ``SearchDirective``."""

    if isinstance(value, SearchDirective):
        return value
    return SearchDirective(
        industry=str(value.get("industry", "")),
        target_prospect_count=int(value.get("target_prospect_count", 0)),
        geographic_area=str(value.get("geographic_area", "")),
        research_criteria=str(value.get("research_criteria", "")),
        negative_criteria=str(value.get("negative_criteria", "")),
        preferred_contact_roles=tuple(
            str(item) for item in value.get("preferred_contact_roles", ())
        ),
        source_preferences=tuple(str(item) for item in value.get("source_preferences", ())),
        evidence_depth=str(value.get("evidence_depth", "standard")),  # type: ignore[arg-type]
        browser_capture=str(value.get("browser_capture", "none")),  # type: ignore[arg-type]
    )


__all__ = [
    "BrowserCapture",
    "BrowserCaptureMode",
    "CompanyProspect",
    "ContactCandidate",
    "ContactPersonalization",
    "EvidenceDepth",
    "ApprovedProspectSelection",
    "FinalEnrichmentRecord",
    "PersonalizationSignal",
    "RoleCategory",
    "SearchDirective",
    "SignalConfidence",
    "SourceConfidence",
    "TieredModelValidationError",
    "TieredResearchRun",
    "TieredWorkflowStatus",
    "coerce_search_directive",
    "tiered_research_schema",
    "validate_tiered_research_run",
]
