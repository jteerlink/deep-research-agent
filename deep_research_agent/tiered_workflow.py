"""Offline-testable tiered prospect research workflow contracts.

This module keeps the G001 tiered workflow boundary deliberately small and
injectable. Search, contact extraction, personalization, and artifact writing are
provided by callers so the orchestration contract can be tested without network,
browser, or model side effects.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, TypeVar, cast
from urllib.parse import urlparse

TieredWorkflowStatus = Literal["completed", "needs_review"]
TierName = Literal[
    "directive_parser",
    "company_discovery",
    "company_qualification",
    "contact_discovery",
    "personalization_research",
    "artifact_writer",
    "review",
]

TIERED_WORKFLOW_TOPOLOGY: tuple[TierName, ...] = (
    "directive_parser",
    "company_discovery",
    "company_qualification",
    "contact_discovery",
    "personalization_research",
    "artifact_writer",
    "review",
)

_REQUIRED_DIRECTIVE_FIELDS = (
    "industry",
    "target_prospect_count",
    "geographic_area",
    "research_criteria",
)

T = TypeVar("T")
MaybeAwaitable = T | Awaitable[T]
CompanyDiscoveryFn = Callable[[Mapping[str, Any]], MaybeAwaitable[Sequence[Mapping[str, Any]]]]
ContactDiscoveryFn = Callable[
    [Mapping[str, Any], Mapping[str, Any]], MaybeAwaitable[Sequence[Mapping[str, Any]]]
]
PersonalizationResearchFn = Callable[
    [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
    MaybeAwaitable[Sequence[Mapping[str, Any]]],
]
ArtifactWriterFn = Callable[[Mapping[str, Any]], MaybeAwaitable[Mapping[str, str] | None]]


@dataclass(frozen=True)
class TieredWorkflowConfig:
    """Execution guardrails for tiered prospect research orchestration."""

    min_company_fit_score: float = 0.55
    max_contacts_per_company: int = 3
    require_human_review: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.min_company_fit_score <= 1:
            raise ValueError("min_company_fit_score must be between 0 and 1")
        if self.max_contacts_per_company < 1:
            raise ValueError("max_contacts_per_company must be at least 1")


@dataclass(frozen=True)
class TieredWorkflowEvent:
    """Serializable audit event emitted at tier boundaries."""

    tier: TierName
    event: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class TieredWorkflowState:
    """Normalized tiered workflow output for downstream artifacts and review."""

    directive: Mapping[str, Any]
    status: TieredWorkflowStatus
    companies: tuple[Mapping[str, Any], ...] = ()
    contacts: tuple[Mapping[str, Any], ...] = ()
    personalization_signals: tuple[Mapping[str, Any], ...] = ()
    evidence: tuple[Mapping[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    events: tuple[Mapping[str, Any], ...] = ()
    artifact_paths: Mapping[str, str] = field(default_factory=dict)
    review_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "directive": dict(self.directive),
            "status": self.status,
            "review_required": self.review_required,
            "companies": [dict(company) for company in self.companies],
            "contacts": [dict(contact) for contact in self.contacts],
            "personalization_signals": [dict(signal) for signal in self.personalization_signals],
            "evidence": [dict(record) for record in self.evidence],
            "warnings": list(self.warnings),
            "events": [dict(event) for event in self.events],
            "artifact_paths": dict(self.artifact_paths),
        }


class TieredResearchWorkflow:
    """Coordinate company, contact, and personalization tiers with injected IO."""

    def __init__(
        self,
        *,
        company_discovery: CompanyDiscoveryFn,
        contact_discovery: ContactDiscoveryFn,
        personalization_research: PersonalizationResearchFn,
        artifact_writer: ArtifactWriterFn | None = None,
        config: TieredWorkflowConfig | None = None,
    ) -> None:
        self.company_discovery = company_discovery
        self.contact_discovery = contact_discovery
        self.personalization_research = personalization_research
        self.artifact_writer = artifact_writer
        self.config = config or TieredWorkflowConfig()

    def run(self, directive: Mapping[str, Any]) -> TieredWorkflowState:
        """Synchronous wrapper around :meth:`arun` for CLI/UI callers."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(directive))
        raise RuntimeError(
            "TieredResearchWorkflow.run cannot be called inside a running event loop"
        )

    async def arun(self, directive: Mapping[str, Any]) -> TieredWorkflowState:
        """Run all tiers and return a review-gated normalized state."""

        events: list[dict[str, Any]] = []
        warnings: list[str] = []
        evidence: list[dict[str, Any]] = []

        events.append(_event("directive_parser", "started"))
        normalized_directive = normalize_search_directive(directive)
        target_count = int(normalized_directive["target_prospect_count"])
        events.append(
            _event(
                "directive_parser",
                "completed",
                required_fields=list(_REQUIRED_DIRECTIVE_FIELDS),
                target_prospect_count=target_count,
            )
        )

        events.append(_event("company_discovery", "started"))
        raw_companies = list(await _await_result(self.company_discovery(normalized_directive)))
        events.append(_event("company_discovery", "completed", candidate_count=len(raw_companies)))

        events.append(_event("company_qualification", "started"))
        companies = qualify_companies(
            raw_companies,
            target_count=target_count,
            min_fit_score=self.config.min_company_fit_score,
        )
        evidence.extend(_evidence_from_items(companies))
        if not companies:
            warnings.append("No qualified companies were found for the directive.")
        elif len(companies) < target_count:
            warnings.append(
                f"Only {len(companies)} qualified compan{'y' if len(companies) == 1 else 'ies'} "
                f"were found for requested target_prospect_count={target_count}."
            )
        events.append(
            _event("company_qualification", "completed", qualified_company_count=len(companies))
        )

        contacts: list[dict[str, Any]] = []
        events.append(_event("contact_discovery", "started"))
        for company in companies:
            raw_contacts = list(
                await _await_result(self.contact_discovery(normalized_directive, company))
            )
            company_contacts = normalize_contacts(
                raw_contacts,
                company,
                max_contacts=self.config.max_contacts_per_company,
            )
            if not company_contacts:
                warnings.append(
                    f"No contact candidates found for company_id={company['company_id']}."
                )
            contacts.extend(company_contacts)
        evidence.extend(_evidence_from_items(contacts))
        events.append(_event("contact_discovery", "completed", contact_count=len(contacts)))

        signals: list[dict[str, Any]] = []
        events.append(_event("personalization_research", "started"))
        companies_by_id = {str(company["company_id"]): company for company in companies}
        for contact in contacts:
            company = companies_by_id[str(contact["company_id"])]
            raw_signals = list(
                await _await_result(
                    self.personalization_research(normalized_directive, company, contact)
                )
            )
            contact_signals = normalize_personalization_signals(raw_signals, contact)
            if not contact_signals:
                warnings.append(
                    f"No personalization signals found for contact_id={contact['contact_id']}."
                )
            signals.extend(contact_signals)
        evidence.extend(_evidence_from_items(signals))
        events.append(
            _event(
                "personalization_research",
                "completed",
                personalization_signal_count=len(signals),
            )
        )

        artifact_paths: Mapping[str, str] = {}
        snapshot = _state_payload(
            directive=normalized_directive,
            status="needs_review" if self.config.require_human_review else "completed",
            companies=companies,
            contacts=contacts,
            personalization_signals=signals,
            evidence=evidence,
            warnings=warnings,
            events=events,
            artifact_paths={},
            review_required=self.config.require_human_review,
        )
        if self.artifact_writer is not None:
            events.append(_event("artifact_writer", "started"))
            artifact_paths = dict(await _await_result(self.artifact_writer(snapshot)) or {})
            events.append(
                _event("artifact_writer", "completed", artifact_count=len(artifact_paths))
            )

        if self.config.require_human_review:
            events.append(_event("review", "required"))

        return TieredWorkflowState(
            directive=normalized_directive,
            status="needs_review" if self.config.require_human_review else "completed",
            companies=tuple(companies),
            contacts=tuple(contacts),
            personalization_signals=tuple(signals),
            evidence=tuple(evidence),
            warnings=tuple(warnings),
            events=tuple(events),
            artifact_paths=artifact_paths,
            review_required=self.config.require_human_review,
        )


def normalize_search_directive(directive: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the required tiered search directive fields."""

    missing = [
        field
        for field in _REQUIRED_DIRECTIVE_FIELDS
        if not str(directive.get(field, "")).strip()
    ]
    if missing:
        raise ValueError(f"search directive missing required field(s): {', '.join(missing)}")

    target_count = _positive_int(directive["target_prospect_count"], "target_prospect_count")
    normalized = dict(directive)
    normalized["industry"] = str(directive["industry"]).strip()
    normalized["geographic_area"] = str(directive["geographic_area"]).strip()
    normalized["research_criteria"] = str(directive["research_criteria"]).strip()
    normalized["target_prospect_count"] = target_count
    normalized.setdefault("preferred_contact_roles", [])
    normalized.setdefault("negative_criteria", "")
    return normalized


def qualify_companies(
    candidates: Sequence[Mapping[str, Any]], *, target_count: int, min_fit_score: float = 0.55
) -> list[dict[str, Any]]:
    """Deduplicate, filter, rank, and cap company candidates for Tier 1."""

    seen: set[str] = set()
    qualified: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        company = _normalize_company(candidate, index)
        key = _company_dedupe_key(company)
        if not key or key in seen:
            continue
        seen.add(key)
        if company["disqualification_flags"]:
            continue
        if float(company["fit_score"]) < min_fit_score:
            continue
        qualified.append(company)

    qualified.sort(key=lambda company: float(company["fit_score"]), reverse=True)
    return qualified[:target_count]


def normalize_contacts(
    contacts: Sequence[Mapping[str, Any]], company: Mapping[str, Any], *, max_contacts: int
) -> list[dict[str, Any]]:
    """Normalize and cap Tier 2 contact candidates for one company."""

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    company_id = str(company["company_id"])
    for index, contact in enumerate(contacts, start=1):
        name = str(contact.get("name") or contact.get("person_name") or "").strip()
        title = str(contact.get("title") or contact.get("role") or "").strip()
        if not name:
            continue
        key = _slug(f"{company_id}-{name}-{title}")
        if key in seen:
            continue
        seen.add(key)
        payload = dict(contact)
        payload["contact_id"] = str(
            contact.get("contact_id") or f"{company_id}_contact_{index:03d}"
        )
        payload["company_id"] = company_id
        payload["name"] = name
        payload["title"] = title
        payload["role_category"] = str(contact.get("role_category") or "")
        payload["contact_confidence"] = _clamp_float(contact.get("contact_confidence", 1.0))
        payload["evidence_ids"] = _string_list(contact.get("evidence_ids", ()))
        normalized.append(payload)
        if len(normalized) >= max_contacts:
            break
    return normalized


def normalize_personalization_signals(
    signals: Sequence[Mapping[str, Any]], contact: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Normalize Tier 3 personalization signals for one contact."""

    normalized: list[dict[str, Any]] = []
    contact_id = str(contact["contact_id"])
    for index, signal in enumerate(signals, start=1):
        text = str(signal.get("signal") or signal.get("summary") or "").strip()
        if not text:
            continue
        payload = dict(signal)
        payload["signal_id"] = str(signal.get("signal_id") or f"{contact_id}_signal_{index:03d}")
        payload["contact_id"] = contact_id
        payload["company_id"] = str(contact["company_id"])
        payload["signal"] = text
        payload["message_angle"] = str(signal.get("message_angle") or "")
        payload["evidence_ids"] = _string_list(signal.get("evidence_ids", ()))
        payload["confidence"] = str(signal.get("confidence") or "medium")
        normalized.append(payload)
    return normalized


def _normalize_company(candidate: Mapping[str, Any], index: int) -> dict[str, Any]:
    name = str(
        candidate.get("name")
        or candidate.get("company_name")
        or candidate.get("organization")
        or ""
    ).strip()
    website = str(candidate.get("website") or candidate.get("url") or "").strip()
    company_id = str(candidate.get("company_id") or _stable_company_id(name, website, index))
    payload = dict(candidate)
    payload["company_id"] = company_id
    payload["name"] = name
    payload["website"] = website
    payload["fit_score"] = _clamp_float(candidate.get("fit_score", 1.0))
    payload["fit_rationale"] = str(candidate.get("fit_rationale") or "")
    payload["disqualification_flags"] = _string_list(candidate.get("disqualification_flags", ()))
    payload["evidence_ids"] = _string_list(candidate.get("evidence_ids", ()))
    payload["source_confidence"] = str(candidate.get("source_confidence") or "medium")
    return payload


async def _await_result(value: MaybeAwaitable[T]) -> T:
    if inspect.isawaitable(value):
        return cast(T, await value)
    return value


def _event(tier: TierName, event: str, **metadata: Any) -> dict[str, Any]:
    return TieredWorkflowEvent(tier=tier, event=event, metadata=metadata).to_dict()


def _state_payload(
    *,
    directive: Mapping[str, Any],
    status: TieredWorkflowStatus,
    companies: Sequence[Mapping[str, Any]],
    contacts: Sequence[Mapping[str, Any]],
    personalization_signals: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    warnings: Sequence[str],
    events: Sequence[Mapping[str, Any]],
    artifact_paths: Mapping[str, str],
    review_required: bool,
) -> dict[str, Any]:
    return TieredWorkflowState(
        directive=directive,
        status=status,
        companies=tuple(companies),
        contacts=tuple(contacts),
        personalization_signals=tuple(personalization_signals),
        evidence=tuple(evidence),
        warnings=tuple(warnings),
        events=tuple(events),
        artifact_paths=artifact_paths,
        review_required=review_required,
    ).to_dict()


def _evidence_from_items(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for item in items:
        for record in item.get("evidence", ()) or ():
            if isinstance(record, Mapping):
                evidence.append(dict(record))
    return evidence


def _positive_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _clamp_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    return min(1.0, max(0.0, parsed))


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _company_dedupe_key(company: Mapping[str, Any]) -> str:
    website = str(company.get("website") or "")
    domain = urlparse(website).netloc.lower().removeprefix("www.")
    if domain:
        return domain
    return _slug(str(company.get("name") or ""))


def _stable_company_id(name: str, website: str, index: int) -> str:
    base = _company_dedupe_key({"name": name, "website": website}) or f"company-{index}"
    return f"company_{_slug(base) or index:0>3}"


def _slug(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")
