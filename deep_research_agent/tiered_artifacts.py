"""Tiered prospect research artifact contracts and writers.

The tiered artifact surface is intentionally separate from the legacy flat
``artifacts.py`` helpers.  It preserves the normalized company -> contact ->
personalization hierarchy described by the tiered prospecting plan while still
emitting operator-friendly CSV and Markdown projections.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, NamedTuple

TIERED_ARTIFACT_SCHEMA_VERSION = "g001.tiered_prospect_research.v1"


class TieredArtifactPaths(NamedTuple):
    """Paths and counts emitted by :func:`write_tiered_artifacts`."""

    json_path: Path
    companies_csv_path: Path
    contacts_csv_path: Path
    personalization_csv_path: Path
    markdown_path: Path
    company_count: int
    contact_count: int
    personalization_count: int


@dataclass(frozen=True)
class SearchDirective:
    """Operator directive that shaped a tiered prospect research run."""

    industry: str
    geography: str
    target_count: int
    criteria: tuple[str, ...] = ()
    raw_query: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.industry:
            raise ValueError("SearchDirective.industry is required")
        if not self.geography:
            raise ValueError("SearchDirective.geography is required")
        if self.target_count < 1:
            raise ValueError("SearchDirective.target_count must be >= 1")
        object.__setattr__(self, "criteria", tuple(self.criteria))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class CompanyProspect:
    """Tier 1 qualified company record."""

    company_id: str
    name: str
    domain: str = ""
    website: str = ""
    industry: str = ""
    geography: str = ""
    qualification_status: str = "qualified"
    fit_score: float | None = None
    confidence: float | None = None
    summary: str = ""
    evidence_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.company_id:
            raise ValueError("CompanyProspect.company_id is required")
        if not self.name:
            raise ValueError("CompanyProspect.name is required")
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class ContactCandidate:
    """Tier 2 company-scoped person/contact candidate."""

    contact_id: str
    company_id: str
    name: str
    role: str = ""
    title: str = ""
    email: str = ""
    profile_url: str = ""
    confidence: float | None = None
    evidence_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.contact_id:
            raise ValueError("ContactCandidate.contact_id is required")
        if not self.company_id:
            raise ValueError("ContactCandidate.company_id is required")
        if not self.name:
            raise ValueError("ContactCandidate.name is required")
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class PersonalizationSignal:
    """Tier 3 person/company personalization evidence."""

    signal_id: str
    company_id: str
    contact_id: str = ""
    signal_type: str = "general"
    summary: str = ""
    suggested_outreach_angle: str = ""
    evidence_ids: tuple[str, ...] = ()
    confidence: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.signal_id:
            raise ValueError("PersonalizationSignal.signal_id is required")
        if not self.company_id:
            raise ValueError("PersonalizationSignal.company_id is required")
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class BrowserCapture:
    """Optional rendered-page capture reference used as audit evidence."""

    browser_capture_id: str
    url: str
    company_id: str = ""
    contact_id: str = ""
    captured_at: str = ""
    text_excerpt: str = ""
    screenshot_path: str = ""
    dom_snapshot_path: str = ""
    evidence_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.browser_capture_id:
            raise ValueError("BrowserCapture.browser_capture_id is required")
        if not self.url:
            raise ValueError("BrowserCapture.url is required")
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class TieredResearchRun:
    """Full normalized tiered prospect research artifact model."""

    directive: SearchDirective | Mapping[str, Any]
    companies: tuple[CompanyProspect | Mapping[str, Any], ...] = ()
    contacts: tuple[ContactCandidate | Mapping[str, Any], ...] = ()
    personalization_signals: tuple[PersonalizationSignal | Mapping[str, Any], ...] = ()
    browser_captures: tuple[BrowserCapture | Mapping[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "directive", _coerce_directive(self.directive))
        object.__setattr__(self, "companies", tuple(_coerce_company(v) for v in self.companies))
        object.__setattr__(self, "contacts", tuple(_coerce_contact(v) for v in self.contacts))
        object.__setattr__(
            self,
            "personalization_signals",
            tuple(_coerce_personalization(v) for v in self.personalization_signals),
        )
        object.__setattr__(
            self, "browser_captures", tuple(_coerce_capture(v) for v in self.browser_captures)
        )
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "metadata", dict(self.metadata))
        _validate_relationships(self)

    def to_dict(self) -> dict[str, Any]:
        return build_tiered_artifact_payload(self)


_COMPANY_FIELDS = (
    "company_id",
    "name",
    "domain",
    "website",
    "industry",
    "geography",
    "qualification_status",
    "fit_score",
    "confidence",
    "summary",
    "evidence_ids",
    "warnings",
    "metadata_json",
)
_CONTACT_FIELDS = (
    "contact_id",
    "company_id",
    "name",
    "role",
    "title",
    "email",
    "profile_url",
    "confidence",
    "evidence_ids",
    "warnings",
    "metadata_json",
)
_PERSONALIZATION_FIELDS = (
    "signal_id",
    "company_id",
    "contact_id",
    "signal_type",
    "summary",
    "suggested_outreach_angle",
    "evidence_ids",
    "confidence",
    "metadata_json",
)


def build_tiered_artifact_payload(
    run: TieredResearchRun | Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the deterministic JSON payload for a tiered research run."""

    normalized = _coerce_run(run)
    merged_metadata = dict(normalized.metadata)
    if metadata:
        merged_metadata.update(metadata)
    return {
        "schema_version": TIERED_ARTIFACT_SCHEMA_VERSION,
        "metadata": _jsonable(merged_metadata),
        "directive": normalized.directive.to_dict(),
        "companies": [company.to_dict() for company in normalized.companies],
        "contacts": [contact.to_dict() for contact in normalized.contacts],
        "personalization_signals": [
            signal.to_dict() for signal in normalized.personalization_signals
        ],
        "browser_captures": [capture.to_dict() for capture in normalized.browser_captures],
        "warnings": list(normalized.warnings),
    }


def write_tiered_artifacts(
    run: TieredResearchRun | Mapping[str, Any],
    output_dir: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> TieredArtifactPaths:
    """Write JSON, CSV, and Markdown artifacts for a tiered prospecting run."""

    normalized = _coerce_run(run)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "tiered_prospect_research.json"
    companies_csv_path = output_path / "companies.csv"
    contacts_csv_path = output_path / "contacts.csv"
    personalization_csv_path = output_path / "personalization.csv"
    markdown_path = output_path / "research_report.md"

    json_path.write_text(
        json.dumps(build_tiered_artifact_payload(normalized, metadata=metadata), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    _write_csv(companies_csv_path, _COMPANY_FIELDS, (_company_row(c) for c in normalized.companies))
    _write_csv(contacts_csv_path, _CONTACT_FIELDS, (_contact_row(c) for c in normalized.contacts))
    _write_csv(
        personalization_csv_path,
        _PERSONALIZATION_FIELDS,
        (_personalization_row(s) for s in normalized.personalization_signals),
    )
    markdown_path.write_text(_render_markdown_report(normalized), encoding="utf-8")

    return TieredArtifactPaths(
        json_path=json_path,
        companies_csv_path=companies_csv_path,
        contacts_csv_path=contacts_csv_path,
        personalization_csv_path=personalization_csv_path,
        markdown_path=markdown_path,
        company_count=len(normalized.companies),
        contact_count=len(normalized.contacts),
        personalization_count=len(normalized.personalization_signals),
    )


def _coerce_run(value: TieredResearchRun | Mapping[str, Any]) -> TieredResearchRun:
    if isinstance(value, TieredResearchRun):
        return value
    return TieredResearchRun(
        directive=_required(value, "directive"),
        companies=tuple(value.get("companies") or ()),
        contacts=tuple(value.get("contacts") or ()),
        personalization_signals=tuple(
            value.get("personalization_signals") or value.get("personalization") or ()
        ),
        browser_captures=tuple(value.get("browser_captures") or ()),
        warnings=tuple(value.get("warnings") or ()),
        metadata=dict(value.get("metadata") or {}),
    )


def _coerce_directive(value: SearchDirective | Mapping[str, Any]) -> SearchDirective:
    if isinstance(value, SearchDirective):
        return value
    return SearchDirective(
        industry=str(_required(value, "industry")),
        geography=str(_required(value, "geography")),
        target_count=int(_required(value, "target_count")),
        criteria=tuple(value.get("criteria") or ()),
        raw_query=str(value.get("raw_query") or ""),
        metadata=dict(value.get("metadata") or {}),
    )


def _coerce_company(value: CompanyProspect | Mapping[str, Any]) -> CompanyProspect:
    if isinstance(value, CompanyProspect):
        return value
    return CompanyProspect(
        company_id=str(_required(value, "company_id")),
        name=str(_required(value, "name")),
        domain=str(value.get("domain") or ""),
        website=str(value.get("website") or ""),
        industry=str(value.get("industry") or ""),
        geography=str(value.get("geography") or ""),
        qualification_status=str(value.get("qualification_status") or "qualified"),
        fit_score=_optional_float(value.get("fit_score")),
        confidence=_optional_float(value.get("confidence")),
        summary=str(value.get("summary") or ""),
        evidence_ids=tuple(value.get("evidence_ids") or ()),
        warnings=tuple(value.get("warnings") or ()),
        metadata=dict(value.get("metadata") or {}),
    )


def _coerce_contact(value: ContactCandidate | Mapping[str, Any]) -> ContactCandidate:
    if isinstance(value, ContactCandidate):
        return value
    return ContactCandidate(
        contact_id=str(_required(value, "contact_id")),
        company_id=str(_required(value, "company_id")),
        name=str(_required(value, "name")),
        role=str(value.get("role") or ""),
        title=str(value.get("title") or ""),
        email=str(value.get("email") or ""),
        profile_url=str(value.get("profile_url") or ""),
        confidence=_optional_float(value.get("confidence")),
        evidence_ids=tuple(value.get("evidence_ids") or ()),
        warnings=tuple(value.get("warnings") or ()),
        metadata=dict(value.get("metadata") or {}),
    )


def _coerce_personalization(
    value: PersonalizationSignal | Mapping[str, Any],
) -> PersonalizationSignal:
    if isinstance(value, PersonalizationSignal):
        return value
    return PersonalizationSignal(
        signal_id=str(_required(value, "signal_id")),
        company_id=str(_required(value, "company_id")),
        contact_id=str(value.get("contact_id") or ""),
        signal_type=str(value.get("signal_type") or "general"),
        summary=str(value.get("summary") or ""),
        suggested_outreach_angle=str(value.get("suggested_outreach_angle") or ""),
        evidence_ids=tuple(value.get("evidence_ids") or ()),
        confidence=_optional_float(value.get("confidence")),
        metadata=dict(value.get("metadata") or {}),
    )


def _coerce_capture(value: BrowserCapture | Mapping[str, Any]) -> BrowserCapture:
    if isinstance(value, BrowserCapture):
        return value
    return BrowserCapture(
        browser_capture_id=str(_required(value, "browser_capture_id")),
        url=str(_required(value, "url")),
        company_id=str(value.get("company_id") or ""),
        contact_id=str(value.get("contact_id") or ""),
        captured_at=str(value.get("captured_at") or ""),
        text_excerpt=str(value.get("text_excerpt") or ""),
        screenshot_path=str(value.get("screenshot_path") or ""),
        dom_snapshot_path=str(value.get("dom_snapshot_path") or ""),
        evidence_ids=tuple(value.get("evidence_ids") or ()),
        metadata=dict(value.get("metadata") or {}),
    )


def _validate_relationships(run: TieredResearchRun) -> None:
    company_ids = {company.company_id for company in run.companies}
    contact_ids = {contact.contact_id for contact in run.contacts}
    for contact in run.contacts:
        if contact.company_id not in company_ids:
            raise ValueError(f"ContactCandidate.company_id is unknown: {contact.company_id}")
    for signal in run.personalization_signals:
        if signal.company_id not in company_ids:
            raise ValueError(f"PersonalizationSignal.company_id is unknown: {signal.company_id}")
        if signal.contact_id and signal.contact_id not in contact_ids:
            raise ValueError(f"PersonalizationSignal.contact_id is unknown: {signal.contact_id}")
    for capture in run.browser_captures:
        if capture.company_id and capture.company_id not in company_ids:
            raise ValueError(f"BrowserCapture.company_id is unknown: {capture.company_id}")
        if capture.contact_id and capture.contact_id not in contact_ids:
            raise ValueError(f"BrowserCapture.contact_id is unknown: {capture.contact_id}")


def _required(value: Mapping[str, Any], key: str) -> Any:
    item = value.get(key)
    if item in (None, ""):
        raise ValueError(f"{key} is required")
    return item


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_jsonable(value), sort_keys=True)
    return str(value)


def _company_row(company: CompanyProspect) -> dict[str, Any]:
    return {
        "company_id": company.company_id,
        "name": company.name,
        "domain": company.domain,
        "website": company.website,
        "industry": company.industry,
        "geography": company.geography,
        "qualification_status": company.qualification_status,
        "fit_score": company.fit_score,
        "confidence": company.confidence,
        "summary": company.summary,
        "evidence_ids": company.evidence_ids,
        "warnings": company.warnings,
        "metadata_json": dict(company.metadata),
    }


def _contact_row(contact: ContactCandidate) -> dict[str, Any]:
    return {
        "contact_id": contact.contact_id,
        "company_id": contact.company_id,
        "name": contact.name,
        "role": contact.role,
        "title": contact.title,
        "email": contact.email,
        "profile_url": contact.profile_url,
        "confidence": contact.confidence,
        "evidence_ids": contact.evidence_ids,
        "warnings": contact.warnings,
        "metadata_json": dict(contact.metadata),
    }


def _personalization_row(signal: PersonalizationSignal) -> dict[str, Any]:
    return {
        "signal_id": signal.signal_id,
        "company_id": signal.company_id,
        "contact_id": signal.contact_id,
        "signal_type": signal.signal_type,
        "summary": signal.summary,
        "suggested_outreach_angle": signal.suggested_outreach_angle,
        "evidence_ids": signal.evidence_ids,
        "confidence": signal.confidence,
        "metadata_json": dict(signal.metadata),
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]] | Any) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_cell(row.get(field, "")) for field in fieldnames})


def _render_markdown_report(run: TieredResearchRun) -> str:
    contacts_by_company: dict[str, list[ContactCandidate]] = {c.company_id: [] for c in run.companies}
    for contact in run.contacts:
        contacts_by_company.setdefault(contact.company_id, []).append(contact)

    signals_by_company: dict[str, list[PersonalizationSignal]] = {
        c.company_id: [] for c in run.companies
    }
    signals_by_contact: dict[str, list[PersonalizationSignal]] = {}
    for signal in run.personalization_signals:
        signals_by_company.setdefault(signal.company_id, []).append(signal)
        if signal.contact_id:
            signals_by_contact.setdefault(signal.contact_id, []).append(signal)

    lines = [
        "# Tiered Prospect Research Report",
        "",
        "## Directive",
        "",
        f"- **Industry**: {_escape_markdown(run.directive.industry)}",
        f"- **Geography**: {_escape_markdown(run.directive.geography)}",
        f"- **Target count**: {run.directive.target_count}",
    ]
    if run.directive.criteria:
        lines.append(f"- **Criteria**: {_escape_markdown('; '.join(run.directive.criteria))}")
    if run.directive.raw_query:
        lines.append(f"- **Raw query**: {_escape_markdown(run.directive.raw_query)}")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Companies: {len(run.companies)}",
            f"- Contacts: {len(run.contacts)}",
            f"- Personalization signals: {len(run.personalization_signals)}",
            f"- Browser captures: {len(run.browser_captures)}",
            "",
            "## Qualified Companies",
        ]
    )

    for company in run.companies:
        lines.extend(["", f"### Company: {_escape_markdown(company.name)}", ""])
        if company.website:
            lines.append(f"- Website: {company.website}")
        if company.domain:
            lines.append(f"- Domain: {_escape_markdown(company.domain)}")
        lines.append(f"- Status: {_escape_markdown(company.qualification_status)}")
        if company.fit_score is not None:
            lines.append(f"- Fit score: {company.fit_score}")
        if company.confidence is not None:
            lines.append(f"- Confidence: {company.confidence}")
        if company.summary:
            lines.extend(["", "#### Fit Evidence", "", _escape_markdown(company.summary)])
        if company.evidence_ids:
            lines.extend(["", "Evidence:"])
            lines.extend(f"- `{evidence_id}`" for evidence_id in company.evidence_ids)

        company_contacts = contacts_by_company.get(company.company_id, [])
        if company_contacts:
            lines.extend(["", "#### Contact Candidates"])
        for contact in company_contacts:
            heading = contact.name
            if contact.role or contact.title:
                heading += f", {contact.role or contact.title}"
            lines.extend(["", f"##### Contact: {_escape_markdown(heading)}", ""])
            if contact.profile_url:
                lines.append(f"- Profile: {contact.profile_url}")
            if contact.confidence is not None:
                lines.append(f"- Confidence: {contact.confidence}")
            contact_signals = signals_by_contact.get(contact.contact_id, [])
            if contact_signals:
                lines.extend(["", "###### Personalization Signals"])
                for signal in contact_signals:
                    lines.append(f"- {_escape_markdown(signal.summary)}")
                    if signal.suggested_outreach_angle:
                        lines.append(
                            f"  - Suggested outreach angle: "
                            f"{_escape_markdown(signal.suggested_outreach_angle)}"
                        )

        company_level_signals = [
            signal for signal in signals_by_company.get(company.company_id, []) if not signal.contact_id
        ]
        if company_level_signals:
            lines.extend(["", "#### Company Personalization Signals"])
            for signal in company_level_signals:
                lines.append(f"- {_escape_markdown(signal.summary)}")

    if run.warnings:
        lines.extend(["", "## Warnings and Human Review Items", ""])
        lines.extend(f"- {_escape_markdown(warning)}" for warning in run.warnings)

    return "\n".join(lines).rstrip() + "\n"


def _escape_markdown(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", "<br>")
