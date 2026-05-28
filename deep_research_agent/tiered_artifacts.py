"""Tiered prospect research artifact writers.

The artifact layer consumes the canonical contracts from ``tiered_models`` and
projects them into JSON, CSV, and Markdown operator artifacts. It accepts mapping
payloads at the boundary for CLI/runtime convenience, but normalizes those
payloads into the shared model classes before writing anything.
"""

from __future__ import annotations

import csv
import html
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, NamedTuple, cast

from .tiered_models import (
    BrowserCapture,
    BrowserCaptureMode,
    CompanyProspect,
    ContactCandidate,
    ContactPersonalization,
    EvidenceDepth,
    FinalEnrichmentRecord,
    PersonalizationSignal,
    RoleCategory,
    SearchDirective,
    SignalConfidence,
    SourceConfidence,
    TieredResearchRun,
)

TIERED_ARTIFACT_SCHEMA_VERSION = "tiered.prospect_artifacts.v1"


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
    final_enrichment_csv_path: Path | None = None
    final_enrichment_count: int = 0


_COMPANY_FIELDS = (
    "company_id",
    "name",
    "website",
    "industry",
    "geographic_area",
    "locations",
    "fit_score",
    "fit_rationale",
    "disqualification_flags",
    "evidence_ids",
    "source_confidence",
)
_CONTACT_FIELDS = (
    "contact_id",
    "company_id",
    "name",
    "title",
    "role_category",
    "profile_urls",
    "email",
    "phone",
    "contact_confidence",
    "evidence_ids",
    "notes",
)
_PERSONALIZATION_FIELDS = (
    "contact_id",
    "signal",
    "message_angle",
    "confidence",
    "evidence_ids",
    "suggested_opening_line",
    "do_not_claim",
)
_FINAL_ENRICHMENT_FIELDS = (
    "enrichment_id",
    "company_id",
    "contact_id",
    "provider",
    "summary",
    "evidence_ids",
    "warnings",
)
_QUALIFICATION_COUNT_KEYS = (
    "ready_contact_count",
    "qualified_company_count",
    "needs_contact_count",
    "rejected_candidate_count",
)


def build_tiered_artifact_payload(
    run: TieredResearchRun | Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
    final_enrichment_records: Sequence[FinalEnrichmentRecord | Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return the deterministic JSON payload for a tiered research run."""

    normalized = _coerce_run(run)
    final_enrichment = tuple(_coerce_final_enrichment(item) for item in final_enrichment_records)
    merged_metadata = dict(metadata or {})
    qualification_audit = _qualification_audit_records(run, merged_metadata)
    qualification_summary = _qualification_summary(
        normalized,
        qualification_audit,
        merged_metadata,
    )
    merged_metadata.update(qualification_summary)
    return {
        "schema_version": TIERED_ARTIFACT_SCHEMA_VERSION,
        "metadata": _jsonable(merged_metadata),
        "run_id": normalized.run_id,
        "directive": normalized.directive.to_dict(),
        "companies": [company.to_dict() for company in normalized.companies],
        "contacts": [contact.to_dict() for contact in normalized.contacts],
        "personalizations": [
            personalization.to_dict() for personalization in normalized.personalizations
        ],
        "personalization_signals": [
            {
                **signal.to_dict(),
                "suggested_opening_line": personalization.suggested_opening_line,
                "do_not_claim": list(personalization.do_not_claim),
            }
            for personalization in normalized.personalizations
            for signal in personalization.personalization_signals
        ],
        "browser_captures": [capture.to_dict() for capture in normalized.browser_captures],
        "final_enrichment": [record.to_dict() for record in final_enrichment],
        "qualification_audit": qualification_audit,
        "warnings": list(normalized.warnings),
    }


def write_tiered_artifacts(
    run: TieredResearchRun | Mapping[str, Any],
    output_dir: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
    final_enrichment_records: Sequence[FinalEnrichmentRecord | Mapping[str, Any]] = (),
) -> TieredArtifactPaths:
    """Write JSON, CSV, and Markdown artifacts for a tiered prospecting run."""

    normalized = _coerce_run(run)
    final_enrichment = tuple(_coerce_final_enrichment(item) for item in final_enrichment_records)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "tiered_prospect_research.json"
    companies_csv_path = output_path / "companies.csv"
    contacts_csv_path = output_path / "contacts.csv"
    personalization_csv_path = output_path / "personalization.csv"
    markdown_path = output_path / "research_report.md"
    final_enrichment_csv_path = (
        output_path / "final_enrichment.csv" if final_enrichment else None
    )

    json_path.write_text(
        json.dumps(
            build_tiered_artifact_payload(
                normalized,
                metadata=metadata,
                final_enrichment_records=final_enrichment,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(companies_csv_path, _COMPANY_FIELDS, (_company_row(c) for c in normalized.companies))
    _write_csv(contacts_csv_path, _CONTACT_FIELDS, (_contact_row(c) for c in normalized.contacts))
    _write_csv(
        personalization_csv_path,
        _PERSONALIZATION_FIELDS,
        (
            _personalization_row(personalization, signal)
            for personalization in normalized.personalizations
            for signal in personalization.personalization_signals
        ),
    )
    if final_enrichment_csv_path is not None:
        _write_csv(
            final_enrichment_csv_path,
            _FINAL_ENRICHMENT_FIELDS,
            (_final_enrichment_row(record) for record in final_enrichment),
        )
    markdown_path.write_text(
        _render_markdown_report(
            normalized,
            final_enrichment,
            qualification_audit=_qualification_audit_records(run, dict(metadata or {})),
            qualification_summary=_qualification_summary(
                normalized,
                _qualification_audit_records(run, dict(metadata or {})),
                dict(metadata or {}),
            ),
        ),
        encoding="utf-8",
    )

    return TieredArtifactPaths(
        json_path=json_path,
        companies_csv_path=companies_csv_path,
        contacts_csv_path=contacts_csv_path,
        personalization_csv_path=personalization_csv_path,
        markdown_path=markdown_path,
        company_count=len(normalized.companies),
        contact_count=len(normalized.contacts),
        personalization_count=sum(
            len(item.personalization_signals) for item in normalized.personalizations
        ),
        final_enrichment_csv_path=final_enrichment_csv_path,
        final_enrichment_count=len(final_enrichment),
    )


def _coerce_run(value: TieredResearchRun | Mapping[str, Any]) -> TieredResearchRun:
    if isinstance(value, TieredResearchRun):
        return value
    return TieredResearchRun(
        run_id=str(value.get("run_id") or "tiered-run"),
        directive=_coerce_directive(cast(Mapping[str, Any], _required(value, "directive"))),
        companies=tuple(_coerce_company(item) for item in value.get("companies") or ()),
        contacts=tuple(_coerce_contact(item) for item in value.get("contacts") or ()),
        personalizations=_coerce_personalizations(value),
        browser_captures=tuple(
            _coerce_capture(item) for item in value.get("browser_captures") or ()
        ),
        warnings=tuple(value.get("warnings") or ()),
    )


def _coerce_directive(value: SearchDirective | Mapping[str, Any]) -> SearchDirective:
    if isinstance(value, SearchDirective):
        return value
    return SearchDirective(
        industry=str(_required(value, "industry")),
        target_prospect_count=int(
            value.get("target_prospect_count") or value.get("target_count") or 0
        ),
        geographic_area=str(value.get("geographic_area") or value.get("geography") or ""),
        research_criteria=str(
            value.get("research_criteria")
            or _criteria_text(value.get("criteria"))
            or value.get("raw_query")
            or ""
        ),
        negative_criteria=str(value.get("negative_criteria") or ""),
        preferred_contact_roles=tuple(value.get("preferred_contact_roles") or ()),
        source_preferences=tuple(value.get("source_preferences") or ()),
        evidence_depth=cast(EvidenceDepth, str(value.get("evidence_depth") or "standard")),
        browser_capture=cast(BrowserCaptureMode, str(value.get("browser_capture") or "none")),
    )


def _coerce_company(value: CompanyProspect | Mapping[str, Any]) -> CompanyProspect:
    if isinstance(value, CompanyProspect):
        return value
    company_id = str(_required(value, "company_id"))
    return CompanyProspect(
        company_id=company_id,
        name=str(_required(value, "name")),
        evidence_ids=_evidence_ids(value, f"company {company_id}"),
        website=str(value.get("website") or value.get("url") or ""),
        industry=str(value.get("industry") or ""),
        geographic_area=str(value.get("geographic_area") or value.get("geography") or ""),
        locations=tuple(value.get("locations") or ()),
        fit_score=_coerce_float(value.get("fit_score"), default=0.0),
        fit_rationale=str(value.get("fit_rationale") or value.get("summary") or ""),
        disqualification_flags=tuple(value.get("disqualification_flags") or ()),
        source_confidence=cast(SourceConfidence, str(value.get("source_confidence") or "medium")),
    )


def _coerce_contact(value: ContactCandidate | Mapping[str, Any]) -> ContactCandidate:
    if isinstance(value, ContactCandidate):
        return value
    contact_id = str(_required(value, "contact_id"))
    return ContactCandidate(
        contact_id=contact_id,
        company_id=str(_required(value, "company_id")),
        name=str(_required(value, "name")),
        evidence_ids=_evidence_ids(value, f"contact {contact_id}"),
        title=str(value.get("title") or value.get("role") or ""),
        role_category=cast(RoleCategory, str(value.get("role_category") or "unknown")),
        profile_urls=tuple(value.get("profile_urls") or _optional_single(value.get("profile_url"))),
        email=_optional_str(value.get("email")),
        phone=_optional_str(value.get("phone")),
        contact_confidence=_coerce_float(
            value.get("contact_confidence", value.get("confidence")),
            default=0.0,
        ),
        notes=str(value.get("notes") or ""),
    )


def _coerce_personalizations(value: Mapping[str, Any]) -> tuple[ContactPersonalization, ...]:
    raw_personalizations = value.get("personalizations")
    if raw_personalizations:
        return tuple(_coerce_contact_personalization(item) for item in raw_personalizations)

    grouped: dict[str, list[PersonalizationSignal]] = {}
    for item in value.get("personalization_signals") or value.get("personalization") or ():
        mapping = cast(Mapping[str, Any], item)
        contact_id = str(_required(mapping, "contact_id"))
        grouped.setdefault(contact_id, []).append(_coerce_signal(mapping))
    return tuple(
        ContactPersonalization(contact_id=contact_id, personalization_signals=tuple(signals))
        for contact_id, signals in grouped.items()
    )


def _coerce_contact_personalization(
    value: ContactPersonalization | Mapping[str, Any],
) -> ContactPersonalization:
    if isinstance(value, ContactPersonalization):
        return value
    contact_id = str(_required(value, "contact_id"))
    return ContactPersonalization(
        contact_id=contact_id,
        personalization_signals=tuple(
            _coerce_signal({**cast(Mapping[str, Any], item), "contact_id": contact_id})
            for item in value.get("personalization_signals") or ()
        ),
        suggested_opening_line=str(value.get("suggested_opening_line") or ""),
        do_not_claim=tuple(value.get("do_not_claim") or ()),
    )


def _coerce_signal(value: PersonalizationSignal | Mapping[str, Any]) -> PersonalizationSignal:
    if isinstance(value, PersonalizationSignal):
        return value
    contact_id = str(_required(value, "contact_id"))
    signal_text = str(value.get("signal") or value.get("summary") or "")
    return PersonalizationSignal(
        contact_id=contact_id,
        signal=signal_text,
        message_angle=str(
            value.get("message_angle") or value.get("suggested_outreach_angle") or signal_text
        ),
        evidence_ids=_evidence_ids(value, f"personalization for contact {contact_id}"),
        confidence=cast(SignalConfidence, str(value.get("confidence") or "medium")),
    )


def _coerce_capture(value: BrowserCapture | Mapping[str, Any]) -> BrowserCapture:
    if isinstance(value, BrowserCapture):
        return value
    capture_id = str(_required(value, "browser_capture_id"))
    return BrowserCapture(
        browser_capture_id=capture_id,
        url=str(_required(value, "url")),
        captured_at=str(_required(value, "captured_at")),
        evidence_ids=_evidence_ids(value, f"browser capture {capture_id}"),
        company_id=str(value.get("company_id") or ""),
        contact_id=str(value.get("contact_id") or ""),
        text_excerpt=str(value.get("text_excerpt") or ""),
        screenshot_path=str(value.get("screenshot_path") or ""),
        dom_snapshot_path=str(value.get("dom_snapshot_path") or ""),
    )


def _coerce_final_enrichment(
    value: FinalEnrichmentRecord | Mapping[str, Any],
) -> FinalEnrichmentRecord:
    if isinstance(value, FinalEnrichmentRecord):
        return value
    enrichment_id = str(value.get("enrichment_id") or value.get("id") or "")
    if not enrichment_id:
        enrichment_id = (
            f"enrichment_{value.get('company_id', 'unknown')}_{value.get('contact_id', '')}"
        )
    return FinalEnrichmentRecord(
        enrichment_id=enrichment_id,
        company_id=str(_required(value, "company_id")),
        contact_id=str(value.get("contact_id") or ""),
        summary=str(value.get("summary") or ""),
        evidence_ids=_evidence_ids(value, f"final enrichment {enrichment_id}"),
        provider=str(value.get("provider") or "mock"),
        warnings=tuple(value.get("warnings") or ()),
    )


def _required(value: Mapping[str, Any], key: str) -> Any:
    item = value.get(key)
    if item in (None, ""):
        raise ValueError(f"{key} is required")
    return item


def _criteria_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        return "; ".join(str(item) for item in value if str(item))
    return str(value)


def _evidence_ids(value: Mapping[str, Any], field_name: str) -> tuple[str, ...]:
    raw = value.get("evidence_ids") or value.get("evidence_id") or ()
    if isinstance(raw, str):
        if raw:
            return (raw,)
        raise ValueError(f"{field_name} requires evidence_ids")
    normalized = tuple(str(item) for item in raw if str(item))
    if not normalized:
        raise ValueError(f"{field_name} requires evidence_ids")
    return normalized


def _coerce_float(value: Any, *, default: float) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _optional_single(value: Any) -> tuple[str, ...]:
    return (str(value),) if value else ()


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


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
        "website": company.website,
        "industry": company.industry,
        "geographic_area": company.geographic_area,
        "locations": company.locations,
        "fit_score": company.fit_score,
        "fit_rationale": company.fit_rationale,
        "disqualification_flags": company.disqualification_flags,
        "evidence_ids": company.evidence_ids,
        "source_confidence": company.source_confidence,
    }


def _contact_row(contact: ContactCandidate) -> dict[str, Any]:
    return {
        "contact_id": contact.contact_id,
        "company_id": contact.company_id,
        "name": contact.name,
        "title": contact.title,
        "role_category": contact.role_category,
        "profile_urls": contact.profile_urls,
        "email": contact.email,
        "phone": contact.phone,
        "contact_confidence": contact.contact_confidence,
        "evidence_ids": contact.evidence_ids,
        "notes": contact.notes,
    }


def _personalization_row(
    personalization: ContactPersonalization,
    signal: PersonalizationSignal,
) -> dict[str, Any]:
    return {
        "contact_id": signal.contact_id,
        "signal": signal.signal,
        "message_angle": signal.message_angle,
        "confidence": signal.confidence,
        "evidence_ids": signal.evidence_ids,
        "suggested_opening_line": personalization.suggested_opening_line,
        "do_not_claim": personalization.do_not_claim,
    }


def _final_enrichment_row(record: FinalEnrichmentRecord) -> dict[str, Any]:
    return {
        "enrichment_id": record.enrichment_id,
        "company_id": record.company_id,
        "contact_id": record.contact_id,
        "provider": record.provider,
        "summary": record.summary,
        "evidence_ids": record.evidence_ids,
        "warnings": record.warnings,
    }


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]] | Any
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_cell(row.get(field, "")) for field in fieldnames})


def _render_markdown_report(
    run: TieredResearchRun,
    final_enrichment: Sequence[FinalEnrichmentRecord],
) -> str:
    contacts_by_company: dict[str, list[ContactCandidate]] = {
        company.company_id: [] for company in run.companies
    }
    for contact in run.contacts:
        contacts_by_company.setdefault(contact.company_id, []).append(contact)

    personalizations_by_contact = {
        personalization.contact_id: personalization for personalization in run.personalizations
    }

    lines = [
        "# Tiered Prospect Research Report",
        "",
        "## Directive",
        "",
        f"- **Industry**: {_escape_markdown(run.directive.industry)}",
        f"- **Geography**: {_escape_markdown(run.directive.geographic_area)}",
        f"- **Target count**: {run.directive.target_prospect_count}",
        f"- **Criteria**: {_escape_markdown(run.directive.research_criteria)}",
        "",
        "## Summary",
        "",
        f"- Companies: {len(run.companies)}",
        f"- Contacts: {len(run.contacts)}",
        f"- Personalization signals: "
        f"{sum(len(item.personalization_signals) for item in run.personalizations)}",
        f"- Browser captures: {len(run.browser_captures)}",
        f"- Final enrichment records: {len(final_enrichment)}",
        "",
        "## Qualified Companies",
    ]

    for company in run.companies:
        lines.extend(["", f"### Company: {_escape_markdown(company.name)}", ""])
        if company.website:
            lines.append(f"- Website: {company.website}")
        lines.append(f"- Fit score: {company.fit_score}")
        lines.append(f"- Source confidence: {_escape_markdown(company.source_confidence)}")
        if company.fit_rationale:
            lines.extend(["", "#### Fit Evidence", "", _escape_markdown(company.fit_rationale)])
        if company.evidence_ids:
            lines.extend(["", "Evidence:"])
            lines.extend(f"- `{evidence_id}`" for evidence_id in company.evidence_ids)

        company_contacts = contacts_by_company.get(company.company_id, [])
        if company_contacts:
            lines.extend(["", "#### Contact Candidates"])
        for contact in company_contacts:
            heading = contact.name
            if contact.title:
                heading += f", {contact.title}"
            lines.extend(["", f"##### Contact: {_escape_markdown(heading)}", ""])
            if contact.profile_urls:
                lines.append(f"- Profile: {contact.profile_urls[0]}")
            lines.append(f"- Confidence: {contact.contact_confidence}")
            personalization = personalizations_by_contact.get(contact.contact_id)
            if personalization and personalization.personalization_signals:
                lines.extend(["", "###### Personalization Signals"])
                for signal in personalization.personalization_signals:
                    lines.append(f"- {_escape_markdown(signal.signal)}")
                    if signal.message_angle:
                        lines.append(
                            f"  - Message angle: {_escape_markdown(signal.message_angle)}"
                        )
                if personalization.do_not_claim:
                    lines.append(
                        f"  - Do not claim: "
                        f"{_escape_markdown('; '.join(personalization.do_not_claim))}"
                    )

    if final_enrichment:
        lines.extend(["", "## Final Enrichment", ""])
        for record in final_enrichment:
            lines.append(
                f"- `{record.enrichment_id}` for `{record.company_id}`"
                f"{' / `' + record.contact_id + '`' if record.contact_id else ''}: "
                f"{_escape_markdown(record.summary)}"
            )

    if run.warnings:
        lines.extend(["", "## Warnings and Human Review Items", ""])
        lines.extend(f"- {_escape_markdown(warning)}" for warning in run.warnings)

    return "\n".join(lines).rstrip() + "\n"


def _escape_markdown(value: Any) -> str:
    return html.escape(str(value), quote=False).replace("|", r"\|").replace("\n", "<br>")


__all__ = [
    "BrowserCapture",
    "CompanyProspect",
    "ContactCandidate",
    "ContactPersonalization",
    "FinalEnrichmentRecord",
    "PersonalizationSignal",
    "SearchDirective",
    "TIERED_ARTIFACT_SCHEMA_VERSION",
    "TieredArtifactPaths",
    "TieredResearchRun",
    "build_tiered_artifact_payload",
    "write_tiered_artifacts",
]
