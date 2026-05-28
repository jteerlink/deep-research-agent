"""Local tiered prospect run/resume/inspect runtime.

This sidecar runtime gives the tiered workflow a durable, offline-testable
execution surface without changing the existing generic G003 graph. Network,
browser, model, and final-enrichment providers stay outside this module; tests
and demos inject mock records through CLI flags or JSON files.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from .geography import normalize_geography
from .tiered_artifacts import build_tiered_artifact_payload, write_tiered_artifacts
from .tiered_models import (
    ApprovedProspectSelection,
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
    TieredWorkflowStatus,
)
from .tiered_qualification import (
    QualificationAuditRecord,
    QualifiedCompanyBatch,
    QualifiedContactBatch,
    qualify_company_hits,
    qualify_contact_hits,
)
from .tiered_search import (
    CompanySearchTarget,
    ProviderPolicy,
    SearchCallable,
    TieredSearchDirective,
    build_contact_discovery_queries,
    collect_company_discovery_search,
    collect_tiered_search,
)

TIERED_CHECKPOINT_SCHEMA_VERSION = "tiered.prospect_checkpoint.v1"
DEFAULT_TIERED_CHECKPOINT_DIR = ".deep_research_agent/tiered_checkpoints"
DEFAULT_TIERED_ARTIFACT_DIR = ".deep_research_agent/tiered_artifacts"
_THREAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
LIVE_COMPANY_OVERSAMPLE_FACTOR = 4
LIVE_COMPANY_DISCOVERY_POOL_FACTOR = 3
LIVE_CONTACTS_PER_COMPANY = 2
MAX_LIVE_COMPANY_CANDIDATES = 50
FinalEnrichmentFactory = Callable[
    [TieredResearchRun, ApprovedProspectSelection],
    Sequence[FinalEnrichmentRecord],
]


@dataclass(frozen=True)
class TieredCheckpoint:
    """Durable local checkpoint for the additive tiered workflow."""

    thread_id: str
    status: TieredWorkflowStatus
    run: TieredResearchRun
    artifact_paths: dict[str, str] = field(default_factory=dict)
    approval: ApprovedProspectSelection | None = None
    final_enrichment: tuple[FinalEnrichmentRecord, ...] = ()
    events: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    qualification_audit: tuple[dict[str, Any], ...] = ()
    qualification_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = build_tiered_artifact_payload(
            self.run,
            metadata={
                "checkpoint_schema_version": TIERED_CHECKPOINT_SCHEMA_VERSION,
                "thread_id": self.thread_id,
                "status": self.status,
                "approval_state": self.approval.to_dict() if self.approval else None,
                "artifact_paths": self.artifact_paths,
                "events": list(self.events),
                "qualification_audit": list(self.qualification_audit),
                "qualification_counts": dict(self.qualification_counts),
            },
            final_enrichment_records=self.final_enrichment,
        )
        payload.update(
            {
                "checkpoint_schema_version": TIERED_CHECKPOINT_SCHEMA_VERSION,
                "thread_id": self.thread_id,
                "status": self.status,
                "approval": self.approval.to_dict() if self.approval else None,
                "artifact_paths": dict(self.artifact_paths),
                "events": list(self.events),
                "warnings": list(self.warnings or self.run.warnings),
                "qualification_audit": list(self.qualification_audit),
                "qualification_counts": dict(self.qualification_counts),
            }
        )
        return payload


def parse_directive_payload(payload: dict[str, Any]) -> SearchDirective:
    """Parse a JSON/CLI directive payload into the canonical directive model."""

    geography_raw = str(payload.get("geographic_area") or payload.get("geography") or "")
    geo = normalize_geography(geography_raw)
    preferred_roles = payload.get("preferred_contact_roles") or payload.get("preferred_roles") or ()
    source_preferences = payload.get("source_preferences") or ()
    return SearchDirective(
        industry=str(payload.get("industry") or ""),
        target_prospect_count=int(
            payload.get("target_prospect_count") or payload.get("target_count") or 0
        ),
        geographic_area=geo.canonical,
        research_criteria=str(payload.get("research_criteria") or payload.get("criteria") or ""),
        negative_criteria=str(payload.get("negative_criteria") or ""),
        preferred_contact_roles=tuple(str(role) for role in preferred_roles if str(role)),
        source_preferences=tuple(str(source) for source in source_preferences if str(source)),
        evidence_depth=cast(EvidenceDepth, str(payload.get("evidence_depth") or "standard")),
        browser_capture=cast(BrowserCaptureMode, str(payload.get("browser_capture") or "none")),
    )


def load_directive_json(path: str | Path) -> SearchDirective:
    """Load a directive JSON file with friendly validation errors."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid directive JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("directive JSON must contain an object")
    return parse_directive_payload(payload)


def run_tiered_research(
    directive: SearchDirective,
    *,
    thread_id: str | None = None,
    checkpoint_dir: str | Path = DEFAULT_TIERED_CHECKPOINT_DIR,
    artifact_dir: str | Path = DEFAULT_TIERED_ARTIFACT_DIR,
    mock_results: tuple[str, ...] = (),
) -> TieredCheckpoint:
    """Create a review-gated tiered checkpoint from injected mock discovery results."""

    active_thread_id = _validate_thread_id(thread_id or f"tiered-{uuid4().hex[:12]}")
    companies = _companies_from_mock_results(mock_results, directive)
    contacts = _contacts_for_companies(companies, directive)
    personalizations = _personalizations_for_contacts(contacts)
    warnings = (
        "No mock company results were provided; no company records were generated."
        if not mock_results
        else "Human review required before final enrichment or outreach use."
    )
    run = TieredResearchRun(
        run_id=active_thread_id,
        directive=directive,
        companies=tuple(companies),
        contacts=tuple(contacts),
        personalizations=tuple(personalizations),
        warnings=(warnings,),
    )
    event = _event("review_required", "tiered run stopped before final enrichment")
    checkpoint = TieredCheckpoint(
        thread_id=active_thread_id,
        status="review_required",
        run=run,
        events=(event,),
        warnings=run.warnings,
    )
    return _persist_checkpoint(
        checkpoint,
        checkpoint_dir=checkpoint_dir,
        artifact_dir=artifact_dir,
    )


async def run_tiered_research_with_search(
    directive: SearchDirective,
    *,
    search: SearchCallable,
    thread_id: str | None = None,
    checkpoint_dir: str | Path = DEFAULT_TIERED_CHECKPOINT_DIR,
    artifact_dir: str | Path = DEFAULT_TIERED_ARTIFACT_DIR,
    max_results: int = 5,
    max_contact_queries_per_company: int = 4,
    provider_policy: ProviderPolicy | None = None,
) -> TieredCheckpoint:
    """Create a review-gated tiered checkpoint from live injected search results."""

    if max_results < 1:
        raise ValueError("max_results must be >= 1")
    if max_contact_queries_per_company < 1:
        raise ValueError("max_contact_queries_per_company must be >= 1")

    active_thread_id = _validate_thread_id(thread_id or f"tiered-{uuid4().hex[:12]}")
    search_directive = _search_directive_from_run_directive(directive)
    policy = provider_policy or ProviderPolicy()
    company_result_cap = _company_candidate_result_cap(directive, max_results)
    company_batch = await collect_company_discovery_search(
        search_directive,
        max_results=company_result_cap,
        search=search,
        provider_policy=policy,
    )
    company_qualification = qualify_company_hits(
        company_batch.hits,
        directive,
        target_count=directive.target_prospect_count,
        oversample_factor=LIVE_COMPANY_OVERSAMPLE_FACTOR,
    )
    company_audit = _company_audit_payloads(company_qualification)
    contact_audit: list[dict[str, Any]] = []
    companies: list[CompanyProspect] = []
    contacts: list[ContactCandidate] = []
    contact_failures = 0
    for candidate in company_qualification.accepted:
        company = candidate.to_company_prospect(directive)
        companies.append(company)
        target = CompanySearchTarget(company.name, website=company.website)
        queries = build_contact_discovery_queries(search_directive, target)[
            :max_contact_queries_per_company
        ]
        contact_batch = await collect_tiered_search(
            "contact_discovery",
            queries,
            max_results=max_results,
            search=search,
            provider_policy=policy,
        )
        contact_failures += len(contact_batch.failures)
        contact_qualification = qualify_contact_hits(
            contact_batch.hits,
            company,
            directive,
            max_contacts=LIVE_CONTACTS_PER_COMPANY,
        )
        company_contacts = [
            candidate.to_contact_candidate()
            for candidate in contact_qualification.accepted
        ]
        contacts.extend(company_contacts)
        contact_audit.extend(
            _contact_audit_payloads(contact_qualification, company)
        )

    qualification_audit = [*company_audit, *contact_audit]
    personalizations = _personalizations_for_contacts(
        contacts,
        source_label="live search discovery",
    )
    warnings = _live_search_warnings(
        company_count=len(companies),
        company_failures=len(company_batch.failures),
        contact_failures=contact_failures,
        needs_contact_count=_count_audit_status(qualification_audit, "needs_contact"),
        rejected_candidate_count=_count_audit_status(qualification_audit, "rejected"),
    )
    qualification_counts = _qualification_counts(
        companies=companies,
        contacts=contacts,
        qualification_audit=qualification_audit,
    )
    run = TieredResearchRun(
        run_id=active_thread_id,
        directive=directive,
        companies=tuple(companies),
        contacts=tuple(contacts),
        personalizations=tuple(personalizations),
        warnings=warnings,
    )
    checkpoint = TieredCheckpoint(
        thread_id=active_thread_id,
        status="review_required",
        run=run,
        events=(
            _event(
                "company_discovery_complete",
                f"{len(company_batch.hits)} company candidate(s) gathered from live search",
                raw_hit_count=len(company_batch.hits),
                result_cap=company_result_cap,
            ),
            _event(
                "company_qualification_complete",
                f"{len(companies)} qualified company record(s) promoted for review",
                accepted_company_count=len(companies),
                rejected_company_count=_count_audit_status(company_audit, "rejected"),
            ),
            _event(
                "contact_qualification_complete",
                f"{len(contacts)} qualified contact-level prospect row(s) promoted for review",
                accepted_contact_count=len(contacts),
                rejected_contact_count=_count_audit_status(contact_audit, "rejected"),
                needs_contact_count=_count_audit_status(contact_audit, "needs_contact"),
            ),
            _event(
                "review_required",
                f"{len(companies)} company prospect row(s) ready for review",
                ready_company_count=len(companies),
                ready_contact_count=len(contacts),
                needs_contact_count=qualification_counts["needs_contact_count"],
            ),
        ),
        warnings=run.warnings,
        qualification_audit=tuple(qualification_audit),
        qualification_counts=qualification_counts,
    )
    return _persist_checkpoint(
        checkpoint,
        checkpoint_dir=checkpoint_dir,
        artifact_dir=artifact_dir,
    )


def resume_tiered_research(
    thread_id: str,
    *,
    checkpoint_dir: str | Path = DEFAULT_TIERED_CHECKPOINT_DIR,
    artifact_dir: str | Path = DEFAULT_TIERED_ARTIFACT_DIR,
    approval_selection: ApprovedProspectSelection | None = None,
    enable_final_enrichment: bool = False,
    final_enrichment_records: Sequence[FinalEnrichmentRecord] | None = None,
    final_enrichment_factory: FinalEnrichmentFactory | None = None,
    mock_final_enrichment: tuple[str, ...] = (),
) -> TieredCheckpoint:
    """Resume a tiered checkpoint, optionally recording approval and enrichment."""

    checkpoint = inspect_tiered_research(thread_id, checkpoint_dir=checkpoint_dir)
    approval = approval_selection or checkpoint.approval
    events = list(checkpoint.events)
    warnings = list(checkpoint.warnings)
    final_enrichment = list(checkpoint.final_enrichment)
    status: TieredWorkflowStatus = checkpoint.status

    if approval_selection is not None:
        _validate_approval(checkpoint.run, approval_selection)
        approval = approval_selection
        status = "approval_recorded"
        events.append(_event("approval_recorded", "human approval payload accepted"))

    if enable_final_enrichment:
        if approval is None:
            status = "final_enrichment_blocked"
            warnings.append("Final enrichment blocked: no approved prospect selection.")
            events.append(_event("final_enrichment_blocked", "missing approved selection"))
        else:
            if final_enrichment_records is not None:
                proposed = tuple(final_enrichment_records)
            elif final_enrichment_factory is not None:
                proposed = tuple(final_enrichment_factory(checkpoint.run, approval))
            else:
                proposed = _final_enrichment_records(
                    mock_final_enrichment,
                    approval=approval,
                    run=checkpoint.run,
                )
            _validate_final_enrichment(proposed, approval, checkpoint.run)
            final_enrichment = list(proposed)
            status = "final_enrichment_complete"
            providers = ", ".join(
                sorted({record.provider for record in final_enrichment if record.provider})
            )
            events.append(
                _event(
                    "final_enrichment_complete",
                    f"{len(final_enrichment)} approved enrichment record(s) stored"
                    + (f" via {providers}" if providers else ""),
                )
            )
    elif approval_selection is None and approval is None:
        events.append(_event("review_required", "resume without approval left state unchanged"))

    next_checkpoint = TieredCheckpoint(
        thread_id=checkpoint.thread_id,
        status=status,
        run=checkpoint.run,
        artifact_paths=checkpoint.artifact_paths,
        approval=approval,
        final_enrichment=tuple(final_enrichment),
        events=tuple(events),
        warnings=tuple(dict.fromkeys(warnings)),
        qualification_audit=checkpoint.qualification_audit,
        qualification_counts=checkpoint.qualification_counts,
    )
    return _persist_checkpoint(
        next_checkpoint,
        checkpoint_dir=checkpoint_dir,
        artifact_dir=artifact_dir,
    )


def inspect_tiered_research(
    thread_id: str,
    *,
    checkpoint_dir: str | Path = DEFAULT_TIERED_CHECKPOINT_DIR,
) -> TieredCheckpoint:
    """Load a tiered checkpoint from disk."""

    path = _checkpoint_path(checkpoint_dir, thread_id)
    if not path.exists():
        raise FileNotFoundError(f"tiered checkpoint not found for thread_id={thread_id}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("checkpoint_schema_version") != TIERED_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported tiered checkpoint schema")
    run = _run_from_payload(payload)
    approval_payload = payload.get("approval")
    approval = _approval_from_payload(approval_payload) if approval_payload else None
    final_enrichment = tuple(
        _final_record_from_payload(item) for item in payload.get("final_enrichment") or ()
    )
    return TieredCheckpoint(
        thread_id=str(payload["thread_id"]),
        status=cast(TieredWorkflowStatus, str(payload["status"])),
        run=run,
        artifact_paths=dict(payload.get("artifact_paths") or {}),
        approval=approval,
        final_enrichment=final_enrichment,
        events=tuple(dict(item) for item in payload.get("events") or ()),
        warnings=tuple(str(item) for item in payload.get("warnings") or ()),
        qualification_audit=tuple(
            dict(item) for item in payload.get("qualification_audit") or ()
        ),
        qualification_counts=_qualification_counts_from_payload(
            payload.get("qualification_counts") or payload.get("metadata") or {}
        ),
    )


def load_approval_selection(path: str | Path) -> ApprovedProspectSelection:
    """Load an approved company/contact selection JSON payload."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid approval JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("approval JSON must contain an object")
    return _approval_from_payload(payload)


def _persist_checkpoint(
    checkpoint: TieredCheckpoint,
    *,
    checkpoint_dir: str | Path,
    artifact_dir: str | Path,
) -> TieredCheckpoint:
    safe_thread_id = _validate_thread_id(checkpoint.thread_id)
    artifact_root = Path(artifact_dir) / safe_thread_id
    artifact_paths = write_tiered_artifacts(
        checkpoint.run,
        artifact_root,
        metadata={
            "checkpoint_schema_version": TIERED_CHECKPOINT_SCHEMA_VERSION,
            "thread_id": checkpoint.thread_id,
            "status": checkpoint.status,
            "approval_state": checkpoint.approval.to_dict() if checkpoint.approval else None,
            "qualification_audit": list(checkpoint.qualification_audit),
            "qualification_counts": dict(checkpoint.qualification_counts),
        },
        final_enrichment_records=checkpoint.final_enrichment,
    )
    artifact_map = {
        "json": str(artifact_paths.json_path),
        "companies_csv": str(artifact_paths.companies_csv_path),
        "contacts_csv": str(artifact_paths.contacts_csv_path),
        "personalization_csv": str(artifact_paths.personalization_csv_path),
        "markdown": str(artifact_paths.markdown_path),
    }
    if artifact_paths.final_enrichment_csv_path:
        artifact_map["final_enrichment_csv"] = str(artifact_paths.final_enrichment_csv_path)
    persisted = TieredCheckpoint(
        thread_id=safe_thread_id,
        status=checkpoint.status,
        run=checkpoint.run,
        artifact_paths=artifact_map,
        approval=checkpoint.approval,
        final_enrichment=checkpoint.final_enrichment,
        events=checkpoint.events,
        warnings=checkpoint.warnings,
        qualification_audit=checkpoint.qualification_audit,
        qualification_counts=checkpoint.qualification_counts,
    )
    checkpoint_path = _checkpoint_path(checkpoint_dir, safe_thread_id)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(persisted.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return persisted


def _checkpoint_path(checkpoint_dir: str | Path, thread_id: str) -> Path:
    safe_thread_id = _validate_thread_id(thread_id)
    return Path(checkpoint_dir) / f"{safe_thread_id}.json"


def _companies_from_mock_results(
    specs: tuple[str, ...],
    directive: SearchDirective,
) -> list[CompanyProspect]:
    if not specs:
        return []
    companies: list[CompanyProspect] = []
    for index, raw in enumerate(specs, start=1):
        title, url, snippet, provider = _split_spec(raw, 4, "--mock-result")
        company_id = f"company_{_slug(title) or index:0>3}"
        companies.append(
            CompanyProspect(
                company_id=company_id,
                name=title,
                website=url,
                industry=directive.industry,
                geographic_area=directive.geographic_area,
                fit_score=0.75,
                fit_rationale=snippet,
                evidence_ids=(f"ev_{company_id}_001",),
                source_confidence="medium" if provider != "mock" else "low",
            )
        )
    return companies[: directive.target_prospect_count]


def _search_directive_from_run_directive(directive: SearchDirective) -> TieredSearchDirective:
    return TieredSearchDirective(
        industry=directive.industry,
        geographic_area=directive.geographic_area,
        target_prospect_count=directive.target_prospect_count,
        research_criteria=directive.research_criteria,
        preferred_contact_roles=directive.preferred_contact_roles,
        source_preferences=directive.source_preferences,
    )


def _company_candidate_result_cap(directive: SearchDirective, max_results: int) -> int:
    requested_pool = max(
        max_results,
        directive.target_prospect_count
        * LIVE_COMPANY_DISCOVERY_POOL_FACTOR
        * LIVE_COMPANY_OVERSAMPLE_FACTOR,
    )
    return min(MAX_LIVE_COMPANY_CANDIDATES, requested_pool)


def _company_audit_payloads(batch: QualifiedCompanyBatch) -> list[dict[str, Any]]:
    accepted_by_evidence_id = {
        candidate.evidence_id: candidate for candidate in batch.accepted
    }
    payloads: list[dict[str, Any]] = []
    for record in batch.audit_records:
        payload = _audit_payload(record)
        candidate = accepted_by_evidence_id.get(record.evidence_id)
        if candidate is not None:
            payload.update(
                {
                    "company_id": candidate.company_id,
                    "company_name": candidate.name,
                }
            )
        elif record.normalized_name:
            payload["company_name"] = record.normalized_name
        payloads.append(payload)
    return payloads


def _contact_audit_payloads(
    batch: QualifiedContactBatch,
    company: CompanyProspect,
) -> list[dict[str, Any]]:
    accepted_by_evidence_id = {
        candidate.evidence_id: candidate for candidate in batch.accepted
    }
    payloads: list[dict[str, Any]] = []
    for record in batch.audit_records:
        payload = _audit_payload(record)
        payload.update(
            {
                "company_id": company.company_id,
                "company_name": company.name,
            }
        )
        candidate = accepted_by_evidence_id.get(record.evidence_id)
        if candidate is not None:
            payload.update(
                {
                    "contact_id": candidate.contact_id,
                    "contact_name": candidate.name,
                    "contact_title": candidate.title,
                }
            )
        elif record.normalized_name and record.normalized_name != company.name:
            payload["contact_name"] = record.normalized_name
        payloads.append(payload)
    return payloads


def _audit_payload(record: QualificationAuditRecord) -> dict[str, Any]:
    payload = record.to_dict()
    payload.setdefault("source_title", record.title)
    payload.setdefault("source_url", record.url)
    return payload


def _qualification_counts(
    *,
    companies: Sequence[CompanyProspect],
    contacts: Sequence[ContactCandidate],
    qualification_audit: Sequence[dict[str, Any]],
) -> dict[str, int]:
    companies_with_contacts = {contact.company_id for contact in contacts}
    return {
        "ready_company_count": len(companies),
        "company_prospect_count": len(companies),
        "ready_contact_count": len(contacts),
        "contact_candidate_count": len(contacts),
        "qualified_company_count": len(companies),
        "companies_with_contacts_count": len(companies_with_contacts),
        "needs_contact_count": _count_audit_status(qualification_audit, "needs_contact"),
        "rejected_candidate_count": _count_audit_status(qualification_audit, "rejected"),
    }


def _qualification_counts_from_payload(payload: Any) -> dict[str, int]:
    if not isinstance(payload, dict):
        return {}
    counts = payload.get("qualification_counts")
    if not isinstance(counts, dict):
        counts = payload
    parsed: dict[str, int] = {}
    for key in (
        "ready_company_count",
        "company_prospect_count",
        "ready_contact_count",
        "contact_candidate_count",
        "qualified_company_count",
        "companies_with_contacts_count",
        "needs_contact_count",
        "rejected_candidate_count",
    ):
        value = counts.get(key)
        if value in (None, ""):
            continue
        try:
            parsed[key] = int(str(value))
        except (TypeError, ValueError):
            continue
    return parsed


def _count_audit_status(
    qualification_audit: Sequence[dict[str, Any]],
    status: str,
) -> int:
    normalized = status.strip().lower().replace("-", "_")
    return sum(
        1
        for record in qualification_audit
        if str(record.get("status") or "").strip().lower().replace("-", "_") == normalized
    )


def _contacts_for_companies(
    companies: list[CompanyProspect],
    directive: SearchDirective,
) -> list[ContactCandidate]:
    role = directive.preferred_contact_roles[0] if directive.preferred_contact_roles else "owner"
    contacts: list[ContactCandidate] = []
    for company in companies:
        contact_id = f"contact_{_slug(company.company_id)}_001"
        contacts.append(
            ContactCandidate(
                contact_id=contact_id,
                company_id=company.company_id,
                name=f"Review Contact at {company.name}",
                title=role,
                role_category=_role_category(role),
                contact_confidence=0.55,
                evidence_ids=(f"ev_{contact_id}_001",),
                notes="Generated from mock company result; requires human review.",
            )
        )
    return contacts


def _personalizations_for_contacts(
    contacts: list[ContactCandidate],
    *,
    source_label: str = "mock discovery",
) -> list[ContactPersonalization]:
    personalizations: list[ContactPersonalization] = []
    for contact in contacts:
        signal_text = (
            contact.notes
            if contact.notes and source_label != "mock discovery"
            else f"Only {source_label} evidence is available."
        )
        message_angle = (
            "Verify live search context before using this for outreach."
            if source_label != "mock discovery"
            else "Use as a review placeholder, not an outreach fact."
        )
        signal = PersonalizationSignal(
            contact_id=contact.contact_id,
            signal=signal_text,
            message_angle=message_angle,
            evidence_ids=(f"ev_{contact.contact_id}_personalization_001",),
            confidence="low",
        )
        personalizations.append(
            ContactPersonalization(
                contact_id=contact.contact_id,
                personalization_signals=(signal,),
                suggested_opening_line="",
                do_not_claim=("Do not claim live verification before final enrichment.",),
            )
        )
    return personalizations


def _live_search_warnings(
    *,
    company_count: int,
    company_failures: int,
    contact_failures: int,
    needs_contact_count: int = 0,
    rejected_candidate_count: int = 0,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if company_count:
        warnings.append("Human review required before final enrichment or outreach use.")
    else:
        warnings.append(
            "No company discovery results were returned; no company records were generated."
        )
    if company_failures:
        warnings.append(f"Company discovery had {company_failures} failed search query/queries.")
    if contact_failures:
        warnings.append(f"Contact discovery had {contact_failures} failed search query/queries.")
    if needs_contact_count:
        warnings.append(
            f"{needs_contact_count} qualified company prospect(s) still need verified contacts."
        )
    if rejected_candidate_count:
        warnings.append(
            f"{rejected_candidate_count} noisy or non-sales-ready candidate(s) were filtered out."
        )
    return tuple(warnings)


def _final_enrichment_records(
    specs: tuple[str, ...],
    *,
    approval: ApprovedProspectSelection,
    run: TieredResearchRun,
) -> tuple[FinalEnrichmentRecord, ...]:
    if specs:
        records = []
        for index, raw in enumerate(specs, start=1):
            company_id, contact_id, summary, provider = _split_spec(
                raw,
                4,
                "--mock-final-enrichment",
            )
            records.append(
                FinalEnrichmentRecord(
                    enrichment_id=f"final_{index:03d}",
                    company_id=company_id,
                    contact_id=contact_id,
                    summary=summary,
                    provider=provider or "mock",
                    evidence_ids=(f"ev_final_{index:03d}",),
                )
            )
        return tuple(records)

    company_names = {company.company_id: company.name for company in run.companies}
    contacts_by_company = _approved_contacts_by_company(run, approval)
    records = []
    for index, company_id in enumerate(approval.approved_company_ids, start=1):
        records.append(
            FinalEnrichmentRecord(
                enrichment_id=f"final_{index:03d}",
                company_id=company_id,
                contact_id=next(iter(contacts_by_company.get(company_id, ())), ""),
                summary=(
                    "Mock final enrichment approved for "
                    f"{company_names.get(company_id, company_id)}."
                ),
                evidence_ids=(f"ev_final_{index:03d}",),
                provider="mock",
            )
        )
    return tuple(records)


def _validate_approval(
    run: TieredResearchRun,
    approval: ApprovedProspectSelection,
) -> None:
    company_ids = {company.company_id for company in run.companies}
    contact_ids = {contact.contact_id for contact in run.contacts}
    contact_company = _contact_company_map(run)
    unknown_companies = set(approval.approved_company_ids) - company_ids
    unknown_contacts = set(approval.approved_contact_ids) - contact_ids
    if unknown_companies:
        raise ValueError(f"approval references unknown company_id: {sorted(unknown_companies)}")
    if unknown_contacts:
        raise ValueError(f"approval references unknown contact_id: {sorted(unknown_contacts)}")
    unapproved_contact_companies = {
        contact_id: contact_company[contact_id]
        for contact_id in approval.approved_contact_ids
        if contact_company[contact_id] not in approval.approved_company_ids
    }
    if unapproved_contact_companies:
        raise ValueError(
            "approval contact_id must belong to an approved company_id: "
            f"{unapproved_contact_companies}"
        )


def _validate_final_enrichment(
    records: tuple[FinalEnrichmentRecord, ...],
    approval: ApprovedProspectSelection,
    run: TieredResearchRun,
) -> None:
    approved_companies = set(approval.approved_company_ids)
    approved_contacts = set(approval.approved_contact_ids)
    contact_company = _contact_company_map(run)
    for record in records:
        if record.company_id not in approved_companies:
            raise ValueError(f"final enrichment outside approved company_id: {record.company_id}")
        if record.contact_id and record.contact_id not in approved_contacts:
            raise ValueError(f"final enrichment outside approved contact_id: {record.contact_id}")
        if record.contact_id and contact_company[record.contact_id] != record.company_id:
            raise ValueError(
                "final enrichment contact_id does not belong to company_id: "
                f"{record.contact_id} -> {contact_company[record.contact_id]}, "
                f"not {record.company_id}"
            )


def _contact_company_map(run: TieredResearchRun) -> dict[str, str]:
    return {contact.contact_id: contact.company_id for contact in run.contacts}


def _approved_contacts_by_company(
    run: TieredResearchRun,
    approval: ApprovedProspectSelection,
) -> dict[str, tuple[str, ...]]:
    approved_contacts = set(approval.approved_contact_ids)
    contacts_by_company: dict[str, list[str]] = {}
    for contact in run.contacts:
        if contact.contact_id in approved_contacts:
            contacts_by_company.setdefault(contact.company_id, []).append(contact.contact_id)
    return {
        company_id: tuple(contact_ids)
        for company_id, contact_ids in contacts_by_company.items()
    }


def _domain_from_url(url: str) -> str:
    return (
        url.removeprefix("https://")
        .removeprefix("http://")
        .removeprefix("www.")
        .split("/", 1)[0]
        .strip()
    )


def _run_from_payload(payload: dict[str, Any]) -> TieredResearchRun:
    return TieredResearchRun(
        run_id=str(payload["run_id"]),
        directive=parse_directive_payload(payload["directive"]),
        companies=tuple(_company_from_payload(item) for item in payload.get("companies") or ()),
        contacts=tuple(_contact_from_payload(item) for item in payload.get("contacts") or ()),
        personalizations=tuple(
            _personalization_from_payload(item)
            for item in payload.get("personalizations") or ()
        ),
        browser_captures=(),
        warnings=tuple(str(item) for item in payload.get("warnings") or ()),
    )


def _company_from_payload(payload: dict[str, Any]) -> CompanyProspect:
    return CompanyProspect(
        company_id=str(payload["company_id"]),
        name=str(payload["name"]),
        evidence_ids=tuple(payload.get("evidence_ids") or ()),
        website=str(payload.get("website") or ""),
        industry=str(payload.get("industry") or ""),
        geographic_area=str(payload.get("geographic_area") or ""),
        locations=tuple(payload.get("locations") or ()),
        fit_score=float(payload.get("fit_score") or 0.0),
        fit_rationale=str(payload.get("fit_rationale") or ""),
        disqualification_flags=tuple(payload.get("disqualification_flags") or ()),
        source_confidence=cast(SourceConfidence, str(payload.get("source_confidence") or "medium")),
    )


def _contact_from_payload(payload: dict[str, Any]) -> ContactCandidate:
    return ContactCandidate(
        contact_id=str(payload["contact_id"]),
        company_id=str(payload["company_id"]),
        name=str(payload["name"]),
        evidence_ids=tuple(payload.get("evidence_ids") or ()),
        title=str(payload.get("title") or ""),
        role_category=cast(RoleCategory, str(payload.get("role_category") or "unknown")),
        profile_urls=tuple(payload.get("profile_urls") or ()),
        email=payload.get("email"),
        phone=payload.get("phone"),
        contact_confidence=float(payload.get("contact_confidence") or 0.0),
        notes=str(payload.get("notes") or ""),
    )


def _personalization_from_payload(payload: dict[str, Any]) -> ContactPersonalization:
    return ContactPersonalization(
        contact_id=str(payload["contact_id"]),
        personalization_signals=tuple(
            PersonalizationSignal(
                contact_id=str(signal["contact_id"]),
                signal=str(signal["signal"]),
                message_angle=str(signal["message_angle"]),
                evidence_ids=tuple(signal.get("evidence_ids") or ()),
                confidence=cast(SignalConfidence, str(signal.get("confidence") or "medium")),
            )
            for signal in payload.get("personalization_signals") or ()
        ),
        suggested_opening_line=str(payload.get("suggested_opening_line") or ""),
        do_not_claim=tuple(payload.get("do_not_claim") or ()),
    )


def _approval_from_payload(payload: dict[str, Any]) -> ApprovedProspectSelection:
    return ApprovedProspectSelection(
        approved_company_ids=tuple(payload.get("approved_company_ids") or ()),
        approved_contact_ids=tuple(payload.get("approved_contact_ids") or ()),
        reviewer=str(payload.get("reviewer") or ""),
        approved_at=str(payload.get("approved_at") or datetime.now(UTC).isoformat()),
        notes=str(payload.get("notes") or ""),
    )


def _final_record_from_payload(payload: dict[str, Any]) -> FinalEnrichmentRecord:
    return FinalEnrichmentRecord(
        enrichment_id=str(payload["enrichment_id"]),
        company_id=str(payload["company_id"]),
        contact_id=str(payload.get("contact_id") or ""),
        summary=str(payload.get("summary") or ""),
        provider=str(payload.get("provider") or "mock"),
        evidence_ids=tuple(payload.get("evidence_ids") or ()),
        warnings=tuple(payload.get("warnings") or ()),
    )


def _split_spec(raw: str, parts: int, flag: str) -> tuple[str, ...]:
    values = raw.split("|", parts - 1)
    if len(values) != parts:
        raise ValueError(f"{flag} must have {parts} pipe-delimited fields")
    return tuple(value.strip() for value in values)


def _role_category(role: str) -> RoleCategory:
    lowered = role.lower()
    if any(token in lowered for token in ("owner", "founder")):
        return "owner"
    if any(token in lowered for token in ("ceo", "president", "executive")):
        return "executive"
    if "marketing" in lowered:
        return "marketing"
    if role:
        return "operator"
    return cast(RoleCategory, "unknown")


def _event(status: str, message: str, **metadata: Any) -> dict[str, Any]:
    return {
        "status": status,
        "message": message,
        "timestamp": datetime.now(UTC).isoformat(),
        **metadata,
    }


def _slug(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")


def _validate_thread_id(thread_id: str) -> str:
    candidate = str(thread_id).strip()
    if not candidate or not _THREAD_ID_PATTERN.fullmatch(candidate):
        raise ValueError(
            "thread_id must contain only letters, numbers, underscore, dash, or dot"
        )
    if candidate in {".", ".."}:
        raise ValueError("thread_id cannot be . or ..")
    return candidate


__all__ = [
    "DEFAULT_TIERED_ARTIFACT_DIR",
    "DEFAULT_TIERED_CHECKPOINT_DIR",
    "FinalEnrichmentFactory",
    "TIERED_CHECKPOINT_SCHEMA_VERSION",
    "TieredCheckpoint",
    "inspect_tiered_research",
    "load_approval_selection",
    "load_directive_json",
    "parse_directive_payload",
    "resume_tiered_research",
    "run_tiered_research",
    "run_tiered_research_with_search",
]
