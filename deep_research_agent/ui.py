"""Local Streamlit UI for running and inspecting research threads.

The helpers in this module are import-safe and testable without Streamlit. The
actual Streamlit dependency is imported only when the UI is launched or rendered.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import sys
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from async_multi_search import AsyncMultiProviderSearch, ExaProvider

try:
    from .config import dotenv_values
    from .final_enrichment import enrich_selected_prospects
    from .geography import normalize_geography
    from .graph import (
        DEFAULT_TARGET_PROSPECT_COUNT,
        MAX_MODEL_TIMEOUT_SECONDS,
        MAX_SEARCH_ITERATIONS,
        MAX_SEARCH_RESULTS_PER_ITERATION,
        MAX_SEARCH_TIMEOUT_SECONDS,
        MAX_TARGET_PROSPECT_COUNT,
        derive_prospect_run_budget,
        inspect_checkpoints,
        resume_research,
        run_research,
    )
    from .models import build_model_client
    from .tiered_models import ApprovedProspectSelection
    from .tiered_runtime import (
        DEFAULT_TIERED_ARTIFACT_DIR,
        DEFAULT_TIERED_CHECKPOINT_DIR,
        TieredCheckpoint,
        inspect_tiered_research,
        parse_directive_payload,
        resume_tiered_research,
        run_tiered_research,
        run_tiered_research_with_search,
    )
    from .tiered_search import (
        CompanySearchTarget,
        ContactSearchTarget,
        ProviderPolicy,
        TieredSearchDirective,
        build_company_discovery_lanes,
        build_company_discovery_queries,
        build_contact_discovery_queries,
        build_personalization_queries,
    )
except ImportError:
    if __package__:
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from deep_research_agent.config import dotenv_values
    from deep_research_agent.final_enrichment import enrich_selected_prospects
    from deep_research_agent.geography import normalize_geography
    from deep_research_agent.graph import (
        DEFAULT_TARGET_PROSPECT_COUNT,
        MAX_MODEL_TIMEOUT_SECONDS,
        MAX_SEARCH_ITERATIONS,
        MAX_SEARCH_RESULTS_PER_ITERATION,
        MAX_SEARCH_TIMEOUT_SECONDS,
        MAX_TARGET_PROSPECT_COUNT,
        derive_prospect_run_budget,
        inspect_checkpoints,
        resume_research,
        run_research,
    )
    from deep_research_agent.models import build_model_client
    from deep_research_agent.tiered_models import ApprovedProspectSelection
    from deep_research_agent.tiered_runtime import (
        DEFAULT_TIERED_ARTIFACT_DIR,
        DEFAULT_TIERED_CHECKPOINT_DIR,
        TieredCheckpoint,
        inspect_tiered_research,
        parse_directive_payload,
        resume_tiered_research,
        run_tiered_research,
        run_tiered_research_with_search,
    )
    from deep_research_agent.tiered_search import (
        CompanySearchTarget,
        ContactSearchTarget,
        ProviderPolicy,
        TieredSearchDirective,
        build_company_discovery_lanes,
        build_company_discovery_queries,
        build_contact_discovery_queries,
        build_personalization_queries,
    )

PROVIDER_API_KEY_FIELDS: tuple[tuple[str, str], ...] = (
    ("Tavily", "TAVILY_API_KEY"),
    ("Exa", "EXA_API_KEY"),
    ("Serper", "SERPER_API_KEY"),
    ("Firecrawl", "FIRECRAWL_API_KEY"),
    ("You.com Developer Cloud", "YDC_API_KEY"),
)

DEFAULT_CHECKPOINT_DIR = ".deep_research_agent/checkpoints"
DEFAULT_ARTIFACT_DIR = ".deep_research_agent/artifacts"
TIERED_SELECTION_SESSION_KEY = "tiered_prospect_selections"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def provider_env_overlay(values: Mapping[str, str]) -> dict[str, str]:
    """Return non-empty provider key overrides without mutating ``os.environ``."""

    allowed = {env_key for _, env_key in PROVIDER_API_KEY_FIELDS}
    return {
        key: value.strip()
        for key, value in values.items()
        if key in allowed and isinstance(value, str) and value.strip()
    }


def provider_env_from_env_file(
    path: str | os.PathLike[str] = ".env",
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return provider keys from shell environment plus ``.env`` values."""

    current = os.environ if environ is None else environ
    merged = dotenv_values(path) if os.fspath(path).strip() else {}
    merged.update(current)
    return provider_env_overlay(merged)


def build_prospect_directive(industry: str, geography: str, criteria: str) -> str:
    """Build the structured free-text directive consumed by the research graph."""

    parts = []
    if industry.strip():
        parts.append(f"industry: {industry.strip()}")
    if geography.strip():
        parts.append(f"geography: {geography.strip()}")
    if criteria.strip():
        parts.append(f"criteria: {criteria.strip()}")
    return "\n".join(parts) if parts else "criteria: business prospects"


def build_tiered_preview(
    industry: str,
    geography: str,
    criteria: str,
    *,
    target_prospect_count: int = DEFAULT_TARGET_PROSPECT_COUNT,
    preferred_contact_roles: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build an offline tiered prospect preview for UI display and tests."""

    missing_fields = []
    if not industry.strip():
        missing_fields.append("industry")
    if not geography.strip():
        missing_fields.append("geographic_area")
    if missing_fields:
        return {
            "ready": False,
            "missing_fields": missing_fields,
            "directive": {
                "industry": industry.strip(),
                "geographic_area": geography.strip(),
                "target_prospect_count": target_prospect_count,
                "research_criteria": criteria.strip(),
                "preferred_contact_roles": list(preferred_contact_roles),
            },
            "company_discovery_queries": [],
            "company_discovery_lanes": [],
            "contact_discovery_query_templates": [],
            "personalization_query_templates": [],
            "search_dependency": "injected",
            "provider_policy": ProviderPolicy().to_dict(),
            "warnings": [
                "Tiered preview is waiting for industry and geographic area.",
                "Preview only: no network search, browser capture, enrichment, "
                "or outreach is executed.",
            ],
        }

    directive = TieredSearchDirective(
        industry=industry,
        geographic_area=geography,
        target_prospect_count=target_prospect_count,
        research_criteria=criteria,
        preferred_contact_roles=preferred_contact_roles,
    )
    example_company = CompanySearchTarget("Example Company", website="https://example.com")
    example_contact = ContactSearchTarget(
        "Example Contact",
        "Example Company",
        title=(preferred_contact_roles[0] if preferred_contact_roles else "Owner"),
    )
    return {
        "ready": True,
        "missing_fields": [],
        "directive": asdict(directive),
        "company_discovery_queries": list(build_company_discovery_queries(directive)),
        "company_discovery_lanes": [
            lane.to_dict() for lane in build_company_discovery_lanes(directive)
        ],
        "contact_discovery_query_templates": list(
            build_contact_discovery_queries(directive, example_company)
        ),
        "personalization_query_templates": list(
            build_personalization_queries(directive, example_contact)
        ),
        "search_dependency": "injected",
        "provider_policy": ProviderPolicy().to_dict(),
        "warnings": [
            "Preview only: no network search, browser capture, enrichment, "
            "or outreach is executed.",
            "Early tiered discovery excludes the legacy default provider chain.",
        ],
    }


def flatten_tiered_prospect_rows(
    checkpoint: TieredCheckpoint | Mapping[str, Any] | None,
    *,
    selected_row_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Flatten a tiered checkpoint into company-level selectable prospect rows.

    Company prospects are the primary review unit. Contacts remain attached
    child evidence, with zero, one, or a small number of candidates per company.
    """

    payload = _tiered_payload(checkpoint)
    selected = set(selected_row_ids)
    audit_records = qualification_audit_rows(
        payload,
        include_statuses=("accepted", "ready", "review_ready"),
    )
    accepted_contact_ids = {
        str(record.get("contact_id") or "")
        for record in audit_records
        if str(record.get("contact_id") or "")
    }
    has_audit = bool(_qualification_audit(payload))
    companies = {
        str(company.get("company_id")): company
        for company in payload.get("companies") or ()
        if isinstance(company, Mapping)
    }
    personalizations = _personalization_summaries(payload)
    contacts_by_company: dict[str, list[Mapping[str, Any]]] = {
        company_id: [] for company_id in companies
    }
    for contact in payload.get("contacts") or ():
        if not isinstance(contact, Mapping):
            continue
        company_id = str(contact.get("company_id") or "")
        contact_id = str(contact.get("contact_id") or "")
        if not company_id or not contact_id or company_id not in companies:
            continue
        if accepted_contact_ids and contact_id not in accepted_contact_ids:
            continue
        if has_audit and _is_placeholder_contact(contact):
            continue
        contacts_by_company.setdefault(company_id, []).append(contact)

    rows: list[dict[str, Any]] = []
    for company_id, company in companies.items():
        company_contacts = contacts_by_company.get(company_id, [])
        contact_ids = [
            str(contact.get("contact_id") or "")
            for contact in company_contacts
            if str(contact.get("contact_id") or "")
        ]
        primary_contact = company_contacts[0] if company_contacts else {}
        contact_summaries = [
            ", ".join(
                part
                for part in (
                    str(contact.get("name") or ""),
                    str(contact.get("title") or ""),
                )
                if part
            )
            for contact in company_contacts
        ]
        personalization_summary = "; ".join(
            personalizations.get(contact_id, "")
            for contact_id in contact_ids
            if personalizations.get(contact_id, "")
        )
        company = companies[company_id]
        row_id = company_id
        rows.append(
            {
                "selected": row_id in selected,
                "selectable": True,
                "row_type": "company",
                "row_id": row_id,
                "company_id": company_id,
                "contact_id": str(primary_contact.get("contact_id") or ""),
                "contact_ids": contact_ids,
                "company_name": str(company.get("name") or ""),
                "website": str(company.get("website") or ""),
                "fit_score": company.get("fit_score", ""),
                "contact_count": len(company_contacts),
                "contact_names": "; ".join(contact_summaries),
                "contact_name": str(primary_contact.get("name") or ""),
                "contact_title": str(primary_contact.get("title") or ""),
                "contact_confidence": primary_contact.get("contact_confidence", ""),
                "personalization_summary": personalization_summary,
            }
        )
    return rows


def selection_from_prospect_rows(
    rows: Any,
    *,
    reviewer: str = "streamlit-ui",
    approved_at: str | None = None,
    notes: str = "",
) -> ApprovedProspectSelection:
    """Build an approval payload from checked company-level prospect rows."""

    selected_rows = [
        row
        for row in _iter_row_records(rows)
        if bool(row.get("selected")) and row.get("selectable", True) is not False
    ]
    if not selected_rows:
        raise ValueError("Select at least one prospect before final enrichment.")

    company_ids = _dedupe_preserve_order(str(row.get("company_id") or "") for row in selected_rows)
    contact_values: list[str] = []
    for row in selected_rows:
        raw_contact_ids = row.get("contact_ids")
        if isinstance(raw_contact_ids, str):
            contact_values.extend(
                item.strip() for item in raw_contact_ids.split(";") if item.strip()
            )
        elif isinstance(raw_contact_ids, Iterable):
            contact_values.extend(str(item) for item in raw_contact_ids if str(item))
        else:
            contact_id = str(row.get("contact_id") or "")
            if contact_id:
                contact_values.append(contact_id)
    contact_ids = _dedupe_preserve_order(contact_values)
    if not company_ids:
        raise ValueError("Selected prospect rows must include company IDs.")

    return ApprovedProspectSelection(
        approved_company_ids=company_ids,
        approved_contact_ids=contact_ids,
        reviewer=reviewer,
        approved_at=approved_at or datetime.now(UTC).isoformat(),
        notes=notes,
    )


def missing_exa_api_key_message(provider_keys: Mapping[str, str]) -> str | None:
    """Return the UI error message when final Exa enrichment cannot run."""

    if provider_keys.get("EXA_API_KEY"):
        return None
    return "EXA_API_KEY is required to run final Exa enrichment for selected prospects."


def prepare_exa_enrichment_selection(
    rows: Any,
    provider_keys: Mapping[str, str],
    *,
    reviewer: str = "streamlit-ui",
    notes: str = "",
) -> tuple[ApprovedProspectSelection | None, str | None]:
    """Validate selected rows and Exa configuration before any checkpoint write."""

    missing_key = missing_exa_api_key_message(provider_keys)
    if missing_key:
        return None, missing_key
    try:
        return selection_from_prospect_rows(rows, reviewer=reviewer, notes=notes), None
    except ValueError as exc:
        return None, str(exc)


def flatten_enriched_prospect_rows(
    checkpoint: TieredCheckpoint | Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Flatten persisted final enrichment records with company/contact labels."""

    payload = _tiered_payload(checkpoint)
    companies = {
        str(company.get("company_id")): company
        for company in payload.get("companies") or ()
        if isinstance(company, Mapping)
    }
    contacts = {
        str(contact.get("contact_id")): contact
        for contact in payload.get("contacts") or ()
        if isinstance(contact, Mapping)
    }
    rows: list[dict[str, Any]] = []
    for record in payload.get("final_enrichment") or ():
        if not isinstance(record, Mapping):
            continue
        company_id = str(record.get("company_id") or "")
        contact_id = str(record.get("contact_id") or "")
        company = companies.get(company_id, {})
        contact = contacts.get(contact_id, {})
        rows.append(
            {
                "enrichment_id": str(record.get("enrichment_id") or ""),
                "provider": str(record.get("provider") or ""),
                "company_id": company_id,
                "company_name": str(company.get("name") or ""),
                "website": str(company.get("website") or ""),
                "contact_id": contact_id,
                "contact_name": str(contact.get("name") or ""),
                "contact_title": str(contact.get("title") or ""),
                "summary": str(record.get("summary") or ""),
                "evidence_ids": "; ".join(str(item) for item in record.get("evidence_ids") or ()),
                "warnings": "; ".join(str(item) for item in record.get("warnings") or ()),
            }
        )
    return rows


def final_enrichment_csv_bytes(
    checkpoint: TieredCheckpoint | Mapping[str, Any] | None,
) -> bytes:
    """Return persisted final enrichment CSV bytes, or generate an equivalent CSV."""

    payload = _tiered_payload(checkpoint)
    artifact_paths = payload.get("artifact_paths") or {}
    if isinstance(artifact_paths, Mapping):
        csv_path = artifact_paths.get("final_enrichment_csv")
        if csv_path:
            path = Path(str(csv_path))
            if path.exists():
                return path.read_bytes()

    rows = flatten_enriched_prospect_rows(payload)
    output = io.StringIO()
    fieldnames = (
        "enrichment_id",
        "provider",
        "company_id",
        "company_name",
        "website",
        "contact_id",
        "contact_name",
        "contact_title",
        "summary",
        "evidence_ids",
        "warnings",
    )
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    return output.getvalue().encode("utf-8")


def tiered_review_state(
    checkpoint: TieredCheckpoint | Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return compact tiered review state for UI empty states and tests."""

    payload = _tiered_payload(checkpoint)
    prospect_count = len(flatten_tiered_prospect_rows(payload))
    enrichment_count = len(flatten_enriched_prospect_rows(payload))
    qualification = qualification_summary(payload, ready_company_count=prospect_count)
    return {
        "status": str(payload.get("status") or ""),
        "prospect_count": prospect_count,
        "ready_prospect_count": qualification["ready_company_count"],
        "ready_company_count": qualification["ready_company_count"],
        "company_prospect_count": qualification["company_prospect_count"],
        "ready_contact_count": qualification["ready_contact_count"],
        "contact_candidate_count": qualification["contact_candidate_count"],
        "qualified_company_count": qualification["qualified_company_count"],
        "companies_with_contacts_count": qualification["companies_with_contacts_count"],
        "needs_contact_count": qualification["needs_contact_count"],
        "rejected_candidate_count": qualification["rejected_candidate_count"],
        "enrichment_count": enrichment_count,
        "review_ready": str(payload.get("status") or "") == "review_required"
        and prospect_count > 0,
        "empty_review": str(payload.get("status") or "") == "review_required"
        and prospect_count == 0,
    }


def qualification_summary(
    checkpoint: TieredCheckpoint | Mapping[str, Any] | None,
    *,
    ready_company_count: int | None = None,
    ready_contact_count: int | None = None,
) -> dict[str, int]:
    """Return additive qualification counts from checkpoint/artifact payloads."""

    payload = _tiered_payload(checkpoint)
    metadata_value = payload.get("metadata")
    metadata: Mapping[str, Any] = (
        metadata_value if isinstance(metadata_value, Mapping) else {}
    )
    audit = _qualification_audit(payload)
    companies = [
        company for company in payload.get("companies") or () if isinstance(company, Mapping)
    ]
    contacts = [
        contact for contact in payload.get("contacts") or () if isinstance(contact, Mapping)
    ]
    contact_company_ids = {
        str(contact.get("company_id") or "")
        for contact in contacts
        if str(contact.get("company_id") or "")
    }
    needs_contact_from_audit = {
        str(record.get("company_id") or record.get("company_name") or record.get("name") or "")
        for record in audit
        if _audit_status(record) == "needs_contact"
    }
    accepted_contact_ids = {
        str(record.get("contact_id") or "")
        for record in audit
        if _audit_status(record) in {"accepted", "ready", "review_ready"}
        and str(record.get("contact_id") or "")
    }
    accepted_contact_company_ids = {
        str(record.get("company_id") or "")
        for record in audit
        if _audit_status(record) in {"accepted", "ready", "review_ready"}
        and str(record.get("company_id") or "")
    }
    companies_with_contacts = accepted_contact_company_ids or contact_company_ids
    defaults = {
        "ready_company_count": (
            ready_company_count
            if ready_company_count is not None
            else len(companies)
        ),
        "company_prospect_count": len(companies),
        "ready_contact_count": (
            ready_contact_count
            if ready_contact_count is not None
            else (len(accepted_contact_ids) if accepted_contact_ids else len(contacts))
        ),
        "contact_candidate_count": len(contacts),
        "qualified_company_count": len(companies),
        "companies_with_contacts_count": len(companies_with_contacts),
        "needs_contact_count": (
            len({item for item in needs_contact_from_audit if item})
            if needs_contact_from_audit
            else sum(
                1
                for company in companies
                if str(company.get("company_id") or "") not in companies_with_contacts
            )
        ),
        "rejected_candidate_count": sum(
            1 for record in audit if _audit_status(record) == "rejected"
        ),
    }
    return {
        key: _payload_count(payload, metadata, key, default=value)
        for key, value in defaults.items()
    }


def qualification_audit_rows(
    checkpoint: TieredCheckpoint | Mapping[str, Any] | None,
    *,
    include_statuses: tuple[str, ...] = ("needs_contact", "rejected"),
) -> list[dict[str, Any]]:
    """Flatten qualification audit records into non-selectable UI table rows."""

    payload = _tiered_payload(checkpoint)
    statuses = {status.lower().replace("-", "_") for status in include_statuses}
    rows: list[dict[str, Any]] = []
    for record in _qualification_audit(payload):
        status = _audit_status(record)
        if statuses and status not in statuses:
            continue
        rows.append(
            {
                "selectable": False,
                "status": status,
                "tier": str(record.get("tier") or ""),
                "company_id": str(record.get("company_id") or ""),
                "contact_id": str(record.get("contact_id") or ""),
                "company_name": str(record.get("company_name") or record.get("name") or ""),
                "contact_name": str(record.get("contact_name") or ""),
                "source_title": str(record.get("source_title") or record.get("title") or ""),
                "source_url": str(record.get("source_url") or record.get("url") or ""),
                "reasons": "; ".join(_audit_reasons(record)),
            }
        )

    if not rows and "needs_contact" in statuses:
        contact_company_ids = {
            str(contact.get("company_id") or "")
            for contact in payload.get("contacts") or ()
            if isinstance(contact, Mapping) and str(contact.get("company_id") or "")
        }
        for company in payload.get("companies") or ():
            if not isinstance(company, Mapping):
                continue
            company_id = str(company.get("company_id") or "")
            if company_id and company_id not in contact_company_ids:
                rows.append(
                    {
                        "selectable": False,
                        "status": "needs_contact",
                        "tier": "company",
                        "company_id": company_id,
                        "contact_id": "",
                        "company_name": str(company.get("name") or ""),
                        "contact_name": "",
                        "source_title": "Qualified company has no verified contact",
                        "source_url": str(company.get("website") or ""),
                        "reasons": "no_verified_contact",
                    }
                )
    return rows


def _tiered_payload(checkpoint: TieredCheckpoint | Mapping[str, Any] | None) -> dict[str, Any]:
    if checkpoint is None:
        return {}
    if isinstance(checkpoint, Mapping):
        return dict(checkpoint)
    return checkpoint.to_dict()


def _qualification_audit(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw: Any = payload.get("qualification_audit")
    metadata = payload.get("metadata")
    if raw is None and isinstance(metadata, Mapping):
        raw = metadata.get("qualification_audit")
        qualification = metadata.get("qualification")
        if raw is None and isinstance(qualification, Mapping):
            raw = qualification.get("audit") or qualification.get("records")
    if isinstance(raw, Mapping):
        raw = raw.get("records") or raw.get("audit") or raw.get("items")
    if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes)):
        return []
    rows: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, Mapping):
            rows.append(dict(item))
    return rows


def _audit_status(record: Mapping[str, Any]) -> str:
    return str(record.get("status") or "").strip().lower().replace("-", "_")


def _audit_reasons(record: Mapping[str, Any]) -> list[str]:
    raw = record.get("reasons") or record.get("reason_flags") or record.get("reason") or ()
    if isinstance(raw, str):
        return [raw] if raw else []
    if isinstance(raw, Iterable):
        return [str(item) for item in raw if str(item)]
    return [str(raw)] if raw else []


def _payload_count(
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    key: str,
    *,
    default: int,
) -> int:
    value = payload.get(key)
    if value in (None, ""):
        value = metadata.get(key)
    if value in (None, ""):
        counts = payload.get("qualification_counts") or metadata.get("qualification_counts")
        if isinstance(counts, Mapping):
            value = counts.get(key)
    if value in (None, ""):
        return default
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _is_placeholder_contact(contact: Mapping[str, Any]) -> bool:
    name = str(contact.get("name") or "").strip().casefold()
    return name.startswith("review contact at ")


def _personalization_summaries(payload: Mapping[str, Any]) -> dict[str, str]:
    summaries: dict[str, str] = {}
    for item in payload.get("personalizations") or ():
        if not isinstance(item, Mapping):
            continue
        contact_id = str(item.get("contact_id") or "")
        if not contact_id:
            continue
        opening = str(item.get("suggested_opening_line") or "").strip()
        if opening:
            summaries[contact_id] = opening
            continue
        signals = []
        for signal in item.get("personalization_signals") or ():
            if not isinstance(signal, Mapping):
                continue
            signals.append(
                str(signal.get("message_angle") or signal.get("signal") or "").strip()
            )
        summaries[contact_id] = "; ".join(signal for signal in signals if signal)
    return summaries


def _prospect_row_id(company_id: str, contact_id: str) -> str:
    return f"{company_id}::{contact_id}"


def _iter_row_records(rows: Any) -> list[dict[str, Any]]:
    if rows is None:
        return []
    if hasattr(rows, "to_dict"):
        records = rows.to_dict("records")
        return [dict(row) for row in records]
    if isinstance(rows, Mapping):
        return [dict(rows)]
    return [dict(row) for row in rows]


def _dedupe_preserve_order(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return tuple(deduped)


def preview_geography_scope(geography: str) -> dict[str, Any]:
    """Return display-safe geography normalization details for the UI."""

    return normalize_geography(geography).to_dict()


def env_file_overlay(
    path: str | os.PathLike[str] = ".env",
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return non-empty ``.env`` values that are not already exported."""

    current = os.environ if environ is None else environ
    values = dotenv_values(path) if os.fspath(path).strip() else {}
    return {key: value for key, value in values.items() if value and key not in current}


def model_preflight_from_env_file(
    path: str | os.PathLike[str] = ".env",
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return redacted model availability using the same env overlay as runs."""

    with temporary_env(env_file_overlay(path, environ=environ)):
        return build_model_client().preflight().to_dict()


@contextmanager
def temporary_env(overrides: Mapping[str, str]) -> Iterator[None]:
    """Apply environment values for one run and restore the previous process state."""

    original: dict[str, str | None] = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key, previous in original.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def result_preview(state: Mapping[str, Any], markdown_limit: int = 4000) -> dict[str, Any]:
    """Build the compact, display-safe result preview used by UI and tests."""

    artifact_paths = dict(state.get("artifact_paths", {}))
    markdown_preview = ""
    markdown_path = artifact_paths.get("markdown")
    if markdown_path:
        path = Path(str(markdown_path))
        if path.exists():
            markdown_preview = path.read_text(encoding="utf-8")[:markdown_limit]

    return {
        "thread_id": state.get("thread_id", ""),
        "status": state.get("status", ""),
        "warnings": list(state.get("warnings", [])),
        "event_count": len(state.get("events", [])),
        "events": list(state.get("events", []))[-25:],
        "evidence": list(state.get("evidence", []))[:10],
        "prospect_targets": list(state.get("prospect_targets", []))[:10],
        "artifact_paths": artifact_paths,
        "markdown_preview": markdown_preview,
        "raw_json": _jsonable(dict(state)),
    }


def _mock_result_lines(value: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in value.splitlines() if line.strip())


def tiered_early_discovery_searcher(timeout: int) -> AsyncMultiProviderSearch:
    """Build the tiered early-discovery searcher without Exa final enrichment."""

    providers = [
        provider for provider in AsyncMultiProviderSearch().providers if provider.name != "exa"
    ]
    return AsyncMultiProviderSearch(providers=providers, timeout=timeout)


def launch_streamlit() -> int:
    """Launch this module with Streamlit's CLI."""

    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError:
        print(
            "Streamlit is not installed. Install it with: pip install -e '.[ui]'",
            file=sys.stderr,
        )
        return 1

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    return int(streamlit_cli.main() or 0)


def render_app() -> None:
    """Render the Streamlit app."""

    import streamlit as st

    st.set_page_config(page_title="Deep Research Agent", layout="wide")
    st.title("Deep Research Agent")
    st.caption("Local execution UI. Provider API keys are read from .env or shell env.")

    if "events" not in st.session_state:
        st.session_state.events = []
    if "last_state" not in st.session_state:
        st.session_state.last_state = None
    if "last_tiered_state" not in st.session_state:
        st.session_state.last_tiered_state = None
    if TIERED_SELECTION_SESSION_KEY not in st.session_state:
        st.session_state[TIERED_SELECTION_SESSION_KEY] = {}

    with st.sidebar:
        st.header("Tiered research")
        industry = st.text_input("Industry / niche", "")
        geography = st.text_input("Geographic area", "")
        geography_preview = preview_geography_scope(geography)
        geography_terms = geography_preview.get("search_terms", [])
        if geography_terms:
            st.caption(f"Search geography: {', '.join(geography_terms[:6])}")
        for warning in geography_preview.get("warnings", []):
            st.warning(str(warning))
        query = st.text_area(
            "Research criteria",
            "Likely need lead reactivation, customer winback, or dormant-database follow-up.",
        )
        thread_id = st.text_input("Thread ID", "")
        target_prospect_count = st.number_input(
            "Target prospects",
            min_value=1,
            max_value=MAX_TARGET_PROSPECT_COUNT,
            value=DEFAULT_TARGET_PROSPECT_COUNT,
        )
        derived_budget = derive_prospect_run_budget(int(target_prospect_count))
        tiered_preview = build_tiered_preview(
            industry,
            geography,
            query,
            target_prospect_count=int(target_prospect_count),
        )
        with st.expander("Tiered prospect preview", expanded=False):
            st.caption("Offline preview: no search, browser capture, enrichment, or outreach runs.")
            st.json(tiered_preview)
        tiered_checkpoint_dir = st.text_input(
            "Tiered checkpoint dir",
            DEFAULT_TIERED_CHECKPOINT_DIR,
        )
        tiered_artifact_dir = st.text_input("Tiered artifact dir", DEFAULT_TIERED_ARTIFACT_DIR)
        with st.expander("Tiered mock company inputs", expanded=False):
            tiered_mock_results_text = st.text_area(
                "Company results",
                "",
                placeholder="Acme Dental|https://acme.example|Strong fit|mock",
            )

        active_budget = derived_budget
        legacy_checkpoint_dir = DEFAULT_CHECKPOINT_DIR
        legacy_artifact_dir = DEFAULT_ARTIFACT_DIR
        require_review = True
        approve_review = False
        enable_llm_judgment = True
        with st.expander("Advanced / Legacy G003", expanded=False):
            st.caption(
                "Legacy G003 remains available for compatibility; the main workflow is tiered."
            )
            st.caption(
                "Budget: "
                f"{derived_budget.max_iterations} searches, "
                f"{derived_budget.max_results} results/search, "
                f"{derived_budget.search_timeout_seconds}s search timeout, "
                f"{derived_budget.model_timeout_seconds}s model timeout"
            )
            for warning in derived_budget.warnings:
                st.warning(warning)
            override_budget = st.checkbox("Override derived budget", value=False)
            if override_budget:
                max_iterations = st.number_input(
                    "Max iterations",
                    min_value=1,
                    max_value=MAX_SEARCH_ITERATIONS,
                    value=derived_budget.max_iterations,
                )
                search_max_results = st.number_input(
                    "Search max results",
                    min_value=1,
                    max_value=MAX_SEARCH_RESULTS_PER_ITERATION,
                    value=derived_budget.max_results,
                )
                timeout_seconds = st.number_input(
                    "Search timeout seconds",
                    min_value=1,
                    max_value=MAX_SEARCH_TIMEOUT_SECONDS,
                    value=derived_budget.search_timeout_seconds,
                )
                model_timeout_seconds = st.number_input(
                    "Model timeout seconds",
                    min_value=1,
                    max_value=MAX_MODEL_TIMEOUT_SECONDS,
                    value=derived_budget.model_timeout_seconds,
                )
                active_budget = derive_prospect_run_budget(
                    int(target_prospect_count),
                    max_iterations=int(max_iterations),
                    max_results=int(search_max_results),
                    search_timeout_seconds=int(timeout_seconds),
                    model_timeout_seconds=int(model_timeout_seconds),
                )
            else:
                active_budget = derived_budget
            legacy_checkpoint_dir = st.text_input(
                "Legacy G003 checkpoint dir",
                DEFAULT_CHECKPOINT_DIR,
            )
            legacy_artifact_dir = st.text_input("Legacy G003 artifact dir", DEFAULT_ARTIFACT_DIR)
            require_review = st.checkbox("Require review", value=True)
            approve_review = st.checkbox("Approve review", value=False)
            enable_llm_judgment = st.checkbox("LLM prospect judgment", value=True)
            run_clicked = st.button("Run G003", width="stretch")
            resume_clicked = st.button("Resume G003", width="stretch")
            inspect_clicked = st.button("Inspect G003", width="stretch")

        st.header("Environment")
        env_file = st.text_input("Env file", ".env")
        runtime_env = {
            **env_file_overlay(env_file),
            "DEEP_RESEARCH_MODEL_TIMEOUT_SECONDS": str(active_budget.model_timeout_seconds),
        }
        provider_keys = provider_env_from_env_file(env_file)
        configured = [
            label for label, env_key in PROVIDER_API_KEY_FIELDS if env_key in provider_keys
        ]
        missing = [
            label for label, env_key in PROVIDER_API_KEY_FIELDS if env_key not in provider_keys
        ]
        st.caption(f"Configured providers: {', '.join(configured) if configured else 'none'}")
        st.caption(f"Missing provider keys: {', '.join(missing) if missing else 'none'}")
        model_preflight = model_preflight_from_env_file(env_file)
        model_label = (
            model_preflight.get("selected_provider")
            if model_preflight.get("live_model_available")
            else "none"
        )
        st.caption(f"Live model provider: {model_label}")

        tiered_run_clicked = st.button("Run tiered research", type="primary", width="stretch")
        tiered_resume_clicked = st.button("Resume tiered research", width="stretch")
        tiered_inspect_clicked = st.button("Inspect tiered research", width="stretch")

    def progress(event: dict[str, Any]) -> None:
        st.session_state.events.append(event)

    try:
        if tiered_run_clicked:
            directive = parse_directive_payload(
                {
                    "industry": industry,
                    "geography": geography,
                    "target_prospect_count": int(target_prospect_count),
                    "research_criteria": query,
                }
            )
            mock_results = _mock_result_lines(tiered_mock_results_text)
            if mock_results:
                checkpoint = run_tiered_research(
                    directive,
                    thread_id=thread_id or None,
                    checkpoint_dir=tiered_checkpoint_dir,
                    artifact_dir=tiered_artifact_dir,
                    mock_results=mock_results,
                )
            else:
                searcher = tiered_early_discovery_searcher(
                    timeout=int(active_budget.search_timeout_seconds)
                )
                with temporary_env(runtime_env):
                    checkpoint = asyncio.run(
                        run_tiered_research_with_search(
                            directive,
                            search=searcher.search,
                            thread_id=thread_id or None,
                            checkpoint_dir=tiered_checkpoint_dir,
                            artifact_dir=tiered_artifact_dir,
                            max_results=active_budget.max_results,
                        )
                    )
            st.session_state.events = list(checkpoint.events)
            st.session_state.last_tiered_state = checkpoint.to_dict()
            st.session_state.last_state = None
            prospect_count = len(flatten_tiered_prospect_rows(checkpoint))
            st.success(
                f"Tiered run {checkpoint.status}: {checkpoint.thread_id}; "
                f"{prospect_count} prospect row(s)"
            )

        if tiered_resume_clicked:
            if not thread_id:
                st.error("Thread ID is required to resume tiered research.")
            else:
                checkpoint = resume_tiered_research(
                    thread_id,
                    checkpoint_dir=tiered_checkpoint_dir,
                    artifact_dir=tiered_artifact_dir,
                )
                st.session_state.events = list(checkpoint.events)
                st.session_state.last_tiered_state = checkpoint.to_dict()
                st.session_state.last_state = None
                st.success(f"Resumed tiered {checkpoint.status}: {thread_id}")

        if tiered_inspect_clicked:
            if not thread_id:
                st.error("Thread ID is required to inspect tiered research.")
            else:
                checkpoint = inspect_tiered_research(
                    thread_id,
                    checkpoint_dir=tiered_checkpoint_dir,
                )
                st.session_state.events = list(checkpoint.events)
                st.session_state.last_tiered_state = checkpoint.to_dict()
                st.session_state.last_state = None
                st.info(f"Loaded tiered checkpoint: {thread_id}")

        if run_clicked:
            st.session_state.events = []
            searcher = AsyncMultiProviderSearch(timeout=int(active_budget.search_timeout_seconds))
            research_query = build_prospect_directive(industry, geography, query)
            with temporary_env(runtime_env):
                state = asyncio.run(
                    run_research(
                        research_query,
                        thread_id=thread_id or None,
                        checkpoint_dir=legacy_checkpoint_dir,
                        search=searcher.search,
                        require_review=require_review,
                        max_iterations=active_budget.max_iterations,
                        max_results=active_budget.max_results,
                        target_prospect_count=active_budget.target_prospect_count,
                        search_timeout_seconds=active_budget.search_timeout_seconds,
                        model_timeout_seconds=active_budget.model_timeout_seconds,
                        enable_llm_judgment=bool(enable_llm_judgment),
                        progress_callback=progress,
                        review_approved=approve_review,
                        artifact_dir=legacy_artifact_dir,
                    )
                )
            st.session_state.last_state = state
            st.session_state.last_tiered_state = None
            st.success(f"Run {state.get('status', 'finished')}: {state.get('thread_id')}")

        if resume_clicked:
            if not thread_id:
                st.error("Thread ID is required to resume.")
            else:
                st.session_state.events = []
                searcher = AsyncMultiProviderSearch(
                    timeout=int(active_budget.search_timeout_seconds)
                )
                with temporary_env(runtime_env):
                    state = asyncio.run(
                        resume_research(
                            thread_id,
                            checkpoint_dir=legacy_checkpoint_dir,
                            approve_review=approve_review,
                            search=searcher.search,
                            max_results=active_budget.max_results,
                            progress_callback=progress,
                            artifact_dir=legacy_artifact_dir,
                        )
                    )
                st.session_state.last_state = state
                st.session_state.last_tiered_state = None
                st.success(f"Resumed {state.get('status', 'finished')}: {thread_id}")

        if inspect_clicked:
            if not thread_id:
                st.error("Thread ID is required to inspect.")
            else:
                state = inspect_checkpoints(thread_id, checkpoint_dir=legacy_checkpoint_dir)
                st.session_state.last_state = state
                st.session_state.last_tiered_state = None
                st.info(f"Loaded checkpoint: {thread_id}")
    except Exception as exc:  # pragma: no cover - exercised manually through Streamlit
        st.exception(exc)

    tiered_state = st.session_state.last_tiered_state
    state = tiered_state or st.session_state.last_state
    preview = result_preview(state or {})

    status_col, warning_col, artifact_col = st.columns(3)
    status_col.metric("Status", str(preview["status"] or "not run"))
    warning_col.metric("Warnings", len(preview["warnings"]))
    artifact_col.metric("Artifacts", len(preview["artifact_paths"]))
    if tiered_state:
        review_state = tiered_review_state(tiered_state)
        if review_state["review_ready"]:
            st.info(
                f"{review_state['ready_prospect_count']} company prospect row(s); "
                f"{review_state['contact_candidate_count']} contact candidate(s); "
                f"{review_state['needs_contact_count']} qualified compan"
                f"{'y' if review_state['needs_contact_count'] == 1 else 'ies'} "
                "need contact discovery; "
                f"{review_state['rejected_candidate_count']} rejected/noisy candidate(s)."
            )
        elif review_state["empty_review"]:
            st.warning(
                "No company prospect rows were generated for this tiered run. "
                f"{review_state['needs_contact_count']} qualified compan"
                f"{'y' if review_state['needs_contact_count'] == 1 else 'ies'} "
                "need contact discovery; "
                f"{review_state['rejected_candidate_count']} rejected/noisy candidate(s)."
            )

    tabs = st.tabs(
        [
            "Prospects",
            "Enriched Prospects",
            "Timeline",
            "Artifacts",
            "Markdown",
            "Raw JSON",
        ]
    )
    with tabs[0]:
        st.subheader("Prospects")
        if tiered_state:
            active_thread_id = str(tiered_state.get("thread_id") or "")
            selections_by_thread = st.session_state[TIERED_SELECTION_SESSION_KEY]
            selected_row_ids = selections_by_thread.get(active_thread_id, set())
            prospect_rows = flatten_tiered_prospect_rows(
                tiered_state,
                selected_row_ids=selected_row_ids,
            )
            if prospect_rows:
                edited_rows = st.data_editor(
                    prospect_rows,
                    key=f"tiered-prospects-{active_thread_id}",
                    hide_index=True,
                    width="stretch",
                    column_order=(
                        "selected",
                        "company_name",
                        "website",
                        "fit_score",
                        "contact_count",
                        "contact_names",
                        "personalization_summary",
                    ),
                    disabled=[
                        "row_id",
                        "company_id",
                        "contact_id",
                        "contact_ids",
                        "company_name",
                        "website",
                        "fit_score",
                        "contact_count",
                        "contact_names",
                        "contact_name",
                        "contact_title",
                        "contact_confidence",
                        "personalization_summary",
                    ],
                    column_config={
                        "selected": st.column_config.CheckboxColumn("Select"),
                        "website": st.column_config.LinkColumn("Website"),
                    },
                )
                current_selection = {
                    str(row.get("row_id"))
                    for row in _iter_row_records(edited_rows)
                    if bool(row.get("selected"))
                }
                selections_by_thread[active_thread_id] = current_selection
                if st.button("Run Exa enrichment for selected prospects", width="stretch"):
                    approval, error = prepare_exa_enrichment_selection(
                        edited_rows,
                        provider_keys,
                    )
                    if error:
                        st.error(error)
                    elif approval is not None:
                        checkpoint = inspect_tiered_research(
                            active_thread_id,
                            checkpoint_dir=tiered_checkpoint_dir,
                        )
                        exa_searcher = AsyncMultiProviderSearch(
                            providers=[ExaProvider()],
                            treat_empty_as_failure=False,
                            timeout=int(active_budget.search_timeout_seconds),
                        )
                        with temporary_env(runtime_env):
                            records = asyncio.run(
                                enrich_selected_prospects(
                                    checkpoint.run,
                                    approval,
                                    exa_searcher.search,
                                    max_results=3,
                                )
                            )
                            enriched = resume_tiered_research(
                                active_thread_id,
                                checkpoint_dir=tiered_checkpoint_dir,
                                artifact_dir=tiered_artifact_dir,
                                approval_selection=approval,
                                enable_final_enrichment=True,
                                final_enrichment_records=records,
                            )
                        st.session_state.events = list(enriched.events)
                        st.session_state.last_tiered_state = enriched.to_dict()
                        st.session_state.last_state = None
                        st.success(
                            "Exa enrichment complete: "
                            f"{len(enriched.final_enrichment)} record(s)."
                        )
                        st.rerun()
            else:
                st.info("No company prospect rows are available in this tiered checkpoint.")
            audit_rows = qualification_audit_rows(tiered_state)
            if audit_rows:
                with st.expander(
                    "Qualification audit: needs contact and rejected/noisy candidates"
                ):
                    st.dataframe(audit_rows, width="stretch")
        else:
            st.dataframe(preview["prospect_targets"], width="stretch")
    with tabs[1]:
        st.subheader("Enriched prospects")
        enriched_rows = flatten_enriched_prospect_rows(tiered_state)
        if enriched_rows:
            st.dataframe(enriched_rows, width="stretch")
            st.download_button(
                "Download final_enrichment.csv",
                data=final_enrichment_csv_bytes(tiered_state),
                file_name="final_enrichment.csv",
                mime="text/csv",
                width="stretch",
            )
        else:
            st.info("No final enrichment records have been persisted for this tiered run.")
    with tabs[2]:
        st.subheader("Progress events")
        st.json(st.session_state.events or preview["events"])
    with tabs[3]:
        st.subheader("Artifact paths")
        st.json(preview["artifact_paths"])
    with tabs[4]:
        st.subheader("Markdown preview")
        st.markdown(preview["markdown_preview"] or "_No markdown artifact yet._")
    with tabs[5]:
        st.subheader("Raw state JSON")
        st.code(json.dumps(preview["raw_json"], indent=2, sort_keys=True), language="json")


if __name__ == "__main__":
    render_app()
