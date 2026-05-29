"""Approval-gated Exa final-enrichment helpers for tiered prospect runs."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from .tiered_models import (
    ApprovedProspectSelection,
    CompanyProspect,
    ContactCandidate,
    FinalEnrichmentRecord,
    TieredResearchRun,
)
from .tiered_search import SearchCallable


async def enrich_selected_prospects(
    run: TieredResearchRun,
    approval: ApprovedProspectSelection,
    search: SearchCallable,
    *,
    max_results: int = 3,
) -> tuple[FinalEnrichmentRecord, ...]:
    """Run Exa-only final enrichment for approved company/contact pairs."""

    if max_results < 1:
        raise ValueError("max_results must be >= 1")

    selected_targets = _approved_enrichment_targets(run, approval)
    records: list[FinalEnrichmentRecord] = []
    for index, (company, contact) in enumerate(selected_targets, start=1):
        query = _final_enrichment_query(run, company, contact)
        raw_results = await search(query, max_results)
        results = tuple(_coerce_search_result(result) for result in raw_results)
        non_exa_providers = sorted(
            {result["provider"] for result in results if result["provider"] != "exa"}
        )
        if non_exa_providers:
            raise ValueError(
                f"final enrichment search must use Exa results only; got {non_exa_providers}"
            )

        evidence_ids = tuple(
            f"ev_final_exa_{index:03d}_{rank:03d}" for rank, _result in enumerate(results, start=1)
        )
        warnings: tuple[str, ...] = ()
        if not evidence_ids:
            evidence_ids = (f"ev_final_exa_{index:03d}_no_results",)
            warnings = ("Exa returned no results for the approved prospect pair.",)

        records.append(
            FinalEnrichmentRecord(
                enrichment_id=(
                    f"final_exa_{index:03d}_{_slug(company.company_id)}_"
                    f"{_slug(contact.contact_id) if contact else 'company'}"
                ),
                company_id=company.company_id,
                contact_id=contact.contact_id if contact else "",
                summary=_final_enrichment_summary(
                    company_name=company.name,
                    contact=contact,
                    results=results,
                ),
                evidence_ids=evidence_ids,
                provider="exa",
                warnings=warnings,
                contact_snapshot=_contact_snapshot(contact),
            )
        )
    return tuple(records)


def _approved_enrichment_targets(
    run: TieredResearchRun,
    approval: ApprovedProspectSelection,
) -> tuple[tuple[CompanyProspect, ContactCandidate | None], ...]:
    companies_by_id = _companies_by_id(run)
    approved_companies = set(approval.approved_company_ids)
    approved_contacts = set(approval.approved_contact_ids)
    contacts_by_id = {contact.contact_id: contact for contact in run.contacts}
    contacts = [
        contact
        for contact in run.contacts
        if contact.contact_id in approved_contacts and contact.company_id in approved_companies
    ]
    missing_contacts = approved_contacts - set(contacts_by_id)
    if missing_contacts:
        raise ValueError(f"approved contact_id not found in approved company: {missing_contacts}")
    targets: list[tuple[CompanyProspect, ContactCandidate | None]] = []
    contacts_by_company: dict[str, list[ContactCandidate]] = {}
    for contact in contacts:
        contacts_by_company.setdefault(contact.company_id, []).append(contact)
    for company_id in approval.approved_company_ids:
        company = companies_by_id[company_id]
        company_contacts = contacts_by_company.get(company_id)
        if company_contacts:
            targets.extend((company, contact) for contact in company_contacts)
        else:
            targets.append((company, None))
    return tuple(targets)


def _companies_by_id(run: TieredResearchRun) -> dict[str, CompanyProspect]:
    return {company.company_id: company for company in run.companies}


def _final_enrichment_query(
    run: TieredResearchRun,
    company: CompanyProspect,
    contact: ContactCandidate | None,
) -> str:
    parts = [
        company.name,
        run.directive.industry,
        run.directive.geographic_area,
        "recent news business context",
    ]
    if contact is not None:
        if contact.contact_kind == "person":
            parts[:0] = [contact.name, contact.title]
            parts.append("leadership context")
        else:
            parts.extend(
                [
                    contact.label or contact.name,
                    contact.email or "",
                    contact.phone or "",
                    contact.contact_url or contact.url or "",
                    contact.source_url or "",
                    "official company contact information",
                ]
            )
    if company.website:
        parts.append(company.website)
    return " ".join(part.strip() for part in parts if part.strip())


def _coerce_search_result(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        title = str(raw.get("title") or "")
        url = str(raw.get("url") or raw.get("href") or "")
        content = str(raw.get("content") or raw.get("text") or raw.get("snippet") or "")
        provider = str(raw.get("provider") or "").lower()
        score = raw.get("score")
    else:
        title = str(getattr(raw, "title", ""))
        url = str(getattr(raw, "url", ""))
        content = str(getattr(raw, "content", ""))
        provider = str(getattr(raw, "provider", "")).lower()
        score = getattr(raw, "score", None)
    return {
        "title": title,
        "url": url,
        "content": content,
        "provider": provider,
        "score": score,
    }


def _final_enrichment_summary(
    *,
    company_name: str,
    contact: ContactCandidate | None,
    results: Sequence[dict[str, Any]],
) -> str:
    subject = _enrichment_subject(company_name, contact)
    if not results:
        return f"Exa found no final enrichment results for {subject}."

    excerpts = []
    for result in results[:3]:
        label = result["title"] or result["url"] or "Untitled Exa result"
        content = " ".join(str(result["content"]).split())
        if len(content) > 180:
            content = content[:177].rstrip() + "..."
        excerpt = label
        if result["url"]:
            excerpt += f" ({result['url']})"
        if content:
            excerpt += f": {content}"
        excerpts.append(excerpt)
    return f"Exa final enrichment for {subject}: " + " | ".join(excerpts)


def _contact_snapshot(contact: ContactCandidate | None) -> dict[str, str] | None:
    if contact is None:
        return None
    return {
        "contact_id": contact.contact_id,
        "company_id": contact.company_id,
        "contact_kind": contact.contact_kind,
        "label": contact.label,
        "name": contact.name,
        "title": contact.title,
        "email": contact.email or "",
        "phone": contact.phone or "",
        "url": contact.url,
        "contact_url": contact.contact_url,
        "source_url": contact.source_url,
    }


def _enrichment_subject(company_name: str, contact: ContactCandidate | None) -> str:
    if contact is None:
        return company_name
    if contact.contact_kind == "person":
        return f"{contact.name} at {company_name}"
    label = contact.label or contact.name or "company contact info"
    return f"{company_name} {label}"


def _slug(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_") or "id"


__all__ = ["enrich_selected_prospects"]
