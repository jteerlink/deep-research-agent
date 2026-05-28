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

    selected_contacts = _approved_contact_pairs(run, approval)
    records: list[FinalEnrichmentRecord] = []
    for index, contact in enumerate(selected_contacts, start=1):
        company = _companies_by_id(run)[contact.company_id]
        query = _final_enrichment_query(run, contact)
        raw_results = await search(query, max_results)
        results = tuple(_coerce_search_result(result) for result in raw_results)
        non_exa_providers = sorted(
            {result["provider"] for result in results if result["provider"] != "exa"}
        )
        if non_exa_providers:
            raise ValueError(
                "final enrichment search must use Exa results only; got "
                f"{non_exa_providers}"
            )

        evidence_ids = tuple(
            f"ev_final_exa_{index:03d}_{rank:03d}"
            for rank, _result in enumerate(results, start=1)
        )
        warnings: tuple[str, ...] = ()
        if not evidence_ids:
            evidence_ids = (f"ev_final_exa_{index:03d}_no_results",)
            warnings = ("Exa returned no results for the approved prospect pair.",)

        records.append(
            FinalEnrichmentRecord(
                enrichment_id=(
                    f"final_exa_{index:03d}_{_slug(contact.company_id)}_"
                    f"{_slug(contact.contact_id)}"
                ),
                company_id=contact.company_id,
                contact_id=contact.contact_id,
                summary=_final_enrichment_summary(
                    company_name=company.name,
                    contact=contact,
                    results=results,
                ),
                evidence_ids=evidence_ids,
                provider="exa",
                warnings=warnings,
            )
        )
    return tuple(records)


def _approved_contact_pairs(
    run: TieredResearchRun,
    approval: ApprovedProspectSelection,
) -> tuple[ContactCandidate, ...]:
    approved_companies = set(approval.approved_company_ids)
    approved_contacts = set(approval.approved_contact_ids)
    contacts = tuple(
        contact
        for contact in run.contacts
        if contact.contact_id in approved_contacts and contact.company_id in approved_companies
    )
    missing_contacts = approved_contacts - {contact.contact_id for contact in contacts}
    if missing_contacts:
        raise ValueError(f"approved contact_id not found in approved company: {missing_contacts}")
    return contacts


def _companies_by_id(run: TieredResearchRun) -> dict[str, CompanyProspect]:
    return {company.company_id: company for company in run.companies}


def _final_enrichment_query(run: TieredResearchRun, contact: ContactCandidate) -> str:
    company = _companies_by_id(run)[contact.company_id]
    parts = [
        contact.name,
        company.name,
        contact.title,
        run.directive.industry,
        run.directive.geographic_area,
        "recent news leadership business context",
    ]
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
    contact: ContactCandidate,
    results: Sequence[dict[str, Any]],
) -> str:
    if not results:
        return f"Exa found no final enrichment results for {contact.name} at {company_name}."

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
    return f"Exa final enrichment for {contact.name} at {company_name}: " + " | ".join(excerpts)


def _slug(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_") or "id"


__all__ = ["enrich_selected_prospects"]
