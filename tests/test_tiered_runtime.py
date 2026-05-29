from __future__ import annotations

import asyncio
import json

import pytest

from async_multi_search import SearchResult
from deep_research_agent.final_enrichment import enrich_selected_prospects
from deep_research_agent.tiered_models import (
    ApprovedProspectSelection,
    CompanyProspect,
    ContactCandidate,
    TieredResearchRun,
)
from deep_research_agent.tiered_runtime import (
    inspect_tiered_research,
    parse_directive_payload,
    resume_tiered_research,
    run_tiered_research,
    run_tiered_research_with_search,
)


def _run_review_checkpoint(tmp_path):
    directive = parse_directive_payload(
        {
            "industry": "dental",
            "geography": "North Texas",
            "target_prospect_count": 1,
            "research_criteria": "multi-location Invisalign",
            "preferred_contact_roles": ["owner"],
        }
    )
    return run_tiered_research(
        directive,
        thread_id="tiered-runtime",
        checkpoint_dir=tmp_path / "checkpoints",
        artifact_dir=tmp_path / "artifacts",
        mock_results=("Acme Dental|https://acme.example|Strong fit|mock",),
    )


def test_tiered_resume_without_approval_preserves_review_state(tmp_path) -> None:
    checkpoint = _run_review_checkpoint(tmp_path)

    resumed = resume_tiered_research(
        checkpoint.thread_id,
        checkpoint_dir=tmp_path / "checkpoints",
        artifact_dir=tmp_path / "artifacts",
    )

    assert resumed.status == "review_required"
    assert resumed.approval is None
    assert resumed.final_enrichment == ()



def test_inspect_rejects_legacy_contact_payload_without_contact_kind(tmp_path) -> None:
    checkpoint = _run_review_checkpoint(tmp_path)
    checkpoint_path = tmp_path / "checkpoints" / f"{checkpoint.thread_id}.json"
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    payload["contacts"] = [
        {
            "contact_id": "contact_legacy_001",
            "company_id": checkpoint.run.companies[0].company_id,
            "name": "Legacy Person",
            "title": "Owner",
            "evidence_ids": ["ev_legacy_contact"],
        }
    ]
    checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="legacy contact payload missing contact_kind"):
        inspect_tiered_research(
            checkpoint.thread_id,
            checkpoint_dir=tmp_path / "checkpoints",
        )


def test_tiered_run_without_mock_results_does_not_invent_company(tmp_path) -> None:
    directive = parse_directive_payload(
        {
            "industry": "dental",
            "geography": "North Texas",
            "target_prospect_count": 1,
            "research_criteria": "multi-location Invisalign",
        }
    )

    checkpoint = run_tiered_research(
        directive,
        thread_id="tiered-empty",
        checkpoint_dir=tmp_path / "checkpoints",
        artifact_dir=tmp_path / "artifacts",
    )

    assert checkpoint.status == "review_required"
    assert checkpoint.run.companies == ()
    assert "no company records were generated" in checkpoint.warnings[0]


def test_tiered_run_with_search_generates_company_contact_rows(tmp_path) -> None:
    directive = parse_directive_payload(
        {
            "industry": "dental",
            "geography": "North Texas",
            "target_prospect_count": 2,
            "research_criteria": "multi-location Invisalign",
        }
    )
    queries: list[str] = []
    max_result_values: list[int] = []

    async def search(query: str, max_results: int):
        queries.append(query)
        max_result_values.append(max_results)
        if "site:acme.example contact" in query:
            assert max_results == 2
            return [
                SearchResult(
                    "Contact Acme Dental",
                    "https://acme.example/contact",
                    '<a href="mailto:hello@acme.example">Email</a>',
                    provider="duckduckgo",
                )
            ]
        if "site:beta.example contact" in query:
            assert max_results == 2
            return [
                SearchResult(
                    "Call Beta Dental",
                    "https://beta.example/contact",
                    '<a href="tel:817-555-0100">Call Beta</a>',
                    provider="duckduckgo",
                )
            ]
        return [
            SearchResult(
                "Acme Dental | Dallas Dentist",
                "https://acme.example",
                "Multi-location practice.",
                provider="duckduckgo",
            ),
            SearchResult(
                "Beta Dental | Fort Worth Dentist",
                "https://beta.example",
                "Dormant patient database.",
                provider="duckduckgo",
            ),
        ][:max_results]

    checkpoint = asyncio.run(
        run_tiered_research_with_search(
            directive,
            search=search,
            thread_id="tiered-live-search",
            checkpoint_dir=tmp_path / "checkpoints",
            artifact_dir=tmp_path / "artifacts",
            max_results=2,
            max_contact_queries_per_company=1,
        )
    )

    assert checkpoint.status == "review_required"
    assert [company.name for company in checkpoint.run.companies] == [
        "Acme Dental",
        "Beta Dental",
    ]
    assert [contact.contact_kind for contact in checkpoint.run.contacts] == [
        "company_email",
        "company_phone",
    ]
    assert checkpoint.run.contacts[0].email == "hello@acme.example"
    assert checkpoint.run.contacts[1].phone == "+18175550100"
    assert "review_required" in [event["status"] for event in checkpoint.events]
    assert any("dental companies" in query for query in queries)
    assert max(max_result_values) > 2
    events = {event["status"]: event for event in checkpoint.events}
    assert events["company_qualification_complete"]["rejected_company_count"] == 0
    assert events["contact_qualification_complete"] == {
        "status": "contact_qualification_complete",
        "message": "2 company contact point(s) promoted for review",
        "timestamp": events["contact_qualification_complete"]["timestamp"],
        "accepted_contact_count": 2,
        "accepted_contact_point_count": 2,
        "rejected_contact_count": 0,
        "needs_contact_count": 0,
    }
    assert checkpoint.qualification_counts == {
        "ready_company_count": 2,
        "company_prospect_count": 2,
        "ready_contact_count": 2,
        "contact_candidate_count": 2,
        "contact_point_count": 2,
        "qualified_company_count": 2,
        "companies_with_contacts_count": 2,
        "companies_with_contact_points_count": 2,
        "needs_contact_count": 0,
        "rejected_candidate_count": 0,
    }
    assert {record["status"] for record in checkpoint.qualification_audit} == {"accepted"}


def test_tiered_live_search_targets_companies_and_caps_contacts_per_company(tmp_path) -> None:
    directive = parse_directive_payload(
        {
            "industry": "hvac",
            "geography": "Dallas",
            "target_prospect_count": 2,
            "research_criteria": "owner operated",
            "preferred_contact_roles": ["owner"],
        }
    )

    async def search(query: str, max_results: int):
        if "site:alpha.example contact" in query:
            return []
        if "site:beta.example contact" in query:
            return [
                SearchResult(
                    "Beta Heating Contact",
                    "https://beta.example/contact",
                    (
                        '<a href="mailto:service@beta.example">Email service</a>'
                        '<a href="tel:214-555-0199">Call service</a>'
                    ),
                    provider="duckduckgo",
                ),
                SearchResult(
                    "Beta Heating Schedule",
                    "https://beta.example/schedule-service",
                    "Schedule service online.",
                    provider="duckduckgo",
                ),
            ]
        return [
            SearchResult(
                "Alpha Air Conditioning DFW | HVAC Repair",
                "https://alpha.example",
                "Alpha Air Conditioning serves Dallas homes.",
                provider="duckduckgo",
            ),
            SearchResult(
                "Beta Heating DFW | HVAC Repair",
                "https://beta.example",
                "Beta Heating serves Dallas homes.",
                provider="duckduckgo",
            ),
        ][:max_results]

    checkpoint = asyncio.run(
        run_tiered_research_with_search(
            directive,
            search=search,
            thread_id="tiered-live-contact-target",
            checkpoint_dir=tmp_path / "checkpoints",
            artifact_dir=tmp_path / "artifacts",
            max_results=3,
            max_contact_queries_per_company=1,
        )
    )

    assert [company.name for company in checkpoint.run.companies] == [
        "Alpha Air Conditioning",
        "Beta Heating",
    ]
    assert [contact.contact_kind for contact in checkpoint.run.contacts] == [
        "company_email",
        "company_phone",
    ]
    assert checkpoint.qualification_counts == {
        "ready_company_count": 2,
        "company_prospect_count": 2,
        "ready_contact_count": 2,
        "contact_candidate_count": 2,
        "contact_point_count": 2,
        "qualified_company_count": 2,
        "companies_with_contacts_count": 1,
        "companies_with_contact_points_count": 1,
        "needs_contact_count": 1,
        "rejected_candidate_count": 0,
    }
    events = {event["status"]: event for event in checkpoint.events}
    assert events["review_required"]["ready_company_count"] == 2
    assert events["review_required"]["ready_contact_count"] == 2
    assert events["review_required"]["needs_contact_count"] == 1


def test_tiered_live_search_uses_optional_company_contact_page_fetcher(tmp_path) -> None:
    directive = parse_directive_payload(
        {
            "industry": "plumbing",
            "geography": "Dallas",
            "target_prospect_count": 1,
            "research_criteria": "emergency repair",
        }
    )
    fetched_urls: list[str] = []

    async def search(query: str, _max_results: int):
        if "site:northpipe.example contact" in query:
            return []
        return [
            SearchResult(
                "North Pipe | Dallas Plumber",
                "https://northpipe.example",
                "Emergency plumbing in Dallas.",
                provider="duckduckgo",
            )
        ]

    def fetcher(url: str) -> str:
        fetched_urls.append(url)
        return (
            '<a href="mailto:dispatch@northpipe.example">Email dispatch</a>'
            '<a href="tel:972-555-0101">Call dispatch</a>'
        )

    checkpoint = asyncio.run(
        run_tiered_research_with_search(
            directive,
            search=search,
            thread_id="tiered-live-fetcher",
            checkpoint_dir=tmp_path / "checkpoints",
            artifact_dir=tmp_path / "artifacts",
            max_results=2,
            max_contact_queries_per_company=1,
            company_contact_page_fetcher=fetcher,
        )
    )

    assert fetched_urls == ["https://northpipe.example"]
    assert [contact.contact_kind for contact in checkpoint.run.contacts] == [
        "company_email",
        "company_phone",
    ]
    assert checkpoint.run.contacts[0].email == "dispatch@northpipe.example"
    assert checkpoint.run.contacts[1].phone == "+19725550101"
    assert checkpoint.qualification_counts["needs_contact_count"] == 0


def test_tiered_live_search_fetcher_skips_external_search_hits(tmp_path) -> None:
    directive = parse_directive_payload(
        {
            "industry": "electrical",
            "geography": "Dallas",
            "target_prospect_count": 1,
            "research_criteria": "home service",
        }
    )
    fetched_urls: list[str] = []

    async def search(query: str, _max_results: int):
        if "site:alpha.example contact" in query:
            return [
                SearchResult(
                    "Alpha Electric on Directory",
                    "https://directory.example/alpha-electric",
                    "Directory listing.",
                    provider="duckduckgo",
                )
            ]
        return [
            SearchResult(
                "Alpha Electric | Dallas Electrician",
                "https://alpha.example",
                "Residential electrical service in Dallas.",
                provider="duckduckgo",
            )
        ]

    def fetcher(url: str) -> str:
        fetched_urls.append(url)
        return '<a href="mailto:service@alpha.example">Email service</a>'

    checkpoint = asyncio.run(
        run_tiered_research_with_search(
            directive,
            search=search,
            thread_id="tiered-live-fetcher-official-only",
            checkpoint_dir=tmp_path / "checkpoints",
            artifact_dir=tmp_path / "artifacts",
            max_results=2,
            max_contact_queries_per_company=1,
            company_contact_page_fetcher=fetcher,
        )
    )

    assert fetched_urls == ["https://alpha.example"]
    assert [contact.email for contact in checkpoint.run.contacts] == ["service@alpha.example"]


def test_tiered_live_search_fetcher_failure_keeps_company_needing_contact(
    tmp_path,
) -> None:
    directive = parse_directive_payload(
        {
            "industry": "plumbing",
            "geography": "Dallas",
            "target_prospect_count": 1,
            "research_criteria": "emergency repair",
        }
    )

    async def search(query: str, _max_results: int):
        if "site:northpipe.example contact" in query:
            return []
        return [
            SearchResult(
                "North Pipe | Dallas Plumber",
                "https://northpipe.example",
                "Emergency plumbing in Dallas.",
                provider="duckduckgo",
            )
        ]

    def fetcher(_url: str) -> str:
        raise RuntimeError("browser unavailable")

    checkpoint = asyncio.run(
        run_tiered_research_with_search(
            directive,
            search=search,
            thread_id="tiered-live-fetcher-failure",
            checkpoint_dir=tmp_path / "checkpoints",
            artifact_dir=tmp_path / "artifacts",
            max_results=2,
            max_contact_queries_per_company=1,
            company_contact_page_fetcher=fetcher,
        )
    )

    assert [company.name for company in checkpoint.run.companies] == ["North Pipe"]
    assert checkpoint.run.contacts == ()
    assert checkpoint.qualification_counts["needs_contact_count"] == 1
    assert any("Contact page fetching had 1 failed" in warning for warning in checkpoint.warnings)
    assert any(
        record["status"] == "fetch_failed"
        and record["source_url"] == "https://northpipe.example"
        and record["error_type"] == "RuntimeError"
        for record in checkpoint.qualification_audit
    )


def test_tiered_live_search_filters_noise_and_does_not_create_contact_placeholder(
    tmp_path,
) -> None:
    directive = parse_directive_payload(
        {
            "industry": "hvac",
            "geography": "Dallas",
            "target_prospect_count": 1,
            "research_criteria": "owner operated",
            "preferred_contact_roles": ["owner"],
        }
    )

    async def search(query: str, _max_results: int):
        if "site:www.houkac.com contact" in query:
            return [
                SearchResult(
                    "7 Best HVAC Companies in Fort Worth, TX of 2026",
                    "https://www.consumeraffairs.com/homeowners/fort-worth-hvac.html",
                    "Reviewed HVAC companies and customer reviews.",
                    provider="duckduckgo",
                )
            ]
        return [
            SearchResult(
                "Local SEO for HVAC Companies in Dallas, TX",
                "https://www.vaza.ai/blog/local-seo-hvac-dallas",
                "Marketing agency page for HVAC lead generation.",
                provider="duckduckgo",
            ),
            SearchResult(
                "Houk Air Conditioning DFW | HVAC Repair",
                "https://www.houkac.com/air-conditioning",
                "Houk Air Conditioning serves homes across Dallas-Fort Worth.",
                provider="duckduckgo",
            ),
        ]

    checkpoint = asyncio.run(
        run_tiered_research_with_search(
            directive,
            search=search,
            thread_id="tiered-live-qualified",
            checkpoint_dir=tmp_path / "checkpoints",
            artifact_dir=tmp_path / "artifacts",
            max_results=2,
            max_contact_queries_per_company=1,
        )
    )

    assert [company.name for company in checkpoint.run.companies] == ["Houk Air Conditioning"]
    assert checkpoint.run.contacts == ()
    assert checkpoint.qualification_counts == {
        "ready_company_count": 1,
        "company_prospect_count": 1,
        "ready_contact_count": 0,
        "contact_candidate_count": 0,
        "contact_point_count": 0,
        "qualified_company_count": 1,
        "companies_with_contacts_count": 0,
        "companies_with_contact_points_count": 0,
        "needs_contact_count": 1,
        "rejected_candidate_count": 2,
    }
    assert {
        reason for record in checkpoint.qualification_audit for reason in record["reasons"]
    } >= {
        "non_target_marketing_page",
        "qualified_company_needs_contact",
        "no_actionable_company_contact_info",
    }
    events = {event["status"]: event for event in checkpoint.events}
    assert events["company_qualification_complete"]["rejected_company_count"] == 1
    assert events["contact_qualification_complete"]["accepted_contact_count"] == 0
    assert events["contact_qualification_complete"]["rejected_contact_count"] == 1
    assert events["contact_qualification_complete"]["needs_contact_count"] == 1
    artifact_payload = checkpoint.to_dict()
    assert artifact_payload["qualification_audit"]
    assert artifact_payload["metadata"]["needs_contact_count"] == 1
    assert artifact_payload["metadata"]["rejected_candidate_count"] == 2


def test_tiered_live_search_honors_negative_criteria(tmp_path) -> None:
    directive = parse_directive_payload(
        {
            "industry": "hvac",
            "geography": "Dallas",
            "target_prospect_count": 1,
            "research_criteria": "owner operated",
            "negative_criteria": "franchise",
            "preferred_contact_roles": ["owner"],
        }
    )

    async def search(query: str, _max_results: int):
        if "hvac companies" not in query:
            return []
        return [
            SearchResult(
                "Franchise HVAC DFW | HVAC Repair",
                "https://franchise-hvac.example",
                "Franchise HVAC is a national franchise serving Dallas.",
                provider="duckduckgo",
            )
        ]

    checkpoint = asyncio.run(
        run_tiered_research_with_search(
            directive,
            search=search,
            thread_id="tiered-live-negative-criteria",
            checkpoint_dir=tmp_path / "checkpoints",
            artifact_dir=tmp_path / "artifacts",
            max_results=1,
            max_contact_queries_per_company=1,
        )
    )

    assert checkpoint.run.companies == ()
    assert checkpoint.run.contacts == ()
    assert checkpoint.qualification_counts == {
        "ready_company_count": 0,
        "company_prospect_count": 0,
        "ready_contact_count": 0,
        "contact_candidate_count": 0,
        "contact_point_count": 0,
        "qualified_company_count": 0,
        "companies_with_contacts_count": 0,
        "companies_with_contact_points_count": 0,
        "needs_contact_count": 0,
        "rejected_candidate_count": 1,
    }
    assert checkpoint.qualification_audit[0]["reasons"] == ["negative_criteria_match:franchise"]
    events = {event["status"]: event for event in checkpoint.events}
    assert events["company_qualification_complete"]["rejected_company_count"] == 1
    assert events["contact_qualification_complete"]["accepted_contact_count"] == 0


def test_tiered_thread_id_rejects_path_traversal(tmp_path) -> None:
    directive = parse_directive_payload(
        {
            "industry": "dental",
            "geography": "North Texas",
            "target_prospect_count": 1,
            "research_criteria": "multi-location Invisalign",
        }
    )

    with pytest.raises(ValueError, match="thread_id"):
        run_tiered_research(
            directive,
            thread_id="../escape",
            checkpoint_dir=tmp_path / "checkpoints",
            artifact_dir=tmp_path / "artifacts",
            mock_results=("Acme Dental|https://acme.example|Strong fit|mock",),
        )


def test_tiered_final_enrichment_requires_approved_ids(tmp_path) -> None:
    checkpoint = _run_review_checkpoint(tmp_path)

    blocked = resume_tiered_research(
        checkpoint.thread_id,
        checkpoint_dir=tmp_path / "checkpoints",
        artifact_dir=tmp_path / "artifacts",
        enable_final_enrichment=True,
    )
    assert blocked.status == "final_enrichment_blocked"

    company_id = checkpoint.run.companies[0].company_id
    approval = ApprovedProspectSelection(
        approved_company_ids=(company_id,),
        approved_contact_ids=(),
        reviewer="tester",
    )
    enriched = resume_tiered_research(
        checkpoint.thread_id,
        checkpoint_dir=tmp_path / "checkpoints",
        artifact_dir=tmp_path / "artifacts",
        approval_selection=approval,
        enable_final_enrichment=True,
        mock_final_enrichment=(f"{company_id}||Approved only|mock",),
    )

    assert enriched.status == "final_enrichment_complete"
    assert enriched.final_enrichment[0].company_id == company_id
    assert enriched.final_enrichment[0].contact_id == ""
    assert enriched.final_enrichment[0].contact_snapshot is None
    assert enriched.run.companies[0].fit_score == checkpoint.run.companies[0].fit_score


def test_tiered_final_enrichment_persists_injected_exa_records(tmp_path) -> None:
    checkpoint = _run_review_checkpoint(tmp_path)
    company_id = checkpoint.run.companies[0].company_id
    approval = ApprovedProspectSelection(
        approved_company_ids=(company_id,),
        approved_contact_ids=(),
        reviewer="tester",
    )
    queries: list[str] = []

    async def exa_search(query: str, max_results: int):
        queries.append(query)
        assert max_results == 3
        return [
            SearchResult(
                "Ada leadership profile",
                "https://exa.example/ada",
                "Leadership context from Exa.",
                provider="exa",
            )
        ]

    records = asyncio.run(
        enrich_selected_prospects(
            checkpoint.run,
            approval,
            exa_search,
        )
    )
    enriched = resume_tiered_research(
        checkpoint.thread_id,
        checkpoint_dir=tmp_path / "checkpoints",
        artifact_dir=tmp_path / "artifacts",
        approval_selection=approval,
        enable_final_enrichment=True,
        final_enrichment_records=records,
    )

    assert "Acme Dental" in queries[0]
    assert enriched.status == "final_enrichment_complete"
    assert enriched.final_enrichment[0].provider == "exa"
    assert enriched.final_enrichment[0].contact_id == ""
    assert "final_enrichment_csv" in enriched.artifact_paths


def test_exa_final_enrichment_uses_company_contact_point_context() -> None:
    directive = parse_directive_payload(
        {
            "industry": "plumbing",
            "geography": "Dallas",
            "target_prospect_count": 1,
            "research_criteria": "emergency repair",
        }
    )
    company = CompanyProspect(
        company_id="company_north_pipe",
        name="North Pipe",
        website="https://northpipe.example",
        evidence_ids=("ev_company",),
    )
    contact = ContactCandidate(
        contact_id="contact_north_pipe_email",
        company_id=company.company_id,
        name="Company email",
        contact_kind="company_email",
        label="Company email",
        email="dispatch@northpipe.example",
        url="mailto:dispatch@northpipe.example",
        contact_url="mailto:dispatch@northpipe.example",
        source_url="https://northpipe.example/contact",
        evidence_ids=("ev_contact",),
    )
    run = TieredResearchRun(
        run_id="company-contact-final-enrichment",
        directive=directive,
        companies=(company,),
        contacts=(contact,),
    )
    approval = ApprovedProspectSelection(
        approved_company_ids=(company.company_id,),
        approved_contact_ids=(contact.contact_id,),
    )
    queries: list[str] = []

    async def exa_search(query: str, _max_results: int):
        queries.append(query)
        return [
            SearchResult(
                "North Pipe company context",
                "https://exa.example/north-pipe",
                "Company-level business context from Exa.",
                provider="exa",
            )
        ]

    records = asyncio.run(enrich_selected_prospects(run, approval, exa_search))

    assert "dispatch@northpipe.example" in queries[0]
    assert "https://northpipe.example/contact" in queries[0]
    assert "leadership context" not in queries[0]
    assert records[0].contact_snapshot is not None
    assert records[0].contact_snapshot["contact_kind"] == "company_email"
    assert records[0].contact_snapshot["email"] == "dispatch@northpipe.example"
    assert "North Pipe Company email" in records[0].summary


def test_tiered_final_enrichment_supports_company_only_approval(tmp_path) -> None:
    checkpoint = _run_review_checkpoint(tmp_path)
    company_id = checkpoint.run.companies[0].company_id
    approval = ApprovedProspectSelection(
        approved_company_ids=(company_id,),
        approved_contact_ids=(),
        reviewer="tester",
    )
    queries: list[str] = []

    async def exa_search(query: str, _max_results: int):
        queries.append(query)
        return [
            SearchResult(
                "Acme company context",
                "https://exa.example/acme",
                "Company-level business context from Exa.",
                provider="exa",
            )
        ]

    records = asyncio.run(
        enrich_selected_prospects(
            checkpoint.run,
            approval,
            exa_search,
        )
    )

    assert len(records) == 1
    assert "Acme Dental" in queries[0]
    assert records[0].company_id == company_id
    assert records[0].contact_id == ""
    assert "Acme Dental" in records[0].summary


def test_exa_final_enrichment_rejects_non_exa_search_results(tmp_path) -> None:
    checkpoint = _run_review_checkpoint(tmp_path)
    approval = ApprovedProspectSelection(
        approved_company_ids=(checkpoint.run.companies[0].company_id,),
        approved_contact_ids=(),
    )

    async def non_exa_search(_query: str, _max_results: int):
        return [
            SearchResult(
                "Fallback result",
                "https://fallback.example",
                "Not allowed.",
                provider="duckduckgo",
            )
        ]

    with pytest.raises(ValueError, match="Exa results only"):
        asyncio.run(enrich_selected_prospects(checkpoint.run, approval, non_exa_search))


def test_tiered_final_enrichment_rejects_unapproved_ids(tmp_path) -> None:
    checkpoint = _run_review_checkpoint(tmp_path)
    approval = ApprovedProspectSelection(
        approved_company_ids=(checkpoint.run.companies[0].company_id,),
        approved_contact_ids=(),
    )

    with pytest.raises(ValueError, match="outside approved company_id"):
        resume_tiered_research(
            checkpoint.thread_id,
            checkpoint_dir=tmp_path / "checkpoints",
            artifact_dir=tmp_path / "artifacts",
            approval_selection=approval,
            enable_final_enrichment=True,
            mock_final_enrichment=("company_other||Not approved|mock",),
        )


def test_tiered_final_enrichment_rejects_cross_company_contact_pair(tmp_path) -> None:
    directive = parse_directive_payload(
        {
            "industry": "dental",
            "geography": "North Texas",
            "target_prospect_count": 2,
            "research_criteria": "multi-location Invisalign",
        }
    )

    async def search(query: str, max_results: int):
        if "site:acme.example contact" in query:
            return [
                SearchResult(
                    "Acme Dental Contact",
                    "https://acme.example/contact",
                    '<a href="mailto:hello@acme.example">Email Acme</a>',
                    provider="duckduckgo",
                )
            ][:max_results]
        if "site:beta.example contact" in query:
            return [
                SearchResult(
                    "Beta Dental Contact",
                    "https://beta.example/contact",
                    '<a href="mailto:hello@beta.example">Email Beta</a>',
                    provider="duckduckgo",
                )
            ][:max_results]
        return [
            SearchResult(
                "Acme Dental | Dallas Dentist",
                "https://acme.example",
                "Strong fit.",
                provider="duckduckgo",
            ),
            SearchResult(
                "Beta Dental | Dallas Dentist",
                "https://beta.example",
                "Strong fit.",
                provider="duckduckgo",
            ),
        ][:max_results]

    checkpoint = asyncio.run(
        run_tiered_research_with_search(
            directive,
            search=search,
            thread_id="tiered-cross-pair",
            checkpoint_dir=tmp_path / "checkpoints",
            artifact_dir=tmp_path / "artifacts",
            max_results=2,
            max_contact_queries_per_company=1,
        )
    )
    company_1 = checkpoint.run.companies[0].company_id
    company_2 = checkpoint.run.companies[1].company_id
    contact_1 = checkpoint.run.contacts[0].contact_id
    contact_2 = checkpoint.run.contacts[1].contact_id
    approval = ApprovedProspectSelection(
        approved_company_ids=(company_1, company_2),
        approved_contact_ids=(contact_1, contact_2),
    )

    with pytest.raises(ValueError, match="does not belong to company_id"):
        resume_tiered_research(
            checkpoint.thread_id,
            checkpoint_dir=tmp_path / "checkpoints",
            artifact_dir=tmp_path / "artifacts",
            approval_selection=approval,
            enable_final_enrichment=True,
            mock_final_enrichment=(f"{company_1}|{contact_2}|Cross-paired|mock",),
        )
