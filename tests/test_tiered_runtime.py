from __future__ import annotations

import asyncio

import pytest

from async_multi_search import SearchResult
from deep_research_agent.final_enrichment import enrich_selected_prospects
from deep_research_agent.tiered_models import ApprovedProspectSelection
from deep_research_agent.tiered_runtime import (
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
        if "Acme Dental owner" in query:
            assert max_results == 2
            return [
                SearchResult(
                    "Dr. Ada Lovelace - Owner at Acme Dental",
                    "https://acme.example/team/ada",
                    "Owner profile.",
                    provider="duckduckgo",
                )
            ]
        if "Beta Dental owner" in query:
            assert max_results == 2
            return [
                SearchResult(
                    "Grace Hopper - CEO at Beta Dental",
                    "https://beta.example/team/grace",
                    "CEO profile.",
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
    assert [contact.name for contact in checkpoint.run.contacts] == [
        "Dr. Ada Lovelace",
        "Grace Hopper",
    ]
    assert "review_required" in [event["status"] for event in checkpoint.events]
    assert any("dental companies" in query for query in queries)
    assert max(max_result_values) > 2
    events = {event["status"]: event for event in checkpoint.events}
    assert events["company_qualification_complete"]["rejected_company_count"] == 0
    assert events["contact_qualification_complete"] == {
        "status": "contact_qualification_complete",
        "message": "2 qualified contact-level prospect row(s) promoted for review",
        "timestamp": events["contact_qualification_complete"]["timestamp"],
        "accepted_contact_count": 2,
        "rejected_contact_count": 0,
        "needs_contact_count": 0,
    }
    assert checkpoint.qualification_counts == {
        "ready_contact_count": 2,
        "qualified_company_count": 2,
        "needs_contact_count": 0,
        "rejected_candidate_count": 0,
    }
    assert {
        record["status"]
        for record in checkpoint.qualification_audit
    } == {"accepted"}


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
        if "Houk Air Conditioning owner" in query:
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

    assert [company.name for company in checkpoint.run.companies] == [
        "Houk Air Conditioning"
    ]
    assert checkpoint.run.contacts == ()
    assert not any(
        contact.name.startswith("Review Contact at")
        for contact in checkpoint.run.contacts
    )
    assert checkpoint.qualification_counts == {
        "ready_contact_count": 0,
        "qualified_company_count": 1,
        "needs_contact_count": 1,
        "rejected_candidate_count": 2,
    }
    assert {
        reason
        for record in checkpoint.qualification_audit
        for reason in record["reasons"]
    } >= {
        "non_target_marketing_page",
        "qualified_company_needs_contact",
        "article_or_list_title",
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
        "ready_contact_count": 0,
        "qualified_company_count": 0,
        "needs_contact_count": 0,
        "rejected_candidate_count": 1,
    }
    assert checkpoint.qualification_audit[0]["reasons"] == [
        "negative_criteria_match:franchise"
    ]
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
    contact_id = checkpoint.run.contacts[0].contact_id
    approval = ApprovedProspectSelection(
        approved_company_ids=(company_id,),
        approved_contact_ids=(contact_id,),
        reviewer="tester",
    )
    enriched = resume_tiered_research(
        checkpoint.thread_id,
        checkpoint_dir=tmp_path / "checkpoints",
        artifact_dir=tmp_path / "artifacts",
        approval_selection=approval,
        enable_final_enrichment=True,
        mock_final_enrichment=(f"{company_id}|{contact_id}|Approved only|mock",),
    )

    assert enriched.status == "final_enrichment_complete"
    assert enriched.final_enrichment[0].company_id == company_id
    assert enriched.run.companies[0].fit_score == checkpoint.run.companies[0].fit_score


def test_tiered_final_enrichment_persists_injected_exa_records(tmp_path) -> None:
    checkpoint = _run_review_checkpoint(tmp_path)
    company_id = checkpoint.run.companies[0].company_id
    contact_id = checkpoint.run.contacts[0].contact_id
    approval = ApprovedProspectSelection(
        approved_company_ids=(company_id,),
        approved_contact_ids=(contact_id,),
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
    assert enriched.final_enrichment[0].contact_id == contact_id
    assert "final_enrichment_csv" in enriched.artifact_paths


def test_exa_final_enrichment_rejects_non_exa_search_results(tmp_path) -> None:
    checkpoint = _run_review_checkpoint(tmp_path)
    approval = ApprovedProspectSelection(
        approved_company_ids=(checkpoint.run.companies[0].company_id,),
        approved_contact_ids=(checkpoint.run.contacts[0].contact_id,),
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
        approved_contact_ids=(checkpoint.run.contacts[0].contact_id,),
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
    checkpoint = run_tiered_research(
        directive,
        thread_id="tiered-cross-pair",
        checkpoint_dir=tmp_path / "checkpoints",
        artifact_dir=tmp_path / "artifacts",
        mock_results=(
            "Acme Dental|https://acme.example|Strong fit|mock",
            "Beta Dental|https://beta.example|Strong fit|mock",
        ),
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
