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

    async def search(query: str, max_results: int):
        queries.append(query)
        assert max_results == 2
        if "Acme Dental owner" in query:
            return [
                SearchResult(
                    "Dr. Ada Lovelace - Owner at Acme Dental",
                    "https://acme.example/team/ada",
                    "Owner profile.",
                    provider="duckduckgo",
                )
            ]
        if "Beta Dental owner" in query:
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
