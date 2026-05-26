from __future__ import annotations

import pytest

from deep_research_agent.tiered_models import ApprovedProspectSelection
from deep_research_agent.tiered_runtime import (
    parse_directive_payload,
    resume_tiered_research,
    run_tiered_research,
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
