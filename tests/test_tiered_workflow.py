from __future__ import annotations

import pytest

from deep_research_agent.tiered_workflow import (
    TIERED_WORKFLOW_TOPOLOGY,
    TieredResearchWorkflow,
    TieredWorkflowConfig,
    normalize_search_directive,
    qualify_companies,
)


def _directive() -> dict[str, object]:
    return {
        "industry": "dental",
        "target_prospect_count": 2,
        "geographic_area": "DFW area",
        "research_criteria": "prioritize multi-location practices",
        "preferred_contact_roles": ["owner", "practice manager"],
    }


def test_search_directive_requires_tiered_contract_fields() -> None:
    with pytest.raises(ValueError, match="industry, target_prospect_count"):
        normalize_search_directive(
            {
                "geographic_area": "DFW",
                "research_criteria": "reactivation opportunity",
            }
        )

    normalized = normalize_search_directive(
        {
            "industry": " dental ",
            "target_prospect_count": "3",
            "geographic_area": " DFW ",
            "research_criteria": " reactivation ",
        }
    )

    assert normalized["industry"] == "dental"
    assert normalized["target_prospect_count"] == 3
    assert normalized["geographic_area"] == "DFW"
    assert normalized["preferred_contact_roles"] == []


_CHAIN_DENTAL = {
    "name": "Chain Dental",
    "website": "https://chain.example",
    "fit_score": 0.96,
    "disqualification_flags": ["national_chain"],
}
_LOW_FIT_DENTAL = {
    "name": "Low Fit Dental",
    "website": "https://low-fit.example",
    "fit_score": 0.2,
}


def test_company_qualification_dedupes_ranks_filters_and_caps() -> None:
    qualified = qualify_companies(
        [
            {"name": "Acme Dental", "website": "https://acme.example", "fit_score": 0.72},
            {
                "name": "Acme Dental Duplicate",
                "website": "https://www.acme.example/about",
                "fit_score": 0.99,
            },
            {"name": "Bright Dental", "website": "https://bright.example", "fit_score": 0.91},
            _CHAIN_DENTAL,
            _LOW_FIT_DENTAL,
        ],
        target_count=2,
        min_fit_score=0.55,
    )

    assert [company["name"] for company in qualified] == ["Bright Dental", "Acme Dental"]
    assert [company["company_id"] for company in qualified] == [
        "company_bright_example",
        "company_acme_example",
    ]


def test_tiered_workflow_runs_in_order_and_review_gates_outputs() -> None:
    calls: list[str] = []

    async def company_discovery(directive):
        calls.append(f"company:{directive['industry']}:{directive['geographic_area']}")
        return [
            {
                "name": "Acme Dental",
                "website": "https://acme.example",
                "fit_score": 0.88,
                "evidence_ids": ["ev_company_1"],
                "evidence": [{"evidence_id": "ev_company_1", "content": "Acme has DFW offices"}],
            },
            {
                "name": "Bright Dental",
                "website": "https://bright.example",
                "fit_score": 0.8,
                "evidence_ids": ["ev_company_2"],
            },
        ]

    def contact_discovery(_directive, company):
        calls.append(f"contact:{company['company_id']}")
        return [
            {
                "name": f"{company['name']} Owner",
                "title": "Owner",
                "evidence_ids": [f"ev_contact_{company['company_id']}"],
            }
        ]

    async def personalization_research(_directive, _company, contact):
        calls.append(f"personalization:{contact['contact_id']}")
        return [
            {
                "signal": f"{contact['name']} is tied to local expansion.",
                "message_angle": "Lead with patient reactivation for growth.",
                "evidence_ids": ["ev_person_1"],
            }
        ]

    def artifact_writer(state):
        calls.append(f"artifact:{state['status']}")
        return {"tiered_prospect_research": "artifacts/tiered_prospect_research.json"}

    state = TieredResearchWorkflow(
        company_discovery=company_discovery,
        contact_discovery=contact_discovery,
        personalization_research=personalization_research,
        artifact_writer=artifact_writer,
    ).run(_directive())

    assert state.status == "review_required"
    assert state.review_required is True
    assert [company["name"] for company in state.companies] == ["Acme Dental", "Bright Dental"]
    assert len(state.contacts) == 2
    assert len(state.personalization_signals) == 2
    assert state.artifact_paths == {
        "tiered_prospect_research": "artifacts/tiered_prospect_research.json"
    }
    started_or_required_tiers = [
        event["tier"] for event in state.events if event["event"] in {"started", "required"}
    ]
    assert started_or_required_tiers == [
        "directive_parser",
        "company_discovery",
        "company_qualification",
        "contact_discovery",
        "personalization_research",
        "artifact_writer",
        "review",
    ]
    assert TIERED_WORKFLOW_TOPOLOGY[-1] == "review"
    assert calls[0] == "company:dental:DFW area"
    assert calls[-1] == "artifact:review_required"


def test_tiered_workflow_preserves_partial_results_with_actionable_warnings() -> None:
    def company_discovery(_directive):
        return [
            {
                "name": "Solo Dental",
                "website": "https://solo.example",
                "fit_score": 0.9,
            },
            _LOW_FIT_DENTAL,
            _CHAIN_DENTAL,
        ]

    def contact_discovery(_directive, _company):
        return []

    def personalization_research(_directive, _company, _contact):
        raise AssertionError("personalization should not run without contacts")

    state = TieredResearchWorkflow(
        company_discovery=company_discovery,
        contact_discovery=contact_discovery,
        personalization_research=personalization_research,
        config=TieredWorkflowConfig(require_human_review=False),
    ).run(_directive())

    assert state.status == "completed"
    assert [company["name"] for company in state.companies] == ["Solo Dental"]
    assert state.contacts == ()
    assert state.personalization_signals == ()
    assert "Only 1 qualified company" in state.warnings[0]
    assert "No contact candidates found for company_id=company_solo_example" in state.warnings[1]
    assert state.events[-1]["tier"] == "personalization_research"
