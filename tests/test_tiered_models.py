from __future__ import annotations

import pytest

from deep_research_agent.tiered_models import (
    BrowserCapture,
    CompanyProspect,
    ContactCandidate,
    ContactPersonalization,
    PersonalizationSignal,
    SearchDirective,
    TieredModelValidationError,
    TieredResearchRun,
    coerce_search_directive,
    tiered_research_schema,
)


def _directive() -> SearchDirective:
    return SearchDirective(
        industry="dental",
        target_prospect_count=2,
        geographic_area="DFW area",
        research_criteria="Prioritize multi-location practices with growth signals.",
        preferred_contact_roles=("owner", "practice manager"),
        source_preferences=("official sites", "directories"),
        evidence_depth="standard",
        browser_capture="text_extraction",
    )


def _company() -> CompanyProspect:
    return CompanyProspect(
        company_id="company_001",
        name="Example Dental Group",
        website="https://example.com",
        industry="dental",
        geographic_area="DFW area",
        locations=("Dallas, TX", "Plano, TX"),
        fit_score=0.82,
        fit_rationale="Matches dental and DFW with multi-location evidence.",
        evidence_ids=("ev_company_001",),
        source_confidence="medium",
    )


def _contact() -> ContactCandidate:
    return ContactCandidate(
        contact_id="contact_001",
        company_id="company_001",
        name="Jane Smith",
        title="Practice Manager",
        role_category="operator",
        profile_urls=("https://example.com/team/jane-smith",),
        contact_confidence=0.74,
        evidence_ids=("ev_contact_001",),
        notes="Listed on the team page.",
    )


def _personalization() -> ContactPersonalization:
    return ContactPersonalization(
        contact_id="contact_001",
        personalization_signals=(
            PersonalizationSignal(
                contact_id="contact_001",
                signal="Practice opened a second DFW location.",
                message_angle="Lead with patient reactivation for multi-location growth.",
                evidence_ids=("ev_person_001",),
                confidence="medium",
            ),
        ),
        suggested_opening_line="Saw Example Dental Group expanded in DFW.",
        do_not_claim=("Do not claim Jane owns marketing unless a source confirms it.",),
    )


def test_search_directive_contract_validates_required_fields_and_serializes_lists() -> None:
    directive = _directive()

    assert directive.preferred_contact_roles == ("owner", "practice manager")
    assert directive.to_dict()["preferred_contact_roles"] == ["owner", "practice manager"]
    assert directive.to_dict()["browser_capture"] == "text_extraction"

    with pytest.raises(TieredModelValidationError, match="target_prospect_count"):
        SearchDirective(
            industry="dental",
            target_prospect_count=0,
            geographic_area="DFW",
            research_criteria="Find practices.",
        )


def test_tiered_research_schema_documents_required_contract_sections() -> None:
    schema = tiered_research_schema()

    assert schema["required"] == [
        "run_id",
        "directive",
        "companies",
        "contacts",
        "personalizations",
    ]
    assert schema["properties"]["directive"]["properties"]["evidence_depth"]["enum"] == [
        "fast",
        "standard",
        "deep",
    ]
    assert "browser_capture" in schema["definitions"]
    assert schema["definitions"]["contact"]["properties"]["role_category"]["enum"] == [
        "owner",
        "executive",
        "operator",
        "marketing",
        "unknown",
    ]


def test_complete_tiered_run_serializes_company_contact_and_personalization_records() -> None:
    run = TieredResearchRun(
        run_id="run_001",
        directive=_directive(),
        companies=(_company(),),
        contacts=(_contact(),),
        personalizations=(_personalization(),),
        browser_captures=(
            BrowserCapture(
                browser_capture_id="cap_001",
                url="https://example.com/team",
                company_id="company_001",
                contact_id="contact_001",
                captured_at="2026-05-26T20:00:00Z",
                text_excerpt="Jane Smith is listed as Practice Manager.",
                screenshot_path="artifacts/run_001/captures/cap_001.png",
                dom_snapshot_path="artifacts/run_001/captures/cap_001.html",
                evidence_ids=("ev_contact_001",),
            ),
        ),
        warnings=("human_review_required_before_outreach",),
    )

    payload = run.to_dict()

    assert payload["directive"]["industry"] == "dental"
    assert payload["companies"][0]["locations"] == ["Dallas, TX", "Plano, TX"]
    assert payload["contacts"][0]["role_category"] == "operator"
    assert payload["personalizations"][0]["personalization_signals"][0]["message_angle"].startswith(
        "Lead with patient reactivation"
    )
    assert payload["browser_captures"][0]["evidence_ids"] == ["ev_contact_001"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"fit_score": 1.1}, "fit_score must be between"),
        ({"source_confidence": "certain"}, "source_confidence"),
        ({"evidence_ids": ()}, "evidence_ids requires"),
    ],
)
def test_company_contract_rejects_invalid_scores_confidence_and_uncited_records(
    kwargs: dict[str, object], match: str
) -> None:
    data = {
        "company_id": "company_001",
        "name": "Example Dental Group",
        "fit_score": 0.5,
        "evidence_ids": ("ev_company_001",),
    }
    data.update(kwargs)

    with pytest.raises(TieredModelValidationError, match=match):
        CompanyProspect(**data)  # type: ignore[arg-type]


def test_cross_record_validation_rejects_unknown_relationships_and_duplicates() -> None:
    with pytest.raises(TieredModelValidationError, match="unknown company_id"):
        TieredResearchRun(
            run_id="run_001",
            directive=_directive(),
            companies=(_company(),),
            contacts=(
                ContactCandidate(
                    contact_id="contact_002",
                    company_id="missing_company",
                    name="John Doe",
                    evidence_ids=("ev_contact_002",),
                ),
            ),
        )

    with pytest.raises(TieredModelValidationError, match="duplicate contact_id"):
        TieredResearchRun(
            run_id="run_001",
            directive=_directive(),
            companies=(_company(),),
            contacts=(_contact(), _contact()),
        )

    with pytest.raises(TieredModelValidationError, match="unknown contact_id"):
        TieredResearchRun(
            run_id="run_001",
            directive=_directive(),
            companies=(_company(),),
            contacts=(_contact(),),
            personalizations=(
                ContactPersonalization(
                    contact_id="missing_contact",
                    personalization_signals=(),
                ),
            ),
        )


def test_personalization_bundle_rejects_signal_for_different_contact() -> None:
    with pytest.raises(TieredModelValidationError, match="signal contact_id must match"):
        ContactPersonalization(
            contact_id="contact_001",
            personalization_signals=(
                PersonalizationSignal(
                    contact_id="contact_002",
                    signal="Recent expansion.",
                    message_angle="Mention growth.",
                    evidence_ids=("ev_person_001",),
                ),
            ),
        )


def test_coerce_search_directive_accepts_mapping_input() -> None:
    directive = coerce_search_directive(
        {
            "industry": "HVAC",
            "target_prospect_count": "3",
            "geographic_area": "North Texas",
            "research_criteria": "Owner-operated businesses.",
            "preferred_contact_roles": ["owner", "marketing director"],
        }
    )

    assert directive.target_prospect_count == 3
    assert directive.preferred_contact_roles == ("owner", "marketing director")
    assert directive.evidence_depth == "standard"
