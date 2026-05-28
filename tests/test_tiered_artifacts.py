from __future__ import annotations

import csv
import json

import pytest

from deep_research_agent.tiered_artifacts import (
    TIERED_ARTIFACT_SCHEMA_VERSION,
    BrowserCapture,
    CompanyProspect,
    ContactCandidate,
    ContactPersonalization,
    PersonalizationSignal,
    SearchDirective,
    TieredResearchRun,
    build_tiered_artifact_payload,
    write_tiered_artifacts,
)


def _sample_run() -> TieredResearchRun:
    return TieredResearchRun(
        run_id="demo-thread",
        directive=SearchDirective(
            industry="orthodontics",
            geographic_area="North Texas",
            target_prospect_count=2,
            research_criteria="multi-location; offers Invisalign",
        ),
        companies=(
            CompanyProspect(
                company_id="company_001",
                name="Acme Ortho",
                website="https://acmeortho.example",
                industry="orthodontics",
                geographic_area="Dallas-Fort Worth",
                locations=("Dallas", "Fort Worth"),
                fit_score=0.91,
                fit_rationale="Multi-location practice | public Invisalign page",
                evidence_ids=("ev_company_001",),
                source_confidence="high",
            ),
        ),
        contacts=(
            ContactCandidate(
                contact_id="contact_001",
                company_id="company_001",
                name="Dr. Ada Lovelace",
                role_category="owner",
                title="owner DDS",
                profile_urls=("https://acmeortho.example/team/ada",),
                contact_confidence=0.82,
                evidence_ids=("ev_contact_001",),
            ),
        ),
        personalizations=(
            ContactPersonalization(
                contact_id="contact_001",
                personalization_signals=(
                    PersonalizationSignal(
                        contact_id="contact_001",
                        signal="Mentions teen Invisalign expansion",
                        message_angle="Lead with teen Invisalign patient reactivation.",
                        evidence_ids=("ev_signal_001",),
                        confidence="high",
                    ),
                    PersonalizationSignal(
                        contact_id="contact_001",
                        signal="Opened a second location",
                        message_angle="Ask about local patient reactivation after expansion.",
                        evidence_ids=("ev_signal_002",),
                    ),
                ),
                suggested_opening_line="Saw the second-location expansion.",
                do_not_claim=("Do not claim patient volume.",),
            ),
        ),
        browser_captures=(
            BrowserCapture(
                browser_capture_id="cap_001",
                url="https://acmeortho.example/team",
                company_id="company_001",
                contact_id="contact_001",
                captured_at="2026-05-26T20:00:00Z",
                text_excerpt="Dr. Ada Lovelace, owner",
                screenshot_path="captures/cap_001.png",
                dom_snapshot_path="captures/cap_001.html",
                evidence_ids=("ev_contact_001",),
            ),
        ),
        warnings=("Human review required before outreach.",),
    )


def test_write_tiered_artifacts_emits_nested_json_csvs_and_markdown(tmp_path) -> None:
    run = _sample_run()

    result = write_tiered_artifacts(run, tmp_path / "nested", metadata={"story": "G001"})

    assert result.company_count == 1
    assert result.contact_count == 1
    assert result.personalization_count == 2
    assert result.json_path.name == "tiered_prospect_research.json"
    assert result.companies_csv_path.name == "companies.csv"
    assert result.contacts_csv_path.name == "contacts.csv"
    assert result.personalization_csv_path.name == "personalization.csv"
    assert result.markdown_path.name == "research_report.md"

    payload = json.loads(result.json_path.read_text())
    assert payload["schema_version"] == TIERED_ARTIFACT_SCHEMA_VERSION
    assert payload["metadata"] == {
        "story": "G001",
        "ready_company_count": 1,
        "company_prospect_count": 1,
        "ready_contact_count": 1,
        "contact_candidate_count": 1,
        "qualified_company_count": 1,
        "companies_with_contacts_count": 1,
        "needs_contact_count": 0,
        "rejected_candidate_count": 0,
    }
    assert payload["run_id"] == "demo-thread"
    assert payload["directive"]["industry"] == "orthodontics"
    assert payload["companies"][0]["company_id"] == "company_001"
    assert payload["contacts"][0]["company_id"] == "company_001"
    assert payload["personalization_signals"][0]["contact_id"] == "contact_001"
    assert payload["browser_captures"][0]["screenshot_path"] == "captures/cap_001.png"
    assert payload["qualification_audit"] == []

    with result.companies_csv_path.open(newline="") as handle:
        company_rows = list(csv.DictReader(handle))
    assert company_rows[0]["evidence_ids"] == '["ev_company_001"]'
    assert company_rows[0]["locations"] == '["Dallas", "Fort Worth"]'

    with result.contacts_csv_path.open(newline="") as handle:
        contact_rows = list(csv.DictReader(handle))
    assert contact_rows[0]["name"] == "Dr. Ada Lovelace"

    with result.personalization_csv_path.open(newline="") as handle:
        signal_rows = list(csv.DictReader(handle))
    assert [row["signal"] for row in signal_rows] == [
        "Mentions teen Invisalign expansion",
        "Opened a second location",
    ]

    markdown = result.markdown_path.read_text()
    assert "# Tiered Prospect Research Report" in markdown
    assert "## Directive" in markdown
    assert "### Company: Acme Ortho" in markdown
    assert "Multi-location practice \\| public Invisalign page" in markdown
    assert "##### Contact: Dr. Ada Lovelace, owner DDS" in markdown
    assert "Lead with teen Invisalign patient reactivation." in markdown
    assert "Do not claim patient volume." in markdown
    assert "Company prospects: 1" in markdown
    assert "Contact candidates: 1" in markdown
    assert "Rejected/noisy candidates: 0" in markdown
    assert "## Warnings and Human Review Items" in markdown


def test_write_tiered_artifacts_renders_additive_qualification_audit(tmp_path) -> None:
    run = _sample_run()

    result = write_tiered_artifacts(
        run,
        tmp_path,
        metadata={
            "qualification_audit": [
                {
                    "audit_id": "qa-needs-contact",
                    "tier": "company",
                    "status": "needs_contact",
                    "company_id": "company_002",
                    "company_name": "Beta Ortho",
                    "source_title": "Beta Ortho official site",
                    "source_url": "https://beta.example",
                    "reasons": ["no_verified_contact"],
                },
                {
                    "audit_id": "qa-rejected",
                    "tier": "company",
                    "status": "rejected",
                    "source_title": "Best orthodontists in Dallas",
                    "source_url": "https://directory.example/best",
                    "reasons": ["listicle_or_directory"],
                },
            ],
        },
    )

    payload = json.loads(result.json_path.read_text())
    assert [record["status"] for record in payload["qualification_audit"]] == [
        "needs_contact",
        "rejected",
    ]
    assert payload["metadata"]["ready_contact_count"] == 1
    assert payload["metadata"]["qualified_company_count"] == 1
    assert payload["metadata"]["needs_contact_count"] == 1
    assert payload["metadata"]["rejected_candidate_count"] == 1
    assert "qualification_audit" not in payload["metadata"]

    markdown = result.markdown_path.read_text()
    assert "## Qualified companies needing contacts" in markdown
    assert "Beta Ortho" in markdown
    assert "no_verified_contact" in markdown
    assert "## Rejected/noisy candidates" in markdown
    assert "Best orthodontists in Dallas" in markdown
    assert "listicle_or_directory" in markdown


def test_build_tiered_payload_accepts_mapping_records() -> None:
    payload = build_tiered_artifact_payload(
        {
            "directive": {
                "industry": "med spa",
                "geography": "Austin",
                "target_count": 1,
                "criteria": "membership offers",
            },
            "companies": [
                {
                    "company_id": "company_001",
                    "name": "Glow Co",
                    "evidence_ids": ["ev_company_001"],
                }
            ],
            "contacts": [
                {
                    "contact_id": "contact_001",
                    "company_id": "company_001",
                    "name": "Jordan Lee",
                    "evidence_ids": ["ev_contact_001"],
                }
            ],
            "personalization": [
                {
                    "contact_id": "contact_001",
                    "summary": "Promotes memberships",
                    "evidence_ids": ["ev_signal_001"],
                }
            ],
        }
    )

    assert payload["directive"]["geographic_area"] == "Austin"
    assert payload["companies"][0]["fit_score"] == 0.0
    assert payload["contacts"][0]["contact_id"] == "contact_001"
    assert payload["personalization_signals"][0]["signal"] == "Promotes memberships"


def test_mapping_records_must_carry_evidence_ids() -> None:
    with pytest.raises(ValueError, match="company company_001 requires evidence_ids"):
        build_tiered_artifact_payload(
            {
                "directive": {
                    "industry": "med spa",
                    "geography": "Austin",
                    "target_count": 1,
                    "criteria": "membership offers",
                },
                "companies": [{"company_id": "company_001", "name": "Glow Co"}],
            }
        )


def test_markdown_report_escapes_html_like_input(tmp_path) -> None:
    run = TieredResearchRun(
        run_id="escape-test",
        directive=SearchDirective(
            industry="<script>",
            geographic_area="Dallas",
            target_prospect_count=1,
            research_criteria="A&B",
        ),
        companies=(
            CompanyProspect(
                company_id="company_001",
                name="<b>Bad Co</b>",
                fit_rationale="Uses A&B | C",
                evidence_ids=("ev_company_001",),
            ),
        ),
    )

    result = write_tiered_artifacts(run, tmp_path)

    markdown = result.markdown_path.read_text()
    assert "&lt;script&gt;" in markdown
    assert "&lt;b&gt;Bad Co&lt;/b&gt;" in markdown
    assert "A&amp;B \\| C" in markdown


def test_tiered_run_rejects_cross_company_or_contact_references() -> None:
    with pytest.raises(ValueError, match="references unknown company_id"):
        TieredResearchRun(
            run_id="run_001",
            directive=SearchDirective(
                industry="legal",
                geographic_area="Houston",
                target_prospect_count=1,
                research_criteria="small firms",
            ),
            companies=(
                CompanyProspect(
                    company_id="company_001",
                    name="Firm A",
                    evidence_ids=("ev_company_001",),
                ),
            ),
            contacts=(
                ContactCandidate(
                    contact_id="contact_001",
                    company_id="company_missing",
                    name="Alex Smith",
                    evidence_ids=("ev_contact_001",),
                ),
            ),
        )

    with pytest.raises(ValueError, match="personalization references unknown contact_id"):
        TieredResearchRun(
            run_id="run_001",
            directive=SearchDirective(
                industry="legal",
                geographic_area="Houston",
                target_prospect_count=1,
                research_criteria="small firms",
            ),
            companies=(
                CompanyProspect(
                    company_id="company_001",
                    name="Firm A",
                    evidence_ids=("ev_company_001",),
                ),
            ),
            personalizations=(
                ContactPersonalization(
                    contact_id="contact_missing",
                    personalization_signals=(
                        PersonalizationSignal(
                            contact_id="contact_missing",
                            signal="Promotes litigation support",
                            message_angle="Ask about intake follow-up",
                            evidence_ids=("ev_signal_001",),
                        ),
                    ),
                ),
            ),
        )


def test_tiered_dataclasses_validate_required_fields() -> None:
    with pytest.raises(ValueError, match="target_prospect_count must be greater than 0"):
        SearchDirective(
            industry="dental",
            geographic_area="Dallas",
            target_prospect_count=0,
            research_criteria="multi-location",
        )

    with pytest.raises(ValueError, match="CompanyProspect.name is required"):
        CompanyProspect(company_id="company_001", name="", evidence_ids=("ev_company_001",))
