from __future__ import annotations

import csv
import json

import pytest

from deep_research_agent.tiered_artifacts import (
    BrowserCapture,
    CompanyProspect,
    ContactCandidate,
    PersonalizationSignal,
    SearchDirective,
    TIERED_ARTIFACT_SCHEMA_VERSION,
    TieredResearchRun,
    build_tiered_artifact_payload,
    write_tiered_artifacts,
)


def _sample_run() -> TieredResearchRun:
    return TieredResearchRun(
        directive=SearchDirective(
            industry="orthodontics",
            geography="North Texas",
            target_count=2,
            criteria=("multi-location", "offers Invisalign"),
            raw_query="orthodontists north texas",
        ),
        companies=(
            CompanyProspect(
                company_id="company_001",
                name="Acme Ortho",
                domain="acmeortho.example",
                website="https://acmeortho.example",
                industry="orthodontics",
                geography="Dallas-Fort Worth",
                fit_score=0.91,
                confidence=0.87,
                summary="Multi-location practice | public Invisalign page",
                evidence_ids=("ev_company_001",),
                metadata={"locations": 3},
            ),
        ),
        contacts=(
            ContactCandidate(
                contact_id="contact_001",
                company_id="company_001",
                name="Dr. Ada Lovelace",
                role="owner",
                title="DDS",
                profile_url="https://acmeortho.example/team/ada",
                confidence=0.82,
                evidence_ids=("ev_contact_001",),
            ),
        ),
        personalization_signals=(
            PersonalizationSignal(
                signal_id="signal_001",
                company_id="company_001",
                contact_id="contact_001",
                signal_type="service_line",
                summary="Mentions teen Invisalign expansion",
                suggested_outreach_angle="Lead with teen Invisalign patient reactivation.",
                evidence_ids=("ev_signal_001",),
                confidence=0.79,
            ),
            PersonalizationSignal(
                signal_id="signal_002",
                company_id="company_001",
                signal_type="company_news",
                summary="Opened a second location",
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
        metadata={"thread_id": "demo-thread"},
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
    assert payload["metadata"] == {"story": "G001", "thread_id": "demo-thread"}
    assert payload["directive"]["industry"] == "orthodontics"
    assert payload["companies"][0]["company_id"] == "company_001"
    assert payload["contacts"][0]["company_id"] == "company_001"
    assert payload["personalization_signals"][0]["contact_id"] == "contact_001"
    assert payload["browser_captures"][0]["screenshot_path"] == "captures/cap_001.png"

    with result.companies_csv_path.open(newline="") as handle:
        company_rows = list(csv.DictReader(handle))
    assert company_rows[0]["evidence_ids"] == '["ev_company_001"]'
    assert company_rows[0]["metadata_json"] == '{"locations": 3}'

    with result.contacts_csv_path.open(newline="") as handle:
        contact_rows = list(csv.DictReader(handle))
    assert contact_rows[0]["name"] == "Dr. Ada Lovelace"

    with result.personalization_csv_path.open(newline="") as handle:
        signal_rows = list(csv.DictReader(handle))
    assert [row["signal_id"] for row in signal_rows] == ["signal_001", "signal_002"]

    markdown = result.markdown_path.read_text()
    assert "# Tiered Prospect Research Report" in markdown
    assert "## Directive" in markdown
    assert "### Company: Acme Ortho" in markdown
    assert "Multi-location practice \\| public Invisalign page" in markdown
    assert "##### Contact: Dr. Ada Lovelace, owner" in markdown
    assert "Lead with teen Invisalign patient reactivation." in markdown
    assert "## Warnings and Human Review Items" in markdown


def test_build_tiered_payload_accepts_mapping_records() -> None:
    payload = build_tiered_artifact_payload(
        {
            "directive": {
                "industry": "med spa",
                "geography": "Austin",
                "target_count": 1,
            },
            "companies": [{"company_id": "company_001", "name": "Glow Co"}],
            "contacts": [
                {
                    "contact_id": "contact_001",
                    "company_id": "company_001",
                    "name": "Jordan Lee",
                }
            ],
            "personalization": [
                {
                    "signal_id": "signal_001",
                    "company_id": "company_001",
                    "summary": "Promotes memberships",
                }
            ],
        }
    )

    assert payload["directive"]["geography"] == "Austin"
    assert payload["companies"][0]["qualification_status"] == "qualified"
    assert payload["contacts"][0]["contact_id"] == "contact_001"
    assert payload["personalization_signals"][0]["signal_type"] == "general"


def test_tiered_run_rejects_cross_company_or_contact_references() -> None:
    with pytest.raises(ValueError, match="ContactCandidate.company_id is unknown"):
        TieredResearchRun(
            directive=SearchDirective(industry="legal", geography="Houston", target_count=1),
            companies=(CompanyProspect(company_id="company_001", name="Firm A"),),
            contacts=(
                ContactCandidate(
                    contact_id="contact_001",
                    company_id="company_missing",
                    name="Alex Smith",
                ),
            ),
        )

    with pytest.raises(ValueError, match="PersonalizationSignal.contact_id is unknown"):
        TieredResearchRun(
            directive=SearchDirective(industry="legal", geography="Houston", target_count=1),
            companies=(CompanyProspect(company_id="company_001", name="Firm A"),),
            personalization_signals=(
                PersonalizationSignal(
                    signal_id="signal_001",
                    company_id="company_001",
                    contact_id="contact_missing",
                ),
            ),
        )


def test_tiered_dataclasses_validate_required_fields() -> None:
    with pytest.raises(ValueError, match="SearchDirective.target_count must be >= 1"):
        SearchDirective(industry="dental", geography="Dallas", target_count=0)

    with pytest.raises(ValueError, match="CompanyProspect.name is required"):
        CompanyProspect(company_id="company_001", name="")
