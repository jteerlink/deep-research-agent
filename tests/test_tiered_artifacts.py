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
    FinalEnrichmentRecord,
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
                name="Company email",
                contact_kind="company_email",
                label="Company email",
                email="hello@acmeortho.example",
                url="mailto:hello@acmeortho.example",
                contact_url="mailto:hello@acmeortho.example",
                source_url="https://acmeortho.example/contact",
                source_confidence="high",
                contact_confidence=0.9,
                evidence_ids=("ev_contact_001",),
            ),
        ),
        browser_captures=(
            BrowserCapture(
                browser_capture_id="cap_001",
                url="https://acmeortho.example/contact",
                company_id="company_001",
                contact_id="contact_001",
                captured_at="2026-05-26T20:00:00Z",
                text_excerpt="Email hello@acmeortho.example",
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
    assert result.personalization_count == 0
    assert result.json_path.name == "tiered_prospect_research.json"
    assert result.companies_csv_path.name == "companies.csv"
    assert result.contacts_csv_path.name == "contacts.csv"
    assert result.personalization_csv_path.name == "personalization.csv"
    assert result.markdown_path.name == "research_report.md"

    payload = json.loads(result.json_path.read_text())
    assert payload["schema_version"] == TIERED_ARTIFACT_SCHEMA_VERSION
    assert payload["metadata"] == {
        "story": "G001",
        "contact_record_semantics": "company_contact_point.v1",
        "legacy_contact_count_aliases": [
            "ready_contact_count",
            "contact_candidate_count",
            "companies_with_contacts_count",
        ],
        "ready_company_count": 1,
        "company_prospect_count": 1,
        "ready_contact_count": 1,
        "contact_candidate_count": 1,
        "contact_point_count": 1,
        "qualified_company_count": 1,
        "companies_with_contacts_count": 1,
        "companies_with_contact_points_count": 1,
        "needs_contact_count": 0,
        "rejected_candidate_count": 0,
    }
    assert payload["run_id"] == "demo-thread"
    assert payload["directive"]["industry"] == "orthodontics"
    assert payload["companies"][0]["company_id"] == "company_001"
    assert payload["contacts"][0]["company_id"] == "company_001"
    assert payload["personalization_signals"] == []
    assert payload["browser_captures"][0]["screenshot_path"] == "captures/cap_001.png"
    assert payload["qualification_audit"] == []

    with result.companies_csv_path.open(newline="") as handle:
        company_rows = list(csv.DictReader(handle))
    assert company_rows[0]["evidence_ids"] == '["ev_company_001"]'
    assert company_rows[0]["locations"] == '["Dallas", "Fort Worth"]'

    with result.contacts_csv_path.open(newline="") as handle:
        contact_rows = list(csv.DictReader(handle))
    assert contact_rows[0]["name"] == "Company email"
    assert contact_rows[0]["contact_kind"] == "company_email"
    assert contact_rows[0]["email"] == "hello@acmeortho.example"

    with result.personalization_csv_path.open(newline="") as handle:
        signal_rows = list(csv.DictReader(handle))
    assert signal_rows == []

    markdown = result.markdown_path.read_text()
    assert "# Tiered Prospect Research Report" in markdown
    assert "## Directive" in markdown
    assert "### Company: Acme Ortho" in markdown
    assert "Multi-location practice \\| public Invisalign page" in markdown
    assert "##### Contact info: Company email" in markdown
    assert "hello@acmeortho.example" in markdown
    assert "Company prospects: 1" in markdown
    assert "Company contact points: 1" in markdown
    assert "Rejected/noisy candidates: 0" in markdown
    assert "## Warnings and Human Review Items" in markdown


def test_canonical_contact_metadata_cannot_be_overridden() -> None:
    payload = build_tiered_artifact_payload(
        _sample_run(),
        metadata={
            "contact_record_semantics": "legacy_person_contact",
            "legacy_contact_count_aliases": ["wrong"],
        },
    )

    assert payload["metadata"]["contact_record_semantics"] == "company_contact_point.v1"
    assert payload["metadata"]["legacy_contact_count_aliases"] == [
        "ready_contact_count",
        "contact_candidate_count",
        "companies_with_contacts_count",
    ]


def test_company_contact_artifacts_reject_person_contacts() -> None:
    run = TieredResearchRun(
        run_id="person-contact",
        directive=SearchDirective(
            industry="legal",
            geographic_area="Dallas",
            target_prospect_count=1,
            research_criteria="small firms",
        ),
        companies=(
            CompanyProspect(
                company_id="company_001",
                name="Firm A",
                evidence_ids=("ev_company",),
            ),
        ),
        contacts=(
            ContactCandidate(
                contact_id="contact_001",
                company_id="company_001",
                name="Alex Smith",
                evidence_ids=("ev_contact",),
            ),
        ),
    )

    with pytest.raises(ValueError, match="does not accept person contact rows"):
        build_tiered_artifact_payload(run)


def test_final_enrichment_csv_hydrates_contact_snapshot_for_direct_writer(
    tmp_path,
) -> None:
    company = CompanyProspect(
        company_id="company_001",
        name="Acme Plumbing",
        website="https://acme.example",
        evidence_ids=("ev_company",),
    )
    contact = ContactCandidate(
        contact_id="contact_001",
        company_id=company.company_id,
        name="Company email",
        contact_kind="company_email",
        label="Company email",
        email="hello@acme.example",
        url="mailto:hello@acme.example",
        contact_url="mailto:hello@acme.example",
        source_url="https://acme.example/contact",
        evidence_ids=("ev_contact",),
    )
    run = TieredResearchRun(
        run_id="direct-writer",
        directive=SearchDirective(
            industry="plumbing",
            geographic_area="Dallas",
            target_prospect_count=1,
            research_criteria="emergency service",
        ),
        companies=(company,),
        contacts=(contact,),
    )

    result = write_tiered_artifacts(
        run,
        tmp_path,
        final_enrichment_records=(
            FinalEnrichmentRecord(
                enrichment_id="final_001",
                company_id=company.company_id,
                contact_id=contact.contact_id,
                summary="Approved enrichment.",
                evidence_ids=("ev_final",),
                provider="exa",
            ),
        ),
    )

    payload = json.loads(result.json_path.read_text())
    snapshot = payload["final_enrichment"][0]["contact_snapshot"]
    assert snapshot["contact_kind"] == "company_email"
    assert snapshot["email"] == "hello@acme.example"

    assert result.final_enrichment_csv_path is not None
    with result.final_enrichment_csv_path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["contact_kind"] == "company_email"
    assert row["email"] == "hello@acme.example"
    assert row["source_url"] == "https://acme.example/contact"


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
    assert "## Qualified companies needing contact info" in markdown
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
                    "name": "Company email",
                    "contact_kind": "company_email",
                    "label": "Company email",
                    "email": "hello@glow.example",
                    "url": "mailto:hello@glow.example",
                    "contact_url": "mailto:hello@glow.example",
                    "source_url": "https://glow.example/contact",
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



def test_mapping_contact_records_must_declare_contact_kind() -> None:
    with pytest.raises(ValueError, match="legacy contact payload missing contact_kind"):
        build_tiered_artifact_payload(
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
                        "contact_id": "contact_legacy_001",
                        "company_id": "company_001",
                        "name": "Legacy Person",
                        "title": "Owner",
                        "evidence_ids": ["ev_contact_001"],
                    }
                ],
            }
        )


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
