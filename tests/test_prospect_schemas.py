from __future__ import annotations

import pytest

from deep_research_agent.prospects import (
    CitationValidationError,
    EvidenceReference,
    ProspectCitation,
    ProspectRecord,
    citation_schema,
    evidence_reference_schema,
    normalize_evidence_catalog,
    prospect_schema,
    validate_citation,
    validate_prospect,
    validate_prospects,
)


def test_prospect_and_citation_schemas_document_supported_semantics() -> None:
    citation = citation_schema()
    prospect = prospect_schema()
    evidence = evidence_reference_schema()

    assert citation["properties"]["purpose"]["enum"] == ["discovery", "fact"]
    assert prospect["required"] == ["organization", "confidence", "citations"]
    assert prospect["properties"]["citations"]["items"] == citation
    assert "decision_maker_leads" in prospect["properties"]
    assert "fit_rationale" in prospect["properties"]
    assert "personalized_angles" in prospect["properties"]
    assert evidence["properties"]["evidence_type"]["enum"] == ["page_read", "snippet"]


def test_page_read_fact_citation_requires_quote_present_in_evidence() -> None:
    evidence = {
        "acme-about": EvidenceReference(
            evidence_id="acme-about",
            url="https://acme.example/about",
            evidence_type="page_read",
            content="Acme builds autonomous research software for revenue teams.",
        )
    }

    citation = validate_citation(
        ProspectCitation(
            evidence_id="acme-about",
            claim="Acme serves revenue teams.",
            purpose="fact",
            quote="research software for revenue teams",
            field="summary",
        ),
        evidence,
    )

    assert citation.evidence_id == "acme-about"


@pytest.mark.parametrize(
    ("citation", "match"),
    [
        (
            ProspectCitation(evidence_id="result-1", claim="Acme serves revenue teams."),
            "snippet evidence can only support discovery",
        ),
        (
            ProspectCitation(evidence_id="page-1", claim="Acme serves revenue teams."),
            "page_read fact citations require a quote",
        ),
        (
            ProspectCitation(
                evidence_id="page-1",
                claim="Acme serves revenue teams.",
                quote="not in the page",
            ),
            "quote is not present",
        ),
        (
            ProspectCitation(evidence_id="missing", claim="Acme serves revenue teams."),
            "unknown evidence_id",
        ),
    ],
)
def test_invalid_citations_report_actionable_validation_errors(
    citation: ProspectCitation, match: str
) -> None:
    evidence = {
        "result-1": EvidenceReference(
            evidence_id="result-1",
            url="https://search.example/result",
            evidence_type="snippet",
            content="Search result says Acme has a revenue team product.",
        ),
        "page-1": EvidenceReference(
            evidence_id="page-1",
            url="https://acme.example/about",
            evidence_type="page_read",
            content="Acme builds autonomous research software for revenue teams.",
        ),
    }

    with pytest.raises(CitationValidationError, match=match):
        validate_citation(citation, evidence)


def test_snippet_discovery_citation_is_allowed_but_not_factual() -> None:
    evidence = {
        "result-1": EvidenceReference(
            evidence_id="result-1",
            url="https://search.example/result",
            evidence_type="snippet",
            content="Search result says Acme has a revenue team product.",
        )
    }

    citation = validate_citation(
        {"evidence_id": "result-1", "claim": "Candidate source discovered", "purpose": "discovery"},
        evidence,
    )

    assert citation.purpose == "discovery"


def test_validate_prospect_accepts_mapping_shapes_and_normalized_evidence_aliases() -> None:
    evidence = normalize_evidence_catalog(
        [
            {
                "id": "page-1",
                "source_url": "https://acme.example/about",
                "source_type": "page_read",
                "text": "Acme builds autonomous research software for revenue teams.",
            }
        ]
    )
    prospect = validate_prospect(
        {
            "company_name": "Acme",
            "website": "https://acme.example",
            "summary": "Autonomous research software for revenue teams.",
            "confidence": 0.82,
            "decision_maker_leads": ["VP Marketing"],
            "fit_rationale": "Revenue-team fit",
            "personalized_angles": ["Autonomous research"],
            "citations": [
                {
                    "evidence_id": "page-1",
                    "claim": "Acme builds research software for revenue teams.",
                    "purpose": "fact",
                    "quote": "research software for revenue teams",
                    "field": "summary",
                }
            ],
        },
        evidence,
    )

    assert prospect.to_dict()["organization"] == "Acme"
    assert prospect.citations[0].field == "summary"
    assert prospect.decision_maker_leads == ("VP Marketing",)
    assert prospect.fit_rationale == "Revenue-team fit"
    assert prospect.personalized_angles == ("Autonomous research",)


def test_validate_prospects_rejects_uncited_or_out_of_range_records() -> None:
    evidence = [
        EvidenceReference(
            evidence_id="page-1",
            url="https://acme.example/about",
            evidence_type="page_read",
            content="Acme builds autonomous research software for revenue teams.",
        )
    ]

    with pytest.raises(CitationValidationError, match="requires at least one citation"):
        validate_prospect(ProspectRecord(organization="Acme", confidence=0.5), evidence)

    with pytest.raises(CitationValidationError, match="confidence must be between"):
        validate_prospects(
            [
                {
                    "organization": "Acme",
                    "confidence": 1.5,
                    "citations": [
                        {
                            "evidence_id": "page-1",
                            "claim": "Acme builds research software.",
                            "purpose": "fact",
                            "quote": "research software",
                        }
                    ],
                }
            ],
            evidence,
        )
