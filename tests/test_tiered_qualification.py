from __future__ import annotations

from deep_research_agent.tiered_models import SearchDirective
from deep_research_agent.tiered_qualification import (
    QualifiedCompanyCandidate,
    qualify_company_hits,
    qualify_contact_hits,
)
from deep_research_agent.tiered_search import TieredSearchHit


def _directive() -> SearchDirective:
    return SearchDirective(
        industry="HVAC",
        geographic_area="Dallas-Fort Worth",
        target_prospect_count=3,
        research_criteria="owner-operated residential HVAC companies",
        preferred_contact_roles=("owner", "general manager"),
    )


def _directive_with_negative_criteria(value: str) -> SearchDirective:
    base = _directive()
    return SearchDirective(
        industry=base.industry,
        geographic_area=base.geographic_area,
        target_prospect_count=base.target_prospect_count,
        research_criteria=base.research_criteria,
        negative_criteria=value,
        preferred_contact_roles=base.preferred_contact_roles,
    )


def _hit(title: str, url: str, content: str = "snippet", provider: str = "duckduckgo"):
    return TieredSearchHit(
        tier="company_discovery",
        query="HVAC companies in Dallas-Fort Worth",
        title=title,
        url=url,
        content=content,
        provider=provider,
        rank=1,
    )


def test_company_qualification_rejects_listicles_and_marketing_pages() -> None:
    batch = qualify_company_hits(
        (
            _hit(
                "Best 13 Hvac Systems in Fort Worth TX",
                "https://www.contractorsup.com/fort-worth-tx/hvac",
                "Directory list of HVAC contractors.",
            ),
            _hit(
                "7 Best HVAC Companies in Fort Worth, TX of 2026",
                "https://www.consumeraffairs.com/homeowners/fort-worth-hvac.html",
                "Reviewed HVAC companies and customer reviews.",
            ),
            _hit(
                "Local SEO for HVAC Companies in Dallas, TX",
                "https://www.vaza.ai/blog/local-seo-hvac-dallas",
                "Marketing agency page for HVAC lead generation.",
            ),
        ),
        _directive(),
    )

    assert batch.accepted == ()
    reasons = {reason for record in batch.rejected for reason in record.reasons}
    assert "listicle_or_directory" in reasons
    assert "non_target_marketing_page" in reasons
    assert {record.status for record in batch.audit_records} == {"rejected"}


def test_company_qualification_rejects_job_boards_articles_and_reversed_listicles() -> None:
    batch = qualify_company_hits(
        (
            _hit(
                "Electrician jobs in Dallas-Fort Worth, TX - Indeed",
                "https://www.indeed.com/q-electrician-l-dallas-fort-worth,-tx-jobs.html",
                "610 Electrician jobs available. Apply to Journeyperson Electrician.",
            ),
            _hit(
                "14 Best Fort Worth, TX Electricians | Expertise.com",
                "https://www.expertise.com/home-improvement/electricians/texas/fort-worth",
                "Reviewed electrician companies and customer reviews.",
            ),
            _hit(
                "How to Choose a Reliable Certified Electrician in Dallas-Fort Worth",
                "https://callw3.com/blog/how-to-choose-a-reliable-certified-electrician-in-dallas-fort-worth",
                "Guide to choosing a certified electrical contractor.",
            ),
            _hit(
                "Commercial Electrical Contractor Dallas, TX",
                "https://fsg.com/locations/dallas",
                "FSG Dallas is a full-service commercial electrical contractor.",
            ),
            _hit(
                "Local Electrician in Dallas, TX | Mr. Electric",
                "https://www.mrelectricdallas.com",
                "Certified electricians in Dallas.",
            ),
        ),
        SearchDirective(
            industry="electrician",
            geographic_area="Dallas-Fort Worth TX",
            target_prospect_count=3,
            research_criteria="has company website",
        ),
    )

    assert [company.name for company in batch.accepted] == ["FSG", "Mr. Electric"]
    rejected_reasons = {
        reason for record in batch.rejected for reason in record.reasons
    }
    assert "job_board_or_career_page" in rejected_reasons
    assert "listicle_or_directory" in rejected_reasons
    assert "article_or_blog_source" in rejected_reasons
    assert "not_specific_company" in rejected_reasons


def test_company_qualification_accepts_official_business_domain() -> None:
    batch = qualify_company_hits(
        (
            _hit(
                "Houk Air Conditioning DFW | HVAC Repair",
                "https://www.houkac.com/air-conditioning",
                "Houk Air Conditioning serves homes across Dallas-Fort Worth.",
            ),
        ),
        _directive(),
        target_count=1,
    )

    assert [company.name for company in batch.accepted] == ["Houk Air Conditioning"]
    company = batch.accepted[0]
    assert company.website == "https://www.houkac.com/air-conditioning"
    assert company.score > 0.7
    assert "official_domain" in company.reasons
    assert batch.audit_records[0].status == "accepted"
    assert batch.audit_records[0].evidence_id == company.evidence_id


def test_company_qualification_rejects_negative_criteria_matches() -> None:
    batch = qualify_company_hits(
        (
            _hit(
                "Franchise HVAC DFW | HVAC Repair",
                "https://franchise-hvac.example",
                "Franchise HVAC is a national franchise serving Dallas.",
            ),
        ),
        _directive_with_negative_criteria("franchise"),
        target_count=1,
    )

    assert batch.accepted == ()
    assert batch.rejected[0].status == "rejected"
    assert "negative_criteria_match:franchise" in batch.rejected[0].reasons


def test_contact_qualification_rejects_article_titles() -> None:
    company = _qualified_houk_company()
    batch = qualify_contact_hits(
        (
            TieredSearchHit(
                tier="contact_discovery",
                query="Houk Air Conditioning owner",
                title="7 Best HVAC Companies in Fort Worth, TX of 2026",
                url="https://www.consumeraffairs.com/homeowners/fort-worth-hvac.html",
                content="Reviewed HVAC companies and customer reviews.",
                provider="duckduckgo",
            ),
        ),
        company,
        _directive(),
    )

    assert batch.accepted == ()
    assert batch.rejected[0].status == "rejected"
    assert "article_or_list_title" in batch.rejected[0].reasons
    assert batch.needs_contact[0].status == "needs_contact"
    assert "qualified_company_needs_contact" in batch.needs_contact[0].reasons


def test_contact_qualification_accepts_person_role_company_evidence() -> None:
    company = _qualified_houk_company()
    batch = qualify_contact_hits(
        (
            TieredSearchHit(
                tier="contact_discovery",
                query="Houk Air Conditioning owner",
                title="Jane Smith - Owner at Houk Air Conditioning",
                url="https://www.linkedin.com/in/jane-smith-houk-air-conditioning",
                content="Jane Smith is Owner at Houk Air Conditioning.",
                provider="duckduckgo",
            ),
        ),
        company,
        _directive(),
    )

    assert [contact.name for contact in batch.accepted] == ["Jane Smith"]
    contact = batch.accepted[0]
    assert contact.title == "Owner"
    assert contact.role_category == "owner"
    assert contact.score > 0.7
    assert "company_tied_evidence" in contact.reasons
    assert "preferred_role" in contact.reasons
    assert not batch.needs_contact


def test_contact_qualification_rejects_negative_criteria_matches() -> None:
    company = _qualified_houk_company()
    batch = qualify_contact_hits(
        (
            TieredSearchHit(
                tier="contact_discovery",
                query="Houk Air Conditioning owner",
                title="Jane Smith - Owner at Houk Air Conditioning",
                url="https://www.linkedin.com/in/jane-smith-houk-air-conditioning",
                content="Jane Smith is an outside consultant for Houk Air Conditioning.",
                provider="duckduckgo",
            ),
        ),
        company,
        _directive_with_negative_criteria("consultant"),
    )

    assert batch.accepted == ()
    assert "negative_criteria_match:consultant" in batch.rejected[0].reasons
    assert batch.needs_contact[0].status == "needs_contact"


def test_qualification_audit_records_preserve_reasons() -> None:
    company_batch = qualify_company_hits(
        (
            _hit(
                "Local SEO for HVAC Companies in Dallas, TX",
                "https://www.vaza.ai/blog/local-seo-hvac-dallas",
                "Marketing agency page for HVAC lead generation.",
            ),
            _hit(
                "Houk Air Conditioning DFW | HVAC Repair",
                "https://www.houkac.com/air-conditioning",
                "Houk Air Conditioning serves Dallas-Fort Worth.",
            ),
        ),
        _directive(),
        target_count=1,
    )
    contact_batch = qualify_contact_hits((), company_batch.accepted[0], _directive())

    audit_payloads = [record.to_dict() for record in company_batch.audit_records]
    audit_payloads.extend(record.to_dict() for record in contact_batch.audit_records)

    assert {payload["status"] for payload in audit_payloads} == {
        "accepted",
        "rejected",
        "needs_contact",
    }
    for payload in audit_payloads:
        assert payload["tier"] in {"company_discovery", "contact_discovery"}
        assert "status" in payload
        assert "title" in payload
        assert "url" in payload
        assert "provider" in payload
        assert "query" in payload
        assert payload["reasons"]


def _qualified_houk_company() -> QualifiedCompanyCandidate:
    batch = qualify_company_hits(
        (
            _hit(
                "Houk Air Conditioning DFW | HVAC Repair",
                "https://www.houkac.com/air-conditioning",
                "Houk Air Conditioning serves Dallas-Fort Worth.",
            ),
        ),
        _directive(),
        target_count=1,
    )
    return batch.accepted[0]
