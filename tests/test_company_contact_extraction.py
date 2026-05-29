from deep_research_agent.company_contact_extraction import (
    CompanyContactSource,
    extract_company_contact_points,
)
from deep_research_agent.tiered_models import CompanyProspect


def _company() -> CompanyProspect:
    return CompanyProspect(
        company_id="company_example",
        name="Example Electric",
        website="https://example-electric.com",
        evidence_ids=("ev_company",),
        source_confidence="high",
    )


def test_extracts_email_and_phone_before_contact_page() -> None:
    contacts = extract_company_contact_points(
        _company(),
        (
            CompanyContactSource(
                url="https://example-electric.com/contact",
                title="Contact Example Electric",
                content="""
                <a href="mailto:Sales@Example-Electric.com">Email sales</a>
                <a href="tel:+12145550100">Call</a>
                <a href="/contact">Contact us</a>
                """,
                evidence_id="ev_contact_page",
                source_confidence="high",
            ),
        ),
    )

    assert [contact.contact_kind for contact in contacts] == ["company_email", "company_phone"]
    assert contacts[0].email == "sales@example-electric.com"
    assert contacts[0].source_url == "https://example-electric.com/contact"
    assert contacts[1].phone == "+12145550100"
    assert all(contact.title == "" for contact in contacts)


def test_emits_contact_page_when_no_direct_channel_exists() -> None:
    contacts = extract_company_contact_points(
        _company(),
        (
            CompanyContactSource(
                url="https://example-electric.com",
                content='<a href="/schedule-service">Schedule Service</a>',
                evidence_id="ev_home",
            ),
        ),
    )

    assert len(contacts) == 1
    assert contacts[0].contact_kind == "company_contact_page"
    assert contacts[0].contact_url == "https://example-electric.com/schedule-service"


def test_schedule_service_source_page_is_actionable_contact_page() -> None:
    contacts = extract_company_contact_points(
        _company(),
        (
            CompanyContactSource(
                url="https://example-electric.com/schedule-service",
                content="Schedule service online.",
                evidence_id="ev_schedule",
            ),
        ),
    )

    assert len(contacts) == 1
    assert contacts[0].contact_kind == "company_contact_page"
    assert contacts[0].contact_url == "https://example-electric.com/schedule-service"


def test_generic_pages_are_not_contact_points_without_actionable_cta() -> None:
    contacts = extract_company_contact_points(
        _company(),
        (
            CompanyContactSource(
                url="https://example-electric.com/about-us",
                title="About Us",
                content="Areas We Serve and Home Automation services.",
                evidence_id="ev_about",
            ),
        ),
    )

    assert contacts == ()


def test_generic_pages_with_contact_text_are_not_contact_page_rows() -> None:
    contacts = extract_company_contact_points(
        _company(),
        (
            CompanyContactSource(
                url="https://example-electric.com/services",
                title="Services",
                content="Schedule service today or contact our home automation team.",
                evidence_id="ev_services",
            ),
            CompanyContactSource(
                url="https://example-electric.com/about-us",
                title="About Us",
                content="Contact our friendly team for areas we serve.",
                evidence_id="ev_about",
            ),
        ),
    )

    assert contacts == ()


def test_generic_links_with_contact_text_are_not_contact_page_rows() -> None:
    contacts = extract_company_contact_points(
        _company(),
        (
            CompanyContactSource(
                url="https://example-electric.com",
                content="""
                <a href="/about-us">Contact our team</a>
                <a href="/services">Schedule service</a>
                """,
                evidence_id="ev_home",
            ),
        ),
    )

    assert contacts == ()


def test_actionable_link_path_still_becomes_contact_page() -> None:
    contacts = extract_company_contact_points(
        _company(),
        (
            CompanyContactSource(
                url="https://example-electric.com",
                content='<a href="/schedule-service">Schedule service</a>',
                evidence_id="ev_home",
            ),
        ),
    )

    assert len(contacts) == 1
    assert contacts[0].contact_kind == "company_contact_page"
    assert contacts[0].contact_url == "https://example-electric.com/schedule-service"


def test_dedupes_visible_and_link_phone_formats() -> None:
    contacts = extract_company_contact_points(
        _company(),
        (
            CompanyContactSource(
                url="https://example-electric.com/contact",
                content="""
                Call (214) 555-0100 today.
                <a href="tel:214-555-0100">Call again</a>
                <a href="mailto:info@example-electric.com">Email</a>
                """,
                evidence_id="ev_contact",
            ),
        ),
    )

    phones = [contact.phone for contact in contacts if contact.phone]
    assert phones == ["+12145550100"]


def test_channel_diversity_keeps_phone_when_multiple_emails_compete_for_cap() -> None:
    contacts = extract_company_contact_points(
        _company(),
        (
            CompanyContactSource(
                url="https://example-electric.com/contact",
                content="""
                <a href="mailto:info@example-electric.com">Email info</a>
                <a href="mailto:sales@example-electric.com">Email sales</a>
                <a href="tel:214-555-0199">Call</a>
                """,
                evidence_id="ev_contact",
            ),
        ),
    )

    assert [contact.contact_kind for contact in contacts] == ["company_email", "company_phone"]
    assert contacts[0].email == "info@example-electric.com"
    assert contacts[1].phone == "+12145550199"


def test_service_slugs_containing_contact_substrings_are_not_contact_pages() -> None:
    contacts = extract_company_contact_points(
        _company(),
        (
            CompanyContactSource(
                url="https://example-electric.com",
                content="""
                <a href="/contact-lenses">Contact lenses</a>
                <a href="/contactless-payments">Contactless payments</a>
                <a href="/services/contactors">Electrical contactors</a>
                """,
                evidence_id="ev_home",
            ),
        ),
    )

    assert contacts == ()
