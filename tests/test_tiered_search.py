from __future__ import annotations

import asyncio
import builtins

import pytest

from deep_research_agent.tiered_search import (
    CompanySearchTarget,
    ContactSearchTarget,
    ProviderPolicy,
    TieredSearchDirective,
    build_company_discovery_lanes,
    build_company_discovery_queries,
    build_contact_discovery_queries,
    build_personalization_queries,
    collect_company_discovery_search,
    collect_tiered_search,
)


def test_provider_policy_keeps_exa_final_enrichment_only(monkeypatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "configured")

    policy = ProviderPolicy()

    assert "exa" not in policy.early_discovery
    assert policy.final_enrichment == ("exa",)
    with pytest.raises(ValueError, match="reserved for final enrichment"):
        ProviderPolicy(early_discovery=("tavily", "exa"))


def test_company_discovery_queries_expand_directive_and_dedupe() -> None:
    directive = TieredSearchDirective(
        industry="dental",
        geographic_area="DFW area",
        target_prospect_count=25,
        research_criteria="multi-location practices",
        source_preferences=("trade associations", "trade associations"),
    )

    queries = build_company_discovery_queries(directive)

    assert queries == (
        "dental companies in DFW area",
        "best dental DFW area",
        "dental owner DFW area",
        "dental marketing DFW area",
        "dental DFW area multi-location practices",
        "dental trade associations DFW area",
    )


def test_company_discovery_lanes_are_bounded_and_family_scoped() -> None:
    directive = TieredSearchDirective(
        industry="dental",
        geographic_area="DFW area",
        research_criteria="multi-location practices",
    )

    lanes = build_company_discovery_lanes(directive)

    assert [lane.family for lane in lanes] == [
        "official_site",
        "local_directory",
        "industry_context",
    ]
    assert len(lanes) == 3
    assert build_company_discovery_lanes(directive, max_parallel_search_lanes=6)
    with pytest.raises(ValueError, match="<= 6"):
        build_company_discovery_lanes(directive, max_parallel_search_lanes=7)


def test_contact_and_personalization_queries_are_scoped_to_targets() -> None:
    directive = TieredSearchDirective(
        industry="HVAC",
        geographic_area="North Texas",
        preferred_contact_roles=("owner", "growth lead"),
    )
    company = CompanySearchTarget("Acme Air", website="https://acme.example/team")
    contact = ContactSearchTarget("Jane Smith", "Acme Air", title="Owner")

    assert build_contact_discovery_queries(directive, company) == (
        "Acme Air owner",
        "Acme Air growth lead",
        "site:linkedin.com/in Acme Air owner OR growth lead",
        "site:acme.example team",
        "site:acme.example about",
        "site:acme.example leadership",
    )
    assert build_personalization_queries(directive, contact) == (
        "Jane Smith Acme Air Owner",
        "Jane Smith Acme Air interview",
        "Jane Smith Acme Air news",
        "Jane Smith Acme Air LinkedIn",
        "Jane Smith HVAC North Texas",
    )


def test_collect_tiered_search_normalizes_hits_and_records_failures() -> None:
    async def fake_search(query: str, max_results: int):
        assert max_results == 2
        if "bad" in query:
            raise RuntimeError("provider unavailable")
        return [
            {
                "title": "Acme",
                "url": "https://acme.example",
                "content": "snippet",
                "provider": "fake",
            }
        ]

    batch = asyncio.run(
        collect_tiered_search(
            "company_discovery",
            ["good query", "bad query"],
            max_results=2,
            search=fake_search,
        )
    )

    assert batch.ok is False
    assert [hit.to_dict() for hit in batch.hits] == [
        {
            "tier": "company_discovery",
            "query": "good query",
            "title": "Acme",
            "url": "https://acme.example",
            "content": "snippet",
            "provider": "fake",
            "rank": 1,
            "score": None,
        }
    ]
    assert batch.failures[0].query == "bad query"
    assert batch.failures[0].error_class == "RuntimeError"


def test_collect_tiered_search_rejects_final_enrichment_provider_hits() -> None:
    async def exa_search(_query: str, _max_results: int):
        return [
            {
                "title": "Exa result",
                "url": "https://example.com",
                "content": "snippet",
                "provider": "exa",
            }
        ]

    batch = asyncio.run(
        collect_tiered_search("company_discovery", ["dental DFW"], search=exa_search)
    )

    assert batch.hits == ()
    assert batch.failures[0].error_class == "ProviderPolicyError"
    assert "final enrichment" in batch.failures[0].error_message


def test_early_company_discovery_does_not_instantiate_default_search_chain(monkeypatch) -> None:
    imports: list[str] = []
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        imports.append(name)
        if name == "async_multi_search":
            raise AssertionError("default AsyncMultiProviderSearch chain should stay unimported")
        return real_import(name, globals, locals, fromlist, level)

    async def injected_search(query: str, max_results: int):
        return [
            {
                "title": f"Result for {query}",
                "url": "https://example.com",
                "content": "search snippet",
                "provider": "injected",
            }
        ][:max_results]

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    batch = asyncio.run(
        collect_company_discovery_search(
            TieredSearchDirective(industry="dental", geographic_area="DFW area"),
            max_results=1,
            search=injected_search,
        )
    )

    assert batch.hits
    assert {hit.provider for hit in batch.hits} == {"injected"}
    assert "async_multi_search" not in imports


def test_collect_tiered_search_requires_injected_search_callable() -> None:
    with pytest.raises(ValueError, match="requires an injected search callable"):
        asyncio.run(collect_tiered_search("company_discovery", ["dental DFW"]))


def test_tiered_search_directive_validates_required_fields() -> None:
    with pytest.raises(ValueError, match="industry is required"):
        TieredSearchDirective(industry=" ", geographic_area="DFW")
    with pytest.raises(ValueError, match="target_prospect_count"):
        TieredSearchDirective(industry="dental", geographic_area="DFW", target_prospect_count=0)
