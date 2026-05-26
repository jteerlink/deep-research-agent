from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from async_multi_search import SearchResult
from deep_research_agent.config import ModelProvider
from deep_research_agent.graph import (
    CHECKPOINT_SCHEMA_VERSION,
    GRAPH_TOPOLOGY,
    LocalCheckpointStore,
    build_graph,
    build_prospect_search_queries,
    inspect_thread,
    resume_thread,
    run_query,
    run_research,
)
from deep_research_agent.models import ModelRequest, ModelResponse
from deep_research_agent.prospect_judgment import (
    MIN_ACCEPTED_CONFIDENCE,
    MIN_ACCEPTED_FIT_SCORE,
    ProspectCandidate,
    build_prospect_judge_prompt,
    business_name_from_title,
    coerce_prospect_judgment,
    prospect_qualification,
    triage_candidate,
)


class FakeProspectJudge:
    async def invoke(self, request: ModelRequest) -> ModelResponse:
        if request.response_schema is None:
            return ModelResponse(
                provider=ModelProvider.OPENAI,
                model="fake-model",
                content="",
                structured={"node": request.node, "status": "metadata_only"},
            )
        payload = json.loads(request.prompt)
        candidate = payload["candidate"]
        accepted = "actualhvac.com" in candidate["root_domain"]
        structured: dict[str, Any] = {
            "accepted": accepted,
            "organization": "Actual HVAC" if accepted else candidate["organization_guess"],
            "canonical_website": candidate["canonical_website"],
            "fit_score": 0.91 if accepted else 0.1,
            "confidence": 0.86 if accepted else 0.2,
            "reject_reason": "" if accepted else "Does not match expected prospect.",
            "fit_rationale": "Owned HVAC business in the target market." if accepted else "",
            "evidence_summary": candidate["snippet"] or candidate["page_text"],
            "personalized_angles": ["HVAC service follow-up"] if accepted else [],
            "decision_maker_leads": ["Owner"] if accepted else [],
            "guardrail_flags": [],
        }
        return ModelResponse(
            provider=ModelProvider.OPENAI,
            model="fake-model",
            content=json.dumps(structured),
            structured=structured,
        )


class MetadataOnlyProspectJudge:
    async def invoke(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            provider=ModelProvider.OPENAI,
            model="metadata-only-model",
            content="",
            structured={"node": request.node, "status": "metadata_only"},
        )


def test_mocked_graph_smoke_delegates_to_researcher_and_interrupts_for_review(tmp_path) -> None:
    store = LocalCheckpointStore(tmp_path)
    state = build_graph(store).invoke(
        {"query": "map agent workflow", "thread_id": "smoke", "max_research_iterations": 2}
    )

    assert GRAPH_TOPOLOGY["main"] == ("supervisor", "review")
    assert state["thread_id"] == "smoke"
    assert state["status"] == "interrupted"
    assert state["interrupt_reason"] == "review_required"
    assert state["research_iterations"] == 2
    assert state["sufficient"] is True
    assert [event["event"] for event in state["events"]].count("delegate") == 2
    assert any(event["node"] == "researcher" for event in state["fallback_events"])

    checkpoint = json.loads((tmp_path / "smoke.json").read_text())
    assert checkpoint["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert checkpoint["thread_id"] == "smoke"


def test_thread_id_resume_approves_review_without_losing_checkpoint_history(tmp_path) -> None:
    first = run_query(
        "resume me",
        thread_id="thread-resume",
        checkpoint_dir=tmp_path,
        max_iterations=1,
    )

    assert first.state["status"] == "interrupted"
    resumed = resume_thread("thread-resume", checkpoint_dir=tmp_path, approve=True)

    assert resumed.thread_id == "thread-resume"
    assert resumed.state["status"] == "completed"
    assert resumed.state["review_status"] == "approved"
    assert resumed.state["research_iterations"] == 1
    inspected = inspect_thread("thread-resume", checkpoint_dir=tmp_path)
    assert inspected["state"]["status"] == "completed"
    assert inspected["checkpoint_path"].endswith("thread-resume.json")


def test_cli_run_resume_inspect_flow_uses_fixed_thread_id(tmp_path) -> None:
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "deep_research_agent",
            "run",
            "cli query",
            "--thread-id",
            "cli-thread",
            "--checkpoint-dir",
            str(tmp_path),
            "--max-iterations",
            "1",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    run_payload = json.loads(run.stdout)
    assert run_payload["thread_id"] == "cli-thread"
    assert run_payload["state"]["status"] == "interrupted"

    resume = subprocess.run(
        [
            sys.executable,
            "-m",
            "deep_research_agent",
            "resume",
            "cli-thread",
            "--checkpoint-dir",
            str(tmp_path),
            "--approve",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    resume_payload = json.loads(resume.stdout)
    assert resume_payload["state"]["status"] == "completed"

    inspect = subprocess.run(
        [
            sys.executable,
            "-m",
            "deep_research_agent",
            "inspect",
            "cli-thread",
            "--checkpoint-dir",
            str(tmp_path),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    inspect_payload = json.loads(inspect.stdout)
    assert inspect_payload["thread_id"] == "cli-thread"
    assert inspect_payload["state"]["review_status"] == "approved"


def test_run_research_passes_max_results_and_emits_progress_events(tmp_path) -> None:
    requested_max_results: list[int] = []
    progress_events: list[dict[str, object]] = []

    async def search(query: str, max_results: int):
        requested_max_results.append(max_results)
        return [
            SearchResult(
                "Acme",
                "https://example.com/acme",
                f"{query} search snippet",
                provider="firecrawl",
            )
        ]

    state = asyncio.run(
        run_research(
            "find prospects",
            thread_id="progress-thread",
            checkpoint_dir=tmp_path,
            search=search,
            require_review=True,
            max_results=7,
            target_prospect_count=1,
            enable_llm_judgment=False,
            progress_callback=progress_events.append,
        )
    )

    assert requested_max_results == [7]
    assert state["max_results"] == 7
    assert state["status"] == "interrupted"
    assert [event["event"] for event in progress_events] == [
        "main_started",
        "model_preflight",
        "supervisor_delegated",
        "search_started",
        "search_completed",
        "fallback_model_used",
        "researcher_iteration",
        "sufficiency_routed",
        "review_interrupt",
    ]


def test_run_research_accumulates_business_targets_before_sufficiency(tmp_path) -> None:
    search_queries: list[str] = []

    async def search(query: str, max_results: int):
        search_queries.append(query)
        suffix = len(search_queries)
        return [
            SearchResult(
                f"Acme {suffix}",
                f"https://acme-{suffix}.com",
                "Local service business with customer reactivation opportunity.",
                provider="tavily",
            )
        ]

    state = asyncio.run(
        run_research(
            "industry: dental practices\ngeography: Dallas\ncriteria: patient reactivation",
            thread_id="target-thread",
            checkpoint_dir=tmp_path,
            search=search,
            target_prospect_count=2,
            max_iterations=3,
            enable_llm_judgment=False,
        )
    )

    assert len(search_queries) == 2
    assert search_queries[0] == "Dallas dental practices official websites"
    assert [target["organization"] for target in state["prospect_targets"]] == [
        "Acme 1",
        "Acme 2",
    ]
    assert state["events"][-2]["reason"] == "target_prospect_count"


def test_prospect_search_queries_expand_directive_into_business_discovery_queries() -> None:
    queries = build_prospect_search_queries(
        "industry: med spas\ngeography: Phoenix\ncriteria: customer reactivation",
        max_iterations=3,
    )

    assert queries == [
        "Phoenix med spas official websites",
        "Phoenix med spas company contact about",
        "Phoenix local med spas service providers",
    ]


def test_prospect_search_queries_expand_ambiguous_geography_terms() -> None:
    queries = build_prospect_search_queries(
        "industry: HVAC\ngeography: North Texas\ncriteria: lead reactivation",
        max_iterations=6,
    )

    assert queries == [
        "North Texas HVAC official websites",
        "Dallas-Fort Worth TX HVAC official websites",
        "DFW HVAC company contact about",
        "Dallas TX HVAC service providers",
        "Fort Worth TX HVAC owner founder",
        "Plano TX HVAC about us contact",
    ]


def test_run_research_records_normalized_geography_scope(tmp_path) -> None:
    progress_events: list[dict[str, Any]] = []

    async def search(query: str, _max_results: int):
        suffix = len([event for event in progress_events if event["event"] == "search_started"])
        return [
            SearchResult(
                f"Actual HVAC {suffix}",
                f"https://actualhvac-{suffix}.com/",
                "We provide AC repair in Fort Worth and Dallas.",
                provider="tavily",
            )
        ]

    state = asyncio.run(
        run_research(
            "industry: HVAC\ngeography: North Texas\ncriteria: lead reactivation",
            thread_id="geo-thread",
            checkpoint_dir=tmp_path,
            search=search,
            target_prospect_count=2,
            max_iterations=2,
            enable_llm_judgment=False,
            progress_callback=progress_events.append,
        )
    )

    search_started = [
        event
        for event in progress_events
        if event["node"] == "search" and event["event"] == "search_started"
    ]
    assert state["geography_scope"]["canonical"] == "Dallas-Fort Worth TX"
    assert state["geography_scope"]["search_terms"][:3] == [
        "North Texas",
        "Dallas-Fort Worth TX",
        "DFW",
    ]
    assert [event["query"] for event in search_started] == [
        "North Texas HVAC official websites",
        "Dallas-Fort Worth TX HVAC official websites",
    ]
    assert search_started[0]["geography_scope"]["canonical"] == "Dallas-Fort Worth TX"
    assert state["evidence"][0]["query"] == "North Texas HVAC official websites"


def test_metadata_only_prospect_judgment_is_review_only_and_continues_geography(
    tmp_path,
) -> None:
    search_queries: list[str] = []
    long_owned_snippet = "We provide AC repair in Fort Worth and Dallas. " + (
        "Factory authorized dealer language should stay review-only. " * 30
    )

    async def search(query: str, _max_results: int):
        search_queries.append(query)
        if len(search_queries) == 1:
            return [
                SearchResult(
                    "Texas Air Conditioning Contractors Association",
                    "https://www.tacca.org/",
                    "Professional trade association for HVAC contractors.",
                    provider="tavily",
                ),
                SearchResult(
                    "Samsung HVAC North America",
                    "https://www.samsunghvac.com/",
                    "Ductless and VRF systems from a manufacturer.",
                    provider="tavily",
                ),
                SearchResult(
                    "North TX Comfort HVAC",
                    "https://northtxcomforthvac.com/",
                    long_owned_snippet,
                    provider="tavily",
                ),
            ]
        return [
            SearchResult(
                "DFW Family HVAC",
                "https://dfwfamilyhvac.com/",
                "Locally owned HVAC company serving Dallas-Fort Worth homes.",
                provider="tavily",
            )
        ]

    artifact_dir = tmp_path / "artifacts"
    state = asyncio.run(
        run_research(
            "industry: HVAC\ngeography: North Texas\ncriteria: lead reactivation",
            thread_id="metadata-only-thread",
            checkpoint_dir=tmp_path,
            search=search,
            model_client=MetadataOnlyProspectJudge(),
            target_prospect_count=1,
            max_iterations=2,
            artifact_dir=artifact_dir,
        )
    )

    assert search_queries == [
        "North Texas HVAC official websites",
        "Dallas-Fort Worth TX HVAC official websites",
    ]
    assert state["prospect_targets"] == []
    assert "model_judgment_metadata_only" in state["warnings"]
    qualifications = [
        review["qualification"]
        for review in state["prospect_reviews"]
        if review.get("qualification")
    ]
    assert any(
        qualification["qualification_status"] == "needs_review"
        and qualification["export_qualified"] is False
        and qualification["sufficiency_qualified"] is False
        and qualification["review_only_reason"] == "model_judgment_metadata_only"
        for qualification in qualifications
    )

    payload = json.loads(Path(state["artifact_paths"]["json"]).read_text())
    assert payload["metadata"]["prospect_targets"] == []
    assert payload["metadata"]["prospect_reviews"]
    assert payload["metadata"]["prospect_rejections"]
    assert payload["metadata"]["review_only_count"] >= 1
    assert payload["records"]
    assert all("qualification_status" in record for record in payload["records"])
    assert all("title" not in record and "url" not in record for record in payload["records"])
    assert any(record["qualification_status"] == "needs_review" for record in payload["records"])
    assert all(len(record["summary"]) <= 600 for record in payload["records"])
    assert any(
        len((review["judgment"] or {}).get("evidence_summary", "")) > 600
        for review in payload["metadata"]["prospect_reviews"]
    )


def test_run_research_surfaces_alias_suggestion_for_unknown_broad_region(tmp_path) -> None:
    state = asyncio.run(
        run_research(
            "industry: HVAC\ngeography: Central Plains region\ncriteria: lead reactivation",
            thread_id="geo-suggestion-thread",
            checkpoint_dir=tmp_path,
            search=None,
            target_prospect_count=1,
            max_iterations=1,
            enable_llm_judgment=False,
        )
    )

    suggestion = state["geography_alias_suggestion"]
    assert suggestion["alias_key"] == "central plains region"
    assert suggestion["source"] == "research_run"
    assert suggestion["search_terms"] == ["Central Plains region"]
    assert "Geography was not recognized" in state["warnings"][0]


def test_prospect_judge_prompt_includes_normalized_geography_scope() -> None:
    candidate = ProspectCandidate.from_evidence(
        {
            "id": "ev_1",
            "title": "Actual HVAC",
            "url": "https://actualhvac.com/",
            "snippet": "We provide AC repair in Fort Worth.",
        }
    )
    prompt = build_prospect_judge_prompt(
        directive={"industry": "HVAC", "geography": "North Texas"},
        geography_scope={
            "raw": "North Texas",
            "canonical": "Dallas-Fort Worth TX",
            "search_terms": ["North Texas", "Dallas-Fort Worth TX", "DFW"],
        },
        candidate=candidate,
        triage=triage_candidate(candidate),
        page_text="",
    )

    payload = json.loads(prompt)
    assert payload["geography_scope"]["canonical"] == "Dallas-Fort Worth TX"
    assert payload["geography_scope"]["search_terms"] == [
        "North Texas",
        "Dallas-Fort Worth TX",
        "DFW",
    ]
    assert payload["task"] == (
        "Evaluate whether this search result is an export-qualified prospect account."
    )
    assert payload["required_output_keys"][:5] == [
        "accepted",
        "organization",
        "canonical_website",
        "fit_score",
        "confidence",
    ]
    assert "Use the exact key 'accepted'; never use 'accept'." in payload["rules"]
    assert "Use the exact key 'reject_reason'; never use 'reason'." in payload["rules"]


def test_prospect_judgment_accepts_live_model_alias_fields_for_owned_business() -> None:
    candidate = ProspectCandidate.from_evidence(
        {
            "id": "ev_1",
            "title": "All Masters Plumbing | Plumbers in DFW & North Texas",
            "url": "https://allmastersplumbing.com/",
            "snippet": (
                "All Masters Plumbing is a trusted Dallas plumbing company offering "
                "24/7 service for leak repair, drains, and water heaters."
            ),
        }
    )
    triage = triage_candidate(
        candidate,
        directive={"industry": "plumber", "geography": "Texas"},
    )

    judgment = coerce_prospect_judgment(
        {
            "accept": True,
            "reason": (
                "All Masters Plumbing is a legitimate plumbing business operating "
                "in the Dallas-Fort Worth area."
            ),
        },
        candidate=candidate,
        triage=triage,
    )
    qualification = prospect_qualification(judgment, triage)

    assert judgment.accepted is True
    assert judgment.fit_score >= MIN_ACCEPTED_FIT_SCORE
    assert judgment.confidence >= MIN_ACCEPTED_CONFIDENCE
    assert judgment.reject_reason == ""
    assert "legitimate plumbing business" in judgment.fit_rationale
    assert "model_output_accept_alias" in judgment.guardrail_flags
    assert qualification.export_qualified is True
    assert qualification.sufficiency_qualified is True


def test_prospect_judgment_alias_fields_cannot_override_deterministic_reject() -> None:
    candidate = ProspectCandidate.from_evidence(
        {
            "id": "ev_1",
            "title": "Yelp",
            "url": "https://www.yelp.com/search?find_desc=Plumbing&find_loc=Dallas",
            "snippet": "Best Plumbing in Dallas, TX.",
        }
    )
    triage = triage_candidate(candidate)

    judgment = coerce_prospect_judgment(
        {"accept": True, "reason": "Contains Dallas plumbers."},
        candidate=candidate,
        triage=triage,
    )

    assert judgment.accepted is False
    assert "deterministic_reject" in judgment.guardrail_flags


def test_deterministic_triage_covers_broad_noise_categories() -> None:
    zoominfo = ProspectCandidate.from_evidence(
        {
            "id": "ev_1",
            "title": "ZoomInfo",
            "url": "https://www.zoominfo.com/c/north-texas-hvac-llc/365490708",
            "snippet": "North Texas HVAC contact info.",
        }
    )
    listicle = ProspectCandidate.from_evidence(
        {
            "id": "ev_2",
            "title": "Top-Rated HVAC Services in North Texas",
            "url": "https://www.northtexashvacguide.com/best-hvac",
            "snippet": "Best HVAC companies in North Texas.",
        }
    )
    contact = ProspectCandidate.from_evidence(
        {
            "id": "ev_3",
            "title": "Contact Us for Emergency HVAC Service in Fort Worth, TX",
            "url": "https://northtxcomforthvac.com/contact",
            "snippet": "We offer residential and commercial service.",
        }
    )
    homepage = ProspectCandidate.from_evidence(
        {
            "id": "ev_4",
            "title": "Actual HVAC",
            "url": "https://actualhvac.com/",
            "snippet": "We provide AC repair in Fort Worth.",
        }
    )
    association = ProspectCandidate.from_evidence(
        {
            "id": "ev_5",
            "title": "Texas Air Conditioning Contractors Association",
            "url": "https://www.tacca.org/",
            "snippet": "Professional trade association and advocacy chapter.",
        }
    )
    school = ProspectCandidate.from_evidence(
        {
            "id": "ev_6",
            "title": "HVAC Technician Training",
            "url": "https://www.techzonehvacr.com/",
            "snippet": "Technician training classes and certification program.",
        }
    )
    regulator = ProspectCandidate.from_evidence(
        {
            "id": "ev_7",
            "title": "Texas Department of Licensing HVAC",
            "url": "https://www.tdlr.texas.gov/acr/",
            "snippet": "Verify a license or renew a license with the state agency.",
        }
    )
    manufacturer = ProspectCandidate.from_evidence(
        {
            "id": "ev_8",
            "title": "Samsung HVAC North America",
            "url": "https://www.samsunghvac.com/",
            "snippet": "Ductless and VRF systems from a manufacturer.",
        }
    )
    national_brand = ProspectCandidate.from_evidence(
        {
            "id": "ev_9",
            "title": "One Hour Heating & Air Conditioning",
            "url": "https://www.onehourheatandair.com/locations/",
            "snippet": "Find a location from a national franchise brand.",
        }
    )

    assert triage_candidate(zoominfo).decision == "reject"
    assert triage_candidate(listicle).decision == "reject"
    assert triage_candidate(contact).decision == "fetch_then_judge"
    assert triage_candidate(homepage).decision == "judge"
    assert triage_candidate(association).source_category == "association"
    assert triage_candidate(association).decision == "reject"
    assert triage_candidate(school).source_category == "education"
    assert triage_candidate(school).decision == "reject"
    assert triage_candidate(regulator).source_category == "government"
    assert triage_candidate(regulator).decision == "reject"
    assert triage_candidate(manufacturer).source_category == "manufacturer"
    assert triage_candidate(manufacturer).decision == "reject"
    assert triage_candidate(national_brand).source_category == "national_brand"
    assert triage_candidate(national_brand).decision == "reject"


def test_owned_entity_noise_routes_to_review_instead_of_rejecting() -> None:
    local_dealer = ProspectCandidate.from_evidence(
        {
            "id": "ev_1",
            "title": "North Texas HVAC",
            "url": "https://northtxhvac.com/",
            "snippet": (
                "We provide AC repair in Fort Worth and are an authorized dealer "
                "for major manufacturer brands."
            ),
        }
    )
    local_training_word = ProspectCandidate.from_evidence(
        {
            "id": "ev_2",
            "title": "Titan Air Solutions",
            "url": "https://titanairdfw.com/",
            "snippet": "Our team receives technician training and provides HVAC service in DFW.",
        }
    )

    dealer_triage = triage_candidate(local_dealer)
    training_triage = triage_candidate(local_training_word)

    assert dealer_triage.source_category == "owned_or_unknown"
    assert "manufacturer" in dealer_triage.flags
    assert dealer_triage.decision == "fetch_then_judge"
    assert training_triage.source_category == "owned_or_unknown"
    assert "education" in training_triage.flags
    assert training_triage.decision == "fetch_then_judge"


def test_prospecting_vendor_noise_is_rejected() -> None:
    lead_vendor = ProspectCandidate.from_evidence(
        {
            "id": "ev_1",
            "title": "D7 Lead Finder",
            "url": "https://d7leadfinder.com/",
            "snippet": "Find contractors and generate HVAC leads with lead finder software.",
        }
    )

    triage = triage_candidate(lead_vendor)

    assert "vendor_noise" in triage.flags
    assert triage.decision == "reject"


def test_requested_entity_categories_are_not_hard_rejected_by_noise_filters() -> None:
    manufacturer = ProspectCandidate.from_evidence(
        {
            "id": "ev_1",
            "title": "Samsung HVAC North America",
            "url": "https://www.samsunghvac.com/",
            "snippet": "Ductless and VRF systems from a manufacturer.",
        }
    )
    national_brand = ProspectCandidate.from_evidence(
        {
            "id": "ev_2",
            "title": "One Hour Heating & Air Conditioning",
            "url": "https://www.onehourheatandair.com/locations/",
            "snippet": "Find a location from a national franchise brand.",
        }
    )

    assert (
        triage_candidate(manufacturer, directive={"industry": "HVAC manufacturers"}).decision
        == "judge"
    )
    assert triage_candidate(
        national_brand, directive={"industry": "HVAC franchises"}
    ).decision == "fetch_then_judge"


def test_business_name_hygiene_uses_domain_when_title_is_only_geography() -> None:
    assert (
        business_name_from_title("Fort Worth, TX", "northtxhvac.com")
        == "North Texas HVAC"
    )
    assert (
        business_name_from_title("Northtxhvac", "northtxhvac.com")
        == "North Texas HVAC"
    )
    assert (
        business_name_from_title(
            "Contact Us for Emergency HVAC Service in Fort Worth, TX",
            "northtxcomforthvac.com",
        )
        == "North Texas Comfort HVAC"
    )
    assert business_name_from_title("HVAC", "primarytx.com") == "Primary TX"
    assert business_name_from_title("United States", "excelgeothermal.com") == (
        "Excel Geothermal"
    )
    assert business_name_from_title("Dallas-fort Worth", "tempoair.com") == "Tempo Air"
    assert business_name_from_title("24/7 Emergency", "mycoolingcompany.com") == (
        "My Cooling Company"
    )
    assert business_name_from_title("24/7 Emergency", "kahnmechanical.com") == (
        "Kahn Mechanical"
    )
    assert business_name_from_title("Repair & Installation", "txairmechanics.com") == (
        "TX Air Mechanics"
    )
    assert business_name_from_title(
        "Texasexpresshvac AC Repair DFW",
        "texasexpresshvac.com",
    ) == "Texas Express HVAC"
    assert business_name_from_title(
        "Reliable HVAC Contractor In Dallas, Texas & Dallas County",
        "reynoldsheatnair.com",
    ) == "Reynolds Heat N Air"
    assert (
        business_name_from_title("Geothermal Installation", "excelgeothermal.com")
        == "Excel Geothermal"
    )
    assert (
        business_name_from_title("Dfw's HVAC Pros", "mycoolingcompany.com")
        == "My Cooling Company"
    )
    assert (
        business_name_from_title(
            "Top Rated HVAC Company in Dallas, TX",
            "proactiveairconditioning.com",
        )
        == "Proactive Air Conditioning"
    )
    assert (
        business_name_from_title(
            "Trane\u00ae | Harold James, Inc | Fort Worth, TX",
            "haroldjames.com",
        )
        == "Harold James, Inc"
    )


def test_run_research_exports_only_llm_accepted_owned_businesses(tmp_path) -> None:
    async def search(_query: str, _max_results: int):
        return [
            SearchResult(
                "ZoomInfo",
                "https://www.zoominfo.com/c/north-texas-hvac-llc/365490708",
                "North Texas HVAC contact info.",
                provider="tavily",
            ),
            SearchResult(
                "Top-Rated HVAC Services in North Texas",
                "https://www.northtexashvacguide.com/best-hvac",
                "Best HVAC companies in North Texas.",
                provider="tavily",
            ),
            SearchResult(
                "Actual HVAC",
                "https://actualhvac.com/",
                "We provide AC repair in Fort Worth.",
                provider="tavily",
            ),
        ]

    state = asyncio.run(
        run_research(
            "industry: HVAC\ngeography: North Texas\ncriteria: lead reactivation",
            thread_id="judge-thread",
            checkpoint_dir=tmp_path,
            search=search,
            model_client=FakeProspectJudge(),
            target_prospect_count=1,
            max_iterations=2,
        )
    )

    assert [target["organization"] for target in state["prospect_targets"]] == ["Actual HVAC"]
    assert state["prospect_targets"][0]["website"] == "https://actualhvac.com"
    metadata = state["prospect_targets"][0]["metadata"]
    assert metadata["qualification_status"] == "qualified"
    assert metadata["export_qualified"] is True
    assert metadata["sufficiency_qualified"] is True
    assert metadata["review_only"] is False
    assert any(
        rejection["triage"]["source_category"] == "directory"
        for rejection in state["prospect_rejections"]
    )


def test_hybrid_fetches_generic_owned_pages_before_judgment(tmp_path) -> None:
    fetched_urls: list[str] = []

    async def search(_query: str, _max_results: int):
        return [
            SearchResult(
                "Contact Us for Emergency HVAC Service in Fort Worth, TX",
                "https://actualhvac.com/contact",
                "We offer residential and commercial service.",
                provider="tavily",
            )
        ]

    async def page_fetch(url: str) -> str:
        fetched_urls.append(url)
        return "Actual HVAC is a locally owned HVAC company serving Fort Worth homeowners."

    state = asyncio.run(
        run_research(
            "industry: HVAC\ngeography: Fort Worth\ncriteria: customer winback",
            thread_id="fetch-thread",
            checkpoint_dir=tmp_path,
            search=search,
            model_client=FakeProspectJudge(),
            page_fetch=page_fetch,
            target_prospect_count=1,
            max_iterations=1,
        )
    )

    assert fetched_urls == ["https://actualhvac.com/contact"]
    assert state["prospect_targets"][0]["organization"] == "Actual HVAC"
    assert state["prospect_targets"][0]["metadata"]["source_url"] == (
        "https://actualhvac.com/contact"
    )
    assert any(record["source_type"] == "page_read" for record in state["evidence"])
