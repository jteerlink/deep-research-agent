from __future__ import annotations

import asyncio
import json
import subprocess
import sys
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
from deep_research_agent.prospect_judgment import ProspectCandidate, triage_candidate


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

    assert triage_candidate(zoominfo).decision == "reject"
    assert triage_candidate(listicle).decision == "reject"
    assert triage_candidate(contact).decision == "fetch_then_judge"
    assert triage_candidate(homepage).decision == "judge"


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
