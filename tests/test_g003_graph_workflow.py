from __future__ import annotations

import asyncio
import json
import subprocess
import sys

from async_multi_search import SearchResult
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
                f"https://acme-{suffix}.example.com",
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
        )
    )

    assert len(search_queries) == 2
    assert search_queries[0] == "Dallas dental practices Contact About"
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
        "Phoenix med spas Contact About",
        "Phoenix med spa official website",
        "Phoenix local med spas company",
    ]
