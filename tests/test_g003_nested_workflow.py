from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from async_multi_search import SearchResult
from deep_research_agent.config import (
    AppConfig,
    CodexConfig,
    ModelProvider,
    OllamaNativeConfig,
    OllamaOpenAIConfig,
    OpenAIConfig,
    SearchConfig,
)
from deep_research_agent.graph import (
    GRAPH_TOPOLOGY,
    LocalCheckpointStore,
    LocalResearchWorkflow,
    inspect_checkpoints,
    route_after_supervisor,
    run_research,
)
from deep_research_agent.models import ModelRequest, build_model_client

ROOT = Path(__file__).resolve().parents[1]


async def _mock_search(query: str, max_results: int):
    assert query == "acme research"
    assert max_results >= 1
    return [
        SearchResult(
            "Acme",
            "https://example.com/acme",
            "Acme builds research tools",
            provider="duckduckgo",
        )
    ]


def test_mocked_graph_smoke_collects_evidence_and_checkpoints(tmp_path) -> None:
    state = asyncio.run(
        run_research(
            "acme research",
            thread_id="smoke-thread",
            checkpoint_dir=tmp_path,
            search=_mock_search,
        )
    )

    assert state["status"] == "completed"
    assert state["thread_id"] == "smoke-thread"
    assert state["evidence"][0]["provider"] == "duckduckgo"
    assert state["model_metadata"]["structured"]["node"] == "researcher"
    assert (tmp_path / "smoke-thread.json").exists()
    assert GRAPH_TOPOLOGY["supervisor"] == ("researcher", "review", "finish")


def test_supervisor_delegates_until_sufficient_or_max_iteration() -> None:
    assert (
        route_after_supervisor({"evidence": [], "iteration": 0, "max_iterations": 2})
        == "researcher"
    )
    assert route_after_supervisor({"evidence": [{"id": "ev_1"}], "iteration": 1}) == "finish"
    assert route_after_supervisor({"evidence": [], "iteration": 2, "max_iterations": 2}) == "finish"
    assert (
        route_after_supervisor(
            {
                "evidence": [{"id": "ev_1"}],
                "iteration": 1,
                "review_required": True,
                "review_approved": False,
            }
        )
        == "review"
    )


def test_model_client_records_fallback_events_for_unavailable_hosted_providers() -> None:
    config = AppConfig(
        primary_provider=ModelProvider.CODEX,
        ollama_native=OllamaNativeConfig(),
        ollama_openai=OllamaOpenAIConfig(),
        openai=OpenAIConfig(api_key="", model="openai-test"),
        codex=CodexConfig(api_key="", model="codex-test"),
        search=SearchConfig(),
    )
    response = asyncio.run(
        build_model_client(config).invoke(ModelRequest(node="researcher", prompt="query"))
    )

    assert response.structured["status"] == "no_available_model"
    assert [event.provider for event in response.fallback_events] == [
        ModelProvider.CODEX,
        ModelProvider.OPENAI,
    ]
    assert [event.trigger for event in response.fallback_events] == [
        "missing_api_key",
        "missing_api_key",
    ]


def test_thread_id_resume_approves_review_interrupt(tmp_path) -> None:
    workflow = LocalResearchWorkflow(checkpoint_store=LocalCheckpointStore(tmp_path))
    interrupted = asyncio.run(
        workflow.arun(
            query="acme research",
            thread_id="review-thread",
            search=_mock_search,
            require_review=True,
        )
    )

    assert interrupted["status"] == "interrupted"
    assert interrupted["review_interrupt"]["thread_id"] == "review-thread"

    same_state = asyncio.run(workflow.aresume("review-thread"))
    assert same_state["status"] == "interrupted"

    completed = asyncio.run(
        workflow.aresume("review-thread", approve_review=True, search=_mock_search)
    )
    assert completed["status"] == "completed"
    assert completed["thread_id"] == "review-thread"
    assert inspect_checkpoints("review-thread", checkpoint_dir=tmp_path)["status"] == "completed"


def test_cli_run_resume_inspect_flow_uses_thread_id_checkpoint(tmp_path) -> None:
    env = {"PYTHONPATH": str(ROOT)}
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "deep_research_agent",
            "run",
            "acme research",
            "--thread-id",
            "cli-thread",
            "--checkpoint-dir",
            str(tmp_path),
            "--require-review",
            "--mock-result",
            "Acme|https://example.com/acme|Acme builds research tools|duckduckgo",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    run_state = json.loads(run.stdout)
    assert run_state["status"] == "interrupted"

    resume = subprocess.run(
        [
            sys.executable,
            "-m",
            "deep_research_agent",
            "resume",
            "cli-thread",
            "--checkpoint-dir",
            str(tmp_path),
            "--approve-review",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(resume.stdout)["status"] == "completed"

    inspected = subprocess.run(
        [
            sys.executable,
            "-m",
            "deep_research_agent",
            "inspect",
            "--thread-id",
            "cli-thread",
            "--checkpoint-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(inspected.stdout)["status"] == "completed"


def test_langgraph_json_src_entrypoint_reexports_packaged_graph() -> None:
    module_path = ROOT / "src" / "deep_research_agent" / "graph.py"
    spec = importlib.util.spec_from_file_location("g003_src_graph_probe", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.GRAPH_TOPOLOGY["main"] == ("supervisor",)
    assert hasattr(module.graph, "invoke") or callable(module.graph)
