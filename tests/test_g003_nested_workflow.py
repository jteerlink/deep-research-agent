from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

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
    resume_research,
    route_after_supervisor,
    run_research,
)
from deep_research_agent.models import (
    ConfiguredModelClient,
    ModelPreflightError,
    ModelRequest,
    build_model_client,
)

ROOT = Path(__file__).resolve().parents[1]


async def _mock_search(query: str, max_results: int):
    assert "acme research" in query
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
    assert (
        route_after_supervisor(
            {
                "prospect_targets": [{"organization": "Acme"}],
                "iteration": 1,
                "max_iterations": 3,
                "target_prospect_count": 2,
            }
        )
        == "researcher"
    )
    assert (
        route_after_supervisor(
            {
                "prospect_targets": [{"organization": "Acme"}, {"organization": "Beta"}],
                "iteration": 1,
                "max_iterations": 3,
                "target_prospect_count": 2,
            }
        )
        == "finish"
    )
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


def test_model_client_does_not_treat_local_ollama_as_available() -> None:
    config = AppConfig(
        primary_provider=ModelProvider.OLLAMA_NATIVE,
        ollama_native=OllamaNativeConfig(
            base_url="http://localhost:11434/api",
            model="gpt-oss:120b",
            api_key="local-key",
        ),
        ollama_openai=OllamaOpenAIConfig(
            base_url="http://localhost:11434/v1",
            model="gpt-oss:120b",
            api_key="ollama",
        ),
        openai=OpenAIConfig(api_key="", model="openai-test"),
        codex=CodexConfig(api_key="", model="codex-test"),
        search=SearchConfig(),
    )
    client = build_model_client(config)

    assert client.provider_available(ModelProvider.OLLAMA_NATIVE) is False
    assert client.provider_available(ModelProvider.OLLAMA_OPENAI) is False


def test_model_preflight_reports_missing_ollama_cloud_key_without_secret_values() -> None:
    config = AppConfig(
        primary_provider=ModelProvider.OLLAMA_NATIVE,
        ollama_native=OllamaNativeConfig(api_key=""),
        ollama_openai=OllamaOpenAIConfig(),
        openai=OpenAIConfig(api_key="", model="openai-test"),
        codex=CodexConfig(api_key="", model="codex-test"),
        search=SearchConfig(),
    )

    payload = build_model_client(config).preflight().to_dict()

    assert payload["live_model_available"] is False
    assert payload["selected_provider"] == ""
    assert payload["providers"][0]["provider"] == "ollama_native"
    assert payload["providers"][0]["unavailable_reason"] == "missing_api_key"
    assert "secret" not in json.dumps(payload).lower()


def test_model_preflight_selects_ollama_cloud_when_api_key_is_present() -> None:
    config = AppConfig(
        primary_provider=ModelProvider.OLLAMA_NATIVE,
        ollama_native=OllamaNativeConfig(api_key="ollama-secret"),
        ollama_openai=OllamaOpenAIConfig(),
        openai=OpenAIConfig(api_key="", model="openai-test"),
        codex=CodexConfig(api_key="", model="codex-test"),
        search=SearchConfig(),
    )

    payload = build_model_client(config).preflight().to_dict()

    assert payload["live_model_available"] is True
    assert payload["selected_provider"] == "ollama_native"
    assert payload["providers"][0]["api_key_configured"] is True
    assert "ollama-secret" not in json.dumps(payload)


def test_model_preflight_rejects_local_ollama_even_with_key() -> None:
    config = AppConfig(
        primary_provider=ModelProvider.OLLAMA_NATIVE,
        ollama_native=OllamaNativeConfig(
            base_url="http://localhost:11434/api",
            model="gpt-oss:120b",
            api_key="local-key",
        ),
        ollama_openai=OllamaOpenAIConfig(),
        openai=OpenAIConfig(api_key="", model="openai-test"),
        codex=CodexConfig(api_key="", model="codex-test"),
        search=SearchConfig(),
    )

    payload = build_model_client(config).preflight().to_dict()

    assert payload["live_model_available"] is False
    assert payload["providers"][0]["unavailable_reason"] == "local_host_not_supported"


def test_require_live_model_fails_before_search_when_no_provider_available() -> None:
    search_called = False

    async def search(_query: str, _max_results: int):
        nonlocal search_called
        search_called = True
        return []

    config = AppConfig(
        primary_provider=ModelProvider.OLLAMA_NATIVE,
        ollama_native=OllamaNativeConfig(api_key=""),
        ollama_openai=OllamaOpenAIConfig(),
        openai=OpenAIConfig(api_key="", model="openai-test"),
        codex=CodexConfig(api_key="", model="codex-test"),
        search=SearchConfig(),
    )

    with pytest.raises(ModelPreflightError, match="Live model required"):
        asyncio.run(
            run_research(
                "industry: HVAC\ngeography: North Texas",
                search=search,
                model_client=build_model_client(config),
                require_live_model=True,
                max_iterations=1,
            )
        )

    assert search_called is False


def test_resume_require_live_model_checks_preflight_before_interrupted_return(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    async def search(_query: str, _max_results: int):
        return [
            SearchResult(
                "Actual HVAC",
                "https://actualhvac.com/",
                "We provide AC repair in Fort Worth.",
                provider="tavily",
            )
        ]

    asyncio.run(
        run_research(
            "industry: HVAC\ngeography: North Texas",
            thread_id="interrupted-thread",
            checkpoint_dir=tmp_path,
            search=search,
            require_review=True,
            enable_llm_judgment=False,
            max_iterations=1,
        )
    )

    with pytest.raises(ModelPreflightError, match="Live model required"):
        asyncio.run(
            resume_research(
                "interrupted-thread",
                checkpoint_dir=tmp_path,
                approve_review=False,
                require_live_model=True,
            )
        )


def test_configured_ollama_live_judgment_can_export_qualified_prospect(monkeypatch) -> None:
    async def transport(
        _self: ConfiguredModelClient,
        provider: ModelProvider,
        model: str,
        request: ModelRequest,
    ) -> str:
        assert provider is ModelProvider.OLLAMA_NATIVE
        assert model == "gpt-oss:120b"
        payload = json.loads(request.prompt)
        candidate = payload["candidate"]
        return json.dumps(
            {
                "accepted": True,
                "organization": "Actual HVAC",
                "canonical_website": candidate["canonical_website"],
                "fit_score": 0.91,
                "confidence": 0.86,
                "reject_reason": "",
                "fit_rationale": "Owned HVAC business in the target market.",
                "evidence_summary": candidate["snippet"],
                "personalized_angles": ["HVAC service follow-up"],
                "decision_maker_leads": ["Owner"],
                "guardrail_flags": [],
            }
        )

    async def search(_query: str, _max_results: int):
        return [
            SearchResult(
                "Actual HVAC",
                "https://actualhvac.com/",
                "We provide AC repair in Fort Worth.",
                provider="tavily",
            )
        ]

    monkeypatch.setattr(ConfiguredModelClient, "_transport", transport)
    config = AppConfig(
        primary_provider=ModelProvider.OLLAMA_NATIVE,
        ollama_native=OllamaNativeConfig(api_key="ollama-key"),
        ollama_openai=OllamaOpenAIConfig(),
        openai=OpenAIConfig(api_key="", model="openai-test"),
        codex=CodexConfig(api_key="", model="codex-test"),
        search=SearchConfig(),
    )

    state = asyncio.run(
        run_research(
            "industry: HVAC\ngeography: North Texas",
            search=search,
            model_client=build_model_client(config),
            require_live_model=True,
            max_iterations=1,
            target_prospect_count=1,
        )
    )

    assert state["model_preflight"]["live_model_available"] is True
    assert state["prospect_targets"][0]["organization"] == "Actual HVAC"
    assert state["prospect_targets"][0]["metadata"]["export_qualified"] is True


def test_ollama_model_transport_uses_configured_ca_bundle(monkeypatch) -> None:
    captured: dict[str, object] = {}
    verify_context = object()

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"response": '{"ok": true}'}

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, *_args, **_kwargs):
            captured["post_json"] = _kwargs.get("json")
            return FakeResponse()

    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/tmp/zscaler.pem")
    monkeypatch.setattr(
        "deep_research_agent.models.ssl.create_default_context",
        lambda *, cafile: verify_context,
    )
    monkeypatch.setattr("deep_research_agent.models.httpx.AsyncClient", FakeAsyncClient)
    config = AppConfig(
        primary_provider=ModelProvider.OLLAMA_NATIVE,
        ollama_native=OllamaNativeConfig(api_key="ollama-key"),
        ollama_openai=OllamaOpenAIConfig(),
        openai=OpenAIConfig(api_key="", model="openai-test"),
        codex=CodexConfig(api_key="", model="codex-test"),
        search=SearchConfig(),
    )

    response = asyncio.run(build_model_client(config).live_smoke())

    assert response.structured == {"ok": True}
    assert captured["verify"] is verify_context
    assert captured["timeout"] == 60
    assert captured["post_json"]["format"]["required"] == ["ok"]


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


def test_resume_preserves_checkpointed_evidence_and_artifacts(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    artifact_dir = tmp_path / "artifacts"
    interrupted = asyncio.run(
        run_research(
            "acme research",
            thread_id="artifact-thread",
            checkpoint_dir=checkpoint_dir,
            search=_mock_search,
            require_review=True,
            artifact_dir=artifact_dir,
        )
    )

    resumed = asyncio.run(
        resume_research(
            "artifact-thread",
            checkpoint_dir=checkpoint_dir,
            approve_review=True,
            artifact_dir=artifact_dir,
        )
    )

    assert interrupted["evidence"]
    assert resumed["status"] == "completed"
    assert resumed["evidence"] == interrupted["evidence"]
    assert resumed["findings"] == interrupted["findings"]
    assert resumed["prospect_targets"] == interrupted["prospect_targets"]
    assert resumed["prospect_reviews"] == interrupted["prospect_reviews"]
    assert resumed["prospect_rejections"] == interrupted["prospect_rejections"]
    artifact_payload = json.loads(Path(resumed["artifact_paths"]["json"]).read_text())
    assert artifact_payload["metadata"]["prospect_reviews"] == interrupted["prospect_reviews"]
    assert artifact_payload["metadata"]["prospect_rejections"] == interrupted["prospect_rejections"]


def test_cli_approve_alias_writes_artifacts(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    artifact_dir = tmp_path / "artifacts"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "deep_research_agent",
            "run",
            "acme research",
            "--thread-id",
            "approve-thread",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--artifact-dir",
            str(artifact_dir),
            "--require-review",
            "--mock-result",
            "Acme|https://example.com/acme|Acme builds research tools|duckduckgo",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    resumed = subprocess.run(
        [
            sys.executable,
            "-m",
            "deep_research_agent",
            "resume",
            "approve-thread",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--artifact-dir",
            str(artifact_dir),
            "--approve",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(resumed.stdout)
    assert payload["status"] == "completed"
    assert Path(payload["artifact_paths"]["json"]).exists()
