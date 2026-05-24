"""Local nested research workflow with durable checkpoint metadata for G003.

The implementation is intentionally local-first: it exposes a small graph-like
``invoke``/``ainvoke`` interface that works without a hosted LangGraph server,
database, or live model provider. Compatibility helpers below also preserve the
worker-lane API variants produced during the G003 team run.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict, cast
from uuid import uuid4

from async_multi_search import SearchResult

from .artifacts import write_research_artifacts
from .config import ModelProvider, load_config
from .models import ModelRequest, build_model_client
from .prospects import ProspectCitation, ProspectRecord

WorkflowStatus = Literal["completed", "interrupted"]
ReviewStatus = Literal["pending", "approved"]
SearchFn = Callable[[str, int], Awaitable[Sequence[SearchResult]] | Sequence[SearchResult]]
ProgressCallback = Callable[[dict[str, Any]], object]

DEFAULT_MAX_RESEARCH_ITERATIONS = 3
CHECKPOINT_SCHEMA_VERSION = "g003.local_checkpoint.v1"
GRAPH_TOPOLOGY = {
    "main": ("supervisor", "review"),
    "supervisor": ("researcher", "review", "finish"),
    "researcher": ("supervisor",),
}


class ResearchState(TypedDict, total=False):
    """State shared by the main, supervisor, and researcher workflow layers."""

    query: str
    thread_id: str
    answer: str
    status: WorkflowStatus
    review_status: ReviewStatus
    interrupt_reason: str
    interrupted: bool
    needs_review: bool
    sufficient: bool
    max_research_iterations: int
    max_iterations: int
    research_iterations: int
    iteration: int
    findings: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    events: list[dict[str, Any]]
    fallback_events: list[dict[str, Any]]
    model_metadata: dict[str, Any]
    review_interrupt: dict[str, Any]
    prospect_targets: list[dict[str, Any]]
    artifact_paths: dict[str, str]
    warnings: list[str]
    next_node: str


@dataclass(frozen=True)
class GraphRunResult:
    """Result returned by CLI helpers after a workflow invocation."""

    thread_id: str
    state: ResearchState
    checkpoint_path: Path


@dataclass(frozen=True)
class ResearchCheckpoint:
    """Legacy-compatible checkpoint DTO for CLI/API round trips."""

    thread_id: str
    query: str
    status: str
    review_required: bool
    sufficient: bool
    events: tuple[dict[str, Any], ...]
    fallback_metadata: dict[str, Any]
    checkpoint_path: Path
    state: ResearchState = field(default_factory=lambda: cast(ResearchState, {}))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "thread_id": self.thread_id,
            "query": self.query,
            "status": self.status,
            "review_required": self.review_required,
            "sufficient": self.sufficient,
            "events": list(self.events),
            "fallback_metadata": _jsonable(self.fallback_metadata),
            "checkpoint_path": str(self.checkpoint_path),
        }
        payload.update(_jsonable(dict(self.state)))
        payload["status"] = self.status
        payload["events"] = list(self.events)
        payload["fallback_metadata"] = _jsonable(self.fallback_metadata)
        payload["review_required"] = self.review_required
        payload["sufficient"] = self.sufficient
        payload["checkpoint_path"] = str(self.checkpoint_path)
        return payload


@dataclass(frozen=True)
class ModelFallbackEvent:
    """Serializable fallback metadata emitted by graph nodes."""

    provider: str
    model: str
    trigger: str
    node: str
    retry_count: int
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    error_class: str = ""
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalCheckpointStore:
    """Durable JSON checkpoint store keyed by LangGraph-style thread id."""

    def __init__(self, root: str | Path | None = None) -> None:
        configured = root or os.environ.get("DEEP_RESEARCH_CHECKPOINT_DIR")
        self.root = Path(configured or ".deep_research_agent/checkpoints")

    def path_for(self, thread_id: str) -> Path:
        safe_thread_id = re.sub(r"[^A-Za-z0-9_.-]", "_", thread_id)
        return self.root / f"{safe_thread_id}.json"

    def save(self, thread_id: str, state: Mapping[str, Any]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(thread_id)
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "thread_id": thread_id,
            "saved_at": datetime.now(UTC).isoformat(),
            "state": _jsonable(dict(state)),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def load(self, thread_id: str) -> ResearchState:
        path = self.path_for(thread_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"unsupported checkpoint schema for thread {thread_id!r}")
        return cast(ResearchState, dict(payload["state"]))

    def inspect(self, thread_id: str) -> dict[str, Any]:
        path = self.path_for(thread_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["checkpoint_path"] = str(path)
        return payload


class LocalResearchGraph:
    """Import-safe nested workflow runner with LangGraph-compatible methods."""

    def __init__(self, checkpoint_store: LocalCheckpointStore | None = None) -> None:
        self.checkpoint_store = checkpoint_store or LocalCheckpointStore()
        self.topology = GRAPH_TOPOLOGY

    def invoke(
        self, state: Mapping[str, Any], config: Mapping[str, Any] | None = None
    ) -> ResearchState:
        return _run_research_workflow(
            state,
            checkpoint_store=self.checkpoint_store,
            config=config,
        )

    async def ainvoke(
        self, state: Mapping[str, Any], config: Mapping[str, Any] | None = None
    ) -> ResearchState:
        return self.invoke(state, config=config)


class LocalResearchWorkflow:
    """Async compatibility wrapper used by G003 smoke/resume tests."""

    def __init__(self, checkpoint_store: LocalCheckpointStore | None = None) -> None:
        self.checkpoint_store = checkpoint_store or LocalCheckpointStore()

    async def arun(
        self,
        *,
        query: str,
        thread_id: str | None = None,
        search: SearchFn | None = None,
        require_review: bool = False,
        max_iterations: int = DEFAULT_MAX_RESEARCH_ITERATIONS,
        max_results: int = 5,
        progress_callback: ProgressCallback | None = None,
    ) -> ResearchState:
        return await run_research(
            query,
            thread_id=thread_id,
            checkpoint_dir=self.checkpoint_store.root,
            search=search,
            require_review=require_review,
            max_iterations=max_iterations,
            max_results=max_results,
            progress_callback=progress_callback,
        )

    async def aresume(
        self,
        thread_id: str,
        *,
        approve_review: bool = False,
        search: SearchFn | None = None,
        max_results: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> ResearchState:
        state = self.checkpoint_store.load(thread_id)
        if not approve_review and state.get("status") == "interrupted":
            return state
        query = str(state.get("query", ""))
        return await run_research(
            query,
            thread_id=thread_id,
            checkpoint_dir=self.checkpoint_store.root,
            search=search,
            require_review=False,
            max_iterations=int(state.get("max_iterations", state.get("iteration", 1)) or 1),
            max_results=_coerce_int(
                max_results if max_results is not None else state.get("max_results"), 5
            ),
            progress_callback=progress_callback,
            review_approved=True,
            existing_state=state,
        )


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, ModelProvider):
        return value.value
    return value


def _coerce_int(value: Any, default: int) -> int:
    return int(value if value is not None and value != "" else default)


def _append_event(
    state: MutableMapping[str, Any], node: str, event: str, **metadata: Any
) -> dict[str, Any]:
    entry = {
        "node": node,
        "event": event,
        "type": event,
        "timestamp": datetime.now(UTC).isoformat(),
        **_jsonable(metadata),
    }
    state.setdefault("events", []).append(entry)
    return entry


async def _notify_progress(
    progress_callback: ProgressCallback | None,
    event: Mapping[str, Any],
) -> None:
    if progress_callback is None:
        return
    result = progress_callback(dict(event))
    if isinstance(result, Awaitable):
        await result


def _append_fallback_event(
    state: MutableMapping[str, Any], *, node: str, trigger: str, retry_count: int = 0
) -> None:
    config = load_config()
    provider = config.primary_provider
    model = config.primary_model
    event = ModelFallbackEvent(
        provider=provider.value,
        model=model,
        trigger=trigger,
        node=node,
        retry_count=retry_count,
    )
    state.setdefault("fallback_events", []).append(event.to_dict())


def _thread_id_from_config(config: Mapping[str, Any] | None) -> str | None:
    if not config:
        return None
    configurable = config.get("configurable")
    if isinstance(configurable, Mapping):
        thread_id = configurable.get("thread_id")
        if thread_id:
            return str(thread_id)
    thread_id = config.get("thread_id")
    return str(thread_id) if thread_id else None


def _initial_state(
    state: Mapping[str, Any], *, config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    prepared = dict(state)
    prepared.setdefault("thread_id", _thread_id_from_config(config) or f"thread-{uuid4()}")
    prepared.setdefault("max_research_iterations", DEFAULT_MAX_RESEARCH_ITERATIONS)
    prepared.setdefault("research_iterations", 0)
    prepared.setdefault("findings", [])
    prepared.setdefault("events", [])
    prepared.setdefault("fallback_events", [])
    prepared.setdefault("sufficient", False)
    prepared.setdefault("needs_review", True)
    return prepared


def _supervisor_node(state: dict[str, Any]) -> str:
    _append_event(state, "supervisor", "evaluate")
    if state.get("sufficient"):
        state["next_node"] = "synthesize"
        return "synthesize"
    if int(state.get("research_iterations", 0)) >= int(
        state.get("max_research_iterations", DEFAULT_MAX_RESEARCH_ITERATIONS)
    ):
        state["sufficient"] = True
        state["next_node"] = "synthesize"
        _append_event(state, "supervisor", "max_iterations_reached")
        return "synthesize"
    state["next_node"] = "researcher"
    _append_event(state, "supervisor", "delegate", target="researcher")
    return "researcher"


def _researcher_node(state: dict[str, Any]) -> None:
    iteration = int(state.get("research_iterations", 0)) + 1
    state["research_iterations"] = iteration
    query = str(state.get("query", "")).strip()
    finding = {
        "iteration": iteration,
        "summary": f"Research pass {iteration} for: {query or 'unspecified query'}",
        "source_type": "mocked_local_research",
    }
    state.setdefault("findings", []).append(finding)
    _append_fallback_event(state, node="researcher", trigger="deterministic_local_research")
    _append_event(state, "researcher", "finding_recorded", iteration=iteration)
    if iteration >= int(state.get("max_research_iterations", DEFAULT_MAX_RESEARCH_ITERATIONS)):
        state["sufficient"] = True
        _append_event(state, "researcher", "sufficiency_reached", reason="max_iterations")


def _synthesize_node(state: dict[str, Any]) -> None:
    query = str(state.get("query", "")).strip()
    count = len(state.get("findings", []))
    state["answer"] = f"Draft answer for {query or 'the research query'} using {count} finding(s)."
    state["next_node"] = "review"
    _append_event(state, "synthesize", "draft_created", finding_count=count)


def _review_node(state: dict[str, Any]) -> None:
    if state.get("review_status") != "approved" and state.get("needs_review", True):
        state["status"] = "interrupted"
        state["interrupted"] = True
        state["interrupt_reason"] = "review_required"
        state["next_node"] = "review"
        _append_event(state, "review", "interrupt", reason="review_required")
        return
    state["status"] = "completed"
    state["interrupted"] = False
    state["review_status"] = "approved"
    state["next_node"] = "end"
    _append_event(state, "review", "approved")


def _run_research_workflow(
    state: Mapping[str, Any],
    *,
    checkpoint_store: LocalCheckpointStore,
    config: Mapping[str, Any] | None = None,
) -> ResearchState:
    current = _initial_state(state, config=config)
    if current.get("review_status") == "approved":
        current["needs_review"] = False
    _append_event(current, "main", "start", topology=GRAPH_TOPOLOGY)
    while True:
        next_node = _supervisor_node(current)
        if next_node == "synthesize":
            break
        _researcher_node(current)
    _synthesize_node(current)
    _review_node(current)
    checkpoint_store.save(str(current["thread_id"]), current)
    return cast(ResearchState, current)


def route_after_supervisor(state: Mapping[str, Any]) -> str:
    """Route to researcher, review, or finish from compatibility supervisor state."""

    evidence = list(state.get("evidence", []))
    iteration = int(state.get("iteration", state.get("research_iterations", 0)) or 0)
    max_iterations = int(state.get("max_iterations", DEFAULT_MAX_RESEARCH_ITERATIONS) or 0)
    if state.get("review_required") and not state.get("review_approved"):
        return "review"
    if evidence or iteration >= max_iterations:
        return "finish"
    return "researcher"


async def _call_search(
    search: SearchFn | None, query: str, max_results: int
) -> Sequence[SearchResult]:
    if search is None:
        return []
    result = search(query, max_results)
    if isinstance(result, Awaitable):
        return await result
    return result


def _evidence_from_results(results: Sequence[SearchResult]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for index, result in enumerate(results, start=1):
        evidence.append(
            {
                "id": f"ev_{index}",
                "title": result.title,
                "url": result.url,
                "snippet": getattr(result, "snippet", getattr(result, "content", "")),
                "provider": result.provider,
                "source_type": "snippet",
            }
        )
    return evidence


def _prospects_from_evidence(evidence: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    prospects: list[dict[str, Any]] = []
    for record in evidence:
        evidence_id = str(record.get("id", ""))
        organization = str(record.get("title") or record.get("url") or "Unknown prospect")
        snippet = str(record.get("snippet") or "")
        citation = ProspectCitation(
            evidence_id=evidence_id,
            claim=f"{organization} surfaced as a prospect discovery candidate.",
            quote="",
            purpose="discovery",
            field="target_account_list",
        )
        prospect = ProspectRecord(
            organization=organization,
            website=str(record.get("url") or ""),
            summary=snippet or f"Discovered candidate for the research query: {organization}.",
            confidence=0.55 if record.get("source_type") == "snippet" else 0.75,
            decision_maker_leads=("Founder/CEO", "Head of Growth"),
            fit_rationale=(
                "Candidate is included for discovery review based on cited search evidence; "
                "confirm fit with page-read evidence before outreach."
            ),
            personalized_angles=(
                snippet[:180] if snippet else "Use cited discovery evidence to tailor outreach.",
            ),
            citations=(citation,),
            metadata={
                "evidence_type": record.get("source_type", "snippet"),
                "provider": record.get("provider", ""),
            },
        )
        prospects.append(prospect.to_dict())
    return prospects


def _artifact_records(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    prospects = list(state.get("prospect_targets", []))
    if prospects:
        return prospects
    return list(state.get("evidence", []))


def _write_artifacts_if_requested(
    state: MutableMapping[str, Any], artifact_dir: str | Path | None
) -> None:
    if artifact_dir is None:
        return
    output_dir = Path(artifact_dir) / str(state["thread_id"])
    paths = write_research_artifacts(
        _artifact_records(state),
        output_dir,
        metadata={"thread_id": state["thread_id"], "query": state.get("query", "")},
    )
    state["artifact_paths"] = {
        "json": str(paths.json_path),
        "csv": str(paths.csv_path),
        "markdown": str(paths.markdown_path),
    }


async def run_research(
    query: str,
    *,
    thread_id: str | None = None,
    checkpoint_dir: str | Path | None = None,
    search: SearchFn | None = None,
    require_review: bool = False,
    max_iterations: int = DEFAULT_MAX_RESEARCH_ITERATIONS,
    max_results: int = 5,
    progress_callback: ProgressCallback | None = None,
    review_approved: bool = False,
    existing_state: Mapping[str, Any] | None = None,
    artifact_dir: str | Path | None = None,
) -> ResearchState:
    """Run the compatibility async research workflow used by G003 tests."""

    store = LocalCheckpointStore(checkpoint_dir)
    selected_thread_id: str
    if existing_state and not thread_id:
        selected_thread_id = str(existing_state.get("thread_id") or f"thread-{uuid4()}")
    else:
        selected_thread_id = thread_id or f"thread-{uuid4()}"
    state: dict[str, Any]
    if existing_state and review_approved:
        state = dict(existing_state)
        state.setdefault("events", [])
        state["thread_id"] = selected_thread_id
        state["status"] = "completed"
        state["sufficient"] = bool(state.get("sufficient", True))
        await _notify_progress(
            progress_callback, _append_event(state, "review", "review_resumed")
        )
        await _notify_progress(
            progress_callback, _append_event(state, "main", "workflow_completed")
        )
        _write_artifacts_if_requested(state, artifact_dir)
        store.save(selected_thread_id, state)
        return cast(ResearchState, state)

    events = list(existing_state.get("events", [])) if existing_state else []
    state = {
        "query": query,
        "thread_id": selected_thread_id,
        "events": events,
        "iteration": 0,
        "max_iterations": max_iterations,
        "max_results": max_results,
        "evidence": [],
        "findings": [],
        "fallback_events": [],
        "prospect_targets": [],
        "warnings": [],
    }

    async def record(node: str, event: str, **metadata: Any) -> None:
        await _notify_progress(progress_callback, _append_event(state, node, event, **metadata))

    await record("main", "main_started")

    while route_after_supervisor(state) == "researcher":
        await record("supervisor", "supervisor_delegated", target="researcher")
        await record("search", "search_started", max_results=max_results)
        results = await _call_search(search, query, max_results=max_results)
        await record("search", "search_completed", result_count=len(results))
        state["evidence"] = _evidence_from_results(results)
        state["findings"] = list(state["evidence"])
        state["prospect_targets"] = _prospects_from_evidence(state["evidence"])
        state["iteration"] = int(state.get("iteration", 0)) + 1
        model_response = await build_model_client().invoke(
            ModelRequest(
                node="researcher",
                prompt=query,
                metadata={"thread_id": selected_thread_id, "iteration": state["iteration"]},
            )
        )
        state["model_metadata"] = model_response.to_dict()
        await record("model", "fallback_model_used", provider=model_response.provider.value)
        await record("researcher", "researcher_iteration", iteration=state["iteration"])
        if state["evidence"] or int(state["iteration"]) >= max_iterations:
            state["sufficient"] = True
            await record("supervisor", "sufficiency_routed")
            break

    if not state["evidence"]:
        state["warnings"].append("No search evidence was captured; prospect exports are empty.")
        await record("warning", "warning_recorded", message=state["warnings"][-1])

    if require_review and not review_approved:
        state["status"] = "interrupted"
        state["review_interrupt"] = {"thread_id": selected_thread_id, "reason": "review_required"}
        await record("review", "review_interrupt", reason="review_required")
    else:
        state["status"] = "completed"
        if review_approved or existing_state:
            await record("review", "review_resumed")
        await record("main", "workflow_completed")
    _write_artifacts_if_requested(state, artifact_dir)
    store.save(selected_thread_id, state)
    return cast(ResearchState, state)


async def resume_research(
    thread_id: str,
    *,
    checkpoint_dir: str | Path | None = None,
    approve_review: bool = True,
    search: SearchFn | None = None,
    max_results: int | None = None,
    progress_callback: ProgressCallback | None = None,
    artifact_dir: str | Path | None = None,
) -> ResearchState:
    """Resume a compatibility async workflow from a checkpoint."""

    store = LocalCheckpointStore(checkpoint_dir)
    state = store.load(thread_id)
    if state.get("status") == "interrupted" and not approve_review:
        return state
    return await run_research(
        str(state.get("query", "")),
        thread_id=thread_id,
        checkpoint_dir=checkpoint_dir,
        search=search,
        require_review=False,
        max_iterations=int(state.get("max_iterations", state.get("iteration", 1)) or 1),
        max_results=_coerce_int(
            max_results if max_results is not None else state.get("max_results"), 5
        ),
        progress_callback=progress_callback,
        review_approved=approve_review,
        existing_state=state,
        artifact_dir=artifact_dir,
    )


def inspect_checkpoints(
    thread_id: str, *, checkpoint_dir: str | Path | None = None
) -> ResearchState:
    """Return the saved state for a compatibility checkpoint."""

    return LocalCheckpointStore(checkpoint_dir).load(thread_id)


def build_graph(checkpoint_store: LocalCheckpointStore | None = None) -> LocalResearchGraph:
    """Build the import-safe local graph runner used by package and CLI entrypoints."""

    return LocalResearchGraph(checkpoint_store=checkpoint_store)


def run_query(
    query: str,
    *,
    thread_id: str | None = None,
    checkpoint_dir: str | Path | None = None,
    max_iterations: int = DEFAULT_MAX_RESEARCH_ITERATIONS,
    approve: bool = False,
) -> GraphRunResult:
    store = LocalCheckpointStore(checkpoint_dir)
    state: ResearchState = {
        "query": query,
        "max_research_iterations": max_iterations,
    }
    if thread_id:
        state["thread_id"] = thread_id
    if approve:
        state["review_status"] = "approved"
    config = {"configurable": {"thread_id": thread_id}} if thread_id else None
    result = build_graph(store).invoke(state, config=config)
    return GraphRunResult(
        thread_id=str(result["thread_id"]),
        state=result,
        checkpoint_path=store.path_for(str(result["thread_id"])),
    )


def resume_thread(
    thread_id: str,
    *,
    checkpoint_dir: str | Path | None = None,
    approve: bool = False,
) -> GraphRunResult:
    store = LocalCheckpointStore(checkpoint_dir)
    state = store.load(thread_id)
    if approve:
        state["review_status"] = "approved"
        state["needs_review"] = False
    result = build_graph(store).invoke(state, config={"configurable": {"thread_id": thread_id}})
    return GraphRunResult(
        thread_id=thread_id, state=result, checkpoint_path=store.path_for(thread_id)
    )


def inspect_thread(
    thread_id: str, *, checkpoint_dir: str | Path | None = None
) -> dict[str, Any]:
    return LocalCheckpointStore(checkpoint_dir).inspect(thread_id)


def _checkpoint_from_state(
    state: ResearchState, checkpoint_dir: str | Path | None = None
) -> ResearchCheckpoint:
    status = "complete" if state.get("status") == "completed" else "needs_review"
    metadata = {
        "provider": "local_mock",
        "model": "deterministic-local",
        "trigger": "metadata_only",
    }
    return ResearchCheckpoint(
        thread_id=str(state["thread_id"]),
        query=str(state.get("query", "")),
        status=status,
        review_required=status == "needs_review",
        sufficient=bool(state.get("sufficient", True)),
        events=tuple(state.get("events", [])),
        fallback_metadata=dict(metadata),
        checkpoint_path=LocalCheckpointStore(checkpoint_dir).path_for(str(state["thread_id"])),
        state=state,
    )


def run_research_workflow(
    query: str,
    *,
    thread_id: str | None = None,
    checkpoint_dir: str | Path | None = None,
    max_results: int = 5,
    artifact_dir: str | Path | None = None,
) -> ResearchCheckpoint:
    state = asyncio.run(
        run_research(
            query,
            thread_id=thread_id,
            checkpoint_dir=checkpoint_dir,
            require_review=True,
            max_results=max_results,
            artifact_dir=artifact_dir,
        )
    )
    return _checkpoint_from_state(state, checkpoint_dir)


def resume_research_workflow(
    thread_id: str,
    *,
    checkpoint_dir: str | Path | None = None,
    artifact_dir: str | Path | None = None,
) -> ResearchCheckpoint:
    state = asyncio.run(
        resume_research(
            thread_id,
            checkpoint_dir=checkpoint_dir,
            approve_review=True,
            artifact_dir=artifact_dir,
        )
    )
    return _checkpoint_from_state(state, checkpoint_dir)


def inspect_research_thread(
    thread_id: str, *, checkpoint_dir: str | Path | None = None
) -> ResearchCheckpoint:
    state = inspect_checkpoints(thread_id, checkpoint_dir=checkpoint_dir)
    return _checkpoint_from_state(state, checkpoint_dir)


def invoke_maybe_async(runnable: Any, state: Mapping[str, Any]) -> Any:
    """Invoke a graph-like object from sync tests without leaking event loops."""

    if hasattr(runnable, "invoke"):
        return runnable.invoke(state)
    result = runnable(state)
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return result


graph = build_graph()
