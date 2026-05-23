"""Local nested research workflow with durable checkpoint metadata for G003.

The implementation is intentionally local-first: it exposes a small graph-like
``invoke``/``ainvoke`` interface that works without a hosted LangGraph server,
database, or live model provider. When the optional LangGraph dependency is not
installed this object remains import-safe for the package and CLI surfaces.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict
from uuid import uuid4

from .config import ModelProvider, load_config

WorkflowStatus = Literal["completed", "interrupted"]
ReviewStatus = Literal["pending", "approved"]

DEFAULT_MAX_RESEARCH_ITERATIONS = 3
CHECKPOINT_SCHEMA_VERSION = "g003.local_checkpoint.v1"
GRAPH_TOPOLOGY = {
    "main": ("supervisor", "review"),
    "supervisor": ("researcher", "synthesize"),
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
    research_iterations: int
    findings: list[dict[str, Any]]
    events: list[dict[str, Any]]
    fallback_events: list[dict[str, Any]]
    next_node: str


@dataclass(frozen=True)
class GraphRunResult:
    """Result returned by CLI helpers after a workflow invocation."""

    thread_id: str
    state: ResearchState
    checkpoint_path: Path


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
        return dict(payload["state"])

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


def _append_event(state: MutableMapping[str, Any], node: str, event: str, **metadata: Any) -> None:
    state.setdefault("events", []).append(
        {
            "node": node,
            "event": event,
            "timestamp": datetime.now(UTC).isoformat(),
            **_jsonable(metadata),
        }
    )


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
    state: Mapping[str, Any], *, checkpoint_store: LocalCheckpointStore, config: Mapping[str, Any] | None = None
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
    return current


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
    result = build_graph(store).invoke(state, config={"configurable": {"thread_id": thread_id}} if thread_id else None)
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
    return GraphRunResult(thread_id=thread_id, state=result, checkpoint_path=store.path_for(thread_id))


def inspect_thread(
    thread_id: str, *, checkpoint_dir: str | Path | None = None) -> dict[str, Any]:
    return LocalCheckpointStore(checkpoint_dir).inspect(thread_id)


def invoke_maybe_async(runnable: Any, state: Mapping[str, Any]) -> Any:
    """Invoke a graph-like object from sync tests without leaking event loops."""

    if hasattr(runnable, "invoke"):
        return runnable.invoke(state)
    result = runnable(state)
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return result


graph = build_graph()
