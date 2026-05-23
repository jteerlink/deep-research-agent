"""Nested local research workflow with durable thread checkpoints.

G003 keeps execution local-first and import-safe while modeling the approved
main -> supervisor -> researcher topology. LangGraph is optional: when it is not
installed, the exported ``graph`` object still supports ``invoke``/``ainvoke``
for mocked smoke tests and CLI execution.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, TypedDict
from uuid import uuid4

from async_multi_search import SearchResult

from .config import AppConfig, load_config
from .evidence import SearchFailure, normalize_search_results
from .models import FallbackEvent, ModelClient, ModelRequest, build_model_client

WorkflowStatus = Literal["running", "interrupted", "completed", "failed"]
RouteName = Literal["researcher", "review", "finish"]
SearchCallable = Callable[[str, int], Sequence[SearchResult] | Awaitable[Sequence[SearchResult]]]

DEFAULT_MAX_ITERATIONS = 3
DEFAULT_MIN_EVIDENCE_RECORDS = 1
CHECKPOINT_SCHEMA_VERSION = "g003.local_checkpoint.v1"
GRAPH_TOPOLOGY = {
    "main": ("supervisor",),
    "supervisor": ("researcher", "review", "finish"),
    "researcher": ("supervisor",),
    "review": ("finish",),
    "finish": (),
}


class ResearchState(TypedDict, total=False):
    """State carried through the nested main/supervisor/researcher workflow."""

    query: str
    thread_id: str
    status: WorkflowStatus
    next: RouteName
    iteration: int
    max_iterations: int
    min_evidence_records: int
    evidence: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    answer: str
    sufficient: bool
    fallback_events: list[dict[str, Any]]
    model_metadata: dict[str, Any]
    review_required: bool
    review_interrupt: dict[str, Any]
    review_approved: bool
    messages: list[dict[str, Any]]


class ReviewInterrupt(RuntimeError):
    """Raised by callers that request exception-style review interruption."""

    def __init__(self, state: ResearchState):
        self.state = state
        super().__init__(f"review required for thread {state.get('thread_id', '')}")


class LocalCheckpointStore:
    """Durable JSON checkpoint store keyed by thread id."""

    def __init__(self, root: str | Path | None = None):
        default_root = os.environ.get(
            "DEEP_RESEARCH_CHECKPOINT_DIR", ".deep_research_agent/checkpoints"
        )
        self.root = Path(default_root if root is None else root)

    def path_for(self, thread_id: str) -> Path:
        """Return the checkpoint path for ``thread_id``."""

        safe_thread_id = thread_id.replace("/", "_")
        return self.root / f"{safe_thread_id}.json"

    def save(self, state: Mapping[str, Any]) -> Path:
        """Persist a checkpoint and return its path."""

        thread_id = str(state.get("thread_id") or "")
        if not thread_id:
            raise ValueError("checkpoint state requires thread_id")
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(thread_id)
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "thread_id": thread_id,
            "state": _jsonable_state(state),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def load(self, thread_id: str) -> ResearchState:
        """Load checkpoint state for ``thread_id``."""

        path = self.path_for(thread_id)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found for thread_id={thread_id!r}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"unsupported checkpoint schema: {payload.get('schema_version')!r}")
        return dict(payload["state"])

    def inspect(self, thread_id: str | None = None) -> dict[str, Any]:
        """Return checkpoint metadata for one thread or all local threads."""

        if thread_id:
            path = self.path_for(thread_id)
            state = self.load(thread_id)
            return {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "thread_id": thread_id,
                "path": str(path),
                "exists": True,
                "status": state.get("status"),
                "iteration": state.get("iteration", 0),
                "next": state.get("next"),
            }
        checkpoints = []
        if self.root.exists():
            for path in sorted(self.root.glob("*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    state = payload.get("state", {})
                except (OSError, json.JSONDecodeError):
                    continue
                checkpoints.append(
                    {
                        "thread_id": payload.get("thread_id") or state.get("thread_id") or path.stem,
                        "path": str(path),
                        "status": state.get("status"),
                        "iteration": state.get("iteration", 0),
                        "next": state.get("next"),
                    }
                )
        return {"schema_version": CHECKPOINT_SCHEMA_VERSION, "checkpoints": checkpoints}


def _jsonable_state(state: Mapping[str, Any]) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, FallbackEvent):
            return value.to_dict()
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return value.to_dict()
        if isinstance(value, Mapping):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, tuple | list):
            return [convert(item) for item in value]
        if isinstance(value, str | int | float | bool) or value is None:
            return value
        return str(value)

    return {str(key): convert(value) for key, value in state.items() if not callable(value)}


def _ensure_state(state: Mapping[str, Any] | None = None, *, query: str | None = None) -> ResearchState:
    current: ResearchState = dict(state or {})
    if query is not None:
        current["query"] = query
    current.setdefault("thread_id", str(uuid4()))
    current.setdefault("status", "running")
    current.setdefault("iteration", 0)
    current.setdefault("max_iterations", DEFAULT_MAX_ITERATIONS)
    current.setdefault("min_evidence_records", DEFAULT_MIN_EVIDENCE_RECORDS)
    current.setdefault("evidence", [])
    current.setdefault("failures", [])
    current.setdefault("fallback_events", [])
    current.setdefault("messages", [])
    current.setdefault("review_required", False)
    current.setdefault("review_approved", False)
    return current


def main_node(state: ResearchState) -> ResearchState:
    """Initialize state for the top-level main graph."""

    current = _ensure_state(state)
    current["status"] = "running"
    current["messages"].append({"node": "main", "event": "started"})
    return current


def route_after_supervisor(state: ResearchState) -> RouteName:
    """Route to researcher until evidence is sufficient or max iterations hit."""

    evidence_count = len(state.get("evidence", []))
    min_evidence = int(state.get("min_evidence_records", DEFAULT_MIN_EVIDENCE_RECORDS))
    iteration = int(state.get("iteration", 0))
    max_iterations = int(state.get("max_iterations", DEFAULT_MAX_ITERATIONS))
    if evidence_count >= min_evidence:
        return "review" if state.get("review_required") and not state.get("review_approved") else "finish"
    if iteration >= max_iterations:
        return "review" if state.get("review_required") and not state.get("review_approved") else "finish"
    return "researcher"


def supervisor_node(state: ResearchState) -> ResearchState:
    """Decide whether to delegate research, interrupt for review, or finish."""

    current: ResearchState = dict(state)
    route = route_after_supervisor(current)
    current["next"] = route
    current["sufficient"] = route != "researcher"
    current.setdefault("messages", []).append(
        {
            "node": "supervisor",
            "event": "route",
            "next": route,
            "iteration": current.get("iteration", 0),
            "evidence_count": len(current.get("evidence", [])),
        }
    )
    return current


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def researcher_node(
    state: ResearchState,
    *,
    search: SearchCallable | None = None,
    model_client: ModelClient | None = None,
    config: AppConfig | None = None,
) -> ResearchState:
    """Run one researcher iteration and append evidence/fallback metadata."""

    current: ResearchState = dict(state)
    current.setdefault("evidence", [])
    current.setdefault("failures", [])
    current.setdefault("fallback_events", [])
    current.setdefault("messages", [])
    iteration = int(current.get("iteration", 0)) + 1
    current["iteration"] = iteration
    query = str(current.get("query", ""))
    max_results = int(current.get("max_results", 5))

    if search is None:
        current["failures"].append(
            SearchFailure(
                query=query,
                error_class="SearchNotConfigured",
                error_message="No search callable was provided for the local workflow.",
            ).to_dict()
        )
    else:
        try:
            results = await _maybe_await(search(query, max_results))
            current["evidence"].extend(
                record.to_dict() for record in normalize_search_results(query, results)
            )
        except Exception as exc:  # pragma: no cover - defensive local workflow boundary
            current["failures"].append(
                SearchFailure(
                    query=query,
                    error_class=type(exc).__name__,
                    error_message=str(exc),
                ).to_dict()
            )

    client = build_model_client(config) if model_client is None else model_client
    response = await client.invoke(
        ModelRequest(
            node="researcher",
            prompt=query,
            metadata={
                "thread_id": current.get("thread_id", ""),
                "iteration": iteration,
                "evidence_count": len(current.get("evidence", [])),
            },
        )
    )
    current["model_metadata"] = response.to_dict()
    current["fallback_events"].extend(event.to_dict() for event in response.fallback_events)
    current["messages"].append(
        {
            "node": "researcher",
            "event": "iteration_complete",
            "iteration": iteration,
            "evidence_count": len(current.get("evidence", [])),
            "fallback_event_count": len(response.fallback_events),
        }
    )
    return current


def review_node(state: ResearchState) -> ResearchState:
    """Interrupt before finalization unless review was explicitly approved."""

    current: ResearchState = dict(state)
    if current.get("review_required") and not current.get("review_approved"):
        current["status"] = "interrupted"
        current["review_interrupt"] = {
            "reason": "review_required",
            "thread_id": current.get("thread_id", ""),
            "iteration": current.get("iteration", 0),
            "evidence_count": len(current.get("evidence", [])),
        }
        current.setdefault("messages", []).append({"node": "review", "event": "interrupted"})
        return current
    return finish_node(current)


def finish_node(state: ResearchState) -> ResearchState:
    """Finalize a workflow state."""

    current: ResearchState = dict(state)
    current["status"] = "completed"
    if current.get("evidence"):
        current["answer"] = f"Collected {len(current['evidence'])} evidence record(s)."
    elif current.get("failures"):
        current["answer"] = "No evidence collected; see failures for local execution details."
    else:
        current["answer"] = "Research workflow completed without evidence."
    current.setdefault("messages", []).append({"node": "finish", "event": "completed"})
    return current


class LocalResearchWorkflow:
    """Executable local workflow with durable checkpoint save/load semantics."""

    def __init__(
        self,
        *,
        checkpoint_store: LocalCheckpointStore | None = None,
        model_client: ModelClient | None = None,
        config: AppConfig | None = None,
    ):
        self.checkpoint_store = checkpoint_store or LocalCheckpointStore()
        self.model_client = model_client
        self.config = config

    async def arun(
        self,
        state: Mapping[str, Any] | None = None,
        *,
        query: str | None = None,
        thread_id: str | None = None,
        search: SearchCallable | None = None,
        require_review: bool | None = None,
        approve_review: bool = False,
        max_iterations: int | None = None,
        min_evidence_records: int | None = None,
        raise_on_interrupt: bool = False,
    ) -> ResearchState:
        current = _ensure_state(state, query=query)
        if thread_id is not None:
            current["thread_id"] = thread_id
        if require_review is not None:
            current["review_required"] = require_review
        if approve_review:
            current["review_approved"] = True
            current.pop("review_interrupt", None)
        if max_iterations is not None:
            current["max_iterations"] = max_iterations
        if min_evidence_records is not None:
            current["min_evidence_records"] = min_evidence_records

        current = main_node(current)
        while True:
            current = supervisor_node(current)
            self.checkpoint_store.save(current)
            route = current.get("next")
            if route == "researcher":
                current = await researcher_node(
                    current,
                    search=search,
                    model_client=self.model_client,
                    config=self.config,
                )
                self.checkpoint_store.save(current)
                continue
            if route == "review":
                current = review_node(current)
                self.checkpoint_store.save(current)
                if current.get("status") == "interrupted" and raise_on_interrupt:
                    raise ReviewInterrupt(current)
                return current
            current = finish_node(current)
            self.checkpoint_store.save(current)
            return current

    async def aresume(
        self,
        thread_id: str,
        *,
        approve_review: bool = False,
        search: SearchCallable | None = None,
        raise_on_interrupt: bool = False,
    ) -> ResearchState:
        """Resume a durable thread checkpoint."""

        state = self.checkpoint_store.load(thread_id)
        if state.get("status") == "interrupted" and not approve_review:
            if raise_on_interrupt:
                raise ReviewInterrupt(state)
            return state
        return await self.arun(
            state,
            thread_id=thread_id,
            search=search,
            approve_review=approve_review,
            raise_on_interrupt=raise_on_interrupt,
        )

    def run(self, *args: Any, **kwargs: Any) -> ResearchState:
        """Synchronous wrapper for local scripts/tests."""

        return asyncio.run(self.arun(*args, **kwargs))

    def resume(self, *args: Any, **kwargs: Any) -> ResearchState:
        """Synchronous resume wrapper for local scripts/tests."""

        return asyncio.run(self.aresume(*args, **kwargs))


class LocalCompiledGraph:
    """Small ``invoke``/``ainvoke`` facade used when LangGraph is unavailable."""

    def __init__(self, workflow: LocalResearchWorkflow | None = None):
        self.workflow = workflow or LocalResearchWorkflow()
        self.topology = deepcopy(GRAPH_TOPOLOGY)

    async def ainvoke(self, state: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> ResearchState:
        configurable = dict((config or {}).get("configurable", {})) if config else {}
        thread_id = configurable.get("thread_id") or state.get("thread_id")
        return await self.workflow.arun(state, thread_id=thread_id)

    def invoke(self, state: Mapping[str, Any], config: Mapping[str, Any] | None = None) -> ResearchState:
        return asyncio.run(self.ainvoke(state, config=config))

    def get_graph(self) -> dict[str, tuple[str, ...]]:
        return deepcopy(self.topology)


def _build_graph() -> Any:
    """Build a LangGraph graph when available, otherwise a local facade."""

    try:
        from langgraph.graph import END, StateGraph  # type: ignore[import-not-found]
    except ImportError:
        return LocalCompiledGraph()

    async def run_researcher(state: ResearchState) -> ResearchState:
        return await researcher_node(state)

    builder = StateGraph(ResearchState)
    builder.add_node("main", main_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", run_researcher)
    builder.add_node("review", review_node)
    builder.add_node("finish", finish_node)
    builder.set_entry_point("main")
    builder.add_edge("main", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next", "finish"),
        {"researcher": "researcher", "review": "review", "finish": "finish"},
    )
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("review", END)
    builder.add_edge("finish", END)
    return builder.compile()


async def run_research(
    query: str,
    *,
    thread_id: str | None = None,
    checkpoint_dir: str | Path | None = None,
    require_review: bool = False,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    min_evidence_records: int = DEFAULT_MIN_EVIDENCE_RECORDS,
    search: SearchCallable | None = None,
) -> ResearchState:
    """Run a local research thread and persist checkpoints."""

    workflow = LocalResearchWorkflow(checkpoint_store=LocalCheckpointStore(checkpoint_dir))
    return await workflow.arun(
        query=query,
        thread_id=thread_id,
        require_review=require_review,
        max_iterations=max_iterations,
        min_evidence_records=min_evidence_records,
        search=search,
    )


async def resume_research(
    thread_id: str,
    *,
    checkpoint_dir: str | Path | None = None,
    approve_review: bool = False,
    search: SearchCallable | None = None,
) -> ResearchState:
    """Resume a local research thread by id."""

    workflow = LocalResearchWorkflow(checkpoint_store=LocalCheckpointStore(checkpoint_dir))
    return await workflow.aresume(thread_id, approve_review=approve_review, search=search)


def inspect_checkpoints(
    thread_id: str | None = None, *, checkpoint_dir: str | Path | None = None
) -> dict[str, Any]:
    """Inspect local durable checkpoint metadata."""

    return LocalCheckpointStore(checkpoint_dir).inspect(thread_id)


graph = _build_graph()
