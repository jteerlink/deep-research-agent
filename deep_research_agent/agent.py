"""Nested research workflow and local checkpoint runner for G003.

The public ``graph`` export remains import-safe for the LangGraph dev server,
but the implementation below also provides a deterministic local runner used by
CLI tests when LangGraph or provider credentials are absent.  The local runner
models the intended topology:

``main -> supervisor -> researcher -> supervisor -> review_interrupt``

Checkpoints are JSON files keyed by thread id so a run can be inspected and
resumed without a hosted service or database.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict
from uuid import uuid4

MAX_RESEARCH_ITERATIONS = 2
DEFAULT_CHECKPOINT_DIR = Path(".deep_research_agent") / "checkpoints"

WorkflowStatus = Literal["needs_review", "complete"]
WorkflowEventType = Literal[
    "main_started",
    "supervisor_delegated",
    "researcher_iteration",
    "fallback_model_used",
    "sufficiency_routed",
    "max_iterations_routed",
    "review_interrupt",
    "review_resumed",
    "workflow_completed",
]


class ResearchState(TypedDict, total=False):
    """State shape shared by the import-safe graph and local checkpoint runner."""

    query: str
    thread_id: str
    answer: str
    status: WorkflowStatus
    iterations: int
    sufficient: bool
    review_required: bool
    review_decision: str
    events: list[dict[str, Any]]
    fallback_metadata: dict[str, Any]


@dataclass(frozen=True)
class WorkflowEvent:
    """A serializable workflow event recorded in thread checkpoints."""

    type: WorkflowEventType
    node: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ThreadCheckpoint:
    """Durable state for one local research workflow thread."""

    thread_id: str
    query: str
    status: WorkflowStatus
    iterations: int
    sufficient: bool
    review_required: bool
    answer: str
    events: tuple[WorkflowEvent, ...]
    fallback_metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["events"] = [asdict(event) for event in self.events]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ThreadCheckpoint:
        events = tuple(WorkflowEvent(**event) for event in payload.get("events", ()))
        return cls(
            thread_id=str(payload["thread_id"]),
            query=str(payload.get("query", "")),
            status=payload.get("status", "needs_review"),  # type: ignore[arg-type]
            iterations=int(payload.get("iterations", 0)),
            sufficient=bool(payload.get("sufficient", False)),
            review_required=bool(payload.get("review_required", False)),
            answer=str(payload.get("answer", "")),
            events=events,
            fallback_metadata=dict(payload.get("fallback_metadata") or {}),
            updated_at=str(payload.get("updated_at", "")),
        )


class LocalCheckpointStore:
    """JSON-file checkpoint store keyed by LangGraph-style thread id."""

    def __init__(self, root: str | Path = DEFAULT_CHECKPOINT_DIR) -> None:
        self.root = Path(root)

    def _path(self, thread_id: str) -> Path:
        safe_thread_id = thread_id.replace("/", "_")
        return self.root / f"{safe_thread_id}.json"

    def save(self, checkpoint: ThreadCheckpoint) -> ThreadCheckpoint:
        self.root.mkdir(parents=True, exist_ok=True)
        updated = replace(checkpoint, updated_at=datetime.now(UTC).isoformat())
        self._path(updated.thread_id).write_text(
            json.dumps(updated.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return updated

    def load(self, thread_id: str) -> ThreadCheckpoint:
        path = self._path(thread_id)
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint found for thread id {thread_id!r}")
        return ThreadCheckpoint.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _event(
    event_type: WorkflowEventType,
    node: str,
    message: str,
    **metadata: Any,
) -> WorkflowEvent:
    return WorkflowEvent(type=event_type, node=node, message=message, metadata=metadata)


def _fallback_metadata(reason: str = "local_mock") -> dict[str, Any]:
    return {
        "provider": "local_mock",
        "model": "deterministic-researcher",
        "reason": reason,
    }


def _research_answer(query: str, iteration: int) -> str:
    return f"Draft research answer for {query!r} after {iteration} iteration(s)."


def _build_checkpoint(
    *,
    query: str,
    thread_id: str,
    approve: bool,
    previous_events: tuple[WorkflowEvent, ...] = (),
) -> ThreadCheckpoint:
    events: list[WorkflowEvent] = list(previous_events)
    if previous_events:
        events.append(_event("review_resumed", "main", "Review decision accepted."))
    else:
        events.append(_event("main_started", "main", "Research workflow started."))

    events.append(
        _event(
            "supervisor_delegated",
            "supervisor",
            "Supervisor delegated the query to the researcher graph.",
            thread_id=thread_id,
        )
    )

    fallback = _fallback_metadata()
    sufficient = False
    answer = ""
    iterations = 0
    for iteration in range(1, MAX_RESEARCH_ITERATIONS + 1):
        iterations = iteration
        answer = _research_answer(query, iteration)
        events.append(
            _event(
                "fallback_model_used",
                "researcher",
                "No configured live model was required; used deterministic fallback metadata.",
                **fallback,
            )
        )
        events.append(
            _event(
                "researcher_iteration",
                "researcher",
                "Researcher produced a draft answer.",
                iteration=iteration,
            )
        )
        sufficient = bool(query.strip()) and iteration >= 1
        if sufficient:
            events.append(
                _event(
                    "sufficiency_routed",
                    "supervisor",
                    "Supervisor routed sufficient research to review.",
                    iteration=iteration,
                )
            )
            break

    if not sufficient:
        events.append(
            _event(
                "max_iterations_routed",
                "supervisor",
                "Supervisor stopped after the maximum researcher iterations.",
                max_iterations=MAX_RESEARCH_ITERATIONS,
            )
        )

    if not approve:
        events.append(
            _event(
                "review_interrupt",
                "review",
                "Human review interrupt reached; resume with approval to complete.",
            )
        )
        return ThreadCheckpoint(
            thread_id=thread_id,
            query=query,
            status="needs_review",
            iterations=iterations,
            sufficient=sufficient,
            review_required=True,
            answer=answer,
            events=tuple(events),
            fallback_metadata=fallback,
        )

    events.append(_event("workflow_completed", "main", "Approved research workflow completed."))
    return ThreadCheckpoint(
        thread_id=thread_id,
        query=query,
        status="complete",
        iterations=iterations,
        sufficient=sufficient,
        review_required=False,
        answer=answer,
        events=tuple(events),
        fallback_metadata=fallback,
    )


def run_research_workflow(
    query: str,
    *,
    thread_id: str | None = None,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    approve: bool = False,
) -> ThreadCheckpoint:
    """Run a local nested workflow and persist its checkpoint."""

    resolved_thread_id = thread_id or f"local-{uuid4()}"
    checkpoint = _build_checkpoint(query=query, thread_id=resolved_thread_id, approve=approve)
    return LocalCheckpointStore(checkpoint_dir).save(checkpoint)


def resume_research_workflow(
    thread_id: str,
    *,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    approve: bool = True,
) -> ThreadCheckpoint:
    """Load a thread checkpoint and continue from its review interrupt."""

    store = LocalCheckpointStore(checkpoint_dir)
    current = store.load(thread_id)
    if current.status == "complete":
        return current
    checkpoint = _build_checkpoint(
        query=current.query,
        thread_id=current.thread_id,
        approve=approve,
        previous_events=current.events,
    )
    return store.save(checkpoint)


def inspect_research_thread(
    thread_id: str,
    *,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
) -> ThreadCheckpoint:
    """Return the durable checkpoint for a workflow thread id."""

    return LocalCheckpointStore(checkpoint_dir).load(thread_id)


async def _fallback_graph(state: ResearchState) -> ResearchState:
    checkpoint = run_research_workflow(
        state.get("query", ""),
        thread_id=state.get("thread_id"),
        approve=bool(state.get("review_decision") == "approve"),
    )
    return {**state, **checkpoint.to_dict()}


def _build_graph() -> Any:
    try:
        from langgraph.graph import END, StateGraph  # type: ignore[import-not-found]
    except ImportError:
        return _fallback_graph

    async def main_node(state: ResearchState) -> ResearchState:
        events = [
            *_state_events(state),
            asdict(_event("main_started", "main", "Research workflow started.")),
        ]
        return {**state, "events": events}

    async def supervisor_node(state: ResearchState) -> ResearchState:
        events = [
            *_state_events(state),
            asdict(
                _event(
                    "supervisor_delegated",
                    "supervisor",
                    "Supervisor delegated the query to the researcher graph.",
                )
            ),
        ]
        return {**state, "events": events}

    async def researcher_node(state: ResearchState) -> ResearchState:
        iteration = int(state.get("iterations", 0)) + 1
        fallback = _fallback_metadata("langgraph_mock")
        events = [
            *_state_events(state),
            asdict(
                _event(
                    "fallback_model_used",
                    "researcher",
                    "Used deterministic fallback metadata for import-safe graph execution.",
                    **fallback,
                )
            ),
            asdict(
                _event(
                    "researcher_iteration",
                    "researcher",
                    "Researcher produced a draft answer.",
                    iteration=iteration,
                )
            ),
        ]
        return {
            **state,
            "iterations": iteration,
            "answer": _research_answer(state.get("query", ""), iteration),
            "sufficient": bool(state.get("query", "")),
            "fallback_metadata": fallback,
            "events": events,
        }

    async def review_node(state: ResearchState) -> ResearchState:
        approved = state.get("review_decision") == "approve"
        checkpoint = ThreadCheckpoint(
            thread_id=state.get("thread_id") or f"local-{uuid4()}",
            query=state.get("query", ""),
            status="complete" if approved else "needs_review",
            iterations=int(state.get("iterations", 0)),
            sufficient=bool(state.get("sufficient", False)),
            review_required=not approved,
            answer=state.get("answer", ""),
            events=tuple(WorkflowEvent(**event) for event in _state_events(state))
            + (
                _event("workflow_completed", "main", "Approved research workflow completed.")
                if approved
                else _event(
                    "review_interrupt",
                    "review",
                    "Human review interrupt reached; resume with approval to complete.",
                )
            ,),
            fallback_metadata=dict(state.get("fallback_metadata") or {}),
        )
        saved = LocalCheckpointStore().save(checkpoint)
        return {**state, **saved.to_dict()}

    builder = StateGraph(ResearchState)
    builder.add_node("main", main_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("review", review_node)
    builder.set_entry_point("main")
    builder.add_edge("main", "supervisor")
    builder.add_edge("supervisor", "researcher")
    builder.add_edge("researcher", "review")
    builder.add_edge("review", END)
    return builder.compile()


def _state_events(state: ResearchState) -> list[dict[str, Any]]:
    return list(state.get("events") or [])


def invoke_graph_sync(state: ResearchState) -> ResearchState:
    """Invoke the exported graph or async fallback from synchronous CLI code."""

    candidate = graph
    if hasattr(candidate, "invoke"):
        return candidate.invoke(state)  # type: ignore[no-any-return]
    if asyncio.iscoroutinefunction(candidate):
        return asyncio.run(candidate(state))
    result = candidate(state)
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return result


graph = _build_graph()
