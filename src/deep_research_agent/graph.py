"""LangGraph entry point for the deep research agent workflow.

The canonical development-server entry point in ``langgraph.json`` points at
this module.  LangGraph remains an optional dependency for local development,
so the exported ``graph`` is an import-safe local runner when LangGraph is not
installed.  The local runner mirrors the G003 topology closely enough for
tests, CLI smoke runs, and checkpoint/resume development without requiring a
hosted UI, database, or live model provider.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Literal, TypedDict

GraphNode = Literal["main", "supervisor", "researcher", "review", "end"]

DEFAULT_MAX_ITERATIONS = 2
DEFAULT_REQUIRED_NOTES = 2
DEFAULT_CHECKPOINT_DIR = ".deep_research_agent/checkpoints"


class ResearchState(TypedDict, total=False):
    """State passed through the nested local research workflow."""

    query: str
    thread_id: str
    answer: str
    current_node: GraphNode
    next_node: GraphNode
    visited_nodes: list[str]
    research_notes: list[str]
    iteration: int
    max_iterations: int
    required_notes: int
    sufficient: bool
    status: str
    interrupted: bool
    review_required: bool
    review_approved: bool
    thread_id: str
    fallback_events: list[dict[str, Any]]


def checkpoint_path(thread_id: str, checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR) -> Path:
    """Return the durable local checkpoint path for a thread id."""

    safe_thread_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in thread_id)
    return Path(checkpoint_dir) / f"{safe_thread_id}.json"


def save_checkpoint(
    state: ResearchState,
    *,
    thread_id: str,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
) -> Path:
    """Persist state for local thread-id resume flows."""

    path = checkpoint_path(thread_id, checkpoint_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_checkpoint(
    thread_id: str,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
) -> ResearchState:
    """Load a previously persisted local checkpoint."""

    path = checkpoint_path(thread_id, checkpoint_dir)
    return json.loads(path.read_text(encoding="utf-8"))


def _configurable(config: dict[str, Any] | None) -> dict[str, Any]:
    if not config:
        return {}
    value = config.get("configurable", {})
    return value if isinstance(value, dict) else {}


def _thread_id(state: ResearchState, config: dict[str, Any] | None) -> str:
    configured = _configurable(config).get("thread_id")
    if configured:
        return str(configured)
    return state.get("thread_id") or "local"


def _checkpoint_dir(config: dict[str, Any] | None) -> str | Path:
    configured = _configurable(config).get("checkpoint_dir")
    return configured if configured else DEFAULT_CHECKPOINT_DIR


def _visit(state: ResearchState, node: GraphNode) -> None:
    state["current_node"] = node
    state.setdefault("visited_nodes", []).append(node)


def _main_node(state: ResearchState) -> ResearchState:
    _visit(state, "main")
    state.setdefault("research_notes", [])
    state.setdefault("fallback_events", [])
    state["iteration"] = int(state.get("iteration", 0))
    state["max_iterations"] = int(state.get("max_iterations", DEFAULT_MAX_ITERATIONS))
    state["required_notes"] = int(state.get("required_notes", DEFAULT_REQUIRED_NOTES))
    state["status"] = "running"
    state["next_node"] = "supervisor"
    return state


def _supervisor_node(state: ResearchState) -> ResearchState:
    _visit(state, "supervisor")
    notes = state.get("research_notes", [])
    required_notes = int(state.get("required_notes", DEFAULT_REQUIRED_NOTES))
    max_iterations = int(state.get("max_iterations", DEFAULT_MAX_ITERATIONS))
    iteration = int(state.get("iteration", 0))
    sufficient = bool(state.get("sufficient")) or len(notes) >= required_notes
    state["sufficient"] = sufficient
    if sufficient:
        state["status"] = "completed"
        state["answer"] = state.get("answer") or f"Research complete for: {state.get('query', '')}"
        state["next_node"] = "end"
    elif iteration >= max_iterations:
        state["status"] = "completed"
        state["answer"] = (
            state.get("answer")
            or f"Research stopped after {max_iterations} iteration(s): {state.get('query', '')}"
        )
        state["next_node"] = "end"
    else:
        state["next_node"] = "researcher"
    return state


def _researcher_node(state: ResearchState) -> ResearchState:
    _visit(state, "researcher")
    iteration = int(state.get("iteration", 0)) + 1
    state["iteration"] = iteration
    query = state.get("query", "")
    state.setdefault("research_notes", []).append(f"mock research note {iteration}: {query}")
    state["next_node"] = "supervisor"
    return state


def _review_node(state: ResearchState) -> ResearchState:
    _visit(state, "review")
    if state.get("review_required") and not state.get("review_approved"):
        state["status"] = "interrupted"
        state["interrupted"] = True
        state["next_node"] = "review"
    else:
        state["interrupted"] = False
        state["next_node"] = "supervisor"
    return state


async def _fallback_graph(state: ResearchState) -> ResearchState:
    return await LocalResearchGraph().ainvoke(state)


class LocalResearchGraph:
    """Small local runner that mirrors the G003 graph contract without LangGraph."""

    node_names = ("main", "supervisor", "researcher", "review")

    async def ainvoke(
        self,
        state: ResearchState,
        config: dict[str, Any] | None = None,
    ) -> ResearchState:
        thread_id = _thread_id(state, config)
        checkpoint_dir = _checkpoint_dir(config)
        if state.get("resume"):
            resumed = load_checkpoint(thread_id, checkpoint_dir)
            resumed.update({key: value for key, value in state.items() if key != "resume"})
            state = resumed
        else:
            state = dict(state)
        state["thread_id"] = thread_id

        if state.get("review_required") and not state.get("review_approved"):
            _main_node(state)
            _review_node(state)
            save_checkpoint(state, thread_id=thread_id, checkpoint_dir=checkpoint_dir)
            return state

        _main_node(state)
        while state.get("next_node") != "end":
            next_node = state["next_node"]
            if next_node == "supervisor":
                _supervisor_node(state)
            elif next_node == "researcher":
                _researcher_node(state)
            elif next_node == "review":
                _review_node(state)
                if state.get("interrupted"):
                    break
            else:  # pragma: no cover - defensive guard for corrupted state.
                raise ValueError(f"unknown graph node: {next_node}")

        save_checkpoint(state, thread_id=thread_id, checkpoint_dir=checkpoint_dir)
        return state

    def invoke(
        self,
        state: ResearchState,
        config: dict[str, Any] | None = None,
    ) -> ResearchState:
        """Synchronous convenience wrapper matching compiled LangGraph shape."""

        return asyncio.run(self.ainvoke(state, config))

    def get_state(
        self,
        config: dict[str, Any] | None = None,
    ) -> ResearchState:
        """Return the last durable local checkpoint for a configured thread id."""

        thread_id = _thread_id({}, config)
        return load_checkpoint(thread_id, _checkpoint_dir(config))


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
