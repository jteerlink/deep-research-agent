"""LangGraph entry point for the deep research agent foundation.

The graph remains intentionally minimal for G001: it imports without requiring a
hosted UI, database, or a full upstream clone. If LangGraph is installed, the
exported `graph` is a compiled one-node graph. Otherwise it is an awaitable
fallback with the same simple echo contract so package imports still work.
"""

from __future__ import annotations

from typing import Any, TypedDict


class ResearchState(TypedDict, total=False):
    """Minimal state shape for future research graph expansion."""

    query: str
    answer: str


async def _fallback_graph(state: ResearchState) -> ResearchState:
    query = state.get("query", "")
    return {**state, "answer": f"Research graph not installed; received query: {query}"}


def _build_graph() -> Any:
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return _fallback_graph

    async def start(state: ResearchState) -> ResearchState:
        query = state.get("query", "")
        return {**state, "answer": f"Ready to research: {query}"}

    builder = StateGraph(ResearchState)
    builder.add_node("start", start)
    builder.set_entry_point("start")
    builder.add_edge("start", END)
    return builder.compile()


graph = _build_graph()
