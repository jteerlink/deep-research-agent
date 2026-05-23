"""LangGraph entry point for the deep research agent foundation.

This file exists at the src/ path declared by langgraph.json. G001 keeps the
graph import-safe and minimal; later stories replace this with the nested
main/supervisor/researcher topology from the approved PRD.
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
