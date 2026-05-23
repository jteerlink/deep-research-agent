"""Compatibility entry point for the G003 research workflow graph.

Historically this module exposed ``graph``. Keep that import path stable while
using :mod:`deep_research_agent.graph` as the canonical implementation.
"""

from __future__ import annotations

from .graph import (
    CHECKPOINT_SCHEMA_VERSION,
    GRAPH_TOPOLOGY,
    DEFAULT_MAX_RESEARCH_ITERATIONS,
    GraphRunResult,
    LocalCheckpointStore,
    LocalResearchGraph,
    ModelFallbackEvent,
    ResearchState,
    build_graph,
    graph,
    inspect_thread,
    invoke_maybe_async,
    resume_thread,
    run_query,
)

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "GRAPH_TOPOLOGY",
    "DEFAULT_MAX_RESEARCH_ITERATIONS",
    "GraphRunResult",
    "LocalCheckpointStore",
    "LocalResearchGraph",
    "ModelFallbackEvent",
    "ResearchState",
    "build_graph",
    "graph",
    "inspect_thread",
    "invoke_maybe_async",
    "resume_thread",
    "run_query",
]
