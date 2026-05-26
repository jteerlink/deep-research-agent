"""Compatibility entry point for the G003 research workflow graph.

Historically this module exposed ``graph``. Keep that import path stable while
using :mod:`deep_research_agent.graph` as the canonical implementation.
"""

from __future__ import annotations

from .graph import (
    CHECKPOINT_SCHEMA_VERSION,
    DEFAULT_MAX_RESEARCH_ITERATIONS,
    GRAPH_TOPOLOGY,
    GraphRunResult,
    LocalCheckpointStore,
    LocalResearchGraph,
    LocalResearchWorkflow,
    ModelFallbackEvent,
    ProspectRunBudget,
    ResearchState,
    build_graph,
    derive_prospect_run_budget,
    graph,
    inspect_checkpoints,
    inspect_research_thread,
    inspect_thread,
    invoke_maybe_async,
    resume_research,
    resume_research_workflow,
    resume_thread,
    route_after_supervisor,
    run_query,
    run_research,
    run_research_workflow,
)

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "GRAPH_TOPOLOGY",
    "DEFAULT_MAX_RESEARCH_ITERATIONS",
    "GraphRunResult",
    "LocalCheckpointStore",
    "LocalResearchGraph",
    "LocalResearchWorkflow",
    "ModelFallbackEvent",
    "ProspectRunBudget",
    "ResearchState",
    "build_graph",
    "derive_prospect_run_budget",
    "inspect_checkpoints",
    "inspect_research_thread",
    "graph",
    "inspect_thread",
    "resume_research",
    "resume_research_workflow",
    "invoke_maybe_async",
    "resume_thread",
    "route_after_supervisor",
    "run_research",
    "run_research_workflow",
    "run_query",
]
