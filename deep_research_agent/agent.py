"""Backward-compatible LangGraph entry point for deep_research_agent.

G003 moved the executable topology to :mod:`deep_research_agent.graph`; this
module preserves the earlier ``deep_research_agent.agent:graph`` import surface.
"""

from __future__ import annotations

from .graph import (  # noqa: F401
    GRAPH_TOPOLOGY,
    CHECKPOINT_SCHEMA_VERSION,
    DEFAULT_MAX_ITERATIONS,
    LocalCheckpointStore,
    LocalCompiledGraph,
    LocalResearchWorkflow,
    ResearchState,
    ReviewInterrupt,
    graph,
    inspect_checkpoints,
    main_node,
    researcher_node,
    resume_research,
    review_node,
    route_after_supervisor,
    run_research,
    supervisor_node,
)

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "DEFAULT_MAX_ITERATIONS",
    "GRAPH_TOPOLOGY",
    "LocalCheckpointStore",
    "LocalCompiledGraph",
    "LocalResearchWorkflow",
    "ResearchState",
    "ReviewInterrupt",
    "graph",
    "inspect_checkpoints",
    "main_node",
    "researcher_node",
    "resume_research",
    "review_node",
    "route_after_supervisor",
    "run_research",
    "supervisor_node",
]
