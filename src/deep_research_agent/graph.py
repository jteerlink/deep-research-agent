"""LangGraph dev-server entry point for the deep research agent.

The packaged implementation lives in ``deep_research_agent.graph``. This src
path is retained because ``langgraph.json`` points here for local LangGraph
dev-server compatibility.
"""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deep_research_agent.graph import (  # noqa: E402,F401
    CHECKPOINT_SCHEMA_VERSION,
    DEFAULT_MAX_ITERATIONS,
    GRAPH_TOPOLOGY,
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
