"""LangGraph dev-server entry point for the packaged G003 workflow graph.

``langgraph.json`` points at this src-path file, while setuptools currently
packages ``deep_research_agent`` from the repository root. Keep this module as a
thin shim so local LangGraph dev-server behavior and installed package imports
use the same graph implementation.
"""

from __future__ import annotations

from deep_research_agent.graph import *  # noqa: F403 - intentional compatibility re-export

# The src LangGraph dev entrypoint preserves the upstream-style main graph
# shape while reusing the packaged local graph object.
GRAPH_TOPOLOGY = dict(GRAPH_TOPOLOGY)  # noqa: F405
GRAPH_TOPOLOGY["main"] = ("supervisor",)
