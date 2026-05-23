"""Deep Research Agent package foundation."""

from .artifacts import ARTIFACT_SCHEMA_VERSION, build_artifact_payload
from .config import (
    AppConfig,
    CodexConfig,
    ModelProvider,
    OllamaNativeConfig,
    OllamaOpenAIConfig,
    OpenAIConfig,
    load_config,
)
from .evidence import EvidenceRecord, SearchEvidenceBatch, SearchFailure, collect_search_evidence

from .graph import (
    CHECKPOINT_SCHEMA_VERSION,
    GRAPH_TOPOLOGY,
    GraphRunResult,
    LocalCheckpointStore,
    LocalResearchGraph,
    build_graph,
    graph,
    inspect_thread,
    resume_thread,
    run_query,
)
from .prospects import CitationValidationError, Prospect, ProspectCitation
from .search import SearchResult, web_search

__all__ = [
    "AppConfig",
    "CodexConfig",
    "ModelProvider",
    "FallbackEvent",
    "ModelRequest",
    "ModelResponse",
    "OllamaNativeConfig",
    "OllamaOpenAIConfig",
    "OpenAIConfig",
    "ARTIFACT_SCHEMA_VERSION",
    "CitationValidationError",
    "EvidenceRecord",
    "Prospect",
    "ProspectCitation",
    "SearchEvidenceBatch",
    "SearchFailure",
    "SearchResult",
    "build_artifact_payload",
    "build_model_client",
    "collect_search_evidence",
    "CHECKPOINT_SCHEMA_VERSION",
    "GRAPH_TOPOLOGY",
    "GraphRunResult",
    "LocalCheckpointStore",
    "LocalResearchGraph",
    "build_graph",
    "graph",
    "inspect_thread",
    "resume_thread",
    "run_query",
    "load_config",
    "web_search",
]
