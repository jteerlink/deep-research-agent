"""Deep Research Agent package foundation."""

from .config import (
    AppConfig,
    CodexConfig,
    ModelProvider,
    OllamaNativeConfig,
    OllamaOpenAIConfig,
    OpenAIConfig,
    load_config,
)
from .artifacts import ARTIFACT_SCHEMA_VERSION, build_artifact_payload
from .evidence import EvidenceRecord, SearchEvidenceBatch, SearchFailure, collect_search_evidence
from .prospects import CitationValidationError, Prospect, ProspectCitation
from .search import SearchResult, web_search

__all__ = [
    "AppConfig",
    "CodexConfig",
    "ModelProvider",
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
    "collect_search_evidence",
    "load_config",
    "web_search",
]
