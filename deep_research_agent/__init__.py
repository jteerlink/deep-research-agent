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
from .search import (
    Citation,
    CitationValidationError,
    EvidenceKind,
    EvidenceRecord,
    Prospect,
    SearchEvidenceArtifact,
    SearchResult,
    collect_search_evidence,
    web_search,
)

__all__ = [
    "AppConfig",
    "CodexConfig",
    "ModelProvider",
    "OllamaNativeConfig",
    "OllamaOpenAIConfig",
    "OpenAIConfig",
    "Citation",
    "CitationValidationError",
    "EvidenceKind",
    "EvidenceRecord",
    "Prospect",
    "SearchEvidenceArtifact",
    "SearchResult",
    "collect_search_evidence",
    "load_config",
    "web_search",
]
