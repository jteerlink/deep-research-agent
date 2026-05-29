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
from .geography import (
    GeoAliasSuggestion,
    GeoScope,
    normalize_geography,
    suggest_geography_alias_update,
)
from .models import (
    FallbackEvent,
    ModelPreflight,
    ModelPreflightError,
    ModelProviderStatus,
    ModelRequest,
    ModelResponse,
    build_model_client,
)
from .prospects import CitationValidationError, Prospect, ProspectCitation
from .search import FirecrawlProvider, SearchResult, YdcProvider, web_search
from .tiered_search import (
    DEFAULT_TARGET_PROSPECT_COUNT,
    MAX_TARGET_PROSPECT_COUNT,
    ProspectRunBudget,
    derive_prospect_run_budget,
)

__all__ = [
    "AppConfig",
    "CodexConfig",
    "ModelProvider",
    "FallbackEvent",
    "GeoAliasSuggestion",
    "GeoScope",
    "ModelPreflight",
    "ModelPreflightError",
    "ModelProviderStatus",
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
    "FirecrawlProvider",
    "YdcProvider",
    "build_artifact_payload",
    "build_model_client",
    "collect_search_evidence",
    "normalize_geography",
    "suggest_geography_alias_update",
    "DEFAULT_TARGET_PROSPECT_COUNT",
    "MAX_TARGET_PROSPECT_COUNT",
    "ProspectRunBudget",
    "derive_prospect_run_budget",
    "load_config",
    "web_search",
]
