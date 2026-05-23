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
from .search import SearchResult, web_search

__all__ = [
    "AppConfig",
    "CodexConfig",
    "ModelProvider",
    "OllamaNativeConfig",
    "OllamaOpenAIConfig",
    "OpenAIConfig",
    "SearchResult",
    "load_config",
    "web_search",
]
