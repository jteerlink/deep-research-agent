"""Foundation package for the local-first deep research agent."""

from .configuration import AgentConfig, ModelProvider, load_config
from .models import FallbackEvent, ModelRequest, ModelResponse, build_model_client

__all__ = [
    "AgentConfig",
    "FallbackEvent",
    "ModelProvider",
    "ModelRequest",
    "ModelResponse",
    "build_model_client",
    "load_config",
]
