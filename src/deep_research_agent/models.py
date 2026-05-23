"""Compatibility wrapper for packaged model metadata contracts."""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from deep_research_agent.models import (  # noqa: E402,F401
    ConfiguredModelClient,
    FallbackEvent,
    ModelClient,
    ModelRequest,
    ModelResponse,
    build_model_client,
)

__all__ = [
    "ConfiguredModelClient",
    "FallbackEvent",
    "ModelClient",
    "ModelRequest",
    "ModelResponse",
    "build_model_client",
]
