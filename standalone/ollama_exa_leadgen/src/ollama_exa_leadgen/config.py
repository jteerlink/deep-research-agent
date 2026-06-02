"""Isolated environment/config handling for standalone leadgen."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .contracts import DEFAULT_TARGET_COUNT

DEFAULT_OLLAMA_BASE_URL = "https://ollama.com/api"
DEFAULT_OLLAMA_MODEL = "deepseek-v4-pro:cloud"


@dataclass(frozen=True)
class LeadgenConfig:
    """Runtime configuration with secrets kept out of artifacts."""

    exa_api_key: str = ""
    ollama_api_key: str = ""
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    output_dir: str = "runs"
    default_target_count: int = DEFAULT_TARGET_COUNT
    max_exa_calls: int = 30
    wave_size: int = 4
    timeout_seconds: int = 120
    ca_bundle: str = ""

    @classmethod
    def from_env(cls, *, env_file: str | Path | None = ".env") -> LeadgenConfig:
        env = dict(os.environ)
        if env_file:
            env.update(_read_env_file(Path(env_file)))
        return cls(
            exa_api_key=env.get("EXA_API_KEY", ""),
            ollama_api_key=env.get("OLLAMA_API_KEY", ""),
            ollama_base_url=env.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/"),
            ollama_model=env.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
            output_dir=env.get("LEADGEN_OUTPUT_DIR", "runs"),
            default_target_count=_int_env(
                env.get("LEADGEN_DEFAULT_TARGET_COUNT"),
                DEFAULT_TARGET_COUNT,
            ),
            max_exa_calls=_int_env(env.get("LEADGEN_MAX_EXA_CALLS"), 30),
            wave_size=_int_env(env.get("LEADGEN_WAVE_SIZE"), 4),
            timeout_seconds=_int_env(env.get("LEADGEN_TIMEOUT_SECONDS"), 120),
            ca_bundle=(
                env.get("LEADGEN_CA_BUNDLE")
                or env.get("SSL_CERT_FILE")
                or env.get("REQUESTS_CA_BUNDLE")
                or ""
            ),
        )

    def redacted(self) -> dict[str, object]:
        return {
            "EXA_API_KEY_present": bool(self.exa_api_key),
            "OLLAMA_API_KEY_present": bool(self.ollama_api_key),
            "ollama_base_url": self.ollama_base_url,
            "ollama_model": self.ollama_model,
            "output_dir": self.output_dir,
            "default_target_count": self.default_target_count,
            "max_exa_calls": self.max_exa_calls,
            "wave_size": self.wave_size,
            "timeout_seconds": self.timeout_seconds,
            "ca_bundle_configured": bool(self.ca_bundle),
        }


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    loaded: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            loaded[key] = value
    return loaded


def _int_env(value: str | None, default: int) -> int:
    try:
        parsed = int(value or "")
    except ValueError:
        return default
    return parsed if parsed > 0 else default
