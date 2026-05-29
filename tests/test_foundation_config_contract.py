from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _parse_env_example(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key] = value
    return values


def test_env_example_documents_explicit_model_provider_split():
    env_example = ROOT / ".env.example"
    if not env_example.exists():
        pytest.skip(".env.example is created by the env/docs lane")

    values = _parse_env_example(env_example)

    assert values["DEEP_RESEARCH_MODEL_PROVIDER"] in {
        "ollama_native",
        "ollama_openai",
        "openai",
        "codex",
    }
    assert values["OLLAMA_NATIVE_BASE_URL"].rstrip("/") == "https://ollama.com/api"
    assert values["OLLAMA_OPENAI_BASE_URL"] == ""
    assert values["OPENAI_BASE_URL"].rstrip("/").endswith("/v1")
    assert "CODEX_OPENAI_BASE_URL" in values
    assert "CODEX_OPENAI_API_KEY" in values
    assert "CODEX_OPENAI_MODEL" in values

    assert values["OLLAMA_NATIVE_BASE_URL"] != values["OLLAMA_OPENAI_BASE_URL"]
    assert values["OLLAMA_OPENAI_MODEL"] == ""
    assert values["OLLAMA_OPENAI_API_KEY"] == ""


def test_legacy_langgraph_entrypoint_is_removed() -> None:
    assert not (ROOT / "langgraph.json").exists()
    assert not (ROOT / "src" / "deep_research_agent" / "graph.py").exists()
