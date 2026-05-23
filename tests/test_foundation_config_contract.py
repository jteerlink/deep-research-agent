import json
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
        pytest.skip(".env.example is created by the env/langgraph/docs lane")

    values = _parse_env_example(env_example)

    assert values["DEEP_RESEARCH_MODEL_PROVIDER"] in {
        "ollama_native",
        "ollama_openai",
        "openai",
        "codex",
    }
    assert values["OLLAMA_NATIVE_BASE_URL"].rstrip("/").endswith(":11434")
    assert values["OLLAMA_OPENAI_BASE_URL"].rstrip("/").endswith(":11434/v1")
    assert values["OPENAI_BASE_URL"].rstrip("/").endswith("/v1")
    assert "CODEX_OPENAI_BASE_URL" in values
    assert "CODEX_OPENAI_API_KEY" in values
    assert "CODEX_OPENAI_MODEL" in values

    assert values["OLLAMA_NATIVE_BASE_URL"] != values["OLLAMA_OPENAI_BASE_URL"]
    assert values["OLLAMA_NATIVE_MODEL"] == values["OLLAMA_OPENAI_MODEL"]
    assert values["OLLAMA_OPENAI_API_KEY"]


def test_langgraph_config_points_to_package_graph_entrypoint():
    langgraph_config = ROOT / "langgraph.json"
    if not langgraph_config.exists():
        pytest.skip("langgraph.json is created by the env/langgraph/docs lane")

    config = json.loads(langgraph_config.read_text())

    assert config["dependencies"] == ["."]
    assert config["env"] == ".env"
    assert config["graphs"]["deep_research_agent"] == "./src/deep_research_agent/graph.py:graph"
