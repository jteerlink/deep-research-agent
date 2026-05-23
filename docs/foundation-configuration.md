# Foundation and configuration

This repository starts as a local-first deep research agent. The G001 foundation
keeps configuration explicit and small so later graph, model, and CLI work can be
added without a hosted UI, database service, or multi-user runtime.

## Local configuration files

- `.env.example` documents every supported environment variable and is safe to
  commit.
- `.env` is the developer-local copy and must stay untracked.
- `langgraph.json` is the LangGraph dev-server entry point and points at the
  package graph module expected by the package skeleton.

## Model provider split

Set `DEEP_RESEARCH_MODEL_PROVIDER` to one of these values:

| Provider | Purpose | Primary variables |
| --- | --- | --- |
| `ollama_native` | Direct calls to Ollama's native API. | `OLLAMA_NATIVE_BASE_URL`, `OLLAMA_NATIVE_MODEL` |
| `ollama_openai` | Ollama's OpenAI-compatible `/v1` API. | `OLLAMA_OPENAI_BASE_URL`, `OLLAMA_OPENAI_MODEL`, `OLLAMA_OPENAI_API_KEY` |
| `openai` | Hosted OpenAI-compatible fallback. | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` |
| `codex` | Codex/OpenAI-compatible experimental fallback. | `CODEX_OPENAI_BASE_URL`, `CODEX_OPENAI_API_KEY`, `CODEX_OPENAI_MODEL` |

The native Ollama and OpenAI-compatible Ollama settings are intentionally
separate. Do not infer one from the other in documentation or code; callers can
choose the transport that matches their client.

## Search compatibility

`async_multi_search.py` remains the backwards-compatible import path for web
search utilities. New package modules may wrap it later, but existing code should
continue to support:

```python
from async_multi_search import web_search
```

## LangGraph entry point

`langgraph.json` declares the graph as:

```text
./src/deep_research_agent/graph.py:graph
```

Until the graph implementation grows beyond the foundation story, this entry
point should remain lightweight and import-safe. Avoid adding hosted UI, model
zoo, database, or multi-user service assumptions to the foundation layer.
