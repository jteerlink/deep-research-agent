# Deep Research Agent

Foundation package for a local-first deep research agent. The initial skeleton
keeps the existing `async_multi_search.py` import path working while adding a
package entry point, environment-driven model configuration, and LangGraph
configuration for future graph execution.

## Quick start

```bash
cp .env.example .env
python -m deep_research_agent --help
```

## Provider configuration

The configuration intentionally separates Ollama's native HTTP API from its
OpenAI-compatible API surface:

- `ollama_native`: uses `DRA_OLLAMA_BASE_URL` and `DRA_OLLAMA_MODEL`.
- `ollama_openai`: uses `DRA_OLLAMA_OPENAI_BASE_URL`,
  `DRA_OLLAMA_OPENAI_MODEL`, and `DRA_OLLAMA_OPENAI_API_KEY`.
- `openai`: uses `OPENAI_API_KEY`, `OPENAI_MODEL`, and `OPENAI_BASE_URL`.
- `codex`: uses `CODEX_API_KEY`, `CODEX_MODEL`, and `CODEX_BASE_URL`.

Search provider keys remain compatible with the standalone module:
`TAVILY_API_KEY`, `EXA_API_KEY`, `BRAVE_API_KEY`, and `SERPER_API_KEY`.
