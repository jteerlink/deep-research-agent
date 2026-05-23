# Deep Research Agent

Foundation package for a local-first deep research agent. The skeleton keeps the
existing `async_multi_search.py` import path working while adding a package entry
point, environment-driven model configuration, LangGraph configuration, and a
local G003 workflow runner with durable thread checkpoints.

## Quick start

```bash
cp .env.example .env
python -m deep_research_agent --help
python -m deep_research_agent config --json
```

See [docs/usage.md](docs/usage.md) for offline smoke, optional live smoke,
checkpoint/resume, and artifact examples.

## Local G003 workflow

The G003 runner models the nested workflow topology locally without a hosted UI
or database:

```text
main -> supervisor -> researcher -> supervisor/review
```

Start a run and stop at the review interrupt:

```bash
python -m deep_research_agent run "find AI agencies" --thread-id demo-thread
```

For a deterministic offline run with mocked search evidence:

```bash
python -m deep_research_agent run "find AI agencies" \
  --thread-id demo-thread \
  --require-review \
  --mock-result "Acme|https://example.com/acme|Acme builds reactivation tooling|duckduckgo"
```

Inspect the durable JSON checkpoint:

```bash
python -m deep_research_agent inspect demo-thread
```

Resume the same thread id and approve the review gate:

```bash
python -m deep_research_agent resume demo-thread
```

By default checkpoints are written under `.deep_research_agent/checkpoints`.
Use `--checkpoint-dir <path>` on `run`, `resume`, or `inspect` for test or
scratch directories.

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
