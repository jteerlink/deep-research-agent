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

Provider priority is `tavily`, `exa`, `serper`, `firecrawl`, `ydc`, then
keyless `duckduckgo`. Firecrawl is enabled with `FIRECRAWL_API_KEY`; You.com
Developer Cloud search is enabled with `YDC_API_KEY`. Both are normalized as
snippet evidence unless a later browser/page-read step captures page content.
For corporate proxy roots, set `DEEP_RESEARCH_CA_BUNDLE` to a PEM bundle. On
macOS, live search will otherwise try a cached bundle generated from the system
keychain.

## Local Streamlit UI

The optional `ui` extra installs Streamlit:

```bash
python -m pip install -e '.[ui]'
deep-research-agent ui
```

The UI is a local operator surface over the same checkpointed workflow. It
reads provider API keys from `.env` or shell environment, captures target
industry/niche, geography, research criteria, and target prospect count, exposes
search max-results/timeout controls, and previews progress events, evidence,
prospects, artifact paths, markdown, and raw JSON without asking for secrets in
the browser.


## G003 local workflow checkpoints

The G003 workflow keeps the LangGraph entry point import-safe while adding a
local nested execution contract for development and tests:

```text
main -> supervisor -> researcher -> supervisor/review
```

`python -m deep_research_agent run <query>` writes a JSON checkpoint containing
the thread id, supervisor delegation event, researcher iteration events, model
fallback metadata, sufficiency or max-iteration routing, and the review
interrupt. `resume <thread-id>` reloads that same local checkpoint and completes
the review gate; `inspect <thread-id>` prints the saved state. The default
checkpoint directory is `.deep_research_agent/checkpoints`, and each command
accepts `--checkpoint-dir` for isolated test runs.

This remains a local-first development surface: no hosted UI, no model zoo, no
database service, and no multi-user runtime are introduced by checkpointing.

## LangGraph entry point

`langgraph.json` declares the graph as:

```text
./src/deep_research_agent/graph.py:graph
```

Until the graph implementation grows beyond the foundation story, this entry
point should remain lightweight and import-safe. Avoid adding a hosted UI, no model zoo, database, or multi-user service assumptions to the foundation layer.
