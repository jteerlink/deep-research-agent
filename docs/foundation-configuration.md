# Foundation and configuration

This repository starts as a local-first deep research agent. The foundation
keeps configuration explicit and small so tiered prospect, model, and CLI work
can be added without a hosted UI, database service, or multi-user runtime.

## Local configuration files

- `.env.example` documents every supported environment variable and is safe to
  commit.
- `.env` is the developer-local copy and must stay untracked.
- Tiered checkpoints and artifacts are local JSON/CSV/Markdown files under
  `.deep_research_agent/` by default.

## Model provider split

Set `DEEP_RESEARCH_MODEL_PROVIDER` to one of these values:

| Provider | Purpose | Primary variables |
| --- | --- | --- |
| `ollama_native` | Direct calls to Ollama Cloud's native API. | `OLLAMA_NATIVE_BASE_URL`, `OLLAMA_NATIVE_MODEL`, `OLLAMA_API_KEY` |
| `ollama_openai` | Non-local hosted OpenAI-compatible Ollama surface, if explicitly provided. | `OLLAMA_OPENAI_BASE_URL`, `OLLAMA_OPENAI_MODEL`, `OLLAMA_OPENAI_API_KEY` |
| `openai` | Hosted OpenAI-compatible fallback. | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` |
| `codex` | Codex/OpenAI-compatible experimental fallback. | `CODEX_OPENAI_BASE_URL`, `CODEX_OPENAI_API_KEY`, `CODEX_OPENAI_MODEL` |

The native Ollama and OpenAI-compatible Ollama settings are intentionally
separate. Ollama defaults point at Cloud (`https://ollama.com/api`), and local
Ollama endpoints are not treated as available model transports.

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

The UI is a local operator surface over the same tiered workflow. It
reads provider API keys from `.env` or shell environment, captures target
industry/niche, geography, research criteria, and target prospect count, exposes
search max-results/timeout controls, and previews progress events, evidence,
prospects, candidate review decisions, artifact paths, markdown, and raw JSON
without asking for secrets in the browser.

## Tiered local workflow checkpoints

The tiered workflow keeps a local execution contract for development and tests:

- `tiered-preview` expands industry/geography criteria into query templates
  without network access.
- `tiered-run` writes a `tiered.prospect_checkpoint.v1` JSON checkpoint and
  review-ready artifacts.
- `tiered-inspect` loads the saved checkpoint by thread id.
- `tiered-resume` applies reviewer approval and optional final enrichment.

The default checkpoint directory is `.deep_research_agent/tiered_checkpoints`,
and each command accepts `--checkpoint-dir` for isolated test runs.

This remains a local-first development surface: no hosted UI, no model zoo, no
database service, and no multi-user runtime are introduced by checkpointing.
