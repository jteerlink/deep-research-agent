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
python -m deep_research_agent search-providers
```

See [docs/usage.md](docs/usage.md) for offline smoke, optional live smoke,
checkpoint/resume, and artifact examples.

Optional local UI:

```bash
python -m pip install -e '.[ui]'
deep-research-agent ui
```

The UI reads provider keys from `.env` or already-exported shell environment
variables; it does not ask for secrets in the browser. Fill industry/niche,
geography, and research criteria so runs search for potential business targets
rather than generic articles about a topic.

## Tiered prospect preview

Use the offline `tiered-preview` command to verify how an industry/geography
directive expands into company, contact, and personalization search tiers before
running any live provider calls:

```bash
python -m deep_research_agent tiered-preview \
  --industry "dental" \
  --geography "DFW area" \
  --criteria "multi-location practices with reactivation opportunity" \
  --preferred-contact-role owner \
  --json
```

The preview does not perform network search, browser capture, enrichment, or
outreach. Early tiered discovery is intentionally wired around injected search
callables so it does not instantiate the legacy `AsyncMultiProviderSearch()`
default provider chain; Exa remains reserved for later approved enrichment.

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
  --artifact-dir artifacts/demo \
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

The configuration intentionally uses Ollama Cloud for Ollama model calls. Local
Ollama endpoints such as `localhost:11434` are not used as defaults or accepted
as available model transports.

- `ollama_native`: uses `OLLAMA_NATIVE_BASE_URL=https://ollama.com/api`,
  `OLLAMA_NATIVE_MODEL`, and `OLLAMA_API_KEY`.
- `ollama_openai`: uses `DRA_OLLAMA_OPENAI_BASE_URL`,
  `DRA_OLLAMA_OPENAI_MODEL`, and `DRA_OLLAMA_OPENAI_API_KEY` only for a
  non-local hosted OpenAI-compatible endpoint.
- `openai`: uses `OPENAI_API_KEY`, `OPENAI_MODEL`, and `OPENAI_BASE_URL`.
- `codex`: uses `CODEX_API_KEY`, `CODEX_MODEL`, and `CODEX_BASE_URL`.

Search provider keys remain compatible with the standalone module:
`TAVILY_API_KEY`, `EXA_API_KEY`, `SERPER_API_KEY`, `FIRECRAWL_API_KEY`, and
`YDC_API_KEY`. Provider priority is `tavily`, `exa`, `serper`, `firecrawl`,
`ydc`, then keyless `duckduckgo`. If a corporate proxy causes certificate
verification failures, set `DEEP_RESEARCH_CA_BUNDLE` to a PEM bundle; on macOS,
the search client auto-generates a local bundle from the system keychain when no
explicit bundle is configured.

Prospect extraction first applies broad deterministic triage to reject obvious
directories, aggregators, social/job pages, listicles, reviews, and vendor-noise
results. Plausible owned-domain candidates then go through structured LLM
judgment when a configured provider is available, with deterministic fallback for
offline runs.

Use `deep-research-agent model-status` before long prospect runs to verify the
redacted live-model configuration. `--live-smoke` performs an optional structured
JSON call against the first available hosted provider. Add `--require-live-model`
to `run` or `resume` when export-qualified prospects are required; without it,
offline or unavailable model paths remain review-only and deterministic fallback
will not satisfy export qualification.
