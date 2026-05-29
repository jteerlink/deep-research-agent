# Usage: local prospect deep research workflow

This repo is a local-first, lean tiered prospect research agent. The purpose is
company prospect identification: ranked target accounts, official company
contact points, fit rationale, personalized angles, and exportable local
artifacts.

The current implementation is intentionally mock-first/offline-safe. It wires
the durable workflow, search/evidence contracts, fallback metadata, and artifact
writers without requiring live Ollama, OpenAI/Codex, or paid search credentials.

## Install and inspect

```bash
python -m pip install -e '.[dev]'
cp .env.example .env
python -m deep_research_agent --help
python -m deep_research_agent config --json  # API keys are redacted by default
python -m deep_research_agent search-providers
```

`async_multi_search.py` remains the web-search foundation. Search provider keys
are optional for offline tests and live smoke can use whichever provider is
configured first.

Provider priority is:

```text
tavily -> exa -> serper -> firecrawl -> ydc -> duckduckgo
```

Set `FIRECRAWL_API_KEY` to enable Firecrawl v2 `/search`. Set `YDC_API_KEY` to
enable You.com Developer Cloud Search. These integrations use web search result
descriptions/snippets only; they do not request scraped markdown or page
content.

If live search fails with `CERTIFICATE_VERIFY_FAILED` behind a corporate proxy,
set `DEEP_RESEARCH_CA_BUNDLE` to a PEM bundle. On macOS, the search client will
try to generate a local CA bundle from the system keychain when no explicit
bundle is configured.

## Optional local UI

Install and launch the Streamlit UI:

```bash
python -m pip install -e '.[ui]'
deep-research-agent ui
```

The UI supports target industry/niche, geography, research criteria, target
prospect count, tiered checkpoint/artifact directories, search max
results/timeout, provider API key status from `.env` or shell environment,
tiered run/resume/inspect buttons, a progress timeline, warnings,
evidence/prospect previews, candidate review audit data, artifact paths,
markdown preview, and raw JSON. Provider keys are not typed into the UI.

For live prospect discovery, fill at least industry/niche and geography. A broad
query such as `AI agency lead reactivation targets` tends to find vendor pages or
articles; a directive such as `dental practices`, `Dallas-Fort Worth`, and
`patient reactivation opportunity` produces company-discovery searches and
returns potential business targets.

Known ambiguous geographies are normalized before search. For example,
`North Texas` keeps the original phrase and adds targeted terms such as
`Dallas-Fort Worth TX`, `DFW`, `Dallas TX`, and `Fort Worth TX` to the generated
search queries. The built-in map also covers common large-market shorthands such
as `Bay Area`, `DMV`, `South Florida`, `Chicagoland`, `Greater Atlanta`,
`Greater Houston`, `Puget Sound`, `Twin Cities`, `Research Triangle`, and
`Inland Empire`. Unknown broad regions are searched as entered, surfaced as
warnings, and included as `geography_alias_suggestion` in the run output so a
reviewed alias-map update can be added later without guessing silently.

Prospect extraction uses broad deterministic triage before model judgment. The
triage layer rejects obvious directories, aggregators, social/job pages,
listicles, review pages, and vendor-noise results; plausible owned-domain
candidates are kept review-gated until a human approves the final-enrichment
selection.

## Model configuration

The root package accepts current `DEEP_RESEARCH_*` variables plus older `DRA_*`
aliases for compatibility:

- Root package/CLI compatibility variables: `DEEP_RESEARCH_MODEL_PROVIDER`,
  `DEEP_RESEARCH_FALLBACK_ORDER`, `OLLAMA_NATIVE_BASE_URL`,
  `OLLAMA_NATIVE_MODEL`, `OLLAMA_API_KEY`, `OLLAMA_OPENAI_BASE_URL`,
  `OLLAMA_OPENAI_MODEL`, `OPENAI_API_KEY`, `CODEX_OPENAI_MODEL`,
  `DRA_PRIMARY_PROVIDER`,
  `DRA_OLLAMA_BASE_URL`, `DRA_OLLAMA_MODEL`, `DRA_OLLAMA_OPENAI_BASE_URL`,
  `DRA_OLLAMA_OPENAI_MODEL`, `DRA_OLLAMA_OPENAI_API_KEY`, `OPENAI_API_KEY`,
  `OPENAI_MODEL`, `CODEX_API_KEY`, and `CODEX_MODEL`.

For the intended Ollama Cloud DeepSeek Pro path, set:

```bash
DRA_PRIMARY_PROVIDER=ollama_native
DRA_OLLAMA_BASE_URL=https://ollama.com/api
DRA_OLLAMA_MODEL=deepseek-v4-pro:cloud
OLLAMA_NATIVE_BASE_URL=https://ollama.com/api
OLLAMA_NATIVE_MODEL=deepseek-v4-pro:cloud
OLLAMA_API_KEY=<your-ollama-cloud-key>
```

Keep native Ollama and OpenAI-compatible Ollama settings separate. Do not infer
`/v1` settings from the native `/api` base URL. Local Ollama endpoints are not
used by this app; direct Ollama model calls should go through Ollama Cloud with
`OLLAMA_API_KEY`.

## Offline workflow smoke

Use `--mock-result` to exercise tiered local checkpointing, review interrupts,
and resume without live network calls:

```bash
CHECKPOINT_DIR=$(mktemp -d)
ARTIFACT_DIR=$(mktemp -d)
python -m deep_research_agent tiered-run \
  --industry "dental practices" \
  --geography "Dallas-Fort Worth" \
  --criteria "patient reactivation opportunity" \
  --thread-id demo-tiered-thread \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --artifact-dir "$ARTIFACT_DIR" \
  --mock-result "Acme Dental|https://example.com/acme|Growing DSO with reactivation need|duckduckgo" \
  --json

python -m deep_research_agent tiered-inspect \
  demo-tiered-thread \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --json

python -m deep_research_agent tiered-resume \
  demo-tiered-thread \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --artifact-dir "$ARTIFACT_DIR" \
  --json
```

Expected results:

- `tiered-run` returns JSON with `status: "review_required"`, a stable
  `thread_id`, evidence from the mocked search result, companies, contact-point
  placeholders when contact info is still needed, and artifact paths.
- Live runs expand the directive into multiple company-discovery searches and
  stop when enough deduplicated potential business targets are found or when the
  iteration limit is reached.
- `tiered-resume` without approval preserves the review-required state; with an
  approval JSON and enabled final enrichment, it writes the approved final
  enrichment artifacts.
- `tiered-inspect` returns the persisted checkpoint state from the same local
  JSON checkpoint file.

The default checkpoint directory is `.deep_research_agent/tiered_checkpoints`;
pass `--checkpoint-dir` for tests or throwaway runs.

## Exportable artifacts

G002 artifact helpers write local JSON/CSV/Markdown evidence or prospect records:

```python
from deep_research_agent.artifacts import write_research_artifacts

records = [
    {
        "query": "AI agency lead reactivation targets",
        "title": "Acme Dental",
        "url": "https://example.com/acme",
        "evidence_type": "snippet",
        "provider": "duckduckgo",
        "citation_ids": ["cite-1"],
    }
]

write_research_artifacts(records, "artifacts/prospect-run", metadata={"thread_id": "demo"})
```

Snippet evidence must stay marked as `snippet`; only fetched/read page evidence
should be marked as `page_read`. Prospect fit rationale and personalized angles
should cite evidence IDs or be clearly marked as low-confidence human-review
inferences.

## Compatibility and migration boundaries

The supported workflow is the tiered command family:
`tiered-preview`, `tiered-run`, `tiered-inspect`, and `tiered-resume`.
The older generic `run`, `resume`, and `inspect` commands have been removed; if
called, the CLI exits non-zero with migration guidance instead of silently
reviving the deprecated workflow.

These compatibility surfaces intentionally remain because they protect current
operator and test workflows:

- `async_multi_search.py` and package `deep_research_agent.search` preserve the
  existing search import path.
- Config aliases such as `DRA_PRIMARY_PROVIDER` remain lower-priority shims for
  older local `.env` files.
- Model fallback metadata remains available for redacted diagnostics and
  offline-safe model status checks.
- Legacy contact-payload guards reject stale person-contact artifacts that lack
  the current company contact-point semantics, rather than migrating ambiguous
  data silently.

## Optional live smoke

Run this only after credentials are present in `.env` or exported in the shell:

```bash
python -m deep_research_agent config --json  # API keys are redacted by default
python -m async_multi_search "AI agency lead reactivation targets" --max-results 3
```

Then run a constrained CLI workflow with a real search result copied into
`--mock-result`, or call the package search wrappers from a small script.
Researcher heartbeat model calls remain metadata-only for offline compatibility;
structured prospect-judgment calls use the configured live provider when
credentials are available and fall back to deterministic judgment when they are
not. The CLI redacts configured API keys by default.

## Verification commands

```bash
python -m pytest -q
python -m ruff check .
python -m deep_research_agent --help
python -m deep_research_agent config --json  # API keys are redacted by default
```

These commands are the required offline gate before marking implementation work
complete.

## Non-goal audit

Current implementation intentionally does **not** add:

- hosted/custom web UI,
- full vendored upstream `open_deep_research` clone,
- broad model zoo,
- database or multi-user service,
- CRM or email-sending integrations.

Those remain later planning decisions only if they improve the prospect target
identification goal.
