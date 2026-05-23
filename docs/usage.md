# Usage: local prospect deep research workflow

This repo is a local-first, lean LangGraph/open_deep_research-style prospect
research agent. The first-pass purpose is prospect target identification:
ranked target accounts, decision-maker leads, fit rationale, personalized
angles, and exportable local artifacts.

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

## Model configuration

There are two compatibility configuration surfaces while the root package and
`src/` LangGraph shim converge:

- Root package/CLI compatibility variables: `DRA_PRIMARY_PROVIDER`,
  `DRA_OLLAMA_BASE_URL`, `DRA_OLLAMA_MODEL`, `DRA_OLLAMA_OPENAI_BASE_URL`,
  `DRA_OLLAMA_OPENAI_MODEL`, `DRA_OLLAMA_OPENAI_API_KEY`, `OPENAI_API_KEY`,
  `OPENAI_MODEL`, `CODEX_API_KEY`, and `CODEX_MODEL`.
- `src/` LangGraph shim variables: `DEEP_RESEARCH_MODEL_PROVIDER`,
  `DEEP_RESEARCH_FALLBACK_ORDER`, `OLLAMA_NATIVE_BASE_URL`,
  `OLLAMA_NATIVE_MODEL`, `OLLAMA_API_KEY`, `OLLAMA_OPENAI_BASE_URL`,
  `OLLAMA_OPENAI_MODEL`, `OPENAI_API_KEY`, `CODEX_OPENAI_MODEL`, and
  related fallback keys.

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
`/v1` settings from the native `/api` base URL.

## Offline workflow smoke

Use `--mock-result` to exercise graph routing, local checkpointing, review
interrupts, and resume without live network calls:

```bash
CHECKPOINT_DIR=$(mktemp -d)
python -m deep_research_agent run \
  "AI agency lead reactivation targets" \
  --thread-id demo-prospect-thread \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --artifact-dir artifacts/demo \
  --require-review \
  --mock-result "Acme Dental|https://example.com/acme|Growing DSO with reactivation need|duckduckgo"

python -m deep_research_agent resume \
  demo-prospect-thread \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --artifact-dir artifacts/demo \
  --approve-review

python -m deep_research_agent inspect \
  --thread-id demo-prospect-thread \
  --checkpoint-dir "$CHECKPOINT_DIR"
```

Expected results:

- `run` returns JSON with `status: "interrupted"`, a stable `thread_id`,
  evidence from the mocked search result, model metadata, and review interrupt
  details.
- `resume --approve-review` returns JSON with `status: "completed"` for the
  same thread id, preserving checkpointed evidence and prospect targets.
- `inspect --thread-id` returns the persisted checkpoint state from the same
  local JSON checkpoint file.

The default checkpoint directory is `.deep_research_agent/checkpoints`; pass
`--checkpoint-dir` for tests or throwaway runs.

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

## Optional live smoke

Run this only after credentials are present in `.env` or exported in the shell:

```bash
python -m deep_research_agent config --json  # API keys are redacted by default
python -m async_multi_search "AI agency lead reactivation targets" --max-results 3
```

Then run a constrained CLI workflow with a real search result copied into
`--mock-result`, or call the package search wrappers from a small script. Live
model transports are still represented by deterministic metadata stubs in the
current G003 foundation; the CLI redacts configured API keys by default and a
later goal should replace those stubs with real Ollama/Codex/OpenAI calls behind
the same fallback event contract.

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
