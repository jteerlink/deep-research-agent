# PRD: Standalone Ollama + Exa lead-generation skill implementation

Status: draft implementation spec  
Created: 2026-05-30  
Scope: new standalone lead-generation tool kept in this repository but isolated from `deep_research_agent` runtime, imports, packaging, and dependency graph.

## 0. Resolved implementation decisions

The user clarified these MVP constraints after reviewing the first draft:

1. **Ollama Cloud only for LLM work.** Do not support or default to local Ollama in MVP. Use `https://ollama.com/api` and `OLLAMA_API_KEY`; because Ollama Cloud does not currently support structured-output `format`, use strict JSON prompts, parser validation, and repair retries.
2. **Exa remains the discovery engine.** Ollama Cloud orchestrates, plans, validates, repairs, and summarizes; Exa Deep Search performs web-grounded prospect discovery.
3. **User-written ICP only for MVP.** The first implementation accepts a complete ICP from the operator. Target-company ICP inference can be a later feature.
4. **Include relevant information returned by Exa.** The CSV/JSON should preserve the useful fields Exa returns for the campaign, not force a minimal fixed schema. The schema builder still enforces safe field-count and flat-field limits.
5. **Default target count is 10, configurable.** MVP starts with 10 prospects to simplify validation while allowing configuration for smaller/larger runs through explicit caps.

## 1. Requirements summary

Build a new lead-generation implementation inspired by the provided `exa-lead-gen` Claude skill, but implemented as a standalone Python CLI/tool with **Ollama Cloud as the only LLM backend** and Exa Deep Search as the structured search/discovery engine.

The implementation must:

- Live inside this repo, but **not import or depend on** `deep_research_agent`, `async_multi_search.py`, Streamlit UI code, existing tiered runtime, or root package config.
- Use its **own subproject directory and pyproject** so dependencies, tests, and commands are isolated.
- Use **Ollama Cloud only** for orchestration tasks:
  - normalize ICP/request input,
  - generate micro-verticals / query variations,
  - generate or validate Exa `systemPrompt` and `outputSchema`,
  - perform optional QA/repair passes on compiled results,
  - summarize the run.
- Use **Exa Search API** with `type: "deep"`, `numResults: 50`, `systemPrompt`, `additionalQueries`, and `outputSchema` for web-grounded lead discovery.
- Produce deterministic local artifacts for every run: raw Exa responses, parsed JSON, deduped JSON, CSV, run manifest, prompt/config audit, and summary markdown.
- Preserve secrets only in environment variables; never write API keys to artifacts.
- Provide offline tests with fake Ollama and fake Exa clients; live smoke tests must be optional and key-gated.

## 2. Current evidence and rationale

### Repo evidence

- Current repo intentionally centers `tiered-run`, `tiered-inspect`, and `tiered-resume`; final enrichment is blocked until human approval (`README.md:32-58`).
- Current early discovery deliberately excludes Exa and reserves Exa for approved final enrichment (`README.md:73-77`, `deep_research_agent/tiered_search.py:17-18`, `deep_research_agent/tiered_search.py:48-54`).
- Current final enrichment rejects non-Exa provider results (`deep_research_agent/final_enrichment.py:19-43`).
- Root project has its own dependency/package setup (`pyproject.toml:15-38`), so the new tool should avoid changing the root dependency graph.

### Head-to-head result

Saved run: `.deep_research_agent/head_to_head/20260530T141017Z-dental-dfw/summary.json`

- Same ICP requested 10 DFW dental prospects from each path.
- Current tiered agent returned 10 prospects but included malformed/vendor/broker-style rows and warnings.
- Exa skill-equivalent returned 10 cleaner prospects and zero overlap with current agent.
- This supports a standalone Exa-first lead-gen tool rather than trying to merge the behavior into the current tiered research process immediately.

### External API facts checked

- Ollama Cloud uses `https://ollama.com/api` and requires `OLLAMA_API_KEY`.
- Ollama local `/api/chat` supports `format: "json"` or a JSON schema, but Ollama Cloud currently does not support structured outputs; this implementation therefore uses strict JSON prompts, local schema validation, and repair retries instead of relying on schema-enforced `format`.
- Exa Search API supports `type: "deep"`, `systemPrompt`, `outputSchema`, and `numResults`; Exa MCP docs list `deep_search_exa` as deprecated in favor of `web_search_advanced_exa`, so the new implementation should call the Search API directly rather than depending on the deprecated Claude MCP tool.

## 3. Non-goals

- No changes to current `deep_research_agent` workflow, UI, models, or artifacts.
- No import reuse from existing modules, even convenient helpers such as CA-bundle handling or search providers.
- No Streamlit UI in MVP.
- No target-company ICP inference in MVP; input is a user-written ICP.
- No local Ollama backend in MVP.
- No CRM upload, email sending, or outreach automation.
- No storing raw secrets in run manifests, logs, errors, prompts, or CSVs.
- No MCP server dependency for MVP; direct Exa API is more portable and avoids deprecated MCP tool naming.

## 4. Proposed isolated project layout

Recommended directory: `standalone/ollama_exa_leadgen/`

```text
standalone/ollama_exa_leadgen/
  pyproject.toml                 # separate project; no root package dependency
  README.md                      # install/run docs for this standalone tool
  src/ollama_exa_leadgen/
    __init__.py
    __main__.py                  # python -m ollama_exa_leadgen
    cli.py                       # argparse only
    config.py                    # env loading/redaction, no shared config import
    contracts.py                 # dataclasses + schema dicts
    ollama_client.py             # stdlib HTTP Ollama client
    exa_client.py                # stdlib HTTP Exa Search API client
    planner.py                   # ICP normalization, micro-vertical planning
    schema_builder.py            # Exa outputSchema generation/validation
    runner.py                    # wave/batch execution and retry policy
    compiler.py                  # parse/dedupe/sort/write CSV
    qa.py                        # optional Ollama result-quality pass
    artifacts.py                 # run dir, JSON/CSV/markdown writers
    errors.py                    # explicit exception types
  tests/
    test_contracts.py
    test_schema_builder.py
    test_compiler.py
    test_runner_offline.py
    test_cli_offline.py
    fixtures/
      fake_exa_deep_response.json
      fake_ollama_microverticals.json
  examples/
    dental-dfw-icp.json
    software-icp.json
```

### Dependency policy

MVP should use only Python standard library modules:

- `argparse`, `csv`, `json`, `dataclasses`, `datetime`, `pathlib`, `urllib.request`, `urllib.error`, `ssl`, `concurrent.futures`, `threading`, `time`, `re`, `hashlib`, `html`, `unittest`/`pytest` for tests if using the root dev environment.

The standalone `pyproject.toml` should not depend on the root project. If tests use `pytest`, keep that in the standalone dev extra only and do not add it to root dependencies.

## 5. Product/API design

### CLI commands

```bash
# Dry-run: no API calls; shows derived plan and estimated calls.
python -m ollama_exa_leadgen plan \
  --icp-file examples/dental-dfw-icp.json \
  --target-count 10

# Run with explicit ICP.
python -m ollama_exa_leadgen run \
  --icp-file examples/dental-dfw-icp.json \
  --target-count 10 \
  --output-dir runs/dental-dfw

# Recompile from saved raw batches without more API calls.
python -m ollama_exa_leadgen compile \
  --run-dir runs/dental-dfw/2026-05-30T...

# Optional live smoke, key-gated.
python -m ollama_exa_leadgen smoke \
  --icp-file examples/dental-dfw-icp.json \
  --target-count 10
```

### Environment variables

```text
EXA_API_KEY=...
OLLAMA_BASE_URL=https://ollama.com/api
OLLAMA_MODEL=deepseek-v4-pro:cloud  # example cloud model; user-selectable
OLLAMA_API_KEY=...
OLLAMA_JSON_MODE=prompt_repair      # cloud-only MVP; no schema format reliance
LEADGEN_OUTPUT_DIR=standalone/ollama_exa_leadgen/runs
LEADGEN_DEFAULT_TARGET_COUNT=10
LEADGEN_MAX_EXA_CALLS=30
LEADGEN_WAVE_SIZE=4
LEADGEN_TIMEOUT_SECONDS=120
```

### Input ICP contract

```json
{
  "campaign_name": "dental-dfw-reactivation",
  "target_company": "Optional company we are selling for",
  "buyer_offer": "Patient reactivation and recall follow-up system",
  "target_count": 10,
  "icp_description": "DFW dental practices or DSOs with multi-location growth or recall/review follow-up opportunity",
  "geographies": ["Dallas-Fort Worth", "North Texas"],
  "include_signals": ["multiple locations", "patient recall", "reviews", "growth", "paid ads"],
  "exclude": ["national chains", "directories", "marketing agencies", "software vendors", "listicles"],
  "preferred_columns": ["company_name", "website", "product_description", "icp_fit_score", "icp_fit_reasoning", "headquarters_location", "recent_signal"]
}
```

### Output lead record contract

MVP must always include the core columns below and may include additional campaign-relevant fields selected from Exa output.

Required core columns:

```text
company_name
website
product_description
icp_fit_score
icp_fit_reasoning
source_micro_vertical
source_query_variations
exa_call_id
quality_flags
```

Common Exa-enrichment columns to preserve when requested or returned:

```text
headquarters_location
recent_signal
estimated_employee_count
funding_stage
decision_maker_titles
hiring_signals
competitor_overlap
key_technologies
recent_news
```

Field-selection rule: the schema builder chooses the most relevant fields for the ICP while respecting Exa’s flat-schema/field-count constraints. The compiler preserves unknown-but-valid returned fields in JSON, and includes configured fields in CSV.

## 6. Execution architecture

### Stage 1: user-written ICP validation

- MVP requires `--icp-file` or equivalent user-written ICP JSON.
- Validate JSON locally before any network calls.
- Optionally call Ollama Cloud to normalize wording, fill safe defaults, and verify the ICP is specific enough to search.
- Do **not** infer ICP from a target company in MVP; that is a later feature.

### Stage 2: Ollama micro-vertical generation

Ollama receives only the normalized ICP and target count. It returns structured JSON:

```json
{
  "micro_verticals": [
    {
      "name": "DFW multi-location dental practices with recall opportunity",
      "objective": "DFW dental practices multiple locations patient recall review follow-up",
      "additional_queries": ["...", "...", "..."],
      "why_distinct": "Focuses on multi-location practices rather than single clinics."
    }
  ]
}
```

Target formula:

- `target_count` defaults to 10 when omitted.
- `required_calls = ceil(target_count / 35)` for normal mode.
- Minimum calls: 1.
- Default max calls: 30 unless overridden.
- Generate 20-30% extra micro-verticals, but execute only the number required unless the compiler undershoots and `--allow-second-wave` is set.

### Stage 3: schema generation and validation

Use deterministic local schema templates by default, with Ollama allowed only to choose optional columns from an allowlist.

Rules:

- Root schema is an object with one `companies` array property.
- Keep no more than 9 item-level fields so the wrapper + fields stay inside Exa’s practical schema limit.
- Every string field description includes a word/character limit.
- Array fields are arrays of strings only.
- No nested objects inside array items.

### Stage 4: Exa Deep batch execution

For each micro-vertical, call:

```json
{
  "query": "<objective>",
  "type": "deep",
  "numResults": 50,
  "systemPrompt": "List exactly 50 companies... Do NOT include ...",
  "additionalQueries": ["...", "...", "..."],
  "outputSchema": { "type": "object", "properties": { "companies": { ... } } }
}
```

Execution:

- Use `ThreadPoolExecutor` or `asyncio.to_thread` with stdlib HTTP to avoid dependencies.
- Wave size default: 4.
- For 10-200 leads, a single wave is usually enough.
- For large runs, execute waves sequentially and compile after each wave.
- Rate limit / 429 policy: backoff once for transient 429/5xx if `--retry-transient` is set; otherwise record failure and continue.
- Each call writes one raw JSON file immediately under `raw_batches/` before parsing.

### Stage 5: compile, dedupe, score, and cap

Compiler reads raw batch files and emits:

- `prospects.raw.jsonl`
- `prospects.deduped.json`
- `prospects.csv`
- `dedupe_audit.json`
- `score_distribution.json`

Dedupe keys:

1. normalized website host,
2. normalized company name without legal suffixes,
3. optional fuzzy name match where one normalized name contains the other and websites are missing.

Selection:

- Prefer higher `icp_fit_score`.
- Prefer records with official-looking website URLs over social/directory pages.
- Preserve all duplicate source micro-verticals in `source_micro_verticals`.
- Sort descending by score, then completeness, then stable name.
- Cap to `target_count`.

### Stage 6: optional Ollama QA pass

For small runs (default `target_count <= 10`) or when `--quality-pass ollama` is set:

- Send compiled rows to Ollama in compact batches.
- Ask Ollama to flag likely non-companies, directories, vendors, chains, bad geography, or weak ICP fit.
- Do not delete flagged rows automatically unless `--drop-qa-failures` is set.
- Store `quality_flags` and `qa_reason` in CSV/JSON.

For large runs, avoid sending all raw lead data through Ollama; use deterministic checks plus optional sampled QA.

### Stage 7: artifact summary

Write `summary.md` with:

- ICP and constraints.
- Exa calls made / failed.
- Ollama model/base URL, with no key values.
- Lead count before/after dedupe.
- Score distribution.
- QA flags.
- Output file paths.

## 7. Artifact contract

Each run creates:

```text
runs/<campaign>/<timestamp>/
  run_manifest.json
  icp.normalized.json
  micro_verticals.json
  output_schema.json
  prompts/
    ollama_icp_normalization.txt
    ollama_microverticals.txt
    exa_system_prompt.txt
  raw_batches/
    batch_001_call_001.json
    batch_001_call_002.json
  parsed_batches/
    batch_001_call_001.companies.json
  prospects.raw.jsonl
  prospects.deduped.json
  prospects.csv
  dedupe_audit.json
  score_distribution.json
  qa_audit.json
  summary.md
```

Secrets redaction requirement:

- `run_manifest.json` may include `EXA_API_KEY_present: true`, never the value.
- Request headers must not be logged.
- Errors should include HTTP status and a capped body excerpt with secret-like tokens redacted.

## 8. Acceptance criteria

### Isolation

1. `grep -R "from deep_research_agent\|import deep_research_agent\|async_multi_search" standalone/ollama_exa_leadgen/src standalone/ollama_exa_leadgen/tests` returns no matches.
2. Root `pyproject.toml` is unchanged unless only documentation points to the standalone project.
3. `python -m ollama_exa_leadgen --help` works from the standalone project without installing the root package.

### Offline behavior

4. `python -m pytest standalone/ollama_exa_leadgen/tests -q` passes with fake Ollama and fake Exa clients.
5. Offline fixtures prove exactly 10 prospects can be compiled from fake Exa batch responses.
6. Schema validator rejects output schemas with more than 9 item-level fields or string fields without length constraints.
7. Compiler dedupes duplicate company names/hosts and keeps the higher score.
8. CLI `plan` command works without `EXA_API_KEY` or `OLLAMA_API_KEY` and prints estimated calls.

### Live smoke

9. If `EXA_API_KEY`, `OLLAMA_API_KEY`, and Ollama Cloud endpoint/model are configured, `smoke --target-count 10` produces up to 10 CSV rows for the DFW dental fixture.
10. Live smoke output includes at least 8 rows with non-empty `company_name`, non-empty `website`, and a populated `icp_fit_score` unless Exa returns fewer candidates; failures are recorded in `summary.md`.
11. A second run can be recompiled from raw artifacts with no network calls.

### Safety and cost control

12. Runs over the configured Exa-call cap require `--max-exa-calls` override.
13. No API key value appears in any file under the run directory.
14. HTTP failures are partial-failure tolerant; one failed Exa call does not fail the whole run unless all calls fail.

## 9. Testing strategy

### Unit tests

- `contracts.py`: validate ICP, micro-vertical, lead record, run manifest serialization.
- `schema_builder.py`: enforce Exa schema constraints.
- `compiler.py`: parse response variants, dedupe, score sort, CSV quoting.
- `config.py`: env loading and secret redaction.
- `ollama_client.py`: request payload shape for Ollama Cloud prompt-repair mode; no local schema mode in MVP.
- `exa_client.py`: request payload shape with `type: deep`, `numResults: 50`, `systemPrompt`, `additionalQueries`, `outputSchema`.

### Integration/offline tests

- Fake Ollama returns micro-vertical JSON.
- Fake Exa returns two overlapping company arrays.
- Runner writes raw batches first, then compiler emits final CSV.
- Recompile command reads existing raw batches and produces identical CSV hash.

### Optional live tests

- `LEADGEN_LIVE_SMOKE=1` gates live API tests.
- Live smoke uses `examples/dental-dfw-icp.json`, target count 10, max Exa calls follows the configured cap and estimated-call formula.
- Test writes artifacts under a temp or ignored run directory.

## 10. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Ollama Cloud lacks structured outputs | JSON parsing can be flaky | Use strict prompts, local schema validation, and bounded repair retries; do not use local Ollama schema mode in MVP |
| Exa output fields may be web-grounded but imperfect | Bad CSV rows | Preserve raw response, citations/grounding when available, and add Ollama QA flags |
| No shared dependencies increases HTTP/client code | More boilerplate | Keep clients small; stdlib only; test request payloads thoroughly |
| Large runs can burn Exa credits | Cost surprise | `target_count` -> call estimate, max-call cap, wave size cap, dry-run plan command |
| Duplicate/affiliate brands | Inflated CSV | Host/name dedupe plus source provenance aggregation |
| Confusion with existing deep research agent | Maintenance drift | Separate directory, separate pyproject, separate README, no imports, no root script entrypoint unless later approved |

## 11. ADR

### Decision

Create a standalone `standalone/ollama_exa_leadgen` subproject that reimplements the winning Exa lead-gen skill pattern with Ollama Cloud as the only orchestration backend and Exa Deep Search as the web-grounded lead discovery engine.

### Drivers

1. The head-to-head test showed Exa-style discovery produced cleaner prospect names for the same ICP.
2. The current `deep_research_agent` intentionally reserves Exa for post-review final enrichment, so merging Exa-first discovery into it would violate current architecture.
3. User requested separation and no shared dependencies.

### Alternatives considered

- **Patch current tiered agent to use Exa early.** Rejected: conflicts with explicit current provider phase policy and would blur review-gated architecture.
- **Build an Ollama-only prospect finder.** Rejected for MVP: user clarified Ollama Cloud should be the LLM while Exa remains discovery; Ollama has no native web index.
- **Support local Ollama as an alternate backend.** Rejected for MVP: user clarified only Ollama Cloud should be used.
- **Implement as a Claude/Codex skill wrapper.** Rejected: user requested Ollama backend; direct API/CLI is more portable and avoids deprecated Exa MCP `deep_search_exa`.

### Consequences

- This repo will contain two prospecting paths with different philosophies:
  - existing `deep_research_agent`: review-gated, local-first, Exa final enrichment only;
  - new standalone tool: Exa-first bulk lead generation, Ollama-orchestrated.
- The new tool can move faster without destabilizing the current agent.
- Comparison/evaluation artifacts should be shared only as files, not shared code.

### Follow-ups

- After MVP, decide whether target-company ICP inference should be added.
- After MVP, decide whether local Ollama support is worth adding despite the current cloud-only constraint.
- Tune default Exa-call caps after real 50-lead runs.
- After MVP, rerun the DFW dental head-to-head and compare current agent, Exa skill-equivalent script, and standalone Ollama+Exa tool.

## 12. Implementation steps

1. Add `standalone/ollama_exa_leadgen/pyproject.toml`, README, package skeleton, and examples.
2. Implement contracts/config/redaction with stdlib only.
3. Implement Ollama Cloud client with strict JSON prompt, parser validation, and repair retry mode.
4. Implement Exa client with direct Search API request builder and response parser.
5. Implement deterministic schema builder and validator.
6. Implement micro-vertical planner using Ollama Cloud for live runs, with deterministic plan-mode previews and offline fixtures for tests.
7. Implement runner waves, raw-batch artifact writes, partial failure handling.
8. Implement compiler/dedupe/sort/CSV writer.
9. Implement optional Ollama QA pass.
10. Add offline tests and fixture-based CLI smoke.
11. Add optional live smoke command.
12. Document operating examples and run the DFW dental 10-prospect comparison.

## 13. Verification commands

Expected after implementation:

```bash
# Isolation grep
grep -R --include='*.py' "from deep_research_agent\|import deep_research_agent\|async_multi_search" \
  standalone/ollama_exa_leadgen/src standalone/ollama_exa_leadgen/tests && exit 1 || true

# Offline tests
python -m pytest standalone/ollama_exa_leadgen/tests -q

# CLI dry run
(cd standalone/ollama_exa_leadgen && python -m ollama_exa_leadgen plan \
  --icp-file examples/dental-dfw-icp.json --target-count 10)

# Optional live smoke
(cd standalone/ollama_exa_leadgen && LEADGEN_LIVE_SMOKE=1 python -m ollama_exa_leadgen smoke \
  --icp-file examples/dental-dfw-icp.json --target-count 10)
```
