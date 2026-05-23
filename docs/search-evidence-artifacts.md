# Search evidence artifacts

G002 writes research output as local files rather than a hosted UI, model zoo,
or database-backed service. Artifact writers accept already-normalized evidence
or prospect records and serialize the same records to JSON, CSV, and Markdown.

## Formats

Use `deep_research_agent.artifacts.write_research_artifacts` to create the
three standard outputs in one directory:

- `search_evidence.json` — versioned envelope with `schema_version`,
  `record_count`, `metadata`, and `records`.
- `search_evidence.csv` — flat table with deterministic column ordering.
- `search_evidence.md` — compact Markdown report table for review handoff.

The schema version is currently `search_evidence_artifact.v1`. The writers do
not call search providers and do not validate citations themselves; callers
should validate citation IDs and evidence semantics before writing. Keep snippet
records and page-read records explicit in a field such as `evidence_type` so
reviewers can distinguish search-result snippets from fetched page content.

## Example

```python
from deep_research_agent.artifacts import write_research_artifacts

records = [
    {
        "query": "acme funding",
        "url": "https://example.com/acme",
        "title": "Acme funding news",
        "evidence_type": "snippet",
        "citation_ids": ["cite-1"],
    }
]

result = write_research_artifacts(records, "artifacts", metadata={"story": "G002"})
print(result.json_path, result.csv_path, result.markdown_path)
```
