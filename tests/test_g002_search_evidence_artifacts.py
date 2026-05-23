from __future__ import annotations

import asyncio
import csv
import json

import pytest

from async_multi_search import AllProvidersFailedError, SearchResult
from deep_research_agent.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    build_artifact_payload,
    write_csv_artifact,
    write_json_artifact,
    write_markdown_artifact,
)
from deep_research_agent.evidence import (
    EvidenceRecord,
    collect_search_evidence,
    normalize_search_results,
)
from deep_research_agent.prospects import (
    CitationValidationError,
    Prospect,
    ProspectCitation,
    validate_prospect_citations,
)


def test_normalize_search_results_preserves_snippet_vs_page_read_semantics() -> None:
    records = normalize_search_results(
        "query",
        [
            SearchResult("Snippet", "https://example.com/snippet", "short text", provider="brave"),
            SearchResult("Page", "https://example.com/page", "full page text", provider="exa"),
        ],
    )

    assert [record.rank for record in records] == [1, 2]
    assert [record.source_type for record in records] == ["snippet", "page_read"]
    assert records[0].id.startswith("ev_")
    assert records[0].query == "query"


def test_collect_search_evidence_recovers_search_failures_without_throwing() -> None:
    async def failing_search(query: str, max_results: int):
        raise AllProvidersFailedError(f"all providers failed for {query}/{max_results}")

    batch = asyncio.run(
        collect_search_evidence("find prospects", max_results=3, search=failing_search)
    )

    assert batch.records == ()
    assert batch.ok is False
    assert len(batch.failures) == 1
    assert batch.failures[0].error_class == "AllProvidersFailedError"
    assert "find prospects/3" in batch.failures[0].error_message


def test_collect_search_evidence_normalizes_successful_results() -> None:
    async def successful_search(query: str, max_results: int):
        assert query == "find prospects"
        assert max_results == 2
        return [
            SearchResult("A", "https://a.example", "alpha snippet", provider="duckduckgo"),
            SearchResult("B", "https://b.example", "beta page", provider="exa"),
        ]

    batch = asyncio.run(
        collect_search_evidence("find prospects", max_results=2, search=successful_search)
    )

    assert batch.ok is True
    assert [record.provider for record in batch.records] == ["duckduckgo", "exa"]
    assert [record.source_type for record in batch.records] == ["snippet", "page_read"]
    assert batch.to_dict()["records"][0]["text"] == "alpha snippet"


def test_validate_prospect_citations_rejects_missing_evidence_id() -> None:
    prospect = Prospect(
        name="Acme",
        summary="Likely buyer",
        citations=(ProspectCitation("missing", "claim"),),
    )

    with pytest.raises(CitationValidationError, match="unknown evidence id"):
        validate_prospect_citations(prospect, [])


def test_validate_prospect_citations_rejects_quote_not_present_in_evidence() -> None:
    evidence = EvidenceRecord(
        id="ev_1",
        query="q",
        title="Title",
        url="https://example.com",
        text="confirmed source text",
        source_type="page_read",
        provider="exa",
        rank=1,
    )
    prospect = Prospect(
        name="Acme",
        summary="Likely buyer",
        citations=(ProspectCitation("ev_1", "claim", quote="not in source"),),
    )

    with pytest.raises(CitationValidationError, match="not present"):
        validate_prospect_citations(prospect, [evidence])


def test_artifact_payload_and_writers_preserve_schema_and_citations(tmp_path) -> None:
    evidence = EvidenceRecord(
        id="ev_1",
        query="q",
        title="Evidence title",
        url="https://example.com",
        text="confirmed source text",
        source_type="page_read",
        provider="exa",
        rank=1,
    )
    prospect = Prospect(
        name="Acme",
        summary="Likely buyer",
        citations=(ProspectCitation("ev_1", "uses source", quote="confirmed source"),),
        metadata={"segment": "midmarket"},
    )

    payload = build_artifact_payload(
        prospects=[prospect],
        evidence_records=[evidence],
        metadata={"query": "q"},
    )
    assert payload["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert payload["prospects"][0]["citations"][0]["evidence_id"] == "ev_1"
    assert payload["evidence"][0]["source_type"] == "page_read"

    json_path = write_json_artifact(
        tmp_path / "artifact.json",
        prospects=[prospect],
        evidence_records=[evidence],
        metadata={"query": "q"},
    )
    assert json.loads(json_path.read_text())["schema_version"] == ARTIFACT_SCHEMA_VERSION

    csv_path = write_csv_artifact(tmp_path / "artifact.csv", prospects=[prospect], evidence_records=[evidence])
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "name": "Acme",
            "summary": "Likely buyer",
            "citation_ids": "ev_1",
            "metadata_json": '{"segment": "midmarket"}',
        }
    ]

    markdown_path = write_markdown_artifact(
        tmp_path / "artifact.md",
        prospects=[prospect],
        evidence_records=[evidence],
    )
    markdown = markdown_path.read_text()
    assert "# Deep Research Artifact" in markdown
    assert "`ev_1`" in markdown
    assert "Schema: `g002.search_evidence.v1`" in markdown
