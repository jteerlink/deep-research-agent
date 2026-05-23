from __future__ import annotations

import asyncio
import csv
import json

import pytest

from async_multi_search import AllProvidersFailedError, SearchResult
from deep_research_agent.artifacts import (
    write_csv_artifact,
    write_json_artifact,
    write_markdown_artifact,
)
from deep_research_agent.search import (
    Citation,
    CitationValidationError,
    EvidenceKind,
    EvidenceRecord,
    Prospect,
    collect_search_evidence,
    evidence_schema,
    prospect_schema,
    validate_citations,
)


def test_collect_search_evidence_wraps_legacy_web_search_and_marks_snippets() -> None:
    calls: list[tuple[str, int]] = []

    async def fake_search(query: str, max_results: int) -> list[SearchResult]:
        calls.append((query, max_results))
        return [
            SearchResult(
                title="Example",
                url="https://example.com/a",
                content="SERP snippet",
                score=0.9,
                provider="brave",
            )
        ]

    artifact = asyncio.run(
        collect_search_evidence("  example query  ", max_results=3, search_fn=fake_search)
    )

    assert calls == [("example query", 3)]
    assert artifact.ok is True
    assert artifact.failures == ()
    assert len(artifact.evidence) == 1
    record = artifact.evidence[0]
    assert record.kind is EvidenceKind.SNIPPET
    assert record.query == "example query"
    assert record.rank == 1
    assert record.provider == "brave"
    assert record.id.startswith("ev_")
    assert record.to_dict()["kind"] == "snippet"


def test_collect_search_evidence_marks_exa_text_as_page_read() -> None:
    async def fake_search(query: str, max_results: int) -> list[SearchResult]:
        return [
            SearchResult(
                title="Full page",
                url="https://example.com/page",
                content="Long page-read text",
                provider="exa",
            )
        ]

    artifact = asyncio.run(collect_search_evidence("page read", search_fn=fake_search))

    assert artifact.ok is True
    assert artifact.evidence[0].kind is EvidenceKind.PAGE_READ
    assert artifact.evidence[0].to_dict()["kind"] == "page_read"


def test_collect_search_evidence_recovers_from_all_provider_failure() -> None:
    async def failing_search(query: str, max_results: int) -> list[SearchResult]:
        raise AllProvidersFailedError("all providers failed: [('duckduckgo', 'empty')]")

    artifact = asyncio.run(collect_search_evidence("no results", search_fn=failing_search))

    assert artifact.ok is False
    assert artifact.evidence == ()
    assert len(artifact.failures) == 1
    assert artifact.failures[0].error_class == "AllProvidersFailedError"
    assert "duckduckgo" in artifact.failures[0].error_message


def test_collect_search_evidence_keeps_malformed_result_as_recoverable_failure() -> None:
    async def malformed_search(query: str, max_results: int) -> list[SearchResult]:
        return [
            SearchResult("Bad", "not-a-url", "content", provider="brave"),
            SearchResult("Good", "https://example.com/good", "content", provider="brave"),
        ]

    artifact = asyncio.run(collect_search_evidence("mixed", search_fn=malformed_search))

    assert artifact.ok is True
    assert [record.url for record in artifact.evidence] == ["https://example.com/good"]
    assert len(artifact.failures) == 1
    assert artifact.failures[0].provider == "brave"
    assert "absolute" in artifact.failures[0].error_message


def test_collect_search_evidence_quarantines_invalid_items_and_accepts_mappings() -> None:
    async def mixed_search(query: str, max_results: int) -> list[object]:  # type: ignore[override]
        return [
            object(),
            {
                "title": "Mapped",
                "url": "https://example.com/mapped",
                "content": "mapped snippet",
                "provider": "custom",
            },
        ]

    artifact = asyncio.run(collect_search_evidence("mixed raw", search_fn=mixed_search))  # type: ignore[arg-type]

    assert artifact.ok is True
    assert [record.url for record in artifact.evidence] == ["https://example.com/mapped"]
    assert artifact.evidence[0].kind is EvidenceKind.SNIPPET
    assert len(artifact.failures) == 1
    assert artifact.failures[0].error_class == "TypeError"
    assert "object" in artifact.failures[0].error_message


def test_prospect_and_evidence_schemas_expose_required_citation_contract() -> None:
    assert set(evidence_schema()["required"]) == {"id", "query", "title", "url", "content", "kind"}
    assert evidence_schema()["properties"]["kind"]["enum"] == ["snippet", "page_read"]
    schema = prospect_schema()
    assert schema["required"] == ["name", "citations"]
    assert schema["properties"]["citations"]["items"]["required"] == ["evidence_id"]


def test_validate_citations_rejects_unknown_evidence_ids() -> None:
    evidence = [
        EvidenceRecord(
            id="ev_known",
            query="q",
            title="Known",
            url="https://example.com",
            content="snippet",
            kind=EvidenceKind.SNIPPET,
        )
    ]
    good = Prospect(name="Good", citations=(Citation(evidence_id="ev_known"),))
    bad = Prospect(name="Bad", citations=(Citation(evidence_id="ev_missing"),))

    validate_citations([good], evidence)
    with pytest.raises(CitationValidationError, match="ev_missing"):
        validate_citations([bad], evidence)


def test_artifact_writers_emit_json_csv_and_markdown(tmp_path) -> None:
    evidence = [
        EvidenceRecord(
            id="ev_known",
            query="q",
            title="Known",
            url="https://example.com",
            content="snippet",
            kind=EvidenceKind.SNIPPET,
            provider="brave",
            rank=1,
        )
    ]
    prospect = Prospect(
        name="Ada Lovelace",
        organization="Analytical Engines Inc.",
        title="Research Lead",
        summary="Relevant prospect.",
        citations=(Citation(evidence_id="ev_known", claim="role"),),
    )

    json_path = write_json_artifact(
        tmp_path / "artifact.json",
        {"evidence": evidence, "prospects": [prospect]},
    )
    csv_path = write_csv_artifact(
        tmp_path / "evidence.csv",
        [record.to_dict() for record in evidence],
    )
    md_path = write_markdown_artifact(
        tmp_path / "artifact.md",
        title="Research Artifact",
        prospects=[prospect],
        evidence=evidence,
    )

    payload = json.loads(json_path.read_text())
    assert payload["evidence"][0]["kind"] == "snippet"
    assert payload["prospects"][0]["citations"][0]["evidence_id"] == "ev_known"
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["id"] == "ev_known"
    markdown = md_path.read_text()
    assert "# Research Artifact" in markdown
    assert "ev_known" in markdown
    assert "https://example.com" in markdown
