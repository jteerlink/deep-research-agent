from __future__ import annotations

import csv
import json
from dataclasses import dataclass

import pytest

from deep_research_agent.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    write_csv_artifact,
    write_json_artifact,
    write_markdown_artifact,
    write_research_artifacts,
)


@dataclass(frozen=True)
class EvidenceRow:
    query: str
    url: str
    title: str
    evidence_type: str
    citation_ids: tuple[str, ...]


def test_write_research_artifacts_emits_json_csv_and_markdown(tmp_path) -> None:
    records = [
        EvidenceRow(
            query="acme funding",
            url="https://example.com/acme",
            title="Acme | funding",
            evidence_type="snippet",
            citation_ids=("cite-1",),
        ),
        {
            "query": "acme leadership",
            "url": "https://example.com/leadership",
            "title": "Leadership",
            "evidence_type": "page_read",
            "citation_ids": ["cite-2", "cite-3"],
            "notes": {"source": "page"},
        },
    ]

    result = write_research_artifacts(records, tmp_path, metadata={"story": "G002"})

    assert result.record_count == 2
    assert result.fieldnames == ("query", "url", "title", "evidence_type", "citation_ids", "notes")
    assert result.json_path.name == "search_evidence.json"
    assert result.csv_path.name == "search_evidence.csv"
    assert result.markdown_path.name == "search_evidence.md"

    payload = json.loads(result.json_path.read_text())
    assert payload["schema_version"] == ARTIFACT_SCHEMA_VERSION
    assert payload["record_count"] == 2
    assert payload["metadata"] == {"story": "G002"}
    assert payload["records"][1]["evidence_type"] == "page_read"

    with result.csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["citation_ids"] == '["cite-1"]'
    assert rows[1]["notes"] == '{"source": "page"}'

    markdown = result.markdown_path.read_text()
    assert "# Search Evidence Artifact" in markdown
    assert f"Schema version: `{ARTIFACT_SCHEMA_VERSION}`" in markdown
    assert "page_read" in markdown


def test_individual_writers_create_parent_directories_and_escape_markdown(tmp_path) -> None:
    records = [{"title": "Pipe | newline", "summary": "one\ntwo"}]

    json_path = write_json_artifact(records, tmp_path / "nested" / "artifact.json")
    csv_path = write_csv_artifact(records, tmp_path / "nested" / "artifact.csv")
    markdown_path = write_markdown_artifact(records, tmp_path / "nested" / "artifact.md")

    assert json_path.exists()
    assert csv_path.exists()
    assert "Pipe \\| newline" in markdown_path.read_text()
    assert "one<br>two" in markdown_path.read_text()


def test_empty_records_can_use_explicit_columns(tmp_path) -> None:
    path = write_csv_artifact([], tmp_path / "empty.csv", fieldnames=("query", "url"))

    assert path.read_text() == "query,url\n"


def test_artifact_writers_reject_non_mapping_records(tmp_path) -> None:
    with pytest.raises(TypeError, match="artifact records must be mappings"):
        write_json_artifact(["not-a-record"], tmp_path / "bad.json")
