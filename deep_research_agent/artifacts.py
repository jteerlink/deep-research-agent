"""JSON, CSV, and Markdown artifact helpers for G002 research outputs."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, NamedTuple

from .evidence import EvidenceRecord, SearchFailure
from .prospects import Prospect, validate_prospects_citations

ARTIFACT_SCHEMA_VERSION = "g002.search_evidence.v1"


class ResearchArtifactPaths(NamedTuple):
    """Paths and schema metadata emitted by write_research_artifacts."""

    json_path: Path
    csv_path: Path
    markdown_path: Path
    record_count: int
    fieldnames: tuple[str, ...]


def _record_to_mapping(record: Mapping[str, Any] | object) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    if is_dataclass(record) and not isinstance(record, type):
        return asdict(record)
    raise TypeError("artifact records must be mappings or dataclass instances")


def _records_to_dicts(records: Sequence[Mapping[str, Any] | object]) -> list[dict[str, Any]]:
    return [_record_to_mapping(record) for record in records]


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _fieldnames(
    records: Sequence[Mapping[str, Any] | object], explicit: Sequence[str] | None = None
) -> tuple[str, ...]:
    if explicit is not None:
        return tuple(explicit)
    seen: list[str] = []
    for record in _records_to_dicts(records):
        for key in record:
            if key not in seen:
                seen.append(key)
    return tuple(seen)


def _ensure_parent(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _escape_markdown(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", "<br>")


def build_artifact_payload(
    *,
    prospects: Sequence[Prospect],
    evidence_records: Sequence[EvidenceRecord],
    failures: Sequence[SearchFailure] = (),
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, JSON-serializable prospect artifact payload."""

    validate_prospects_citations(prospects, evidence_records)
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "metadata": dict(metadata or {}),
        "prospects": [prospect.to_dict() for prospect in prospects],
        "evidence": [record.to_dict() for record in evidence_records],
        "failures": [asdict(failure) for failure in failures],
    }


def write_json_artifact(*args: Any, **kwargs: Any) -> Path:
    """Write JSON artifact.

    Supports both the prospect-specific form:
        write_json_artifact(path, prospects=[...], evidence_records=[...])
    and the generic search-record form:
        write_json_artifact(records, path)
    """

    if args and isinstance(args[0], (str, Path)):
        path = _ensure_parent(args[0])
        payload = build_artifact_payload(**kwargs)
    else:
        records, path_arg = args[0], args[1]
        path = _ensure_parent(path_arg)
        record_dicts = _records_to_dicts(records)
        payload = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "metadata": dict(kwargs.get("metadata") or {}),
            "record_count": len(record_dicts),
            "records": [_jsonable(record) for record in record_dicts],
        }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_csv_artifact(*args: Any, **kwargs: Any) -> Path:
    """Write CSV artifact in prospect-specific or generic record form."""

    if args and isinstance(args[0], (str, Path)):
        path = _ensure_parent(args[0])
        prospects = kwargs["prospects"]
        evidence_records = kwargs["evidence_records"]
        validate_prospects_citations(prospects, evidence_records)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["name", "summary", "citation_ids", "metadata_json"],
            )
            writer.writeheader()
            for prospect in prospects:
                writer.writerow(
                    {
                        "name": prospect.name,
                        "summary": prospect.summary,
                        "citation_ids": ";".join(
                            citation.evidence_id for citation in prospect.citations
                        ),
                        "metadata_json": json.dumps(dict(prospect.metadata), sort_keys=True),
                    }
                )
        return path

    records, path_arg = args[0], args[1]
    path = _ensure_parent(path_arg)
    fieldnames = _fieldnames(records, kwargs.get("fieldnames"))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in _records_to_dicts(records):
            writer.writerow(
                {
                    field: json.dumps(_jsonable(record[field]), sort_keys=True)
                    if isinstance(record.get(field), (dict, list, tuple))
                    else record.get(field, "")
                    for field in fieldnames
                }
            )
    return path


def write_markdown_artifact(*args: Any, **kwargs: Any) -> Path:
    """Write Markdown artifact in prospect-specific or generic record form."""

    if args and isinstance(args[0], (str, Path)):
        path = _ensure_parent(args[0])
        prospects = kwargs["prospects"]
        evidence_records = kwargs["evidence_records"]
        failures = kwargs.get("failures", ())
        metadata = kwargs.get("metadata")
        validate_prospects_citations(prospects, evidence_records)
        lines = ["# Deep Research Artifact", "", f"Schema: `{ARTIFACT_SCHEMA_VERSION}`", ""]
        if metadata:
            lines.extend(["## Metadata", ""])
            for key, value in sorted(metadata.items()):
                lines.append(f"- **{key}**: {value}")
            lines.append("")
        lines.extend(["## Prospects", ""])
        for prospect in prospects:
            lines.extend([f"### {prospect.name}", "", prospect.summary, ""])
            if prospect.citations:
                lines.append("Citations:")
                for citation in prospect.citations:
                    quote = f' — "{citation.quote}"' if citation.quote else ""
                    lines.append(f"- `{citation.evidence_id}`: {citation.claim}{quote}")
                lines.append("")
        lines.extend(["## Evidence", ""])
        for record in evidence_records:
            lines.append(
                f"- `{record.id}` ({record.source_type}, {record.provider}, rank {record.rank}): "
                f"[{record.title}]({record.url})"
            )
        if failures:
            lines.extend(["", "## Search Failures", ""])
            for failure in failures:
                lines.append(f"- `{failure.query}`: {failure.error_class}: {failure.error_message}")
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return path

    records, path_arg = args[0], args[1]
    path = _ensure_parent(path_arg)
    record_dicts = _records_to_dicts(records)
    fieldnames = _fieldnames(record_dicts)
    lines = ["# Search Evidence Artifact", "", f"Schema version: `{ARTIFACT_SCHEMA_VERSION}`", ""]
    if record_dicts:
        lines.append("| " + " | ".join(fieldnames) + " |")
        lines.append("| " + " | ".join("---" for _ in fieldnames) + " |")
        for record in record_dicts:
            lines.append(
                "| " + " | ".join(_escape_markdown(record.get(f, "")) for f in fieldnames) + " |"
            )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def write_research_artifacts(
    records: Sequence[Mapping[str, Any] | object],
    output_dir: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ResearchArtifactPaths:
    """Write JSON/CSV/Markdown search evidence artifacts."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(records)
    json_path = write_json_artifact(
        records, output_path / "search_evidence.json", metadata=metadata
    )
    csv_path = write_csv_artifact(
        records, output_path / "search_evidence.csv", fieldnames=fieldnames
    )
    markdown_path = write_markdown_artifact(records, output_path / "search_evidence.md")
    return ResearchArtifactPaths(
        json_path=json_path,
        csv_path=csv_path,
        markdown_path=markdown_path,
        record_count=len(records),
        fieldnames=fieldnames,
    )
