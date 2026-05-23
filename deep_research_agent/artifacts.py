"""Artifact writers for normalized research evidence records.

The helpers in this module are intentionally schema-light: callers own the
record validation step, while the writers provide deterministic JSON, CSV, and
Markdown serialization for already-normalized evidence/prospect records. This
keeps G001 import compatibility intact and avoids coupling artifact generation
to a hosted UI, database, or provider runtime.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

JSONPrimitive = str | int | float | bool | None
JSONValue = JSONPrimitive | list["JSONValue"] | dict[str, "JSONValue"]
ArtifactRecord = Mapping[str, Any]

ARTIFACT_SCHEMA_VERSION = "search_evidence_artifact.v1"
DEFAULT_BASENAME = "search_evidence"
DEFAULT_MARKDOWN_TITLE = "Search Evidence Artifact"


@dataclass(frozen=True)
class ArtifactWriteResult:
    """Paths and counts produced by a multi-format artifact write."""

    json_path: Path
    csv_path: Path
    markdown_path: Path
    record_count: int
    fieldnames: tuple[str, ...]


def _coerce_record(record: ArtifactRecord | object) -> dict[str, Any]:
    if is_dataclass(record) and not isinstance(record, type):
        raw_record = asdict(record)
    elif isinstance(record, Mapping):
        raw_record = dict(record)
    else:
        raise TypeError(f"artifact records must be mappings or dataclass instances, got {type(record)!r}")

    coerced: dict[str, Any] = {}
    for key, value in raw_record.items():
        if not isinstance(key, str):
            raise TypeError(f"artifact record keys must be strings, got {key!r}")
        coerced[key] = value
    return coerced


def _normalize_records(records: Iterable[ArtifactRecord | object]) -> list[dict[str, Any]]:
    return [_coerce_record(record) for record in records]


def _resolve_fieldnames(
    records: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None
) -> tuple[str, ...]:
    if fieldnames is not None:
        return tuple(fieldnames)

    ordered: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record:
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    return tuple(ordered)


def _jsonable(value: Any) -> JSONValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list | set):
        return [_jsonable(item) for item in value]
    return str(value)


def _csv_cell(value: Any) -> str:
    jsonable = _jsonable(value)
    if jsonable is None:
        return ""
    if isinstance(jsonable, str):
        return jsonable
    if isinstance(jsonable, int | float | bool):
        return str(jsonable)
    return json.dumps(jsonable, sort_keys=True)


def _markdown_cell(value: Any) -> str:
    text = _csv_cell(value)
    return text.replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json_artifact(
    records: Iterable[ArtifactRecord | object],
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write normalized records as a versioned JSON artifact envelope."""

    output_path = Path(path)
    normalized = _normalize_records(records)
    payload: dict[str, JSONValue] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "record_count": len(normalized),
        "metadata": _jsonable(dict(metadata or {})),
        "records": _jsonable(normalized),
    }

    _ensure_parent(output_path)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def write_csv_artifact(
    records: Iterable[ArtifactRecord | object],
    path: str | Path,
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Write normalized records as CSV with deterministic column order."""

    output_path = Path(path)
    normalized = _normalize_records(records)
    columns = _resolve_fieldnames(normalized, fieldnames)

    _ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if columns:
            writer.writeheader()
        for record in normalized:
            writer.writerow({column: _csv_cell(record.get(column)) for column in columns})
    return output_path


def write_markdown_artifact(
    records: Iterable[ArtifactRecord | object],
    path: str | Path,
    *,
    title: str = DEFAULT_MARKDOWN_TITLE,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Write normalized records as a compact Markdown report table."""

    output_path = Path(path)
    normalized = _normalize_records(records)
    columns = _resolve_fieldnames(normalized, fieldnames)

    lines = [f"# {title}", "", f"Schema version: `{ARTIFACT_SCHEMA_VERSION}`", "", f"Records: {len(normalized)}"]
    if columns:
        lines.extend(
            [
                "",
                "| " + " | ".join(columns) + " |",
                "| " + " | ".join("---" for _ in columns) + " |",
            ]
        )
        for record in normalized:
            lines.append("| " + " | ".join(_markdown_cell(record.get(column)) for column in columns) + " |")
    else:
        lines.extend(["", "No records."])

    _ensure_parent(output_path)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_research_artifacts(
    records: Iterable[ArtifactRecord | object],
    output_dir: str | Path,
    *,
    basename: str = DEFAULT_BASENAME,
    metadata: Mapping[str, Any] | None = None,
    fieldnames: Sequence[str] | None = None,
    markdown_title: str = DEFAULT_MARKDOWN_TITLE,
) -> ArtifactWriteResult:
    """Write JSON, CSV, and Markdown artifacts for the same normalized records."""

    normalized = _normalize_records(records)
    columns = _resolve_fieldnames(normalized, fieldnames)
    directory = Path(output_dir)

    json_path = write_json_artifact(normalized, directory / f"{basename}.json", metadata=metadata)
    csv_path = write_csv_artifact(normalized, directory / f"{basename}.csv", fieldnames=columns)
    markdown_path = write_markdown_artifact(
        normalized,
        directory / f"{basename}.md",
        title=markdown_title,
        fieldnames=columns,
    )
    return ArtifactWriteResult(
        json_path=json_path,
        csv_path=csv_path,
        markdown_path=markdown_path,
        record_count=len(normalized),
        fieldnames=columns,
    )


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "DEFAULT_BASENAME",
    "DEFAULT_MARKDOWN_TITLE",
    "ArtifactRecord",
    "ArtifactWriteResult",
    "write_csv_artifact",
    "write_json_artifact",
    "write_markdown_artifact",
    "write_research_artifacts",
]
