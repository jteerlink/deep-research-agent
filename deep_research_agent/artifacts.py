"""JSON, CSV, and Markdown artifact writers for G002 research outputs."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .evidence import EvidenceRecord, Prospect, SearchEvidenceArtifact, validate_citations


def _write_text(path: str | Path, content: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination


def _coerce_payload(payload: Any) -> Any:
    if isinstance(payload, SearchEvidenceArtifact):
        return payload.to_dict()
    if isinstance(payload, EvidenceRecord | Prospect):
        return payload.to_dict()
    if isinstance(payload, Mapping):
        return dict(payload)
    if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes)):
        return [_coerce_payload(item) for item in payload]
    return payload


def write_json_artifact(path: str | Path, payload: Any) -> Path:
    """Write a JSON artifact with deterministic formatting."""

    content = json.dumps(_coerce_payload(payload), indent=2, sort_keys=True) + "\n"
    return _write_text(path, content)


def write_csv_artifact(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write rows as CSV, preserving keys from the first row as the header."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
    return destination


def write_markdown_artifact(
    path: str | Path,
    *,
    title: str,
    prospects: Sequence[Prospect] = (),
    evidence: Sequence[EvidenceRecord] = (),
) -> Path:
    """Write a human-readable Markdown research artifact.

    Prospect citations are validated before writing so Markdown cannot silently
    reference missing evidence ids.
    """

    validate_citations(prospects, evidence)
    lines = [f"# {title}", ""]
    if prospects:
        lines.extend(["## Prospects", ""])
        for prospect in prospects:
            lines.append(f"### {prospect.name}")
            if prospect.organization or prospect.title:
                descriptor = " — ".join(
                    part for part in (prospect.title, prospect.organization) if part
                )
                lines.append(descriptor)
            if prospect.url:
                lines.append(f"Source: {prospect.url}")
            if prospect.summary:
                lines.extend(["", prospect.summary])
            if prospect.citations:
                lines.append("")
                lines.append("Citations: " + ", ".join(c.evidence_id for c in prospect.citations))
            lines.append("")
    if evidence:
        lines.extend(["## Evidence", ""])
        for record in evidence:
            lines.append(f"- [{record.id}] **{record.title or record.url}** ({record.kind.value})")
            lines.append(f"  - URL: {record.url}")
            lines.append(f"  - Provider: {record.provider or 'unknown'}")
            lines.append(f"  - Content: {record.content}")
    return _write_text(path, "\n".join(lines).rstrip() + "\n")
