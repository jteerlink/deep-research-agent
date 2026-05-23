"""JSON, CSV, and Markdown artifact helpers for G002 research outputs."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .evidence import EvidenceRecord, SearchFailure
from .prospects import Prospect, validate_prospects_citations

ARTIFACT_SCHEMA_VERSION = "g002.search_evidence.v1"


def build_artifact_payload(
    *,
    prospects: Sequence[Prospect],
    evidence_records: Sequence[EvidenceRecord],
    failures: Sequence[SearchFailure] = (),
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, JSON-serializable research artifact payload."""

    validate_prospects_citations(prospects, evidence_records)
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "metadata": dict(metadata or {}),
        "prospects": [prospect.to_dict() for prospect in prospects],
        "evidence": [record.to_dict() for record in evidence_records],
        "failures": [asdict(failure) for failure in failures],
    }


def write_json_artifact(
    path: str | Path,
    *,
    prospects: Sequence[Prospect],
    evidence_records: Sequence[EvidenceRecord],
    failures: Sequence[SearchFailure] = (),
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write the canonical JSON artifact and return its path."""

    output_path = Path(path)
    payload = build_artifact_payload(
        prospects=prospects,
        evidence_records=evidence_records,
        failures=failures,
        metadata=metadata,
    )
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def write_csv_artifact(
    path: str | Path,
    *,
    prospects: Sequence[Prospect],
    evidence_records: Sequence[EvidenceRecord],
) -> Path:
    """Write a prospect-oriented CSV artifact with citation ids."""

    validate_prospects_citations(prospects, evidence_records)
    output_path = Path(path)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
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
    return output_path


def write_markdown_artifact(
    path: str | Path,
    *,
    prospects: Sequence[Prospect],
    evidence_records: Sequence[EvidenceRecord],
    failures: Sequence[SearchFailure] = (),
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a readable Markdown artifact with prospects, citations, and evidence."""

    validate_prospects_citations(prospects, evidence_records)
    output_path = Path(path)
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
                quote = f" — \"{citation.quote}\"" if citation.quote else ""
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
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path
