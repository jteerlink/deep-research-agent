"""Artifact helpers for standalone leadgen runs."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .compiler import write_json
from .contracts import IcpSpec, RunSummary


def create_run_dir(output_root: str | Path, icp: IcpSpec) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = _slug(icp.campaign_name)
    run_dir = Path(output_root) / slug / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ("raw_batches", "parsed_batches", "prompts"):
        (run_dir / child).mkdir(exist_ok=True)
    return run_dir


def write_summary_markdown(path: Path, summary: RunSummary, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Ollama Exa Leadgen Summary",
        "",
        f"Run dir: `{summary.run_dir}`",
        f"Target count: {summary.target_count}",
        f"Exa calls: {summary.exa_calls_succeeded}/{summary.exa_calls_planned} succeeded",
        f"Raw leads: {summary.raw_lead_count}",
        f"Deduped leads: {summary.deduped_lead_count}",
        f"Final leads: {summary.final_lead_count}",
        "",
        "## Warnings",
        "",
    ]
    if summary.warnings:
        lines.extend(f"- {warning}" for warning in summary.warnings)
    else:
        lines.append("- None")
    lines.extend(["", "## Leads", ""])
    for index, row in enumerate(rows, 1):
        lines.append(
            f"{index}. **{row.get('company_name', '')}** — {row.get('website', '')} — "
            f"score {row.get('icp_fit_score', '')} — {row.get('icp_fit_reasoning', '')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(
    path: Path,
    *,
    icp: IcpSpec,
    config_redacted: dict[str, object],
    plan: dict[str, Any],
) -> None:
    write_json(path, {"icp": icp.to_dict(), "config": config_redacted, "plan": plan})


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "campaign"
