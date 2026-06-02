"""Top-level orchestration for plan/run/compile commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import create_run_dir, write_manifest, write_summary_markdown
from .compiler import (
    compile_call_results,
    csv_columns,
    extract_companies,
    write_csv,
    write_json,
    write_jsonl,
)
from .config import LeadgenConfig
from .contracts import IcpSpec, MicroVertical, RunSummary
from .errors import ConfigError, ProviderError, ValidationError
from .exa_client import ExaDeepClient
from .ollama_client import OllamaCloudClient
from .planner import (
    ollama_micro_verticals,
    planning_payload,
    required_exa_calls,
)
from .schema_builder import build_output_schema


def load_icp_file(path: str | Path, *, target_count: int | None = None) -> IcpSpec:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid ICP JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValidationError("ICP file must contain a JSON object")
    return IcpSpec.from_mapping(payload, target_count=target_count)


def plan_run(icp: IcpSpec, config: LeadgenConfig) -> dict[str, Any]:
    return planning_payload(icp, max_exa_calls=config.max_exa_calls)


def run_live(icp: IcpSpec, config: LeadgenConfig, *, output_dir: str | None = None) -> Path:
    calls = required_exa_calls(icp.target_count)
    if calls > config.max_exa_calls:
        raise ConfigError(
            f"run needs {calls} Exa calls, above LEADGEN_MAX_EXA_CALLS={config.max_exa_calls}"
        )
    run_dir = create_run_dir(output_dir or config.output_dir, icp)
    plan = plan_run(icp, config)
    write_manifest(
        run_dir / "run_manifest.json",
        icp=icp,
        config_redacted=config.redacted(),
        plan=plan,
    )
    write_json(run_dir / "icp.normalized.json", icp.to_dict())
    schema = build_output_schema(icp)
    write_json(run_dir / "output_schema.json", schema)

    ollama = OllamaCloudClient(config)
    exa = ExaDeepClient(config)
    exa.require_available()
    micro_verticals = _micro_verticals_for_run(icp, ollama, calls)
    write_json(run_dir / "micro_verticals.json", [item.to_dict() for item in micro_verticals])

    call_results: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, micro in enumerate(micro_verticals[:calls], 1):
        call_id = f"exa_{index:03d}"
        try:
            result = exa.search_micro_vertical(icp, micro, call_id=call_id)
        except ProviderError as exc:
            failures.append(f"{call_id}: {exc}")
            continue
        payload = result.to_dict()
        call_results.append(payload)
        write_json(run_dir / "raw_batches" / f"{call_id}.json", payload)

    rows, audit = compile_call_results(call_results, icp)
    for index, row in enumerate(rows, 1):
        row.setdefault("rank", index)
    _write_compiled_artifacts(run_dir, icp, rows, call_results, audit)
    summary = RunSummary(
        run_dir=str(run_dir),
        target_count=icp.target_count,
        exa_calls_planned=calls,
        exa_calls_succeeded=len(call_results),
        exa_calls_failed=len(failures),
        raw_lead_count=audit.get("raw_count", 0),
        deduped_lead_count=audit.get("deduped_count", 0),
        final_lead_count=len(rows),
        warnings=failures,
    )
    write_json(run_dir / "summary.json", summary.to_dict())
    write_summary_markdown(run_dir / "summary.md", summary, rows)
    return run_dir


def compile_existing(run_dir: str | Path, icp: IcpSpec | None = None) -> Path:
    path = Path(run_dir)
    if icp is None:
        manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
        icp = IcpSpec.from_mapping(manifest["icp"])
    call_results = [
        json.loads(item.read_text(encoding="utf-8"))
        for item in sorted((path / "raw_batches").glob("*.json"))
    ]
    rows, audit = compile_call_results(call_results, icp)
    _write_compiled_artifacts(path, icp, rows, call_results, audit)
    summary = RunSummary(
        run_dir=str(path),
        target_count=icp.target_count,
        exa_calls_planned=len(call_results),
        exa_calls_succeeded=len(call_results),
        raw_lead_count=audit.get("raw_count", 0),
        deduped_lead_count=audit.get("deduped_count", 0),
        final_lead_count=len(rows),
    )
    write_json(path / "summary.json", summary.to_dict())
    write_summary_markdown(path / "summary.md", summary, rows)
    return path


def _micro_verticals_for_run(
    icp: IcpSpec,
    ollama: OllamaCloudClient,
    calls: int,
) -> tuple[MicroVertical, ...]:
    desired = max(calls, 3)
    if ollama.available:
        return ollama_micro_verticals(icp, ollama, count=desired)
    raise ConfigError("OLLAMA_API_KEY is required to generate live run micro-verticals")


def _write_compiled_artifacts(
    run_dir: Path,
    icp: IcpSpec,
    rows: list[dict[str, Any]],
    call_results: list[dict[str, Any]],
    audit: dict[str, Any],
) -> None:
    raw_rows: list[dict[str, Any]] = []
    for call in call_results:
        for row in extract_companies(call.get("response_payload") or {}):
            raw_rows.append(row)
    write_jsonl(run_dir / "prospects.raw.jsonl", raw_rows)
    write_json(run_dir / "prospects.deduped.json", rows)
    write_json(run_dir / "dedupe_audit.json", audit)
    write_json(run_dir / "score_distribution.json", _score_distribution(rows))
    write_csv(run_dir / "prospects.csv", rows, csv_columns(rows, icp))


def _score_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    buckets = {"8-10": 0, "5-7": 0, "1-4": 0, "unknown": 0}
    for row in rows:
        try:
            score = float(row.get("icp_fit_score") or 0)
        except (TypeError, ValueError):
            buckets["unknown"] += 1
            continue
        if score >= 8:
            buckets["8-10"] += 1
        elif score >= 5:
            buckets["5-7"] += 1
        elif score >= 1:
            buckets["1-4"] += 1
        else:
            buckets["unknown"] += 1
    return buckets
