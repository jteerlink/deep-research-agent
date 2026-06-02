"""Parse Exa responses, dedupe companies, and write final rows."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .contracts import CORE_COLUMNS, IcpSpec

LEGAL_SUFFIX_RE = re.compile(
    r"\b(inc|ltd|llc|corp|co|company|gmbh|ag|sa|sas|bv|pty|pllc)\.?\b",
    re.I,
)


def extract_companies(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract companies from known Exa structured-output response shapes."""

    candidates: list[Any] = []
    output = response.get("output")
    if isinstance(output, dict):
        candidates.append(output.get("content"))
        candidates.append(output)
    for key in ("content", "answer", "data"):
        candidates.append(response.get(key))
    candidates.append(response)
    for candidate in candidates:
        parsed = _coerce_content(candidate)
        if isinstance(parsed, dict) and isinstance(parsed.get("companies"), list):
            return [item for item in parsed["companies"] if isinstance(item, dict)]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    return []


def compile_call_results(
    call_results: list[dict[str, Any]],
    icp: IcpSpec,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    for call in call_results:
        response = call.get("response_payload") or call.get("response") or {}
        micro = call.get("micro_vertical") or {}
        call_id = str(call.get("call_id") or "call")
        for company in extract_companies(response if isinstance(response, dict) else {}):
            row = dict(company)
            row["source_micro_vertical"] = micro.get("name") or micro.get("objective") or ""
            row["source_query_variations"] = micro.get("additional_queries") or []
            row["exa_call_id"] = call_id
            row.setdefault("quality_flags", [])
            raw_rows.append(row)
    deduped, audit = dedupe_rows(raw_rows)
    sorted_rows = sorted(deduped, key=_sort_key, reverse=True)[: icp.target_count]
    return sorted_rows, {"raw_count": len(raw_rows), "deduped_count": len(deduped), **audit}


def dedupe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    for row in rows:
        key = _dedupe_key(row)
        if not key:
            key = f"row:{len(seen)}:{row.get('company_name', '')}"
        existing = seen.get(key)
        if existing is None:
            seen[key] = row
            continue
        duplicates.append(
            {
                "kept": existing.get("company_name"),
                "dropped": row.get("company_name"),
                "key": key,
            }
        )
        if _row_score(row) > _row_score(existing):
            _merge_sources(row, existing)
            seen[key] = row
        else:
            _merge_sources(existing, row)
    return list(seen.values()), {"duplicates_removed": len(duplicates), "duplicates": duplicates}


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column, "")) for column in columns})


def csv_columns(rows: list[dict[str, Any]], icp: IcpSpec) -> list[str]:
    preferred = list(CORE_COLUMNS)
    for column in icp.preferred_columns:
        if column not in preferred:
            preferred.append(column)
    for column in (
        "source_micro_vertical",
        "source_query_variations",
        "exa_call_id",
        "quality_flags",
    ):
        if column not in preferred:
            preferred.append(column)
    extras = sorted({key for row in rows for key in row if key not in preferred})
    return preferred + extras


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True, default=str)
    path.write_text(f"{content}\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, sort_keys=True, default=str) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _coerce_content(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def _dedupe_key(row: dict[str, Any]) -> str:
    host = _host(str(row.get("website") or row.get("url") or ""))
    if host:
        return f"host:{host}"
    name = normalize_name(str(row.get("company_name") or row.get("name") or ""))
    return f"name:{name}" if name else ""


def normalize_name(name: str) -> str:
    name = LEGAL_SUFFIX_RE.sub("", name)
    return re.sub(r"\s+", " ", name).strip().casefold()


def _host(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}" if url else "")
    host = parsed.netloc.lower().removeprefix("www.")
    return host


def _row_score(row: dict[str, Any]) -> float:
    try:
        return float(row.get("icp_fit_score") or row.get("fit_score") or 0)
    except (TypeError, ValueError):
        return 0.0


def _sort_key(row: dict[str, Any]) -> tuple[float, int, str]:
    completeness = sum(1 for value in row.values() if value not in (None, "", [], {}))
    return (_row_score(row), completeness, str(row.get("company_name") or ""))


def _merge_sources(keeper: dict[str, Any], dropped: dict[str, Any]) -> None:
    for field in ("source_micro_vertical", "exa_call_id"):
        values = _as_list(keeper.get(field)) + _as_list(dropped.get(field))
        keeper[field] = _dedupe_values(values)
    for field in ("source_query_variations", "quality_flags"):
        values = _as_list(keeper.get(field)) + _as_list(dropped.get(field))
        keeper[field] = _dedupe_values(values)


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _dedupe_values(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for value in values:
        key = json.dumps(value, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list | tuple):
        return " | ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)
