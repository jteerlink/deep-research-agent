from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ollama_exa_leadgen.runner import compile_existing

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "dental-dfw-icp.json"
FIXTURE = ROOT / "tests" / "fixtures" / "fake_exa_deep_response.json"


def test_cli_plan_runs_without_api_keys() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ollama_exa_leadgen",
            "plan",
            "--icp-file",
            str(EXAMPLE),
            "--json",
        ],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["target_count"] == 10
    assert payload["estimated_exa_calls"] == 1
    assert payload["within_cap"] is True


def test_compile_existing_run_from_raw_batches(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    raw = run_dir / "raw_batches"
    raw.mkdir(parents=True)
    icp_payload = json.loads(EXAMPLE.read_text())
    (run_dir / "run_manifest.json").write_text(json.dumps({"icp": icp_payload}))
    response = json.loads(FIXTURE.read_text())
    (raw / "exa_001.json").write_text(
        json.dumps(
            {
                "call_id": "exa_001",
                "micro_vertical": {"name": "core", "additional_queries": ["q1"]},
                "response_payload": response,
            }
        )
    )

    compile_existing(run_dir)

    csv_path = run_dir / "prospects.csv"
    summary_path = run_dir / "summary.json"
    raw_jsonl_path = run_dir / "prospects.raw.jsonl"
    assert csv_path.exists()
    assert summary_path.exists()
    raw_rows = [json.loads(line) for line in raw_jsonl_path.read_text().splitlines()]
    assert len(raw_rows) == 11
    assert json.loads(summary_path.read_text())["final_lead_count"] == 10
