from __future__ import annotations

import json
from pathlib import Path

from ollama_exa_leadgen.compiler import compile_call_results, csv_columns, write_csv
from ollama_exa_leadgen.contracts import IcpSpec


def test_compiler_extracts_dedupes_and_keeps_higher_score(tmp_path: Path) -> None:
    response = json.loads(Path("tests/fixtures/fake_exa_deep_response.json").read_text())
    icp = IcpSpec.from_mapping(
        {
            "campaign_name": "dental",
            "icp_description": "DFW dental",
            "target_count": 10,
        }
    )
    call_results = [
        {
            "call_id": "exa_001",
            "micro_vertical": {"name": "core", "additional_queries": ["q1"]},
            "response_payload": response,
        }
    ]

    rows, audit = compile_call_results(call_results, icp)

    assert len(rows) == 10
    assert audit["raw_count"] == 11
    assert audit["duplicates_removed"] == 1
    assert rows[0]["company_name"] == "Alpha Dental LLC"
    assert rows[0]["icp_fit_score"] == 10
    assert rows[0]["source_micro_vertical"]

    output = tmp_path / "prospects.csv"
    write_csv(output, rows, csv_columns(rows, icp))
    assert "Alpha Dental LLC" in output.read_text()
