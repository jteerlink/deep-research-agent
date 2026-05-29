from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _isolated_env() -> dict[str, str]:
    env = {"PYTHONPATH": str(ROOT)}
    if os.environ.get("PATH"):
        env["PATH"] = os.environ["PATH"]
    return env


def test_module_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "deep_research_agent", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Local-first deep research agent foundation" in result.stdout
    assert "config" in result.stdout
    assert "run" in result.stdout
    assert "resume" in result.stdout
    assert "inspect" in result.stdout
    assert "tiered-run" in result.stdout
    assert "tiered-resume" in result.stdout
    assert "tiered-inspect" in result.stdout
    assert "ui" in result.stdout


def test_cli_config_json() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "deep_research_agent", "config", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"primary_provider"' in result.stdout
    assert '"ollama_native"' in result.stdout


def test_legacy_module_cli_help_does_not_run_demo() -> None:
    result = subprocess.run(
        [sys.executable, "async_multi_search.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Run the async multi-provider search fallback chain" in result.stdout
    assert "--max-results" in result.stdout
    assert "AllProvidersFailedError" not in result.stderr


def test_cli_config_json_redacts_api_keys() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "deep_research_agent", "config", "--json"],
        check=True,
        capture_output=True,
        text=True,
        env={"OPENAI_API_KEY": "secret-openai", "CODEX_API_KEY": "secret-codex"},
    )

    assert "secret-openai" not in result.stdout
    assert "secret-codex" not in result.stdout
    assert "<redacted>" in result.stdout


def test_cli_search_providers_lists_env_example_order_without_brave() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "deep_research_agent", "search-providers"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == [
        "tavily: TAVILY_API_KEY",
        "exa: EXA_API_KEY",
        "serper: SERPER_API_KEY",
        "firecrawl: FIRECRAWL_API_KEY",
        "ydc: YDC_API_KEY",
        "duckduckgo: no key required",
    ]


def test_cli_tiered_preview_json_is_offline_and_query_shaped() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deep_research_agent",
            "tiered-preview",
            "--industry",
            "dental",
            "--geography",
            "DFW area",
            "--criteria",
            "multi-location",
            "--preferred-contact-role",
            "owner",
            "--source-preference",
            "directories",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["directive"]["industry"] == "dental"
    assert payload["directive"]["geographic_area"] == "DFW area"
    assert payload["tiers"]["company_discovery"]["search_dependency"] == "injected"
    assert payload["tiers"]["company_discovery"][
        "excludes_default_async_multi_provider_chain"
    ] is True
    assert payload["provider_policy"]["final_enrichment"] == ["exa"]
    assert "exa" not in payload["provider_policy"]["early_discovery"]
    assert [lane["family"] for lane in payload["tiers"]["company_discovery"]["lanes"]][:3] == [
        "official_site",
        "local_directory",
        "industry_context",
    ]
    assert "dental companies in DFW area" in payload["tiers"]["company_discovery"]["queries"]
    assert "tiered_prospect_research.json" in payload["artifact_plan"]


def test_cli_tiered_run_resume_inspect_with_mock_data(tmp_path) -> None:
    directive_path = tmp_path / "directive.json"
    checkpoint_dir = tmp_path / "checkpoints"
    artifact_dir = tmp_path / "artifacts"
    directive_path.write_text(
        json.dumps(
            {
                "industry": "dental",
                "geography": "DFW area",
                "target_prospect_count": 1,
                "research_criteria": "multi-location Invisalign",
                "preferred_contact_roles": ["owner"],
            }
        )
    )

    run_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deep_research_agent",
            "tiered-run",
            "--directive-json",
            str(directive_path),
            "--thread-id",
            "tiered-test",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--artifact-dir",
            str(artifact_dir),
            "--mock-result",
            "Acme Dental|https://acme.example|Multi-location dental group|mock",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    run_payload = json.loads(run_result.stdout)
    assert run_payload["checkpoint_schema_version"] == "tiered.prospect_checkpoint.v1"
    assert run_payload["status"] == "review_required"
    assert run_payload["directive"]["geographic_area"] == "Dallas-Fort Worth TX"
    assert Path(run_payload["artifact_paths"]["json"]).exists()
    company_id = run_payload["companies"][0]["company_id"]
    assert run_payload["contacts"] == []
    assert run_payload["metadata"]["needs_contact_count"] == 1

    inspect_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deep_research_agent",
            "tiered-inspect",
            "tiered-test",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(inspect_result.stdout)["thread_id"] == "tiered-test"

    approval_path = tmp_path / "approval.json"
    approval_path.write_text(
        json.dumps(
            {
                "approved_company_ids": [company_id],
                "approved_contact_ids": [],
                "reviewer": "tester",
            }
        )
    )
    resume_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deep_research_agent",
            "tiered-resume",
            "tiered-test",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--artifact-dir",
            str(artifact_dir),
            "--approve-selection",
            str(approval_path),
            "--enable-final-enrichment",
            "--mock-final-enrichment",
            f"{company_id}||Approved enrichment only|mock",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    resume_payload = json.loads(resume_result.stdout)
    assert resume_payload["status"] == "final_enrichment_complete"
    assert resume_payload["approval"]["approved_company_ids"] == [company_id]
    assert resume_payload["final_enrichment"][0]["company_id"] == company_id
    assert Path(resume_payload["artifact_paths"]["final_enrichment_csv"]).exists()


def test_cli_tiered_run_invalid_directive_json_has_no_traceback(tmp_path) -> None:
    directive_path = tmp_path / "bad.json"
    directive_path.write_text("{not json")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deep_research_agent",
            "tiered-run",
            "--directive-json",
            str(directive_path),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "invalid directive JSON" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_model_status_json_reports_redacted_missing_ollama_key(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "deep_research_agent", "model-status", "--json"],
        check=True,
        capture_output=True,
        text=True,
        env=_isolated_env(),
        cwd=tmp_path,
    )

    payload = json.loads(result.stdout)
    assert payload["primary_provider"] == "ollama_native"
    assert payload["live_model_available"] is False
    assert payload["providers"][0]["unavailable_reason"] == "missing_api_key"
    assert "secret" not in result.stdout.lower()


def test_cli_run_require_live_model_fails_before_mock_search_without_stacktrace(
    tmp_path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deep_research_agent",
            "run",
            "industry: HVAC\ngeography: North Texas",
            "--require-live-model",
            "--mock-result",
            "Actual HVAC|https://actualhvac.com|We provide AC repair|tavily",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_isolated_env(),
        cwd=tmp_path,
    )

    assert result.returncode != 0
    assert "Live model required" in result.stderr
    assert "Traceback" not in result.stderr
