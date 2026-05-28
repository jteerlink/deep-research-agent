from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from deep_research_agent.tiered_models import ApprovedProspectSelection
from deep_research_agent.tiered_runtime import (
    parse_directive_payload,
    resume_tiered_research,
    run_tiered_research,
)
from deep_research_agent.ui import (
    build_prospect_directive,
    build_tiered_preview,
    env_file_overlay,
    final_enrichment_csv_bytes,
    flatten_enriched_prospect_rows,
    flatten_tiered_prospect_rows,
    model_preflight_from_env_file,
    prepare_exa_enrichment_selection,
    preview_geography_scope,
    provider_env_from_env_file,
    provider_env_overlay,
    qualification_audit_rows,
    qualification_summary,
    result_preview,
    selection_from_prospect_rows,
    temporary_env,
    tiered_early_discovery_searcher,
    tiered_review_state,
)


def _tiered_checkpoint(tmp_path):
    directive = parse_directive_payload(
        {
            "industry": "dental",
            "geography": "North Texas",
            "target_prospect_count": 2,
            "research_criteria": "multi-location Invisalign",
            "preferred_contact_roles": ["owner"],
        }
    )
    return run_tiered_research(
        directive,
        thread_id="ui-tiered",
        checkpoint_dir=tmp_path / "checkpoints",
        artifact_dir=tmp_path / "artifacts",
        mock_results=(
            "Acme Dental|https://acme.example|Strong fit|mock",
            "Beta Dental|https://beta.example|Strong fit|mock",
        ),
    )


def test_provider_env_overlay_keeps_only_non_empty_supported_keys() -> None:
    overlay = provider_env_overlay(
        {
            "FIRECRAWL_API_KEY": " fc-test ",
            "SERPER_API_KEY": "",
            "OPENAI_API_KEY": "not-a-search-key",
        }
    )

    assert overlay == {"FIRECRAWL_API_KEY": "fc-test"}


def test_provider_env_from_env_file_preserves_shell_precedence(tmp_path, monkeypatch) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "TAVILY_API_KEY=from-dotenv\nFIRECRAWL_API_KEY=from-dotenv-firecrawl\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TAVILY_API_KEY", "from-shell")

    overlay = provider_env_from_env_file(dotenv)

    assert overlay == {
        "TAVILY_API_KEY": "from-shell",
        "FIRECRAWL_API_KEY": "from-dotenv-firecrawl",
    }


def test_env_file_overlay_loads_non_exported_runtime_values(tmp_path, monkeypatch) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "TAVILY_API_KEY=from-dotenv\n"
        "SSL_CERT_FILE=/tmp/corp-ca.pem\n"
        "REQUESTS_CA_BUNDLE=/tmp/requests-ca.pem\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TAVILY_API_KEY", "from-shell")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)

    overlay = env_file_overlay(dotenv)

    assert overlay == {
        "SSL_CERT_FILE": "/tmp/corp-ca.pem",
        "REQUESTS_CA_BUNDLE": "/tmp/requests-ca.pem",
    }


def test_model_preflight_from_env_file_uses_runtime_overlay(tmp_path, monkeypatch) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("OLLAMA_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)

    payload = model_preflight_from_env_file(dotenv)

    assert payload["live_model_available"] is True
    assert payload["selected_provider"] == "ollama_native"
    assert "from-dotenv" not in str(payload)


def test_build_prospect_directive_includes_business_target_context() -> None:
    assert build_prospect_directive(
        "dental practices",
        "Dallas-Fort Worth",
        "patient reactivation opportunity",
    ) == (
        "industry: dental practices\n"
        "geography: Dallas-Fort Worth\n"
        "criteria: patient reactivation opportunity"
    )


def test_build_tiered_preview_is_offline_and_query_shaped() -> None:
    preview = build_tiered_preview(
        "dental",
        "DFW area",
        "multi-location",
        target_prospect_count=3,
        preferred_contact_roles=("owner",),
    )

    assert preview["directive"]["target_prospect_count"] == 3
    assert preview["search_dependency"] == "injected"
    assert preview["provider_policy"]["final_enrichment"] == ["exa"]
    assert "exa" not in preview["provider_policy"]["early_discovery"]
    assert [lane["family"] for lane in preview["company_discovery_lanes"]] == [
        "official_site",
        "local_directory",
        "industry_context",
    ]
    assert preview["company_discovery_queries"][:2] == [
        "dental companies in DFW area",
        "best dental DFW area",
    ]
    assert "Example Company owner" in preview["contact_discovery_query_templates"]
    assert preview["warnings"]


def test_tiered_early_discovery_searcher_excludes_exa() -> None:
    searcher = tiered_early_discovery_searcher(timeout=3)

    assert "exa" not in [provider.name for provider in searcher.providers]
    assert "duckduckgo" in [provider.name for provider in searcher.providers]


def test_build_tiered_preview_waits_for_required_fields() -> None:
    preview = build_tiered_preview(
        "",
        " ",
        "multi-location",
        target_prospect_count=3,
    )

    assert preview["ready"] is False
    assert preview["missing_fields"] == ["industry", "geographic_area"]
    assert preview["directive"]["research_criteria"] == "multi-location"
    assert preview["company_discovery_queries"] == []
    assert preview["provider_policy"]["final_enrichment"] == ["exa"]
    assert "waiting for industry" in preview["warnings"][0]


def test_flatten_tiered_prospect_rows_uses_company_level_identity(tmp_path) -> None:
    checkpoint = _tiered_checkpoint(tmp_path)
    selected = {checkpoint.run.companies[0].company_id}

    rows = flatten_tiered_prospect_rows(checkpoint, selected_row_ids=selected)

    assert [row["row_id"] for row in rows] == [
        checkpoint.run.companies[0].company_id,
        checkpoint.run.companies[1].company_id,
    ]
    assert rows[0]["selected"] is True
    assert rows[1]["selected"] is False
    assert rows[0]["row_type"] == "company"
    assert rows[0]["company_name"] == "Acme Dental"
    assert rows[0]["contact_count"] == 1
    assert rows[0]["contact_ids"] == [checkpoint.run.contacts[0].contact_id]
    assert rows[0]["contact_name"] == "Review Contact at Acme Dental"
    assert "review placeholder" in rows[0]["personalization_summary"]


def test_tiered_review_state_distinguishes_ready_and_empty_review(tmp_path) -> None:
    ready = _tiered_checkpoint(tmp_path)
    empty = run_tiered_research(
        ready.run.directive,
        thread_id="ui-tiered-empty",
        checkpoint_dir=tmp_path / "checkpoints",
        artifact_dir=tmp_path / "artifacts",
    )

    assert tiered_review_state(ready) == {
        "status": "review_required",
        "prospect_count": 2,
        "ready_prospect_count": 2,
        "ready_company_count": 2,
        "company_prospect_count": 2,
        "ready_contact_count": 2,
        "contact_candidate_count": 2,
        "qualified_company_count": 2,
        "companies_with_contacts_count": 2,
        "needs_contact_count": 0,
        "rejected_candidate_count": 0,
        "enrichment_count": 0,
        "review_ready": True,
        "empty_review": False,
    }
    assert tiered_review_state(empty) == {
        "status": "review_required",
        "prospect_count": 0,
        "ready_prospect_count": 0,
        "ready_company_count": 0,
        "company_prospect_count": 0,
        "ready_contact_count": 0,
        "contact_candidate_count": 0,
        "qualified_company_count": 0,
        "companies_with_contacts_count": 0,
        "needs_contact_count": 0,
        "rejected_candidate_count": 0,
        "enrichment_count": 0,
        "review_ready": False,
        "empty_review": True,
    }


def test_tiered_review_state_counts_ready_needs_contact_and_rejected() -> None:
    payload = {
        "status": "review_required",
        "companies": [
            {"company_id": "company_ready", "name": "Ready Dental", "website": "https://ready.example"},
            {"company_id": "company_needs", "name": "Needs Contact Dental"},
        ],
        "contacts": [
            {
                "company_id": "company_ready",
                "contact_id": "contact_ready",
                "name": "Jane Smith",
                "title": "Owner",
                "contact_confidence": 0.86,
            },
            {
                "company_id": "company_needs",
                "contact_id": "contact_placeholder",
                "name": "Review Contact at Needs Contact Dental",
                "title": "review contact",
                "contact_confidence": 0.2,
            },
        ],
        "qualification_audit": [
            {
                "tier": "contact",
                "status": "accepted",
                "company_id": "company_ready",
                "contact_id": "contact_ready",
                "company_name": "Ready Dental",
                "contact_name": "Jane Smith",
            },
            {
                "tier": "company",
                "status": "needs_contact",
                "company_id": "company_needs",
                "company_name": "Needs Contact Dental",
                "reasons": ["no_verified_contact"],
            },
            {
                "tier": "company",
                "status": "rejected",
                "source_title": "Best dental practices in Dallas",
                "source_url": "https://directory.example/best",
                "reasons": ["listicle_or_directory"],
            },
        ],
    }

    rows = flatten_tiered_prospect_rows(
        payload,
        selected_row_ids={"company_ready", "company_needs"},
    )

    assert rows == [
        {
            "selected": True,
            "selectable": True,
            "row_type": "company",
            "row_id": "company_ready",
            "company_id": "company_ready",
            "contact_id": "contact_ready",
            "contact_ids": ["contact_ready"],
            "company_name": "Ready Dental",
            "website": "https://ready.example",
            "fit_score": "",
            "contact_count": 1,
            "contact_names": "Jane Smith, Owner",
            "contact_name": "Jane Smith",
            "contact_title": "Owner",
            "contact_confidence": 0.86,
            "personalization_summary": "",
        },
        {
            "selected": True,
            "selectable": True,
            "row_type": "company",
            "row_id": "company_needs",
            "company_id": "company_needs",
            "contact_id": "",
            "contact_ids": [],
            "company_name": "Needs Contact Dental",
            "website": "",
            "fit_score": "",
            "contact_count": 0,
            "contact_names": "",
            "contact_name": "",
            "contact_title": "",
            "contact_confidence": "",
            "personalization_summary": "",
        },
    ]
    assert qualification_summary(payload) == {
        "ready_company_count": 2,
        "company_prospect_count": 2,
        "ready_contact_count": 1,
        "contact_candidate_count": 2,
        "qualified_company_count": 2,
        "companies_with_contacts_count": 1,
        "needs_contact_count": 1,
        "rejected_candidate_count": 1,
    }
    assert tiered_review_state(payload) == {
        "status": "review_required",
        "prospect_count": 2,
        "ready_prospect_count": 2,
        "ready_company_count": 2,
        "company_prospect_count": 2,
        "ready_contact_count": 1,
        "contact_candidate_count": 2,
        "qualified_company_count": 2,
        "companies_with_contacts_count": 1,
        "needs_contact_count": 1,
        "rejected_candidate_count": 1,
        "enrichment_count": 0,
        "review_ready": True,
        "empty_review": False,
    }
    audit_rows = qualification_audit_rows(payload)
    assert [row["status"] for row in audit_rows] == ["needs_contact", "rejected"]
    assert all(row["selectable"] is False for row in audit_rows)
    assert audit_rows[0]["reasons"] == "no_verified_contact"


def test_selected_prospect_rows_build_approval_for_checked_pairs(tmp_path) -> None:
    rows = flatten_tiered_prospect_rows(_tiered_checkpoint(tmp_path))
    rows[0]["selected"] = True
    rows[1]["selected"] = False

    approval = selection_from_prospect_rows(rows, reviewer="tester", approved_at="now")

    assert approval.approved_company_ids == (rows[0]["company_id"],)
    assert approval.approved_contact_ids == (rows[0]["contact_id"],)
    assert approval.reviewer == "tester"
    assert approval.approved_at == "now"


def test_selected_company_prospect_without_contact_builds_company_only_approval() -> None:
    rows = [
        {
            "selected": True,
            "selectable": True,
            "company_id": "company_needs",
            "contact_ids": [],
        }
    ]

    approval = selection_from_prospect_rows(rows, reviewer="tester", approved_at="now")

    assert approval.approved_company_ids == ("company_needs",)
    assert approval.approved_contact_ids == ()


def test_missing_exa_key_blocks_ui_enrichment_without_checkpoint_write(tmp_path) -> None:
    checkpoint = _tiered_checkpoint(tmp_path)
    rows = flatten_tiered_prospect_rows(checkpoint)
    rows[0]["selected"] = True

    approval, error = prepare_exa_enrichment_selection(rows, {})

    assert approval is None
    assert error == "EXA_API_KEY is required to run final Exa enrichment for selected prospects."
    assert resume_tiered_research(
        checkpoint.thread_id,
        checkpoint_dir=tmp_path / "checkpoints",
        artifact_dir=tmp_path / "artifacts",
    ).final_enrichment == ()


def test_enriched_rows_join_display_labels_and_csv_bytes(tmp_path) -> None:
    checkpoint = _tiered_checkpoint(tmp_path)
    company_id = checkpoint.run.companies[0].company_id
    contact_id = checkpoint.run.contacts[0].contact_id
    approval = ApprovedProspectSelection(
        approved_company_ids=(company_id,),
        approved_contact_ids=(contact_id,),
    )
    enriched = resume_tiered_research(
        checkpoint.thread_id,
        checkpoint_dir=tmp_path / "checkpoints",
        artifact_dir=tmp_path / "artifacts",
        approval_selection=approval,
        enable_final_enrichment=True,
        mock_final_enrichment=(f"{company_id}|{contact_id}|Approved enrichment only|exa",),
    )

    rows = flatten_enriched_prospect_rows(enriched)
    csv_bytes = final_enrichment_csv_bytes(enriched)

    assert rows == [
        {
            "enrichment_id": "final_001",
            "provider": "exa",
            "company_id": company_id,
            "company_name": "Acme Dental",
            "website": "https://acme.example",
            "contact_id": contact_id,
            "contact_name": "Review Contact at Acme Dental",
            "contact_title": "owner",
            "summary": "Approved enrichment only",
            "evidence_ids": "ev_final_001",
            "warnings": "",
        }
    ]
    assert b"final_001" in csv_bytes
    assert b"Approved enrichment only" in csv_bytes


def test_preview_geography_scope_expands_known_regions() -> None:
    preview = preview_geography_scope("North Texas")

    assert preview["canonical"] == "Dallas-Fort Worth TX"
    assert preview["search_terms"][:3] == ["North Texas", "Dallas-Fort Worth TX", "DFW"]


def test_temporary_env_restores_secret_values(monkeypatch) -> None:
    monkeypatch.setenv("FIRECRAWL_API_KEY", "original")

    with temporary_env({"FIRECRAWL_API_KEY": "session-only", "SERPER_API_KEY": "serper"}):
        assert os.environ["FIRECRAWL_API_KEY"] == "session-only"
        assert os.environ["SERPER_API_KEY"] == "serper"

    assert os.environ["FIRECRAWL_API_KEY"] == "original"
    assert "SERPER_API_KEY" not in os.environ


def test_result_preview_reads_markdown_and_limits_lists(tmp_path) -> None:
    markdown_path = tmp_path / "artifact.md"
    markdown_path.write_text("# Artifact\n\nPreview text", encoding="utf-8")
    state = {
        "thread_id": "ui-thread",
        "status": "completed",
        "warnings": ["warn"],
        "events": [{"event": f"e{i}"} for i in range(30)],
        "evidence": [{"id": str(i)} for i in range(12)],
        "prospect_targets": [{"name": str(i)} for i in range(12)],
        "artifact_paths": {"markdown": str(markdown_path)},
    }

    preview = result_preview(state)

    assert preview["thread_id"] == "ui-thread"
    assert preview["event_count"] == 30
    assert len(preview["events"]) == 25
    assert len(preview["evidence"]) == 10
    assert len(preview["prospect_targets"]) == 10
    assert preview["markdown_preview"].startswith("# Artifact")


def test_ui_imports_when_loaded_as_streamlit_script() -> None:
    path = Path(__file__).resolve().parents[1] / "deep_research_agent" / "ui.py"
    spec = importlib.util.spec_from_file_location("streamlit_script_ui", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    assert module.DEFAULT_CHECKPOINT_DIR == ".deep_research_agent/checkpoints"
