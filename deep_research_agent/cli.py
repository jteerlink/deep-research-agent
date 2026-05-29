"""Command-line interface for the deep research agent foundation."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .config import load_config
from .models import build_model_client
from .tiered_runtime import (
    DEFAULT_TIERED_ARTIFACT_DIR,
    DEFAULT_TIERED_CHECKPOINT_DIR,
    inspect_tiered_research,
    load_approval_selection,
    load_directive_json,
    parse_directive_payload,
    resume_tiered_research,
    run_tiered_research,
)
from .tiered_search import (
    DEFAULT_TARGET_PROSPECT_COUNT,
    CompanySearchTarget,
    ContactSearchTarget,
    ProviderPolicy,
    TieredSearchDirective,
    build_company_discovery_lanes,
    build_company_discovery_queries,
    build_contact_discovery_queries,
    build_personalization_queries,
)

REMOVED_COMMAND_MIGRATIONS = {
    "run": "tiered-run",
    "resume": "tiered-resume",
    "inspect": "tiered-inspect",
}


def _removed_command_message(command: str) -> str:
    replacement = REMOVED_COMMAND_MIGRATIONS[command]
    return (
        f"The legacy `{command}` workflow command was removed. "
        f"Use `{replacement}` for the supported tiered prospect workflow. "
        "Run `python -m deep_research_agent --help` for current commands."
    )



def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if "key" in key.lower() and item:
                redacted[key] = "<redacted>"
            else:
                redacted[key] = _redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deep-research-agent",
        description="Local-first deep research agent foundation utilities.",
    )
    subparsers = parser.add_subparsers(dest="command")

    config_parser = subparsers.add_parser("config", help="Print resolved configuration.")
    config_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit configuration as JSON instead of a short summary.",
    )
    config_parser.add_argument(
        "--show-secrets",
        action="store_true",
        help="Include raw API keys in config JSON output. Redacted by default.",
    )

    subparsers.add_parser(
        "search-providers",
        help="List async_multi_search.py provider order and required environment keys.",
    )

    tiered_preview_parser = subparsers.add_parser(
        "tiered-preview",
        help=(
            "Preview the tiered prospect directive and early search queries "
            "without network calls."
        ),
    )
    tiered_preview_parser.add_argument("--industry", required=True, help="Target industry/niche.")
    tiered_preview_parser.add_argument("--geography", required=True, help="Target geography.")
    tiered_preview_parser.add_argument(
        "--criteria",
        default="",
        help="Freeform prospecting criteria to preserve in the directive.",
    )
    tiered_preview_parser.add_argument(
        "--target-prospect-count",
        type=int,
        default=DEFAULT_TARGET_PROSPECT_COUNT,
        help="Requested company count after qualification.",
    )
    tiered_preview_parser.add_argument(
        "--preferred-contact-role",
        action="append",
        default=[],
        help="Preferred contact role; repeat for multiple roles.",
    )
    tiered_preview_parser.add_argument(
        "--source-preference",
        action="append",
        default=[],
        help="Preferred source family; repeat for multiple source hints.",
    )
    tiered_preview_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit preview as JSON instead of a text summary.",
    )

    tiered_run_parser = subparsers.add_parser(
        "tiered-run",
        help="Run the additive tiered prospect workflow to the review gate.",
    )
    tiered_run_parser.add_argument("--directive-json", help="Path to tiered directive JSON.")
    tiered_run_parser.add_argument("--industry", help="Target industry/niche.")
    tiered_run_parser.add_argument("--geography", help="Target geography.")
    tiered_run_parser.add_argument("--criteria", help="Freeform prospecting criteria.")
    tiered_run_parser.add_argument(
        "--target-prospect-count",
        type=int,
        default=DEFAULT_TARGET_PROSPECT_COUNT,
        help="Requested company count after qualification.",
    )
    tiered_run_parser.add_argument(
        "--preferred-contact-role",
        action="append",
        default=[],
        help="Preferred contact role; repeat for multiple roles.",
    )
    tiered_run_parser.add_argument("--thread-id", help="Durable tiered thread id.")
    tiered_run_parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_TIERED_CHECKPOINT_DIR,
        help="Directory for tiered JSON checkpoints.",
    )
    tiered_run_parser.add_argument(
        "--artifact-dir",
        default=DEFAULT_TIERED_ARTIFACT_DIR,
        help="Directory for tiered JSON/CSV/Markdown artifacts.",
    )
    tiered_run_parser.add_argument(
        "--mock-result",
        action="append",
        default=[],
        help="Add a mocked company result as title|url|snippet|provider.",
    )
    tiered_run_parser.add_argument("--json", action="store_true", help="Emit run state as JSON.")

    tiered_resume_parser = subparsers.add_parser(
        "tiered-resume",
        help="Resume a tiered workflow checkpoint after review.",
    )
    tiered_resume_parser.add_argument("thread_id", help="Tiered thread id to resume.")
    tiered_resume_parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_TIERED_CHECKPOINT_DIR,
        help="Directory for tiered JSON checkpoints.",
    )
    tiered_resume_parser.add_argument(
        "--artifact-dir",
        default=DEFAULT_TIERED_ARTIFACT_DIR,
        help="Directory for tiered JSON/CSV/Markdown artifacts.",
    )
    tiered_resume_parser.add_argument(
        "--approve-selection",
        help="Path to approved company/contact selection JSON.",
    )
    tiered_resume_parser.add_argument(
        "--enable-final-enrichment",
        action="store_true",
        help="Allow final enrichment for approved IDs only.",
    )
    tiered_resume_parser.add_argument(
        "--mock-final-enrichment",
        action="append",
        default=[],
        help="Add mock final enrichment as company_id|contact_id|summary|provider.",
    )
    tiered_resume_parser.add_argument("--json", action="store_true", help="Emit state as JSON.")

    tiered_inspect_parser = subparsers.add_parser(
        "tiered-inspect",
        help="Inspect a tiered workflow checkpoint.",
    )
    tiered_inspect_parser.add_argument("thread_id", nargs="?", help="Tiered thread id.")
    tiered_inspect_parser.add_argument("--thread-id", dest="thread_id_option")
    tiered_inspect_parser.add_argument(
        "--checkpoint-dir",
        default=DEFAULT_TIERED_CHECKPOINT_DIR,
        help="Directory for tiered JSON checkpoints.",
    )
    tiered_inspect_parser.add_argument("--json", action="store_true", help="Emit state as JSON.")

    model_status_parser = subparsers.add_parser(
        "model-status", help="Show redacted live model availability diagnostics."
    )
    model_status_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit model status as JSON instead of a short summary.",
    )
    model_status_parser.add_argument(
        "--live-smoke",
        action="store_true",
        help="Run a live structured-output smoke call against the first available provider.",
    )

    subparsers.add_parser("ui", help="Launch the local Streamlit research UI.")
    return parser


def _print_json(payload: Any) -> None:
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))


def _print_model_status(payload: dict[str, Any]) -> None:
    print(f"primary_provider={payload.get('primary_provider', '')}")
    print(f"live_model_available={str(payload.get('live_model_available', False)).lower()}")
    selected = payload.get("selected_provider") or "none"
    print(f"selected_provider={selected}")
    for provider in payload.get("providers", []):
        status = "available" if provider.get("available") else provider.get("unavailable_reason")
        print(
            f"{provider.get('provider')}: {status}; "
            f"model_set={str(bool(provider.get('model'))).lower()}; "
            f"base_url_set={str(bool(provider.get('base_url'))).lower()}; "
            f"api_key_configured={str(bool(provider.get('api_key_configured'))).lower()}"
        )
    live_smoke = payload.get("live_smoke")
    if live_smoke:
        structured = live_smoke.get("structured") or {}
        print(f"live_smoke_status={structured.get('status') or structured.get('ok')}")


def build_tiered_preview_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Build an offline tiered prospect preview for CLI/UI wiring."""

    directive = TieredSearchDirective(
        industry=args.industry,
        geographic_area=args.geography,
        target_prospect_count=args.target_prospect_count,
        research_criteria=args.criteria,
        preferred_contact_roles=tuple(args.preferred_contact_role or ()),
        source_preferences=tuple(args.source_preference or ()),
    )
    example_company = CompanySearchTarget("Example Company", website="https://example.com")
    example_contact = ContactSearchTarget(
        "Example Contact",
        "Example Company",
        title=(
            directive.preferred_contact_roles[0]
            if directive.preferred_contact_roles
            else "Owner"
        ),
    )
    return {
        "directive": asdict(directive),
        "tiers": {
            "company_discovery": {
                "queries": list(build_company_discovery_queries(directive)),
                "lanes": [
                    lane.to_dict() for lane in build_company_discovery_lanes(directive)
                ],
                "search_dependency": "injected",
                "excludes_default_async_multi_provider_chain": True,
            },
            "contact_discovery": {
                "example_company": asdict(example_company),
                "query_templates": list(
                    build_contact_discovery_queries(directive, example_company)
                ),
                "search_dependency": "injected",
            },
            "personalization": {
                "example_contact": asdict(example_contact),
                "query_templates": list(
                    build_personalization_queries(directive, example_contact)
                ),
                "search_dependency": "injected",
            },
        },
        "warnings": [
            "Preview only: no network search, browser capture, enrichment, "
            "or outreach is executed.",
            "Early tiered discovery requires an injected search callable and excludes "
            "the default async search provider chain.",
        ],
        "provider_policy": ProviderPolicy().to_dict(),
        "artifact_plan": [
            "tiered_prospect_research.json",
            "companies.csv",
            "contacts.csv",
            "personalization.csv",
            "research_report.md",
        ],
    }


def _print_tiered_preview(payload: Mapping[str, Any]) -> None:
    directive = payload["directive"]
    print(
        "directive="
        f"{directive['industry']} | {directive['geographic_area']} | "
        f"target={directive['target_prospect_count']}"
    )
    tiers = payload["tiers"]
    for tier_name, tier_payload in tiers.items():
        queries = tier_payload.get("queries") or tier_payload.get("query_templates") or []
        print(f"{tier_name}:")
        for query in queries:
            print(f"- {query}")
    for warning in payload["warnings"]:
        print(f"warning: {warning}")


def _print_tiered_checkpoint(payload: Mapping[str, Any]) -> None:
    print(f"thread_id={payload.get('thread_id', '')}")
    print(f"status={payload.get('status', '')}")
    artifact_paths = payload.get("artifact_paths") or {}
    if isinstance(artifact_paths, Mapping):
        for name, path in artifact_paths.items():
            print(f"{name}={path}")


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] in REMOVED_COMMAND_MIGRATIONS:
        print(_removed_command_message(raw_args[0]), file=sys.stderr)
        return 2

    parser = build_parser()
    args = parser.parse_args(raw_args)

    if args.command == "config":
        config = load_config()
        if args.json:
            payload = asdict(config)
            if not args.show_secrets:
                payload = _redact_secrets(payload)
            print(json.dumps(payload, indent=2, default=str))
        else:
            print(f"primary_provider={config.primary_provider.value}")
            print(f"primary_model={config.primary_model}")
        return 0

    if args.command == "search-providers":
        from async_multi_search import AsyncMultiProviderSearch

        for provider in AsyncMultiProviderSearch().providers:
            print(f"{provider.name}: {provider.env_key or 'no key required'}")
        return 0

    if args.command == "tiered-preview":
        try:
            payload = build_tiered_preview_payload(args)
        except ValueError as exc:
            parser.error(str(exc))
        if args.json:
            _print_json(payload)
        else:
            _print_tiered_preview(payload)
        return 0

    if args.command == "tiered-run":
        try:
            if args.directive_json:
                directive = load_directive_json(args.directive_json)
            else:
                directive = parse_directive_payload(
                    {
                        "industry": args.industry or "",
                        "geography": args.geography or "",
                        "target_prospect_count": args.target_prospect_count,
                        "research_criteria": args.criteria or "",
                        "preferred_contact_roles": args.preferred_contact_role or (),
                    }
                )
            tiered_checkpoint = run_tiered_research(
                directive,
                thread_id=args.thread_id,
                checkpoint_dir=args.checkpoint_dir,
                artifact_dir=args.artifact_dir,
                mock_results=tuple(args.mock_result or ()),
            )
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        payload = tiered_checkpoint.to_dict()
        if args.json:
            _print_json(payload)
        else:
            _print_tiered_checkpoint(payload)
        return 0

    if args.command == "tiered-resume":
        try:
            approval = (
                load_approval_selection(args.approve_selection)
                if args.approve_selection
                else None
            )
            tiered_checkpoint = resume_tiered_research(
                args.thread_id,
                checkpoint_dir=args.checkpoint_dir,
                artifact_dir=args.artifact_dir,
                approval_selection=approval,
                enable_final_enrichment=args.enable_final_enrichment,
                mock_final_enrichment=tuple(args.mock_final_enrichment or ()),
            )
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        payload = tiered_checkpoint.to_dict()
        if args.json:
            _print_json(payload)
        else:
            _print_tiered_checkpoint(payload)
        return 0

    if args.command == "tiered-inspect":
        thread_id = args.thread_id_option or args.thread_id
        if not thread_id:
            parser.error("tiered-inspect requires a thread id")
        try:
            tiered_checkpoint = inspect_tiered_research(
                thread_id,
                checkpoint_dir=args.checkpoint_dir,
            )
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        payload = tiered_checkpoint.to_dict()
        if args.json:
            _print_json(payload)
        else:
            _print_tiered_checkpoint(payload)
        return 0

    if args.command == "model-status":
        client = build_model_client()
        payload = client.preflight().to_dict()
        if args.live_smoke:
            payload["live_smoke"] = asyncio.run(client.live_smoke()).to_dict()
        if args.json:
            _print_json(payload)
        else:
            _print_model_status(payload)
        return 0

    if args.command == "ui":
        from .ui import launch_streamlit

        return launch_streamlit()

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
