"""Command-line interface for the deep research agent foundation."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from async_multi_search import SearchResult

from .config import load_config
from .graph import (
    DEFAULT_MAX_RESEARCH_ITERATIONS,
    DEFAULT_TARGET_PROSPECT_COUNT,
    derive_prospect_run_budget,
    inspect_research_thread,
    inspect_thread,
    resume_research,
    resume_research_workflow,
    resume_thread,
    run_query,
    run_research,
    run_research_workflow,
)
from .models import ModelPreflightError, build_model_client


def _checkpoint_payload(checkpoint: object) -> str:
    return json.dumps(checkpoint.to_dict(), indent=2, sort_keys=True)  # type: ignore[attr-defined]


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

    run_parser = subparsers.add_parser("run", help="Run the local G003 research workflow.")
    run_parser.add_argument("query", help="Research query to run through the nested workflow.")
    run_parser.add_argument("--thread-id", help="Durable thread id to use for checkpointing.")
    run_parser.add_argument(
        "--checkpoint-dir", help="Directory for local JSON checkpoints.", default=None
    )
    run_parser.add_argument(
        "--artifact-dir", help="Directory for local JSON/CSV/Markdown prospect artifacts."
    )
    run_parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override derived researcher iterations before sufficiency routing.",
    )
    run_parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Override derived search results requested per researcher iteration.",
    )
    run_parser.add_argument(
        "--target-prospect-count",
        type=int,
        default=DEFAULT_TARGET_PROSPECT_COUNT,
        help="Potential business targets to collect before sufficiency routing.",
    )
    run_parser.add_argument(
        "--no-llm-judgment",
        action="store_true",
        help="Disable structured LLM prospect judgment and use deterministic triage only.",
    )
    run_parser.add_argument(
        "--require-live-model",
        action="store_true",
        help="Fail before search if no hosted model provider is available for judgment.",
    )
    run_parser.add_argument(
        "--approve",
        action="store_true",
        help="Approve review immediately instead of stopping at the review interrupt.",
    )
    run_parser.add_argument("--json", action="store_true", help="Emit run result as JSON.")
    run_parser.add_argument(
        "--require-review",
        action="store_true",
        help="Stop at the compatibility review interrupt.",
    )
    run_parser.add_argument(
        "--mock-result",
        action="append",
        default=[],
        help="Add a mocked search result as title|url|snippet|provider.",
    )

    resume_parser = subparsers.add_parser("resume", help="Resume a local G003 workflow thread.")
    resume_parser.add_argument("thread_id", help="Thread id to resume from checkpoint.")
    resume_parser.add_argument(
        "--checkpoint-dir", help="Directory for local JSON checkpoints.", default=None
    )
    resume_parser.add_argument(
        "--artifact-dir", help="Directory for local JSON/CSV/Markdown prospect artifacts."
    )
    resume_parser.add_argument(
        "--approve",
        action="store_true",
        help="Mark review approved before resuming the thread.",
    )
    resume_parser.add_argument(
        "--approve-review",
        action="store_true",
        help="Compatibility alias for --approve.",
    )
    resume_parser.add_argument(
        "--require-live-model",
        action="store_true",
        help="Fail before resuming if no hosted model provider is available for judgment.",
    )
    resume_parser.add_argument("--json", action="store_true", help="Emit resume result as JSON.")

    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect a local G003 workflow checkpoint."
    )
    inspect_parser.add_argument("thread_id", nargs="?", help="Thread id to inspect.")
    inspect_parser.add_argument(
        "--thread-id", dest="thread_id_option", help="Thread id to inspect."
    )
    inspect_parser.add_argument(
        "--checkpoint-dir", help="Directory for local JSON checkpoints.", default=None
    )
    inspect_parser.add_argument("--json", action="store_true", help="Emit checkpoint as JSON.")

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


def _mock_search_from_specs(specs: Sequence[str]):
    results: list[SearchResult] = []
    for raw in specs:
        parts = raw.split("|", 3)
        if len(parts) != 4:
            raise ValueError("--mock-result must be title|url|snippet|provider")
        title, url, snippet, provider = parts
        results.append(SearchResult(title, url, snippet, provider=provider))

    async def search(_query: str, _max_results: int) -> list[SearchResult]:
        return results

    return search


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

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

    if args.command == "run":
        try:
            budget = derive_prospect_run_budget(
                args.target_prospect_count,
                max_iterations=args.max_iterations,
                max_results=args.max_results,
            )
            if args.json:
                if args.artifact_dir:
                    state = asyncio.run(
                        run_research(
                            args.query,
                            thread_id=args.thread_id,
                            checkpoint_dir=args.checkpoint_dir,
                            require_review=not args.approve,
                            max_iterations=budget.max_iterations,
                            max_results=budget.max_results,
                            target_prospect_count=budget.target_prospect_count,
                            model_timeout_seconds=budget.model_timeout_seconds,
                            enable_llm_judgment=not args.no_llm_judgment,
                            require_live_model=args.require_live_model,
                            review_approved=args.approve,
                            artifact_dir=args.artifact_dir,
                        )
                    )
                    _print_json(state)
                    return 0
                if args.require_live_model:
                    state = asyncio.run(
                        run_research(
                            args.query,
                            thread_id=args.thread_id,
                            checkpoint_dir=args.checkpoint_dir,
                            require_review=not args.approve,
                            max_iterations=budget.max_iterations,
                            max_results=budget.max_results,
                            target_prospect_count=budget.target_prospect_count,
                            model_timeout_seconds=budget.model_timeout_seconds,
                            enable_llm_judgment=not args.no_llm_judgment,
                            require_live_model=True,
                            review_approved=args.approve,
                        )
                    )
                    _print_json(state)
                    return 0
                result = run_query(
                    args.query,
                    thread_id=args.thread_id,
                    checkpoint_dir=args.checkpoint_dir,
                    max_iterations=args.max_iterations or DEFAULT_MAX_RESEARCH_ITERATIONS,
                    approve=args.approve,
                )
                _print_json(result)
                return 0

            if args.require_review or args.mock_result:
                state = asyncio.run(
                    run_research(
                        args.query,
                        thread_id=args.thread_id,
                        checkpoint_dir=args.checkpoint_dir,
                        search=_mock_search_from_specs(args.mock_result)
                        if args.mock_result
                        else None,
                        require_review=args.require_review,
                        max_iterations=budget.max_iterations,
                        max_results=budget.max_results,
                        target_prospect_count=budget.target_prospect_count,
                        model_timeout_seconds=budget.model_timeout_seconds,
                        enable_llm_judgment=not args.no_llm_judgment,
                        require_live_model=args.require_live_model,
                        review_approved=args.approve,
                        artifact_dir=args.artifact_dir,
                    )
                )
                _print_json(state)
                return 0

            checkpoint = run_research_workflow(
                args.query,
                thread_id=args.thread_id,
                checkpoint_dir=args.checkpoint_dir,
                max_iterations=budget.max_iterations,
                max_results=budget.max_results,
                target_prospect_count=budget.target_prospect_count,
                require_live_model=args.require_live_model,
                artifact_dir=args.artifact_dir,
            )
            _print_json(checkpoint.to_dict())
            return 0
        except ModelPreflightError as exc:
            parser.error(str(exc))

    if args.command == "resume":
        approve = bool(args.approve or args.approve_review)
        try:
            if args.json:
                if args.require_live_model:
                    state = asyncio.run(
                        resume_research(
                            args.thread_id,
                            checkpoint_dir=args.checkpoint_dir,
                            approve_review=approve,
                            require_live_model=True,
                            artifact_dir=args.artifact_dir,
                        )
                    )
                    _print_json(state)
                    return 0
                result = resume_thread(
                    args.thread_id,
                    checkpoint_dir=args.checkpoint_dir,
                    approve=approve,
                )
                _print_json(result)
                return 0

            if approve:
                state = asyncio.run(
                    resume_research(
                        args.thread_id,
                        checkpoint_dir=args.checkpoint_dir,
                        approve_review=True,
                        require_live_model=args.require_live_model,
                        artifact_dir=args.artifact_dir,
                    )
                )
                _print_json(state)
                return 0

            checkpoint = resume_research_workflow(
                args.thread_id,
                checkpoint_dir=args.checkpoint_dir,
                require_live_model=args.require_live_model,
                artifact_dir=args.artifact_dir,
            )
            _print_json(checkpoint.to_dict())
            return 0
        except ModelPreflightError as exc:
            parser.error(str(exc))

    if args.command == "inspect":
        thread_id = args.thread_id_option or args.thread_id
        if not thread_id:
            parser.error("inspect requires a thread id")
        if args.json:
            _print_json(inspect_thread(thread_id, checkpoint_dir=args.checkpoint_dir))
            return 0
        if args.thread_id_option:
            checkpoint = inspect_research_thread(
                thread_id, checkpoint_dir=args.checkpoint_dir
            )
            _print_json(checkpoint.state)
            return 0
        checkpoint = inspect_research_thread(thread_id, checkpoint_dir=args.checkpoint_dir)
        _print_json(checkpoint.to_dict())
        return 0

    if args.command == "ui":
        from .ui import launch_streamlit

        return launch_streamlit()

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
