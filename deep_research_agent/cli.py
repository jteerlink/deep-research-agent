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
    inspect_research_thread,
    inspect_thread,
    resume_research,
    resume_research_workflow,
    resume_thread,
    run_query,
    run_research,
    run_research_workflow,
)


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
        default=3,
        help="Maximum researcher iterations before sufficiency routing.",
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
    return parser


def _print_json(payload: Any) -> None:
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))


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

    if args.command == "run":
        if args.json:
            if args.artifact_dir:
                state = asyncio.run(
                    run_research(
                        args.query,
                        thread_id=args.thread_id,
                        checkpoint_dir=args.checkpoint_dir,
                        require_review=not args.approve,
                        max_iterations=args.max_iterations,
                        review_approved=args.approve,
                        artifact_dir=args.artifact_dir,
                    )
                )
                _print_json(state)
                return 0
            result = run_query(
                args.query,
                thread_id=args.thread_id,
                checkpoint_dir=args.checkpoint_dir,
                max_iterations=args.max_iterations,
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
                    search=_mock_search_from_specs(args.mock_result) if args.mock_result else None,
                    require_review=args.require_review,
                    max_iterations=args.max_iterations,
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
            artifact_dir=args.artifact_dir,
        )
        _print_json(checkpoint.to_dict())
        return 0

    if args.command == "resume":
        approve = bool(args.approve or args.approve_review)
        if args.json:
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
                    artifact_dir=args.artifact_dir,
                )
            )
            _print_json(state)
            return 0

        checkpoint = resume_research_workflow(
            args.thread_id, checkpoint_dir=args.checkpoint_dir, artifact_dir=args.artifact_dir
        )
        _print_json(checkpoint.to_dict())
        return 0

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

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
