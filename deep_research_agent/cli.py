"""Command-line interface for the deep research agent foundation."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from async_multi_search import SearchResult

from .config import load_config
from .graph import inspect_checkpoints, resume_research, run_research


def _checkpoint_payload(checkpoint: object) -> str:
    return json.dumps(checkpoint.to_dict(), indent=2, sort_keys=True)  # type: ignore[attr-defined]


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

    subparsers.add_parser(
        "search-providers",
        help="List async_multi_search.py provider order and required environment keys.",
    )

    run_parser = subparsers.add_parser("run", help="Run a local research workflow thread.")
    run_parser.add_argument("query", help="Research query to execute.")
    run_parser.add_argument("--thread-id", help="Optional durable thread id to use.")
    run_parser.add_argument(
        "--checkpoint-dir",
        help=(
            "Directory for local JSON checkpoints "
            "(default: DEEP_RESEARCH_CHECKPOINT_DIR or .deep_research_agent/checkpoints)."
        ),
    )
    run_parser.add_argument(
        "--require-review",
        action="store_true",
        help="Persist a review interrupt instead of finalizing immediately.",
    )
    run_parser.add_argument("--max-iterations", type=int, default=3)
    run_parser.add_argument("--min-evidence-records", type=int, default=1)
    run_parser.add_argument(
        "--mock-result",
        action="append",
        default=[],
        metavar="TITLE|URL|TEXT|PROVIDER",
        help="Add a deterministic mocked search result for local smoke tests. May be repeated.",
    )

    resume_parser = subparsers.add_parser("resume", help="Resume a local research workflow thread.")
    resume_parser.add_argument("thread_id", help="Durable thread id to resume.")
    resume_parser.add_argument(
        "--checkpoint-dir", help="Directory containing local JSON checkpoints."
    )
    resume_parser.add_argument(
        "--approve-review",
        action="store_true",
        help="Approve a pending review interrupt and finish the thread.",
    )
    resume_parser.add_argument(
        "--mock-result",
        action="append",
        default=[],
        metavar="TITLE|URL|TEXT|PROVIDER",
        help="Add a deterministic mocked search result for resumed local smoke tests.",
    )

    inspect_parser = subparsers.add_parser("inspect", help="Inspect local research checkpoints.")
    inspect_parser.add_argument("--thread-id", help="Optional thread id to inspect.")
    inspect_parser.add_argument(
        "--checkpoint-dir", help="Directory containing local JSON checkpoints."
    )
    return parser


def _mock_search_from_args(values: Sequence[str]):
    if not values:
        return None

    parsed: list[SearchResult] = []
    for raw in values:
        parts = raw.split("|", 3)
        if len(parts) != 4:
            raise ValueError("--mock-result must be TITLE|URL|TEXT|PROVIDER")
        title, url, text, provider = parts
        parsed.append(SearchResult(title=title, url=url, content=text, provider=provider))

    async def search(_query: str, max_results: int):
        return parsed[:max_results]

    return search


def _print_state(state: dict) -> None:
    print(json.dumps(state, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "config":
        config = load_config()
        if args.json:
            print(json.dumps(asdict(config), indent=2, default=str))
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
        try:
            search = _mock_search_from_args(args.mock_result)
            state = asyncio.run(
                run_research(
                    args.query,
                    thread_id=args.thread_id,
                    checkpoint_dir=args.checkpoint_dir,
                    require_review=args.require_review,
                    max_iterations=args.max_iterations,
                    min_evidence_records=args.min_evidence_records,
                    search=search,
                )
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        _print_state(state)
        return 0

    if args.command == "resume":
        try:
            search = _mock_search_from_args(args.mock_result)
            state = asyncio.run(
                resume_research(
                    args.thread_id,
                    checkpoint_dir=args.checkpoint_dir,
                    approve_review=args.approve_review,
                    search=search,
                )
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        _print_state(state)
        return 0

    if args.command == "inspect":
        try:
            _print_state(inspect_checkpoints(args.thread_id, checkpoint_dir=args.checkpoint_dir))
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
