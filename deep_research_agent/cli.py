"""Command-line interface for the deep research agent foundation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from .agent import inspect_research_thread, resume_research_workflow, run_research_workflow
from .config import load_config


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

    run_parser = subparsers.add_parser(
        "run",
        help="Run a local G003 research workflow and write a thread checkpoint.",
    )
    run_parser.add_argument("query", help="Research query to run through the nested workflow.")
    run_parser.add_argument(
        "--thread-id",
        help="Optional stable thread id. Defaults to a generated local-* id.",
    )
    run_parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for local JSON checkpoints (default: .deep_research_agent/checkpoints).",
    )
    run_parser.add_argument(
        "--approve",
        action="store_true",
        help="Complete the review gate immediately instead of stopping at the review interrupt.",
    )

    resume_parser = subparsers.add_parser(
        "resume",
        help="Resume a local G003 workflow checkpoint by thread id.",
    )
    resume_parser.add_argument("thread_id", help="Thread id to resume.")
    resume_parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for local JSON checkpoints (default: .deep_research_agent/checkpoints).",
    )
    resume_parser.add_argument(
        "--no-approve",
        action="store_true",
        help="Re-enter the review interrupt instead of approving completion.",
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Print a local G003 workflow checkpoint by thread id.",
    )
    inspect_parser.add_argument("thread_id", help="Thread id to inspect.")
    inspect_parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for local JSON checkpoints (default: .deep_research_agent/checkpoints).",
    )
    return parser


def _path_or_default(raw: str | None) -> str | Path:
    return raw if raw is not None else Path(".deep_research_agent") / "checkpoints"


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
        checkpoint = run_research_workflow(
            args.query,
            thread_id=args.thread_id,
            checkpoint_dir=_path_or_default(args.checkpoint_dir),
            approve=args.approve,
        )
        print(_checkpoint_payload(checkpoint))
        return 0

    if args.command == "resume":
        checkpoint = resume_research_workflow(
            args.thread_id,
            checkpoint_dir=_path_or_default(args.checkpoint_dir),
            approve=not args.no_approve,
        )
        print(_checkpoint_payload(checkpoint))
        return 0

    if args.command == "inspect":
        checkpoint = inspect_research_thread(
            args.thread_id,
            checkpoint_dir=_path_or_default(args.checkpoint_dir),
        )
        print(_checkpoint_payload(checkpoint))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
