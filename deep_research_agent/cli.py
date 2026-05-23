"""Command-line interface for the deep research agent foundation."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .config import load_config
from .graph import inspect_checkpoints, resume_research, run_research


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

    run_parser = subparsers.add_parser("run", help="Run the local G003 research workflow.")
    run_parser.add_argument("query", help="Research query to run through the nested workflow.")
    run_parser.add_argument("--thread-id", help="Durable thread id to use for checkpointing.")
    run_parser.add_argument(
        "--checkpoint-dir", help="Directory for local JSON checkpoints.", default=None
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

    resume_parser = subparsers.add_parser("resume", help="Resume a local G003 workflow thread.")
    resume_parser.add_argument("thread_id", help="Thread id to resume from checkpoint.")
    resume_parser.add_argument(
        "--checkpoint-dir", help="Directory for local JSON checkpoints.", default=None
    )
    resume_parser.add_argument(
        "--approve",
        action="store_true",
        help="Mark review approved before resuming the thread.",
    )
    resume_parser.add_argument("--json", action="store_true", help="Emit resume result as JSON.")

    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect a local G003 workflow checkpoint."
    )
    inspect_parser.add_argument("thread_id", help="Thread id to inspect.")
    inspect_parser.add_argument(
        "--checkpoint-dir", help="Directory for local JSON checkpoints.", default=None
    )
    inspect_parser.add_argument("--json", action="store_true", help="Emit checkpoint as JSON.")
    return parser


def _print_graph_result(result: Any, *, as_json: bool) -> None:
    payload = _jsonable(result)
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    state = payload["state"]
    print(f"thread_id={payload['thread_id']}")
    print(f"status={state.get('status')}")
    print(f"checkpoint_path={payload['checkpoint_path']}")
    if state.get("interrupt_reason"):
        print(f"interrupt_reason={state['interrupt_reason']}")
    if state.get("answer"):
        print(f"answer={state['answer']}")


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
        from .graph import run_query

        result = run_query(
            args.query,
            thread_id=args.thread_id,
            checkpoint_dir=args.checkpoint_dir,
            max_iterations=args.max_iterations,
            approve=args.approve,
        )
        _print_graph_result(result, as_json=args.json)
        return 0

    if args.command == "resume":
        from .graph import resume_thread

        result = resume_thread(
            args.thread_id,
            checkpoint_dir=args.checkpoint_dir,
            approve=args.approve,
        )
        _print_graph_result(result, as_json=args.json)
        return 0

    if args.command == "inspect":
        from .graph import inspect_thread

        payload = inspect_thread(args.thread_id, checkpoint_dir=args.checkpoint_dir)
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            state = payload.get("state", {})
            print(f"thread_id={payload.get('thread_id')}")
            print(f"status={state.get('status')}")
            print(f"checkpoint_path={payload.get('checkpoint_path')}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
