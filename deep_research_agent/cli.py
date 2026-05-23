"""Command line interface for Deep Research Agent foundation utilities."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without importing optional provider clients."""
    parser = argparse.ArgumentParser(
        prog="deep-research-agent",
        description="Foundation CLI for Deep Research Agent utilities.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    search_parser = subparsers.add_parser(
        "search",
        help="run the async multi-provider web search fallback chain",
    )
    search_parser.add_argument("query", help="search query to run")
    search_parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="maximum number of results to return from the first successful provider",
    )

    return parser


async def _run_search(query: str, max_results: int) -> int:
    from .search import web_search

    results = await web_search(query, max_results=max_results)
    for result in results:
        print(f"[{result.provider}] {result.title}")
        if result.url:
            print(f"  {result.url}")
        if result.content:
            print(f"  {result.content}")
        print()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "search":
        return asyncio.run(_run_search(args.query, args.max_results))

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
