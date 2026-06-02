"""CLI for standalone Ollama Cloud + Exa lead generation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

from .config import LeadgenConfig
from .errors import LeadgenError
from .runner import compile_existing, load_icp_file, plan_run, run_live


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ollama-exa-leadgen",
        description="Standalone Ollama Cloud + Exa Deep Search lead generation.",
    )
    subparsers = parser.add_subparsers(dest="command")

    plan = subparsers.add_parser("plan", help="Dry-run an ICP into call estimates and query lanes.")
    _add_icp_args(plan)
    plan.add_argument("--json", action="store_true", help="Emit JSON only.")

    run = subparsers.add_parser("run", help="Run live Exa discovery using Ollama Cloud planning.")
    _add_icp_args(run)
    run.add_argument("--output-dir", help="Output root for run artifacts.")

    compile_cmd = subparsers.add_parser(
        "compile",
        help="Recompile an existing run from raw batches.",
    )
    compile_cmd.add_argument("--run-dir", required=True, help="Existing run directory.")
    compile_cmd.add_argument("--icp-file", help="Optional ICP file override.")

    smoke = subparsers.add_parser("smoke", help="Key-gated live smoke for a small ICP run.")
    _add_icp_args(smoke)
    smoke.add_argument("--output-dir", help="Output root for run artifacts.")

    return parser


def _add_icp_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--icp-file", required=True, help="Path to user-written ICP JSON.")
    parser.add_argument(
        "--target-count",
        type=int,
        help="Override target prospect count; default 10.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Optional env file to read without printing secrets.",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if not args.command:
        parser.print_help()
        return 0
    try:
        config = LeadgenConfig.from_env(env_file=getattr(args, "env_file", ".env"))
        if args.command == "plan":
            icp = load_icp_file(
                args.icp_file,
                target_count=args.target_count or config.default_target_count,
            )
            payload = plan_run(icp, config)
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(f"campaign={icp.campaign_name}")
                print(f"target_count={icp.target_count}")
                print(f"estimated_exa_calls={payload['estimated_exa_calls']}")
                print(f"within_cap={str(payload['within_cap']).lower()}")
                for item in payload["micro_vertical_preview"]:
                    print(f"- {item['name']}: {item['objective']}")
            return 0

        if args.command == "run":
            icp = load_icp_file(
                args.icp_file,
                target_count=args.target_count or config.default_target_count,
            )
            run_dir = run_live(icp, config, output_dir=args.output_dir)
            print(f"run_dir={run_dir}")
            print(f"csv={run_dir / 'prospects.csv'}")
            return 0

        if args.command == "compile":
            compile_icp = load_icp_file(args.icp_file) if args.icp_file else None
            run_dir = compile_existing(args.run_dir, icp=compile_icp)
            print(f"run_dir={run_dir}")
            print(f"csv={run_dir / 'prospects.csv'}")
            return 0

        if args.command == "smoke":
            if os.getenv("LEADGEN_LIVE_SMOKE") not in {"1", "true", "TRUE", "yes"}:
                raise LeadgenError("Set LEADGEN_LIVE_SMOKE=1 to allow live smoke calls")
            icp = load_icp_file(
                args.icp_file,
                target_count=args.target_count or config.default_target_count,
            )
            run_dir = run_live(icp, config, output_dir=args.output_dir)
            summary_path = run_dir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0

    except LeadgenError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


__all__ = ["main", "build_parser"]
