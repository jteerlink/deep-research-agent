from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from deep_research_agent.agent import (
    inspect_research_thread,
    resume_research_workflow,
    run_research_workflow,
)


def _event_types(payload: dict[str, object]) -> set[str]:
    return {str(event["type"]) for event in payload["events"]}  # type: ignore[index]


def test_local_workflow_checkpoint_records_review_interrupt_and_metadata(tmp_path: Path) -> None:
    checkpoint = run_research_workflow(
        "find AI agencies",
        thread_id="thread-g003",
        checkpoint_dir=tmp_path,
    )

    payload = checkpoint.to_dict()
    assert payload["thread_id"] == "thread-g003"
    assert payload["status"] == "needs_review"
    assert payload["review_required"] is True
    assert payload["sufficient"] is True
    assert payload["fallback_metadata"]["provider"] == "local_mock"
    assert (tmp_path / "thread-g003.json").exists()
    assert {
        "main_started",
        "supervisor_delegated",
        "fallback_model_used",
        "researcher_iteration",
        "sufficiency_routed",
        "review_interrupt",
    } <= _event_types(payload)


def test_resume_uses_existing_thread_id_and_completes_review(tmp_path: Path) -> None:
    run_research_workflow("resume me", thread_id="thread-resume", checkpoint_dir=tmp_path)

    resumed = resume_research_workflow("thread-resume", checkpoint_dir=tmp_path)
    inspected = inspect_research_thread("thread-resume", checkpoint_dir=tmp_path)

    assert resumed.thread_id == "thread-resume"
    assert resumed.status == "complete"
    assert inspected.status == "complete"
    event_types = _event_types(inspected.to_dict())
    assert "review_resumed" in event_types
    assert "workflow_completed" in event_types


def test_cli_run_resume_inspect_round_trip(tmp_path: Path) -> None:
    run_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deep_research_agent",
            "run",
            "cli query",
            "--thread-id",
            "thread-cli",
            "--checkpoint-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    run_payload = json.loads(run_result.stdout)
    assert run_payload["thread_id"] == "thread-cli"
    assert run_payload["status"] == "needs_review"
    assert "review_interrupt" in _event_types(run_payload)

    resume_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deep_research_agent",
            "resume",
            "thread-cli",
            "--checkpoint-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    resume_payload = json.loads(resume_result.stdout)
    assert resume_payload["thread_id"] == "thread-cli"
    assert resume_payload["status"] == "complete"

    inspect_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deep_research_agent",
            "inspect",
            "thread-cli",
            "--checkpoint-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    inspect_payload = json.loads(inspect_result.stdout)
    assert inspect_payload["thread_id"] == "thread-cli"
    assert inspect_payload["status"] == "complete"
    assert "workflow_completed" in _event_types(inspect_payload)
