import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_package_cli_help_exits_successfully_without_provider_credentials():
    package_missing = find_spec("deep_research_agent") is None
    src_package_missing = not (ROOT / "src" / "deep_research_agent").exists()
    if package_missing and src_package_missing:
        pytest.skip("deep_research_agent package CLI is created by the package/CLI lanes")

    env = os.environ.copy()
    pythonpath_entries = [str(ROOT), str(ROOT / "src")]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    result = subprocess.run(
        [sys.executable, "-m", "deep_research_agent", "--help"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    output = f"{result.stdout}\n{result.stderr}".lower()
    assert result.returncode == 0, output
    assert "usage" in output or "help" in output
    assert "api_key" not in output


def test_cli_help_lists_tiered_preview_without_credentials():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    result = subprocess.run(
        [sys.executable, "-m", "deep_research_agent", "--help"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    )

    output = result.stdout.lower()
    assert "tiered-preview" in output
    assert "tiered-run" in output
    assert "tiered-resume" in output
    assert "tiered-inspect" in output
    assert "api_key" not in output
