from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_standalone_sources_do_not_import_root_agent() -> None:
    forbidden = ("deep" + "_research" + "_agent", "async" + "_multi" + "_search")
    offenders: list[str] = []
    for directory in (ROOT / "src", ROOT / "tests"):
        for path in directory.rglob("*.py"):
            text = path.read_text()
            if any(token in text for token in forbidden):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
