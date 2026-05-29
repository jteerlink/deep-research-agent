# Development quickstart

1. Copy `.env.example` to `.env` and fill only the provider keys you need.
2. Install the package in editable mode once package metadata is present:
   `python -m pip install -e .[dev]`.
3. Run import and discovery checks:
   `python -m pytest --collect-only`.
4. Run CLI help once the CLI module is available:
   `python -m deep_research_agent --help`.
   List provider priority with `python -m deep_research_agent search-providers`.
5. Exercise the tiered run/resume/inspect flow without provider credentials:
   `python -m deep_research_agent tiered-run --industry dental --geography "DFW area" --thread-id demo-tiered --mock-result "Demo|https://example.com|Snippet|duckduckgo" --json`,
   `python -m deep_research_agent tiered-inspect demo-tiered --json`, and
   `python -m deep_research_agent tiered-resume demo-tiered --json`.
6. For the optional local UI, install `python -m pip install -e '.[ui]'` and
   launch `deep-research-agent ui`. Keep provider keys in `.env` or local
   environment; do not commit them.

The foundation story is intentionally local-first: no hosted UI, no full upstream
clone, no model zoo, and no database-backed multi-user service. Tiered
checkpoints are local JSON files keyed by thread id so review interrupts can be
inspected and resumed without adding a server-side persistence layer.


See `docs/usage.md` for the full offline smoke, optional live smoke, artifact examples, and non-goal audit.
