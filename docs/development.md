# Development quickstart

1. Copy `.env.example` to `.env` and fill only the provider keys you need.
2. Install the package in editable mode once package metadata is present:
   `python -m pip install -e .[dev]`.
3. Run import and discovery checks:
   `python -m pytest --collect-only`.
4. Run CLI help once the CLI module is available:
   `python -m deep_research_agent --help`.
   List provider priority with `python -m deep_research_agent search-providers`.
5. Exercise the local G003 run/resume/inspect flow without provider credentials:
   `python -m deep_research_agent run "demo query" --thread-id demo-thread --artifact-dir artifacts/demo --require-review --mock-result "Demo|https://example.com|Snippet|duckduckgo"`,
   `python -m deep_research_agent resume demo-thread --artifact-dir artifacts/demo --approve-review`, and
   `python -m deep_research_agent inspect --thread-id demo-thread`.
6. For the optional local UI, install `python -m pip install -e '.[ui]'` and
   launch `deep-research-agent ui`. Keep provider keys in the UI password fields
   or local environment; do not commit them.
7. Run a LangGraph dev server from the repository root after dependencies are
   installed: `langgraph dev`.

The foundation story is intentionally local-first: no hosted UI, no full upstream
clone, no model zoo, and no database-backed multi-user service. G003 checkpoints
are local JSON files keyed by thread id so review interrupts can be inspected and
resumed without adding a server-side persistence layer.


See `docs/usage.md` for the full offline smoke, optional live smoke, artifact examples, and non-goal audit.
