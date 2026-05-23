# Development quickstart

1. Copy `.env.example` to `.env` and fill only the provider keys you need.
2. Install the package in editable mode once package metadata is present:
   `python -m pip install -e .[dev]`.
3. Run import and discovery checks:
   `python -m pytest --collect-only`.
4. Run CLI help once the CLI module is available:
   `python -m deep_research_agent --help`.
5. Run a LangGraph dev server from the repository root after dependencies are
   installed: `langgraph dev`.

The foundation story is intentionally local-first: no hosted UI, no full upstream
clone, no model zoo, and no database-backed multi-user service.
