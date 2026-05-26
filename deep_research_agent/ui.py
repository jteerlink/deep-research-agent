"""Local Streamlit UI for running and inspecting research threads.

The helpers in this module are import-safe and testable without Streamlit. The
actual Streamlit dependency is imported only when the UI is launched or rendered.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from async_multi_search import AsyncMultiProviderSearch

try:
    from .config import dotenv_values
    from .geography import normalize_geography
    from .graph import (
        DEFAULT_TARGET_PROSPECT_COUNT,
        MAX_MODEL_TIMEOUT_SECONDS,
        MAX_SEARCH_ITERATIONS,
        MAX_SEARCH_RESULTS_PER_ITERATION,
        MAX_SEARCH_TIMEOUT_SECONDS,
        MAX_TARGET_PROSPECT_COUNT,
        derive_prospect_run_budget,
        inspect_checkpoints,
        resume_research,
        run_research,
    )
    from .models import build_model_client
    from .tiered_search import (
        CompanySearchTarget,
        ContactSearchTarget,
        ProviderPolicy,
        TieredSearchDirective,
        build_company_discovery_lanes,
        build_company_discovery_queries,
        build_contact_discovery_queries,
        build_personalization_queries,
    )
except ImportError:
    if __package__:
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from deep_research_agent.config import dotenv_values
    from deep_research_agent.geography import normalize_geography
    from deep_research_agent.graph import (
        DEFAULT_TARGET_PROSPECT_COUNT,
        MAX_MODEL_TIMEOUT_SECONDS,
        MAX_SEARCH_ITERATIONS,
        MAX_SEARCH_RESULTS_PER_ITERATION,
        MAX_SEARCH_TIMEOUT_SECONDS,
        MAX_TARGET_PROSPECT_COUNT,
        derive_prospect_run_budget,
        inspect_checkpoints,
        resume_research,
        run_research,
    )
    from deep_research_agent.models import build_model_client
    from deep_research_agent.tiered_search import (
        CompanySearchTarget,
        ContactSearchTarget,
        ProviderPolicy,
        TieredSearchDirective,
        build_company_discovery_lanes,
        build_company_discovery_queries,
        build_contact_discovery_queries,
        build_personalization_queries,
    )

PROVIDER_API_KEY_FIELDS: tuple[tuple[str, str], ...] = (
    ("Tavily", "TAVILY_API_KEY"),
    ("Exa", "EXA_API_KEY"),
    ("Serper", "SERPER_API_KEY"),
    ("Firecrawl", "FIRECRAWL_API_KEY"),
    ("You.com Developer Cloud", "YDC_API_KEY"),
)

DEFAULT_CHECKPOINT_DIR = ".deep_research_agent/checkpoints"
DEFAULT_ARTIFACT_DIR = ".deep_research_agent/artifacts"


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


def provider_env_overlay(values: Mapping[str, str]) -> dict[str, str]:
    """Return non-empty provider key overrides without mutating ``os.environ``."""

    allowed = {env_key for _, env_key in PROVIDER_API_KEY_FIELDS}
    return {
        key: value.strip()
        for key, value in values.items()
        if key in allowed and isinstance(value, str) and value.strip()
    }


def provider_env_from_env_file(
    path: str | os.PathLike[str] = ".env",
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return provider keys from shell environment plus ``.env`` values."""

    current = os.environ if environ is None else environ
    merged = dotenv_values(path) if os.fspath(path).strip() else {}
    merged.update(current)
    return provider_env_overlay(merged)


def build_prospect_directive(industry: str, geography: str, criteria: str) -> str:
    """Build the structured free-text directive consumed by the research graph."""

    parts = []
    if industry.strip():
        parts.append(f"industry: {industry.strip()}")
    if geography.strip():
        parts.append(f"geography: {geography.strip()}")
    if criteria.strip():
        parts.append(f"criteria: {criteria.strip()}")
    return "\n".join(parts) if parts else "criteria: business prospects"



def build_tiered_preview(
    industry: str,
    geography: str,
    criteria: str,
    *,
    target_prospect_count: int = DEFAULT_TARGET_PROSPECT_COUNT,
    preferred_contact_roles: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build an offline tiered prospect preview for UI display and tests."""

    directive = TieredSearchDirective(
        industry=industry,
        geographic_area=geography,
        target_prospect_count=target_prospect_count,
        research_criteria=criteria,
        preferred_contact_roles=preferred_contact_roles,
    )
    example_company = CompanySearchTarget("Example Company", website="https://example.com")
    example_contact = ContactSearchTarget(
        "Example Contact",
        "Example Company",
        title=(preferred_contact_roles[0] if preferred_contact_roles else "Owner"),
    )
    return {
        "directive": asdict(directive),
        "company_discovery_queries": list(build_company_discovery_queries(directive)),
        "company_discovery_lanes": [
            lane.to_dict() for lane in build_company_discovery_lanes(directive)
        ],
        "contact_discovery_query_templates": list(
            build_contact_discovery_queries(directive, example_company)
        ),
        "personalization_query_templates": list(
            build_personalization_queries(directive, example_contact)
        ),
        "search_dependency": "injected",
        "provider_policy": ProviderPolicy().to_dict(),
        "warnings": [
            "Preview only: no network search, browser capture, enrichment, "
            "or outreach is executed.",
            "Early tiered discovery excludes the legacy default provider chain.",
        ],
    }

def preview_geography_scope(geography: str) -> dict[str, Any]:
    """Return display-safe geography normalization details for the UI."""

    return normalize_geography(geography).to_dict()


def env_file_overlay(
    path: str | os.PathLike[str] = ".env",
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return non-empty ``.env`` values that are not already exported."""

    current = os.environ if environ is None else environ
    values = dotenv_values(path) if os.fspath(path).strip() else {}
    return {key: value for key, value in values.items() if value and key not in current}


def model_preflight_from_env_file(
    path: str | os.PathLike[str] = ".env",
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return redacted model availability using the same env overlay as runs."""

    with temporary_env(env_file_overlay(path, environ=environ)):
        return build_model_client().preflight().to_dict()


@contextmanager
def temporary_env(overrides: Mapping[str, str]) -> Iterator[None]:
    """Apply environment values for one run and restore the previous process state."""

    original: dict[str, str | None] = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            os.environ[key] = value
        yield
    finally:
        for key, previous in original.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def result_preview(state: Mapping[str, Any], markdown_limit: int = 4000) -> dict[str, Any]:
    """Build the compact, display-safe result preview used by UI and tests."""

    artifact_paths = dict(state.get("artifact_paths", {}))
    markdown_preview = ""
    markdown_path = artifact_paths.get("markdown")
    if markdown_path:
        path = Path(str(markdown_path))
        if path.exists():
            markdown_preview = path.read_text(encoding="utf-8")[:markdown_limit]

    return {
        "thread_id": state.get("thread_id", ""),
        "status": state.get("status", ""),
        "warnings": list(state.get("warnings", [])),
        "event_count": len(state.get("events", [])),
        "events": list(state.get("events", []))[-25:],
        "evidence": list(state.get("evidence", []))[:10],
        "prospect_targets": list(state.get("prospect_targets", []))[:10],
        "artifact_paths": artifact_paths,
        "markdown_preview": markdown_preview,
        "raw_json": _jsonable(dict(state)),
    }


def launch_streamlit() -> int:
    """Launch this module with Streamlit's CLI."""

    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError:
        print(
            "Streamlit is not installed. Install it with: pip install -e '.[ui]'",
            file=sys.stderr,
        )
        return 1

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    return int(streamlit_cli.main() or 0)


def render_app() -> None:
    """Render the Streamlit app."""

    import streamlit as st

    st.set_page_config(page_title="Deep Research Agent", layout="wide")
    st.title("Deep Research Agent")
    st.caption("Local execution UI. Provider API keys are read from .env or shell env.")

    if "events" not in st.session_state:
        st.session_state.events = []
    if "last_state" not in st.session_state:
        st.session_state.last_state = None

    with st.sidebar:
        st.header("Execution")
        industry = st.text_input("Industry / niche", "")
        geography = st.text_input("Geographic area", "")
        geography_preview = preview_geography_scope(geography)
        geography_terms = geography_preview.get("search_terms", [])
        if geography_terms:
            st.caption(f"Search geography: {', '.join(geography_terms[:6])}")
        for warning in geography_preview.get("warnings", []):
            st.warning(str(warning))
        query = st.text_area(
            "Research criteria",
            "Likely need lead reactivation, customer winback, or dormant-database follow-up.",
        )
        thread_id = st.text_input("Thread ID", "")
        target_prospect_count = st.number_input(
            "Target prospects",
            min_value=1,
            max_value=MAX_TARGET_PROSPECT_COUNT,
            value=DEFAULT_TARGET_PROSPECT_COUNT,
        )
        derived_budget = derive_prospect_run_budget(int(target_prospect_count))
        st.caption(
            "Budget: "
            f"{derived_budget.max_iterations} searches, "
            f"{derived_budget.max_results} results/search, "
            f"{derived_budget.search_timeout_seconds}s search timeout, "
            f"{derived_budget.model_timeout_seconds}s model timeout"
        )
        for warning in derived_budget.warnings:
            st.warning(warning)
        tiered_preview = build_tiered_preview(
            industry,
            geography,
            query,
            target_prospect_count=int(target_prospect_count),
        )
        with st.expander("Tiered prospect preview", expanded=False):
            st.caption("Offline preview: no search, browser capture, enrichment, or outreach runs.")
            st.json(tiered_preview)
        with st.expander("Advanced budget"):
            override_budget = st.checkbox("Override derived budget", value=False)
            if override_budget:
                max_iterations = st.number_input(
                    "Max iterations",
                    min_value=1,
                    max_value=MAX_SEARCH_ITERATIONS,
                    value=derived_budget.max_iterations,
                )
                search_max_results = st.number_input(
                    "Search max results",
                    min_value=1,
                    max_value=MAX_SEARCH_RESULTS_PER_ITERATION,
                    value=derived_budget.max_results,
                )
                timeout_seconds = st.number_input(
                    "Search timeout seconds",
                    min_value=1,
                    max_value=MAX_SEARCH_TIMEOUT_SECONDS,
                    value=derived_budget.search_timeout_seconds,
                )
                model_timeout_seconds = st.number_input(
                    "Model timeout seconds",
                    min_value=1,
                    max_value=MAX_MODEL_TIMEOUT_SECONDS,
                    value=derived_budget.model_timeout_seconds,
                )
                active_budget = derive_prospect_run_budget(
                    int(target_prospect_count),
                    max_iterations=int(max_iterations),
                    max_results=int(search_max_results),
                    search_timeout_seconds=int(timeout_seconds),
                    model_timeout_seconds=int(model_timeout_seconds),
                )
            else:
                active_budget = derived_budget
        checkpoint_dir = st.text_input("Checkpoint dir", DEFAULT_CHECKPOINT_DIR)
        artifact_dir = st.text_input("Artifact dir", DEFAULT_ARTIFACT_DIR)
        require_review = st.checkbox("Require review", value=True)
        approve_review = st.checkbox("Approve review", value=False)
        enable_llm_judgment = st.checkbox("LLM prospect judgment", value=True)

        st.header("Environment")
        env_file = st.text_input("Env file", ".env")
        runtime_env = {
            **env_file_overlay(env_file),
            "DEEP_RESEARCH_MODEL_TIMEOUT_SECONDS": str(active_budget.model_timeout_seconds),
        }
        provider_keys = provider_env_from_env_file(env_file)
        configured = [
            label for label, env_key in PROVIDER_API_KEY_FIELDS if env_key in provider_keys
        ]
        missing = [
            label for label, env_key in PROVIDER_API_KEY_FIELDS if env_key not in provider_keys
        ]
        st.caption(f"Configured providers: {', '.join(configured) if configured else 'none'}")
        st.caption(f"Missing provider keys: {', '.join(missing) if missing else 'none'}")
        model_preflight = model_preflight_from_env_file(env_file)
        model_label = (
            model_preflight.get("selected_provider")
            if model_preflight.get("live_model_available")
            else "none"
        )
        st.caption(f"Live model provider: {model_label}")

        run_clicked = st.button("Run G003", type="primary", width="stretch")
        resume_clicked = st.button("Resume G003", width="stretch")
        inspect_clicked = st.button("Inspect G003", width="stretch")

    def progress(event: dict[str, Any]) -> None:
        st.session_state.events.append(event)

    try:
        if run_clicked:
            st.session_state.events = []
            searcher = AsyncMultiProviderSearch(timeout=int(active_budget.search_timeout_seconds))
            research_query = build_prospect_directive(industry, geography, query)
            with temporary_env(runtime_env):
                state = asyncio.run(
                    run_research(
                        research_query,
                        thread_id=thread_id or None,
                        checkpoint_dir=checkpoint_dir,
                        search=searcher.search,
                        require_review=require_review,
                        max_iterations=active_budget.max_iterations,
                        max_results=active_budget.max_results,
                        target_prospect_count=active_budget.target_prospect_count,
                        search_timeout_seconds=active_budget.search_timeout_seconds,
                        model_timeout_seconds=active_budget.model_timeout_seconds,
                        enable_llm_judgment=bool(enable_llm_judgment),
                        progress_callback=progress,
                        review_approved=approve_review,
                        artifact_dir=artifact_dir,
                    )
                )
            st.session_state.last_state = state
            st.success(f"Run {state.get('status', 'finished')}: {state.get('thread_id')}")

        if resume_clicked:
            if not thread_id:
                st.error("Thread ID is required to resume.")
            else:
                st.session_state.events = []
                searcher = AsyncMultiProviderSearch(
                    timeout=int(active_budget.search_timeout_seconds)
                )
                with temporary_env(runtime_env):
                    state = asyncio.run(
                        resume_research(
                            thread_id,
                            checkpoint_dir=checkpoint_dir,
                            approve_review=approve_review,
                            search=searcher.search,
                            max_results=active_budget.max_results,
                            progress_callback=progress,
                            artifact_dir=artifact_dir,
                        )
                    )
                st.session_state.last_state = state
                st.success(f"Resumed {state.get('status', 'finished')}: {thread_id}")

        if inspect_clicked:
            if not thread_id:
                st.error("Thread ID is required to inspect.")
            else:
                state = inspect_checkpoints(thread_id, checkpoint_dir=checkpoint_dir)
                st.session_state.last_state = state
                st.info(f"Loaded checkpoint: {thread_id}")
    except Exception as exc:  # pragma: no cover - exercised manually through Streamlit
        st.exception(exc)

    state = st.session_state.last_state
    preview = result_preview(state or {})

    status_col, warning_col, artifact_col = st.columns(3)
    status_col.metric("Status", str(preview["status"] or "not run"))
    warning_col.metric("Warnings", len(preview["warnings"]))
    artifact_col.metric("Artifacts", len(preview["artifact_paths"]))

    tabs = st.tabs(
        [
            "Timeline",
            "Evidence",
            "Prospects",
            "Candidate Reviews",
            "Artifacts",
            "Markdown",
            "Raw JSON",
        ]
    )
    with tabs[0]:
        st.subheader("Progress events")
        st.json(st.session_state.events or preview["events"])
    with tabs[1]:
        st.subheader("Evidence preview")
        st.dataframe(preview["evidence"], width="stretch")
    with tabs[2]:
        st.subheader("Prospect preview")
        st.dataframe(preview["prospect_targets"], width="stretch")
    with tabs[3]:
        st.subheader("Candidate reviews")
        st.dataframe(preview["raw_json"].get("prospect_reviews", []), width="stretch")
    with tabs[4]:
        st.subheader("Artifact paths")
        st.json(preview["artifact_paths"])
    with tabs[5]:
        st.subheader("Markdown preview")
        st.markdown(preview["markdown_preview"] or "_No markdown artifact yet._")
    with tabs[6]:
        st.subheader("Raw state JSON")
        st.code(json.dumps(preview["raw_json"], indent=2, sort_keys=True), language="json")


if __name__ == "__main__":
    render_app()
