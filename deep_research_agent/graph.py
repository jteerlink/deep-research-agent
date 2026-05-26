"""Local nested research workflow with durable checkpoint metadata for G003.

The implementation is intentionally local-first: it exposes a small graph-like
``invoke``/``ainvoke`` interface that works without a hosted LangGraph server,
database, or live model provider. Compatibility helpers below also preserve the
worker-lane API variants produced during the G003 team run.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict, cast
from urllib.parse import urlparse
from uuid import uuid4

from async_multi_search import SearchResult

from .artifacts import write_research_artifacts
from .config import ModelProvider, load_config
from .geography import GeoScope, normalize_geography, suggest_geography_alias_update
from .models import ModelClient, ModelRequest, build_model_client
from .prospect_judgment import (
    CandidateReview,
    ProspectCandidate,
    ProspectQualification,
    build_prospect_judge_prompt,
    business_name_from_title,
    canonical_website,
    coerce_prospect_judgment,
    deterministic_judgment,
    domain_from_url,
    fetch_page_text,
    prospect_judgment_schema,
    prospect_qualification,
    registrable_domain,
    triage_candidate,
)
from .prospects import ProspectCitation, ProspectRecord

WorkflowStatus = Literal["completed", "interrupted"]
ReviewStatus = Literal["pending", "approved"]
SearchFn = Callable[[str, int], Awaitable[Sequence[SearchResult]] | Sequence[SearchResult]]
PageFetchFn = Callable[[str], Awaitable[str] | str]
ProgressCallback = Callable[[dict[str, Any]], object]

DEFAULT_MAX_RESEARCH_ITERATIONS = 3
DEFAULT_TARGET_PROSPECT_COUNT = 10
CHECKPOINT_SCHEMA_VERSION = "g003.local_checkpoint.v1"
GRAPH_TOPOLOGY = {
    "main": ("supervisor", "review"),
    "supervisor": ("researcher", "review", "finish"),
    "researcher": ("supervisor",),
}
ARTIFACT_SUMMARY_MAX_CHARS = 600


class ResearchState(TypedDict, total=False):
    """State shared by the main, supervisor, and researcher workflow layers."""

    query: str
    thread_id: str
    answer: str
    status: WorkflowStatus
    review_status: ReviewStatus
    interrupt_reason: str
    interrupted: bool
    needs_review: bool
    sufficient: bool
    max_research_iterations: int
    max_iterations: int
    max_results: int
    raw_search_max_results: int
    research_iterations: int
    iteration: int
    findings: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    events: list[dict[str, Any]]
    fallback_events: list[dict[str, Any]]
    model_metadata: dict[str, Any]
    review_interrupt: dict[str, Any]
    prospect_targets: list[dict[str, Any]]
    prospect_reviews: list[dict[str, Any]]
    prospect_rejections: list[dict[str, Any]]
    geography_scope: dict[str, Any]
    geography_alias_suggestion: dict[str, Any]
    target_prospect_count: int
    llm_judgment_enabled: bool
    artifact_paths: dict[str, str]
    warnings: list[str]
    next_node: str


@dataclass(frozen=True)
class ProspectExtractionResult:
    """Prospects plus audit details derived from evidence."""

    prospects: list[dict[str, Any]]
    reviews: list[dict[str, Any]]
    rejections: list[dict[str, Any]]
    extra_evidence: list[dict[str, Any]]
    warnings: list[str]


@dataclass(frozen=True)
class GraphRunResult:
    """Result returned by CLI helpers after a workflow invocation."""

    thread_id: str
    state: ResearchState
    checkpoint_path: Path


@dataclass(frozen=True)
class ResearchCheckpoint:
    """Legacy-compatible checkpoint DTO for CLI/API round trips."""

    thread_id: str
    query: str
    status: str
    review_required: bool
    sufficient: bool
    events: tuple[dict[str, Any], ...]
    fallback_metadata: dict[str, Any]
    checkpoint_path: Path
    state: ResearchState = field(default_factory=lambda: cast(ResearchState, {}))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "thread_id": self.thread_id,
            "query": self.query,
            "status": self.status,
            "review_required": self.review_required,
            "sufficient": self.sufficient,
            "events": list(self.events),
            "fallback_metadata": _jsonable(self.fallback_metadata),
            "checkpoint_path": str(self.checkpoint_path),
        }
        payload.update(_jsonable(dict(self.state)))
        payload["status"] = self.status
        payload["events"] = list(self.events)
        payload["fallback_metadata"] = _jsonable(self.fallback_metadata)
        payload["review_required"] = self.review_required
        payload["sufficient"] = self.sufficient
        payload["checkpoint_path"] = str(self.checkpoint_path)
        return payload


@dataclass(frozen=True)
class ModelFallbackEvent:
    """Serializable fallback metadata emitted by graph nodes."""

    provider: str
    model: str
    trigger: str
    node: str
    retry_count: int
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    error_class: str = ""
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalCheckpointStore:
    """Durable JSON checkpoint store keyed by LangGraph-style thread id."""

    def __init__(self, root: str | Path | None = None) -> None:
        configured = root or os.environ.get("DEEP_RESEARCH_CHECKPOINT_DIR")
        self.root = Path(configured or ".deep_research_agent/checkpoints")

    def path_for(self, thread_id: str) -> Path:
        safe_thread_id = re.sub(r"[^A-Za-z0-9_.-]", "_", thread_id)
        return self.root / f"{safe_thread_id}.json"

    def save(self, thread_id: str, state: Mapping[str, Any]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(thread_id)
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "thread_id": thread_id,
            "saved_at": datetime.now(UTC).isoformat(),
            "state": _jsonable(dict(state)),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def load(self, thread_id: str) -> ResearchState:
        path = self.path_for(thread_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"unsupported checkpoint schema for thread {thread_id!r}")
        return cast(ResearchState, dict(payload["state"]))

    def inspect(self, thread_id: str) -> dict[str, Any]:
        path = self.path_for(thread_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["checkpoint_path"] = str(path)
        return payload


class LocalResearchGraph:
    """Import-safe nested workflow runner with LangGraph-compatible methods."""

    def __init__(self, checkpoint_store: LocalCheckpointStore | None = None) -> None:
        self.checkpoint_store = checkpoint_store or LocalCheckpointStore()
        self.topology = GRAPH_TOPOLOGY

    def invoke(
        self, state: Mapping[str, Any], config: Mapping[str, Any] | None = None
    ) -> ResearchState:
        return _run_research_workflow(
            state,
            checkpoint_store=self.checkpoint_store,
            config=config,
        )

    async def ainvoke(
        self, state: Mapping[str, Any], config: Mapping[str, Any] | None = None
    ) -> ResearchState:
        return self.invoke(state, config=config)


class LocalResearchWorkflow:
    """Async compatibility wrapper used by G003 smoke/resume tests."""

    def __init__(self, checkpoint_store: LocalCheckpointStore | None = None) -> None:
        self.checkpoint_store = checkpoint_store or LocalCheckpointStore()

    async def arun(
        self,
        *,
        query: str,
        thread_id: str | None = None,
        search: SearchFn | None = None,
        require_review: bool = False,
        max_iterations: int = DEFAULT_MAX_RESEARCH_ITERATIONS,
        max_results: int = 5,
        target_prospect_count: int = DEFAULT_TARGET_PROSPECT_COUNT,
        progress_callback: ProgressCallback | None = None,
    ) -> ResearchState:
        return await run_research(
            query,
            thread_id=thread_id,
            checkpoint_dir=self.checkpoint_store.root,
            search=search,
            require_review=require_review,
            max_iterations=max_iterations,
            max_results=max_results,
            target_prospect_count=target_prospect_count,
            progress_callback=progress_callback,
        )

    async def aresume(
        self,
        thread_id: str,
        *,
        approve_review: bool = False,
        search: SearchFn | None = None,
        max_results: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> ResearchState:
        state = self.checkpoint_store.load(thread_id)
        if not approve_review and state.get("status") == "interrupted":
            return state
        query = str(state.get("query", ""))
        return await run_research(
            query,
            thread_id=thread_id,
            checkpoint_dir=self.checkpoint_store.root,
            search=search,
            require_review=False,
            max_iterations=int(state.get("max_iterations", state.get("iteration", 1)) or 1),
            max_results=_coerce_int(
                max_results if max_results is not None else state.get("max_results"), 5
            ),
            progress_callback=progress_callback,
            review_approved=True,
            existing_state=state,
        )


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
    if isinstance(value, ModelProvider):
        return value.value
    return value


def _coerce_int(value: Any, default: int) -> int:
    return int(value if value is not None and value != "" else default)


def _clean_query_part(value: str) -> str:
    return " ".join(value.strip().split())


def _directive_fields(query: str) -> dict[str, str]:
    fields: dict[str, str] = {"criteria": _clean_query_part(query)}
    aliases = {
        "industry": "industry",
        "vertical": "industry",
        "niche": "industry",
        "geography": "geography",
        "geographic_area": "geography",
        "geo": "geography",
        "location": "geography",
        "criteria": "criteria",
        "research_criteria": "criteria",
    }
    for line in query.splitlines():
        if ":" not in line:
            continue
        raw_key, raw_value = line.split(":", 1)
        key = aliases.get(raw_key.strip().lower())
        value = _clean_query_part(raw_value)
        if key and value:
            fields[key] = value
    return fields


def geography_scope_for_query(query: str) -> GeoScope:
    """Return normalized geography context for a structured research directive."""

    return normalize_geography(_directive_fields(query).get("geography", ""))


def build_prospect_search_queries(query: str, max_iterations: int) -> list[str]:
    """Expand a prospect directive into business-discovery search queries."""

    fields = _directive_fields(query)
    criteria = fields.get("criteria", _clean_query_part(query))
    industry = fields.get("industry", "")
    geography = fields.get("geography", "")
    geography_scope = normalize_geography(geography)
    geography_terms = list(geography_scope.search_terms)
    if industry and geography and len(geography_terms) > 1:
        singular_industry = industry[:-1] if industry.endswith("s") else industry
        patterns = (
            "{geo} {industry} official websites",
            "{geo} {industry} official websites",
            "{geo} {industry} company contact about",
            "{geo} {industry} service providers",
            "{geo} {singular_industry} owner founder",
            "{geo} {industry} about us contact",
        )
        queries = [
            _clean_query_part(
                patterns[min(index, len(patterns) - 1)].format(
                    geo=term, industry=industry, singular_industry=singular_industry
                )
            )
            for index, term in enumerate(geography_terms)
        ]
        base = _clean_query_part(f"{industry} {geography_scope.canonical}")
        if criteria:
            queries.append(_clean_query_part(f"{base} {criteria}"))
        deduped = list(dict.fromkeys(query for query in queries if query))
        return deduped[: max(1, max_iterations)]

    focus = _clean_query_part(" ".join(part for part in (industry, geography) if part))
    base = focus or criteria or "business prospects"
    if focus:
        singular_industry = industry[:-1] if industry.endswith("s") else industry
        queries = [
            _clean_query_part(f"{geography} {industry} official websites"),
            _clean_query_part(f"{geography} {industry} company contact about"),
            _clean_query_part(f"{geography} local {industry} service providers"),
            _clean_query_part(f"{geography} {singular_industry} owner founder"),
            _clean_query_part(f"{geography} {industry} about us contact"),
            _clean_query_part(f"{base} {criteria}"),
        ]
    else:
        queries = [
            _clean_query_part(f"{base} target businesses official websites"),
            _clean_query_part(f"{base} potential clients local companies"),
            _clean_query_part(f"{base} company contact about owner founder"),
            _clean_query_part(f"{base} service providers official websites"),
            _clean_query_part(f"{base} service providers contact"),
        ]
    deduped = list(dict.fromkeys(query for query in queries if query))
    return deduped[: max(1, max_iterations)]


def _search_query_for_iteration(query: str, iteration: int, max_iterations: int) -> str:
    queries = build_prospect_search_queries(query, max_iterations)
    return queries[min(iteration - 1, len(queries) - 1)]


def _domain_from_url(url: str) -> str:
    return domain_from_url(url)


def _business_name_from_title(title: str, domain: str) -> str:
    return business_name_from_title(title, domain)


def _looks_like_business_target(record: Mapping[str, Any]) -> bool:
    candidate = ProspectCandidate.from_evidence(record)
    return triage_candidate(candidate).decision != "reject"


def _dedupe_evidence(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for record in records:
        key = _evidence_dedupe_key(record)
        if not key or key in seen:
            continue
        seen.add(key)
        copied = dict(record)
        copied["id"] = f"ev_{len(deduped) + 1}"
        deduped.append(copied)
    return deduped


def _evidence_dedupe_key(record: Mapping[str, Any]) -> str:
    url = str(record.get("url") or "").strip()
    source_type = str(record.get("source_type") or "snippet")
    if url:
        parsed = urlparse(url)
        path = parsed.path.rstrip("/") or "/"
        return f"{source_type}:{parsed.scheme}://{parsed.netloc.lower()}{path}"
    return f"{source_type}:{record.get('title', '')}"


def _append_event(
    state: MutableMapping[str, Any], node: str, event: str, **metadata: Any
) -> dict[str, Any]:
    entry = {
        "node": node,
        "event": event,
        "type": event,
        "timestamp": datetime.now(UTC).isoformat(),
        **_jsonable(metadata),
    }
    state.setdefault("events", []).append(entry)
    return entry


async def _notify_progress(
    progress_callback: ProgressCallback | None,
    event: Mapping[str, Any],
) -> None:
    if progress_callback is None:
        return
    result = progress_callback(dict(event))
    if isinstance(result, Awaitable):
        await result


def _append_fallback_event(
    state: MutableMapping[str, Any], *, node: str, trigger: str, retry_count: int = 0
) -> None:
    config = load_config()
    provider = config.primary_provider
    model = config.primary_model
    event = ModelFallbackEvent(
        provider=provider.value,
        model=model,
        trigger=trigger,
        node=node,
        retry_count=retry_count,
    )
    state.setdefault("fallback_events", []).append(event.to_dict())


def _thread_id_from_config(config: Mapping[str, Any] | None) -> str | None:
    if not config:
        return None
    configurable = config.get("configurable")
    if isinstance(configurable, Mapping):
        thread_id = configurable.get("thread_id")
        if thread_id:
            return str(thread_id)
    thread_id = config.get("thread_id")
    return str(thread_id) if thread_id else None


def _initial_state(
    state: Mapping[str, Any], *, config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    prepared = dict(state)
    prepared.setdefault("thread_id", _thread_id_from_config(config) or f"thread-{uuid4()}")
    prepared.setdefault("max_research_iterations", DEFAULT_MAX_RESEARCH_ITERATIONS)
    prepared.setdefault("research_iterations", 0)
    prepared.setdefault("findings", [])
    prepared.setdefault("events", [])
    prepared.setdefault("fallback_events", [])
    prepared.setdefault("sufficient", False)
    prepared.setdefault("needs_review", True)
    return prepared


def _supervisor_node(state: dict[str, Any]) -> str:
    _append_event(state, "supervisor", "evaluate")
    if state.get("sufficient"):
        state["next_node"] = "synthesize"
        return "synthesize"
    if int(state.get("research_iterations", 0)) >= int(
        state.get("max_research_iterations", DEFAULT_MAX_RESEARCH_ITERATIONS)
    ):
        state["sufficient"] = True
        state["next_node"] = "synthesize"
        _append_event(state, "supervisor", "max_iterations_reached")
        return "synthesize"
    state["next_node"] = "researcher"
    _append_event(state, "supervisor", "delegate", target="researcher")
    return "researcher"


def _researcher_node(state: dict[str, Any]) -> None:
    iteration = int(state.get("research_iterations", 0)) + 1
    state["research_iterations"] = iteration
    query = str(state.get("query", "")).strip()
    finding = {
        "iteration": iteration,
        "summary": f"Research pass {iteration} for: {query or 'unspecified query'}",
        "source_type": "mocked_local_research",
    }
    state.setdefault("findings", []).append(finding)
    _append_fallback_event(state, node="researcher", trigger="deterministic_local_research")
    _append_event(state, "researcher", "finding_recorded", iteration=iteration)
    if iteration >= int(state.get("max_research_iterations", DEFAULT_MAX_RESEARCH_ITERATIONS)):
        state["sufficient"] = True
        _append_event(state, "researcher", "sufficiency_reached", reason="max_iterations")


def _synthesize_node(state: dict[str, Any]) -> None:
    query = str(state.get("query", "")).strip()
    count = len(state.get("findings", []))
    state["answer"] = f"Draft answer for {query or 'the research query'} using {count} finding(s)."
    state["next_node"] = "review"
    _append_event(state, "synthesize", "draft_created", finding_count=count)


def _review_node(state: dict[str, Any]) -> None:
    if state.get("review_status") != "approved" and state.get("needs_review", True):
        state["status"] = "interrupted"
        state["interrupted"] = True
        state["interrupt_reason"] = "review_required"
        state["next_node"] = "review"
        _append_event(state, "review", "interrupt", reason="review_required")
        return
    state["status"] = "completed"
    state["interrupted"] = False
    state["review_status"] = "approved"
    state["next_node"] = "end"
    _append_event(state, "review", "approved")


def _run_research_workflow(
    state: Mapping[str, Any],
    *,
    checkpoint_store: LocalCheckpointStore,
    config: Mapping[str, Any] | None = None,
) -> ResearchState:
    current = _initial_state(state, config=config)
    if current.get("review_status") == "approved":
        current["needs_review"] = False
    _append_event(current, "main", "start", topology=GRAPH_TOPOLOGY)
    while True:
        next_node = _supervisor_node(current)
        if next_node == "synthesize":
            break
        _researcher_node(current)
    _synthesize_node(current)
    _review_node(current)
    checkpoint_store.save(str(current["thread_id"]), current)
    return cast(ResearchState, current)


def route_after_supervisor(state: Mapping[str, Any]) -> str:
    """Route to researcher, review, or finish from compatibility supervisor state."""

    evidence = list(state.get("evidence", []))
    prospects = list(state.get("prospect_targets", []))
    iteration = int(state.get("iteration", state.get("research_iterations", 0)) or 0)
    max_iterations = int(state.get("max_iterations", DEFAULT_MAX_RESEARCH_ITERATIONS) or 0)
    target_count = int(state.get("target_prospect_count", 0) or 0)
    if state.get("review_required") and not state.get("review_approved"):
        return "review"
    if target_count > 0:
        if len(prospects) >= target_count or iteration >= max_iterations:
            return "finish"
        return "researcher"
    if evidence or iteration >= max_iterations:
        return "finish"
    return "researcher"


async def _call_search(
    search: SearchFn | None, query: str, max_results: int
) -> Sequence[SearchResult]:
    if search is None:
        return []
    result = search(query, max_results)
    if isinstance(result, Awaitable):
        return await result
    return result


def _raw_search_result_count(max_results: int, target_prospect_count: int) -> int:
    """Request a broader raw pool while keeping final prospect count focused."""

    return max(max_results, min(25, max(1, target_prospect_count) * 2))


def _evidence_from_results(
    results: Sequence[SearchResult], *, start_index: int = 1, search_query: str = ""
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for index, result in enumerate(results, start=1):
        evidence.append(
            {
                "id": f"ev_{start_index + index - 1}",
                "query": search_query,
                "title": result.title,
                "url": result.url,
                "snippet": getattr(result, "snippet", getattr(result, "content", "")),
                "provider": result.provider,
                "source_type": "snippet",
            }
        )
    return evidence


async def _prospects_from_evidence(
    evidence: Sequence[Mapping[str, Any]],
    *,
    query: str,
    model_client: ModelClient,
    enable_llm_judgment: bool = True,
    page_fetch: PageFetchFn | None = None,
    geography_scope: Mapping[str, Any] | None = None,
) -> ProspectExtractionResult:
    directive = _directive_fields(query)
    candidates = _candidate_reviews_from_evidence(evidence, directive=directive)
    selected, duplicate_rejections = _select_best_reviews_by_domain(candidates)
    prospects: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = list(duplicate_rejections)
    extra_evidence: list[dict[str, Any]] = []
    warnings: list[str] = []
    page_text_by_domain = _page_text_by_domain(evidence)
    normalized_geography_scope = dict(geography_scope or geography_scope_for_query(query).to_dict())

    for review in selected:
        candidate = review.candidate
        triage = review.triage
        if triage.decision == "reject":
            judgment = deterministic_judgment(candidate, triage)
            qualification = prospect_qualification(judgment, triage)
            rejected = CandidateReview(
                candidate=candidate,
                triage=triage,
                judgment=judgment,
                qualification=qualification,
            ).to_dict()
            rejections.append(rejected)
            reviews.append(rejected)
            continue

        page_text = page_text_by_domain.get(candidate.root_domain, "")
        page_evidence_id = _page_evidence_id_for_domain(evidence, candidate.root_domain)
        if triage.decision == "fetch_then_judge" and not page_text:
            page_text = await _fetch_candidate_page_text(candidate, page_fetch)
            if page_text:
                page_evidence_id = f"ev_{len(evidence) + len(extra_evidence) + 1}"
                extra_evidence.append(
                    {
                        "id": page_evidence_id,
                        "query": candidate.search_query,
                        "title": f"Page read: {candidate.organization_guess}",
                        "url": candidate.source_url or candidate.canonical_website,
                        "snippet": page_text,
                        "provider": "page_fetch",
                        "source_type": "page_read",
                    }
                )

        judgment_error = ""
        model_status = ""
        if enable_llm_judgment:
            try:
                model_response = await model_client.invoke(
                    ModelRequest(
                        node="prospect_judge",
                        prompt=build_prospect_judge_prompt(
                            directive=directive,
                            geography_scope=normalized_geography_scope,
                            candidate=candidate,
                            triage=triage,
                            page_text=page_text,
                        ),
                        response_schema=prospect_judgment_schema(),
                        metadata={
                            "evidence_id": candidate.evidence_id,
                            "domain": candidate.root_domain,
                            "triage_decision": triage.decision,
                        },
                    )
                )
                structured = model_response.structured or {}
                status = str(structured.get("status") or "")
                if status in {"metadata_only", "model_unavailable", "no_available_model"}:
                    judgment = deterministic_judgment(candidate, triage, page_text=page_text)
                    judgment_error = status
                    model_status = status
                else:
                    judgment = coerce_prospect_judgment(
                        structured, candidate=candidate, triage=triage
                    )
            except Exception as exc:  # pragma: no cover - exact model parse failures vary
                judgment = deterministic_judgment(candidate, triage, page_text=page_text)
                judgment_error = f"{type(exc).__name__}: {exc}"
                model_status = "model_exception"
        else:
            judgment = deterministic_judgment(
                candidate, triage, page_text=page_text, mode="deterministic_disabled"
            )

        qualification = prospect_qualification(judgment, triage, model_status=model_status)
        reviewed = CandidateReview(
            candidate=candidate,
            triage=triage,
            judgment=judgment,
            page_evidence_id=page_evidence_id,
            page_text=page_text,
            error=judgment_error,
            qualification=qualification,
        )
        reviews.append(reviewed.to_dict())
        warnings.extend(qualification.qualification_warnings)
        if not qualification.export_qualified:
            rejections.append(reviewed.to_dict())
            continue
        prospects.append(_prospect_record_from_judgment(reviewed).to_dict())

    return ProspectExtractionResult(
        prospects=prospects,
        reviews=reviews,
        rejections=rejections,
        extra_evidence=extra_evidence,
        warnings=list(dict.fromkeys(warnings)),
    )


def _candidate_reviews_from_evidence(
    evidence: Sequence[Mapping[str, Any]],
    *,
    directive: Mapping[str, str] | None = None,
) -> list[CandidateReview]:
    reviews: list[CandidateReview] = []
    for record in evidence:
        if str(record.get("source_type") or "snippet") != "snippet":
            continue
        candidate = ProspectCandidate.from_evidence(record)
        reviews.append(
            CandidateReview(
                candidate=candidate,
                triage=triage_candidate(candidate, directive=directive),
            )
        )
    return reviews


def _select_best_reviews_by_domain(
    reviews: Sequence[CandidateReview],
) -> tuple[list[CandidateReview], list[dict[str, Any]]]:
    selected_by_domain: dict[str, CandidateReview] = {}
    rejections: list[dict[str, Any]] = []
    for review in reviews:
        domain = review.candidate.root_domain or review.candidate.domain
        if not domain:
            rejections.append(review.to_dict())
            continue
        existing = selected_by_domain.get(domain)
        if existing is None or _review_rank(review) > _review_rank(existing):
            if existing is not None:
                rejections.append(_duplicate_rejection(existing, kept=review))
            selected_by_domain[domain] = review
        else:
            rejections.append(_duplicate_rejection(review, kept=existing))
    return list(selected_by_domain.values()), rejections


def _review_rank(review: CandidateReview) -> tuple[int, int, int]:
    decision_rank = {"reject": 0, "fetch_then_judge": 1, "judge": 2}
    intent_rank = {
        "homepage": 5,
        "about": 4,
        "contact": 4,
        "service": 3,
        "unknown": 2,
        "listicle": 0,
        "article": 0,
        "job": 0,
        "review": 0,
        "third_party_profile": 0,
        "invalid_url": 0,
    }
    owned_signal = int("owned_business_signal" in review.triage.flags)
    return (
        decision_rank.get(review.triage.decision, 0),
        intent_rank.get(review.triage.page_intent, 1),
        owned_signal,
    )


def _duplicate_rejection(review: CandidateReview, *, kept: CandidateReview) -> dict[str, Any]:
    payload = review.to_dict()
    payload["triage"]["reason"] = f"Duplicate domain suppressed in favor of {kept.candidate.url}"
    payload["triage"]["flags"] = [*payload["triage"].get("flags", []), "duplicate_domain"]
    return payload


async def _fetch_candidate_page_text(
    candidate: ProspectCandidate, page_fetch: PageFetchFn | None
) -> str:
    urls = [candidate.source_url, candidate.canonical_website]
    for url in dict.fromkeys(url for url in urls if url):
        try:
            result = page_fetch(url) if page_fetch else fetch_page_text(url)
            text = await result if isinstance(result, Awaitable) else result
        except Exception:
            text = ""
        if text:
            return str(text)
    return ""


def _page_text_by_domain(evidence: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    pages: dict[str, str] = {}
    for record in evidence:
        if str(record.get("source_type") or "") != "page_read":
            continue
        root = registrable_domain(domain_from_url(str(record.get("url") or "")))
        if root and root not in pages:
            pages[root] = str(record.get("snippet") or record.get("content") or "")
    return pages


def _page_evidence_id_for_domain(evidence: Sequence[Mapping[str, Any]], domain: str) -> str:
    for record in evidence:
        if str(record.get("source_type") or "") != "page_read":
            continue
        if registrable_domain(domain_from_url(str(record.get("url") or ""))) == domain:
            return str(record.get("id") or "")
    return ""


def _prospect_record_from_judgment(review: CandidateReview) -> ProspectRecord:
    judgment = review.judgment
    if judgment is None:
        raise ValueError("accepted prospect review requires a judgment")
    qualification = review.qualification
    candidate = review.candidate
    citation_id = review.page_evidence_id or candidate.evidence_id
    citation = ProspectCitation(
        evidence_id=citation_id,
        claim=f"{judgment.organization} surfaced as a prospect discovery candidate.",
        quote="",
        purpose="discovery",
        field="target_account_list",
    )
    summary = judgment.evidence_summary or candidate.snippet
    return ProspectRecord(
        organization=judgment.organization,
        website=judgment.canonical_website or canonical_website(candidate.url),
        summary=summary or f"Discovered candidate for the research query: {judgment.organization}.",
        confidence=judgment.confidence,
        decision_maker_leads=judgment.decision_maker_leads
        or ("Owner/Founder", "General Manager"),
        fit_rationale=judgment.fit_rationale
        or "Candidate matched broad prospect triage and structured judgment guardrails.",
        personalized_angles=judgment.personalized_angles
        or (summary[:180] if summary else "Use cited discovery evidence to tailor outreach.",),
        citations=(citation,),
        metadata={
            "evidence_type": candidate.source_type,
            "provider": candidate.provider,
            "domain": candidate.root_domain,
            "source_url": candidate.source_url,
            "search_query": candidate.search_query,
            "source_category": review.triage.source_category,
            "page_intent": review.triage.page_intent,
            "triage_decision": review.triage.decision,
            "triage_reason": review.triage.reason,
            "judgment_mode": judgment.mode,
            "fit_score": judgment.fit_score,
            "guardrail_flags": list(judgment.guardrail_flags),
            **_qualification_metadata(qualification),
        },
    )


def _qualification_metadata(
    qualification: ProspectQualification | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if qualification is None:
        return {
            "qualification_status": "",
            "export_qualified": False,
            "sufficiency_qualified": False,
            "review_only": False,
            "review_only_reason": "",
            "qualification_reasons": [],
            "qualification_warnings": [],
            "fallback_metadata": {},
            "model_judgment_status": "",
        }
    payload = (
        qualification.to_dict()
        if isinstance(qualification, ProspectQualification)
        else dict(qualification)
    )
    fallback_metadata = dict(payload.get("fallback_metadata") or {})
    return {
        "qualification_status": str(payload.get("qualification_status") or ""),
        "export_qualified": bool(payload.get("export_qualified", False)),
        "sufficiency_qualified": bool(payload.get("sufficiency_qualified", False)),
        "review_only": bool(payload.get("review_only", False)),
        "review_only_reason": str(payload.get("review_only_reason") or ""),
        "qualification_reasons": list(payload.get("qualification_reasons") or []),
        "qualification_warnings": list(payload.get("qualification_warnings") or []),
        "fallback_metadata": fallback_metadata,
        "model_judgment_status": str(fallback_metadata.get("model_judgment_status") or ""),
    }


def _artifact_records(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    prospect_rows = _prospect_artifact_records(state)
    if prospect_rows:
        return prospect_rows
    return list(state.get("evidence", []))


def _prospect_artifact_records(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for target in list(state.get("prospect_targets", [])):
        row = _artifact_row_from_target(target)
        key = _artifact_row_key(row)
        if key and key in seen:
            continue
        seen.add(key)
        rows.append(row)

    for review in [
        *list(state.get("prospect_reviews", [])),
        *list(state.get("prospect_rejections", [])),
    ]:
        row = _artifact_row_from_review(review)
        if not row:
            continue
        key = _artifact_row_key(row)
        if key and key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def _artifact_row_from_target(target: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(target.get("metadata") or {})
    citations = list(target.get("citations") or [])
    citation = dict(citations[0]) if citations and isinstance(citations[0], Mapping) else {}
    return _compact_artifact_row(
        record_type="qualified",
        organization=str(target.get("organization") or ""),
        website=str(target.get("website") or ""),
        summary=str(target.get("summary") or ""),
        confidence=target.get("confidence", ""),
        fit_score=metadata.get("fit_score", ""),
        qualification_status=str(metadata.get("qualification_status") or "qualified"),
        export_qualified=bool(metadata.get("export_qualified", True)),
        sufficiency_qualified=bool(metadata.get("sufficiency_qualified", True)),
        review_only=bool(metadata.get("review_only", False)),
        review_only_reason=str(metadata.get("review_only_reason") or ""),
        reject_reason="",
        judgment_mode=str(metadata.get("judgment_mode") or ""),
        model_judgment_status=str(metadata.get("model_judgment_status") or ""),
        source_category=str(metadata.get("source_category") or ""),
        page_intent=str(metadata.get("page_intent") or ""),
        source_url=str(metadata.get("source_url") or target.get("website") or ""),
        search_query=str(metadata.get("search_query") or ""),
        guardrail_flags=list(metadata.get("guardrail_flags") or []),
        evidence_id=str(citation.get("evidence_id") or ""),
    )


def _artifact_row_from_review(review: Mapping[str, Any]) -> dict[str, Any]:
    candidate = dict(review.get("candidate") or {})
    if not candidate:
        return {}
    triage = dict(review.get("triage") or {})
    judgment = dict(review.get("judgment") or {})
    qualification = _qualification_metadata(review.get("qualification"))
    qualification_status = qualification["qualification_status"]
    if not qualification_status:
        qualification_status = "rejected" if triage.get("decision") == "reject" else "needs_review"
    reject_reason = ""
    if qualification_status == "rejected":
        reject_reason = str(
            judgment.get("reject_reason") or triage.get("reason") or review.get("error") or ""
        )
    fallback_metadata = dict(qualification.get("fallback_metadata") or {})
    judgment_mode = str(judgment.get("mode") or fallback_metadata.get("judgment_mode") or "")
    model_status = str(
        fallback_metadata.get("model_judgment_status") or review.get("error") or ""
    )
    guardrail_flags = list(judgment.get("guardrail_flags") or triage.get("flags") or [])
    return _compact_artifact_row(
        record_type=qualification_status,
        organization=str(judgment.get("organization") or candidate.get("organization_guess") or ""),
        website=str(
            judgment.get("canonical_website") or candidate.get("canonical_website") or ""
        ),
        summary=str(judgment.get("evidence_summary") or candidate.get("snippet") or ""),
        confidence=judgment.get("confidence", ""),
        fit_score=judgment.get("fit_score", ""),
        qualification_status=qualification_status,
        export_qualified=bool(qualification["export_qualified"]),
        sufficiency_qualified=bool(qualification["sufficiency_qualified"]),
        review_only=bool(qualification["review_only"]),
        review_only_reason=str(qualification["review_only_reason"]),
        reject_reason=reject_reason,
        judgment_mode=judgment_mode,
        model_judgment_status=model_status,
        source_category=str(triage.get("source_category") or ""),
        page_intent=str(triage.get("page_intent") or ""),
        source_url=str(candidate.get("source_url") or candidate.get("url") or ""),
        search_query=str(candidate.get("search_query") or ""),
        guardrail_flags=guardrail_flags,
        evidence_id=str(candidate.get("evidence_id") or ""),
    )


def _compact_artifact_row(**values: Any) -> dict[str, Any]:
    return {
        "record_type": values.get("record_type", ""),
        "organization": values.get("organization", ""),
        "website": values.get("website", ""),
        "summary": _compact_artifact_summary(values.get("summary", "")),
        "confidence": values.get("confidence", ""),
        "fit_score": values.get("fit_score", ""),
        "qualification_status": values.get("qualification_status", ""),
        "export_qualified": values.get("export_qualified", False),
        "sufficiency_qualified": values.get("sufficiency_qualified", False),
        "review_only": values.get("review_only", False),
        "review_only_reason": values.get("review_only_reason", ""),
        "reject_reason": values.get("reject_reason", ""),
        "judgment_mode": values.get("judgment_mode", ""),
        "model_judgment_status": values.get("model_judgment_status", ""),
        "source_category": values.get("source_category", ""),
        "page_intent": values.get("page_intent", ""),
        "source_url": values.get("source_url", ""),
        "search_query": values.get("search_query", ""),
        "guardrail_flags": values.get("guardrail_flags", []),
        "evidence_id": values.get("evidence_id", ""),
    }


def _compact_artifact_summary(value: Any) -> str:
    summary = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(summary) <= ARTIFACT_SUMMARY_MAX_CHARS:
        return summary
    return summary[: ARTIFACT_SUMMARY_MAX_CHARS - 3].rstrip() + "..."


def _artifact_row_key(row: Mapping[str, Any]) -> str:
    return str(row.get("website") or row.get("source_url") or row.get("evidence_id") or "")


def _review_only_count(state: Mapping[str, Any]) -> int:
    count = 0
    for review in list(state.get("prospect_reviews", [])):
        qualification = dict(review.get("qualification") or {})
        if qualification.get("review_only"):
            count += 1
    return count


def _sufficiency_qualified_count(prospects: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for prospect in prospects:
        metadata = dict(prospect.get("metadata") or {})
        if metadata.get("sufficiency_qualified", True):
            count += 1
    return count


def _write_artifacts_if_requested(
    state: MutableMapping[str, Any], artifact_dir: str | Path | None
) -> None:
    if artifact_dir is None:
        return
    output_dir = Path(artifact_dir) / str(state["thread_id"])
    paths = write_research_artifacts(
        _artifact_records(state),
        output_dir,
        metadata={
            "thread_id": state["thread_id"],
            "query": state.get("query", ""),
            "geography_scope": state.get("geography_scope", {}),
            "geography_alias_suggestion": state.get("geography_alias_suggestion", {}),
            "artifact_projection": "prospect_run_compact_v1",
            "prospect_targets": list(state.get("prospect_targets", [])),
            "prospect_reviews": list(state.get("prospect_reviews", [])),
            "prospect_rejections": list(state.get("prospect_rejections", [])),
            "warnings": list(state.get("warnings", [])),
            "review_only_count": _review_only_count(state),
            "export_qualified_count": len(list(state.get("prospect_targets", []))),
        },
    )
    state["artifact_paths"] = {
        "json": str(paths.json_path),
        "csv": str(paths.csv_path),
        "markdown": str(paths.markdown_path),
    }


async def run_research(
    query: str,
    *,
    thread_id: str | None = None,
    checkpoint_dir: str | Path | None = None,
    search: SearchFn | None = None,
    model_client: ModelClient | None = None,
    page_fetch: PageFetchFn | None = None,
    require_review: bool = False,
    max_iterations: int = DEFAULT_MAX_RESEARCH_ITERATIONS,
    max_results: int = 5,
    target_prospect_count: int = DEFAULT_TARGET_PROSPECT_COUNT,
    enable_llm_judgment: bool = True,
    progress_callback: ProgressCallback | None = None,
    review_approved: bool = False,
    existing_state: Mapping[str, Any] | None = None,
    artifact_dir: str | Path | None = None,
) -> ResearchState:
    """Run the compatibility async research workflow used by G003 tests."""

    store = LocalCheckpointStore(checkpoint_dir)
    selected_model_client = model_client or build_model_client()
    geography_scope = geography_scope_for_query(query).to_dict()
    geography_alias_suggestion = suggest_geography_alias_update(
        str(geography_scope.get("raw", ""))
    )
    selected_thread_id: str
    if existing_state and not thread_id:
        selected_thread_id = str(existing_state.get("thread_id") or f"thread-{uuid4()}")
    else:
        selected_thread_id = thread_id or f"thread-{uuid4()}"
    state: dict[str, Any]
    if existing_state and review_approved:
        state = dict(existing_state)
        state.setdefault("events", [])
        state.setdefault("geography_scope", geography_scope)
        if geography_alias_suggestion is not None:
            state.setdefault(
                "geography_alias_suggestion", geography_alias_suggestion.to_dict()
            )
        state["thread_id"] = selected_thread_id
        state["status"] = "completed"
        state["sufficient"] = bool(state.get("sufficient", True))
        await _notify_progress(
            progress_callback, _append_event(state, "review", "review_resumed")
        )
        await _notify_progress(
            progress_callback, _append_event(state, "main", "workflow_completed")
        )
        _write_artifacts_if_requested(state, artifact_dir)
        store.save(selected_thread_id, state)
        return cast(ResearchState, state)

    events = list(existing_state.get("events", [])) if existing_state else []
    state = {
        "query": query,
        "thread_id": selected_thread_id,
        "events": events,
        "geography_scope": geography_scope,
        "iteration": 0,
        "max_iterations": max_iterations,
        "max_results": max_results,
        "raw_search_max_results": _raw_search_result_count(max_results, target_prospect_count),
        "target_prospect_count": target_prospect_count,
        "llm_judgment_enabled": enable_llm_judgment,
        "evidence": [],
        "findings": [],
        "fallback_events": [],
        "prospect_targets": [],
        "prospect_reviews": [],
        "prospect_rejections": [],
        "warnings": list(geography_scope.get("warnings", [])),
    }
    if geography_alias_suggestion is not None:
        state["geography_alias_suggestion"] = geography_alias_suggestion.to_dict()

    async def record(node: str, event: str, **metadata: Any) -> None:
        await _notify_progress(progress_callback, _append_event(state, node, event, **metadata))

    await record("main", "main_started")
    for warning in state["warnings"]:
        await record("warning", "warning_recorded", message=warning)

    while route_after_supervisor(state) == "researcher":
        next_iteration = int(state.get("iteration", 0)) + 1
        search_query = _search_query_for_iteration(query, next_iteration, max_iterations)
        raw_search_results = _raw_search_result_count(max_results, target_prospect_count)
        await record("supervisor", "supervisor_delegated", target="researcher")
        await record(
            "search",
            "search_started",
            query=search_query,
            max_results=raw_search_results,
            target_prospect_count=target_prospect_count,
            geography_scope=state["geography_scope"],
        )
        results = await _call_search(search, search_query, max_results=raw_search_results)
        await record("search", "search_completed", query=search_query, result_count=len(results))
        state["evidence"] = _dedupe_evidence(
            [
                *list(state.get("evidence", [])),
                *_evidence_from_results(
                    results,
                    start_index=len(state.get("evidence", [])) + 1,
                    search_query=search_query,
                ),
            ]
        )
        extraction = await _prospects_from_evidence(
            state["evidence"],
            query=query,
            model_client=selected_model_client,
            enable_llm_judgment=enable_llm_judgment,
            page_fetch=page_fetch,
            geography_scope=state["geography_scope"],
        )
        if extraction.extra_evidence:
            state["evidence"] = _dedupe_evidence(
                [*list(state.get("evidence", [])), *extraction.extra_evidence]
            )
        for warning in extraction.warnings:
            if warning not in state["warnings"]:
                state["warnings"].append(warning)
                await record("warning", "warning_recorded", message=warning)
        state["findings"] = list(state["evidence"])
        state["prospect_targets"] = extraction.prospects
        state["prospect_reviews"] = extraction.reviews
        state["prospect_rejections"] = extraction.rejections
        _append_event(
            state,
            "prospects",
            "prospect_candidates_reviewed",
            accepted_count=len(extraction.prospects),
            rejected_count=len(extraction.rejections),
            qualified_count=_sufficiency_qualified_count(extraction.prospects),
            review_only_count=_review_only_count(state),
            llm_judgment_enabled=enable_llm_judgment,
        )
        state["iteration"] = next_iteration
        model_response = await selected_model_client.invoke(
            ModelRequest(
                node="researcher",
                prompt=search_query,
                metadata={"thread_id": selected_thread_id, "iteration": state["iteration"]},
            )
        )
        state["model_metadata"] = model_response.to_dict()
        await record("model", "fallback_model_used", provider=model_response.provider.value)
        await record(
            "researcher",
            "researcher_iteration",
            iteration=state["iteration"],
            prospect_count=len(state["prospect_targets"]),
        )
        if len(state["prospect_targets"]) >= target_prospect_count or int(
            state["iteration"]
        ) >= max_iterations:
            state["sufficient"] = True
            reason = (
                "target_prospect_count"
                if len(state["prospect_targets"]) >= target_prospect_count
                else "max_iterations"
            )
            await record("supervisor", "sufficiency_routed", reason=reason)
            break

    if not state["evidence"]:
        state["warnings"].append("No search evidence was captured; prospect exports are empty.")
        await record("warning", "warning_recorded", message=state["warnings"][-1])
    elif not state["prospect_targets"]:
        review_only_count = _review_only_count(state)
        review_suffix = (
            f" {review_only_count} candidate(s) were retained for review-only artifacts."
            if review_only_count
            else ""
        )
        state["warnings"].append(
            "Search evidence was captured, but no export-qualified business prospects were "
            f"extracted.{review_suffix} Add a target industry/niche and geography for stronger "
            "company discovery."
        )
        await record("warning", "warning_recorded", message=state["warnings"][-1])
    elif len(state["prospect_targets"]) < target_prospect_count:
        state["warnings"].append(
            f"Only {len(state['prospect_targets'])} potential business target(s) were found "
            f"before the iteration limit of {max_iterations}."
        )
        await record("warning", "warning_recorded", message=state["warnings"][-1])

    if require_review and not review_approved:
        state["status"] = "interrupted"
        state["review_interrupt"] = {"thread_id": selected_thread_id, "reason": "review_required"}
        await record("review", "review_interrupt", reason="review_required")
    else:
        state["status"] = "completed"
        if review_approved or existing_state:
            await record("review", "review_resumed")
        await record("main", "workflow_completed")
    _write_artifacts_if_requested(state, artifact_dir)
    store.save(selected_thread_id, state)
    return cast(ResearchState, state)


async def resume_research(
    thread_id: str,
    *,
    checkpoint_dir: str | Path | None = None,
    approve_review: bool = True,
    search: SearchFn | None = None,
    max_results: int | None = None,
    progress_callback: ProgressCallback | None = None,
    artifact_dir: str | Path | None = None,
) -> ResearchState:
    """Resume a compatibility async workflow from a checkpoint."""

    store = LocalCheckpointStore(checkpoint_dir)
    state = store.load(thread_id)
    if state.get("status") == "interrupted" and not approve_review:
        return state
    return await run_research(
        str(state.get("query", "")),
        thread_id=thread_id,
        checkpoint_dir=checkpoint_dir,
        search=search,
        require_review=False,
        max_iterations=int(state.get("max_iterations", state.get("iteration", 1)) or 1),
        max_results=_coerce_int(
            max_results if max_results is not None else state.get("max_results"), 5
        ),
        progress_callback=progress_callback,
        review_approved=approve_review,
        existing_state=state,
        artifact_dir=artifact_dir,
    )


def inspect_checkpoints(
    thread_id: str, *, checkpoint_dir: str | Path | None = None
) -> ResearchState:
    """Return the saved state for a compatibility checkpoint."""

    return LocalCheckpointStore(checkpoint_dir).load(thread_id)


def build_graph(checkpoint_store: LocalCheckpointStore | None = None) -> LocalResearchGraph:
    """Build the import-safe local graph runner used by package and CLI entrypoints."""

    return LocalResearchGraph(checkpoint_store=checkpoint_store)


def run_query(
    query: str,
    *,
    thread_id: str | None = None,
    checkpoint_dir: str | Path | None = None,
    max_iterations: int = DEFAULT_MAX_RESEARCH_ITERATIONS,
    approve: bool = False,
) -> GraphRunResult:
    store = LocalCheckpointStore(checkpoint_dir)
    state: ResearchState = {
        "query": query,
        "max_research_iterations": max_iterations,
    }
    if thread_id:
        state["thread_id"] = thread_id
    if approve:
        state["review_status"] = "approved"
    config = {"configurable": {"thread_id": thread_id}} if thread_id else None
    result = build_graph(store).invoke(state, config=config)
    return GraphRunResult(
        thread_id=str(result["thread_id"]),
        state=result,
        checkpoint_path=store.path_for(str(result["thread_id"])),
    )


def resume_thread(
    thread_id: str,
    *,
    checkpoint_dir: str | Path | None = None,
    approve: bool = False,
) -> GraphRunResult:
    store = LocalCheckpointStore(checkpoint_dir)
    state = store.load(thread_id)
    if approve:
        state["review_status"] = "approved"
        state["needs_review"] = False
    result = build_graph(store).invoke(state, config={"configurable": {"thread_id": thread_id}})
    return GraphRunResult(
        thread_id=thread_id, state=result, checkpoint_path=store.path_for(thread_id)
    )


def inspect_thread(
    thread_id: str, *, checkpoint_dir: str | Path | None = None
) -> dict[str, Any]:
    return LocalCheckpointStore(checkpoint_dir).inspect(thread_id)


def _checkpoint_from_state(
    state: ResearchState, checkpoint_dir: str | Path | None = None
) -> ResearchCheckpoint:
    status = "complete" if state.get("status") == "completed" else "needs_review"
    metadata = {
        "provider": "local_mock",
        "model": "deterministic-local",
        "trigger": "metadata_only",
    }
    return ResearchCheckpoint(
        thread_id=str(state["thread_id"]),
        query=str(state.get("query", "")),
        status=status,
        review_required=status == "needs_review",
        sufficient=bool(state.get("sufficient", True)),
        events=tuple(state.get("events", [])),
        fallback_metadata=dict(metadata),
        checkpoint_path=LocalCheckpointStore(checkpoint_dir).path_for(str(state["thread_id"])),
        state=state,
    )


def run_research_workflow(
    query: str,
    *,
    thread_id: str | None = None,
    checkpoint_dir: str | Path | None = None,
    max_results: int = 5,
    artifact_dir: str | Path | None = None,
) -> ResearchCheckpoint:
    state = asyncio.run(
        run_research(
            query,
            thread_id=thread_id,
            checkpoint_dir=checkpoint_dir,
            require_review=True,
            max_results=max_results,
            artifact_dir=artifact_dir,
        )
    )
    return _checkpoint_from_state(state, checkpoint_dir)


def resume_research_workflow(
    thread_id: str,
    *,
    checkpoint_dir: str | Path | None = None,
    artifact_dir: str | Path | None = None,
) -> ResearchCheckpoint:
    state = asyncio.run(
        resume_research(
            thread_id,
            checkpoint_dir=checkpoint_dir,
            approve_review=True,
            artifact_dir=artifact_dir,
        )
    )
    return _checkpoint_from_state(state, checkpoint_dir)


def inspect_research_thread(
    thread_id: str, *, checkpoint_dir: str | Path | None = None
) -> ResearchCheckpoint:
    state = inspect_checkpoints(thread_id, checkpoint_dir=checkpoint_dir)
    return _checkpoint_from_state(state, checkpoint_dir)


def invoke_maybe_async(runnable: Any, state: Mapping[str, Any]) -> Any:
    """Invoke a graph-like object from sync tests without leaking event loops."""

    if hasattr(runnable, "invoke"):
        return runnable.invoke(state)
    result = runnable(state)
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return result


graph = build_graph()
