"""Tier-specific search helpers for prospect research.

The tiered workflow starts with broad company discovery before spending budget on
contact and person-level enrichment.  This module intentionally keeps the search
provider boundary injectable so early tiers and tests can run without
constructing the default ``AsyncMultiProviderSearch`` provider chain.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

TierName = Literal["company_discovery", "contact_discovery", "personalization"]
SearchCallable = Callable[[str, int], Awaitable[Sequence[Any]]]
EARLY_DISCOVERY_PROVIDERS = ("tavily", "serper", "firecrawl", "ydc", "duckduckgo")
FINAL_ENRICHMENT_PROVIDERS = ("exa",)
DEFAULT_MAX_PARALLEL_SEARCH_LANES = 3
MAX_PARALLEL_SEARCH_LANES = 6
DEFAULT_TARGET_PROSPECT_COUNT = 10
MAX_TARGET_PROSPECT_COUNT = 100
MAX_SEARCH_ITERATIONS = 10
MAX_SEARCH_RESULTS_PER_ITERATION = 50
MAX_RAW_SEARCH_RESULTS_PER_RUN = 300
MAX_MODEL_JUDGMENTS_PER_RUN = 150
MAX_SEARCH_TIMEOUT_SECONDS = 30
MAX_MODEL_TIMEOUT_SECONDS = 120


class SearchResultLike(Protocol):
    """Structural shape consumed from any search backend result."""

    title: str
    url: str
    content: str
    score: float | None
    provider: str


@dataclass(frozen=True)
class ProviderPolicy:
    """Provider phase policy for tiered prospect research."""

    early_discovery: tuple[str, ...] = EARLY_DISCOVERY_PROVIDERS
    final_enrichment: tuple[str, ...] = FINAL_ENRICHMENT_PROVIDERS

    def __post_init__(self) -> None:
        early = tuple(provider.lower() for provider in self.early_discovery)
        final = tuple(provider.lower() for provider in self.final_enrichment)
        if "exa" in early:
            raise ValueError("Exa is reserved for final enrichment, not early discovery")
        object.__setattr__(self, "early_discovery", early)
        object.__setattr__(self, "final_enrichment", final)

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "early_discovery": list(self.early_discovery),
            "final_enrichment": list(self.final_enrichment),
        }


@dataclass(frozen=True)
class TieredSearchDirective:
    """Minimal search directive fields needed to build tiered queries."""

    industry: str
    geographic_area: str
    target_prospect_count: int = DEFAULT_TARGET_PROSPECT_COUNT
    research_criteria: str = ""
    preferred_contact_roles: tuple[str, ...] = ()
    source_preferences: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.industry.strip():
            raise ValueError("TieredSearchDirective.industry is required")
        if not self.geographic_area.strip():
            raise ValueError("TieredSearchDirective.geographic_area is required")
        if self.target_prospect_count < 1:
            raise ValueError("TieredSearchDirective.target_prospect_count must be >= 1")
        object.__setattr__(
            self,
            "preferred_contact_roles",
            tuple(role.strip() for role in self.preferred_contact_roles if role.strip()),
        )
        object.__setattr__(
            self,
            "source_preferences",
            tuple(source.strip() for source in self.source_preferences if source.strip()),
        )


@dataclass(frozen=True)
class CompanySearchTarget:
    """Company context used by contact-discovery search."""

    name: str
    website: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("CompanySearchTarget.name is required")


@dataclass(frozen=True)
class ContactSearchTarget:
    """Contact context used by person-level personalization search."""

    name: str
    company_name: str
    title: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ContactSearchTarget.name is required")
        if not self.company_name.strip():
            raise ValueError("ContactSearchTarget.company_name is required")


@dataclass(frozen=True)
class CompanyDiscoveryLane:
    """One bounded early-discovery query lane."""

    family: str
    query: str
    ordinal: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TieredSearchHit:
    """Normalized, backend-neutral search hit captured for a tier query."""

    tier: TierName
    query: str
    title: str
    url: str
    content: str
    provider: str = ""
    rank: int = 1
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TieredSearchFailure:
    """Recoverable search failure for one tier query."""

    tier: TierName
    query: str
    error_class: str
    error_message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class TieredSearchBatch:
    """Normalized result of executing one or more queries for a tier."""

    tier: TierName
    queries: tuple[str, ...]
    hits: tuple[TieredSearchHit, ...] = ()
    failures: tuple[TieredSearchFailure, ...] = ()

    @property
    def ok(self) -> bool:
        return bool(self.hits) and not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "ok": self.ok,
            "queries": list(self.queries),
            "hits": [hit.to_dict() for hit in self.hits],
            "failures": [failure.to_dict() for failure in self.failures],
        }


@dataclass(frozen=True)
class ProspectRunBudget:
    """Bounded tiered-search knobs derived from the requested prospect count."""

    requested_target_prospect_count: int
    target_prospect_count: int
    max_iterations: int
    max_results: int
    raw_search_max_results: int
    search_timeout_seconds: int
    model_timeout_seconds: int
    max_model_judgments: int
    estimated_raw_search_results: int
    warnings: tuple[str, ...] = ()


def _dedupe_preserve_order(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = " ".join(value.split())
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            deduped.append(normalized)
    return tuple(deduped)


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _positive_int(value: Any, default: int, *, label: str, warnings: list[str]) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        warnings.append(f"{label} defaulted to {default} because it was not an integer")
        return default
    if number < 1:
        warnings.append(f"{label} raised to 1 because run budgets require positive values")
        return 1
    return number


def _bounded_budget_value(
    value: int | None,
    *,
    default: int,
    minimum: int,
    maximum: int,
    label: str,
    warnings: list[str],
) -> int:
    if value is None:
        return default
    number = _positive_int(value, default, label=label, warnings=warnings)
    if number < minimum:
        warnings.append(f"{label} raised to {minimum} by run guardrail")
        return minimum
    if number > maximum:
        warnings.append(f"{label} capped at {maximum} by run guardrail")
        return maximum
    return number


def derive_prospect_run_budget(
    target_prospect_count: int,
    *,
    max_iterations: int | None = None,
    max_results: int | None = None,
    search_timeout_seconds: int | None = None,
    model_timeout_seconds: int | None = None,
) -> ProspectRunBudget:
    """Return bounded runtime knobs for tiered company discovery."""

    warnings: list[str] = []
    requested_target = _positive_int(
        target_prospect_count,
        DEFAULT_TARGET_PROSPECT_COUNT,
        label="target_prospect_count",
        warnings=warnings,
    )
    target = min(requested_target, MAX_TARGET_PROSPECT_COUNT)
    if target != requested_target:
        warnings.append(
            f"target_prospect_count capped at {MAX_TARGET_PROSPECT_COUNT} by run guardrail"
        )

    derived_iterations = min(MAX_SEARCH_ITERATIONS, max(1, _ceil_div(target, 8) + 1))
    iterations = _bounded_budget_value(
        max_iterations,
        default=derived_iterations,
        minimum=1,
        maximum=MAX_SEARCH_ITERATIONS,
        label="max_iterations",
        warnings=warnings,
    )

    desired_raw_pool = min(MAX_RAW_SEARCH_RESULTS_PER_RUN, max(target * 4, target + 10))
    derived_results = min(
        MAX_SEARCH_RESULTS_PER_ITERATION,
        max(5, _ceil_div(desired_raw_pool, iterations)),
    )
    results = _bounded_budget_value(
        max_results,
        default=derived_results,
        minimum=1,
        maximum=MAX_SEARCH_RESULTS_PER_ITERATION,
        label="max_results",
        warnings=warnings,
    )
    if results * iterations > MAX_RAW_SEARCH_RESULTS_PER_RUN:
        capped_results = max(1, MAX_RAW_SEARCH_RESULTS_PER_RUN // iterations)
        if capped_results < results:
            results = capped_results
            warnings.append(
                f"raw search budget capped at {MAX_RAW_SEARCH_RESULTS_PER_RUN} results per run"
            )

    search_timeout = _bounded_budget_value(
        search_timeout_seconds,
        default=min(
            MAX_SEARCH_TIMEOUT_SECONDS,
            10 + (_ceil_div(max(target - 10, 0), 10) * 2),
        ),
        minimum=1,
        maximum=MAX_SEARCH_TIMEOUT_SECONDS,
        label="search_timeout_seconds",
        warnings=warnings,
    )
    model_timeout = _bounded_budget_value(
        model_timeout_seconds,
        default=min(
            MAX_MODEL_TIMEOUT_SECONDS,
            60 + (_ceil_div(max(target - 10, 0), 10) * 5),
        ),
        minimum=1,
        maximum=MAX_MODEL_TIMEOUT_SECONDS,
        label="model_timeout_seconds",
        warnings=warnings,
    )
    max_model_judgments = min(
        MAX_MODEL_JUDGMENTS_PER_RUN,
        max(10, target * 3),
        MAX_RAW_SEARCH_RESULTS_PER_RUN,
    )
    return ProspectRunBudget(
        requested_target_prospect_count=requested_target,
        target_prospect_count=target,
        max_iterations=iterations,
        max_results=results,
        raw_search_max_results=results,
        search_timeout_seconds=search_timeout,
        model_timeout_seconds=model_timeout,
        max_model_judgments=max_model_judgments,
        estimated_raw_search_results=iterations * results,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def build_company_discovery_queries(directive: TieredSearchDirective) -> tuple[str, ...]:
    """Build broad Tier 1 company-discovery queries from a directive."""

    industry = directive.industry.strip()
    geography = directive.geographic_area.strip()
    criteria = directive.research_criteria.strip()
    seeds = [
        f"{industry} companies in {geography}",
        f"best {industry} {geography}",
        f"{industry} owner {geography}",
        f"{industry} marketing {geography}",
    ]
    if criteria:
        seeds.append(f"{industry} {geography} {criteria}")
    seeds.extend(f"{industry} {source} {geography}" for source in directive.source_preferences)
    return _dedupe_preserve_order(seeds)


def build_company_discovery_lanes(
    directive: TieredSearchDirective,
    *,
    max_parallel_search_lanes: int = DEFAULT_MAX_PARALLEL_SEARCH_LANES,
) -> tuple[CompanyDiscoveryLane, ...]:
    """Build bounded, deterministic early company-discovery lanes."""

    if max_parallel_search_lanes < 1:
        raise ValueError("max_parallel_search_lanes must be >= 1")
    if max_parallel_search_lanes > MAX_PARALLEL_SEARCH_LANES:
        raise ValueError(f"max_parallel_search_lanes must be <= {MAX_PARALLEL_SEARCH_LANES}")

    industry = directive.industry.strip()
    geography = directive.geographic_area.strip()
    criteria = directive.research_criteria.strip()
    lane_specs = [
        ("official_site", f"{industry} official site {geography} {criteria}".strip()),
        ("local_directory", f"best {industry} companies {geography} directory"),
        ("industry_context", f"{industry} association {geography} {criteria}".strip()),
        ("growth_trigger", f"{industry} expansion hiring new location {geography}"),
    ]
    return tuple(
        CompanyDiscoveryLane(family=family, query=query, ordinal=index)
        for index, (family, query) in enumerate(lane_specs[:max_parallel_search_lanes], 1)
    )


def build_contact_discovery_queries(
    directive: TieredSearchDirective,
    company: CompanySearchTarget,
) -> tuple[str, ...]:
    """Build Tier 2 company-owned contact-channel discovery queries."""

    del directive  # Contact discovery is company-channel based, not person-role based.
    seeds: list[str] = []
    if company.website:
        domain = company.website.removeprefix("https://").removeprefix("http://").split("/", 1)[0]
        seeds.extend(
            (
                f"site:{domain} contact",
                f"site:{domain} contact us",
                f"site:{domain} phone",
                f"site:{domain} email",
                f"site:{domain} locations",
                f"site:{domain} schedule service",
            )
        )
    seeds.extend(
        (
            f"{company.name} contact",
            f"{company.name} phone",
            f"{company.name} email",
            f"{company.name} locations",
            f"{company.name} schedule service",
        )
    )
    return _dedupe_preserve_order(seeds)


def build_personalization_queries(
    directive: TieredSearchDirective,
    contact: ContactSearchTarget,
) -> tuple[str, ...]:
    """Build Tier 3 person-context queries for personalization research."""

    role = f" {contact.title.strip()}" if contact.title.strip() else ""
    return _dedupe_preserve_order(
        (
            f"{contact.name} {contact.company_name}{role}",
            f"{contact.name} {contact.company_name} interview",
            f"{contact.name} {contact.company_name} news",
            f"{contact.name} {contact.company_name} LinkedIn",
            f"{contact.name} {directive.industry} {directive.geographic_area}",
        )
    )


def _coerce_hit(tier: TierName, query: str, raw: Any, rank: int) -> TieredSearchHit:
    if isinstance(raw, Mapping):
        title = str(raw.get("title", ""))
        url = str(raw.get("url") or raw.get("href") or "")
        content = str(raw.get("content") or raw.get("body") or raw.get("snippet") or "")
        provider = str(raw.get("provider", ""))
        score_value = raw.get("score")
    else:
        title = str(getattr(raw, "title", ""))
        url = str(getattr(raw, "url", ""))
        content = str(getattr(raw, "content", ""))
        provider = str(getattr(raw, "provider", ""))
        score_value = getattr(raw, "score", None)
    score = float(score_value) if isinstance(score_value, int | float) else None
    return TieredSearchHit(
        tier=tier,
        query=query,
        title=title,
        url=url,
        content=content,
        provider=provider,
        rank=rank,
        score=score,
    )


def _is_final_enrichment_provider(provider: str, policy: ProviderPolicy) -> bool:
    return provider.lower() in set(policy.final_enrichment) - set(policy.early_discovery)


async def collect_tiered_search(
    tier: TierName,
    queries: Sequence[str],
    *,
    max_results: int = 5,
    search: SearchCallable | None = None,
    provider_policy: ProviderPolicy | None = None,
) -> TieredSearchBatch:
    """Run tier queries and normalize results without hiding partial failures.

    The required ``search`` dependency keeps early workflow tiers independent
    from the default ``AsyncMultiProviderSearch`` chain, including the Exa-backed
    provider list reserved for later approved enrichment.
    """

    if search is None:
        raise ValueError("collect_tiered_search requires an injected search callable")

    search_func = search
    policy = provider_policy or ProviderPolicy()
    normalized_queries = _dedupe_preserve_order(queries)
    hits: list[TieredSearchHit] = []
    failures: list[TieredSearchFailure] = []
    seen_hits: set[str] = set()
    for query in normalized_queries:
        try:
            results = await search_func(query, max_results)
        except Exception as exc:  # pragma: no cover - provider exception types vary
            failures.append(
                TieredSearchFailure(
                    tier=tier,
                    query=query,
                    error_class=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            continue
        for rank, result in enumerate(results, 1):
            hit = _coerce_hit(tier, query, result, rank)
            if _is_final_enrichment_provider(hit.provider, policy):
                failures.append(
                    TieredSearchFailure(
                        tier=tier,
                        query=query,
                        error_class="ProviderPolicyError",
                        error_message=(
                            f"provider {hit.provider!r} is reserved for final enrichment"
                        ),
                    )
                )
                continue
            hit_key = _hit_dedupe_key(hit)
            if hit_key in seen_hits:
                continue
            seen_hits.add(hit_key)
            hits.append(hit)
    return TieredSearchBatch(
        tier=tier,
        queries=normalized_queries,
        hits=tuple(hits),
        failures=tuple(failures),
    )


def _hit_dedupe_key(hit: TieredSearchHit) -> str:
    if hit.url.strip():
        return f"url:{hit.url.strip().casefold()}"
    return f"title:{hit.title.strip().casefold()}|content:{hit.content.strip().casefold()}"


async def collect_company_discovery_search(
    directive: TieredSearchDirective,
    *,
    max_results: int = 5,
    search: SearchCallable | None = None,
    provider_policy: ProviderPolicy | None = None,
) -> TieredSearchBatch:
    """Run Tier 1 company-discovery search for a directive."""

    return await collect_tiered_search(
        "company_discovery",
        build_company_discovery_queries(directive),
        max_results=max_results,
        search=search,
        provider_policy=provider_policy,
    )
