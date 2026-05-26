"""Tier-specific search helpers for prospect research.

The tiered workflow starts with broad company discovery before spending budget on
contact and person-level enrichment.  This module intentionally keeps the search
provider boundary injectable so early tiers and tests can run without
constructing the legacy ``AsyncMultiProviderSearch`` provider chain.
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
    target_prospect_count: int = 10
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
    """Build Tier 2 company-scoped contact-discovery queries."""

    roles = directive.preferred_contact_roles or (
        "owner",
        "founder",
        "CEO",
        "marketing director",
    )
    seeds = [f"{company.name} {role}" for role in roles]
    if company.website:
        domain = company.website.removeprefix("https://").removeprefix("http://").split("/", 1)[0]
        seeds.extend((f"site:{domain} team", f"site:{domain} about", f"site:{domain} leadership"))
    seeds.append(f"site:linkedin.com/in {company.name} {' OR '.join(roles[:3])}")
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
            hits.append(hit)
    return TieredSearchBatch(
        tier=tier,
        queries=normalized_queries,
        hits=tuple(hits),
        failures=tuple(failures),
    )


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
