"""Search evidence normalization for the G002 research artifact contract."""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import async_multi_search
from async_multi_search import DEFAULT_MAX_RESULTS, SearchResult

EvidenceSourceType = Literal["snippet", "page_read"]

# Exa returns requested text contents in the legacy adapter; the other configured
# providers expose search-result summaries/snippets in async_multi_search.py.
_PAGE_READ_PROVIDERS = frozenset({"exa"})


@dataclass(frozen=True)
class EvidenceRecord:
    """A normalized citation target produced from web search results.

    ``source_type`` distinguishes full-page/read evidence from search snippets so
    downstream prospect summaries can avoid treating snippets as verified page
    reads.
    """

    id: str
    query: str
    title: str
    url: str
    text: str
    source_type: EvidenceSourceType
    provider: str
    rank: int
    score: float | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("EvidenceRecord.id is required")
        if self.source_type not in {"snippet", "page_read"}:
            raise ValueError("EvidenceRecord.source_type must be 'snippet' or 'page_read'")
        if self.rank < 1:
            raise ValueError("EvidenceRecord.rank must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True)
class SearchFailure:
    """Non-throwing wrapper metadata for a failed search attempt."""

    query: str
    error_class: str
    error_message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class SearchEvidenceBatch:
    """The normalized result of running a search query for evidence."""

    query: str
    records: tuple[EvidenceRecord, ...]
    failures: tuple[SearchFailure, ...] = ()

    @property
    def ok(self) -> bool:
        return bool(self.records) and not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "ok": self.ok,
            "records": [record.to_dict() for record in self.records],
            "failures": [failure.to_dict() for failure in self.failures],
        }


def evidence_source_type(provider: str) -> EvidenceSourceType:
    """Return the evidence semantics for a legacy search provider name."""

    return "page_read" if provider.lower() in _PAGE_READ_PROVIDERS else "snippet"


def evidence_id_for(query: str, result: SearchResult, rank: int) -> str:
    """Build a stable evidence id from query/result identity."""

    provider = result.provider or "search"
    digest = hashlib.sha1(  # noqa: S324 - deterministic identifier, not security-sensitive
        f"{query}\0{provider}\0{result.url}\0{rank}".encode()
    ).hexdigest()[:12]
    return f"ev_{digest}"


def normalize_search_results(query: str, results: Sequence[SearchResult]) -> tuple[EvidenceRecord, ...]:
    """Convert legacy ``SearchResult`` items into citation-ready evidence records."""

    records: list[EvidenceRecord] = []
    for index, result in enumerate(results, start=1):
        provider = result.provider or "search"
        records.append(
            EvidenceRecord(
                id=evidence_id_for(query, result, index),
                query=query,
                title=result.title,
                url=result.url,
                text=result.content,
                source_type=evidence_source_type(provider),
                provider=provider,
                rank=index,
                score=result.score,
            )
        )
    return tuple(records)


async def collect_search_evidence(
    query: str,
    *,
    max_results: int = DEFAULT_MAX_RESULTS,
    search: Callable[[str, int], Awaitable[Sequence[SearchResult]]] | None = None,
) -> SearchEvidenceBatch:
    """Run ``async_multi_search.web_search`` and normalize results without throwing.

    The legacy function remains available and unchanged. This wrapper catches
    exhausted-provider failures so artifact generation can record recoverable
    search failures instead of crashing the full research run.
    """

    search_func = async_multi_search.web_search if search is None else search
    try:
        results = await search_func(query, max_results)
    except Exception as exc:  # pragma: no cover - exact exception class is provider-dependent
        return SearchEvidenceBatch(
            query=query,
            records=(),
            failures=(
                SearchFailure(
                    query=query,
                    error_class=type(exc).__name__,
                    error_message=str(exc),
                ),
            ),
        )
    return SearchEvidenceBatch(query=query, records=normalize_search_results(query, results))
