"""Search evidence records, prospect schemas, and citation validation.

G002 keeps the legacy ``async_multi_search.web_search`` API intact while adding
package-level records that distinguish search-result snippets from page-read text.
The wrappers are deterministic and import-safe so graph nodes can depend on them
without requiring live provider credentials during tests.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse

from async_multi_search import AllProvidersFailedError, SearchResult, web_search

SearchCallable = Callable[[str, int], Awaitable[list[SearchResult]]]


class EvidenceKind(StrEnum):
    """Supported evidence content semantics."""

    SNIPPET = "snippet"
    PAGE_READ = "page_read"


@dataclass(frozen=True)
class SearchFailure:
    """Recoverable search failure metadata captured by the wrapper."""

    query: str
    error_class: str
    error_message: str
    provider: str = ""

    @classmethod
    def from_exception(cls, query: str, error: Exception) -> SearchFailure:
        return cls(
            query=query,
            error_class=type(error).__name__,
            error_message=str(error),
        )


@dataclass(frozen=True)
class EvidenceRecord:
    """Normalized evidence item derived from search or page-read content."""

    id: str
    query: str
    title: str
    url: str
    content: str
    kind: EvidenceKind
    provider: str = ""
    rank: int | None = None
    score: float | None = None
    retrieved_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("EvidenceRecord.id is required")
        if not self.query.strip():
            raise ValueError("EvidenceRecord.query is required")
        if not self.url.strip():
            raise ValueError("EvidenceRecord.url is required")
        parsed = urlparse(self.url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"EvidenceRecord.url must be absolute, got {self.url!r}")
        if not self.content.strip():
            raise ValueError("EvidenceRecord.content is required")

    @classmethod
    def from_search_result(
        cls,
        result: SearchResult,
        *,
        query: str,
        rank: int,
        kind: EvidenceKind | str | None = None,
        retrieved_at: str | None = None,
    ) -> EvidenceRecord:
        """Convert a legacy search result into a stable evidence record."""

        normalized_kind = EvidenceKind(kind) if kind is not None else infer_evidence_kind(result)
        timestamp = retrieved_at or datetime.now(UTC).isoformat()
        return cls(
            id=stable_evidence_id(query=query, url=result.url, rank=rank, provider=result.provider),
            query=query,
            title=result.title.strip(),
            url=result.url.strip(),
            content=result.content.strip(),
            kind=normalized_kind,
            provider=result.provider,
            rank=rank,
            score=result.score,
            retrieved_at=timestamp,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data


@dataclass(frozen=True)
class SearchEvidenceArtifact:
    """A complete, serializable result of one evidence-gathering search."""

    query: str
    evidence: tuple[EvidenceRecord, ...]
    failures: tuple[SearchFailure, ...] = ()
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def ok(self) -> bool:
        return bool(self.evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "created_at": self.created_at,
            "ok": self.ok,
            "evidence": [record.to_dict() for record in self.evidence],
            "failures": [asdict(failure) for failure in self.failures],
        }


@dataclass(frozen=True)
class Citation:
    """Reference from a prospect field back to an evidence record."""

    evidence_id: str
    claim: str = ""
    quote: str = ""


@dataclass(frozen=True)
class Prospect:
    """Minimal prospect schema for research outputs."""

    name: str
    organization: str = ""
    title: str = ""
    url: str = ""
    summary: str = ""
    citations: tuple[Citation, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["citations"] = [asdict(citation) for citation in self.citations]
        return data


class CitationValidationError(ValueError):
    """Raised when a prospect citation does not resolve to evidence."""


def stable_evidence_id(*, query: str, url: str, rank: int, provider: str = "") -> str:
    """Build a deterministic evidence id for a query/result pair."""

    digest = hashlib.sha1(  # noqa: S324 - stable non-security identifier only.
        "|".join([query.strip(), url.strip(), str(rank), provider.strip()]).encode("utf-8")
    ).hexdigest()[:12]
    return f"ev_{digest}"


def infer_evidence_kind(result: SearchResult) -> EvidenceKind:
    """Infer whether a legacy result is page-read text or a SERP snippet.

    Exa is requested with ``contents.text`` in ``async_multi_search.py`` and maps
    that response into ``SearchResult.content``; the other configured providers
    currently map provider snippets/descriptions into the same field.
    """

    if result.provider.lower() == "exa":
        return EvidenceKind.PAGE_READ
    return EvidenceKind.SNIPPET


def evidence_schema() -> dict[str, Any]:
    """Return a JSON-schema-like contract for evidence artifacts."""

    return {
        "type": "object",
        "required": ["id", "query", "title", "url", "content", "kind"],
        "properties": {
            "id": {"type": "string"},
            "query": {"type": "string"},
            "title": {"type": "string"},
            "url": {"type": "string", "format": "uri"},
            "content": {"type": "string"},
            "kind": {"enum": [kind.value for kind in EvidenceKind]},
            "provider": {"type": "string"},
            "rank": {"type": ["integer", "null"]},
            "score": {"type": ["number", "null"]},
            "retrieved_at": {"type": "string"},
            "metadata": {"type": "object"},
        },
    }


def prospect_schema() -> dict[str, Any]:
    """Return a JSON-schema-like contract for prospect artifacts."""

    return {
        "type": "object",
        "required": ["name", "citations"],
        "properties": {
            "name": {"type": "string"},
            "organization": {"type": "string"},
            "title": {"type": "string"},
            "url": {"type": "string"},
            "summary": {"type": "string"},
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["evidence_id"],
                    "properties": {
                        "evidence_id": {"type": "string"},
                        "claim": {"type": "string"},
                        "quote": {"type": "string"},
                    },
                },
            },
            "metadata": {"type": "object"},
        },
    }


def validate_citations(
    prospects: Iterable[Prospect], evidence: Iterable[EvidenceRecord]
) -> None:
    """Ensure every prospect citation references an existing evidence record."""

    valid_ids = {record.id for record in evidence}
    missing: list[str] = []
    for prospect in prospects:
        for citation in prospect.citations:
            if citation.evidence_id not in valid_ids:
                missing.append(f"{prospect.name}: {citation.evidence_id}")
    if missing:
        raise CitationValidationError("Unresolved citation evidence ids: " + ", ".join(missing))


def coerce_prospect(raw: Mapping[str, Any]) -> Prospect:
    """Coerce a mapping into a ``Prospect`` and validate required shape."""

    name = str(raw.get("name", "")).strip()
    if not name:
        raise ValueError("Prospect.name is required")
    citations = tuple(
        Citation(
            evidence_id=str(citation.get("evidence_id", "")).strip(),
            claim=str(citation.get("claim", "")),
            quote=str(citation.get("quote", "")),
        )
        for citation in raw.get("citations", ())
    )
    for citation in citations:
        if not citation.evidence_id:
            raise ValueError("Citation.evidence_id is required")
    metadata = raw.get("metadata", {})
    return Prospect(
        name=name,
        organization=str(raw.get("organization", "")),
        title=str(raw.get("title", "")),
        url=str(raw.get("url", "")),
        summary=str(raw.get("summary", "")),
        citations=citations,
        metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
    )


def coerce_search_result(raw: Any) -> SearchResult:
    """Coerce legacy/mapping search output into ``SearchResult``.

    The async dispatcher normally returns ``SearchResult`` instances, but custom
    injected search callables and malformed providers can leak mapping-like or
    invalid objects. Coercion at the evidence boundary keeps failures local and
    serializable.
    """

    if isinstance(raw, SearchResult):
        return raw
    if isinstance(raw, Mapping):
        url = raw.get("url", raw.get("link", raw.get("href", "")))
        content = raw.get("content", raw.get("text", raw.get("snippet", raw.get("body", ""))))
        return SearchResult(
            title=str(raw.get("title", "")),
            url=str(url),
            content=str(content),
            score=raw.get("score") if isinstance(raw.get("score"), int | float) else None,
            provider=str(raw.get("provider", "")),
        )
    raise TypeError(f"search result must be SearchResult or mapping, got {type(raw).__name__}")


async def collect_search_evidence(
    query: str,
    *,
    max_results: int = 5,
    search_fn: SearchCallable = web_search,
) -> SearchEvidenceArtifact:
    """Run web search and normalize results into evidence records.

    The wrapper recovers from provider-chain failures by returning an artifact
    with ``ok == False`` and a populated ``failures`` list instead of raising.
    This gives graph nodes a stable branch for retry/fallback decisions.
    """

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query is required")
    try:
        results = await search_fn(normalized_query, max_results)
    except AllProvidersFailedError as exc:
        return SearchEvidenceArtifact(
            query=normalized_query,
            evidence=(),
            failures=(SearchFailure.from_exception(normalized_query, exc),),
        )
    except Exception as exc:  # defensive wrapper for injected/custom search callables.
        return SearchEvidenceArtifact(
            query=normalized_query,
            evidence=(),
            failures=(SearchFailure.from_exception(normalized_query, exc),),
        )

    retrieved_at = datetime.now(UTC).isoformat()
    evidence: list[EvidenceRecord] = []
    failures: list[SearchFailure] = []
    for index, raw_result in enumerate(results, start=1):
        provider = getattr(raw_result, "provider", "")
        try:
            result = coerce_search_result(raw_result)
            provider = result.provider
            evidence.append(
                EvidenceRecord.from_search_result(
                    result,
                    query=normalized_query,
                    rank=index,
                    retrieved_at=retrieved_at,
                )
            )
        except (TypeError, ValueError) as exc:
            failures.append(
                SearchFailure(
                    query=normalized_query,
                    provider=str(provider),
                    error_class=type(exc).__name__,
                    error_message=str(exc),
                )
            )
    return SearchEvidenceArtifact(
        query=normalized_query,
        evidence=tuple(evidence),
        failures=tuple(failures),
    )


def evidence_records_to_rows(records: Sequence[EvidenceRecord]) -> list[dict[str, Any]]:
    """Flatten evidence records for CSV-like tabular artifacts."""

    return [record.to_dict() for record in records]
