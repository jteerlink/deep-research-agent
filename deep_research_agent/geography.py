"""Geography normalization for targeted prospect-search queries."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GeoScope:
    """Normalized geography context used for search targeting and audit metadata."""

    raw: str
    canonical: str
    scope_type: str
    confidence: float
    search_terms: tuple[str, ...] = ()
    must_include_any: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "canonical": self.canonical,
            "scope_type": self.scope_type,
            "confidence": self.confidence,
            "search_terms": list(self.search_terms),
            "must_include_any": list(self.must_include_any),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class GeoAliasSuggestion:
    """Draft payload for reviewing and adding a future geography alias."""

    raw: str
    alias_key: str
    canonical: str
    scope_type: str
    confidence: float
    search_terms: tuple[str, ...]
    must_include_any: tuple[str, ...]
    source: str = "research_run"

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "alias_key": self.alias_key,
            "canonical": self.canonical,
            "scope_type": self.scope_type,
            "confidence": self.confidence,
            "search_terms": list(self.search_terms),
            "must_include_any": list(self.must_include_any),
            "source": self.source,
        }


_NORTH_TEXAS_TERMS = (
    "North Texas",
    "Dallas-Fort Worth TX",
    "DFW",
    "Dallas TX",
    "Fort Worth TX",
    "Plano TX",
    "Frisco TX",
    "Denton TX",
    "Arlington TX",
    "McKinney TX",
)

_DFW_TERMS = (
    "DFW",
    "Dallas-Fort Worth TX",
    "Dallas TX",
    "Fort Worth TX",
    "Plano TX",
    "Frisco TX",
    "Denton TX",
    "Arlington TX",
    "McKinney TX",
)

GeoAliasRecord = tuple[str, str, tuple[str, ...], tuple[str, ...], float]


def _clean(value: str) -> str:
    return " ".join(str(value).strip().split())


def _alias_key(value: str) -> str:
    lowered = value.lower().replace("&", " and ")
    lowered = re.sub(r"[/_,]+", " ", lowered)
    lowered = re.sub(r"[-]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _dedupe(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: list[str] = []
    normalized_seen: set[str] = set()
    for value in values:
        cleaned = _clean(value)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key not in normalized_seen:
            normalized_seen.add(key)
            seen.append(cleaned)
    return tuple(seen)


def _alias_group(
    *,
    canonical: str,
    scope_type: str,
    terms: tuple[str, ...],
    aliases: tuple[str, ...],
    must_include_any: tuple[str, ...],
    confidence: float,
) -> dict[str, GeoAliasRecord]:
    record = (canonical, scope_type, terms, must_include_any, confidence)
    return {_alias_key(alias): record for alias in aliases}


_ALIASES: dict[str, GeoAliasRecord] = {}
_ALIASES.update(
    _alias_group(
        canonical="Dallas-Fort Worth TX",
        scope_type="regional_area",
        terms=_NORTH_TEXAS_TERMS,
        aliases=("North Texas", "N Texas", "North TX", "N TX"),
        must_include_any=("North Texas", "Dallas", "Fort Worth", "DFW"),
        confidence=0.75,
    )
)
_ALIASES.update(
    _alias_group(
        canonical="Dallas-Fort Worth TX",
        scope_type="metro_area",
        terms=_DFW_TERMS,
        aliases=(
            "DFW",
            "DFW area",
            "Dallas-Fort Worth",
            "Dallas-Fort Worth TX",
            "Dallas Fort Worth",
            "Dallas Fort Worth TX",
            "Dallas Forth Worth",
            "Dallas Forth Worth TX",
        ),
        must_include_any=("DFW", "Dallas", "Fort Worth"),
        confidence=0.9,
    )
)
_ALIASES.update(
    _alias_group(
        canonical="San Francisco Bay Area CA",
        scope_type="regional_area",
        terms=(
            "Bay Area CA",
            "San Francisco Bay Area",
            "San Francisco CA",
            "San Jose CA",
            "Oakland CA",
            "Palo Alto CA",
            "Sunnyvale CA",
            "Berkeley CA",
        ),
        aliases=("Bay Area", "SF Bay Area", "San Francisco Bay Area", "Bay Area CA"),
        must_include_any=("Bay Area", "San Francisco", "San Jose", "Oakland"),
        confidence=0.82,
    )
)
_ALIASES.update(
    _alias_group(
        canonical="Washington DC metro",
        scope_type="regional_area",
        terms=(
            "DMV",
            "Washington DC metro",
            "Washington DC",
            "Northern Virginia",
            "Arlington VA",
            "Alexandria VA",
            "Fairfax VA",
            "Bethesda MD",
            "Silver Spring MD",
        ),
        aliases=("DMV", "DC area", "Washington DC area", "Washington DC metro"),
        must_include_any=("DMV", "Washington DC", "Northern Virginia", "Maryland"),
        confidence=0.78,
    )
)
_ALIASES.update(
    _alias_group(
        canonical="Miami-Fort Lauderdale-West Palm Beach FL",
        scope_type="regional_area",
        terms=(
            "South Florida",
            "Miami FL",
            "Fort Lauderdale FL",
            "West Palm Beach FL",
            "Boca Raton FL",
            "Hollywood FL",
        ),
        aliases=("South Florida", "Miami-Fort Lauderdale", "Miami Fort Lauderdale"),
        must_include_any=("South Florida", "Miami", "Fort Lauderdale", "West Palm Beach"),
        confidence=0.8,
    )
)
_ALIASES.update(
    _alias_group(
        canonical="Orlando-Central Florida",
        scope_type="regional_area",
        terms=(
            "Central Florida",
            "Orlando FL",
            "Kissimmee FL",
            "Sanford FL",
            "Winter Park FL",
            "Lakeland FL",
        ),
        aliases=("Central Florida", "Orlando area", "Orlando metro"),
        must_include_any=("Central Florida", "Orlando", "Kissimmee", "Sanford"),
        confidence=0.74,
    )
)
_ALIASES.update(
    _alias_group(
        canonical="Tampa Bay FL",
        scope_type="regional_area",
        terms=(
            "Tampa Bay",
            "Tampa FL",
            "St Petersburg FL",
            "Clearwater FL",
            "Brandon FL",
            "Sarasota FL",
        ),
        aliases=("Tampa Bay", "Tampa Bay area", "Tampa-St Petersburg", "Tampa St Petersburg"),
        must_include_any=("Tampa Bay", "Tampa", "St Petersburg", "Clearwater"),
        confidence=0.82,
    )
)
_ALIASES.update(
    _alias_group(
        canonical="Chicago metro IL",
        scope_type="regional_area",
        terms=(
            "Chicagoland",
            "Chicago IL",
            "Naperville IL",
            "Schaumburg IL",
            "Evanston IL",
            "Joliet IL",
            "Aurora IL",
        ),
        aliases=("Chicagoland", "Chicago area", "Chicago metro", "Greater Chicago"),
        must_include_any=("Chicagoland", "Chicago", "Naperville"),
        confidence=0.82,
    )
)
_ALIASES.update(
    _alias_group(
        canonical="Atlanta metro GA",
        scope_type="metro_area",
        terms=(
            "Greater Atlanta",
            "Atlanta GA",
            "Sandy Springs GA",
            "Marietta GA",
            "Alpharetta GA",
            "Decatur GA",
        ),
        aliases=("Greater Atlanta", "Atlanta metro", "Atlanta area"),
        must_include_any=("Greater Atlanta", "Atlanta", "Sandy Springs"),
        confidence=0.86,
    )
)
_ALIASES.update(
    _alias_group(
        canonical="Houston metro TX",
        scope_type="metro_area",
        terms=(
            "Greater Houston",
            "Houston TX",
            "The Woodlands TX",
            "Sugar Land TX",
            "Katy TX",
            "Pasadena TX",
        ),
        aliases=("Greater Houston", "Houston metro", "Houston area"),
        must_include_any=("Greater Houston", "Houston", "The Woodlands", "Sugar Land"),
        confidence=0.86,
    )
)
_ALIASES.update(
    _alias_group(
        canonical="Phoenix metro AZ",
        scope_type="metro_area",
        terms=(
            "Phoenix metro",
            "Phoenix AZ",
            "Scottsdale AZ",
            "Mesa AZ",
            "Tempe AZ",
            "Chandler AZ",
            "Glendale AZ",
        ),
        aliases=("Greater Phoenix", "Phoenix metro", "Phoenix area", "Valley of the Sun"),
        must_include_any=("Phoenix", "Scottsdale", "Mesa", "Tempe"),
        confidence=0.82,
    )
)
_ALIASES.update(
    _alias_group(
        canonical="Seattle-Puget Sound WA",
        scope_type="regional_area",
        terms=(
            "Puget Sound",
            "Seattle metro",
            "Seattle WA",
            "Bellevue WA",
            "Tacoma WA",
            "Everett WA",
            "Redmond WA",
        ),
        aliases=("Puget Sound", "Greater Seattle", "Seattle metro", "Seattle area"),
        must_include_any=("Puget Sound", "Seattle", "Bellevue", "Tacoma"),
        confidence=0.8,
    )
)
_ALIASES.update(
    _alias_group(
        canonical="Minneapolis-Saint Paul MN",
        scope_type="metro_area",
        terms=(
            "Twin Cities",
            "Minneapolis MN",
            "Saint Paul MN",
            "St Paul MN",
            "Bloomington MN",
            "Eden Prairie MN",
            "Minnetonka MN",
        ),
        aliases=("Twin Cities", "Minneapolis-St Paul", "Minneapolis Saint Paul", "MSP"),
        must_include_any=("Twin Cities", "Minneapolis", "Saint Paul", "St Paul"),
        confidence=0.84,
    )
)
_ALIASES.update(
    _alias_group(
        canonical="Research Triangle NC",
        scope_type="regional_area",
        terms=(
            "Research Triangle NC",
            "Raleigh NC",
            "Durham NC",
            "Chapel Hill NC",
            "Cary NC",
        ),
        aliases=("Research Triangle", "The Triangle NC", "Raleigh-Durham", "Raleigh Durham"),
        must_include_any=("Research Triangle", "Raleigh", "Durham", "Chapel Hill"),
        confidence=0.82,
    )
)
_ALIASES.update(
    _alias_group(
        canonical="Inland Empire CA",
        scope_type="regional_area",
        terms=(
            "Inland Empire CA",
            "Riverside CA",
            "San Bernardino CA",
            "Ontario CA",
            "Rancho Cucamonga CA",
            "Corona CA",
        ),
        aliases=("Inland Empire", "Inland Empire CA", "Riverside-San Bernardino"),
        must_include_any=("Inland Empire", "Riverside", "San Bernardino"),
        confidence=0.82,
    )
)
_ALIASES.update(
    _alias_group(
        canonical="Philadelphia metro",
        scope_type="regional_area",
        terms=(
            "Philadelphia metro",
            "Philadelphia PA",
            "Camden NJ",
            "Wilmington DE",
            "King of Prussia PA",
            "Cherry Hill NJ",
        ),
        aliases=("Delaware Valley", "Greater Philadelphia", "Philadelphia metro", "Philly area"),
        must_include_any=("Philadelphia", "Delaware Valley", "Camden", "Wilmington"),
        confidence=0.78,
    )
)
_ALIASES.update(
    _alias_group(
        canonical="Boston metro MA",
        scope_type="metro_area",
        terms=(
            "Greater Boston",
            "Boston MA",
            "Cambridge MA",
            "Somerville MA",
            "Newton MA",
            "Quincy MA",
            "Waltham MA",
        ),
        aliases=("Greater Boston", "Boston metro", "Boston area"),
        must_include_any=("Greater Boston", "Boston", "Cambridge"),
        confidence=0.86,
    )
)
_ALIASES.update(
    _alias_group(
        canonical="Denver Front Range CO",
        scope_type="regional_area",
        terms=(
            "Front Range CO",
            "Denver CO",
            "Aurora CO",
            "Boulder CO",
            "Lakewood CO",
            "Fort Collins CO",
        ),
        aliases=("Front Range", "Colorado Front Range", "Denver metro", "Greater Denver"),
        must_include_any=("Front Range", "Denver", "Boulder", "Aurora"),
        confidence=0.72,
    )
)

_AMBIGUOUS_REGION_PATTERNS = (
    re.compile(r"\b(area|region|metro|tri state|tri-state|valley)\b"),
    re.compile(r"^(north|south|east|west|central)\b"),
)


def normalize_geography(raw: str) -> GeoScope:
    """Normalize an operator-entered geography into search-targeting terms."""

    cleaned = _clean(raw)
    if not cleaned:
        return GeoScope(
            raw="",
            canonical="",
            scope_type="none",
            confidence=1.0,
        )

    alias_key = _alias_key(cleaned)
    alias = _ALIASES.get(alias_key)
    if alias is not None:
        canonical, scope_type, terms, must_include_any, confidence = alias
        search_terms = _dedupe((cleaned, *terms))
        return GeoScope(
            raw=cleaned,
            canonical=canonical,
            scope_type=scope_type,
            confidence=confidence,
            search_terms=search_terms,
            must_include_any=_dedupe(must_include_any),
        )

    warnings: tuple[str, ...] = ()
    if _looks_region_ambiguous(cleaned):
        warnings = ("Geography was not recognized; searching the raw phrase only.",)

    return GeoScope(
        raw=cleaned,
        canonical=cleaned,
        scope_type="raw",
        confidence=0.5 if warnings else 0.65,
        search_terms=(cleaned,),
        must_include_any=(cleaned,),
        warnings=warnings,
    )


def suggest_geography_alias_update(
    raw: str,
    observed_terms: Sequence[str] = (),
    *,
    source: str = "research_run",
) -> GeoAliasSuggestion | None:
    """Draft a reviewed alias-map update for an unknown broad geography.

    The helper intentionally does not mutate ``_ALIASES`` or source files. It
    creates an auditable payload that a later workflow can review, edit, and
    promote into the static alias map.
    """

    scope = normalize_geography(raw)
    if not scope.raw or not scope.warnings:
        return None

    terms = _dedupe((scope.raw, *(str(term) for term in observed_terms)))
    must_include_any = terms[: min(4, len(terms))]
    return GeoAliasSuggestion(
        raw=scope.raw,
        alias_key=_alias_key(scope.raw),
        canonical=scope.canonical,
        scope_type="regional_area",
        confidence=0.4,
        search_terms=terms,
        must_include_any=must_include_any,
        source=source,
    )


def _looks_region_ambiguous(value: str) -> bool:
    alias_key = _alias_key(value)
    return any(pattern.search(alias_key) for pattern in _AMBIGUOUS_REGION_PATTERNS)
