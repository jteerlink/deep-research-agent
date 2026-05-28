"""Deterministic qualification gates for tiered prospect discovery.

Search hits are discovery evidence, not review-ready prospects.  This module
keeps the first quality gate offline and dependency-free so live search results
can be filtered, scored, and audited before later runtime/UI layers promote
them into selectable pre-enrichment rows.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse

from .tiered_models import (
    CompanyProspect,
    ContactCandidate,
    RoleCategory,
    SearchDirective,
    SourceConfidence,
)
from .tiered_search import TieredSearchHit, TierName

QualificationStatus = Literal["accepted", "rejected", "needs_contact"]

_YEAR_PATTERN = re.compile(r"\b20\d{2}\b")
_LISTICLE_PHRASES = (
    "best",
    "top",
    "list",
    "companies",
    "company list",
    "contractors",
    "near me",
    "directory",
    "reviews",
    "reviewed",
)
_DIRECTORY_DOMAINS = (
    "angi.com",
    "bbb.org",
    "chamberofcommerce.com",
    "consumeraffairs.com",
    "contractorsup.com",
    "homeadvisor.com",
    "houzz.com",
    "thumbtack.com",
    "yelp.com",
    "yellowpages.com",
)
_CONTENT_DOMAINS = (
    "blogspot.",
    "forbes.com",
    "medium.com",
    "news.",
    "substack.com",
)
_MARKETING_TERMS = (
    "seo",
    "marketing",
    "lead generation",
    "advertising",
    "digital agency",
    "growth agency",
    "website design",
)
_COMPANY_SUFFIXES = (
    "air conditioning",
    "heating",
    "cooling",
    "dental",
    "orthodontics",
    "hvac",
    "plumbing",
    "roofing",
    "electric",
    "med spa",
    "law",
    "clinic",
    "services",
    "service",
)
_GENERIC_COMPANY_TITLES = {
    "home",
    "about",
    "about us",
    "contact",
    "services",
    "team",
    "leadership",
}
_ROLE_KEYWORDS: tuple[tuple[str, RoleCategory], ...] = (
    ("owner", "owner"),
    ("founder", "owner"),
    ("president", "executive"),
    ("ceo", "executive"),
    ("chief", "executive"),
    ("principal", "executive"),
    ("general manager", "operator"),
    ("manager", "operator"),
    ("director", "operator"),
    ("operations", "operator"),
    ("marketing", "marketing"),
    ("growth", "marketing"),
)
_PERSON_RE = re.compile(
    r"^(?P<name>(?:Dr\.\s*)?[A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){1,3})"
    r"(?:\s*[-–—|,]\s*(?P<title>.+?))?(?:\s+(?:at|@)\s+(?P<company>.+))?$"
)


@dataclass(frozen=True)
class QualificationAuditRecord:
    """Traceable decision made while qualifying one discovery hit."""

    tier: TierName
    status: QualificationStatus
    title: str
    url: str
    provider: str = ""
    query: str = ""
    reasons: tuple[str, ...] = ()
    evidence_id: str = ""
    normalized_name: str = ""
    source_confidence: SourceConfidence = "low"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "status": self.status,
            "title": self.title,
            "url": self.url,
            "provider": self.provider,
            "query": self.query,
            "reasons": list(self.reasons),
            "evidence_id": self.evidence_id,
            "normalized_name": self.normalized_name,
            "source_confidence": self.source_confidence,
        }


@dataclass(frozen=True)
class QualifiedCompanyCandidate:
    """Accepted company evidence ready to become a company prospect."""

    company_id: str
    name: str
    website: str
    score: float
    confidence: SourceConfidence
    evidence_hit: TieredSearchHit
    evidence_id: str
    reasons: tuple[str, ...]

    def to_company_prospect(self, directive: SearchDirective) -> CompanyProspect:
        return CompanyProspect(
            company_id=self.company_id,
            name=self.name,
            website=self.website,
            industry=directive.industry,
            geographic_area=directive.geographic_area,
            fit_score=self.score,
            fit_rationale=_safe_summary(self.evidence_hit.content, self.reasons),
            evidence_ids=(self.evidence_id,),
            source_confidence=self.confidence,
        )


@dataclass(frozen=True)
class QualifiedContactCandidate:
    """Accepted person/contact evidence tied to a qualified company."""

    contact_id: str
    company_id: str
    name: str
    title: str
    role_category: RoleCategory
    score: float
    evidence_hit: TieredSearchHit
    evidence_id: str
    source_url: str
    reasons: tuple[str, ...]

    def to_contact_candidate(self) -> ContactCandidate:
        return ContactCandidate(
            contact_id=self.contact_id,
            company_id=self.company_id,
            name=self.name,
            title=self.title,
            role_category=self.role_category,
            profile_urls=(self.source_url,) if self.source_url else (),
            contact_confidence=self.score,
            evidence_ids=(self.evidence_id,),
            notes=_safe_summary(self.evidence_hit.content, self.reasons),
        )


@dataclass(frozen=True)
class QualifiedCompanyBatch:
    """Company qualification result plus a full accepted/rejected audit."""

    accepted: tuple[QualifiedCompanyCandidate, ...] = ()
    audit_records: tuple[QualificationAuditRecord, ...] = ()

    @property
    def rejected(self) -> tuple[QualificationAuditRecord, ...]:
        return tuple(record for record in self.audit_records if record.status == "rejected")


@dataclass(frozen=True)
class QualifiedContactBatch:
    """Contact qualification result plus accepted/rejected/needs-contact audit."""

    accepted: tuple[QualifiedContactCandidate, ...] = ()
    audit_records: tuple[QualificationAuditRecord, ...] = ()

    @property
    def rejected(self) -> tuple[QualificationAuditRecord, ...]:
        return tuple(record for record in self.audit_records if record.status == "rejected")

    @property
    def needs_contact(self) -> tuple[QualificationAuditRecord, ...]:
        return tuple(record for record in self.audit_records if record.status == "needs_contact")


def qualify_company_hits(
    hits: Iterable[TieredSearchHit],
    directive: SearchDirective,
    target_count: int | None = None,
    oversample_factor: int = 4,
) -> QualifiedCompanyBatch:
    """Qualify raw company-discovery hits into real-company candidates.

    ``oversample_factor`` bounds how many raw hits are inspected before the
    target is met; it does not trigger provider calls itself.  The caller owns
    live-search budgets and can pass a pre-bounded hit collection.
    """

    target = target_count or directive.target_prospect_count
    if target < 1:
        raise ValueError("target_count must be >= 1")
    if oversample_factor < 1:
        raise ValueError("oversample_factor must be >= 1")

    accepted: list[QualifiedCompanyCandidate] = []
    audit: list[QualificationAuditRecord] = []
    seen: set[str] = set()
    max_evaluated = target * oversample_factor
    for index, hit in enumerate(tuple(hits)[:max_evaluated], start=1):
        candidate = _company_candidate_from_hit(hit)
        reasons = _company_rejection_reasons(hit, candidate, directive)
        key = candidate.casefold()
        if key in seen:
            reasons = (*reasons, "duplicate_company")
        if reasons:
            audit.append(_audit(hit, "rejected", reasons, candidate))
            continue

        seen.add(key)
        score = _company_score(hit, candidate)
        confidence = _confidence(score)
        company_id = f"company_{_slug(candidate) or index:0>3}"
        evidence_id = f"ev_{company_id}_search_{hit.rank:03d}"
        accepted_reasons = _accepted_company_reasons(hit)
        qualified = QualifiedCompanyCandidate(
            company_id=company_id,
            name=candidate,
            website=hit.url,
            score=score,
            confidence=confidence,
            evidence_hit=hit,
            evidence_id=evidence_id,
            reasons=accepted_reasons,
        )
        accepted.append(qualified)
        audit.append(_audit(hit, "accepted", accepted_reasons, candidate, evidence_id, confidence))
        if len(accepted) >= target:
            break

    return QualifiedCompanyBatch(accepted=tuple(accepted), audit_records=tuple(audit))


def qualify_contact_hits(
    hits: Iterable[TieredSearchHit],
    company: CompanyProspect | QualifiedCompanyCandidate,
    directive: SearchDirective,
    max_contacts: int = 3,
) -> QualifiedContactBatch:
    """Qualify contact-discovery hits into person-like company-tied contacts."""

    if max_contacts < 1:
        raise ValueError("max_contacts must be >= 1")

    accepted: list[QualifiedContactCandidate] = []
    audit: list[QualificationAuditRecord] = []
    seen: set[str] = set()
    company_name = company.name
    company_id = company.company_id
    for hit in hits:
        name, title = _contact_name_and_title(hit)
        reasons = _contact_rejection_reasons(hit, name, title, company, directive)
        key = name.casefold()
        if key and key in seen:
            reasons = (*reasons, "duplicate_contact")
        if reasons:
            audit.append(_audit(hit, "rejected", reasons, name or title or company_name))
            continue

        seen.add(key)
        score = _contact_score(hit, title, directive)
        role = _role_category(title)
        contact_id = f"contact_{_slug(company_id)}_{len(accepted) + 1:03d}"
        evidence_id = f"ev_{contact_id}_search_{hit.rank:03d}"
        accepted_reasons = _accepted_contact_reasons(hit, title, company, directive)
        qualified = QualifiedContactCandidate(
            contact_id=contact_id,
            company_id=company_id,
            name=name,
            title=title,
            role_category=role,
            score=score,
            evidence_hit=hit,
            evidence_id=evidence_id,
            source_url=hit.url,
            reasons=accepted_reasons,
        )
        accepted.append(qualified)
        audit.append(
            _audit(hit, "accepted", accepted_reasons, name, evidence_id, _confidence(score))
        )
        if len(accepted) >= max_contacts:
            break

    if not accepted:
        audit.append(
            QualificationAuditRecord(
                tier="contact_discovery",
                status="needs_contact",
                title=f"No verified contact for {company_name}",
                url=company.website,
                reasons=("qualified_company_needs_contact",),
                normalized_name=company_name,
                source_confidence="medium",
            )
        )

    return QualifiedContactBatch(accepted=tuple(accepted), audit_records=tuple(audit))


def _company_candidate_from_hit(hit: TieredSearchHit) -> str:
    title = _clean_title(hit.title)
    primary = re.split(r"\s+[|–—]\s+|\s+-\s+", title, maxsplit=1)[0].strip()
    primary = _strip_company_tail(primary)
    if _is_generic_company_name(primary):
        from_content = _company_from_content(hit.content)
        if from_content:
            return from_content
        return ""
    return primary


def _company_rejection_reasons(
    hit: TieredSearchHit,
    candidate: str,
    directive: SearchDirective,
) -> tuple[str, ...]:
    text = f"{hit.title} {hit.content}".casefold()
    domain = _domain(hit.url)
    reasons: list[str] = []
    if not candidate:
        reasons.append("not_specific_company")
    if _is_marketing_page(text):
        reasons.append("non_target_marketing_page")
    if _looks_like_listicle(hit.title, hit.content, domain):
        reasons.append("listicle_or_directory")
    if _is_content_domain(domain):
        reasons.append("article_or_blog_source")
    if candidate and not _looks_official_domain(domain) and _is_generic_company_name(candidate):
        reasons.append("not_specific_company")
    reasons.extend(
        _negative_criteria_reasons(
            directive.negative_criteria,
            hit.title,
            hit.content,
            candidate,
        )
    )
    return _dedupe(reasons)


def _accepted_company_reasons(hit: TieredSearchHit) -> tuple[str, ...]:
    reasons = ["specific_company_identity"]
    if _looks_official_domain(_domain(hit.url)):
        reasons.append("official_domain")
    if hit.provider:
        reasons.append("provider_evidence")
    return tuple(reasons)


def _contact_name_and_title(hit: TieredSearchHit) -> tuple[str, str]:
    title = _clean_title(hit.title)
    title = re.split(r"\s+[|–—]\s+LinkedIn\b", title, maxsplit=1, flags=re.IGNORECASE)[0]
    match = _PERSON_RE.match(title)
    if not match:
        return "", ""
    name = _normalize_person_name(match.group("name") or "")
    raw_title = (match.group("title") or "").strip()
    company = match.group("company")
    if company and not re.search(r"\b(at|@)\b", raw_title, flags=re.IGNORECASE):
        raw_title = re.sub(r"\s+\b(?:at|@)\b\s+.+$", "", raw_title, flags=re.IGNORECASE)
    raw_title = re.sub(r"\s+\b(?:at|@)\b\s+.+$", "", raw_title, flags=re.IGNORECASE).strip()
    return name, _strip_title_noise(raw_title)


def _contact_rejection_reasons(
    hit: TieredSearchHit,
    name: str,
    title: str,
    company: CompanyProspect | QualifiedCompanyCandidate,
    directive: SearchDirective,
) -> tuple[str, ...]:
    text = f"{hit.title} {hit.content}".casefold()
    reasons: list[str] = []
    if _looks_like_listicle(hit.title, hit.content, _domain(hit.url)):
        reasons.append("article_or_list_title")
    if not name or not _looks_like_person(name):
        reasons.append("missing_person_name")
    if not _contact_tied_to_company(hit, company):
        reasons.append("missing_company_evidence")
    if _is_marketing_page(text) and directive.industry.casefold() in text:
        reasons.append("non_target_marketing_page")
    if name and _looks_like_company_name(name):
        reasons.append("not_person_identity")
    if title and _looks_like_listicle(title, "", _domain(hit.url)):
        reasons.append("article_or_list_title")
    reasons.extend(
        _negative_criteria_reasons(
            directive.negative_criteria,
            hit.title,
            hit.content,
            name,
            title,
        )
    )
    return _dedupe(reasons)


def _negative_criteria_reasons(
    negative_criteria: str,
    *values: str,
) -> tuple[str, ...]:
    text = " ".join(value for value in values if value).casefold()
    reasons = [
        f"negative_criteria_match:{phrase}"
        for phrase in _negative_criteria_phrases(negative_criteria)
        if phrase.casefold() in text
    ]
    return tuple(reasons)


def _negative_criteria_phrases(value: str) -> tuple[str, ...]:
    return tuple(
        phrase.strip()
        for phrase in re.split(r"[,;]", value)
        if phrase.strip()
    )


def _accepted_contact_reasons(
    hit: TieredSearchHit,
    title: str,
    company: CompanyProspect | QualifiedCompanyCandidate,
    directive: SearchDirective,
) -> tuple[str, ...]:
    reasons = ["person_like_name", "company_tied_evidence"]
    if _role_category(title) != "unknown":
        reasons.append("role_evidence")
    preferred = tuple(role.casefold() for role in directive.preferred_contact_roles)
    if preferred and any(role in title.casefold() for role in preferred):
        reasons.append("preferred_role")
    if _company_domain(company) and _company_domain(company) in _domain(hit.url):
        reasons.append("company_domain_source")
    if "linkedin.com/in" in hit.url.casefold():
        reasons.append("linkedin_profile_source")
    return tuple(reasons)


def _contact_tied_to_company(
    hit: TieredSearchHit,
    company: CompanyProspect | QualifiedCompanyCandidate,
) -> bool:
    text = f"{hit.title} {hit.content}".casefold()
    name_key = _token_key(company.name)
    domain = _domain(hit.url)
    company_domain = _company_domain(company)
    if company_domain and company_domain in domain:
        return True
    if company_domain and company_domain in text:
        return True
    return bool(name_key and name_key in _token_key(text))


def _company_score(hit: TieredSearchHit, candidate: str) -> float:
    score = 0.5
    if _looks_official_domain(_domain(hit.url)):
        score += 0.25
    if hit.provider:
        score += 0.08
    if any(term in candidate.casefold() for term in _COMPANY_SUFFIXES):
        score += 0.07
    if hit.score:
        score += min(max(hit.score, 0.0), 1.0) * 0.1
    return round(min(score, 0.95), 2)


def _contact_score(
    hit: TieredSearchHit,
    title: str,
    directive: SearchDirective,
) -> float:
    score = 0.52
    if _role_category(title) != "unknown":
        score += 0.13
    if any(role.casefold() in title.casefold() for role in directive.preferred_contact_roles):
        score += 0.12
    if _looks_official_domain(_domain(hit.url)):
        score += 0.1
    if "linkedin.com/in" in hit.url.casefold():
        score += 0.08
    if hit.provider:
        score += 0.05
    return round(min(score, 0.92), 2)


def _audit(
    hit: TieredSearchHit,
    status: QualificationStatus,
    reasons: Iterable[str],
    normalized_name: str,
    evidence_id: str = "",
    confidence: SourceConfidence = "low",
) -> QualificationAuditRecord:
    return QualificationAuditRecord(
        tier=hit.tier,
        status=status,
        title=hit.title,
        url=hit.url,
        provider=hit.provider,
        query=hit.query,
        reasons=tuple(reasons),
        evidence_id=evidence_id,
        normalized_name=normalized_name,
        source_confidence=confidence,
    )


def _looks_like_listicle(title: str, content: str, domain: str) -> bool:
    title_key = title.casefold()
    text = f"{title} {content}".casefold()
    if any(domain.endswith(noisy) for noisy in _DIRECTORY_DOMAINS):
        return True
    if _YEAR_PATTERN.search(title_key) and any(term in title_key for term in _LISTICLE_PHRASES):
        return True
    if re.search(r"\b(?:best|top)\s+\d+\b", title_key):
        return True
    return any(term in text for term in _LISTICLE_PHRASES) and bool(
        re.search(r"\b(?:companies|contractors|near me|directory|reviews)\b", text)
    )


def _is_marketing_page(text: str) -> bool:
    return any(term in text for term in _MARKETING_TERMS)


def _is_content_domain(domain: str) -> bool:
    return any(marker in domain for marker in _CONTENT_DOMAINS)


def _looks_official_domain(domain: str) -> bool:
    return bool(domain) and not any(domain.endswith(noisy) for noisy in _DIRECTORY_DOMAINS)


def _is_generic_company_name(value: str) -> bool:
    normalized = value.strip().casefold()
    if not normalized or normalized in _GENERIC_COMPANY_TITLES:
        return True
    if _YEAR_PATTERN.search(normalized):
        return True
    return bool(
        re.search(
            r"\b(?:best|top)\b.*\b(?:companies|contractors|near me|reviews|directory)\b",
            normalized,
        )
    )


def _looks_like_company_name(value: str) -> bool:
    normalized = value.casefold()
    return any(term in normalized for term in _COMPANY_SUFFIXES)


def _looks_like_person(value: str) -> bool:
    words = value.split()
    if len(words) < 2 or len(words) > 5:
        return False
    generic_words = {"best", "top", "hvac", "companies", "contractors"}
    if any(word.casefold() in generic_words for word in words):
        return False
    return all(word[:1].isupper() or word in {"Dr."} for word in words)


def _company_from_content(content: str) -> str:
    match = re.search(
        r"\b([A-Z][A-Za-z&'.-]+(?:\s+[A-Z][A-Za-z&'.-]+){1,5})\s+"
        r"(?:is|serves|offers|provides|specializes|has)\b",
        content,
    )
    if not match:
        return ""
    return _strip_company_tail(match.group(1))


def _strip_company_tail(value: str) -> str:
    candidate = re.sub(r"\s+", " ", value).strip(" -–—|")
    candidate = re.sub(r"\b(?:DFW|Dallas|Fort Worth|TX|Texas)\b\.?", "", candidate).strip()
    candidate = re.sub(r"\b(?:Official Site|Home Page|Homepage|Website)\b", "", candidate).strip()
    candidate = re.sub(r"\s+", " ", candidate).strip(" -–—|")
    return candidate


def _strip_title_noise(value: str) -> str:
    title = re.split(r"\s+[|–—]\s+", value, maxsplit=1)[0]
    return re.sub(r"\s+", " ", title).strip(" -–—|")


def _clean_title(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _normalize_person_name(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" -–—|,")


def _safe_summary(content: str, reasons: Iterable[str]) -> str:
    content = re.sub(r"\s+", " ", content).strip()
    reason_text = ", ".join(reasons)
    if content and reason_text:
        return f"{content} | qualification: {reason_text}"
    return content or f"qualification: {reason_text}"


def _confidence(score: float) -> SourceConfidence:
    if score >= 0.78:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def _role_category(title: str) -> RoleCategory:
    normalized = title.casefold()
    for keyword, category in _ROLE_KEYWORDS:
        if keyword in normalized:
            return category
    return "unknown"


def _company_domain(company: CompanyProspect | QualifiedCompanyCandidate) -> str:
    return _domain(company.website)


def _domain(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    return (parsed.netloc or parsed.path).casefold().removeprefix("www.")


def _token_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _slug(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    return tuple(deduped)


__all__ = [
    "QualificationAuditRecord",
    "QualificationStatus",
    "QualifiedCompanyBatch",
    "QualifiedCompanyCandidate",
    "QualifiedContactBatch",
    "QualifiedContactCandidate",
    "qualify_company_hits",
    "qualify_contact_hits",
]
