"""Prospect candidate triage and structured judgment helpers."""

from __future__ import annotations

import html
import json
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal
from urllib.parse import urlparse

import httpx

TriageDecision = Literal["reject", "judge", "fetch_then_judge"]
QualificationStatus = Literal["qualified", "needs_review", "rejected"]

MIN_ACCEPTED_FIT_SCORE = 0.55
MIN_ACCEPTED_CONFIDENCE = 0.5

_TWO_PART_SUFFIX_PREFIXES = {"ac", "co", "com", "edu", "gov", "net", "org"}

_DOMAIN_CATEGORIES: dict[str, tuple[str, ...]] = {
    "directory": (
        "zoominfo.com",
        "yelp.com",
        "yellowpages.com",
        "angi.com",
        "angieslist.com",
        "homeadvisor.com",
        "thumbtack.com",
        "houzz.com",
        "bbb.org",
        "mapquest.com",
        "manta.com",
        "chamberofcommerce.com",
        "buildzoom.com",
        "porch.com",
        "bark.com",
        "nextdoor.com",
        "merchantcircle.com",
        "superpages.com",
        "dandb.com",
        "datanyze.com",
        "apollo.io",
        "rocketreach.co",
        "crunchbase.com",
    ),
    "social": (
        "facebook.com",
        "instagram.com",
        "linkedin.com",
        "x.com",
        "twitter.com",
        "tiktok.com",
        "youtube.com",
    ),
    "job_board": (
        "indeed.com",
        "glassdoor.com",
        "ziprecruiter.com",
        "monster.com",
        "simplyhired.com",
        "dentalpost.net",
    ),
    "health_directory": (
        "zocdoc.com",
        "healthgrades.com",
        "opencare.com",
        "ada.org",
        "dcds.org",
        "vitals.com",
        "webmd.com",
    ),
    "reference": ("wikipedia.org", "wikidata.org"),
    "search": ("google.com", "bing.com", "duckduckgo.com"),
    "association": (
        "acca.org",
        "phcc-tx.org",
        "tacca.org",
    ),
    "education": (
        "edu",
        "techzonehvacr.com",
    ),
    "government": ("gov",),
    "manufacturer": (
        "amana-hac.com",
        "carrier.com",
        "daikincomfort.com",
        "goodmanmfg.com",
        "lennox.com",
        "northamerica-daikin.com",
        "nortekair.com",
        "nortekhvac.com",
        "rheem.com",
        "samsunghvac.com",
        "trane.com",
        "york.com",
    ),
    "national_brand": (
        "aireserv.com",
        "ars.com",
        "onehourheatandair.com",
        "serviceexperts.com",
    ),
}

_REJECTED_CATEGORIES = {
    "association",
    "directory",
    "education",
    "government",
    "social",
    "job_board",
    "health_directory",
    "manufacturer",
    "national_brand",
    "reference",
    "search",
}

_ARTICLE_PATH_TOKENS = (
    "/blog",
    "/news",
    "/article",
    "/articles",
    "/guide",
    "/guides",
    "/resources",
    "/insights",
)
_JOB_PATH_TOKENS = ("/jobs", "/careers", "/career", "/hiring", "/employment")
_OWNED_PATH_TOKENS = (
    "/about",
    "/contact",
    "/services",
    "/service",
    "/locations",
    "/location",
    "/team",
)
_PROFILE_PATH_TOKENS = ("/profile", "/profiles", "/company/", "/companies/", "/biz/", "/c/")
_REVIEW_PATH_TOKENS = ("/reviews", "/review", "/ratings", "/rating")

_LISTICLE_TITLE_PATTERNS = (
    r"\bbest\s+\d*\s*",
    r"\btop\s+\d*\s*",
    r"\btop-rated\b",
    r"\bnear me\b",
    r"\bdirectory\b",
    r"\breviews?\b",
    r"\bratings?\b",
)

_GENERIC_TITLE_STARTS = (
    "contact",
    "contact us",
    "about",
    "about us",
    "home",
    "homepage",
    "official website",
    "top-rated",
    "top rated",
    "best ",
    "local ",
    "emergency ",
    "commercial ",
    "residential ",
)

_VENDOR_NOISE_PATTERNS = (
    "lead finder",
    "leadfinder",
    "lead generation software",
    "marketing automation",
    "crm software",
    "sales automation",
    "prospecting software",
    "business directory",
    "find a contractor",
    "find contractors",
)

_ENTITY_NOISE_PATTERNS: dict[str, tuple[str, ...]] = {
    "association": (
        "association",
        "professional trade association",
        "not for profit professional",
        "industry leadership",
        "advocacy",
        "chapter",
    ),
    "education": (
        "school",
        "college",
        "campus",
        "training program",
        "technician training",
        "become a certified",
        "classes",
        "tuition",
    ),
    "government": (
        "department of licensing",
        "verify a license",
        "renew a license",
        "apply for a license",
        "advisory board",
        "state agency",
    ),
    "manufacturer": (
        "manufacturing",
        "manufacturer",
        "heating and cooling products",
        "ductless and vrf systems",
        "our brands",
        "professional portal",
        "dealer locator",
    ),
    "national_brand": (
        "find a location",
        "franchise",
        "national franchise",
        "national account",
        "corporate office",
    ),
}

_ENTITY_NOISE_CATEGORIES = {
    "association",
    "education",
    "government",
    "manufacturer",
    "national_brand",
}

_REQUESTED_ENTITY_TERMS: dict[str, tuple[str, ...]] = {
    "association": ("association", "associations", "chapter", "chapters"),
    "education": (
        "college",
        "colleges",
        "school",
        "schools",
        "training",
        "training program",
        "trade school",
    ),
    "government": ("government", "agency", "agencies", "regulator", "public sector"),
    "manufacturer": (
        "manufacturer",
        "manufacturers",
        "manufacturing",
        "distributor",
        "distributors",
    ),
    "national_brand": (
        "chain",
        "chains",
        "franchise",
        "franchises",
        "multi location",
        "national brand",
        "national brands",
    ),
}

_REVIEW_ONLY_MODEL_STATUSES = {
    "metadata_only",
    "model_unavailable",
    "no_available_model",
    "model_budget_exhausted",
}
_REVIEW_ONLY_JUDGMENT_MODES = {"deterministic_fallback"}

_OWNED_SNIPPET_SIGNALS = (
    "we provide",
    "we offer",
    "our team",
    "our services",
    "serving",
    "family owned",
    "locally owned",
    "call us",
    "schedule",
    "request service",
    "residential and commercial",
)


@dataclass(frozen=True)
class ProspectCandidate:
    """Normalized search candidate passed into triage and judgment."""

    evidence_id: str
    title: str
    url: str
    source_url: str
    canonical_website: str
    domain: str
    root_domain: str
    organization_guess: str
    snippet: str = ""
    provider: str = ""
    source_type: str = "snippet"
    search_query: str = ""

    @classmethod
    def from_evidence(cls, record: Mapping[str, Any]) -> ProspectCandidate:
        url = str(record.get("url") or "")
        domain = domain_from_url(url)
        root_domain = registrable_domain(domain)
        title = " ".join(str(record.get("title") or "").split())
        return cls(
            evidence_id=str(record.get("id") or ""),
            title=title,
            url=url,
            source_url=url,
            canonical_website=canonical_website(url),
            domain=domain,
            root_domain=root_domain,
            organization_guess=business_name_from_title(title, root_domain),
            snippet=str(record.get("snippet") or ""),
            provider=str(record.get("provider") or ""),
            source_type=str(record.get("source_type") or "snippet"),
            search_query=str(record.get("query") or ""),
        )

    def to_prompt_dict(self, page_text: str = "") -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "canonical_website": self.canonical_website,
            "domain": self.domain,
            "root_domain": self.root_domain,
            "organization_guess": self.organization_guess,
            "snippet": self.snippet[:1200],
            "page_text": page_text[:2400],
            "provider": self.provider,
            "search_query": self.search_query,
        }


@dataclass(frozen=True)
class CandidateTriage:
    """Deterministic routing decision before model judgment."""

    decision: TriageDecision
    source_category: str
    page_intent: str
    reason: str
    flags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["flags"] = list(self.flags)
        return payload


@dataclass(frozen=True)
class ProspectJudgment:
    """Structured prospect-fit decision from an LLM or deterministic fallback."""

    accepted: bool
    organization: str
    canonical_website: str
    fit_score: float
    confidence: float
    reject_reason: str = ""
    fit_rationale: str = ""
    evidence_summary: str = ""
    personalized_angles: tuple[str, ...] = ()
    decision_maker_leads: tuple[str, ...] = ()
    guardrail_flags: tuple[str, ...] = ()
    mode: str = "llm"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["personalized_angles"] = list(self.personalized_angles)
        payload["decision_maker_leads"] = list(self.decision_maker_leads)
        payload["guardrail_flags"] = list(self.guardrail_flags)
        return payload


@dataclass(frozen=True)
class ProspectQualification:
    """Export and sufficiency eligibility for a reviewed prospect candidate."""

    qualification_status: QualificationStatus
    export_qualified: bool
    sufficiency_qualified: bool
    review_only: bool
    review_only_reason: str = ""
    qualification_reasons: tuple[str, ...] = ()
    qualification_warnings: tuple[str, ...] = ()
    fallback_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["qualification_reasons"] = list(self.qualification_reasons)
        payload["qualification_warnings"] = list(self.qualification_warnings)
        payload["fallback_metadata"] = dict(self.fallback_metadata or {})
        return payload


@dataclass(frozen=True)
class CandidateReview:
    """Auditable result for one candidate after triage and judgment."""

    candidate: ProspectCandidate
    triage: CandidateTriage
    judgment: ProspectJudgment | None = None
    page_evidence_id: str = ""
    page_text: str = ""
    error: str = ""
    qualification: ProspectQualification | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": asdict(self.candidate),
            "triage": self.triage.to_dict(),
            "judgment": self.judgment.to_dict() if self.judgment else None,
            "page_evidence_id": self.page_evidence_id,
            "error": self.error,
            "qualification": self.qualification.to_dict() if self.qualification else None,
        }


def domain_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if "@" in host:
        host = host.rsplit("@", 1)[-1]
    host = host.split(":", 1)[0]
    return host[4:] if host.startswith("www.") else host


def registrable_domain(domain: str) -> str:
    parts = [part for part in domain.lower().split(".") if part]
    if len(parts) <= 2:
        return ".".join(parts)
    if len(parts[-1]) == 2 and parts[-2] in _TWO_PART_SUFFIX_PREFIXES:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def canonical_website(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return ""
    root = registrable_domain(domain_from_url(url))
    if not root:
        return ""
    return f"{parsed.scheme}://{root}"


def business_name_from_title(title: str, domain: str) -> str:
    domain_name = ""
    if domain:
        first_label = registrable_domain(domain).split(".")[0]
        domain_name = _readable_business_name(first_label)

    normalized = re.sub(r"\s+", " ", html.unescape(title)).strip()
    if normalized:
        parts = [normalized]
        for separator in (" | ", " - ", " — ", " – ", ": "):
            if separator in normalized:
                parts = [part.strip() for part in normalized.split(separator) if part.strip()]
                break
        brand_parts = []
        for part in parts:
            if _looks_generic_title(part) or _looks_geography_only(part):
                continue
            readable = _readable_business_name(part)
            if _prefer_domain_name_for_title(part, readable, domain_name):
                readable = domain_name
            brand_parts.append(readable)
        brand_parts = [part for part in brand_parts if part]
        if brand_parts:
            return min(brand_parts, key=len).strip(" -|:")
    if domain_name:
        return domain_name
    return "Unknown prospect"


def triage_candidate(
    candidate: ProspectCandidate,
    *,
    directive: Mapping[str, str] | None = None,
) -> CandidateTriage:
    if not candidate.url.startswith(("http://", "https://")) or not candidate.root_domain:
        return CandidateTriage("reject", "invalid", "invalid_url", "URL is not an absolute web URL")

    source_category = source_category_for_domain(candidate.root_domain)
    page_intent = page_intent_for_candidate(candidate)
    flags = _candidate_flags(candidate, source_category, page_intent)
    requested_categories = _requested_entity_categories(directive)

    if source_category in _REJECTED_CATEGORIES and source_category not in requested_categories:
        return CandidateTriage(
            "reject",
            source_category,
            page_intent,
            f"{source_category} sources are not owned business websites",
            flags,
        )
    if page_intent in {"article", "job", "listicle", "review", "third_party_profile"}:
        return CandidateTriage(
            "reject",
            source_category,
            page_intent,
            f"{page_intent} pages are not direct prospect accounts",
            flags,
        )
    if "vendor_noise" in flags:
        return CandidateTriage(
            "reject",
            source_category,
            page_intent,
            "Result appears to sell prospecting or marketing software instead of being a target",
            flags,
        )
    entity_noise = sorted(
        flag
        for flag in flags
        if flag in _ENTITY_NOISE_CATEGORIES and flag not in requested_categories
    )
    if entity_noise:
        if source_category == "owned_or_unknown" and "owned_business_signal" in flags:
            return CandidateTriage(
                "fetch_then_judge",
                source_category,
                page_intent,
                f"{entity_noise[0]} signals on an owned domain require review",
                flags,
            )
        return CandidateTriage(
            "reject",
            source_category,
            page_intent,
            f"{entity_noise[0]} sources are not local operating prospect accounts",
            flags,
        )
    if "generic_title" in flags or page_intent in {"contact", "about", "service"}:
        return CandidateTriage(
            "fetch_then_judge",
            source_category,
            page_intent,
            "Owned-domain signals need more evidence before judgment",
            flags,
        )
    return CandidateTriage(
        "judge",
        source_category,
        page_intent,
        "Plausible owned business candidate",
        flags,
    )


def source_category_for_domain(root_domain: str) -> str:
    for category, domains in _DOMAIN_CATEGORIES.items():
        if any(root_domain == domain or root_domain.endswith(f".{domain}") for domain in domains):
            return category
    return "owned_or_unknown"


def _requested_entity_categories(directive: Mapping[str, str] | None) -> set[str]:
    if not directive:
        return set()
    haystack = " ".join(str(value) for value in directive.values()).lower()
    haystack = re.sub(r"[^a-z0-9]+", " ", haystack)
    requested: set[str] = set()
    for category, terms in _REQUESTED_ENTITY_TERMS.items():
        for term in terms:
            normalized = re.sub(r"[^a-z0-9]+", " ", term.lower()).strip()
            if re.search(rf"\b{re.escape(normalized)}\b", haystack):
                requested.add(category)
                break
    return requested


def page_intent_for_candidate(candidate: ProspectCandidate) -> str:
    parsed = urlparse(candidate.url)
    path = parsed.path.lower()
    title = candidate.title.lower()
    title_url = f"{title} {candidate.url}".lower()
    if path in {"", "/"}:
        return "homepage"
    if any(token in path for token in _JOB_PATH_TOKENS) or "jobs" in title:
        return "job"
    if any(token in path for token in _ARTICLE_PATH_TOKENS) or any(
        word in title for word in (" guide", " article", " blog ")
    ):
        return "article"
    if any(re.search(pattern, title_url) for pattern in _LISTICLE_TITLE_PATTERNS):
        return "listicle"
    if any(token in path for token in _REVIEW_PATH_TOKENS):
        return "review"
    if any(token in path for token in _PROFILE_PATH_TOKENS):
        return "third_party_profile"
    if "/contact" in path:
        return "contact"
    if "/about" in path:
        return "about"
    if any(token in path for token in _OWNED_PATH_TOKENS):
        return "service"
    return "unknown"


def prospect_judgment_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": [
            "accepted",
            "organization",
            "canonical_website",
            "fit_score",
            "confidence",
            "reject_reason",
            "fit_rationale",
            "evidence_summary",
            "guardrail_flags",
        ],
        "properties": {
            "accepted": {"type": "boolean"},
            "organization": {"type": "string"},
            "canonical_website": {"type": "string"},
            "fit_score": {"type": "number", "minimum": 0, "maximum": 1},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reject_reason": {"type": "string"},
            "fit_rationale": {"type": "string"},
            "evidence_summary": {"type": "string"},
            "personalized_angles": {"type": "array", "items": {"type": "string"}},
            "decision_maker_leads": {"type": "array", "items": {"type": "string"}},
            "guardrail_flags": {"type": "array", "items": {"type": "string"}},
        },
    }


def build_prospect_judge_prompt(
    *,
    directive: Mapping[str, str],
    geography_scope: Mapping[str, Any] | None = None,
    candidate: ProspectCandidate,
    triage: CandidateTriage,
    page_text: str,
) -> str:
    schema = prospect_judgment_schema()
    payload = {
        "directive": dict(directive),
        "geography_scope": dict(geography_scope or {}),
        "candidate": candidate.to_prompt_dict(page_text=page_text),
        "deterministic_triage": triage.to_dict(),
        "task": "Evaluate whether this search result is an export-qualified prospect account.",
        "required_output_schema": schema,
        "required_output_keys": [
            "accepted",
            "organization",
            "canonical_website",
            "fit_score",
            "confidence",
            "reject_reason",
            "fit_rationale",
            "evidence_summary",
            "personalized_angles",
            "decision_maker_leads",
            "guardrail_flags",
        ],
        "rules": [
            "Accept only real operating businesses that match the requested industry or niche.",
            (
                "Accept only candidates that serve or are located in the requested geography "
                "when geography is provided."
            ),
            (
                "Reject directories, aggregators, third-party profiles, articles, reviews, "
                "job pages, and listicles."
            ),
            (
                "Reject vendors selling lead generation, CRM, or marketing automation unless "
                "that exact vendor category is the requested niche."
            ),
            "Do not invent facts. Use only title, URL, snippet, and page_text evidence.",
            "Use the exact key 'accepted'; never use 'accept'.",
            "Use the exact key 'reject_reason'; never use 'reason'.",
            (
                "For a real owned operating business matching industry and geography, set "
                "accepted=true with fit_score from 0.65 to 0.95 and confidence from 0.55 to 0.95."
            ),
            "If accepted=true, reject_reason must be empty and fit_rationale must explain the fit.",
            (
                "Reject only when evidence shows the candidate is not a direct "
                "matching prospect account."
            ),
            "Return only JSON with the requested fields.",
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def coerce_prospect_judgment(
    value: Mapping[str, Any],
    *,
    candidate: ProspectCandidate,
    triage: CandidateTriage,
    mode: str = "llm",
) -> ProspectJudgment:
    flags = _string_tuple(value.get("guardrail_flags", []))
    accepted_value = _coerce_bool(value.get("accepted"))
    used_accept_alias = False
    if accepted_value is None and "accept" in value:
        accepted_value = _coerce_bool(value.get("accept"))
        used_accept_alias = True
        flags = (*flags, "model_output_accept_alias")
    elif accepted_value is None:
        flags = (*flags, "model_output_missing_accepted")
    accepted = bool(accepted_value)

    fit_score_missing = _model_field_missing(value, "fit_score")
    confidence_missing = _model_field_missing(value, "confidence")
    fit_score = _bounded_float(value.get("fit_score"), 0.0)
    confidence = _bounded_float(value.get("confidence"), 0.0)
    if accepted and fit_score_missing:
        fit_score = 0.72
        flags = (*flags, "model_output_missing_fit_score_defaulted")
    if accepted and confidence_missing:
        confidence = 0.62
        flags = (*flags, "model_output_missing_confidence_defaulted")
    if accepted and (fit_score < MIN_ACCEPTED_FIT_SCORE or confidence < MIN_ACCEPTED_CONFIDENCE):
        accepted = False
        flags = (*flags, "below_acceptance_threshold")
    if triage.decision == "reject":
        accepted = False
        flags = (*flags, "deterministic_reject")
    organization = str(value.get("organization") or candidate.organization_guess).strip()
    website = str(value.get("canonical_website") or candidate.canonical_website).strip()
    reason_alias = str(value.get("reason") or "").strip()
    reject_reason = str(value.get("reject_reason") or "").strip()
    fit_rationale = str(value.get("fit_rationale") or "").strip()
    if reason_alias and not reject_reason and not accepted:
        reject_reason = reason_alias
        flags = (*flags, "model_output_reason_alias")
    if reason_alias and not fit_rationale and accepted:
        fit_rationale = reason_alias
        flags = (*flags, "model_output_reason_alias")
    if used_accept_alias and accepted and not fit_rationale:
        fit_rationale = "Model accepted the owned business prospect using alias fields."
    return ProspectJudgment(
        accepted=accepted,
        organization=organization or candidate.organization_guess,
        canonical_website=website or candidate.canonical_website,
        fit_score=fit_score,
        confidence=confidence,
        reject_reason=reject_reason,
        fit_rationale=fit_rationale,
        evidence_summary=str(value.get("evidence_summary") or "").strip(),
        personalized_angles=_string_tuple(value.get("personalized_angles", [])),
        decision_maker_leads=_string_tuple(value.get("decision_maker_leads", [])),
        guardrail_flags=tuple(dict.fromkeys(flags)),
        mode=mode,
    )


def deterministic_judgment(
    candidate: ProspectCandidate,
    triage: CandidateTriage,
    *,
    page_text: str = "",
    mode: str = "deterministic_fallback",
) -> ProspectJudgment:
    owned_signal = _has_owned_signal(candidate, page_text)
    accepted = triage.decision != "reject" and owned_signal
    score = 0.62 if accepted else 0.2
    confidence = 0.58 if accepted else 0.3
    reason = "" if accepted else triage.reason
    summary_source = page_text or candidate.snippet
    summary = _clean_summary(summary_source) or candidate.title
    return ProspectJudgment(
        accepted=accepted,
        organization=candidate.organization_guess,
        canonical_website=candidate.canonical_website,
        fit_score=score,
        confidence=confidence,
        reject_reason=reason,
        fit_rationale=(
            "Owned-domain prospect candidate surfaced by broad deterministic triage; "
            "confirm fit with page-read evidence before outreach."
            if accepted
            else ""
        ),
        evidence_summary=summary,
        personalized_angles=(summary[:180],) if summary else (),
        decision_maker_leads=("Owner/Founder", "General Manager"),
        guardrail_flags=triage.flags,
        mode=mode,
    )


async def fetch_page_text(url: str, *, timeout: float = 8.0) -> str:
    if not url.startswith(("http://", "https://")):
        return ""
    verify = _httpx_verify_value()
    headers = {"User-Agent": "deep-research-agent/0.1 prospect-evidence-fetch"}
    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=True, verify=verify, headers=headers
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
    return html_to_text(response.text)


def html_to_text(value: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", value)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()[:5000]


def is_accepted_prospect(judgment: ProspectJudgment) -> bool:
    return (
        judgment.accepted
        and judgment.fit_score >= MIN_ACCEPTED_FIT_SCORE
        and judgment.confidence >= MIN_ACCEPTED_CONFIDENCE
        and bool(judgment.organization)
        and bool(judgment.canonical_website)
    )


def prospect_qualification(
    judgment: ProspectJudgment,
    triage: CandidateTriage,
    *,
    model_status: str = "",
) -> ProspectQualification:
    """Return explicit export/sufficiency status for a candidate judgment."""

    fallback_metadata = {
        "judgment_mode": judgment.mode,
        "model_judgment_status": model_status,
    }
    reasons: list[str] = []
    warnings: list[str] = []
    threshold_accepted = is_accepted_prospect(judgment)
    if triage.decision == "reject":
        reasons.append(triage.reason)
        return ProspectQualification(
            "rejected",
            export_qualified=False,
            sufficiency_qualified=False,
            review_only=False,
            qualification_reasons=tuple(reasons),
            fallback_metadata=fallback_metadata,
        )
    if not threshold_accepted:
        reasons.append(judgment.reject_reason or "Candidate did not meet prospect thresholds")
        return ProspectQualification(
            "rejected",
            export_qualified=False,
            sufficiency_qualified=False,
            review_only=False,
            qualification_reasons=tuple(reasons),
            fallback_metadata=fallback_metadata,
        )
    if model_status in _REVIEW_ONLY_MODEL_STATUSES:
        reason = f"model_judgment_{model_status}"
        warnings.append(reason)
        return ProspectQualification(
            "needs_review",
            export_qualified=False,
            sufficiency_qualified=False,
            review_only=True,
            review_only_reason=reason,
            qualification_reasons=(reason,),
            qualification_warnings=tuple(warnings),
            fallback_metadata=fallback_metadata,
        )
    if judgment.mode in _REVIEW_ONLY_JUDGMENT_MODES:
        reason = "deterministic_fallback_requires_review"
        warnings.append(reason)
        return ProspectQualification(
            "needs_review",
            export_qualified=False,
            sufficiency_qualified=False,
            review_only=True,
            review_only_reason=reason,
            qualification_reasons=(reason,),
            qualification_warnings=tuple(warnings),
            fallback_metadata=fallback_metadata,
        )
    return ProspectQualification(
        "qualified",
        export_qualified=True,
        sufficiency_qualified=True,
        review_only=False,
        qualification_reasons=("accepted_by_structured_judgment",),
        fallback_metadata=fallback_metadata,
    )


def _candidate_flags(
    candidate: ProspectCandidate, source_category: str, page_intent: str
) -> tuple[str, ...]:
    text = f"{candidate.title} {candidate.url} {candidate.snippet}".lower()
    flags: list[str] = []
    if source_category != "owned_or_unknown":
        flags.append(source_category)
    if page_intent != "unknown":
        flags.append(page_intent)
    if _looks_generic_title(candidate.title):
        flags.append("generic_title")
    if any(pattern in text for pattern in _VENDOR_NOISE_PATTERNS):
        flags.append("vendor_noise")
    for flag, patterns in _ENTITY_NOISE_PATTERNS.items():
        if any(pattern in text for pattern in patterns):
            flags.append(flag)
    if _has_owned_signal(candidate):
        flags.append("owned_business_signal")
    return tuple(dict.fromkeys(flags))


def _has_owned_signal(candidate: ProspectCandidate, page_text: str = "") -> bool:
    text = f"{candidate.title} {candidate.url} {candidate.snippet} {page_text}".lower()
    if (
        candidate.root_domain
        and source_category_for_domain(candidate.root_domain) == "owned_or_unknown"
    ):
        if (
            candidate.organization_guess
            and candidate.organization_guess != "Unknown prospect"
            and not _looks_geography_only(candidate.organization_guess)
        ):
            return True
    return any(signal in text for signal in _OWNED_SNIPPET_SIGNALS)


def _looks_generic_title(title: str) -> bool:
    value = re.sub(r"\s+", " ", title).strip().lower()
    value = re.sub(r"[\u00ae\u2122\u00a9]", "", value).strip()
    value = value.replace("a/c", "ac")
    if not value:
        return True
    if _looks_geography_only(value):
        return True
    generic_exact = {
        "home",
        "homepage",
        "contact",
        "contact us",
        "about",
        "about us",
        "hvac",
        "24/7 emergency",
        "24 7 emergency",
        "repair & installation",
        "repair and installation",
        "geothermal installation",
        "trane",
        "carrier",
        "lennox",
        "goodman",
        "daikin",
        "rheem",
    }
    if value in generic_exact:
        return True
    if any(value.startswith(prefix) for prefix in _GENERIC_TITLE_STARTS):
        return True
    if any(re.search(pattern, value) for pattern in _LISTICLE_TITLE_PATTERNS):
        return True
    service_terms = (
        "hvac",
        "ac",
        "a c",
        "air conditioning",
        "heating",
        "heater",
        "furnace",
        "plumbing",
        "roofing",
        "dental",
        "dentist",
        "med spa",
        "refrigeration",
        "geothermal",
    )
    if not any(term in value for term in service_terms):
        return False
    generic_descriptors = (
        "company",
        "contractor",
        "installation",
        "repair",
        "repairs",
        "service",
        "services",
        "supply",
        "pros",
    )
    if any(re.search(rf"\b{descriptor}\b", value) for descriptor in generic_descriptors):
        return True
    return _starts_with_service_and_mentions_geography(value)


def _looks_geography_only(value: str) -> bool:
    normalized = re.sub(r"\s+", " ", value).strip().lower()
    normalized = normalized.strip(" -|:")
    normalized = re.sub(r"[-_/]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    if re.fullmatch(r"[a-z .'-]+,\s*(tx|texas|ca|fl|ga|az|wa|mn|ma|pa|nc|dc|md|va)", normalized):
        return True
    return normalized in {
        "north texas",
        "dfw",
        "united states",
        "dallas",
        "fort worth",
        "dallas fort worth",
        "fort worth tx",
        "dallas tx",
        "plano tx",
        "frisco tx",
        "arlington tx",
    }


def _readable_business_name(value: str) -> str:
    cleaned = re.sub(r"^#+\s*", "", html.unescape(value)).strip(" -|:")
    cleaned = re.sub(r"(?i)^at\s+", "", cleaned)
    cleaned = re.split(r"(?i)\bwe\s+(?:are|provide|offer)\b", cleaned, maxsplit=1)[0]
    cleaned = cleaned.strip(" -|:")
    if re.search(r"[\s&]", cleaned):
        return _title_preserving_acronyms(cleaned)

    compact = re.sub(r"[^A-Za-z0-9]+", " ", cleaned).strip().lower()
    compact = re.sub(r"\bnorthtexas\b", "north texas", compact)
    compact = re.sub(r"\bnorthtx\b", "north texas", compact)
    compact = re.sub(r"northtexas", "north texas ", compact)
    compact = re.sub(r"northtx", "north texas ", compact)
    compact = re.sub(r"\btx(?=[a-z])", "tx ", compact)
    compact = re.sub(r"(?<=[a-z])tx\b", " tx", compact)
    compact = re.sub(r"dfw", " dfw ", compact)
    compact = re.sub(r"hvacr", " hvacr ", compact)
    compact = re.sub(r"hvac", " hvac ", compact)
    compact = re.sub(r"(?<=[a-z])air(?=\s+conditioning\b)", " air", compact)
    compact = re.sub(r"(?<=[a-z])express(?=\s|$)", " express", compact)
    compact = re.sub(r"(?<=[a-z])pro(?=(?:dfw|dallas|tx|$))", " pro ", compact)
    for token in (
        "brothers",
        "conditioning",
        "express",
        "geothermal",
        "heat",
        "mechanical",
        "mechanics",
        "cooling",
        "heating",
        "plumbing",
        "company",
        "air",
    ):
        compact = re.sub(rf"(?<=\w){token}(?=\w|$)", f" {token} ", compact)
    compact = re.sub(r"(?<=[a-z])air(?=\s+conditioning\b)", " air", compact)
    compact = re.sub(r"(?<=\w)comfort(?=hvac|$)", " comfort ", compact)
    compact = re.sub(r"(?<=\w)services?$", " service", compact)
    compact = re.sub(r"\s+", " ", compact).strip()
    return _title_preserving_acronyms(compact or cleaned)


def _prefer_domain_name_for_title(raw_title: str, readable_title: str, domain_name: str) -> bool:
    if not domain_name or not readable_title:
        return False
    title_key = _compact_name_key(raw_title)
    domain_key = _compact_name_key(domain_name)
    readable_key = _compact_name_key(readable_title)
    if not title_key or not domain_key:
        return False
    if title_key == domain_key:
        return True
    if _looks_generic_title(raw_title) and domain_key in title_key:
        return True
    if readable_key and title_key == readable_key and title_key in domain_key:
        return True
    return False


def _compact_name_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _starts_with_service_and_mentions_geography(value: str) -> bool:
    starts_with_service = re.match(
        r"^(?:ac|air conditioning|heating|heater|furnace|hvac|plumbing)\b", value
    )
    if not starts_with_service:
        return False
    return bool(
        re.search(
            r"\b(?:dallas|fort worth|dfw|texas|tx|county|haltom city|plano|frisco|arlington)\b",
            value,
        )
    )


def _title_preserving_acronyms(value: str) -> str:
    acronyms = {"ac", "dfw", "hts", "hvac", "hvacr", "mfg", "tx", "vrf"}
    words = re.split(r"(\s+)", value)
    titled: list[str] = []
    for word in words:
        lower = word.lower()
        if lower in acronyms:
            titled.append(lower.upper())
        else:
            titled.append(word[:1].upper() + word[1:].lower())
    return "".join(titled).strip()


def _bounded_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, number))


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1"}:
            return True
        if normalized in {"false", "no", "n", "0"}:
            return False
    return None


def _model_field_missing(value: Mapping[str, Any], key: str) -> bool:
    return key not in value or value.get(key) in {None, ""}


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        items = (value,)
    elif isinstance(value, Mapping):
        items = tuple(value.values())
    else:
        try:
            items = tuple(value)
        except TypeError:
            items = (value,)
    return tuple(str(item).strip() for item in items if str(item).strip())


def _clean_summary(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _httpx_verify_value() -> bool | str:
    for key in ("DEEP_RESEARCH_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return True
