"""Offline-testable company contact-channel extraction.

The tiered pre-enrichment workflow treats companies as the primary prospect and
company-owned phone/email/contact URLs as supporting contact points.  This module
keeps extraction deterministic and dependency-free; callers may inject fetched
HTML/text from a browser or HTTP adapter, but no network access happens here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Literal
from urllib.parse import urljoin, urlparse, urlunparse

from .tiered_models import CompanyProspect, ContactCandidate, SourceConfidence

ContactPointKind = Literal["company_email", "company_phone", "company_contact_page"]

_EMAIL_RE = re.compile(r"(?<![\w.+-])([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})(?![\w.+-])", re.I)
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[\s.\-]?)?(?:\(?\d{3}\)?[\s.\-]?)\d{3}[\s.\-]?\d{4}(?!\d)")
_CONTACT_PATH_TERMS = (
    "contact",
    "contact-us",
    "contactus",
    "schedule",
    "schedule-service",
    "appointment",
    "appointments",
    "book",
    "booking",
    "quote",
    "estimate",
    "location",
    "locations",
)
_GENERIC_PAGE_TERMS = (
    "about",
    "about-us",
    "areas-we-serve",
    "services",
    "service",
    "home-automation",
)
_PUBLIC_EMAIL_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "aol.com",
    "icloud.com",
}


@dataclass(frozen=True)
class CompanyContactSource:
    """Official-domain page/search content inspected for contact channels."""

    url: str
    title: str = ""
    content: str = ""
    evidence_id: str = ""
    source_confidence: SourceConfidence = "medium"


@dataclass(frozen=True)
class _Link:
    href: str
    text: str


class _ContactHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[_Link] = []
        self.text_parts: list[str] = []
        self._href_stack: list[str] = []
        self._link_text_stack: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = ""
        for key, value in attrs:
            if key.lower() == "href" and value:
                href = value
                break
        self._href_stack.append(href)
        self._link_text_stack.append([])

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or not self._href_stack:
            return
        href = self._href_stack.pop()
        text = " ".join(self._link_text_stack.pop()).strip()
        if href:
            self.links.append(_Link(href=href, text=text))

    def handle_data(self, data: str) -> None:
        cleaned = " ".join(data.split())
        if not cleaned:
            return
        self.text_parts.append(cleaned)
        if self._link_text_stack:
            self._link_text_stack[-1].append(cleaned)

    @property
    def text(self) -> str:
        return " ".join(self.text_parts)


def extract_company_contact_points(
    company: CompanyProspect,
    sources: Iterable[CompanyContactSource],
    *,
    max_contact_points: int = 2,
) -> tuple[ContactCandidate, ...]:
    """Return ranked official company contact points for ``company``.

    Only official-domain sources/links are eligible.  Phone and email channels
    outrank contact-page fallbacks; generic about/service pages are retained only
    as source evidence unless they contain an actionable contact CTA/channel.
    """

    if max_contact_points < 1:
        raise ValueError("max_contact_points must be >= 1")

    company_host = _host(company.website)
    candidates: list[dict[str, Any]] = []
    for ordinal, source in enumerate(sources, start=1):
        source_url = _canonical_url(source.url)
        source_host = _host(source_url)
        official_host = company_host or source_host
        if official_host and source_host and not _same_domain(source_host, official_host):
            continue
        parser = _parse(source.content)
        text = " ".join(part for part in (source.title, parser.text or source.content) if part)
        evidence_id = (
            source.evidence_id or f"ev_{_slug(company.company_id)}_contact_source_{ordinal:03d}"
        )
        evidence_ids = (evidence_id,)

        for email in _emails(source.content, parser.links):
            candidates.append(
                {
                    "kind": "company_email",
                    "value": email.casefold(),
                    "label": "Company email",
                    "email": email.casefold(),
                    "phone": None,
                    "url": f"mailto:{email.casefold()}",
                    "source_url": source_url,
                    "source_confidence": _channel_confidence(source, email=email),
                    "contact_confidence": 0.9,
                    "notes": _notes("Email found on official company source", text),
                    "evidence_ids": evidence_ids,
                    "priority": 0,
                }
            )

        for phone in _phones(source.content, parser.links):
            candidates.append(
                {
                    "kind": "company_phone",
                    "value": phone,
                    "label": "Company phone",
                    "email": None,
                    "phone": phone,
                    "url": f"tel:{phone}",
                    "source_url": source_url,
                    "source_confidence": source.source_confidence,
                    "contact_confidence": 0.88,
                    "notes": _notes("Phone found on official company source", text),
                    "evidence_ids": evidence_ids,
                    "priority": 1,
                }
            )

        for page_url in _contact_page_urls(source_url, parser.links, text, official_host):
            candidates.append(
                {
                    "kind": "company_contact_page",
                    "value": page_url,
                    "label": "Contact page",
                    "email": None,
                    "phone": None,
                    "url": page_url,
                    "source_url": source_url,
                    "source_confidence": source.source_confidence,
                    "contact_confidence": 0.72,
                    "notes": _notes("Official contact page or CTA found", text),
                    "evidence_ids": evidence_ids,
                    "priority": 2,
                }
            )

    ranked = _ranked_unique(candidates)
    if any(candidate["kind"] in {"company_email", "company_phone"} for candidate in ranked):
        ranked = [candidate for candidate in ranked if candidate["kind"] != "company_contact_page"]
    ranked = _diversify_contact_channels(ranked, max_contact_points=max_contact_points)
    contacts: list[ContactCandidate] = []
    for index, candidate in enumerate(ranked, start=1):
        contact_id = f"contact_{_slug(company.company_id)}_{index:03d}"
        contacts.append(
            ContactCandidate(
                contact_id=contact_id,
                company_id=company.company_id,
                name=str(candidate["label"]),
                evidence_ids=tuple(candidate["evidence_ids"]),
                email=candidate["email"],
                phone=candidate["phone"],
                contact_confidence=float(candidate["contact_confidence"]),
                notes=str(candidate["notes"]),
                contact_kind=candidate["kind"],
                label=str(candidate["label"]),
                url=str(candidate["url"]),
                contact_url=str(candidate["url"]),
                source_url=str(candidate["source_url"]),
                source_confidence=candidate["source_confidence"],
            )
        )
    return tuple(contacts)


def _parse(content: str) -> _ContactHTMLParser:
    parser = _ContactHTMLParser()
    try:
        parser.feed(content or "")
    except Exception:  # pragma: no cover - HTMLParser is tolerant, keep extraction safe.
        return _ContactHTMLParser()
    return parser


def _emails(content: str, links: Iterable[_Link]) -> tuple[str, ...]:
    found: list[str] = []
    for link in links:
        href = link.href.strip()
        if href.lower().startswith("mailto:"):
            found.append(href.split(":", 1)[1].split("?", 1)[0])
    found.extend(match.group(1) for match in _EMAIL_RE.finditer(content or ""))
    return _dedupe(email for email in found if _valid_email(email))


def _phones(content: str, links: Iterable[_Link]) -> tuple[str, ...]:
    found: list[str] = []
    for link in links:
        href = link.href.strip()
        if href.lower().startswith("tel:"):
            found.append(href.split(":", 1)[1])
    found.extend(match.group(0) for match in _PHONE_RE.finditer(content or ""))
    normalized = [_normalize_phone(phone) for phone in found]
    return _dedupe(phone for phone in normalized if phone)


def _contact_page_urls(
    source_url: str,
    links: Iterable[_Link],
    text: str,
    official_host: str,
) -> tuple[str, ...]:
    urls: list[str] = []
    if _actionable_page_url(source_url, text):
        urls.append(source_url)
    for link in links:
        absolute = _canonical_url(urljoin(source_url, link.href))
        host = _host(absolute)
        if official_host and host and not _same_domain(host, official_host):
            continue
        if _actionable_link(absolute, link.text):
            urls.append(absolute)
    return _dedupe(urls)


def _actionable_link(url: str, text: str) -> bool:
    return _actionable_path(urlparse(url).path)


def _actionable_page_url(url: str, text: str) -> bool:
    return _actionable_path(urlparse(url).path)


def _actionable_path(path: str) -> bool:
    normalized = path.casefold().replace("_", "-")
    if _generic_page_path(normalized):
        return False
    segments = [segment.strip("/") for segment in normalized.split("/") if segment.strip("/")]
    return any(segment in _CONTACT_PATH_TERMS for segment in segments)


def _generic_page_path(path: str) -> bool:
    segments = [segment.strip("/") for segment in path.split("/") if segment.strip("/")]
    return any(segment in _GENERIC_PAGE_TERMS for segment in segments)


def _ranked_unique(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda item: (int(item["priority"]), str(item["value"])))
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for candidate in ordered:
        key = (str(candidate["kind"]), str(candidate["value"]).casefold())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _diversify_contact_channels(
    candidates: list[dict[str, Any]],
    *,
    max_contact_points: int,
) -> list[dict[str, Any]]:
    if max_contact_points >= len(candidates):
        return candidates
    diversified: list[dict[str, Any]] = []
    used_indexes: set[int] = set()
    for kind in ("company_email", "company_phone", "company_contact_page"):
        for index, candidate in enumerate(candidates):
            if index in used_indexes or candidate["kind"] != kind:
                continue
            diversified.append(candidate)
            used_indexes.add(index)
            break
        if len(diversified) >= max_contact_points:
            return diversified
    for index, candidate in enumerate(candidates):
        if index in used_indexes:
            continue
        diversified.append(candidate)
        if len(diversified) >= max_contact_points:
            break
    return diversified


def _channel_confidence(source: CompanyContactSource, *, email: str) -> SourceConfidence:
    domain = email.rsplit("@", 1)[-1].casefold()
    if domain not in _PUBLIC_EMAIL_DOMAINS and source.source_confidence in {"medium", "high"}:
        return "high"
    return source.source_confidence


def _valid_email(email: str) -> bool:
    email = email.strip().strip(".,;:'\")<>")
    if not email or ".." in email:
        return False
    return bool(_EMAIL_RE.fullmatch(email))


def _normalize_phone(phone: str) -> str:
    digits = re.sub(r"\D+", "", phone)
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    if len(digits) == 10:
        return f"+1{digits}"
    return ""


def _canonical_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url if "://" in url or url.startswith("mailto:") else f"https://{url}")
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path or ""
    if path != "/":
        path = path.rstrip("/")
    return urlunparse((scheme, netloc, path, "", parsed.query, ""))


def _host(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = parsed.netloc.casefold()
    return host[4:] if host.startswith("www.") else host


def _same_domain(host: str, official_host: str) -> bool:
    host = host.casefold().removeprefix("www.")
    official_host = official_host.casefold().removeprefix("www.")
    return host == official_host or host.endswith(f".{official_host}")


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = value.strip()
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return tuple(output)


def _notes(prefix: str, text: str) -> str:
    excerpt = " ".join(text.split())
    if len(excerpt) > 160:
        excerpt = excerpt[:157].rstrip() + "..."
    return f"{prefix}." + (f" Evidence: {excerpt}" if excerpt else "")


def _slug(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_") or "id"


__all__ = ["CompanyContactSource", "extract_company_contact_points"]
