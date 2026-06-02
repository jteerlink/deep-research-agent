"""Serializable contracts for the standalone leadgen flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import ValidationError

DEFAULT_TARGET_COUNT = 10
MAX_SCHEMA_ITEM_FIELDS = 9
CORE_COLUMNS = (
    "company_name",
    "website",
    "product_description",
    "icp_fit_score",
    "icp_fit_reasoning",
)
DEFAULT_OPTIONAL_COLUMNS = (
    "headquarters_location",
    "recent_signal",
    "decision_maker_titles",
)
SUPPORTED_OPTIONAL_COLUMNS: dict[str, dict[str, Any]] = {
    "headquarters_location": {
        "type": "string",
        "description": "city, region, or country in 6 words or less",
    },
    "recent_signal": {
        "type": "string",
        "description": "growth, hiring, reviews, funding, or buying signal in 15 words or less",
    },
    "decision_maker_titles": {
        "type": "array",
        "description": "up to 3 likely buyer titles, each 5 words or less",
        "items": {"type": "string"},
    },
    "estimated_employee_count": {
        "type": "string",
        "description": "employee range in 5 words or less",
    },
    "funding_stage": {
        "type": "string",
        "description": "funding or ownership stage in 6 words or less",
    },
    "hiring_signals": {
        "type": "string",
        "description": "relevant hiring signal in 12 words or less, or None found",
    },
    "competitor_overlap": {
        "type": "string",
        "description": "known competitor or adjacent product in 10 words or less, or None found",
    },
    "key_technologies": {
        "type": "array",
        "description": "up to 4 technologies, each 3 words or less",
        "items": {"type": "string"},
    },
    "recent_news": {
        "type": "string",
        "description": "recent company news in 15 words or less, or None found",
    },
}


@dataclass(frozen=True)
class IcpSpec:
    """Operator-written ICP used to generate leads."""

    campaign_name: str
    icp_description: str
    buyer_offer: str = ""
    target_count: int = DEFAULT_TARGET_COUNT
    geographies: tuple[str, ...] = ()
    include_signals: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    preferred_columns: tuple[str, ...] = DEFAULT_OPTIONAL_COLUMNS

    @classmethod
    def from_mapping(
        cls,
        payload: dict[str, Any],
        *,
        target_count: int | None = None,
    ) -> IcpSpec:
        count = (
            target_count
            if target_count is not None
            else payload.get("target_count", DEFAULT_TARGET_COUNT)
        )
        try:
            parsed_count = int(count)
        except (TypeError, ValueError) as exc:
            raise ValidationError("target_count must be an integer") from exc
        spec = cls(
            campaign_name=str(payload.get("campaign_name") or "leadgen-campaign"),
            icp_description=str(payload.get("icp_description") or ""),
            buyer_offer=str(payload.get("buyer_offer") or ""),
            target_count=parsed_count,
            geographies=_text_tuple(payload.get("geographies") or ()),
            include_signals=_text_tuple(payload.get("include_signals") or ()),
            exclude=_text_tuple(payload.get("exclude") or ()),
            preferred_columns=_text_tuple(
                payload.get("preferred_columns") or DEFAULT_OPTIONAL_COLUMNS
            ),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        if not self.campaign_name.strip():
            raise ValidationError("campaign_name is required")
        if not self.icp_description.strip():
            raise ValidationError("icp_description is required")
        if self.target_count < 1:
            raise ValidationError("target_count must be >= 1")
        if self.target_count > 1000:
            raise ValidationError("target_count must be <= 1000")

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_name": self.campaign_name,
            "buyer_offer": self.buyer_offer,
            "target_count": self.target_count,
            "icp_description": self.icp_description,
            "geographies": list(self.geographies),
            "include_signals": list(self.include_signals),
            "exclude": list(self.exclude),
            "preferred_columns": list(self.preferred_columns),
        }


@dataclass(frozen=True)
class MicroVertical:
    """One Exa Deep Search lane derived from the ICP."""

    name: str
    objective: str
    additional_queries: tuple[str, ...] = ()
    why_distinct: str = ""

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> MicroVertical:
        item = cls(
            name=str(payload.get("name") or payload.get("objective") or "micro-vertical"),
            objective=str(payload.get("objective") or payload.get("name") or ""),
            additional_queries=_text_tuple(
                payload.get("additional_queries")
                or payload.get("additionalQueries")
                or ()
            ),
            why_distinct=str(payload.get("why_distinct") or payload.get("whyDistinct") or ""),
        )
        item.validate()
        return item

    def validate(self) -> None:
        if not self.objective.strip():
            raise ValidationError("micro-vertical objective is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "objective": self.objective,
            "additional_queries": list(self.additional_queries),
            "why_distinct": self.why_distinct,
        }


@dataclass
class RunSummary:
    """Compact run summary persisted in summary JSON/Markdown."""

    run_dir: str
    target_count: int
    exa_calls_planned: int
    exa_calls_succeeded: int = 0
    exa_calls_failed: int = 0
    raw_lead_count: int = 0
    deduped_lead_count: int = 0
    final_lead_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "target_count": self.target_count,
            "exa_calls_planned": self.exa_calls_planned,
            "exa_calls_succeeded": self.exa_calls_succeeded,
            "exa_calls_failed": self.exa_calls_failed,
            "raw_lead_count": self.raw_lead_count,
            "deduped_lead_count": self.deduped_lead_count,
            "final_lead_count": self.final_lead_count,
            "warnings": list(self.warnings),
        }


def _text_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    try:
        return tuple(str(item).strip() for item in value if str(item).strip())
    except TypeError:
        return ()
