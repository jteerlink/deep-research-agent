"""Direct Exa Search API client for deep lead discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import LeadgenConfig
from .contracts import IcpSpec, MicroVertical
from .errors import ConfigError
from .net import post_json
from .schema_builder import build_output_schema

EXA_SEARCH_URL = "https://api.exa.ai/search"


@dataclass(frozen=True)
class ExaCallResult:
    call_id: str
    micro_vertical: MicroVertical
    request_payload: dict[str, Any]
    response_payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "micro_vertical": self.micro_vertical.to_dict(),
            "request_payload": self.request_payload,
            "response_payload": self.response_payload,
        }


class ExaDeepClient:
    def __init__(self, config: LeadgenConfig):
        self.config = config

    def require_available(self) -> None:
        if not self.config.exa_api_key:
            raise ConfigError("EXA_API_KEY is required for live Exa discovery")

    def search_micro_vertical(
        self,
        icp: IcpSpec,
        micro: MicroVertical,
        *,
        call_id: str,
    ) -> ExaCallResult:
        self.require_available()
        payload = build_exa_payload(icp, micro)
        response = post_json(
            EXA_SEARCH_URL,
            payload,
            headers={"x-api-key": self.config.exa_api_key},
            timeout=self.config.timeout_seconds,
            ca_bundle=self.config.ca_bundle,
        )
        return ExaCallResult(
            call_id=call_id,
            micro_vertical=micro,
            request_payload=_redacted_request(payload),
            response_payload=response,
        )


def build_exa_payload(icp: IcpSpec, micro: MicroVertical) -> dict[str, Any]:
    return {
        "query": micro.objective,
        "type": "deep",
        "numResults": 50,
        "systemPrompt": build_system_prompt(icp),
        "additionalQueries": list(micro.additional_queries)[:5],
        "outputSchema": build_output_schema(icp),
    }


def build_system_prompt(icp: IcpSpec) -> str:
    exclusions = (
        ", ".join(icp.exclude)
        if icp.exclude
        else "non-prospects, directories, listicles, vendors"
    )
    offer = f" for this offer: {icp.buyer_offer}" if icp.buyer_offer else ""
    signals = ", ".join(icp.include_signals) if icp.include_signals else "strong buying signals"
    return (
        f"List exactly {icp.target_count} companies in the final output JSON. "
        f"Find real companies matching this ICP{offer}: {icp.icp_description}. "
        f"Prioritize these signals: {signals}. "
        "Score each company from 1 to 10 on ICP fit. "
        "Use official company websites when possible. "
        f"Do NOT include: {exclusions}."
    )


def _redacted_request(payload: dict[str, Any]) -> dict[str, Any]:
    # Payload has no secret headers, but return a deep-ish copy for artifact safety.
    return dict(payload)
