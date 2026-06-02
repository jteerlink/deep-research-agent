"""ICP planning and micro-vertical generation."""

from __future__ import annotations

import math
from typing import Any

from .contracts import IcpSpec, MicroVertical
from .errors import ValidationError
from .ollama_client import OllamaCloudClient


def required_exa_calls(target_count: int) -> int:
    return max(1, math.ceil(target_count / 35))


def deterministic_micro_verticals(
    icp: IcpSpec,
    *,
    count: int | None = None,
) -> tuple[MicroVertical, ...]:
    """Deterministic plan-mode micro-vertical preview from a written ICP."""

    needed = count or max(required_exa_calls(icp.target_count), 3)
    geography = ", ".join(icp.geographies) if icp.geographies else "target geography"
    signals = (
        ", ".join(icp.include_signals[:4])
        if icp.include_signals
        else "high-intent buying signals"
    )
    base = icp.icp_description
    specs = [
        (
            "Core ICP companies",
            f"{base} {geography} {signals}",
            [f"{base} {geography}", f"{signals} {geography}", f"best-fit companies {base}"],
            "Direct ICP match.",
        ),
        (
            "Growth signal companies",
            f"{base} growth expansion multiple locations {geography}",
            [f"{base} expansion {geography}", f"{base} multiple locations {geography}"],
            "Prioritizes expansion and multi-site growth signals.",
        ),
        (
            "Marketing opportunity companies",
            f"{base} reviews paid ads recall follow-up marketing opportunity {geography}",
            [f"{base} reviews {geography}", f"{base} paid ads marketing {geography}"],
            "Targets likely outreach pain points.",
        ),
        (
            "Competitor-adjacent alternatives",
            f"companies similar to high-fit ICP accounts {base} {geography}",
            [f"alternatives in {base} {geography}", f"similar companies {base} {geography}"],
            "Finds companies near known ICP clusters.",
        ),
    ]
    micro = [
        MicroVertical(
            name=name,
            objective=objective,
            additional_queries=tuple(queries),
            why_distinct=why,
        )
        for name, objective, queries, why in specs
    ]
    while len(micro) < needed:
        index = len(micro) + 1
        micro.append(
            MicroVertical(
                name=f"ICP variant {index}",
                objective=f"{base} {geography} {signals} variant {index}",
                additional_queries=(f"{base} {geography} prospect variant {index}",),
                why_distinct="Generated deterministic overshoot variant.",
            )
        )
    return tuple(micro[:needed])


def ollama_micro_verticals(
    icp: IcpSpec,
    client: OllamaCloudClient,
    *,
    count: int | None = None,
) -> tuple[MicroVertical, ...]:
    """Use Ollama Cloud to generate micro-verticals and validate response shape."""

    desired = count or max(required_exa_calls(icp.target_count), 3)
    response = client.json_object(
        task=(
            "Generate distinct Exa Deep Search micro-verticals for a lead-generation run. "
            f"Return exactly {desired} items in a micro_verticals array. Each item must include "
            "name, objective, additional_queries, and why_distinct."
        ),
        payload={"icp": icp.to_dict(), "desired_micro_vertical_count": desired},
    )
    raw_items = response.get("micro_verticals")
    if not isinstance(raw_items, list):
        raise ValidationError("Ollama micro-vertical response must contain micro_verticals list")
    parsed = tuple(MicroVertical.from_mapping(item) for item in raw_items if isinstance(item, dict))
    if not parsed:
        raise ValidationError("Ollama returned no valid micro-verticals")
    return parsed[:desired]


def planning_payload(icp: IcpSpec, *, max_exa_calls: int) -> dict[str, Any]:
    calls = required_exa_calls(icp.target_count)
    micro = deterministic_micro_verticals(icp, count=max(calls, 3))
    return {
        "icp": icp.to_dict(),
        "target_count": icp.target_count,
        "estimated_exa_calls": calls,
        "max_exa_calls": max_exa_calls,
        "within_cap": calls <= max_exa_calls,
        "micro_vertical_preview": [item.to_dict() for item in micro],
    }
