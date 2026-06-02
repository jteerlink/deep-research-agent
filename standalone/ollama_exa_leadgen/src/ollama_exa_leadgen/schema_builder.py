"""Build and validate Exa structured output schemas."""

from __future__ import annotations

import re
from typing import Any

from .contracts import (
    CORE_COLUMNS,
    DEFAULT_OPTIONAL_COLUMNS,
    MAX_SCHEMA_ITEM_FIELDS,
    SUPPORTED_OPTIONAL_COLUMNS,
    IcpSpec,
)
from .errors import ValidationError

CORE_FIELD_DEFS: dict[str, dict[str, Any]] = {
    "company_name": {"type": "string", "description": "company name in 5 words or less"},
    "website": {"type": "string", "description": "homepage URL in 12 words or less"},
    "product_description": {
        "type": "string",
        "description": "what the company sells or offers in 12 words or less",
    },
    "icp_fit_score": {"type": "integer", "description": "ICP fit score from 1 to 10"},
    "icp_fit_reasoning": {
        "type": "string",
        "description": "specific sales-prioritization reason in 20 words or less",
    },
}
_LIMIT_RE = re.compile(
    r"\b(?:\d+\s+words?|\d+\s+characters?|less|under|max(?:imum)?)\b",
    re.I,
)


def selected_exa_fields(icp: IcpSpec) -> tuple[str, ...]:
    """Return schema item fields, preserving relevant preferred columns within limits."""

    fields: list[str] = list(CORE_COLUMNS)
    candidates = list(icp.preferred_columns or DEFAULT_OPTIONAL_COLUMNS)
    for candidate in DEFAULT_OPTIONAL_COLUMNS:
        if candidate not in candidates:
            candidates.append(candidate)
    for field in candidates:
        if field in fields:
            continue
        if field not in SUPPORTED_OPTIONAL_COLUMNS:
            continue
        if len(fields) >= MAX_SCHEMA_ITEM_FIELDS:
            break
        fields.append(field)
    return tuple(fields)


def build_output_schema(icp: IcpSpec) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for field in selected_exa_fields(icp):
        if field in CORE_FIELD_DEFS:
            properties[field] = dict(CORE_FIELD_DEFS[field])
        else:
            properties[field] = dict(SUPPORTED_OPTIONAL_COLUMNS[field])
    required = [field for field in CORE_COLUMNS if field in properties]
    schema = {
        "type": "object",
        "required": ["companies"],
        "properties": {
            "companies": {
                "type": "array",
                "description": (
                    f"Exactly {icp.target_count} companies matching the ICP; "
                    "exclude non-prospects"
                ),
                "items": {
                    "type": "object",
                    "required": required,
                    "properties": properties,
                },
            }
        },
    }
    validate_output_schema(schema)
    return schema


def validate_output_schema(schema: dict[str, Any]) -> None:
    if schema.get("type") != "object":
        raise ValidationError('outputSchema root must have type "object"')
    companies = (schema.get("properties") or {}).get("companies")
    if not isinstance(companies, dict) or companies.get("type") != "array":
        raise ValidationError('outputSchema must contain a companies array property')
    items = companies.get("items")
    if not isinstance(items, dict) or items.get("type") != "object":
        raise ValidationError("companies.items must be a flat object")
    properties = items.get("properties")
    if not isinstance(properties, dict) or not properties:
        raise ValidationError("companies.items.properties is required")
    if len(properties) > MAX_SCHEMA_ITEM_FIELDS:
        raise ValidationError(f"Exa item schema must have <= {MAX_SCHEMA_ITEM_FIELDS} fields")
    for name, spec in properties.items():
        if not isinstance(spec, dict):
            raise ValidationError(f"field {name} must be an object")
        field_type = spec.get("type")
        if field_type not in {"string", "integer", "boolean", "array"}:
            raise ValidationError(f"field {name} has unsupported type {field_type!r}")
        if field_type == "string" and not _has_length_limit(str(spec.get("description", ""))):
            raise ValidationError(f"string field {name} must include a word/character limit")
        if field_type == "array":
            item_type = (spec.get("items") or {}).get("type")
            if item_type != "string":
                raise ValidationError(f"array field {name} must contain strings only")


def _has_length_limit(description: str) -> bool:
    return bool(_LIMIT_RE.search(description))
