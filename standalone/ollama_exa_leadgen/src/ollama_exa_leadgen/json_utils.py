"""JSON extraction helpers for LLM/provider responses."""

from __future__ import annotations

import json
import re
from typing import Any

from .errors import ValidationError

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from raw text, markdown fence, or surrounding prose."""

    candidates = [text.strip()]
    candidates.extend(match.group(1).strip() for match in _FENCE_RE.finditer(text))
    brace = _extract_brace_object(text)
    if brace:
        candidates.append(brace)
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValidationError("could not parse JSON object from provider response")


def _extract_brace_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start : end + 1]
