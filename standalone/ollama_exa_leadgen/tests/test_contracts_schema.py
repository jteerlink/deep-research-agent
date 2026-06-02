from __future__ import annotations

import pytest
from ollama_exa_leadgen.contracts import IcpSpec
from ollama_exa_leadgen.errors import ValidationError
from ollama_exa_leadgen.schema_builder import build_output_schema, validate_output_schema


def test_icp_defaults_to_ten_leads() -> None:
    icp = IcpSpec.from_mapping({"campaign_name": "x", "icp_description": "DFW dental practices"})

    assert icp.target_count == 10


def test_schema_includes_relevant_fields_within_exa_limit() -> None:
    icp = IcpSpec.from_mapping(
        {
            "campaign_name": "x",
            "icp_description": "DFW dental practices",
            "preferred_columns": [
                "company_name",
                "website",
                "product_description",
                "icp_fit_score",
                "icp_fit_reasoning",
                "headquarters_location",
                "recent_signal",
                "decision_maker_titles",
                "hiring_signals",
                "recent_news",
            ],
        }
    )

    schema = build_output_schema(icp)
    fields = schema["properties"]["companies"]["items"]["properties"]

    assert len(fields) == 9
    assert "decision_maker_titles" in fields
    assert "recent_news" not in fields


def test_schema_rejects_string_without_length_limit() -> None:
    schema = build_output_schema(
        IcpSpec.from_mapping({"campaign_name": "x", "icp_description": "ICP"})
    )
    fields = schema["properties"]["companies"]["items"]["properties"]
    fields["company_name"]["description"] = "company name"

    with pytest.raises(ValidationError, match="word/character limit"):
        validate_output_schema(schema)
