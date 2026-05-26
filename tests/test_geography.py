from __future__ import annotations

import pytest

from deep_research_agent.geography import normalize_geography, suggest_geography_alias_update


def test_normalize_north_texas_expands_to_targeted_terms() -> None:
    scope = normalize_geography(" North   Texas ")

    assert scope.raw == "North Texas"
    assert scope.canonical == "Dallas-Fort Worth TX"
    assert scope.scope_type == "regional_area"
    assert scope.confidence == 0.75
    assert scope.search_terms[:6] == (
        "North Texas",
        "Dallas-Fort Worth TX",
        "DFW",
        "Dallas TX",
        "Fort Worth TX",
        "Plano TX",
    )
    assert "DFW" in scope.must_include_any
    assert scope.warnings == ()


def test_normalize_dfw_aliases_share_canonical_scope() -> None:
    dfw = normalize_geography("DFW")
    spelled = normalize_geography("Dallas-Fort Worth")

    assert dfw.canonical == "Dallas-Fort Worth TX"
    assert spelled.canonical == "Dallas-Fort Worth TX"
    assert dfw.search_terms[0] == "DFW"
    assert spelled.search_terms[:2] == ("Dallas-Fort Worth", "DFW")


def test_empty_geography_is_noop_scope() -> None:
    scope = normalize_geography("   ")

    assert scope.raw == ""
    assert scope.scope_type == "none"
    assert scope.search_terms == ()
    assert scope.warnings == ()


def test_unknown_ambiguous_region_preserves_raw_and_warns() -> None:
    scope = normalize_geography("Central Plains region")

    assert scope.canonical == "Central Plains region"
    assert scope.search_terms == ("Central Plains region",)
    assert scope.must_include_any == ("Central Plains region",)
    assert scope.warnings == ("Geography was not recognized; searching the raw phrase only.",)


@pytest.mark.parametrize(
    ("raw", "canonical", "expected_terms"),
    [
        ("Bay Area", "San Francisco Bay Area CA", ("San Francisco CA", "San Jose CA")),
        ("DMV", "Washington DC metro", ("Washington DC", "Arlington VA")),
        ("South Florida", "Miami-Fort Lauderdale-West Palm Beach FL", ("Miami FL",)),
        ("Chicagoland", "Chicago metro IL", ("Chicago IL", "Naperville IL")),
        ("Greater Atlanta", "Atlanta metro GA", ("Atlanta GA", "Marietta GA")),
        ("Greater Houston", "Houston metro TX", ("Houston TX", "The Woodlands TX")),
        ("Greater Phoenix", "Phoenix metro AZ", ("Phoenix AZ", "Scottsdale AZ")),
        ("Puget Sound", "Seattle-Puget Sound WA", ("Seattle WA", "Bellevue WA")),
        ("Twin Cities", "Minneapolis-Saint Paul MN", ("Minneapolis MN", "Saint Paul MN")),
        ("Research Triangle", "Research Triangle NC", ("Raleigh NC", "Durham NC")),
        ("Inland Empire", "Inland Empire CA", ("Riverside CA", "San Bernardino CA")),
        ("Delaware Valley", "Philadelphia metro", ("Philadelphia PA", "Camden NJ")),
        ("Greater Boston", "Boston metro MA", ("Boston MA", "Cambridge MA")),
        ("Front Range", "Denver Front Range CO", ("Denver CO", "Boulder CO")),
    ],
)
def test_common_large_market_aliases_expand_to_targeted_terms(
    raw: str, canonical: str, expected_terms: tuple[str, ...]
) -> None:
    scope = normalize_geography(raw)

    assert scope.canonical == canonical
    for term in expected_terms:
        assert term in scope.search_terms
    assert scope.warnings == ()


def test_alias_update_suggestion_is_drafted_for_unknown_broad_region() -> None:
    suggestion = suggest_geography_alias_update(
        "Central Plains region",
        observed_terms=("Omaha NE", "Kansas City MO"),
    )

    assert suggestion is not None
    assert suggestion.alias_key == "central plains region"
    assert suggestion.canonical == "Central Plains region"
    assert suggestion.search_terms == ("Central Plains region", "Omaha NE", "Kansas City MO")
    assert suggestion.must_include_any == (
        "Central Plains region",
        "Omaha NE",
        "Kansas City MO",
    )


def test_alias_update_suggestion_skips_known_or_specific_geographies() -> None:
    assert suggest_geography_alias_update("North Texas") is None
    assert suggest_geography_alias_update("Dallas") is None
