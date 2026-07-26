"""Cost-model tests: surface tiers, main-road detection, extra-km penalties, and modes."""

import pytest

from bike_router.constants import Mode
from bike_router.cost import _as_values, edge_cost, road_included, road_tier, surface_included, surface_tier
from tests.conftest import ZERO_PARAMS, zero_params


def test_as_values_normalizes_types():
    assert _as_values(tag=None) == []
    assert _as_values(tag="Asphalt") == ["asphalt"]
    assert _as_values(tag=["Gravel", "DIRT"]) == ["gravel", "dirt"]
    assert _as_values(tag=42) == ["42"]


def test_surface_tier_mapping_and_worst_wins():
    assert surface_tier(surface="asphalt") == 0
    assert surface_tier(surface="concrete:plates") == 0  # paved variant → good
    assert surface_tier(surface="gravel") == 1  # loose
    assert surface_tier(surface="ground") == 2  # natural/rough but rideable
    assert surface_tier(surface="dirt") == 2
    assert surface_tier(surface=["asphalt", "gravel"]) == 1  # worst wins (0 vs 1 → 1)
    assert surface_tier(surface=["gravel", "ground"]) == 2  # worst wins (1 vs 2 → 2)
    assert surface_tier(surface="spacedust") == 1  # unknown → DEFAULT_TIER (loose)
    assert surface_tier(surface=None) == 1


def test_surface_included_allowlist():
    assert surface_included(surface="asphalt") is True  # allowlisted paved (tier 0)
    assert surface_included(surface="gravel") is True  # allowlisted loose (tier 1)
    assert surface_included(surface="ground") is True  # allowlisted natural-rough (tier 2)
    assert surface_included(surface="dirt") is True  # tier 2 — kept, not dropped
    assert surface_included(surface=None) is True  # untagged → kept as DEFAULT_TIER
    assert surface_included(surface="sand") is False  # genuinely impassable → excluded
    assert surface_included(surface="mud") is False
    assert surface_included(surface="gravel;mud") is False  # any disallowed value → excluded


def test_road_tier_mapping_and_worst_wins():
    assert road_tier(highway="secondary") == 1
    assert road_tier(highway="primary") == 1
    assert road_tier(highway="unclassified") == 1  # unclassified is a main road
    assert road_tier(highway="trunk") == 1
    assert road_tier(highway="primary_link") == 1  # arterial link → main
    assert road_tier(highway="residential") == 0  # quiet way
    assert road_tier(highway="cycleway") == 0
    assert road_tier(highway="tertiary") == 0  # quiet way, not a main road
    assert road_tier(highway=["residential", "secondary"]) == 1  # worst wins → main
    assert road_tier(highway=None) == 1  # untagged → DEFAULT_TIER (main, pessimistic)


def test_road_included_allowlist():
    assert road_included(highway="residential") is True  # allowlisted quiet
    assert road_included(highway="secondary") is True  # allowlisted main
    assert road_included(highway=None) is True  # untagged → kept as DEFAULT_TIER
    assert road_included(highway="motorway") is False  # not in allowlist → excluded (no bikes)
    assert road_included(highway="raceway") is False
    assert road_included(highway=["residential", "motorway"]) is False  # any disallowed → excluded


def test_distance_only_cost_is_length():
    cost = edge_cost(
        mode=Mode.BIKE,
        length=1000.0,
        surface="mud",
        highway="primary",
        elev_source=0.0,
        elev_target=50.0,
        params=ZERO_PARAMS,
    )
    assert cost == 1000.0  # every penalty disabled → pure distance


def test_uphill_penalty_matches_extra_km_contract():
    # 100 m climb, 5 extra km per 100 m → +5000 m on top of length
    params = zero_params(extra_km_per_uphill_100m=5.0)
    cost = edge_cost(
        mode=Mode.BIKE,
        length=1000.0,
        surface="asphalt",
        highway="residential",
        elev_source=0.0,
        elev_target=100.0,
        params=params,
    )
    assert cost == pytest.approx(1000.0 + 5000.0)


def test_uphill_only_downhill_is_free():
    params = zero_params(extra_km_per_uphill_100m=5.0)
    downhill = edge_cost(
        mode=Mode.BIKE,
        length=1000.0,
        surface="asphalt",
        highway="residential",
        elev_source=100.0,
        elev_target=0.0,
        params=params,
    )
    assert downhill == 1000.0  # descending adds no penalty


def test_unpaved_penalty_scales_with_tier():
    # tier 0 (paved) → no penalty; tier 1 (moderate) at 1 extra km/km → +1000 m.
    params = zero_params(extra_km_per_unpaved_km=1.0)
    paved = edge_cost(
        mode=Mode.BIKE,
        length=1000.0,
        surface="asphalt",
        highway="residential",
        elev_source=0.0,
        elev_target=0.0,
        params=params,
    )
    moderate = edge_cost(
        mode=Mode.BIKE,
        length=1000.0,
        surface="gravel",
        highway="residential",
        elev_source=0.0,
        elev_target=0.0,
        params=params,
    )
    assert paved == 1000.0  # tier 0 adds nothing
    assert moderate == pytest.approx(1000.0 + 1000.0)


def test_unpaved_tier2_doubles_the_penalty():
    # A natural/rough surface (tier 2) adds TWICE the tier-1 penalty — the tier is a
    # literal multiplier. 1 km, 1 extra km/km: tier 1 → +1000 m, tier 2 → +2000 m.
    params = zero_params(extra_km_per_unpaved_km=1.0)
    loose = edge_cost(
        mode=Mode.BIKE,
        length=1000.0,
        surface="gravel",
        highway="residential",
        elev_source=0.0,
        elev_target=0.0,
        params=params,
    )
    rough = edge_cost(
        mode=Mode.BIKE,
        length=1000.0,
        surface="ground",
        highway="residential",
        elev_source=0.0,
        elev_target=0.0,
        params=params,
    )
    assert loose == pytest.approx(1000.0 + 1000.0)  # tier 1 → ×1
    assert rough == pytest.approx(1000.0 + 2000.0)  # tier 2 → ×2 (doubled)


def test_main_road_penalty():
    params = zero_params(extra_km_per_main_road_km=2.0)
    on_main = edge_cost(
        mode=Mode.BIKE,
        length=1000.0,
        surface="asphalt",
        highway="secondary",
        elev_source=0.0,
        elev_target=0.0,
        params=params,
    )
    off_main = edge_cost(
        mode=Mode.BIKE,
        length=1000.0,
        surface="asphalt",
        highway="residential",
        elev_source=0.0,
        elev_target=0.0,
        params=params,
    )
    assert on_main == pytest.approx(1000.0 + 2000.0)
    assert off_main == 1000.0


def test_penalties_are_additive():
    params = zero_params(extra_km_per_uphill_100m=5.0, extra_km_per_unpaved_km=1.0, extra_km_per_main_road_km=1.0)
    # 1 km, +100 m climb, gravel (tier 1), secondary main road
    cost = edge_cost(
        mode=Mode.BIKE,
        length=1000.0,
        surface="gravel",
        highway="secondary",
        elev_source=0.0,
        elev_target=100.0,
        params=params,
    )
    assert cost == pytest.approx(1000.0 + 5000.0 + 1000.0 + 1000.0)


def test_cost_never_below_length():
    # Penalties actually fire (gravel + main road + a climb) so the ">= length" invariant is
    # non-vacuous; assert the exact total: 500 + 100m-climb + 0.5km·gravel + 0.5km·main.
    params = zero_params(extra_km_per_uphill_100m=10.0, extra_km_per_unpaved_km=3.0, extra_km_per_main_road_km=3.0)
    cost = edge_cost(
        mode=Mode.BIKE,
        length=500.0,
        surface="gravel",
        highway="primary",
        elev_source=0.0,
        elev_target=100.0,
        params=params,
    )
    # 500 + (100/100·10·1000) + (0.5·3·1000) + (0.5·3·1000) = 500 + 10000 + 1500 + 1500
    assert cost == pytest.approx(500.0 + 10000.0 + 1500.0 + 1500.0)
    assert cost > 500.0  # never below raw length


def test_rail_cost_uses_per_km_only_no_boarding_no_terrain_penalty():
    # 10 km rail, 2 extra km/km rail → length + 20000. Boarding is NOT on the rail edge
    # (it lives on the station edges); uphill/surface/main-road knobs must NOT affect rail.
    params = zero_params(extra_km_per_rail_km=2.0, extra_km_per_boarding=15.0)
    cost = edge_cost(
        mode=Mode.RAIL,
        length=10_000.0,
        surface="mud",
        highway="primary",
        elev_source=0.0,
        elev_target=500.0,  # big climb — ignored for rail
        params=params,
    )
    assert cost == pytest.approx(10_000.0 + 20_000.0)


def test_rail_sliders_scale_cost_so_high_values_deter_rail():
    cheap = zero_params(extra_km_per_rail_km=0.1, extra_km_per_boarding=1.0)
    dear = zero_params(extra_km_per_rail_km=5.0, extra_km_per_boarding=80.0)
    kwargs = dict(length=5_000.0, surface=None, highway=None, elev_source=0.0, elev_target=0.0)
    assert edge_cost(mode=Mode.RAIL, params=dear, **kwargs) > edge_cost(mode=Mode.RAIL, params=cheap, **kwargs)


def test_station_cost_is_length_plus_half_boarding():
    # Station edge = straight-line length + half the boarding charge. With boarding 0 it is
    # pure length; with boarding 10 km it adds 0.5·10·1000 = 5000 m on top. Length ≤ radius.
    assert (
        edge_cost(
            mode=Mode.STATION,
            length=150.0,
            surface=None,
            highway=None,
            elev_source=0.0,
            elev_target=100.0,  # elevation ignored for station edges
            params=ZERO_PARAMS,
        )
        == 150.0
    )
    params = zero_params(extra_km_per_boarding=10.0)
    cost = edge_cost(
        mode=Mode.STATION, length=150.0, surface=None, highway=None, elev_source=0.0, elev_target=0.0, params=params
    )
    assert cost == pytest.approx(150.0 + 5000.0)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown edge mode"):
        edge_cost(
            mode="teleport",
            length=1.0,
            surface=None,
            highway=None,
            elev_source=0.0,
            elev_target=0.0,
            params=ZERO_PARAMS,
        )
