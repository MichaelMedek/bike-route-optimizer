"""Cost-model tests: surface tiers, main-road detection, extra-km penalties, and modes."""

import pytest

from bike_router.constants import Mode, RoutingParams
from bike_router.cost import _as_values, edge_cost, road_included, road_tier, surface_included, surface_tier

# Distance-only params (all penalties off) → bike edge cost == length.
_DIST_ONLY = RoutingParams(
    extra_km_per_uphill_100m=0.0,
    extra_km_per_unpaved_km=0.0,
    extra_km_per_main_road_km=0.0,
    extra_km_per_rail_km=0.0,
    extra_km_per_boarding=0.0,
)


def _bike_params(*, uphill: float = 0.0, unpaved: float = 0.0, main_road: float = 0.0) -> RoutingParams:
    """RoutingParams with only the bike knobs set (rail knobs zeroed)."""
    return RoutingParams(
        extra_km_per_uphill_100m=uphill,
        extra_km_per_unpaved_km=unpaved,
        extra_km_per_main_road_km=main_road,
        extra_km_per_rail_km=0.0,
        extra_km_per_boarding=0.0,
    )


def test_as_values_normalizes_types():
    assert _as_values(tag=None) == []
    assert _as_values(tag="Asphalt") == ["asphalt"]
    assert _as_values(tag=["Gravel", "DIRT"]) == ["gravel", "dirt"]
    assert _as_values(tag=42) == ["42"]


def test_surface_tier_mapping_and_worst_wins():
    assert surface_tier(surface="asphalt") == 0
    assert surface_tier(surface="concrete:plates") == 0  # paved variant → good
    assert surface_tier(surface="gravel") == 1
    assert surface_tier(surface=["asphalt", "gravel"]) == 1  # worst wins
    assert surface_tier(surface="spacedust") == 1  # unknown → DEFAULT_TIER (moderate)
    assert surface_tier(surface=None) == 1


def test_surface_included_allowlist():
    assert surface_included(surface="asphalt") is True  # allowlisted good
    assert surface_included(surface="gravel") is True  # allowlisted moderate
    assert surface_included(surface=None) is True  # untagged → kept as DEFAULT_TIER
    assert surface_included(surface="sand") is False  # not in allowlist → excluded
    assert surface_included(surface="dirt") is False
    assert surface_included(surface="gravel;dirt") is False  # any disallowed value → excluded


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
        params=_DIST_ONLY,
    )
    assert cost == 1000.0  # every penalty disabled → pure distance


def test_uphill_penalty_matches_extra_km_contract():
    # 100 m climb, 5 extra km per 100 m → +5000 m on top of length
    params = _bike_params(uphill=5.0)
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
    params = _bike_params(uphill=5.0)
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
    params = _bike_params(unpaved=1.0)
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


def test_main_road_penalty():
    params = _bike_params(main_road=2.0)
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
    params = _bike_params(uphill=5.0, unpaved=1.0, main_road=1.0)
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
    params = _bike_params(uphill=10.0, unpaved=3.0, main_road=3.0)
    cost = edge_cost(
        mode=Mode.BIKE,
        length=500.0,
        surface="asphalt",
        highway="residential",
        elev_source=0.0,
        elev_target=0.0,
        params=params,
    )
    assert cost >= 500.0


def test_rail_cost_uses_per_km_only_no_boarding_no_terrain_penalty():
    # 10 km rail, 2 extra km/km rail → length + 20000. Boarding is NOT on the rail edge
    # (it lives on the station edges); uphill/surface/main-road knobs must NOT affect rail.
    params = RoutingParams(
        extra_km_per_uphill_100m=50.0,
        extra_km_per_unpaved_km=50.0,
        extra_km_per_main_road_km=50.0,
        extra_km_per_rail_km=2.0,
        extra_km_per_boarding=15.0,
    )
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
    cheap = RoutingParams(
        extra_km_per_uphill_100m=0.0,
        extra_km_per_unpaved_km=0.0,
        extra_km_per_main_road_km=0.0,
        extra_km_per_rail_km=0.1,
        extra_km_per_boarding=1.0,
    )
    dear = RoutingParams(
        extra_km_per_uphill_100m=0.0,
        extra_km_per_unpaved_km=0.0,
        extra_km_per_main_road_km=0.0,
        extra_km_per_rail_km=5.0,
        extra_km_per_boarding=80.0,
    )
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
            params=_DIST_ONLY,
        )
        == 150.0
    )
    params = RoutingParams(
        extra_km_per_uphill_100m=0.0,
        extra_km_per_unpaved_km=0.0,
        extra_km_per_main_road_km=0.0,
        extra_km_per_rail_km=0.0,
        extra_km_per_boarding=10.0,
    )
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
            params=_DIST_ONLY,
        )
