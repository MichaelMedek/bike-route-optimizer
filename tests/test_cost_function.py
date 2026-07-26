"""Cost-model tests: surface tiers, main-road detection, extra-km penalties, and modes."""

import pytest

from bike_router.constants import Mode, RoutingParams
from bike_router.cost import _as_values, edge_cost, is_main_road, surface_tier

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
    assert surface_tier(surface="gravel") == 1
    assert surface_tier(surface="mud") == 2
    assert surface_tier(surface=["asphalt", "mud"]) == 2  # worst wins
    assert surface_tier(surface="spacedust") == 1  # unknown → DEFAULT_TIER (moderate)
    assert surface_tier(surface=None) == 1


def test_is_main_road():
    assert is_main_road(highway="secondary") is True
    assert is_main_road(highway="primary") is True
    assert is_main_road(highway="unclassified") is True
    assert is_main_road(highway="residential") is False
    assert is_main_road(highway="cycleway") is False
    assert is_main_road(highway=["residential", "secondary"]) is True  # any main → main
    assert is_main_road(highway=None) is True  # unknown → treated as main road


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
    # 1 km on moderate (tier 1) at 1 extra km/km → +1000 m; heavy (tier 2) → +2000 m
    params = _bike_params(unpaved=1.0)
    moderate = edge_cost(
        mode=Mode.BIKE,
        length=1000.0,
        surface="gravel",
        highway="residential",
        elev_source=0.0,
        elev_target=0.0,
        params=params,
    )
    heavy = edge_cost(
        mode=Mode.BIKE,
        length=1000.0,
        surface="mud",
        highway="residential",
        elev_source=0.0,
        elev_target=0.0,
        params=params,
    )
    assert moderate == pytest.approx(1000.0 + 1000.0)
    assert heavy == pytest.approx(1000.0 + 2000.0)


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


def test_rail_cost_uses_boarding_plus_per_km_no_terrain_penalty():
    # 10 km rail, 2 extra km/km rail + 15 km per boarding → length + 20000 + 15000.
    # Uphill/surface/main-road knobs set high must NOT affect a rail edge.
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
    assert cost == pytest.approx(10_000.0 + 15_000.0 + 20_000.0)


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


def test_transfer_cost_is_plain_length():
    cost = edge_cost(
        mode=Mode.TRANSFER,
        length=250.0,
        surface=None,
        highway=None,
        elev_source=0.0,
        elev_target=100.0,
        params=_DIST_ONLY,
    )
    assert cost == 250.0


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
