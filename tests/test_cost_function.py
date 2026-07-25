"""Cost-model tests: surface tiers, main-road detection, and the extra-km penalties."""

import pytest

from bike_router.constants import RoutingParams
from bike_router.cost import _as_values, edge_cost, is_main_road, surface_tier

# Distance-only params (all penalties off) → edge cost == length.
_DIST_ONLY = RoutingParams(extra_km_per_uphill_100m=0.0, extra_km_per_unpaved_km=0.0, extra_km_per_main_road_km=0.0)


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
        length=1000.0, surface="mud", highway="primary", elev_source=0.0, elev_target=50.0, params=_DIST_ONLY
    )
    assert cost == 1000.0  # every penalty disabled → pure distance


def test_uphill_penalty_matches_extra_km_contract():
    # 100 m climb, 5 extra km per 100 m → +5000 m on top of length
    params = RoutingParams(extra_km_per_uphill_100m=5.0, extra_km_per_unpaved_km=0.0, extra_km_per_main_road_km=0.0)
    cost = edge_cost(
        length=1000.0, surface="asphalt", highway="residential", elev_source=0.0, elev_target=100.0, params=params
    )
    assert cost == pytest.approx(1000.0 + 5000.0)


def test_uphill_only_downhill_is_free():
    params = RoutingParams(extra_km_per_uphill_100m=5.0, extra_km_per_unpaved_km=0.0, extra_km_per_main_road_km=0.0)
    downhill = edge_cost(
        length=1000.0, surface="asphalt", highway="residential", elev_source=100.0, elev_target=0.0, params=params
    )
    assert downhill == 1000.0  # descending adds no penalty


def test_unpaved_penalty_scales_with_tier():
    # 1 km on moderate (tier 1) at 1 extra km/km → +1000 m; heavy (tier 2) → +2000 m
    params = RoutingParams(extra_km_per_uphill_100m=0.0, extra_km_per_unpaved_km=1.0, extra_km_per_main_road_km=0.0)
    moderate = edge_cost(
        length=1000.0, surface="gravel", highway="residential", elev_source=0.0, elev_target=0.0, params=params
    )
    heavy = edge_cost(
        length=1000.0, surface="mud", highway="residential", elev_source=0.0, elev_target=0.0, params=params
    )
    assert moderate == pytest.approx(1000.0 + 1000.0)
    assert heavy == pytest.approx(1000.0 + 2000.0)


def test_main_road_penalty():
    params = RoutingParams(extra_km_per_uphill_100m=0.0, extra_km_per_unpaved_km=0.0, extra_km_per_main_road_km=2.0)
    on_main = edge_cost(
        length=1000.0, surface="asphalt", highway="secondary", elev_source=0.0, elev_target=0.0, params=params
    )
    off_main = edge_cost(
        length=1000.0, surface="asphalt", highway="residential", elev_source=0.0, elev_target=0.0, params=params
    )
    assert on_main == pytest.approx(1000.0 + 2000.0)
    assert off_main == 1000.0


def test_penalties_are_additive():
    params = RoutingParams(extra_km_per_uphill_100m=5.0, extra_km_per_unpaved_km=1.0, extra_km_per_main_road_km=1.0)
    # 1 km, +100 m climb, gravel (tier 1), secondary main road
    cost = edge_cost(
        length=1000.0, surface="gravel", highway="secondary", elev_source=0.0, elev_target=100.0, params=params
    )
    assert cost == pytest.approx(1000.0 + 5000.0 + 1000.0 + 1000.0)


def test_cost_never_below_length():
    params = RoutingParams(extra_km_per_uphill_100m=10.0, extra_km_per_unpaved_km=3.0, extra_km_per_main_road_km=3.0)
    cost = edge_cost(
        length=500.0, surface="asphalt", highway="residential", elev_source=0.0, elev_target=0.0, params=params
    )
    assert cost >= 500.0
