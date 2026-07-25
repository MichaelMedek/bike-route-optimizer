"""RoutingParams validation — out-of-range values raise loudly (no silent clamp)."""

import pytest

from bike_router.constants import RoutingDefaults, RoutingParams


def test_valid_params_accepted():
    at_max = RoutingDefaults.MAX_EXTRA_KM
    params = RoutingParams(extra_km_per_uphill_100m=0.0, extra_km_per_unpaved_km=1.0, extra_km_per_main_road_km=at_max)
    assert params.extra_km_per_main_road_km == at_max


def test_negative_value_raises():
    with pytest.raises(ValueError):
        RoutingParams(extra_km_per_uphill_100m=-1.0, extra_km_per_unpaved_km=0.0, extra_km_per_main_road_km=0.0)


def test_above_max_raises():
    over = RoutingDefaults.MAX_EXTRA_KM + 1
    with pytest.raises(ValueError):
        RoutingParams(extra_km_per_uphill_100m=over, extra_km_per_unpaved_km=0.0, extra_km_per_main_road_km=0.0)
