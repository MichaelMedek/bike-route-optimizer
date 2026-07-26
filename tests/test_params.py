"""RoutingParams validation — out-of-range values raise loudly (no silent clamp)."""

import pytest

from bike_router.constants import RoutingDefaults
from bike_router.errors import ParamOutOfRangeError
from tests.conftest import params


def test_valid_params_accepted():
    at_max = RoutingDefaults.MAX_EXTRA_KM
    at_max_params = params(extra_km_per_main_road_km=at_max)
    assert at_max_params.extra_km_per_main_road_km == at_max


def test_negative_value_raises():
    with pytest.raises(ParamOutOfRangeError):
        params(extra_km_per_uphill_100m=-1.0)


def test_above_max_raises():
    with pytest.raises(ParamOutOfRangeError):
        params(extra_km_per_uphill_100m=RoutingDefaults.MAX_EXTRA_KM + 1)


def test_rail_params_validated_too():
    with pytest.raises(ParamOutOfRangeError):
        params(extra_km_per_boarding=-5.0)
    with pytest.raises(ParamOutOfRangeError):
        params(extra_km_per_rail_km=RoutingDefaults.MAX_EXTRA_KM + 1)
