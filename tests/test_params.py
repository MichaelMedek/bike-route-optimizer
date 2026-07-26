"""RoutingParams validation — out-of-range values raise loudly (no silent clamp)."""

import pytest

from bike_router.constants import PARAM_SPECS, RoutingDefaults, RoutingParams


def _params(**overrides: float) -> RoutingParams:
    """Build RoutingParams from the spec defaults, overriding named fields."""
    values = {spec.field: spec.default for spec in PARAM_SPECS}
    values.update(overrides)
    return RoutingParams(**values)


def test_valid_params_accepted():
    at_max = RoutingDefaults.MAX_EXTRA_KM
    params = _params(extra_km_per_main_road_km=at_max)
    assert params.extra_km_per_main_road_km == at_max


def test_negative_value_raises():
    with pytest.raises(ValueError):
        _params(extra_km_per_uphill_100m=-1.0)


def test_above_max_raises():
    with pytest.raises(ValueError):
        _params(extra_km_per_uphill_100m=RoutingDefaults.MAX_EXTRA_KM + 1)


def test_rail_params_validated_too():
    with pytest.raises(ValueError):
        _params(extra_km_per_boarding=-5.0)
    with pytest.raises(ValueError):
        _params(extra_km_per_rail_km=RoutingDefaults.MAX_EXTRA_KM + 1)
