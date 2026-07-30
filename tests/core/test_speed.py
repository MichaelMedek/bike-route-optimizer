"""Speed-model tests — surface base speeds + linear grade interpolation + floor."""

import pytest

from bike_router.core.constants import SpeedConfig
from bike_router.core.speed import effective_speed_kmh


def test_flat_uses_surface_base_speed():
    assert effective_speed_kmh(surface_tier=0, grade=0.0) == 25.0  # paved
    assert effective_speed_kmh(surface_tier=1, grade=0.0) == 20.0  # loose
    assert effective_speed_kmh(surface_tier=2, grade=0.0) == 15.0  # natural-rough


def test_downhill_holds_base_speed():
    assert effective_speed_kmh(surface_tier=0, grade=-0.10) == 25.0


def test_walk_grade_hits_walk_speed():
    assert effective_speed_kmh(surface_tier=0, grade=SpeedConfig.WALK_GRADE) == pytest.approx(SpeedConfig.WALK_KMH)
    # steeper than WALK_GRADE stays at walking pace
    assert effective_speed_kmh(surface_tier=0, grade=0.30) == pytest.approx(SpeedConfig.WALK_KMH)


def test_linear_interpolation_midpoint():
    # halfway to WALK_GRADE on good asphalt → halfway between 25 and 5 = 15 km/h
    mid = effective_speed_kmh(surface_tier=0, grade=SpeedConfig.WALK_GRADE / 2)
    assert mid == pytest.approx((25.0 + SpeedConfig.WALK_KMH) / 2)


def test_speed_monotonic_decreasing_with_grade():
    grades = [0.0, 0.02, 0.05, 0.08, 0.12]
    speeds = [effective_speed_kmh(surface_tier=0, grade=g) for g in grades]
    assert all(speeds[i] >= speeds[i + 1] for i in range(len(speeds) - 1))


def test_unknown_tier_raises():
    with pytest.raises(AssertionError):
        effective_speed_kmh(surface_tier=9, grade=0.0)
