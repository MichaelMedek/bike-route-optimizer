"""Speed-model tests — continuous surface base speeds, linear grade interpolation, walk floor, unit conv."""

import pytest

from bike_router.core.constants import GpxConfig, SpeedConfig
from bike_router.core.speed import base_speed_kmh, effective_speed_kmh, kmh_to_ms


def test_kmh_to_ms():
    # The single km/h → m/s conversion: 36 km/h = 10 m/s; 0 stays 0.
    assert kmh_to_ms(kmh=36.0) == pytest.approx(10.0)
    assert kmh_to_ms(kmh=0.0) == 0.0
    assert kmh_to_ms(kmh=SpeedConfig.WALK_KMH) == pytest.approx(
        SpeedConfig.WALK_KMH * GpxConfig.METERS_PER_KM / GpxConfig.SECONDS_PER_HOUR
    )


def test_base_speed_kmh():
    # Linear from paved anchor (weight 0 → 25) to the rough floor (weight MAX → 15); weights beyond
    # MAX clamp to the floor and never dip below it; monotonic non-increasing in weight.
    assert base_speed_kmh(surface_weight=0.0) == pytest.approx(SpeedConfig.BASE_KMH_AT_WEIGHT0)
    assert base_speed_kmh(surface_weight=SpeedConfig.SURFACE_WEIGHT_MAX) == pytest.approx(
        SpeedConfig.BASE_KMH_AT_WEIGHT_MAX
    )
    assert base_speed_kmh(surface_weight=10.0) == pytest.approx(SpeedConfig.BASE_KMH_AT_WEIGHT_MAX)  # clamps
    mid = base_speed_kmh(surface_weight=SpeedConfig.SURFACE_WEIGHT_MAX / 2)
    assert mid == pytest.approx((SpeedConfig.BASE_KMH_AT_WEIGHT0 + SpeedConfig.BASE_KMH_AT_WEIGHT_MAX) / 2)


def test_continuous_surface_speed_is_strictly_monotone():
    # REGRESSION (continuous tiers): at a fixed non-walk grade, base ride speed strictly DECREASES as
    # the surface weight rises across [0, SURFACE_WEIGHT_MAX] — a gradient the old int buckets couldn't
    # express; beyond the max it clamps to the rough floor (tested in test_base_speed_kmh).
    weights = [0.0, 0.2, 0.4, 0.6, 0.8, SpeedConfig.SURFACE_WEIGHT_MAX]
    speeds = [effective_speed_kmh(surface_weight=w, grade=0.03) for w in weights]
    assert all(speeds[i] > speeds[i + 1] for i in range(len(speeds) - 1))


def test_effective_speed_kmh():
    # Flat/downhill hold the surface base (continuous in weight); the grade penalty ramps linearly to
    # WALK_KMH at WALK_GRADE and stays there beyond; monotonic non-increasing; negative weight fails loud.
    assert effective_speed_kmh(surface_weight=0.0, grade=0.0) == pytest.approx(25.0)  # paved
    assert effective_speed_kmh(surface_weight=SpeedConfig.SURFACE_WEIGHT_MAX, grade=0.0) == pytest.approx(15.0)  # rough
    assert effective_speed_kmh(surface_weight=0.0, grade=-0.10) == pytest.approx(25.0)  # downhill holds base
    assert effective_speed_kmh(surface_weight=0.0, grade=SpeedConfig.WALK_GRADE) == pytest.approx(SpeedConfig.WALK_KMH)
    assert effective_speed_kmh(surface_weight=0.0, grade=0.30) == pytest.approx(SpeedConfig.WALK_KMH)  # steeper → walk
    # linear midpoint between base (25) and walk at half WALK_GRADE
    assert effective_speed_kmh(surface_weight=0.0, grade=SpeedConfig.WALK_GRADE / 2) == pytest.approx(
        (25.0 + SpeedConfig.WALK_KMH) / 2
    )
    speeds = [effective_speed_kmh(surface_weight=0.0, grade=g) for g in (0.0, 0.02, 0.05, 0.08, 0.12)]
    assert all(speeds[i] >= speeds[i + 1] for i in range(len(speeds) - 1))  # monotonic ↓ with grade
    # a rougher surface (higher weight) is never faster than a smoother one at the same grade
    assert effective_speed_kmh(surface_weight=1.0, grade=0.03) <= effective_speed_kmh(surface_weight=0.0, grade=0.03)
    with pytest.raises(AssertionError):
        effective_speed_kmh(surface_weight=-1.0, grade=0.0)  # negative weight → fail loud
