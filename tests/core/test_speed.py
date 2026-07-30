"""Speed-model tests — surface base speeds, linear grade interpolation, walk floor, unit conv."""

import pytest

from bike_router.core.constants import GpxConfig, SpeedConfig
from bike_router.core.speed import effective_speed_kmh, kmh_to_ms


def test_kmh_to_ms():
    # The single km/h → m/s conversion: 36 km/h = 10 m/s; 0 stays 0.
    assert kmh_to_ms(kmh=36.0) == pytest.approx(10.0)
    assert kmh_to_ms(kmh=0.0) == 0.0
    assert kmh_to_ms(kmh=SpeedConfig.WALK_KMH) == pytest.approx(
        SpeedConfig.WALK_KMH * GpxConfig.METERS_PER_KM / GpxConfig.SECONDS_PER_HOUR
    )


def test_effective_speed_kmh():
    # Flat/downhill hold the surface base speed (per tier); the grade penalty ramps linearly to
    # WALK_KMH at WALK_GRADE and stays there beyond; monotonic non-increasing; bad tier fails loud.
    assert effective_speed_kmh(surface_tier=0, grade=0.0) == 25.0  # paved
    assert effective_speed_kmh(surface_tier=1, grade=0.0) == 20.0  # loose
    assert effective_speed_kmh(surface_tier=2, grade=0.0) == 15.0  # natural-rough
    assert effective_speed_kmh(surface_tier=0, grade=-0.10) == 25.0  # downhill holds base
    assert effective_speed_kmh(surface_tier=0, grade=SpeedConfig.WALK_GRADE) == pytest.approx(SpeedConfig.WALK_KMH)
    assert effective_speed_kmh(surface_tier=0, grade=0.30) == pytest.approx(SpeedConfig.WALK_KMH)  # steeper → walk
    # linear midpoint between base (25) and walk at half WALK_GRADE
    assert effective_speed_kmh(surface_tier=0, grade=SpeedConfig.WALK_GRADE / 2) == pytest.approx(
        (25.0 + SpeedConfig.WALK_KMH) / 2
    )
    speeds = [effective_speed_kmh(surface_tier=0, grade=g) for g in (0.0, 0.02, 0.05, 0.08, 0.12)]
    assert all(speeds[i] >= speeds[i + 1] for i in range(len(speeds) - 1))  # monotonic ↓ with grade
    with pytest.raises(AssertionError):
        effective_speed_kmh(surface_tier=9, grade=0.0)  # unknown tier → fail loud
