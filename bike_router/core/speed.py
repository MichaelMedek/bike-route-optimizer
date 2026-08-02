"""Surface- and grade-adaptive cycling speed (km/h); each edge is a single linear grade.

Speed drops linearly from the surface's base speed at 0 % grade to walking pace at WALK_GRADE, then
stays at walking pace on anything steeper. Flat and downhill ride at the base speed.
"""

from bike_router.core.constants import GpxConfig, SpeedConfig


def kmh_to_ms(kmh: float) -> float:
    """Convert km/h to m/s — the single source for this unit conversion."""
    return kmh * GpxConfig.METERS_PER_KM / GpxConfig.SECONDS_PER_HOUR


def base_speed_kmh(surface_weight: float) -> float:
    """Flat-ground base speed (km/h) for a continuous surface weight.

    Linear from BASE_KMH_AT_WEIGHT0 (weight 0.0, paved) to BASE_KMH_AT_WEIGHT_MAX (SURFACE_WEIGHT_MAX);
    weights beyond the max clamp to the rough-surface floor (never below it).
    """
    frac = min(surface_weight / SpeedConfig.SURFACE_WEIGHT_MAX, 1.0)
    return SpeedConfig.BASE_KMH_AT_WEIGHT0 + frac * (
        SpeedConfig.BASE_KMH_AT_WEIGHT_MAX - SpeedConfig.BASE_KMH_AT_WEIGHT0
    )


def effective_speed_kmh(surface_weight: float, grade: float) -> float:
    """Ride speed (km/h) for a continuous surface weight and linear grade.

    Args:
        surface_weight: Crr-ordered surface cost weight (0.0 paved … ~1.7 rough; see SurfaceConfig).
        grade: rise/run fraction over the edge (<= 0 = flat/downhill).
    """
    assert surface_weight >= 0.0, "surface weight must be non-negative"
    base = base_speed_kmh(surface_weight=surface_weight)
    if grade <= 0.0:
        return base
    if grade >= SpeedConfig.WALK_GRADE:
        return SpeedConfig.WALK_KMH
    # linear interpolate base (at grade 0) → walk (at WALK_GRADE)
    frac = grade / SpeedConfig.WALK_GRADE
    return base + frac * (SpeedConfig.WALK_KMH - base)
