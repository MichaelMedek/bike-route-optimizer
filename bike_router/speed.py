"""Surface- and grade-adaptive cycling speed (km/h).

Each edge is treated as a single linear grade. Speed drops linearly from the
surface's base speed at 0 % grade to walking pace at WALK_GRADE, and stays at
walking pace on anything steeper. Flat and downhill ride at the base speed.
"""

from bike_router.constants import SpeedConfig


def effective_speed_kmh(surface_tier: int, grade: float) -> float:
    """Ride speed (km/h) for a surface tier and linear grade.

    Args:
        surface_tier: 0 good / 1 moderate / 2 heavy (see SurfaceConfig).
        grade: rise/run fraction over the edge (<= 0 = flat/downhill).
    """
    assert surface_tier in SpeedConfig.BASE_KMH_BY_TIER, "unknown surface tier"
    base = SpeedConfig.BASE_KMH_BY_TIER[surface_tier]
    if grade <= 0.0:
        return base
    if grade >= SpeedConfig.WALK_GRADE:
        return SpeedConfig.WALK_KMH
    # linear interpolate base (at grade 0) → walk (at WALK_GRADE)
    frac = grade / SpeedConfig.WALK_GRADE
    return base + frac * (SpeedConfig.WALK_KMH - base)
