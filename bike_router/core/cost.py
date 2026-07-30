"""Per-edge cost in intuitive "extra kilometres".

The cost of a directed BIKE edge is its real length plus three user-controlled
penalties, all measured in metres so they add cleanly:

    cost = length
         + uphill_penalty     = (climb_m / 100) * extra_km_per_uphill_100m   * 1000
         + unpaved_penalty    = surface_tier * extra_km_per_unpaved_km       * length/1000 * 1000
         + main_road_penalty  = road_tier * extra_km_per_main_road_km        * length/1000 * 1000

A RAIL edge instead costs only a per-km rail charge (slider-controlled, in the same
"extra km" currency), with NO terrain penalties — a train doesn't care about
hills/surface/traffic. The boarding charge is NOT here; it lives on the two station
edges (see below), so a board→ride→alight pays it exactly once each way:

    cost = extra_km_per_rail_km * length/1000 * 1000

A STATION edge (bike node ↔ station node) costs its straight-line length PLUS HALF the
boarding charge. Entry (board) + exit (alight) each carry half, so any use of a station
sums to the full boarding hassle — which also makes cycling THROUGH a station (in one
entrance, out another) cost a full boarding, naturally deterring cut-through. With the
boarding slider at 0 a cut-through is free; we accept that (the felt cost is honest):

    cost = length + 0.5 * extra_km_per_boarding * 1000

Each `extra_km_*` is how many extra virtual kilometres the rider will accept to
avoid one unit of the bad thing. All penalties are >= 0, so the cheapest possible
edge is pure distance — which keeps the A* great-circle heuristic admissible.

Because the graph is directed, only the uphill direction of a street is penalised
(the downhill direction has climb_m = 0), so uphill costs more (flat-preferring).
"""

import math

import numpy as np
import pandas as pd

from bike_router.core.constants import (
    CostConfig,
    GpxConfig,
    Mode,
    RailConfig,
    RoadConfig,
    RoutingParams,
    SurfaceConfig,
)


def _as_values(tag: object) -> list[str]:
    """Normalize an OSM tag (str | list | None | float nan) to a list of lowercased strings.
    pandas/pyrosm encode a missing tag as float ``nan`` and simplify splices ``nan`` INSIDE merged
    lists (e.g. ``['asphalt', nan]``); stringifying it to "nan" would reject most roads, so drop nans.
    """
    if _is_missing(value=tag):
        return []
    if isinstance(tag, list | tuple | set):
        return [str(value).lower() for value in tag if not _is_missing(value=value)]
    return [str(tag).lower()]


def _is_missing(value: object) -> bool:
    """True for an absent OSM tag value: ``None`` or float ``nan`` (how pandas encodes a missing tag)."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def tag_tier(tag: object, tier_map: dict[str, int], default_tier: int) -> int:
    """Penalty tier for an OSM tag against its tier map: worst (highest) wins.

    Shared by surface_tier and road_tier. Unknown values in the tag are ignored;
    an all-unknown / missing tag falls back to ``default_tier``.
    """
    tiers = [tier_map[value] for value in _as_values(tag=tag) if value in tier_map]
    tier = max(tiers) if tiers else default_tier
    assert tier in {0, 1, 2}, "tier must be 0, 1, or 2"
    return tier


def tag_included(tag: object, tier_map: dict[str, int]) -> bool:
    """False iff the tag names a category outside the allowlist (→ excluded).

    Shared by surface_included and road_included. Missing/untagged (no values) → True
    (kept as DEFAULT_TIER). A tag naming ONLY categories in ``tier_map`` → True.
    """
    values = _as_values(tag=tag)
    if not values:
        return True
    return all(value in tier_map for value in values)


def surface_tier(surface: object) -> int:
    """Penalty tier for a surface tag: 0 good, 1 moderate (worst wins; untagged → default)."""
    return tag_tier(tag=surface, tier_map=SurfaceConfig.SURFACE_TIER, default_tier=SurfaceConfig.DEFAULT_TIER)


def surface_included(surface: object) -> bool:
    """False iff the surface names a category outside the allowlist (missing → kept)."""
    return tag_included(tag=surface, tier_map=SurfaceConfig.SURFACE_TIER)


def road_tier(highway: object) -> int:
    """Penalty tier for a highway tag: 0 quiet, 1 main road (worst wins; untagged → default)."""
    return tag_tier(tag=highway, tier_map=RoadConfig.ROAD_TIER, default_tier=RoadConfig.DEFAULT_TIER)


def road_included(highway: object) -> bool:
    """False iff the highway names a class outside the allowlist (missing → kept)."""
    return tag_included(tag=highway, tier_map=RoadConfig.ROAD_TIER)


def edge_cost_array(*, edges_df: pd.DataFrame, elev_by_osmid: dict[int, float], params: RoutingParams) -> np.ndarray:
    """Per-edge cost in metres for a whole edge table — the ONE cost formula, fully vectorized.
    Branches on mode (``np.select``): bike = length + uphill + unpaved + main-road penalties (uphill
    only); rail = per-km charge; station = length + half the boarding charge (board + alight = full).
    """
    mpk = GpxConfig.METERS_PER_KM
    mode = edges_df["mode"].to_numpy()
    length = edges_df["length_m"].to_numpy(dtype=np.float64)
    length_km = length / mpk
    from_elev = edges_df["from_node"].map(elev_by_osmid).to_numpy(dtype=np.float64)
    to_elev = edges_df["to_node"].map(elev_by_osmid).to_numpy(dtype=np.float64)
    s_tier = edges_df["surface"].map(surface_tier).to_numpy(dtype=np.float64)
    r_tier = edges_df["highway"].map(road_tier).to_numpy(dtype=np.float64)

    climb_m = np.maximum(to_elev - from_elev, 0.0)  # uphill only; downhill = 0
    bike_cost = (
        length
        + (climb_m / CostConfig.UPHILL_REFERENCE_M) * params.extra_km_per_uphill_100m * mpk
        + s_tier * params.extra_km_per_unpaved_km * length_km * mpk
        + r_tier * params.extra_km_per_main_road_km * length_km * mpk
    )
    rail_cost = length + params.extra_km_per_rail_km * length_km * mpk
    # Half the boarding charge per station edge: entry + exit sum to one full boarding.
    station_cost = length + 0.5 * params.extra_km_per_boarding * mpk

    is_station = mode == Mode.STATION
    assert bool(np.all(length[is_station] <= RailConfig.STATION_RADIUS_M)), "station link exceeds station radius"
    assert bool(np.all(length >= 0)), "edge length must be non-negative"
    cost: np.ndarray = np.select(
        [mode == Mode.BIKE, mode == Mode.RAIL, is_station], [bike_cost, rail_cost, station_cost], default=np.nan
    )
    assert not np.isnan(cost).any(), "unknown edge mode in cost array"
    assert bool(np.all(cost >= 0)), "edge cost must be non-negative"
    return cost
