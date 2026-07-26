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

import networkx as nx

from bike_router.constants import (
    CostConfig,
    GpxConfig,
    Mode,
    RailConfig,
    RoadConfig,
    RoutingParams,
    SurfaceConfig,
)


def _as_values(tag: object) -> list[str]:
    """Normalize an OSM tag (str | list | None) to a list of lowercased strings."""
    if tag is None:
        return []
    if isinstance(tag, list | tuple | set):
        return [str(value).lower() for value in tag]
    return [str(tag).lower()]


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


def edge_cost(
    *,
    mode: str,
    length: float,
    surface: object,
    highway: object,
    elev_source: float,
    elev_target: float,
    params: RoutingParams,
) -> float:
    """Total edge cost in metres, branching on travel mode.

    bike: length + uphill + unpaved + main-road penalties (terrain-aware).
    rail: per-km rail charge only (no terrain penalties; boarding lives on station edges).
    station: straight-line length + half the boarding charge (board + alight = full).
    """
    assert length >= 0, "edge length must be non-negative"
    length_km = length / GpxConfig.METERS_PER_KM

    if mode == Mode.RAIL:
        total = length + params.extra_km_per_rail_km * length_km * GpxConfig.METERS_PER_KM
    elif mode == Mode.STATION:
        # Half the boarding charge per station edge: entry + exit sum to one full boarding.
        assert length <= RailConfig.STATION_RADIUS_M, "station link exceeds station radius"
        total = length + 0.5 * params.extra_km_per_boarding * GpxConfig.METERS_PER_KM
    elif mode == Mode.BIKE:
        climb_m = max(elev_target - elev_source, 0.0)  # uphill only; downhill = 0
        uphill_penalty = (
            (climb_m / CostConfig.UPHILL_REFERENCE_M) * params.extra_km_per_uphill_100m * GpxConfig.METERS_PER_KM
        )
        unpaved_penalty = (
            surface_tier(surface=surface) * params.extra_km_per_unpaved_km * length_km * GpxConfig.METERS_PER_KM
        )
        main_road_penalty = (
            road_tier(highway=highway) * params.extra_km_per_main_road_km * length_km * GpxConfig.METERS_PER_KM
        )
        total = length + uphill_penalty + unpaved_penalty + main_road_penalty
    else:
        raise ValueError(f"unknown edge mode: {mode!r}")

    assert total >= 0, "edge cost must be non-negative"
    return total


def assign_edge_costs(graph: nx.MultiDiGraph, params: RoutingParams) -> None:
    """Store the total cost on every directed edge of the graph, in place.

    Requires node ``elevation`` (enrich_elevations) and edge ``length``/``mode``
    (baked at build time) — internal invariants, accessed strictly so a gap fails
    loud. ``surface``/``highway`` stay optional (genuine external OSM data).
    """
    for node_a, node_b, _key, data in graph.edges(keys=True, data=True):
        data[CostConfig.EDGE_COST] = edge_cost(
            mode=data["mode"],
            length=float(data["length"]),
            surface=data.get("surface"),  # OSM surface tag is genuinely optional
            highway=data.get("highway"),  # ditto highway (external OSM data)
            elev_source=float(graph.nodes[node_a]["elevation"]),
            elev_target=float(graph.nodes[node_b]["elevation"]),
            params=params,
        )
    assert graph.number_of_edges() > 0, "graph must have edges to cost"
