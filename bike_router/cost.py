"""Per-edge cost in intuitive "extra kilometres".

The cost of a directed edge is its real length plus three user-controlled penalties,
all measured in metres so they add cleanly:

    cost = length
         + uphill_penalty     = (climb_m / 100) * extra_km_per_uphill_100m   * 1000
         + unpaved_penalty    = surface_tier * extra_km_per_unpaved_km       * length/1000 * 1000
         + main_road_penalty  = is_main_road * extra_km_per_main_road_km     * length/1000 * 1000

Each `extra_km_*` is how many extra virtual kilometres the rider will accept to
avoid one unit of the bad thing. All penalties are >= 0, so the cheapest possible
edge is pure distance — which keeps the A* great-circle heuristic admissible.

Because the graph is directed, only the uphill direction of a street is penalised
(the downhill direction has climb_m = 0), so uphill costs more (flat-preferring).
"""

import networkx as nx

from bike_router.constants import CostConfig, GpxConfig, RoadConfig, RoutingParams, SurfaceConfig


def _as_values(tag: object) -> list[str]:
    """Normalize an OSM tag (str | list | None) to a list of lowercased strings."""
    if tag is None:
        return []
    if isinstance(tag, list | tuple | set):
        return [str(value).lower() for value in tag]
    return [str(tag).lower()]


def surface_tier(surface: object) -> int:
    """Penalty tier for a surface tag: 0 good, 1 moderate, 2 heavy.

    Worst (highest) tier wins for list-valued tags. Unknown/untagged → DEFAULT_TIER.
    """
    tiers = [
        SurfaceConfig.SURFACE_TIER[value] for value in _as_values(tag=surface) if value in SurfaceConfig.SURFACE_TIER
    ]
    tier = max(tiers) if tiers else SurfaceConfig.DEFAULT_TIER
    assert tier in {0, 1, 2}, "surface tier must be 0, 1, or 2"
    return tier


def is_main_road(highway: object) -> bool:
    """True if any of the highway values is a penalised main road.

    Unknown/untagged highway (≈0% in practice) is treated as a main road.
    """
    values = _as_values(tag=highway)
    if not values:
        return True
    return any(value in RoadConfig.MAIN_ROADS for value in values)


def edge_cost(
    length: float, surface: object, highway: object, elev_source: float, elev_target: float, params: RoutingParams
) -> float:
    """Total edge cost in metres = length + uphill + unpaved + main-road penalties."""
    assert length >= 0, "edge length must be non-negative"

    climb_m = max(elev_target - elev_source, 0.0)  # uphill only; downhill = 0
    uphill_penalty = (
        (climb_m / CostConfig.UPHILL_REFERENCE_M) * params.extra_km_per_uphill_100m * GpxConfig.METERS_PER_KM
    )

    length_km = length / GpxConfig.METERS_PER_KM
    unpaved_penalty = (
        surface_tier(surface=surface) * params.extra_km_per_unpaved_km * length_km * GpxConfig.METERS_PER_KM
    )
    main_road_penalty = (
        (1 if is_main_road(highway=highway) else 0)
        * params.extra_km_per_main_road_km
        * length_km
        * GpxConfig.METERS_PER_KM
    )

    total = length + uphill_penalty + unpaved_penalty + main_road_penalty
    assert total >= 0, "edge cost must be non-negative"
    return total


def assign_edge_costs(graph: nx.MultiDiGraph, params: RoutingParams) -> None:
    """Store the total cost on every directed edge of the graph, in place.

    Requires node ``elevation`` (enrich_elevations) and edge ``length`` (OSMnx, in
    metres) — both internal invariants, accessed strictly so a gap fails loud.
    """
    for node_a, node_b, _key, data in graph.edges(keys=True, data=True):
        data[CostConfig.EDGE_COST] = edge_cost(
            length=float(data["length"]),
            surface=data.get("surface"),  # OSM surface tag is genuinely optional
            highway=data.get("highway"),  # ditto highway (external OSM data)
            elev_source=float(graph.nodes[node_a]["elevation"]),
            elev_target=float(graph.nodes[node_b]["elevation"]),
            params=params,
        )
    assert graph.number_of_edges() > 0, "graph must have edges to cost"
