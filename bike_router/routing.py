"""Routing: A* over the directed graph for a given RouteProfile.

Each edge carries three cost components (cost.py). For a profile we minimize
their weighted sum via a NetworkX weight callable. The A* heuristic estimates
remaining cost as (min per-metre weighted cost) × straight-line distance — never
an overestimate, so it stays admissible → A* returns the optimal path.
"""

from collections.abc import Callable

import networkx as nx

from bike_router.constants import CostConfig, RouteProfile
from bike_router.cost import combine, edge_stored_components
from bike_router.geo import haversine_distance_m

Heuristic = Callable[[int, int], float]
WeightFn = Callable[[int, int, dict[int, dict[str, object]]], float]


def weighted_cost_fn(profile: RouteProfile) -> WeightFn:
    """NetworkX MultiDiGraph weight callable: cheapest parallel edge's weighted cost.

    NetworkX hands the weight function the dict of ALL parallel edges (keyed by
    edge key), so we combine each and return the minimum.
    """

    def weight(node_a: int, node_b: int, edges: dict[int, dict[str, object]]) -> float:
        return min(combine(components=edge_stored_components(data=data), profile=profile) for data in edges.values())

    return weight


def _min_per_metre_cost(profile: RouteProfile) -> float:
    """Lowest achievable weighted cost per metre (perfect flat asphalt cycleway).

    dist contributes w_dist*1; surface contributes w_surface*(MIN_SF_RF-1) which
    is negative for a cycleway, so the floor can dip below w_dist; elevation adds
    ≥0. Clamped at 0 so the heuristic never goes negative (stays admissible).
    """
    floor = profile.w_dist + profile.w_surface * (CostConfig.MIN_SF_RF - 1.0)
    scale = max(floor, 0.0)
    assert scale >= 0.0, "heuristic scale must be non-negative for admissibility"
    return scale


def make_heuristic(graph: nx.MultiDiGraph, profile: RouteProfile) -> Heuristic:
    """Build the admissible A* heuristic for this profile (scaled great-circle)."""
    scale = _min_per_metre_cost(profile=profile)

    def heuristic(current: int, target: int) -> float:
        node_now, node_end = graph.nodes[current], graph.nodes[target]
        estimate = scale * haversine_distance_m(
            lat_a=node_now["y"], lon_a=node_now["x"], lat_b=node_end["y"], lon_b=node_end["x"]
        )
        assert estimate >= 0, "heuristic estimate must be non-negative"
        return estimate

    return heuristic


def shortest_route(graph: nx.MultiDiGraph, source: int, target: int, profile: RouteProfile) -> list[int]:
    """Return the optimal node path from source to target for ``profile``.

    Raises:
        networkx.NetworkXNoPath: if no path exists (should not happen on the
            strongly-connected core).
    """
    assert source in graph, "source node must be in the graph"
    assert target in graph, "target node must be in the graph"
    path: list[int] = nx.astar_path(
        graph,
        source=source,
        target=target,
        heuristic=make_heuristic(graph=graph, profile=profile),
        weight=weighted_cost_fn(profile=profile),
    )
    assert path[0] == source and path[-1] == target, "A* path must start at source and end at target"
    return path
