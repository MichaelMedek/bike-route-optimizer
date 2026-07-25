"""Routing: A* over the directed graph using the stored per-edge cost.

Each edge's ``custom_cost`` (cost.assign_edge_costs) is length plus non-negative
penalties, so the cheapest possible edge is its raw length. The A* heuristic is
therefore the plain great-circle distance to the target — it never overestimates
the true remaining cost, so it is admissible and A* returns the optimal path.
"""

from collections.abc import Callable
from typing import Any

import networkx as nx

from bike_router.constants import CostConfig
from bike_router.geo import haversine_distance_m

Heuristic = Callable[[int, int], float]


def _edge_cost(edges: dict[int, dict[str, Any]]) -> float:
    """Cheapest parallel-edge cost (NetworkX hands the weight fn all parallel edges)."""
    return min(float(data[CostConfig.EDGE_COST]) for data in edges.values())


def make_heuristic(graph: nx.MultiDiGraph) -> Heuristic:
    """Admissible A* heuristic: straight-line metres to target (cost floor = length)."""

    def heuristic(current: int, target: int) -> float:
        node_now, node_end = graph.nodes[current], graph.nodes[target]
        estimate = haversine_distance_m(
            lat_a=node_now["y"], lon_a=node_now["x"], lat_b=node_end["y"], lon_b=node_end["x"]
        )
        assert estimate >= 0, "heuristic estimate must be non-negative"
        return estimate

    return heuristic


def shortest_route(graph: nx.MultiDiGraph, source: int, target: int) -> list[int]:
    """Return the optimal node path from source to target under the stored cost.

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
        heuristic=make_heuristic(graph=graph),
        weight=lambda node_a, node_b, edges: _edge_cost(edges=edges),
    )
    assert path[0] == source and path[-1] == target, "A* path must start at source and end at target"
    return path
