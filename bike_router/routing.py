"""Routing: A* over the directed graph using the stored per-edge cost.

The cheapest edge equals its raw length, so a great-circle heuristic never
overestimates — admissible, giving optimal paths. ``custom_cost`` is passed as a
STRING weight so NetworkX does a fast C-level lookup, auto-picking cheapest parallel edges.
"""

from collections.abc import Callable

import networkx as nx

from bike_router.constants import CostConfig
from bike_router.geo import haversine_distance_m

Heuristic = Callable[[int, int], float]


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
        weight=CostConfig.EDGE_COST,  # string attr → C-level lookup, min over parallel edges
    )
    assert path[0] == source and path[-1] == target, "A* path must start at source and end at target"
    return path
