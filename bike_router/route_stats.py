"""Per-route summary stats: distance, ride time, total ascent/descent.

Computed from the traversed edges (real ``length`` in metres) and the node
elevations, so the numbers match what the GPX/heatmap show.
"""

from dataclasses import dataclass

import networkx as nx

from bike_router.constants import GpxConfig, RouteProfile
from bike_router.cost import combine, edge_stored_components


@dataclass(frozen=True)
class RouteStats:
    """Summary of one computed route."""

    distance_km: float
    duration_min: float
    ascent_m: float
    descent_m: float


def route_stats(graph: nx.MultiDiGraph, node_path: list[int], profile: RouteProfile) -> RouteStats:
    """Distance / time / ascent / descent for ``node_path`` under ``profile``.

    Distance sums the traversed (profile-cheapest) edge lengths; ascent/descent
    sum positive/negative node-elevation deltas along the path.
    """
    total_m = 0.0
    for node_a, node_b in zip(node_path[:-1], node_path[1:], strict=True):
        edges = graph.get_edge_data(node_a, node_b)
        best_key = min(
            edges, key=lambda key: combine(components=edge_stored_components(data=edges[key]), profile=profile)
        )
        total_m += float(edges[best_key]["length"])

    ascent = descent = 0.0
    for node_a, node_b in zip(node_path[:-1], node_path[1:], strict=True):
        delta = float(graph.nodes[node_b]["elevation"]) - float(graph.nodes[node_a]["elevation"])
        if delta > 0:
            ascent += delta
        else:
            descent += -delta

    distance_km = total_m / GpxConfig.METERS_PER_KM
    duration_min = (distance_km / GpxConfig.SPEED_KMH) * GpxConfig.MINUTES_PER_HOUR
    return RouteStats(distance_km=distance_km, duration_min=duration_min, ascent_m=ascent, descent_m=descent)
