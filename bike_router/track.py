"""Unified route-track builder — the single source of per-point time & elevation.

Walks the A* node path once, and for each traversed (cheapest) edge computes its
length, surface tier, linear grade and adaptive speed, accumulating ride time. The
GPX writer, the total-time/stats, and the elevation profile all derive from this
one structure, so the printed total time and the GPX end-timestamp agree by
construction.
"""

from dataclasses import dataclass
from typing import Any

import networkx as nx

from bike_router.constants import CostConfig, GpxConfig, SpeedConfig
from bike_router.cost import surface_tier
from bike_router.speed import effective_speed_kmh


@dataclass(frozen=True)
class TrackPoint:
    """One point along the route: position, elevation, and cumulative ride time."""

    lat: float
    lon: float
    elevation_m: float
    elapsed_s: float


@dataclass(frozen=True)
class Track:
    """The full traversed route: ordered points + rolled-up totals."""

    points: list[TrackPoint]
    distance_km: float
    duration_min: float
    ascent_m: float
    descent_m: float


def _cheapest_edge(edges: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """The parallel edge A* would traverse (lowest stored cost)."""
    best_key = min(edges, key=lambda key: float(edges[key][CostConfig.EDGE_COST]))
    return edges[best_key]


def build_track(graph: nx.MultiDiGraph, node_path: list[int]) -> Track:
    """Build the full route track with adaptive-speed timing from a node path."""
    assert len(node_path) >= 2, "route must have >= 2 nodes"

    first = graph.nodes[node_path[0]]
    points = [TrackPoint(lat=first["y"], lon=first["x"], elevation_m=float(first["elevation"]), elapsed_s=0.0)]
    total_m = ascent = descent = elapsed_s = 0.0

    for node_a, node_b in zip(node_path[:-1], node_path[1:], strict=True):
        data = _cheapest_edge(edges=graph.get_edge_data(node_a, node_b))
        length_m = float(data["length"])
        elev_a = float(graph.nodes[node_a]["elevation"])
        elev_b = float(graph.nodes[node_b]["elevation"])
        delta = elev_b - elev_a
        if delta > 0:
            ascent += delta
        else:
            descent += -delta

        grade = delta / length_m if length_m > 0 else 0.0
        speed_kmh = effective_speed_kmh(surface_tier=surface_tier(surface=data.get("surface")), grade=grade)
        speed_ms = speed_kmh * GpxConfig.METERS_PER_KM / GpxConfig.SECONDS_PER_HOUR
        elapsed_s += length_m / speed_ms
        total_m += length_m

        target = graph.nodes[node_b]
        points.append(TrackPoint(lat=target["y"], lon=target["x"], elevation_m=elev_b, elapsed_s=elapsed_s))

    assert total_m > 0, "route distance must be positive"
    assert elapsed_s > 0, "route duration must be positive"
    distance_km = total_m / GpxConfig.METERS_PER_KM
    duration_min = elapsed_s / GpxConfig.SECONDS_PER_HOUR * GpxConfig.MINUTES_PER_HOUR
    # sanity: average speed must sit between the walking floor and the best base speed
    avg_kmh = distance_km / (elapsed_s / GpxConfig.SECONDS_PER_HOUR)
    assert SpeedConfig.WALK_KMH - 1e-9 <= avg_kmh <= max(SpeedConfig.BASE_KMH_BY_TIER.values()) + 1e-9, (
        f"implausible average speed {avg_kmh:.1f} km/h"
    )
    return Track(points=points, distance_km=distance_km, duration_min=duration_min, ascent_m=ascent, descent_m=descent)
