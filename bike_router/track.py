"""Unified route-track builder — the single source of per-point time & elevation.

Walks the A* node path once, computing per-edge length, grade and adaptive speed
while accumulating ride time. GPX, stats, and elevation profile all derive from this
one structure, so total time and GPX end-timestamp agree by construction.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import networkx as nx

from bike_router.constants import CostConfig, GpxConfig, Mode, RailConfig, SpeedConfig
from bike_router.cost import surface_tier
from bike_router.geo import haversine_distance_m
from bike_router.speed import effective_speed_kmh, kmh_to_ms


@dataclass(frozen=True)
class TrackPoint:
    """One point along the route: position, elevation, cumulative time, and the
    travel mode of the edge arriving at it (start point takes the first edge's mode).
    """

    lat: float
    lon: float
    elevation_m: float
    elapsed_s: float
    mode: str


@dataclass(frozen=True)
class Track:
    """The full traversed route: ordered points + rolled-up totals."""

    points: list[TrackPoint]
    distance_km: float
    duration_min: float
    ascent_m: float
    descent_m: float


def cheapest_edge(edges: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """The parallel edge A* would traverse (lowest stored cost)."""
    best_key = min(edges, key=lambda key: float(edges[key][CostConfig.EDGE_COST]))
    return edges[best_key]


def iter_route_edges(graph: nx.MultiDiGraph, node_path: list[int]) -> Iterator[tuple[int, int, dict[str, Any]]]:
    """Yield ``(node_a, node_b, cheapest_edge_data)`` for each hop on the A* path."""
    for node_a, node_b in zip(node_path[:-1], node_path[1:], strict=True):
        yield node_a, node_b, cheapest_edge(edges=graph.get_edge_data(node_a, node_b))


def build_track(graph: nx.MultiDiGraph, node_path: list[int]) -> Track:
    """Build the full route track with adaptive-speed timing from a node path.

    Bike edges use the surface/grade speed model and count toward ascent/descent; rail
    edges ride at RAIL_SPEED_KMH; boarding (a transfer entering a station) adds
    BOARDING_WAIT_S once. All leg times are DERIVED from length + rail constants.
    """
    assert len(node_path) >= 2, "route must have >= 2 nodes"

    first = graph.nodes[node_path[0]]
    first_mode = str(cheapest_edge(edges=graph.get_edge_data(node_path[0], node_path[1]))["mode"])
    points = [
        TrackPoint(
            lat=first["y"], lon=first["x"], elevation_m=float(first["elevation"]), elapsed_s=0.0, mode=first_mode
        )
    ]
    total_m = ascent = descent = elapsed_s = 0.0
    bike_m = bike_s = 0.0  # only bike legs feed the avg-speed assert (rail is far faster)
    rail_speed_ms = kmh_to_ms(kmh=RailConfig.RAIL_SPEED_KMH)

    for node_a, node_b in zip(node_path[:-1], node_path[1:], strict=True):
        data = cheapest_edge(edges=graph.get_edge_data(node_a, node_b))
        length_m = float(data["length"])
        elev_a = float(graph.nodes[node_a]["elevation"])
        elev_b = float(graph.nodes[node_b]["elevation"])
        delta = elev_b - elev_a

        if data["mode"] == Mode.BIKE:
            if delta > 0:
                ascent += delta
            else:
                descent += -delta
            grade = delta / length_m if length_m > 0 else 0.0
            speed_kmh = effective_speed_kmh(surface_tier=surface_tier(surface=data.get("surface")), grade=grade)
            speed_ms = kmh_to_ms(kmh=speed_kmh)
            leg_s = length_m / speed_ms
            bike_m += length_m
            bike_s += leg_s
        elif data["mode"] == Mode.RAIL:
            leg_s = length_m / rail_speed_ms  # train ride time, derived from length
        else:  # Mode.TRANSFER — boarding (entering a station) waits; alighting is free
            boarding = bool(graph.nodes[node_b].get("is_station"))
            leg_s = RailConfig.BOARDING_WAIT_S if boarding else 0.0
        elapsed_s += leg_s
        total_m += length_m

        target = graph.nodes[node_b]
        points.append(
            TrackPoint(
                lat=target["y"], lon=target["x"], elevation_m=elev_b, elapsed_s=elapsed_s, mode=str(data["mode"])
            )
        )

    assert total_m > 0, "route distance must be positive"
    assert elapsed_s > 0, "route duration must be positive"
    distance_km = total_m / GpxConfig.METERS_PER_KM
    duration_min = elapsed_s / GpxConfig.SECONDS_PER_HOUR * GpxConfig.MINUTES_PER_HOUR
    # sanity: average BIKE speed must sit between the walking floor and the best base
    # speed (rail legs are excluded — 80 km/h would trip it).
    if bike_s > 0:
        avg_kmh = (bike_m / GpxConfig.METERS_PER_KM) / (bike_s / GpxConfig.SECONDS_PER_HOUR)
        assert SpeedConfig.WALK_KMH - 1e-9 <= avg_kmh <= max(SpeedConfig.BASE_KMH_BY_TIER.values()) + 1e-9, (
            f"implausible average speed {avg_kmh:.1f} km/h"
        )
    return Track(points=points, distance_km=distance_km, duration_min=duration_min, ascent_m=ascent, descent_m=descent)


def densify_track(graph: nx.MultiDiGraph, node_path: list[int], track: Track) -> Track:
    """Expand the node-level track into the full 3D road polyline (no DEM at inference).

    Walks each edge's baked 3D geometry so the profile follows the true road; rail/
    transfer hops contribute a straight segment. Each leg's time is spread by along-edge
    distance and ascent/descent are recomputed from the baked vertex elevations.
    """
    assert len(node_path) >= 2, "route must have >= 2 nodes to densify"
    assert len(track.points) == len(node_path), "track points must align with node path"
    out: list[TrackPoint] = []

    for index, (node_a, node_b) in enumerate(zip(node_path[:-1], node_path[1:], strict=True)):
        data = cheapest_edge(edges=graph.get_edge_data(node_a, node_b))
        mode = str(data["mode"])
        verts = _edge_vertices_3d(graph=graph, node_a=node_a, node_b=node_b, data=data)
        t_start, t_end = track.points[index].elapsed_s, track.points[index + 1].elapsed_s
        seg_lengths = [
            haversine_distance_m(lat_a=verts[i][1], lon_a=verts[i][0], lat_b=verts[i + 1][1], lon_b=verts[i + 1][0])
            for i in range(len(verts) - 1)
        ]
        total = sum(seg_lengths) or 1.0
        cum = 0.0
        last_leg = index == len(node_path) - 2
        stop = len(verts) if last_leg else len(verts) - 1  # avoid duplicating shared node vertex
        for i in range(stop):
            lon, lat, elev = verts[i]
            out.append(
                TrackPoint(
                    lat=lat,
                    lon=lon,
                    elevation_m=elev,
                    elapsed_s=t_start + (t_end - t_start) * (cum / total),
                    mode=mode,
                )
            )
            if i < len(seg_lengths):
                cum += seg_lengths[i]

    # Ascent/descent come from build_track's node-to-node elevations.
    return Track(
        points=out,
        distance_km=track.distance_km,
        duration_min=track.duration_min,
        ascent_m=track.ascent_m,
        descent_m=track.descent_m,
    )


def _edge_vertices_3d(
    graph: nx.MultiDiGraph, node_a: int, node_b: int, data: dict[str, Any]
) -> list[tuple[float, float, float]]:
    """(lon, lat, elev) vertices of edge a→b from its baked 3D geometry.

    Bike edges carry a baked 3D LineString; rail/transfer hops have no geometry, so
    fall back to a straight segment at the two node elevations.
    """
    geom = data.get("geometry")
    ea, eb = float(graph.nodes[node_a]["elevation"]), float(graph.nodes[node_b]["elevation"])
    if geom is None:
        return [
            (graph.nodes[node_a]["x"], graph.nodes[node_a]["y"], ea),
            (graph.nodes[node_b]["x"], graph.nodes[node_b]["y"], eb),
        ]
    coords = [(c[0], c[1], c[2] if len(c) > 2 else ea) for c in geom.coords]
    # Orient a→b by matching the first vertex to node_a's coords.
    ax, ay = graph.nodes[node_a]["x"], graph.nodes[node_a]["y"]
    if abs(coords[0][0] - ax) + abs(coords[0][1] - ay) > abs(coords[-1][0] - ax) + abs(coords[-1][1] - ay):
        coords = coords[::-1]
    return coords
