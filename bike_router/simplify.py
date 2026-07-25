"""Route geometry helpers.

Two distinct jobs, deliberately kept apart:

* ``route_to_linestring`` — stitch the node path into the FULL OSM geometry
  (every real vertex). This is what the GPX track and debug PNG follow.
* ``simplify_track`` — Douglas-Peucker-simplify that full track for file size:
  drop near-collinear points on straights, keep sharp turns. Shape is preserved.
* ``select_waypoints`` — reduce to exactly N perceptually-significant points for
  the Google Maps URL, via Visvalingam-Whyatt effective-area ranking.
"""

import logging

import networkx as nx
import numpy as np
from shapely.geometry import LineString

from bike_router.constants import GeoConfig, RouteProfile, SimplifyConfig
from bike_router.cost import combine, edge_stored_components

logger = logging.getLogger(__name__)


def route_to_linestring(graph: nx.MultiDiGraph, node_path: list[int], profile: RouteProfile) -> LineString:
    """Stitch a node path into the full lon/lat OSM geometry (x=lon, y=lat).

    For each consecutive node pair pick the profile's min-cost parallel edge and
    use its stored ``geometry`` if present, else a straight segment between node
    coordinates.
    """
    assert len(node_path) >= 2, "route must have >= 2 nodes (distinct source/target)"
    coords: list[tuple[float, float]] = []
    for node_a, node_b in zip(node_path[:-1], node_path[1:], strict=True):
        edges = graph.get_edge_data(node_a, node_b)
        best_key = min(
            edges, key=lambda key: combine(components=edge_stored_components(data=edges[key]), profile=profile)
        )
        data = edges[best_key]
        geometry = data.get("geometry")  # OSM geometry is genuinely optional
        if geometry is not None:
            segment = list(geometry.coords)
        else:
            segment = [
                (graph.nodes[node_a]["x"], graph.nodes[node_a]["y"]),
                (graph.nodes[node_b]["x"], graph.nodes[node_b]["y"]),
            ]
        if coords and segment and coords[-1] == segment[0]:
            segment = segment[1:]  # avoid duplicating the shared vertex
        coords.extend(segment)
    return LineString(coords)


def simplify_track(line: LineString) -> list[tuple[float, float]]:
    """Douglas-Peucker-simplify the full route track (returns (lon, lat) points).

    RDP drops points within TRACK_TOLERANCE_M of the local chord (straights) while
    keeping points far from it (sharp turns) — one tolerance controls smoothness.
    The tolerance is metres, converted to degrees via the equator scale (shapely
    simplifies in coordinate units).
    """
    tolerance_deg = SimplifyConfig.TRACK_TOLERANCE_M / GeoConfig.METERS_PER_DEGREE_EQUATOR
    simplified = line.simplify(tolerance_deg, preserve_topology=False)
    return list(simplified.coords)


def select_waypoints(line: LineString, count: int = 10) -> list[tuple[float, float]]:
    """Reduce ``line`` to exactly ``count`` significant points, returned (lat, lon).

    Uses Visvalingam-Whyatt: repeatedly drop the interior point whose triangle
    (with its two current neighbours) has the smallest area, until ``count`` remain.
    Endpoints are always kept. If the line has <= count points it is padded by
    interpolation so callers always receive exactly ``count``.
    """
    coords = list(line.coords)  # (lon, lat)
    if len(coords) <= count:
        coords = _interpolate_to_n(coords=coords, count=count)
    else:
        coords = _visvalingam(coords=coords, count=count)
    return [(lat, lon) for lon, lat in coords]


def _triangle_area(point_a: tuple[float, float], point_b: tuple[float, float], point_c: tuple[float, float]) -> float:
    """Absolute area of the triangle a-b-c (planar; degrees are fine for ranking)."""
    return (
        abs(
            (point_b[0] - point_a[0]) * (point_c[1] - point_a[1])
            - (point_c[0] - point_a[0]) * (point_b[1] - point_a[1])
        )
        / 2.0
    )


def _visvalingam(coords: list[tuple[float, float]], count: int) -> list[tuple[float, float]]:
    """Drop the smallest-effective-area interior point until ``count`` remain."""
    survivors = list(coords)
    while len(survivors) > count:
        smallest_index = 1
        smallest_area = float("inf")
        for index in range(1, len(survivors) - 1):
            area = _triangle_area(point_a=survivors[index - 1], point_b=survivors[index], point_c=survivors[index + 1])
            if area < smallest_area:
                smallest_area, smallest_index = area, index
        survivors.pop(smallest_index)
    return survivors


def _interpolate_to_n(coords: list[tuple[float, float]], count: int) -> list[tuple[float, float]]:
    """Resample the polyline to exactly ``count`` points evenly spaced by arc length."""
    if len(coords) == 1:
        return [coords[0]] * count
    points = np.asarray(coords, dtype=np.float64)
    seg_lengths = np.sqrt(((points[1:] - points[:-1]) ** 2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cumulative[-1]
    if total == 0:
        return [tuple(points[0])] * count
    targets = np.linspace(0.0, total, count)
    interp_lons = np.interp(targets, cumulative, points[:, 0])
    interp_lats = np.interp(targets, cumulative, points[:, 1])
    return [(float(lon), float(lat)) for lon, lat in zip(interp_lons, interp_lats, strict=True)]
