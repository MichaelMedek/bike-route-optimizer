"""Route geometry helpers.

``route_to_linestring`` stitches the node path into the full OSM geometry;
``select_waypoints`` reduces it to N significant points (Visvalingam-Whyatt) for
the Google Maps URL.
"""

import logging

import networkx as nx
import numpy as np
from shapely.geometry import LineString

from bike_router.track import cheapest_edge

logger = logging.getLogger(__name__)


def route_to_linestring(graph: nx.MultiDiGraph, node_path: list[int]) -> LineString:
    """Stitch a node path into the full lon/lat OSM geometry (x=lon, y=lat).

    For each consecutive node pair pick the cheapest parallel edge (matching A*)
    and use its stored ``geometry`` if present, else a straight segment between
    node coordinates.
    """
    assert len(node_path) >= 2, "route must have >= 2 nodes (distinct source/target)"
    coords: list[tuple[float, float]] = []
    for node_a, node_b in zip(node_path[:-1], node_path[1:], strict=True):
        data = cheapest_edge(edges=graph.get_edge_data(node_a, node_b))
        geometry = data.get("geometry")  # OSM geometry is genuinely optional
        if geometry is not None:
            segment = [(c[0], c[1]) for c in geometry.coords]  # drop any z → 2D lon/lat
        else:
            segment = [
                (graph.nodes[node_a]["x"], graph.nodes[node_a]["y"]),
                (graph.nodes[node_b]["x"], graph.nodes[node_b]["y"]),
            ]
        if coords and segment and coords[-1] == segment[0]:
            segment = segment[1:]  # avoid duplicating the shared vertex
        coords.extend(segment)
    return LineString(coords)


def select_waypoints(line: LineString, count: int = 10) -> list[tuple[float, float]]:
    """Reduce ``line`` to exactly ``count`` significant points, returned (lat, lon).

    Visvalingam-Whyatt: repeatedly drop the interior point with the smallest
    triangle area until ``count`` remain (endpoints kept). Lines with <= count
    points are padded by interpolation.
    """
    coords: list[tuple[float, float]] = [(c[0], c[1]) for c in line.coords]  # (lon, lat); ignore any z
    assert count >= 2, "need at least origin + destination waypoints"
    if len(coords) <= count:
        coords = _interpolate_to_n(coords=coords, count=count)
    else:
        coords = _visvalingam(coords=coords, count=count)
    assert len(coords) == count, "select_waypoints must return exactly `count` points"
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
    assert len(coords) >= count, "cannot reduce below the requested count"
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
