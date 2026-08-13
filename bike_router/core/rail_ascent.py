"""Station extrema (local max/min rail stations) + "good ascent" rail legs between them.

Maxima (green, click→Start) and minima (red, click→End) share ONE dominance/prominence pass; a
min→max rail path through no other extremum with a steep-enough direct-line gain is a purple ascent.
"""

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bike_router.core.constants import GpxConfig, GraphConfig, Mode, NodeType, RailConfig, Schema
from bike_router.core.geo import haversine_distance_m, haversine_vec
from bike_router.core.graph_store import NODE_COLS, oriented_polyline, read_tiles

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RailAscent:
    """One "good ascent": a rail leg from a low (min) station up to a high (max) station.

    ``points`` is the real serpentining track polyline ``[lon, lat, z]`` at terrain height (the UI
    lifts it); ``grade`` is the DIRECT-LINE rise/run, not the track grade.
    """

    low_name: str
    high_name: str
    gain_m: float
    direct_km: float
    grade: float
    points: list[list[float]]


@dataclass(frozen=True)
class StationExtrema:
    """The whole 🚞 payload from ONE graph scan: green maxima, red minima, purple ascent legs."""

    maxima: list[tuple[float, float, float, str]]
    minima: list[tuple[float, float, float, str]]
    ascents: list[RailAscent]


def load_stations(*, graph_dir: Path) -> pd.DataFrame:
    """Every named rail station across the coverage area (osmid, lat, lon, elevation_m, name)."""
    nodes_df = read_tiles(directory=graph_dir / GraphConfig.NODES_SUBDIR, columns=NODE_COLS, tiles=None, filters=None)
    stations = nodes_df[(nodes_df["node_type"] == NodeType.RAIL) & nodes_df["station_name"].notna()].reset_index(
        drop=True
    )
    assert not stations.empty, "no station found"
    return stations


def extremum_stations(*, stations_df: pd.DataFrame, want_high: bool) -> pd.DataFrame:
    """The prominent local-extremum stations — the ONE dominance/prominence pass for BOTH max and min.

    ``want_high`` picks maxima (highest within the radius, rising ≥ prominence above the lowest) vs
    minima (lowest, falling ≥ prominence below the highest). Sorted highest-first (max) / lowest-first (min).
    """
    assert not stations_df.empty, "stations frame is empty"
    lats = stations_df["lat"].to_numpy(dtype=float)
    lons = stations_df["lon"].to_numpy(dtype=float)
    elevs = stations_df[Schema.ELEVATION_M].to_numpy(dtype=float)
    keep: list[int] = []
    for i in range(len(stations_df)):
        dists_km = haversine_vec(lat_a=lats[i], lon_a=lons[i], lat_b=lats, lon_b=lons) / GpxConfig.METERS_PER_KM
        near = elevs[dists_km <= RailConfig.TOP_STATION_DOMINANCE_KM]
        if want_high:
            dominant = elevs[i] >= near.max()  # Dominanz: highest station within the radius
            prominent = elevs[i] - near.min() >= RailConfig.TOP_STATION_PROMINENCE_M  # Schartenhöhe up
        else:
            dominant = elevs[i] <= near.min()  # inverse Dominanz: lowest station within the radius
            prominent = near.max() - elevs[i] >= RailConfig.TOP_STATION_PROMINENCE_M  # relief above it
        if dominant and prominent:
            keep.append(i)
    return stations_df.iloc[keep].sort_values(Schema.ELEVATION_M, ascending=not want_high).reset_index(drop=True)


def station_markers(*, stations_df: pd.DataFrame) -> list[tuple[float, float, float, str]]:
    """(lat, lon, elevation_m, name) marker tuples for a station DataFrame — the map-layer shape."""
    return [
        (float(row.lat), float(row.lon), float(row.elevation_m), str(row.station_name))
        for row in stations_df.itertuples(index=False)
    ]


def rail_adjacency(*, edges_df: pd.DataFrame) -> dict[int, list[int]]:
    """Map each osmid → its rail-neighbour osmids from a light ``(from_node, to_node)`` rail-edge frame.

    Rail edges are bidirectional (both directed rows present), so this reads as undirected reachability.
    """
    adjacency: dict[int, list[int]] = {}
    for u, v in zip(edges_df["from_node"].astype(int), edges_df["to_node"].astype(int), strict=True):
        adjacency.setdefault(int(u), []).append(int(v))
    return adjacency


def bfs_extremum_pairs(
    *, adjacency: dict[int, list[int]], sources: set[int], absorbing: set[int], targets: set[int]
) -> list[tuple[int, int, list[int]]]:
    """(source, target, node_path) for every target reached from a source through NO absorbing node.

    BFS from each source; a reached extremum (in ``absorbing``) is recorded but not expanded, so a path
    passing through another min/max is excluded. Traverse sources (maxima) only → each pair once.
    """
    results: list[tuple[int, int, list[int]]] = []
    for source in sorted(sources):
        visited = {source}
        pred = {source: source}
        queue = deque([source])
        while queue:
            cur = queue.popleft()
            for nxt in adjacency.get(cur, ()):
                if nxt in visited:
                    continue
                visited.add(nxt)
                pred[nxt] = cur
                if nxt in targets:
                    path = [nxt]
                    while path[-1] != source:
                        path.append(pred[path[-1]])
                    results.append((source, nxt, list(reversed(path))))
                if nxt not in absorbing:  # don't route THROUGH another extremum
                    queue.append(nxt)
    return results


def ascent_grade(*, gain_m: float, direct_km: float) -> float:
    """DIRECT-LINE rise/run for a min→max leg — gain over the straight distance, not the track length."""
    assert direct_km > 0, "direct-line distance must be positive"
    return gain_m / (direct_km * GpxConfig.METERS_PER_KM)


def interpolate_z_by_distance(*, points_2d: list[tuple[float, float]], z_start: float, z_end: float) -> list[float]:
    """Linear z per vertex by cumulative great-circle distance — a schematic climb from z_start to z_end."""
    assert len(points_2d) >= 2, "need at least two points to interpolate z"
    lons = np.array([lon for lon, _lat in points_2d], dtype=np.float64)
    lats = np.array([lat for _lon, lat in points_2d], dtype=np.float64)
    seg = haversine_vec(lat_a=lats[:-1], lon_a=lons[:-1], lat_b=lats[1:], lon_b=lons[1:])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    assert total > 0, "polyline must have positive length"
    return [float(z_start + (c / total) * (z_end - z_start)) for c in cum]


def ascent_polyline(
    *, node_seq: list[tuple[int, float, float, float]], wkt_by_hop: dict[tuple[int, int], str]
) -> list[list[float]]:
    """Concatenated ``[lon, lat, z]`` track polyline along a low→high node sequence.

    Each hop uses its real oriented rail geometry; a geometry-less access hop falls back to a straight
    endpoint segment. Seam vertices are de-duped; z is interpolated end-to-end (see interpolate_z_by_distance).
    """
    points_2d: list[tuple[float, float]] = []
    for (a_id, a_lon, a_lat, _a_e), (b_id, b_lon, b_lat, _b_e) in zip(node_seq[:-1], node_seq[1:], strict=True):
        wkt = wkt_by_hop.get((a_id, b_id)) or wkt_by_hop.get((b_id, a_id))
        coords, _zs = oriented_polyline(wkt=wkt, start_lon=a_lon, start_lat=a_lat)
        if coords is None:
            coords = [(a_lon, a_lat), (b_lon, b_lat)]
        points_2d.extend(coords[1:] if points_2d else coords)
    zs = interpolate_z_by_distance(points_2d=points_2d, z_start=node_seq[0][3], z_end=node_seq[-1][3])
    return [[lon, lat, z] for (lon, lat), z in zip(points_2d, zs, strict=True)]


def rail_ascents(
    *,
    graph_dir: Path,
    stations_df: pd.DataFrame,
    maxima_df: pd.DataFrame,
    minima_df: pd.DataFrame,
    grade_threshold: float,
) -> list[RailAscent]:
    """Purple "good ascent" legs: min→max rail paths (no extremum between) over the grade threshold.

    Two-phase like load_path_edges: a light rail-edge scan drives BFS + grade filter, then geometry is
    re-read only for the surviving paths' hops so the hot loop stays lean.
    """
    max_ids = set(maxima_df["osmid"].astype(int))
    min_ids = set(minima_df["osmid"].astype(int))
    if not max_ids or not min_ids:
        return []
    edges_lite = read_tiles(
        directory=graph_dir / GraphConfig.EDGES_SUBDIR,
        columns=[Schema.FROM_NODE, Schema.TO_NODE],
        tiles=None,
        filters=[(Schema.MODE, Schema.FILTER_IN, [Mode.RAIL])],
    )
    pairs = bfs_extremum_pairs(
        adjacency=rail_adjacency(edges_df=edges_lite), sources=max_ids, absorbing=max_ids | min_ids, targets=min_ids
    )
    station_by_id = {int(row.osmid): row for row in stations_df.itertuples(index=False)}
    kept: list[tuple[int, int, list[int], float, float, float]] = []
    for max_id, min_id, node_path in pairs:
        high, low = station_by_id[max_id], station_by_id[min_id]
        gain_m = float(high.elevation_m - low.elevation_m)
        direct_km = (
            haversine_distance_m(lat_a=low.lat, lon_a=low.lon, lat_b=high.lat, lon_b=high.lon) / GpxConfig.METERS_PER_KM
        )
        grade = ascent_grade(gain_m=gain_m, direct_km=direct_km)
        if grade >= grade_threshold:
            kept.append((max_id, min_id, node_path, gain_m, direct_km, grade))
    if not kept:
        return []
    all_ids = list({osmid for _mx, _mn, path, _g, _d, _gr in kept for osmid in path})
    nodes_df = read_tiles(
        directory=graph_dir / GraphConfig.NODES_SUBDIR,
        columns=NODE_COLS,
        tiles=None,
        filters=[(Schema.OSMID, Schema.FILTER_IN, all_ids)],
    ).set_index(Schema.OSMID)
    geom_edges = read_tiles(
        directory=graph_dir / GraphConfig.EDGES_SUBDIR,
        columns=[Schema.FROM_NODE, Schema.TO_NODE, Schema.GEOMETRY_WKT],
        tiles=None,
        filters=[(Schema.MODE, Schema.FILTER_IN, [Mode.RAIL]), (Schema.FROM_NODE, Schema.FILTER_IN, all_ids)],
    )
    wkt_by_hop = {
        (int(f), int(t)): w
        for f, t, w in zip(geom_edges["from_node"], geom_edges["to_node"], geom_edges["geometry_wkt"], strict=True)
        if isinstance(w, str)
    }
    ascents: list[RailAscent] = []
    for max_id, min_id, node_path, gain_m, direct_km, grade in kept:
        node_seq = [
            (
                nid,
                float(nodes_df.loc[nid, Schema.LON]),
                float(nodes_df.loc[nid, Schema.LAT]),
                float(nodes_df.loc[nid, Schema.ELEVATION_M]),
            )
            for nid in reversed(node_path)  # BFS path is max→min; reverse to low→high so z climbs
        ]
        ascents.append(
            RailAscent(
                low_name=str(station_by_id[min_id].station_name),
                high_name=str(station_by_id[max_id].station_name),
                gain_m=gain_m,
                direct_km=direct_km,
                grade=grade,
                points=ascent_polyline(node_seq=node_seq, wkt_by_hop=wkt_by_hop),
            )
        )
    return ascents


def station_extrema(*, graph_dir: Path, grade_threshold: float) -> StationExtrema:
    """The full 🚞 payload from ONE scan: green maxima + red minima markers + purple ascent legs."""
    stations_df = load_stations(graph_dir=graph_dir)
    maxima_df = extremum_stations(stations_df=stations_df, want_high=True)
    minima_df = extremum_stations(stations_df=stations_df, want_high=False)
    ascents = rail_ascents(
        graph_dir=graph_dir,
        stations_df=stations_df,
        maxima_df=maxima_df,
        minima_df=minima_df,
        grade_threshold=grade_threshold,
    )
    return StationExtrema(
        maxima=station_markers(stations_df=maxima_df),
        minima=station_markers(stations_df=minima_df),
        ascents=ascents,
    )
