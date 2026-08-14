"""STRICT bike-edge geometry validators — build-time invariants that fail LOUD on a corrupt graph.

Guards the two corruption classes that shipped bad graphs: sparse polylines that shortcut across streets,
and baked z that leaves the endpoint-elevation band. Run on the on-disk tables so the build gates itself.
"""

import logging

import numpy as np
import pandas as pd
from shapely import from_wkt, get_coordinates, get_num_coordinates

from bike_router.core.constants import BuildValidationConfig, Mode, Schema
from bike_router.core.geo import haversine_vec

logger = logging.getLogger(__name__)


def assert_no_long_straight_edges(*, edges_df: pd.DataFrame) -> None:
    """Fail LOUD if any edge longer than MAX_STRAIGHT_EDGE_M has only straight 2-point geometry.

    A long edge MUST trace real infrastructure (a densified polyline); a >1 km straight jump between two
    points is a corrupt shortcut that never follows the actual track/road. Runs on the whole built graph.
    """
    max_straight = BuildValidationConfig.MAX_STRAIGHT_EDGE_M
    wkt = edges_df[Schema.GEOMETRY_WKT].to_numpy()
    length_m = edges_df[Schema.LENGTH_M].to_numpy(dtype=np.float64)
    logger.info(f"validate: checking {len(edges_df)} edges for long straight jumps (> {max_straight:.0f} m) …")
    has_geom = np.array([isinstance(w, str) for w in wkt])
    n_vertices = np.zeros(len(wkt), dtype=np.int64)
    n_vertices[has_geom] = get_num_coordinates(from_wkt(wkt[has_geom]))
    # A long edge is invalid if it has NO geometry, or only its 2 endpoints (a straight line, no real trace).
    invalid = (length_m > max_straight) & (n_vertices <= 2)
    if invalid.any():
        i = int(np.where(invalid)[0][np.argmax(length_m[invalid])])
        raise AssertionError(
            f"edge {(int(edges_df[Schema.FROM_NODE].iloc[i]), int(edges_df[Schema.TO_NODE].iloc[i]))} is "
            f"{length_m[i] / 1000:.1f} km but has {n_vertices[i]} vertices — a >{max_straight:.0f} m edge must trace "
            f"real line geometry, not a straight jump"
        )
    logger.info(f"validate: OK — no straight edge exceeds {max_straight:.0f} m")


def assert_bike_geometry_valid(*, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> None:
    """Fail LOUD if any BIKE edge violates the vertex-spacing or elevation-band invariant.

    Rail/station edges are exempt (trains legitimately tunnel/bridge; station links are straight).
    """
    max_gap = BuildValidationConfig.MAX_VERTEX_SPACING_M
    band_margin = BuildValidationConfig.ELEV_BAND_MARGIN_M
    bike = edges_df[edges_df[Schema.MODE] == Mode.BIKE].reset_index(drop=True)
    if bike.empty:
        return  # nothing to validate (e.g. a rail-only synthetic frame)
    wkt = bike[Schema.GEOMETRY_WKT].to_numpy()
    assert all(isinstance(w, str) for w in wkt), "a bike edge has no geometry — every bike edge must"
    logger.info(f"validate: parsing {len(bike)} bike-edge geometries (vectorized) …")
    geoms = from_wkt(wkt)
    counts = get_num_coordinates(geoms)  # vertices per edge
    assert counts.min() >= 2, "a bike edge polyline has < 2 vertices"
    coords = get_coordinates(geoms, include_z=True)  # (total_vertices, 3) flat, edge-ordered
    assert not np.isnan(coords[:, 2]).any(), "a bike edge vertex is missing z"

    # Segment gaps: haversine between consecutive vertices, but NOT across an edge boundary. The last vertex
    # of each edge has no "next" in its own edge, so mask those cross-edge segments out before the per-edge max.
    logger.info("validate: checking vertex gaps + elevation bands …")
    seg_m = haversine_vec(lat_a=coords[:-1, 1], lon_a=coords[:-1, 0], lat_b=coords[1:, 1], lon_b=coords[1:, 0])
    edge_id = np.repeat(np.arange(len(bike)), counts)  # which edge each vertex belongs to
    same_edge = edge_id[:-1] == edge_id[1:]  # keep only within-edge segments
    gap_per_edge = np.zeros(len(bike))
    np.maximum.at(gap_per_edge, edge_id[:-1][same_edge], np.where(same_edge, seg_m, 0.0)[same_edge])
    worst_gap_i = int(gap_per_edge.argmax())

    # Elevation band: every vertex z must sit within [min,max endpoint elevation]; measure the worst excursion.
    elev = dict(
        zip(nodes_df[Schema.OSMID].to_numpy(), nodes_df[Schema.ELEVATION_M].to_numpy(dtype=np.float64), strict=True)
    )
    fe = bike[Schema.FROM_NODE].map(elev).to_numpy(dtype=np.float64)
    te = bike[Schema.TO_NODE].map(elev).to_numpy(dtype=np.float64)
    lo, hi = np.minimum(fe, te)[edge_id], np.maximum(fe, te)[edge_id]
    excursion = np.maximum(np.maximum(coords[:, 2] - hi, lo - coords[:, 2]), 0.0)
    band_per_edge = np.zeros(len(bike))
    np.maximum.at(band_per_edge, edge_id, excursion)
    worst_band_i = int(band_per_edge.argmax())

    assert gap_per_edge[worst_gap_i] <= max_gap, (
        f"bike edge {(int(bike[Schema.FROM_NODE][worst_gap_i]), int(bike[Schema.TO_NODE][worst_gap_i]))} has a "
        f"{gap_per_edge[worst_gap_i]:.0f} m vertex gap > {max_gap:.0f} m max — shortcuts across streets; densify it"
    )
    assert band_per_edge[worst_band_i] <= band_margin, (
        f"bike edge {(int(bike[Schema.FROM_NODE][worst_band_i]), int(bike[Schema.TO_NODE][worst_band_i]))} has baked z "
        f"{band_per_edge[worst_band_i]:.0f} m outside its endpoint band (> {band_margin:.0f} m) — split it at that extremum"
    )
    logger.info(
        f"validate: OK — worst gap {gap_per_edge[worst_gap_i]:.0f} m, worst band {band_per_edge[worst_band_i]:.0f} m"
    )
