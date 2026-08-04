"""STRICT bike-edge geometry validators — build-time invariants that fail LOUD on a corrupt graph.

Guards the two corruption classes that shipped bad graphs: sparse polylines that shortcut across streets,
and baked z that leaves the endpoint-elevation band. Run on the on-disk tables so the build gates itself.
"""

import numpy as np
import pandas as pd
from shapely import from_wkt

from bike_router.core.constants import BuildValidationConfig, Mode, Schema
from bike_router.core.geo import haversine_vec


def bike_edge_max_vertex_gap_m(*, geometry_wkt: str) -> float:
    """Largest consecutive-vertex great-circle gap (m) along a WKT LINESTRING (needs ≥ 2 vertices)."""
    coords = list(from_wkt(geometry_wkt).coords)
    assert len(coords) >= 2, f"bike edge polyline has < 2 vertices: {geometry_wkt!r}"
    xy = np.asarray([(c[0], c[1]) for c in coords], dtype=np.float64)
    return float(haversine_vec(lat_a=xy[:-1, 1], lon_a=xy[:-1, 0], lat_b=xy[1:, 1], lon_b=xy[1:, 0]).max())


def bike_edge_z_out_of_band_m(*, geometry_wkt: str, from_elev: float, to_elev: float) -> float:
    """Max metres any baked z exceeds the [min,max endpoint elevation] band (every vertex must carry z).

    A bike edge hugs terrain, so every vertex z must sit between its two node elevations (plus DEM
    slack); a z far outside means the polyline dips through a valley / over a hill it can't actually take.
    """
    coords = list(from_wkt(geometry_wkt).coords)
    assert coords and all(len(c) >= 3 for c in coords), f"bike edge polyline missing per-vertex z: {geometry_wkt!r}"
    zs = np.asarray([c[2] for c in coords], dtype=np.float64)
    lo, hi = min(from_elev, to_elev), max(from_elev, to_elev)
    over = np.maximum(zs - hi, 0.0)
    under = np.maximum(lo - zs, 0.0)
    return float(np.maximum(over, under).max())


def assert_bike_geometry_valid(*, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> None:
    """Fail LOUD if any BIKE edge violates the vertex-spacing or elevation-band invariant.

    Rail/station edges are exempt (trains legitimately tunnel/bridge; station links are straight). Runs on
    the whole built graph so a defective artifact never ships. Raises AssertionError naming the worst edge.
    """
    elev = {int(o): float(e) for o, e in zip(nodes_df[Schema.OSMID], nodes_df[Schema.ELEVATION_M], strict=True)}
    max_gap = BuildValidationConfig.MAX_VERTEX_SPACING_M
    band_margin = BuildValidationConfig.ELEV_BAND_MARGIN_M
    bike = edges_df[edges_df[Schema.MODE] == Mode.BIKE]
    worst_gap: tuple[float, tuple[int, int] | None] = (0.0, None)
    worst_band: tuple[float, tuple[int, int] | None] = (0.0, None)
    for row in bike.itertuples(index=False):
        wkt = getattr(row, Schema.GEOMETRY_WKT)
        assert isinstance(wkt, str), f"bike edge {(row.from_node, row.to_node)} has no geometry — every bike edge must"
        gap = bike_edge_max_vertex_gap_m(geometry_wkt=wkt)
        if gap > worst_gap[0]:
            worst_gap = (gap, (row.from_node, row.to_node))
        fe, te = elev[int(row.from_node)], elev[int(row.to_node)]
        band = bike_edge_z_out_of_band_m(geometry_wkt=wkt, from_elev=fe, to_elev=te)
        if band > worst_band[0]:
            worst_band = (band, (row.from_node, row.to_node))
    assert worst_gap[0] <= max_gap, (
        f"bike edge {worst_gap[1]} has a {worst_gap[0]:.0f} m vertex gap > {max_gap:.0f} m max — "
        f"geometry shortcuts across streets; densify the polyline in preprocessing"
    )
    assert worst_band[0] <= band_margin, (
        f"bike edge {worst_band[1]} has baked z {worst_band[0]:.0f} m outside its endpoint band "
        f"(> {band_margin:.0f} m margin) — it dips/climbs terrain a bike can't take; split the edge at that extremum"
    )
