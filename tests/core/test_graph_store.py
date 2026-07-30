"""graph_store tests — tile grid math, windowed corridor load, path re-read, snapping.

One test_<fn> per production symbol (exact-name mirror). Reads the committed FIXTURE_ROUNDTRIP_STORE
(no build stack), then exercises the runtime read side end to end. The graph_writer symbols this file
used to also cover now live in tests/preprocessing/test_graph_writer.py.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from shapely.geometry import box

from bike_router.core import graph_store
from bike_router.core.constants import GraphConfig, Mode, NodeType
from bike_router.core.errors import OutOfCoverageError
from bike_router.core.graph_store import (
    _covering_tiles,
    _intersecting_tiles,
    _load_layer,
    _oriented_geometry,
    _read_tiles,
    _select_path_edges,
    _str_or_none,
    _tile_name,
    download_graph_from_hf,
    load_meta,
    load_path_edges,
    load_route_tables,
    snap_to_node,
    tile_index,
)
from bike_router.core.route_path import RouteNode
from tests.conftest import DEFAULT_PARAMS, FIXTURE_GRAPH_DIR

_META = {"bbox": [7.9, 47.9, 8.2, 48.1], "tile_deg": 0.5, "tolerance_m": 25.0}


def _bike_node(osmid: int, *, lat: float, lon: float, elev: float = 100.0) -> RouteNode:
    """A bike RouteNode for the path-selection helpers."""
    return RouteNode(osmid=osmid, lat=lat, lon=lon, elevation_m=elev, node_type=NodeType.BIKE, station_name=None)


# --- tile grid math ----------------------------------------------------------


def test_tile_index():
    # (row, col) = floor(lat/deg), floor(lon/deg); negative-safe.
    assert tile_index(lat=48.2, lon=8.1, tile_deg=0.5) == (96, 16)
    assert tile_index(lat=-0.1, lon=-0.1, tile_deg=0.5) == (-1, -1)  # floor, not truncate


def test_tile_name():
    # Filename stem for a tile, negative-safe.
    assert _tile_name(row=96, col=16) == "tile_96_16"
    assert _tile_name(row=-1, col=-2) == "tile_-1_-2"


def test_covering_tiles():
    # All tiles overlapping a bbox, grown by margin each side: a single-tile bbox + margin 1 → 3×3.
    assert len(_covering_tiles(bounds=(8.0, 48.0, 8.2, 48.2), tile_deg=0.5, margin=1)) == 9
    assert len(_covering_tiles(bounds=(8.0, 48.0, 8.2, 48.2), tile_deg=0.5, margin=0)) == 1


def test_intersecting_tiles():
    # Only cells the polygon truly crosses (NOT the bbox+margin rectangle): a tall thin box spans
    # exactly two rows; a diagonal tube crosses strictly fewer cells than bbox+margin, no false hits.
    corridor = box(8.1, 48.1, 8.2, 48.6)  # lon in col 16; lat 48.1→48.6 spans rows 96,97
    assert set(_intersecting_tiles(corridor=corridor, tile_deg=0.5)) == {(96, 16), (97, 16)}

    from bike_router.core.corridor import build_corridor

    diagonal = build_corridor(start_latlon=(48.0, 8.0), dest_latlon=(51.0, 11.0), half_width_km=10.0, extend_km=0.0)
    intersecting = _intersecting_tiles(corridor=diagonal, tile_deg=0.5)
    bbox_margin = _covering_tiles(bounds=diagonal.bounds, tile_deg=0.5, margin=1)
    assert len(intersecting) < len(bbox_margin)
    for row, col in intersecting:  # no false positives
        assert diagonal.intersects(box(col * 0.5, row * 0.5, (col + 1) * 0.5, (row + 1) * 0.5))


# --- meta / tile reads -------------------------------------------------------


def test_load_meta(roundtrip_store: Path):
    store = roundtrip_store
    meta = load_meta(graph_dir=store)
    assert meta["tile_deg"] == 0.5 and meta["bbox"] == [7.9, 47.9, 8.2, 48.1]


def test_read_tiles(roundtrip_store: Path):
    # Reads only the named tiles (missing skipped); a mode filter pushes the predicate to parquet;
    # tiles=None reads every tile; an empty result yields the requested-columns empty frame.
    store = roundtrip_store
    tiles = _intersecting_tiles(corridor=box(7.99, 47.99, 8.02, 48.02), tile_deg=0.5)
    rail_only = _read_tiles(
        directory=store / GraphConfig.EDGES_SUBDIR,
        columns=graph_store._EDGE_COLS,
        tiles=tiles,
        filters=[("mode", "==", "rail")],
    )
    assert not rail_only.empty and set(rail_only["mode"]) == {Mode.RAIL}
    all_nodes = _read_tiles(directory=store / GraphConfig.NODES_SUBDIR, columns=graph_store._NODE_COLS, tiles=None)
    assert len(all_nodes) == 6  # every node tile
    missing = _read_tiles(
        directory=store / GraphConfig.NODES_SUBDIR, columns=graph_store._NODE_COLS, tiles=[(999, 999)]
    )
    assert missing.empty and list(missing.columns) == graph_store._NODE_COLS


def test_load_layer(roundtrip_store: Path):
    # Reads one mode-layer confined to nodes the corridor COVERS; a bike layer yields only bike nodes
    # + bike edges among them, all inside the window.
    store = roundtrip_store
    wide = box(7.9, 47.9, 8.2, 48.1)
    nodes_df, edges_df, inside_ids = _load_layer(
        corridor=wide,
        graph_dir=store,
        node_type=NodeType.BIKE,
        edge_modes=[Mode.BIKE],
        node_columns=graph_store._ROUTE_NODE_COLS,
        edge_columns=graph_store._ROUTE_EDGE_COLS,
    )
    assert set(nodes_df["node_type"]) == {NodeType.BIKE} and len(nodes_df) == 4
    assert inside_ids == {1, 2, 3, 4}
    assert set(edges_df["mode"]) == {Mode.BIKE}
    assert edges_df["from_node"].isin(inside_ids).all()


# --- combined routing window -------------------------------------------------


def test_load_route_tables(roundtrip_store: Path):
    # Both corridors covering the fixture recombine bike ring + rail + station into the minimal
    # routing tables (6 nodes, 14 edges); the two layers load independently; outside coverage fails.
    store = roundtrip_store
    wide = box(7.9, 47.9, 8.2, 48.1)
    nodes_df, edges_df = load_route_tables(bike_corridor=wide, rail_corridor=wide, graph_dir=store)
    assert len(nodes_df) == 6 and len(edges_df) == 14
    assert set(edges_df["mode"]) == {Mode.BIKE, Mode.RAIL, Mode.STATION}

    both = box(7.999, 47.999, 8.011, 48.011)
    nodes2, _edges2 = load_route_tables(bike_corridor=both, rail_corridor=both, graph_dir=store)
    node_types = set(nodes2["node_type"])
    assert NodeType.BIKE in node_types and NodeType.RAIL in node_types

    far = box(20.0, 60.0, 20.1, 60.1)  # no tiles there
    with pytest.raises(AssertionError, match="bike corridor is outside"):
        load_route_tables(bike_corridor=far, rail_corridor=far, graph_dir=store)


# --- final-path re-read ------------------------------------------------------


def test_load_path_edges(roundtrip_store: Path):
    # Re-reads ONLY the chosen path's edges (with geometry) into an ordered RoutePath: the bike
    # square 1→2→3 yields two bike edges joining exactly those nodes, in order.
    store = roundtrip_store
    path_nodes = [(1, 48.00, 8.00), (2, 48.00, 8.01), (3, 48.01, 8.01)]
    route = load_path_edges(path_nodes=path_nodes, params=DEFAULT_PARAMS, graph_dir=store)
    assert route.osmids == [1, 2, 3]
    assert [e.mode for e in route.edges] == [Mode.BIKE, Mode.BIKE]
    assert [(e.from_node, e.to_node) for e in route.edges] == [(1, 2), (2, 3)]


def test_select_path_edges():
    # Picks the cheapest parallel candidate per consecutive hop, oriented a→b; a missing hop fails loud.
    nodes = [_bike_node(1, lat=48.0, lon=8.0), _bike_node(2, lat=48.0, lon=8.01)]
    edges_df = pd.DataFrame(
        [
            {
                "from_node": 1,
                "to_node": 2,
                "length_m": 800.0,
                "surface": "gravel",
                "highway": "residential",
                "mode": Mode.BIKE,
                "geometry_wkt": None,
            },
            {
                "from_node": 1,
                "to_node": 2,
                "length_m": 500.0,
                "surface": "asphalt",
                "highway": "residential",
                "mode": Mode.BIKE,
                "geometry_wkt": None,
            },
        ]
    )
    chosen = _select_path_edges(nodes=nodes, edges_df=edges_df, params=DEFAULT_PARAMS)
    assert len(chosen) == 1 and chosen[0].length_m == 500.0  # cheaper (shorter, paved) parallel kept

    with pytest.raises(AssertionError, match="no edge found for path hop"):
        _select_path_edges(nodes=nodes, edges_df=edges_df.iloc[0:0], params=DEFAULT_PARAMS)


def test_oriented_geometry():
    # WKT → [(lon, lat), ...] oriented to START at node_a (reversed if it ends closer); None if absent.
    node_a = _bike_node(1, lat=48.0, lon=8.0)
    forward = _oriented_geometry(wkt="LINESTRING (8.0 48.0, 8.01 48.0)", node_a=node_a)
    assert forward[0] == (8.0, 48.0)
    reverse = _oriented_geometry(wkt="LINESTRING (8.01 48.0, 8.0 48.0)", node_a=node_a)
    assert reverse[0] == (8.0, 48.0)  # flipped so it starts at node_a
    assert _oriented_geometry(wkt=None, node_a=node_a) is None  # a straight hop has no polyline


# --- snapping ----------------------------------------------------------------


def test_snap_to_node():
    # A point inside the shipped fixture snaps to a real node (close by, baked elevation); a point far
    # outside fails loud with the HANDLED user-facing error, not a bare AssertionError.
    lat, lon, elev = snap_to_node(lat=48.50, lon=8.43, graph_dir=FIXTURE_GRAPH_DIR)
    assert abs(lat - 48.50) < 0.05 and abs(lon - 8.43) < 0.05
    assert elev > 0  # baked elevation, no DEM involved
    with pytest.raises(OutOfCoverageError, match="outside the covered region"):
        snap_to_node(lat=52.52, lon=13.40, graph_dir=FIXTURE_GRAPH_DIR)


# --- download / coercion -----------------------------------------------------


def test_download_graph_from_hf(tmp_path: Path, monkeypatch):
    # Idempotent: meta.json present → skipped, no network. When missing, it runs snapshot_download
    # (HF's xet-coordinated fetch) and reports progress from the on-disk file count, ending at
    # (total, total) on the main thread. A download failure propagates.
    present = tmp_path / "present"
    present.mkdir()
    (present / GraphConfig.META_FILENAME).write_text("{}")
    called = MagicMock()
    monkeypatch.setattr(graph_store, "snapshot_download", called)
    assert download_graph_from_hf(target_dir=present) == present
    called.assert_not_called()  # already present → no network

    fresh = tmp_path / "fresh"
    repo_files = ["meta.json", "nodes/tile_0_0.parquet", "edges/tile_0_0.parquet"]
    monkeypatch.setattr(graph_store, "list_repo_files", lambda **_kwargs: repo_files)
    monkeypatch.setattr(graph_store, "_DOWNLOAD_POLL_S", 0.001)  # fast poll so the loop body runs

    def _fake_snapshot_download(*, repo_id, repo_type, local_dir, max_workers):
        # Write files with a tiny gap so the main-thread poll loop samples a partial on-disk count.
        for name in repo_files:
            Path(local_dir, name).parent.mkdir(parents=True, exist_ok=True)
            Path(local_dir, name).write_text("{}")
            time.sleep(0.005)

    monkeypatch.setattr(graph_store, "snapshot_download", _fake_snapshot_download)
    seen: list[tuple[int, int]] = []
    result = download_graph_from_hf(target_dir=fresh, progress=lambda d, t: seen.append((d, t)))
    assert result == fresh and (fresh / GraphConfig.META_FILENAME).exists()
    assert seen[-1] == (3, 3)  # progress ends at (total, total)
    assert all(0 <= d <= t == 3 for d, t in seen)  # every sample within [0, total], correct total
    assert [d for d, _t in seen] == sorted(d for d, _t in seen)  # monotonic (disk count only grows)

    def _boom(**_kwargs):
        raise OSError("network died mid-download")

    monkeypatch.setattr(graph_store, "snapshot_download", _boom)
    with pytest.raises(OSError, match="network died"):
        download_graph_from_hf(target_dir=tmp_path / "boom")


def test_str_or_none():
    # The ONE 'non-str/NaN → None' coercion for tag/name columns.
    assert _str_or_none(value="asphalt") == "asphalt"
    assert _str_or_none(value=None) is None
    assert _str_or_none(value=float("nan")) is None
    assert _str_or_none(value=42) is None
