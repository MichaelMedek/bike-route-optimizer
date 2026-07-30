"""graph_writer tests — build-time GeoParquet writer + MultiDiGraph↔tables round-trip.

One test_<fn> per production symbol (exact-name mirror). These run offline (networkx) — the
inference read side lives in tests/core/test_graph_store.py. Preserves the round-trip, dangling-
edge drop, and the two structural fail-loud invariants from the former test_graph_store.py.
"""

from pathlib import Path

import pandas as pd
import pytest
from shapely.geometry import LineString

from bike_router.core import graph_store
from bike_router.core.constants import Mode, NodeType
from bike_router.preprocessing.graph_writer import (
    _assert_height_diffs_consistent,
    _assert_node_edge_types_consistent,
    _geometry_wkt,
    _scalar,
    compute_bbox,
    graph_from_tables,
    graph_to_tables,
    read_full_graph,
    read_region_tables,
    undirected_graph_from_edges,
    write_graph_parquet,
)
from tests.conftest import FIXTURE_ROUNDTRIP_STORE, make_store_roundtrip_graph, write_store_roundtrip_fixture

_META = {"bbox": [7.9, 47.9, 8.2, 48.1], "tile_deg": 0.5, "tolerance_m": 25.0}


def _node_row(osmid: int, *, elev: float, node_type: str = NodeType.BIKE, name: str | None = None) -> dict:
    """One node table row in the on-disk schema."""
    return {
        "osmid": osmid,
        "lat": 48.0 + osmid * 0.01,
        "lon": 8.0,
        "elevation_m": elev,
        "node_type": node_type,
        "station_name": name,
    }


def _edge_row(u: int, v: int, *, mode: str, height_diff: float = 0.0) -> dict:
    """One edge table row in the on-disk schema."""
    return {
        "from_node": u,
        "to_node": v,
        "key": 0,
        "length_m": 1200.0,
        "height_diff_m": height_diff,
        "surface": "asphalt",
        "highway": "residential",
        "mode": mode,
        "geometry_wkt": None,
    }


# --- bbox / scalar / wkt helpers ---------------------------------------------


def test_compute_bbox():
    # (west, south, east, north) bounds of a node table.
    nodes_df = pd.DataFrame({"lon": [8.0, 8.2, 8.1], "lat": [48.0, 48.3, 48.1]})
    assert compute_bbox(nodes_df=nodes_df) == (8.0, 48.0, 8.2, 48.3)


def test_undirected_graph_from_edges():
    # Endpoint columns → undirected graph; weight_col stores the column as edge 'weight'.
    edges = pd.DataFrame({"from_node": [1, 2], "to_node": [2, 3], "length_m": [100.0, 200.0]})
    plain = undirected_graph_from_edges(edges_df=edges)
    assert plain.number_of_nodes() == 3 and plain.has_edge(1, 2) and plain.has_edge(2, 3)
    weighted = undirected_graph_from_edges(edges_df=edges, weight_col="length_m")
    assert weighted[1][2]["weight"] == 100.0 and weighted[2][3]["weight"] == 200.0


def test_scalar():
    # Collapses a list-valued OSM tag to its first element; unknown/empty/NaN → explicit None.
    assert _scalar(value=["asphalt", "gravel"]) == "asphalt"
    assert _scalar(value=[]) is None
    assert _scalar(value="paved") == "paved"
    assert _scalar(value=None) is None
    assert _scalar(value=float("nan")) is None


def test_geometry_wkt():
    # A real >=2-point polyline → WKT; anything else (None, degenerate) → None.
    wkt = _geometry_wkt(geom=LineString([(8.0, 48.0), (8.01, 48.0)]))
    assert isinstance(wkt, str) and wkt.startswith("LINESTRING")
    assert _geometry_wkt(geom=None) is None


# --- round-trip --------------------------------------------------------------


def test_graph_to_tables():
    # Flattens a MultiDiGraph to node/edge tables in the on-disk schema; height_diff is derived
    # from node elevations (rail -1 100 → -2 130 = +30).
    nodes_df, edges_df = graph_to_tables(graph=make_store_roundtrip_graph())
    assert list(nodes_df.columns) == graph_store._NODE_COLS
    assert list(edges_df.columns) == graph_store._EDGE_COLS
    rail_row = edges_df[(edges_df["mode"] == Mode.RAIL) & (edges_df["from_node"] == -1)].iloc[0]
    assert rail_row["height_diff_m"] == pytest.approx(30.0)


def test_graph_from_tables():
    # Rebuilds an OSMnx-shaped MultiDiGraph; node/edge counts + types round-trip; an edge whose
    # endpoint is outside the loaded window is dropped (dangling).
    nodes_df, edges_df = graph_to_tables(graph=make_store_roundtrip_graph())
    rebuilt = graph_from_tables(nodes_df=nodes_df, edges_df=edges_df)
    original = make_store_roundtrip_graph()
    assert rebuilt.number_of_nodes() == original.number_of_nodes()
    assert rebuilt.number_of_edges() == original.number_of_edges()
    assert rebuilt.nodes[3]["elevation"] == 130.0
    assert rebuilt.nodes[-1]["node_type"] == NodeType.RAIL

    lone = pd.DataFrame([_node_row(1, elev=100.0)], columns=graph_store._NODE_COLS)
    dangling = pd.DataFrame([_edge_row(1, 999, mode=Mode.BIKE)], columns=graph_store._EDGE_COLS)
    assert graph_from_tables(nodes_df=lone, edges_df=dangling).number_of_edges() == 0


def test_read_region_tables(tmp_path: Path):
    # Reads a region artifact's full node + edge tables back — every tile concatenated, standard
    # schemas — even when nodes span MULTIPLE 0.5° tiles (written to separate tile files).
    nodes_df, edges_df = graph_to_tables(graph=make_store_roundtrip_graph())
    write_graph_parquet(nodes_df=nodes_df, edges_df=edges_df, meta=_META, out_dir=tmp_path)
    got_nodes, got_edges = read_region_tables(region_dir=tmp_path)
    assert len(got_nodes) == len(nodes_df) and len(got_edges) == len(edges_df)
    assert list(got_nodes.columns) == graph_store._NODE_COLS

    multi_dir = tmp_path / "multi"
    two_tile_nodes = pd.DataFrame(
        [_node_row(0, elev=0.0), {**_node_row(1, elev=0.0), "lat": 48.6, "lon": 8.6}],  # tiles 96_16 and 97_17
        columns=graph_store._NODE_COLS,
    )
    two_tile_edges = pd.DataFrame([_edge_row(0, 1, mode=Mode.BIKE)], columns=graph_store._EDGE_COLS)
    write_graph_parquet(nodes_df=two_tile_nodes, edges_df=two_tile_edges, meta=_META, out_dir=multi_dir)
    back_nodes, back_edges = read_region_tables(region_dir=multi_dir)
    assert len(back_nodes) == 2 and len(back_edges) == 1 and set(back_nodes["osmid"]) == {0, 1}


def test_read_full_graph(tmp_path: Path):
    # Reconstructs the WHOLE graph from every tile (Phase-4 validation), matching the saved counts.
    nodes_df, edges_df = graph_to_tables(graph=make_store_roundtrip_graph())
    write_graph_parquet(nodes_df=nodes_df, edges_df=edges_df, meta=_META, out_dir=tmp_path)
    graph = read_full_graph(graph_dir=tmp_path)
    assert graph.number_of_nodes() == 6 and graph.number_of_edges() == 14


def test_write_graph_parquet(tmp_path: Path):
    # Writes node/edge tables as lat/lon-tiled parquet + meta.json; a schema drift fails loud; the
    # zstd codec (final HF artifact) round-trips losslessly (readers auto-detect the codec).
    nodes_df, edges_df = graph_to_tables(graph=make_store_roundtrip_graph())
    write_graph_parquet(nodes_df=nodes_df, edges_df=edges_df, meta=_META, out_dir=tmp_path)
    assert (tmp_path / graph_store.GraphConfig.META_FILENAME).exists()
    assert list((tmp_path / graph_store.GraphConfig.NODES_SUBDIR).glob("tile_*.parquet"))
    assert list((tmp_path / graph_store.GraphConfig.EDGES_SUBDIR).glob("tile_*.parquet"))
    with pytest.raises(AssertionError, match="nodes schema drift"):
        write_graph_parquet(
            nodes_df=nodes_df.drop(columns=["station_name"]), edges_df=edges_df, meta=_META, out_dir=tmp_path
        )

    zstd_dir = tmp_path / "zstd"
    wkt = LineString([(8.0, 48.0), (8.1, 48.1)])
    z_nodes = pd.DataFrame([_node_row(0, elev=1.0), _node_row(1, elev=2.0)], columns=graph_store._NODE_COLS)
    z_edges = pd.DataFrame(
        [{**_edge_row(0, 1, mode=Mode.BIKE), "geometry_wkt": _geometry_wkt(geom=wkt)}], columns=graph_store._EDGE_COLS
    )
    write_graph_parquet(nodes_df=z_nodes, edges_df=z_edges, meta=_META, out_dir=zstd_dir, compression="zstd")
    back_nodes, back_edges = read_region_tables(region_dir=zstd_dir)
    assert set(back_nodes["osmid"]) == {0, 1}
    assert len(back_edges) == 1 and back_edges.iloc[0]["geometry_wkt"] == _geometry_wkt(geom=wkt)  # lossless


# --- structural invariants (fail loud) ---------------------------------------


def test_assert_node_edge_types_consistent():
    # A well-typed graph passes; a BIKE edge touching a rail node fails loud (no bike route may
    # pass through a station — reaching one always crosses a station edge).
    _assert_node_edge_types_consistent(make_store_roundtrip_graph())  # clean graph passes

    nodes_df = pd.DataFrame(
        [_node_row(1, elev=100.0), _node_row(-1, elev=100.0, node_type=NodeType.RAIL, name="A")],
        columns=graph_store._NODE_COLS,
    )
    edges_df = pd.DataFrame([_edge_row(1, -1, mode=Mode.BIKE)], columns=graph_store._EDGE_COLS)
    with pytest.raises(AssertionError, match="inconsistent node types"):
        graph_from_tables(nodes_df=nodes_df, edges_df=edges_df)


def test_assert_height_diffs_consistent():
    # A graph whose stored height_diff matches node elevations passes; a tampered value fails loud.
    clean_nodes, clean_edges = graph_to_tables(graph=make_store_roundtrip_graph())
    _assert_height_diffs_consistent(graph_from_tables(nodes_df=clean_nodes, edges_df=clean_edges))  # passes

    nodes_df = pd.DataFrame([_node_row(1, elev=100.0), _node_row(2, elev=130.0)], columns=graph_store._NODE_COLS)
    edges_df = pd.DataFrame([_edge_row(1, 2, mode=Mode.BIKE, height_diff=5.0)], columns=graph_store._EDGE_COLS)
    with pytest.raises(AssertionError, match="height_diff mismatch"):  # real diff is 30, not 5
        graph_from_tables(nodes_df=nodes_df, edges_df=edges_df)


# --- committed fixture drift guard -------------------------------------------


def test_write_store_roundtrip_fixture(tmp_path: Path):
    # The committed FIXTURE_ROUNDTRIP_STORE (read by CORE tests without networkx) must still equal a
    # fresh build — single source of truth. If the builder/schema changes, regenerate the committed store.
    fresh = write_store_roundtrip_fixture(out_dir=tmp_path / "fresh")

    def _tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        def _concat(sub: str) -> pd.DataFrame:
            parts = [pd.read_parquet(p) for p in sorted((root / sub).glob("tile_*.parquet"))]
            return pd.concat(parts).sort_values(list(parts[0].columns)).reset_index(drop=True)

        return _concat(graph_store.GraphConfig.NODES_SUBDIR), _concat(graph_store.GraphConfig.EDGES_SUBDIR)

    fresh_nodes, fresh_edges = _tables(fresh)
    committed_nodes, committed_edges = _tables(FIXTURE_ROUNDTRIP_STORE)
    pd.testing.assert_frame_equal(fresh_nodes, committed_nodes)
    pd.testing.assert_frame_equal(fresh_edges, committed_edges)
    assert graph_store.load_meta(graph_dir=FIXTURE_ROUNDTRIP_STORE) == graph_store.load_meta(graph_dir=fresh)
