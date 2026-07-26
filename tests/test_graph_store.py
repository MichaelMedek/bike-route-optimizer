"""graph_store tests — tiled parquet round-trip, table↔graph, corridor windowing."""

import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import box

from bike_router import graph_store
from bike_router.constants import Mode
from bike_router.graph_store import (
    _covering_tiles,
    _scalar,
    graph_from_tables,
    graph_to_tables,
    load_corridor_graph,
    load_meta,
    snap_to_node,
    write_graph_parquet,
)
from tests.conftest import FIXTURE_GRAPH_DIR


def _toy_graph() -> nx.MultiDiGraph:
    """A 4-node square near (8.0, 48.0) with one rail edge, elevations baked."""
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    pts = {1: (8.00, 48.00, 100.0), 2: (8.01, 48.00, 110.0), 3: (8.01, 48.01, 130.0), 4: (8.00, 48.01, 120.0)}
    for nid, (lon, lat, elev) in pts.items():
        graph.add_node(nid, x=lon, y=lat, elevation=elev, is_station=False, station_name=None)
    ring = [(1, 2), (2, 3), (3, 4), (4, 1), (2, 1), (3, 2), (4, 3), (1, 4)]
    for a, b in ring:
        graph.add_edge(a, b, key=0, length=800.0, surface="asphalt", highway="residential", mode=Mode.BIKE)
    graph.add_edge(1, 3, key=0, length=1500.0, surface=None, highway=None, mode=Mode.RAIL)
    return graph


def test_scalar_collapses_lists():
    assert _scalar(["asphalt", "gravel"]) == "asphalt"
    assert _scalar([]) is None
    assert _scalar("paved") == "paved"
    assert _scalar(None) is None


def test_graph_to_tables_and_back_roundtrip():
    graph = _toy_graph()
    nodes_df, edges_df = graph_to_tables(graph)
    assert list(nodes_df.columns) == graph_store._NODE_COLS
    assert list(edges_df.columns) == graph_store._EDGE_COLS
    # height_diff derived from node elevations
    rail_row = edges_df[edges_df["mode"] == Mode.RAIL].iloc[0]
    assert rail_row["height_diff_m"] == pytest.approx(30.0)  # node3 130 − node1 100

    rebuilt = graph_from_tables(nodes_df=nodes_df, edges_df=edges_df)
    assert rebuilt.number_of_nodes() == graph.number_of_nodes()
    assert rebuilt.number_of_edges() == graph.number_of_edges()
    assert rebuilt.nodes[3]["elevation"] == 130.0


def test_graph_from_tables_drops_edges_with_missing_endpoint():
    nodes_df = pd.DataFrame(
        [{"osmid": 1, "lat": 48.0, "lon": 8.0, "elevation_m": 100.0, "station_name": None}],
        columns=graph_store._NODE_COLS,
    )
    edges_df = pd.DataFrame(
        [
            {
                "from_node": 1,
                "to_node": 999,
                "key": 0,
                "length_m": 10.0,
                "height_diff_m": 0.0,
                "surface": None,
                "highway": None,
                "mode": Mode.BIKE,
                "geometry_wkt": None,
            }
        ],
        columns=graph_store._EDGE_COLS,
    )
    graph = graph_from_tables(nodes_df=nodes_df, edges_df=edges_df)
    assert graph.number_of_edges() == 0  # dangling edge dropped


def test_graph_from_tables_rejects_inconsistent_height_diff():
    # height_diff_m must match to_elev − from_elev; a tampered value fails loud.
    nodes_df = pd.DataFrame(
        [
            {"osmid": 1, "lat": 48.0, "lon": 8.0, "elevation_m": 100.0, "station_name": None},
            {"osmid": 2, "lat": 48.01, "lon": 8.0, "elevation_m": 130.0, "station_name": None},
        ],
        columns=graph_store._NODE_COLS,
    )
    edges_df = pd.DataFrame(
        [
            {
                "from_node": 1,
                "to_node": 2,
                "key": 0,
                "length_m": 1200.0,
                "height_diff_m": 5.0,  # WRONG: real diff is 130 − 100 = 30 m
                "surface": "asphalt",
                "highway": "residential",
                "mode": Mode.BIKE,
                "geometry_wkt": None,
            }
        ],
        columns=graph_store._EDGE_COLS,
    )
    with pytest.raises(AssertionError, match="height_diff mismatch"):
        graph_from_tables(nodes_df=nodes_df, edges_df=edges_df)


def test_covering_tiles_grows_by_margin():
    tiles = _covering_tiles(bounds=(8.0, 48.0, 8.2, 48.2), tile_deg=0.5, margin=1)
    # bbox falls in a single 0.5° tile; margin 1 → 3×3 = 9 tiles
    assert len(tiles) == 9


def test_write_then_load_corridor_roundtrip(tmp_path):
    graph = _toy_graph()
    nodes_df, edges_df = graph_to_tables(graph)
    meta = {"bbox": [7.9, 47.9, 8.2, 48.1], "tile_deg": 0.5, "tolerance_m": 25.0}
    write_graph_parquet(nodes_df=nodes_df, edges_df=edges_df, meta=meta, out_dir=tmp_path)
    assert (tmp_path / "meta.json").exists()
    assert load_meta(graph_dir=tmp_path)["tile_deg"] == 0.5

    corridor = box(7.99, 47.99, 8.02, 48.02)
    loaded = load_corridor_graph(corridor=corridor, graph_dir=tmp_path)
    assert loaded.number_of_nodes() > 0
    # rail edge preserved through the round-trip
    modes = {d["mode"] for _u, _v, _k, d in loaded.edges(keys=True, data=True)}
    assert Mode.RAIL in modes


def test_load_corridor_outside_coverage_raises(tmp_path):
    graph = _toy_graph()
    nodes_df, edges_df = graph_to_tables(graph)
    meta = {"bbox": [7.9, 47.9, 8.2, 48.1], "tile_deg": 0.5, "tolerance_m": 25.0}
    write_graph_parquet(nodes_df=nodes_df, edges_df=edges_df, meta=meta, out_dir=tmp_path)
    far = box(20.0, 60.0, 20.1, 60.1)  # no tiles there
    with pytest.raises(AssertionError):
        load_corridor_graph(corridor=far, graph_dir=tmp_path)


def test_snap_to_node_returns_nearest_node_with_elevation():
    # A point inside the fixture snaps to a real node: lat/lon close by, elevation baked.
    lat, lon, elev = snap_to_node(lat=48.50, lon=8.43, graph_dir=FIXTURE_GRAPH_DIR)
    assert abs(lat - 48.50) < 0.05 and abs(lon - 8.43) < 0.05  # nearest node is close
    assert elev > 0  # baked elevation, no DEM involved
