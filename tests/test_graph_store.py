"""graph_store tests — tiled parquet round-trip, table↔graph, corridor windowing."""

import pandas as pd
import pytest
from shapely.geometry import box

from bike_router.core import graph_store
from bike_router.core.constants import Mode, NodeType
from bike_router.core.errors import OutOfCoverageError
from bike_router.core.graph_store import (
    _covering_tiles,
    _intersecting_tiles,
    _read_tiles,
    load_route_tables,
    snap_to_node,
)
from bike_router.preprocessing.graph_writer import (
    _scalar,
    graph_from_tables,
    graph_to_tables,
    write_graph_parquet,
)
from tests.conftest import FIXTURE_GRAPH_DIR, make_store_roundtrip_graph


def test_scalar_collapses_lists():
    assert _scalar(value=["asphalt", "gravel"]) == "asphalt"
    assert _scalar(value=[]) is None
    assert _scalar(value="paved") == "paved"
    assert _scalar(value=None) is None


def test_graph_to_tables_and_back_roundtrip():
    graph = make_store_roundtrip_graph()
    nodes_df, edges_df = graph_to_tables(graph=graph)
    assert list(nodes_df.columns) == graph_store._NODE_COLS
    assert list(edges_df.columns) == graph_store._EDGE_COLS
    # height_diff derived from node elevations (rail -1 100 → -2 130 = +30)
    rail_row = edges_df[(edges_df["mode"] == Mode.RAIL) & (edges_df["from_node"] == -1)].iloc[0]
    assert rail_row["height_diff_m"] == pytest.approx(30.0)

    rebuilt = graph_from_tables(nodes_df=nodes_df, edges_df=edges_df)
    assert rebuilt.number_of_nodes() == graph.number_of_nodes()
    assert rebuilt.number_of_edges() == graph.number_of_edges()
    assert rebuilt.nodes[3]["elevation"] == 130.0
    assert rebuilt.nodes[-1]["node_type"] == NodeType.RAIL  # node type round-trips


def test_graph_from_tables_drops_edges_with_missing_endpoint():
    nodes_df = pd.DataFrame(
        [{"osmid": 1, "lat": 48.0, "lon": 8.0, "elevation_m": 100.0, "node_type": NodeType.BIKE, "station_name": None}],
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
            {
                "osmid": 1,
                "lat": 48.0,
                "lon": 8.0,
                "elevation_m": 100.0,
                "node_type": NodeType.BIKE,
                "station_name": None,
            },
            {
                "osmid": 2,
                "lat": 48.01,
                "lon": 8.0,
                "elevation_m": 130.0,
                "node_type": NodeType.BIKE,
                "station_name": None,
            },
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


def test_graph_from_tables_rejects_bike_edge_touching_rail_node():
    # A bike edge whose endpoint is a rail node must fail loud (structural invariant:
    # no bike route may pass through a station).
    nodes_df = pd.DataFrame(
        [
            {
                "osmid": 1,
                "lat": 48.0,
                "lon": 8.0,
                "elevation_m": 100.0,
                "node_type": NodeType.BIKE,
                "station_name": None,
            },
            {
                "osmid": -1,
                "lat": 48.01,
                "lon": 8.0,
                "elevation_m": 100.0,
                "node_type": NodeType.RAIL,
                "station_name": "A",
            },
        ],
        columns=graph_store._NODE_COLS,
    )
    edges_df = pd.DataFrame(
        [
            {
                "from_node": 1,
                "to_node": -1,
                "key": 0,
                "length_m": 1200.0,
                "height_diff_m": 0.0,
                "surface": "asphalt",
                "highway": "residential",
                "mode": Mode.BIKE,  # WRONG: bike edge into a rail node
                "geometry_wkt": None,
            }
        ],
        columns=graph_store._EDGE_COLS,
    )
    with pytest.raises(AssertionError, match="inconsistent node types"):
        graph_from_tables(nodes_df=nodes_df, edges_df=edges_df)


def test_covering_tiles_grows_by_margin():
    tiles = _covering_tiles(bounds=(8.0, 48.0, 8.2, 48.2), tile_deg=0.5, margin=1)
    # bbox falls in a single 0.5° tile; margin 1 → 3×3 = 9 tiles
    assert len(tiles) == 9


def test_intersecting_tiles_only_cells_the_polygon_crosses():
    # A tall thin box spanning exactly two 0.5° rows in one column touches exactly 2 tiles —
    # NOT the 3×3 a bbox+margin would return. Confirms the polygon-intersection math.
    corridor = box(8.1, 48.1, 8.2, 48.6)  # lon in tile col 16; lat 48.1→48.6 spans rows 96,97
    tiles = _intersecting_tiles(corridor=corridor, tile_deg=0.5)
    assert set(tiles) == {(96, 16), (97, 16)}


def test_intersecting_tiles_fewer_than_bbox_margin_on_diagonal():
    # A diagonal tube's bbox spans a big rectangle, but the tube only crosses cells near the
    # diagonal — strictly fewer than _covering_tiles(margin=1), and every returned cell truly touches.
    from bike_router.core.corridor import build_corridor

    corridor = build_corridor(start_latlon=(48.0, 8.0), dest_latlon=(51.0, 11.0), half_width_km=10.0, extend_km=0.0)
    intersecting = _intersecting_tiles(corridor=corridor, tile_deg=0.5)
    bbox_margin = _covering_tiles(bounds=corridor.bounds, tile_deg=0.5, margin=1)
    assert len(intersecting) < len(bbox_margin)
    for row, col in intersecting:  # no false positives
        assert corridor.intersects(box(col * 0.5, row * 0.5, (col + 1) * 0.5, (row + 1) * 0.5))


def test_read_tiles_mode_pushdown(tmp_path):
    # _read_tiles with a mode filter returns ONLY matching rows (parquet predicate pushdown).
    graph = make_store_roundtrip_graph()
    nodes_df, edges_df = graph_to_tables(graph=graph)
    meta = {"bbox": [7.9, 47.9, 8.2, 48.1], "tile_deg": 0.5, "tolerance_m": 25.0}
    write_graph_parquet(nodes_df=nodes_df, edges_df=edges_df, meta=meta, out_dir=tmp_path)
    tiles = _intersecting_tiles(corridor=box(7.99, 47.99, 8.02, 48.02), tile_deg=0.5)
    rail_only = _read_tiles(
        directory=tmp_path / "edges", columns=graph_store._EDGE_COLS, tiles=tiles, filters=[("mode", "==", "rail")]
    )
    assert not rail_only.empty and set(rail_only["mode"]) == {Mode.RAIL}


def _write_fixture_store(tmp_path):
    graph = make_store_roundtrip_graph()
    nodes_df, edges_df = graph_to_tables(graph=graph)
    meta = {"bbox": [7.9, 47.9, 8.2, 48.1], "tile_deg": 0.5, "tolerance_m": 25.0}
    write_graph_parquet(nodes_df=nodes_df, edges_df=edges_df, meta=meta, out_dir=tmp_path)
    return tmp_path


def test_load_route_tables_recombines_bike_rail_station(tmp_path):
    # Both corridors cover the tiny fixture → full bike ring + rail + station edges recombine
    # into the minimal routing tables (no geometry column), returned as-built (no pruning).
    store = _write_fixture_store(tmp_path)
    wide = box(7.9, 47.9, 8.2, 48.1)  # covers all fixture nodes for both layers
    nodes_df, edges_df = load_route_tables(bike_corridor=wide, rail_corridor=wide, graph_dir=store)
    assert len(nodes_df) == 6 and len(edges_df) == 14  # 4 bike + 2 rail nodes; ring+rail+station edges
    assert set(edges_df["mode"]) == {Mode.BIKE, Mode.RAIL, Mode.STATION}


def test_load_route_tables_bike_confined_rail_generous(tmp_path):
    # A bike corridor covering only the 4 bike nodes + a rail corridor also covering them yields the
    # bike ring plus the station/rail bridge; proves the two layers load independently and recombine.
    store = _write_fixture_store(tmp_path)
    both = box(7.999, 47.999, 8.011, 48.011)
    nodes_df, _edges_df = load_route_tables(bike_corridor=both, rail_corridor=both, graph_dir=store)
    node_types = set(nodes_df["node_type"])
    assert NodeType.BIKE in node_types and NodeType.RAIL in node_types


def test_load_route_tables_outside_coverage_raises(tmp_path):
    store = _write_fixture_store(tmp_path)
    far = box(20.0, 60.0, 20.1, 60.1)  # no tiles there
    with pytest.raises(AssertionError, match="bike corridor is outside"):
        load_route_tables(bike_corridor=far, rail_corridor=far, graph_dir=store)


def test_snap_to_node_returns_nearest_node_with_elevation():
    # A point inside the fixture snaps to a real node: lat/lon close by, elevation baked.
    lat, lon, elev = snap_to_node(lat=48.50, lon=8.43, graph_dir=FIXTURE_GRAPH_DIR)
    assert abs(lat - 48.50) < 0.05 and abs(lon - 8.43) < 0.05  # nearest node is close
    assert elev > 0  # baked elevation, no DEM involved


def test_snap_to_node_outside_coverage_raises_user_facing_error():
    # A point far outside the fixture (Berlin) fails loud with a HANDLED user-facing error,
    # not a bare AssertionError — so the web app shows a toast instead of a traceback.
    with pytest.raises(OutOfCoverageError, match="outside the covered region"):
        snap_to_node(lat=52.52, lon=13.40, graph_dir=FIXTURE_GRAPH_DIR)
