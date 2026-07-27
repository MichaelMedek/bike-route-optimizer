"""Phase-3 combine + Phase-4 validate logic of the DACH build script (synthetic artifacts).

Builds two tiny per-region artifacts on disk (no pbf/DEM), then exercises the combine's
cumulative-offset reindex + geometry dedup and the connectivity validation, without the
slow real region build.
"""

import networkx as nx
import pandas as pd
import pytest

from bike_router.constants import GraphConfig
from bike_router.graph_store import read_full_graph, read_region_tables, write_graph_parquet
from scripts import build_dach_graph as bd

_NODE_COLS = ["osmid", "lat", "lon", "elevation_m", "node_type", "station_name"]
_EDGE_COLS = ["from_node", "to_node", "key", "length_m", "height_diff_m", "surface", "highway", "mode", "geometry_wkt"]


def _region_artifact(region_dir, nodes: list[tuple], edges: list[tuple]) -> None:  # noqa: ANN001
    """Write a minimal confirmed_complete per-region artifact to region_dir."""
    nodes_df = pd.DataFrame(nodes, columns=_NODE_COLS)
    edges_df = pd.DataFrame(edges, columns=_EDGE_COLS)
    meta = {"tile_deg": GraphConfig.TILE_DEG, "confirmed_complete": True}
    write_graph_parquet(nodes_df=nodes_df, edges_df=edges_df, meta=meta, out_dir=region_dir)


def _line(lat1, lon1, lat2, lon2) -> str:  # noqa: ANN001
    return f"LINESTRING ({lon1} {lat1}, {lon2} {lat2})"


def test_combine_regions_reindexes_and_dedups(monkeypatch, tmp_path):
    # Two regions each with contiguous ids 0,1 that would COLLIDE; region B's node 0 is a seam
    # duplicate of region A's node 1 (same lat/lon). Combine must offset then collapse the dup.
    regions_dir = tmp_path / "per_region"
    monkeypatch.setattr(bd, "_REGIONS_DIR", regions_dir)
    _region_artifact(
        regions_dir / "a",
        nodes=[(0, 48.0, 8.0, 0.0, "bike", None), (1, 48.2, 8.2, 0.0, "bike", None)],
        edges=[(0, 1, 0, 100.0, 0.0, "asphalt", "residential", "bike", _line(48.0, 8.0, 48.2, 8.2))],
    )
    _region_artifact(
        regions_dir / "b",
        nodes=[(0, 48.2, 8.2, 0.0, "bike", None), (1, 48.4, 8.4, 0.0, "bike", None)],  # node 0 == A's node 1
        edges=[(0, 1, 0, 100.0, 0.0, "asphalt", "residential", "bike", _line(48.2, 8.2, 48.4, 8.4))],
    )
    nodes_df, edges_df = bd._combine_regions(regions=["a", "b"])
    # 4 raw nodes → 3 after the seam duplicate collapses; ids globally unique AND contiguous 0..N-1
    # (the final remap closes the hole dedup left).
    assert len(nodes_df) == 3
    assert sorted(nodes_df["osmid"]) == [0, 1, 2]
    # both distinct edges survive (no false dedup — different geometry); endpoints valid ids
    assert len(edges_df) == 2
    assert set(edges_df["from_node"]) | set(edges_df["to_node"]) <= set(nodes_df["osmid"])


def test_validate_connectivity_passes_and_fails(monkeypatch, tmp_path):
    # A fully-connected 2-tile line passes; an artifact split into two disconnected tiles fails.
    def _write(out_dir, nodes, edges):  # noqa: ANN001, ANN202
        write_graph_parquet(
            nodes_df=pd.DataFrame(nodes, columns=_NODE_COLS),
            edges_df=pd.DataFrame(edges, columns=_EDGE_COLS),
            meta={"tile_deg": GraphConfig.TILE_DEG},
            out_dir=out_dir,
        )

    monkeypatch.setattr(bd, "_VALIDATION_PROBES", 5)
    # Connected: two nodes in different tiles joined both ways.
    ok_dir = tmp_path / "ok"
    _write(
        ok_dir,
        nodes=[(0, 48.0, 8.0, 0.0, "bike", None), (1, 48.6, 8.6, 0.0, "bike", None)],
        edges=[
            (0, 1, 0, 1.0, 0.0, "asphalt", "residential", "bike", None),
            (1, 0, 0, 1.0, 0.0, "asphalt", "residential", "bike", None),
        ],
    )
    bd._validate_connectivity(out_dir=ok_dir)  # no raise

    # Disconnected: same two cross-tile nodes, NO edge between them (fresh dir → no stale tiles).
    bad_dir = tmp_path / "bad"
    _write(bad_dir, nodes=[(0, 48.0, 8.0, 0.0, "bike", None), (1, 48.6, 8.6, 0.0, "bike", None)], edges=[])
    with pytest.raises(ValueError, match="fragmented|<2 nodes|path|components"):
        bd._validate_connectivity(out_dir=bad_dir)

    # One-way only (0→1 but not back): weakly connected but NOT strongly → the SCC gate must fail,
    # since a routable graph needs mutual reachability.
    oneway_dir = tmp_path / "oneway"
    _write(
        oneway_dir,
        nodes=[(0, 48.0, 8.0, 0.0, "bike", None), (1, 48.6, 8.6, 0.0, "bike", None)],
        edges=[(0, 1, 0, 1.0, 0.0, "asphalt", "residential", "bike", None)],
    )
    with pytest.raises(ValueError, match="components"):
        bd._validate_connectivity(out_dir=oneway_dir)


def test_assert_all_regions_complete_fails_loud(monkeypatch, tmp_path):
    monkeypatch.setattr(bd, "_REGIONS_DIR", tmp_path)
    (tmp_path / "done").mkdir()
    (tmp_path / "done" / GraphConfig.META_FILENAME).write_text('{"confirmed_complete": true}')
    # "missing" region has no artifact → gate raises naming it.
    with pytest.raises(ValueError, match="missing"):
        bd._assert_all_regions_complete(regions=["done", "missing"])
    bd._assert_all_regions_complete(regions=["done"])  # all complete → no raise


def test_region_complete_guard(monkeypatch, tmp_path):
    # The Phase-2 skip guard: only a present meta.json with confirmed_complete==true counts as done.
    monkeypatch.setattr(bd, "_REGIONS_DIR", tmp_path)
    assert bd._region_complete(region_key="absent") is False  # no dir at all
    (tmp_path / "partial").mkdir()  # dir exists, no meta → NOT complete (rebuilt, not skipped)
    assert bd._region_complete(region_key="partial") is False
    (tmp_path / "unflagged").mkdir()
    (tmp_path / "unflagged" / GraphConfig.META_FILENAME).write_text('{"n_nodes": 5}')  # meta, flag absent
    assert bd._region_complete(region_key="unflagged") is False
    (tmp_path / "ok").mkdir()
    (tmp_path / "ok" / GraphConfig.META_FILENAME).write_text('{"confirmed_complete": true}')
    assert bd._region_complete(region_key="ok") is True


def test_combine_regions_fails_fast_on_node_ceiling(monkeypatch, tmp_path):
    # The grand-total sanity ceiling: if the combined node count would blow past it, fail loud.
    regions_dir = tmp_path / "per_region"
    monkeypatch.setattr(bd, "_REGIONS_DIR", regions_dir)
    monkeypatch.setattr(bd, "_MAX_TOTAL_NODES", 2)  # tiny ceiling to trip deterministically
    _region_artifact(
        regions_dir / "a",
        nodes=[(0, 48.0, 8.0, 0.0, "bike", None), (1, 48.2, 8.2, 0.0, "bike", None), (2, 48.4, 8.4, 0.0, "bike", None)],
        edges=[(0, 1, 0, 1.0, 0.0, "asphalt", "residential", "bike", None)],
    )
    with pytest.raises(ValueError, match="ceiling"):
        bd._combine_regions(regions=["a"])


def test_read_region_tables_roundtrips_all_tiles(tmp_path):
    # Nodes spanning two 0.5° tiles are written then read back whole (all tiles concatenated),
    # returning the standard node/edge schemas regardless of how many tile files exist.
    nodes = pd.DataFrame(
        [
            (0, 48.0, 8.0, 0.0, "bike", None),  # tile 96_16
            (1, 48.6, 8.6, 0.0, "bike", None),  # tile 97_17 — a different tile
        ],
        columns=_NODE_COLS,
    )
    edges = pd.DataFrame([(0, 1, 0, 1.0, 0.0, "asphalt", "residential", "bike", None)], columns=_EDGE_COLS)
    write_graph_parquet(nodes_df=nodes, edges_df=edges, meta={"tile_deg": GraphConfig.TILE_DEG}, out_dir=tmp_path)
    nodes_back, edges_back = read_region_tables(region_dir=tmp_path)
    assert len(nodes_back) == 2 and len(edges_back) == 1  # every tile concatenated
    assert list(nodes_back.columns) == _NODE_COLS and list(edges_back.columns) == _EDGE_COLS
    assert set(nodes_back["osmid"]) == {0, 1}


def test_read_full_graph_roundtrip(tmp_path):
    write_graph_parquet(
        nodes_df=pd.DataFrame(
            [(0, 48.0, 8.0, 0.0, "bike", None), (1, 48.1, 8.1, 0.0, "bike", None)], columns=_NODE_COLS
        ),
        edges_df=pd.DataFrame([(0, 1, 0, 1.0, 0.0, "asphalt", "residential", "bike", None)], columns=_EDGE_COLS),
        meta={"tile_deg": GraphConfig.TILE_DEG},
        out_dir=tmp_path,
    )
    graph = read_full_graph(graph_dir=tmp_path)
    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.number_of_nodes() == 2 and graph.number_of_edges() == 1
