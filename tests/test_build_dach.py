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


def test_region_split_config_shares_pbf_and_overlaps():
    # Austria/Switzerland are bbox-split east/west: the two halves must share ONE pbf download
    # (dedup by geofabrik_path) and their boxes must OVERLAP at the seam (> the ~29 km longest
    # rail edge) so Phase-3 dedup can stitch them. Every region key is unique.
    regions = bd.DACH_REGIONS
    assert len({r.key for r in regions}) == len(regions)  # unique keys
    for country in ("austria", "switzerland"):
        west = next(r for r in regions if r.key == f"{country}-west")
        east = next(r for r in regions if r.key == f"{country}-east")
        assert west.pbf_name == east.pbf_name == country  # ONE shared pbf download
        assert west.bbox is not None and east.bbox is not None
        overlap_deg = west.bbox[2] - east.bbox[0]  # west's east-edge minus east's west-edge
        assert overlap_deg >= 0.5, f"{country} seam overlap {overlap_deg}° too small for a 29 km edge"


def test_split_halves_download_once(monkeypatch, tmp_path):
    # The four split halves reference only two distinct pbfs — Phase 1 must download each once.
    calls: list[str] = []
    monkeypatch.setattr(bd, "_download_region", lambda *, geofabrik_path: calls.append(geofabrik_path) or tmp_path)
    paths = sorted({r.geofabrik_path for r in bd.DACH_REGIONS if r.bbox})
    for gp in paths:
        bd._download_region(geofabrik_path=gp)
    assert calls == ["austria", "switzerland"]  # 4 halves → 2 downloads


def test_write_graph_parquet_zstd_lossless_roundtrip(tmp_path):
    # Phase 3 writes the final artifact with compression="zstd"; readers auto-detect the codec,
    # so a zstd round-trip must return identical tables (the download side needs no codec hint).
    nodes = pd.DataFrame([(0, 48.0, 8.0, 1.0, "bike", None), (1, 48.1, 8.1, 2.0, "bike", None)], columns=_NODE_COLS)
    edges = pd.DataFrame(
        [(0, 1, 0, 1.0, 1.0, "asphalt", "residential", "bike", _line(48.0, 8.0, 48.1, 8.1))], columns=_EDGE_COLS
    )
    write_graph_parquet(
        nodes_df=nodes, edges_df=edges, meta={"tile_deg": GraphConfig.TILE_DEG}, out_dir=tmp_path, compression="zstd"
    )
    nodes_back, edges_back = read_region_tables(region_dir=tmp_path)
    assert set(nodes_back["osmid"]) == {0, 1}
    assert len(edges_back) == 1 and edges_back.iloc[0]["geometry_wkt"] == _line(48.0, 8.0, 48.1, 8.1)


def test_split_geofabrik_path_leaf():
    # The shared-pbf cache name is the Geofabrik leaf: nested German paths and bare country paths.
    assert bd.split_geofabrik_path(geofabrik_path="germany/baden-wuerttemberg/freiburg-regbez") == "freiburg-regbez"
    assert bd.split_geofabrik_path(geofabrik_path="austria") == "austria"


def test_assert_output_empty_guard(tmp_path):
    # Absent dir → OK; empty dir → OK; a single leftover file → fail fast (no stale-artifact reuse).
    bd._assert_output_empty(out_dir=tmp_path / "absent")  # absent: no raise
    bd._assert_output_empty(out_dir=tmp_path)  # empty: no raise
    (tmp_path / "leftover.parquet").write_text("x")
    with pytest.raises(ValueError, match="not empty"):
        bd._assert_output_empty(out_dir=tmp_path)


class _FakeDEM:
    def __init__(self, bounds: tuple[float, float, float, float]) -> None:
        self.bounds = bounds


def test_assert_dem_covers_guard():
    # DEM must fully contain the build area; a too-small DEM (misses the east edge) fails fast.
    dem = _FakeDEM(bounds=(5.0, 45.0, 18.0, 56.0))  # covers all DACH
    bd._assert_dem_covers(dem=dem, area=(5.9, 45.8, 17.2, 55.1))  # inside → no raise
    small = _FakeDEM(bounds=(5.0, 45.0, 16.0, 56.0))  # east edge 16 < area's 17.2
    with pytest.raises(ValueError, match="does not contain"):
        bd._assert_dem_covers(dem=small, area=(5.9, 45.8, 17.2, 55.1))


def test_random_cross_tile_pair_differs(monkeypatch):
    # Phase-4 probe: the returned node pair must come from DIFFERENT tiles (a real cross-region check).
    import random

    graph = nx.MultiDiGraph()
    graph.add_node(0, x=8.0, y=48.0)  # tile A
    graph.add_node(1, x=13.0, y=52.0)  # far tile B
    src, tgt = bd._random_cross_tile_pair(graph=graph, nodes=[0, 1], rng=random.Random(0))
    s = bd.tile_index(lat=graph.nodes[src]["y"], lon=graph.nodes[src]["x"])
    t = bd.tile_index(lat=graph.nodes[tgt]["y"], lon=graph.nodes[tgt]["x"])
    assert s != t  # deliberately cross-tile
