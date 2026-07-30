"""Phase-3 combine + Phase-4 validate logic of the DACH build script (synthetic artifacts).

Builds two tiny per-region artifacts on disk (no pbf/DEM), then exercises the combine's
cumulative-offset reindex + geometry dedup and the connectivity validation, without the
slow real region build.
"""

import networkx as nx
import pandas as pd
import pytest

from bike_router.core.constants import GraphConfig
from bike_router.preprocessing.graph_writer import read_full_graph, read_region_tables, write_graph_parquet
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
    # Two regions each with contiguous ids that would COLLIDE; region B's node 0 is a seam
    # duplicate of region A's node 1 (same lat/lon). Combine must offset then collapse the dup.
    # Edges are 30 km each so the combined bike chain (60 km) clears the MIN_BIKE_COMPONENT_KM prune;
    # region A also carries a rail pair (rail is a mandatory layer).
    regions_dir = tmp_path / "per_region"
    monkeypatch.setattr(bd, "_REGIONS_DIR", regions_dir)
    long_m = 30_000.0
    _region_artifact(
        regions_dir / "a",
        nodes=[
            (0, 48.0, 8.0, 0.0, "bike", None),
            (1, 48.2, 8.2, 0.0, "bike", None),
            (2, 47.0, 7.0, 0.0, "rail", None),
            (3, 47.1, 7.1, 0.0, "rail", None),  # rail pair (mandatory layer)
        ],
        edges=[
            (0, 1, 0, long_m, 0.0, "asphalt", "residential", "bike", _line(48.0, 8.0, 48.2, 8.2)),
            (1, 0, 0, long_m, 0.0, "asphalt", "residential", "bike", _line(48.2, 8.2, 48.0, 8.0)),
            (2, 3, 0, 40_000.0, 0.0, None, None, "rail", _line(47.0, 7.0, 47.1, 7.1)),
            (3, 2, 0, 40_000.0, 0.0, None, None, "rail", _line(47.1, 7.1, 47.0, 7.0)),
        ],
    )
    _region_artifact(
        regions_dir / "b",
        nodes=[(0, 48.2, 8.2, 0.0, "bike", None), (1, 48.4, 8.4, 0.0, "bike", None)],  # node 0 == A's node 1
        edges=[
            (0, 1, 0, long_m, 0.0, "asphalt", "residential", "bike", _line(48.2, 8.2, 48.4, 8.4)),
            (1, 0, 0, long_m, 0.0, "asphalt", "residential", "bike", _line(48.4, 8.4, 48.2, 8.2)),
        ],
    )
    nodes_df, edges_df = bd._combine_regions(regions=["a", "b"])
    # A has 4 nodes (2 bike + 2 rail), B has 2 bike; B's node 0 duplicates A's bike node 1 → 6-1 = 5
    # surviving nodes, renumbered dense 0..4.
    assert len(nodes_df) == 5
    assert sorted(nodes_df["osmid"]) == [0, 1, 2, 3, 4]
    # bike: 4 directed edges (A's 2 + B's 2, no false dedup) + rail: 2 directed = 6 total.
    assert len(edges_df) == 6
    assert set(edges_df["from_node"]) | set(edges_df["to_node"]) <= set(nodes_df["osmid"])
    assert set(edges_df["from_node"]) | set(edges_df["to_node"]) <= set(nodes_df["osmid"])


def _pe(u, v, m, km):  # noqa: ANN001, ANN202
    """One directed prune-test edge; km → length_m. surface/highway None (irrelevant to the prune)."""
    return (u, v, 0, km * 1000.0, 0.0, None, None, m, None)


def _bidir(u, v, m, km):  # noqa: ANN001, ANN202
    """A bidirectional road/track as the builder emits it — both directions of one physical edge."""
    return [_pe(u, v, m, km), _pe(v, u, m, km)]


def _mk_nodes(bike, rail, stations):  # noqa: ANN001, ANN202
    """Node frame from id-lists: bike (node_type bike), rail track (rail), stations (rail + a name)."""
    rows = (
        [(i, 48.0, 8.0 + i * 0.01, 0.0, "bike", None) for i in bike]
        + [(i, 47.0, 7.0 + i * 0.01, 0.0, "rail", None) for i in rail]
        + [(i, 46.0, 6.0, 0.0, "rail", f"St{i}") for i in stations]
    )
    return pd.DataFrame(rows, columns=_NODE_COLS)


class TestPruneComponents:
    """_prune_components policy: RAIL keeps ONLY its largest weakly-connected component; BIKE keeps every
    component with total (undirected) length ≥ MIN_BIKE_COMPONENT_KM; a node/edge survives iff on a kept
    component. Stations (rail nodes wired by a RAIL link) follow their rail component. Both layers are
    MANDATORY. Tests assert only real behaviour — not the tautological edges⊆nodes filter identity.
    """

    _OVER = GraphConfig.MIN_BIKE_COMPONENT_KM + 5  # comfortably above threshold
    _UNDER = 1.0  # comfortably below

    @pytest.mark.parametrize(
        ("bike_km", "kept"),
        [
            (GraphConfig.MIN_BIKE_COMPONENT_KM - 1, False),  # below → dropped
            (GraphConfig.MIN_BIKE_COMPONENT_KM, True),  # exactly at → kept (>=)
            (GraphConfig.MIN_BIKE_COMPONENT_KM + 1, True),  # above → kept
        ],
    )
    def test_bike_component_threshold_boundary(self, bike_km, kept):  # noqa: ANN001
        # BIKE keep-iff-big at the exact boundary. Undirected: one bidirectional edge counts ONCE.
        nodes = _mk_nodes(bike=[0, 1], rail=[2, 3], stations=[])
        edges = pd.DataFrame(_bidir(0, 1, "bike", bike_km) + _bidir(2, 3, "rail", 40.0), columns=_EDGE_COLS)
        surviving = set(bd._prune_components(nodes_df=nodes, edges_df=edges)[0]["osmid"])
        assert ({0, 1} <= surviving) is kept

    def test_bike_keeps_big_island_drops_small_stray(self):
        # Two bike components: a big island (kept) and a tiny stray (dropped) — the core policy.
        nodes = _mk_nodes(bike=[0, 1, 2, 3], rail=[4, 5], stations=[])
        edges = pd.DataFrame(
            _bidir(0, 1, "bike", self._OVER)  # big island
            + _bidir(2, 3, "bike", self._UNDER)  # tiny stray
            + _bidir(4, 5, "rail", 40.0),
            columns=_EDGE_COLS,
        )
        surviving = set(bd._prune_components(nodes_df=nodes, edges_df=edges)[0]["osmid"])
        assert {0, 1} <= surviving and {2, 3}.isdisjoint(surviving)

    def test_rail_keeps_only_largest_component(self):
        # RAIL is strict: a smaller rail component is dropped even though it is a valid local network.
        nodes = _mk_nodes(bike=[0, 1], rail=[2, 3, 4, 5, 6], stations=[])
        edges = pd.DataFrame(
            _bidir(0, 1, "bike", self._OVER)
            + _bidir(2, 3, "rail", 40.0)
            + _bidir(3, 4, "rail", 40.0)  # big rail comp {2,3,4}
            + _bidir(5, 6, "rail", 5.0),  # small rail comp {5,6} → dropped
            columns=_EDGE_COLS,
        )
        surviving = set(bd._prune_components(nodes_df=nodes, edges_df=edges)[0]["osmid"])
        assert {2, 3, 4} <= surviving and {5, 6}.isdisjoint(surviving)

    @pytest.mark.parametrize("station_on_big_rail", [True, False])
    def test_station_follows_its_rail_component(self, station_on_big_rail):  # noqa: ANN001
        # A station (rail node -1, wired by a RAIL link) survives IFF its rail component survives —
        # regardless of its bike entrance. Big rail comp {2,3,6,7} is kept, small comp {4,5} is dropped.
        # station_on_big_rail=True → station links to 2 (kept); False → links to 4 (dropped).
        nodes = _mk_nodes(bike=[0, 1], rail=[2, 3, 4, 5, 6, 7], stations=[-1])
        link_target = 2 if station_on_big_rail else 4
        edges = pd.DataFrame(
            _bidir(0, 1, "bike", self._OVER)  # big bike comp (kept)
            + _bidir(2, 3, "rail", 40.0)
            + _bidir(3, 6, "rail", 40.0)
            + _bidir(6, 7, "rail", 40.0)  # big rail comp {2,3,6,7} → kept
            + _bidir(4, 5, "rail", 5.0)  # small rail comp {4,5} → dropped (even with station it's 3<4)
            + _bidir(-1, link_target, "rail", 0.05)  # station's RAIL link to one comp
            + _bidir(0, -1, "station", 0.1),  # entrance (kept bike node 0) ↔ station
            columns=_EDGE_COLS,
        )
        kept_nodes, kept_edges = bd._prune_components(nodes_df=nodes, edges_df=edges)
        surviving = set(kept_nodes["osmid"])
        n_station_edges = int((kept_edges["mode"] == "station").sum())
        if station_on_big_rail:
            assert -1 in surviving and n_station_edges == 2  # station + both station-edge dirs kept
        else:
            assert -1 not in surviving and n_station_edges == 0  # dropped with its small rail comp

    @pytest.mark.parametrize(
        ("nodes", "edges", "match"),
        [
            (_mk_nodes(bike=[0, 1], rail=[], stations=[]), _bidir(0, 1, "bike", _OVER), "no rail edges"),
            (_mk_nodes(bike=[], rail=[0, 1], stations=[]), _bidir(0, 1, "rail", 40.0), "no bike edges"),
        ],
    )
    def test_empty_layer_raises(self, nodes, edges, match):  # noqa: ANN001
        # Both bike AND rail are mandatory — an empty layer is a corrupt build → fail loud.
        with pytest.raises(ValueError, match=match):
            bd._prune_components(nodes_df=nodes, edges_df=pd.DataFrame(edges, columns=_EDGE_COLS))


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
    # Bbox-split siblings (austria 3-way, switzerland 2-way) must share ONE pbf download
    # (grouped by geofabrik_path) so Phase 1 fetches each country once. Every region key is unique.
    # The seam-overlap geometry itself is owned by _assert_split_overlaps (asserted below) — not
    # re-derived here, to avoid a second, drift-prone copy of the overlap math.
    regions = bd.DACH_REGIONS
    assert len({r.key for r in regions}) == len(regions)  # unique keys
    by_pbf: dict[str, list[bd.Region]] = {}
    for r in regions:
        if r.bbox is not None:
            by_pbf.setdefault(r.geofabrik_path, []).append(r)
    assert {p.rsplit("/", 1)[-1] for p in by_pbf} == {"austria", "switzerland"}
    assert sorted(len(v) for v in by_pbf.values()) == [2, 3]  # switzerland 2-way, austria 3-way
    for slices in by_pbf.values():
        assert len({r.pbf_name for r in slices}) == 1  # each split group shares ONE pbf download
    bd._assert_split_overlaps(regions)  # single source of truth for the seam-overlap invariant


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


# ===================== _assert_split_overlaps (split-config invariant) =====================


def _R(key, bbox):  # noqa: ANN001, ANN202
    """A split Region sharing one pbf with the given bbox (None → a whole, non-split region)."""
    return bd.Region(key=key, geofabrik_path="country/shared", bbox=bbox)


class TestRectangularTilingAxisAlignment:
    """FIRST gate: tiles must be an aligned band — exactly ONE axis may differ between two siblings.

    If BOTH lat and lon ranges differ the pair is ragged/diagonal (not a rectangular tiling) and is
    rejected BEFORE any overlap is measured. Verified from both the lon-split and lat-split framing,
    then the two clean single-axis splits that must pass.
    """

    def test_rejects_lon_split_when_lat_also_differs(self):
        # Meant as an E/W (lon) split, but the lat ranges ALSO differ → >1 axis differs → reject.
        with pytest.raises(AssertionError, match="not a rectangular tiling"):
            bd._assert_split_overlaps([_R("w", (5.0, 45.0, 8.5, 48.0)), _R("e", (8.0, 45.5, 11.0, 48.0))])

    def test_rejects_lat_split_when_lon_also_differs(self):
        # Meant as an N/S (lat) split, but the lon ranges ALSO differ → >1 axis differs → reject.
        with pytest.raises(AssertionError, match="not a rectangular tiling"):
            bd._assert_split_overlaps([_R("s", (5.0, 45.0, 8.0, 47.5)), _R("n", (5.5, 47.0, 8.5, 49.0))])

    def test_rejects_diagonal_both_axes_offset(self):
        # Both lon AND lat shifted (diagonal tiles) → neither axis aligned → reject.
        with pytest.raises(AssertionError, match="not a rectangular tiling"):
            bd._assert_split_overlaps([_R("a", (5.0, 45.0, 8.5, 47.5)), _R("b", (8.0, 47.0, 11.0, 49.0))])

    def test_passes_clean_lon_band_split(self):
        # Only lon differs; lat range identical → clean E/W band → accepted.
        bd._assert_split_overlaps([_R("w", (5.0, 45.0, 8.5, 48.0)), _R("e", (8.0, 45.0, 11.0, 48.0))])

    def test_passes_clean_lat_band_split(self):
        # Only lat differs; lon range identical → clean N/S band → accepted.
        bd._assert_split_overlaps([_R("s", (5.0, 45.0, 8.0, 47.5)), _R("n", (5.0, 47.0, 8.0, 49.0))])


class TestSplitOverlap:
    """SECOND gate (once alignment holds): the split axis must overlap ≥0.5° — no gap, touch, or sliver.

    The FULL set of overlap cases exists SYMMETRICALLY for BOTH axes — lon (E/W bands) and lat (N/S
    bands): a gap fails, an exact touch (zero) fails, a too-small sliver fails, the exact 0.5° minimum
    passes, and a generous overlap passes. Plus the unordered 3-way, whole-region/singleton no-ops,
    and the real shipped DACH config.
    """

    # ---- longitude (E/W band) split: identical lat range, lon offset ----

    def test_lon_fails_on_gap(self):
        # A GAP (west ends 8.0, east starts 8.3) → negative overlap → fail loud.
        with pytest.raises(AssertionError, match="lon overlap"):
            bd._assert_split_overlaps([_R("w", (5.0, 45.0, 8.0, 48.0)), _R("e", (8.3, 45.0, 11.0, 48.0))])

    def test_lon_fails_on_exact_touch(self):
        # Slices that merely TOUCH (edges equal at 8.0) → 0 overlap → fail.
        with pytest.raises(AssertionError, match="lon overlap"):
            bd._assert_split_overlaps([_R("w", (5.0, 45.0, 8.0, 48.0)), _R("e", (8.0, 45.0, 11.0, 48.0))])

    def test_lon_fails_on_insufficient_overlap(self):
        # Overlap of only 0.3 deg (< 0.5 min) → fail; message reports the actual overlap.
        with pytest.raises(AssertionError, match="lon overlap 0.30"):
            bd._assert_split_overlaps([_R("w", (5.0, 45.0, 8.3, 48.0)), _R("e", (8.0, 45.0, 11.0, 48.0))])

    def test_lon_passes_exact_minimum_overlap(self):
        # Overlap of EXACTLY 0.5 deg (west ends 8.5, east starts 8.0) → boundary → passes.
        bd._assert_split_overlaps([_R("w", (5.0, 45.0, 8.5, 48.0)), _R("e", (8.0, 45.0, 11.0, 48.0))])

    def test_lon_passes_generous_overlap(self):
        # Overlap of 2.0 deg (west ends 10.0, east starts 8.0) → comfortably ≥0.5 → passes.
        bd._assert_split_overlaps([_R("w", (5.0, 45.0, 10.0, 48.0)), _R("e", (8.0, 45.0, 13.0, 48.0))])

    # ---- latitude (N/S band) split: identical lon range, lat offset (SYMMETRIC to the above) ----

    def test_lat_fails_on_gap(self):
        # A GAP (south ends 47.0, north starts 47.3) → negative overlap → fail loud.
        with pytest.raises(AssertionError, match="lat overlap"):
            bd._assert_split_overlaps([_R("s", (5.0, 45.0, 8.0, 47.0)), _R("n", (5.0, 47.3, 8.0, 49.0))])

    def test_lat_fails_on_exact_touch(self):
        # Bands that merely TOUCH (edges equal at 47.0) → 0 overlap → fail.
        with pytest.raises(AssertionError, match="lat overlap"):
            bd._assert_split_overlaps([_R("s", (5.0, 45.0, 8.0, 47.0)), _R("n", (5.0, 47.0, 8.0, 49.0))])

    def test_lat_fails_on_insufficient_overlap(self):
        # Overlap of only 0.3 deg (< 0.5 min) → fail; message reports the actual overlap.
        with pytest.raises(AssertionError, match="lat overlap 0.30"):
            bd._assert_split_overlaps([_R("s", (5.0, 45.0, 8.0, 47.3)), _R("n", (5.0, 47.0, 8.0, 49.0))])

    def test_lat_passes_exact_minimum_overlap(self):
        # Overlap of EXACTLY 0.5 deg (south ends 47.5, north starts 47.0) → boundary → passes.
        bd._assert_split_overlaps([_R("s", (5.0, 45.0, 8.0, 47.5)), _R("n", (5.0, 47.0, 8.0, 49.0))])

    def test_lat_passes_generous_overlap(self):
        # Overlap of 2.0 deg (south ends 49.0, north starts 47.0) → comfortably ≥0.5 → passes.
        bd._assert_split_overlaps([_R("s", (5.0, 45.0, 8.0, 49.0)), _R("n", (5.0, 47.0, 8.0, 51.0))])

    # ---- axis-agnostic: multi-slice ordering, no-ops, and the shipped config ----

    def test_passes_three_way_unordered(self):
        # Three slices passed OUT of west→east order → sorted internally; each seam overlaps ≥0.5.
        bd._assert_split_overlaps(
            [_R("e", (13.0, 46.0, 17.0, 49.0)), _R("w", (9.0, 46.0, 13.5, 49.0)), _R("c", (13.0, 46.0, 15.5, 49.0))]
        )

    def test_ignores_whole_regions_and_singletons(self):
        # bbox=None regions skipped; a lone split slice has no seam → nothing to assert (no raise).
        bd._assert_split_overlaps([_R("whole", None), _R("only", (5.0, 45.0, 8.0, 48.0))])

    def test_real_dach_config_is_valid(self):
        # The shipped DACH_REGIONS must satisfy the invariant (this is what runs at import time).
        bd._assert_split_overlaps(bd.DACH_REGIONS)
