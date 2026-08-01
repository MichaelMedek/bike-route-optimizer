"""regions tests — DACH split config + Phase-3 combine/prune/validate (synthetic artifacts).

One test_<fn> per production symbol (exact-name mirror) and TestRegion for the dataclass. Builds
tiny per-region artifacts on disk (no pbf/DEM) to exercise the cumulative-offset combine + seam
dedup and the component prune, plus the split-overlap invariant. Folds the former test_build_dach.py.
"""

import pandas as pd
import pytest

from bike_router.core.constants import GraphConfig
from bike_router.preprocessing.graph_writer import write_graph_parquet
from bike_router.preprocessing.regions import (
    DACH_REGIONS,
    Region,
    _assert_rectangular_tiling,
    _assert_split_overlaps,
    assert_all_regions_complete,
    base_meta,
    combine_regions,
    prune_components,
    region_complete,
    split_geofabrik_path,
)

_NODE_COLS = ["osmid", "lat", "lon", "elevation_m", "node_type", "station_name"]
_EDGE_COLS = ["from_node", "to_node", "key", "length_m", "height_diff_m", "surface", "highway", "mode", "geometry_wkt"]


def _region_artifact(region_dir, nodes: list[tuple], edges: list[tuple]) -> None:  # noqa: ANN001
    """Write a minimal confirmed_complete per-region artifact to region_dir."""
    nodes_df = pd.DataFrame(nodes, columns=_NODE_COLS)
    edges_df = pd.DataFrame(edges, columns=_EDGE_COLS)
    meta = {"tile_deg": GraphConfig.TILE_DEG, "confirmed_complete": True}
    write_graph_parquet(nodes_df=nodes_df, edges_df=edges_df, meta=meta, out_dir=region_dir, compression="snappy")


def _line(lat1, lon1, lat2, lon2) -> str:  # noqa: ANN001
    return f"LINESTRING ({lon1} {lat1}, {lon2} {lat2})"


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


def _R(key, bbox):  # noqa: ANN001, ANN202
    """A split Region sharing one pbf with the given bbox (None → a whole, non-split region)."""
    return Region(key=key, geofabrik_path="country/shared", bbox=bbox)


# --- split config ------------------------------------------------------------


def test_split_geofabrik_path():
    # The shared-pbf cache name is the Geofabrik leaf: nested German paths and bare country paths.
    assert split_geofabrik_path(geofabrik_path="germany/baden-wuerttemberg/freiburg-regbez") == "freiburg-regbez"
    assert split_geofabrik_path(geofabrik_path="austria") == "austria"


class TestRegion:
    def test_pbf_name_is_the_geofabrik_leaf_shared_across_splits(self):
        # Two bbox-split halves of one country share ONE pbf download (same leaf name).
        west = Region(key="austria-west", geofabrik_path="austria", bbox=(9.4, 46.3, 13.5, 49.1))
        east = Region(key="austria-east", geofabrik_path="austria", bbox=(15.3, 46.3, 17.25, 49.1))
        assert west.pbf_name == east.pbf_name == "austria"

    def test_is_frozen(self):
        with pytest.raises(AttributeError):
            Region(key="x", geofabrik_path="y").key = "z"  # type: ignore[misc]

    def test_shipped_dach_regions_have_unique_keys_and_share_split_pbfs(self):
        # Bbox-split siblings must share ONE pbf (grouped by geofabrik_path); every region key unique.
        assert len({r.key for r in DACH_REGIONS}) == len(DACH_REGIONS)
        by_pbf: dict[str, list[Region]] = {}
        for r in DACH_REGIONS:
            if r.bbox is not None:
                by_pbf.setdefault(r.geofabrik_path, []).append(r)
        assert {p.rsplit("/", 1)[-1] for p in by_pbf} == {"austria", "switzerland"}
        assert sorted(len(v) for v in by_pbf.values()) == [2, 3]  # switzerland 2-way, austria 3-way
        for slices in by_pbf.values():
            assert len({r.pbf_name for r in slices}) == 1  # each split group shares ONE pbf download


def test_assert_rectangular_tiling():
    # Two clean bands pass (only one axis differs); a pair differing on BOTH axes is ragged → reject.
    _assert_rectangular_tiling(  # clean E/W band: identical lat, lon overlaps ≥0.5
        pbf="p", a_key="w", a=(5.0, 45.0, 8.5, 48.0), b_key="e", b=(8.0, 45.0, 11.0, 48.0)
    )
    with pytest.raises(AssertionError, match="not a rectangular tiling"):
        _assert_rectangular_tiling(pbf="p", a_key="a", a=(5.0, 45.0, 8.5, 47.5), b_key="b", b=(8.0, 47.0, 11.0, 49.0))
    with pytest.raises(AssertionError, match="lon overlap"):  # aligned lat, but a gap on lon
        _assert_rectangular_tiling(pbf="p", a_key="w", a=(5.0, 45.0, 8.0, 48.0), b_key="e", b=(8.3, 45.0, 11.0, 48.0))


def test_assert_split_overlaps():
    # The FULL split-overlap invariant: axis alignment FIRST, then ≥0.5° overlap on the split axis,
    # symmetric for lon (E/W) and lat (N/S) bands. Plus 3-way ordering, no-ops, and the shipped config.
    with pytest.raises(AssertionError, match="not a rectangular tiling"):  # lon-split, lat also differs
        _assert_split_overlaps([_R("w", (5.0, 45.0, 8.5, 48.0)), _R("e", (8.0, 45.5, 11.0, 48.0))])
    with pytest.raises(AssertionError, match="not a rectangular tiling"):  # both axes offset (diagonal)
        _assert_split_overlaps([_R("a", (5.0, 45.0, 8.5, 47.5)), _R("b", (8.0, 47.0, 11.0, 49.0))])

    # lon (E/W) band: gap / exact-touch / sliver fail; exact 0.5 + generous pass.
    with pytest.raises(AssertionError, match="lon overlap"):
        _assert_split_overlaps([_R("w", (5.0, 45.0, 8.0, 48.0)), _R("e", (8.3, 45.0, 11.0, 48.0))])
    with pytest.raises(AssertionError, match="lon overlap"):
        _assert_split_overlaps([_R("w", (5.0, 45.0, 8.0, 48.0)), _R("e", (8.0, 45.0, 11.0, 48.0))])
    with pytest.raises(AssertionError, match="lon overlap 0.30"):
        _assert_split_overlaps([_R("w", (5.0, 45.0, 8.3, 48.0)), _R("e", (8.0, 45.0, 11.0, 48.0))])
    _assert_split_overlaps([_R("w", (5.0, 45.0, 8.5, 48.0)), _R("e", (8.0, 45.0, 11.0, 48.0))])  # exact 0.5 → ok
    _assert_split_overlaps([_R("w", (5.0, 45.0, 10.0, 48.0)), _R("e", (8.0, 45.0, 13.0, 48.0))])  # generous → ok

    # lat (N/S) band: SYMMETRIC — gap / exact-touch / sliver fail; exact 0.5 passes.
    with pytest.raises(AssertionError, match="lat overlap"):
        _assert_split_overlaps([_R("s", (5.0, 45.0, 8.0, 47.0)), _R("n", (5.0, 47.3, 8.0, 49.0))])
    with pytest.raises(AssertionError, match="lat overlap"):
        _assert_split_overlaps([_R("s", (5.0, 45.0, 8.0, 47.0)), _R("n", (5.0, 47.0, 8.0, 49.0))])
    with pytest.raises(AssertionError, match="lat overlap 0.30"):
        _assert_split_overlaps([_R("s", (5.0, 45.0, 8.0, 47.3)), _R("n", (5.0, 47.0, 8.0, 49.0))])
    _assert_split_overlaps([_R("s", (5.0, 45.0, 8.0, 47.5)), _R("n", (5.0, 47.0, 8.0, 49.0))])  # exact 0.5

    # 3-way out of order → sorted internally; whole-region/singleton no-ops; the shipped config holds.
    _assert_split_overlaps(
        [_R("e", (13.0, 46.0, 17.0, 49.0)), _R("w", (9.0, 46.0, 13.5, 49.0)), _R("c", (13.0, 46.0, 15.5, 49.0))]
    )
    _assert_split_overlaps([_R("whole", None), _R("only", (5.0, 45.0, 8.0, 48.0))])  # no seam → no-op
    _assert_split_overlaps(DACH_REGIONS)  # the shipped config must satisfy the invariant


# --- completion gate ---------------------------------------------------------


def test_region_complete(tmp_path):
    # Only a present meta.json flagged confirmed_complete==true counts as done.
    assert region_complete(regions_dir=tmp_path, region_key="absent") is False  # no dir
    (tmp_path / "partial").mkdir()
    assert region_complete(regions_dir=tmp_path, region_key="partial") is False  # dir, no meta
    (tmp_path / "unflagged").mkdir()
    (tmp_path / "unflagged" / GraphConfig.META_FILENAME).write_text('{"n_nodes": 5}')  # meta, flag absent
    assert region_complete(regions_dir=tmp_path, region_key="unflagged") is False
    (tmp_path / "ok").mkdir()
    (tmp_path / "ok" / GraphConfig.META_FILENAME).write_text('{"confirmed_complete": true}')
    assert region_complete(regions_dir=tmp_path, region_key="ok") is True


def test_assert_all_regions_complete(tmp_path):
    (tmp_path / "done").mkdir()
    (tmp_path / "done" / GraphConfig.META_FILENAME).write_text('{"confirmed_complete": true}')
    assert_all_regions_complete(regions_dir=tmp_path, regions=["done"])  # all complete → no raise
    with pytest.raises(ValueError, match="missing"):  # a region with no artifact → gate names it
        assert_all_regions_complete(regions_dir=tmp_path, regions=["done", "missing"])


def test_base_meta():
    # The meta keys common to per-region + combined artifacts: bbox, tile grid, tolerance, counts.
    nodes_df = pd.DataFrame([(0, 48.0, 8.0, 0.0, "bike", None), (1, 48.2, 8.2, 0.0, "rail", "S")], columns=_NODE_COLS)
    edges_df = pd.DataFrame([(0, 1, 0, 1.0, 0.0, None, None, "station", None)], columns=_EDGE_COLS)
    meta = base_meta(nodes_df=nodes_df, edges_df=edges_df, tolerance_m=25.0)
    assert meta["bbox"] == [8.0, 48.0, 8.2, 48.2] and meta["tile_deg"] == GraphConfig.TILE_DEG
    assert meta["tolerance_m"] == 25.0 and meta["n_nodes"] == 2 and meta["n_edges"] == 1
    assert meta["n_stations"] == 1  # one rail node


# --- combine / prune ---------------------------------------------------------


def test_combine_regions(tmp_path):
    # Two regions with contiguous ids that would COLLIDE; region B's node 0 is a seam duplicate of
    # A's node 1 (same lat/lon). Combine offsets, dedups the seam, prunes, and re-densifies. A tiny
    # node ceiling trips the sanity guard.
    regions_dir = tmp_path / "per_region"
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
    nodes_df, edges_df = combine_regions(regions_dir=regions_dir, regions=["a", "b"])
    assert len(nodes_df) == 5 and sorted(nodes_df["osmid"]) == [0, 1, 2, 3, 4]  # 6-1 seam dup, dense
    assert len(edges_df) == 6  # 4 bike (no false dedup) + 2 rail
    assert set(edges_df["from_node"]) | set(edges_df["to_node"]) <= set(nodes_df["osmid"])

    ceiling_dir = tmp_path / "ceiling"
    _region_artifact(
        ceiling_dir / "a",
        nodes=[(0, 48.0, 8.0, 0.0, "bike", None), (1, 48.2, 8.2, 0.0, "bike", None), (2, 48.4, 8.4, 0.0, "bike", None)],
        edges=[(0, 1, 0, 1.0, 0.0, "asphalt", "residential", "bike", None)],
    )
    import bike_router.preprocessing.regions as regions_mod

    monkey = regions_mod._MAX_TOTAL_NODES
    regions_mod._MAX_TOTAL_NODES = 2  # tiny ceiling to trip deterministically
    try:
        with pytest.raises(ValueError, match="ceiling"):
            combine_regions(regions_dir=ceiling_dir, regions=["a"])
    finally:
        regions_mod._MAX_TOTAL_NODES = monkey


def test_prune_components():
    # RAIL keeps ONLY its largest weakly-connected component; BIKE keeps every component ≥
    # MIN_BIKE_COMPONENT_KM; both layers mandatory. Full boundary/station cases in TestPruneComponents.
    over = GraphConfig.MIN_BIKE_COMPONENT_KM + 5
    nodes = _mk_nodes(bike=[0, 1, 2, 3], rail=[4, 5, 6], stations=[])
    edges = pd.DataFrame(
        _bidir(0, 1, "bike", over)  # big bike island → kept
        + _bidir(2, 3, "bike", 1.0)  # tiny bike stray → dropped
        + _bidir(4, 5, "rail", 40.0)  # big rail comp {4,5} → kept
        + _bidir(6, 6, "rail", 5.0),  # isolated rail self-loop node → dropped (not largest)
        columns=_EDGE_COLS,
    )
    surviving = set(prune_components(nodes_df=nodes, edges_df=edges)[0]["osmid"])
    assert {0, 1, 4, 5} <= surviving and {2, 3}.isdisjoint(surviving)


class TestPruneComponents:
    """prune_components policy: RAIL keeps ONLY its largest weakly-connected component; BIKE keeps
    every component with total (undirected) length ≥ MIN_BIKE_COMPONENT_KM; a station follows its
    rail component; both layers are MANDATORY (an empty layer fails loud).
    """

    _OVER = GraphConfig.MIN_BIKE_COMPONENT_KM + 5
    _UNDER = 1.0

    @pytest.mark.parametrize(
        ("bike_km", "kept"),
        [
            (GraphConfig.MIN_BIKE_COMPONENT_KM - 1, False),  # below → dropped
            (GraphConfig.MIN_BIKE_COMPONENT_KM, True),  # exactly at → kept (>=)
            (GraphConfig.MIN_BIKE_COMPONENT_KM + 1, True),  # above → kept
        ],
    )
    def test_bike_component_threshold_boundary(self, bike_km, kept):  # noqa: ANN001
        nodes = _mk_nodes(bike=[0, 1], rail=[2, 3], stations=[])
        edges = pd.DataFrame(_bidir(0, 1, "bike", bike_km) + _bidir(2, 3, "rail", 40.0), columns=_EDGE_COLS)
        surviving = set(prune_components(nodes_df=nodes, edges_df=edges)[0]["osmid"])
        assert ({0, 1} <= surviving) is kept

    def test_bike_keeps_big_island_drops_small_stray(self):
        nodes = _mk_nodes(bike=[0, 1, 2, 3], rail=[4, 5], stations=[])
        edges = pd.DataFrame(
            _bidir(0, 1, "bike", self._OVER) + _bidir(2, 3, "bike", self._UNDER) + _bidir(4, 5, "rail", 40.0),
            columns=_EDGE_COLS,
        )
        surviving = set(prune_components(nodes_df=nodes, edges_df=edges)[0]["osmid"])
        assert {0, 1} <= surviving and {2, 3}.isdisjoint(surviving)

    def test_rail_keeps_only_largest_component(self):
        nodes = _mk_nodes(bike=[0, 1], rail=[2, 3, 4, 5, 6], stations=[])
        edges = pd.DataFrame(
            _bidir(0, 1, "bike", self._OVER)
            + _bidir(2, 3, "rail", 40.0)
            + _bidir(3, 4, "rail", 40.0)  # big rail comp {2,3,4}
            + _bidir(5, 6, "rail", 5.0),  # small rail comp {5,6} → dropped
            columns=_EDGE_COLS,
        )
        surviving = set(prune_components(nodes_df=nodes, edges_df=edges)[0]["osmid"])
        assert {2, 3, 4} <= surviving and {5, 6}.isdisjoint(surviving)

    @pytest.mark.parametrize("station_on_big_rail", [True, False])
    def test_station_follows_its_rail_component(self, station_on_big_rail):  # noqa: ANN001
        nodes = _mk_nodes(bike=[0, 1], rail=[2, 3, 4, 5, 6, 7], stations=[-1])
        link_target = 2 if station_on_big_rail else 4
        edges = pd.DataFrame(
            _bidir(0, 1, "bike", self._OVER)
            + _bidir(2, 3, "rail", 40.0)
            + _bidir(3, 6, "rail", 40.0)
            + _bidir(6, 7, "rail", 40.0)  # big rail comp {2,3,6,7} → kept
            + _bidir(4, 5, "rail", 5.0)  # small rail comp {4,5} → dropped
            + _bidir(-1, link_target, "rail", 0.05)
            + _bidir(0, -1, "station", 0.1),
            columns=_EDGE_COLS,
        )
        kept_nodes, kept_edges = prune_components(nodes_df=nodes, edges_df=edges)
        surviving = set(kept_nodes["osmid"])
        n_station_edges = int((kept_edges["mode"] == "station").sum())
        if station_on_big_rail:
            assert -1 in surviving and n_station_edges == 2
        else:
            assert -1 not in surviving and n_station_edges == 0

    @pytest.mark.parametrize(
        ("nodes", "edges", "match"),
        [
            (_mk_nodes(bike=[0, 1], rail=[], stations=[]), _bidir(0, 1, "bike", _OVER), "no rail edges"),
            (_mk_nodes(bike=[], rail=[0, 1], stations=[]), _bidir(0, 1, "rail", 40.0), "no bike edges"),
        ],
    )
    def test_empty_layer_raises(self, nodes, edges, match):  # noqa: ANN001
        with pytest.raises(ValueError, match=match):
            prune_components(nodes_df=nodes, edges_df=pd.DataFrame(edges, columns=_EDGE_COLS))
