"""graph_validation tests — the STRICT bike-edge geometry invariants that gate the build.

One test_<fn> per production symbol. Each folds the pass case + the loud-fail case, and confirms
rail/station edges are exempt from both invariants.
"""

import pandas as pd
import pytest

from bike_router.core.constants import Mode, Schema
from bike_router.core.graph_validation import assert_bike_geometry_valid, assert_no_long_straight_edges


def _edge_row(*, from_node: int, to_node: int, mode: str, wkt: str | None) -> dict:
    """One on-disk edge row for the validator (only the columns it reads)."""
    return {Schema.FROM_NODE: from_node, Schema.TO_NODE: to_node, Schema.MODE: mode, Schema.GEOMETRY_WKT: wkt}


def test_assert_no_long_straight_edges():
    # A long edge with a real multi-vertex polyline passes; a >1 km straight 2-point jump fails loud;
    # short 2-point edges (≤1 km) are fine (a straight hop over a small gap is legitimate).
    def row(f, t, length_m, wkt):
        return {**_edge_row(from_node=f, to_node=t, mode=Mode.RAIL, wkt=wkt), Schema.LENGTH_M: length_m}

    ok = pd.DataFrame(
        [
            row(1, 2, 5000.0, "LINESTRING (8.0 48.0, 8.02 48.0, 8.04 48.0)"),  # 5 km but traced → OK
            row(3, 4, 100.0, "LINESTRING (8.0 48.0, 8.001 48.0)"),  # 100 m straight → OK (≤ threshold)
        ]
    )
    assert_no_long_straight_edges(edges_df=ok)  # no raise
    bad = pd.DataFrame([row(5, 6, 3000.0, "LINESTRING (8.0 48.0, 8.04 48.0)")])  # 3 km straight jump → fail
    with pytest.raises(AssertionError, match="trace real line geometry"):
        assert_no_long_straight_edges(edges_df=bad)


def test_assert_bike_geometry_valid():
    # A clean dense-and-in-band bike edge passes; a sparse-gap bike edge and an out-of-band bike edge
    # each fail LOUD; the SAME violations on rail/station edges are exempt (trains tunnel/bridge).
    nodes = pd.DataFrame({Schema.OSMID: [1, 2], Schema.ELEVATION_M: [100.0, 200.0]})
    good = pd.DataFrame(
        [
            _edge_row(
                from_node=1,
                to_node=2,
                mode=Mode.BIKE,
                wkt="LINESTRING Z (8.0 48.0 100, 8.0005 48.0 150, 8.001 48.0 200)",
            )
        ]
    )
    assert_bike_geometry_valid(nodes_df=nodes, edges_df=good)  # no raise

    # sparse: ~743 m single gap > 100 m max
    sparse = pd.DataFrame(
        [_edge_row(from_node=1, to_node=2, mode=Mode.BIKE, wkt="LINESTRING Z (8.0 48.0 100, 8.01 48.0 200)")]
    )
    with pytest.raises(AssertionError, match="vertex gap"):
        assert_bike_geometry_valid(nodes_df=nodes, edges_df=sparse)

    # out of band: dense but midpoint z=0 (band [100,200], margin 30 → 100 m under)
    dipped = pd.DataFrame(
        [
            _edge_row(
                from_node=1, to_node=2, mode=Mode.BIKE, wkt="LINESTRING Z (8.0 48.0 100, 8.0005 48.0 0, 8.001 48.0 200)"
            )
        ]
    )
    with pytest.raises(AssertionError, match="outside its endpoint band"):
        assert_bike_geometry_valid(nodes_df=nodes, edges_df=dipped)

    # the SAME sparse+dipped geometry on a RAIL edge is exempt — no raise
    rail = pd.DataFrame(
        [_edge_row(from_node=1, to_node=2, mode=Mode.RAIL, wkt="LINESTRING Z (8.0 48.0 100, 8.01 48.0 0)")]
    )
    assert_bike_geometry_valid(nodes_df=nodes, edges_df=rail)


def test_assert_bike_geometry_valid_reports_worst_across_many_edges():
    # With many valid edges, the failure names the WORST offender.
    nodes = pd.DataFrame({Schema.OSMID: [1, 2, 3], Schema.ELEVATION_M: [100.0, 100.0, 100.0]})
    edges = pd.DataFrame(
        [
            _edge_row(from_node=1, to_node=2, mode=Mode.BIKE, wkt="LINESTRING Z (8.0 48.0 100, 8.0005 48.0 100)"),  # ok
            _edge_row(
                from_node=2, to_node=3, mode=Mode.BIKE, wkt="LINESTRING Z (8.0 48.0 100, 8.02 48.0 100)"
            ),  # ~1486 m gap
        ]
    )
    with pytest.raises(AssertionError, match=r"\(2, 3\)"):  # names the worst-gap edge
        assert_bike_geometry_valid(nodes_df=nodes, edges_df=edges)

    # a bike edge with NO geometry is corrupt → fails loud (never silently skipped)
    missing = pd.DataFrame([_edge_row(from_node=1, to_node=2, mode=Mode.BIKE, wkt=None)])
    with pytest.raises(AssertionError, match="no geometry"):
        assert_bike_geometry_valid(nodes_df=nodes, edges_df=missing)


def test_assert_bike_geometry_valid_band_boundary():
    # A z exactly at the endpoint band (no overshoot) passes even with a big node-to-node climb.
    nodes = pd.DataFrame({Schema.OSMID: [1, 2], Schema.ELEVATION_M: [100.0, 500.0]})
    ok = pd.DataFrame(
        [
            _edge_row(
                from_node=1,
                to_node=2,
                mode=Mode.BIKE,
                wkt="LINESTRING Z (8.0 48.0 100, 8.0005 48.0 300, 8.001 48.0 500)",
            )
        ]
    )
    assert_bike_geometry_valid(nodes_df=nodes, edges_df=ok)  # monotone in-band climb → no raise


def test_assert_bike_geometry_valid_within_margin_slack():
    # A z modestly outside the endpoint band but WITHIN the DEM margin passes (real DEM noise tolerance);
    # pushing it past the margin fails. Band [100,100], margin 30 → z=125 ok, z=200 fails.
    nodes = pd.DataFrame({Schema.OSMID: [1, 2], Schema.ELEVATION_M: [100.0, 100.0]})
    slack = pd.DataFrame(
        [
            _edge_row(
                from_node=1,
                to_node=2,
                mode=Mode.BIKE,
                wkt="LINESTRING Z (8.0 48.0 100, 8.0005 48.0 125, 8.001 48.0 100)",
            )
        ]
    )
    assert_bike_geometry_valid(nodes_df=nodes, edges_df=slack)  # +25 m ≤ 30 m margin → no raise
    over = pd.DataFrame(
        [
            _edge_row(
                from_node=1,
                to_node=2,
                mode=Mode.BIKE,
                wkt="LINESTRING Z (8.0 48.0 100, 8.0005 48.0 200, 8.001 48.0 100)",
            )
        ]
    )
    with pytest.raises(AssertionError, match="outside its endpoint band"):
        assert_bike_geometry_valid(nodes_df=nodes, edges_df=over)
