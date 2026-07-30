"""RouteGraph (CSR) tests — parity with the analytic path, snapping, cost array, parallel min.

The CSR engine is the ONLY router; these pin that a route over the compact matrix equals the
expected shortest path, that parallel edges collapse to the minimum, that endpoint snapping
stays bike-only, that the array cost matches the per-edge formula, and that dangling edges drop.
"""

import numpy as np
import pandas as pd
import pytest

from bike_router.core.constants import Mode, NodeType
from bike_router.core.cost import edge_cost_array
from bike_router.core.errors import NoRouteError
from bike_router.core.route_graph import RouteGraph, shortest_path
from tests.conftest import DEFAULT_PARAMS, ZERO_PARAMS, make_choice_edges, make_line_edges, zero_params


def test_from_arrays_route_matches_reference_line():
    rg = RouteGraph.from_arrays(**make_line_edges().route_graph_args(params=ZERO_PARAMS))
    assert shortest_path(route_graph=rg, source_osmid=1, target_osmid=3) == [1, 2, 3]


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        (zero_params(extra_km_per_uphill_100m=5.0, extra_km_per_main_road_km=1.0), [1, 3, 5]),  # avoid hill+road
        (ZERO_PARAMS, [1, 2, 5]),  # distance-only takes the short steep path
    ],
)
def test_csr_route_picks_expected_path(params, expected):
    rg = RouteGraph.from_arrays(**make_choice_edges().route_graph_args(params=params))
    assert shortest_path(route_graph=rg, source_osmid=1, target_osmid=5) == expected


def test_shortest_path_raises_when_unreachable():
    arr = make_line_edges()
    arr.add_node(99, lon=20.0, lat=60.0, elevation=0.0)
    rg = RouteGraph.from_arrays(**arr.route_graph_args(params=ZERO_PARAMS))
    with pytest.raises(NoRouteError):
        shortest_path(route_graph=rg, source_osmid=1, target_osmid=99)


def test_parallel_edges_collapse_to_minimum():
    # Two parallel 1→2 edges of different cost; the matrix must keep the cheaper.
    rg = RouteGraph.from_arrays(
        osmids=np.array([1, 2], dtype="int64"),
        lat=np.array([48.0, 48.0]),
        lon=np.array([8.0, 8.01]),
        node_type=np.array([NodeType.BIKE, NodeType.BIKE], dtype=object),
        from_osmid=np.array([1, 1], dtype="int64"),
        to_osmid=np.array([2, 2], dtype="int64"),
        cost=np.array([800.0, 500.0]),  # cheaper parallel second
    )
    assert rg.matrix[rg.index[1], rg.index[2]] == pytest.approx(500.0)


def test_snap_bike_node_ignores_rail():
    # A rail node sits exactly on the query point; snap must still pick the nearest BIKE node.
    rg = RouteGraph.from_arrays(
        osmids=np.array([1, 2, -1], dtype="int64"),
        lat=np.array([48.0, 48.0, 48.0]),
        lon=np.array([8.000, 8.020, 8.010]),
        node_type=np.array([NodeType.BIKE, NodeType.BIKE, NodeType.RAIL], dtype=object),
        from_osmid=np.array([1], dtype="int64"),
        to_osmid=np.array([2], dtype="int64"),
        cost=np.array([1500.0]),
    )
    snapped = rg.snap_bike_node(lat=48.0, lon=8.010)  # nearest overall is the rail node
    assert snapped != -1 and rg.node_type[rg.index[snapped]] == NodeType.BIKE


def test_edge_cost_array_costs_a_whole_table():
    """The vectorized array cost applies the penalty formula per row (uphill costs more)."""
    edges_df = pd.DataFrame(
        {
            "from_node": [1, 2, 2],
            "to_node": [2, 1, 3],
            "length_m": [800.0, 800.0, 800.0],
            "surface": ["asphalt", "asphalt", "gravel"],
            "highway": ["residential", "residential", "secondary"],
            "mode": [Mode.BIKE, Mode.BIKE, Mode.BIKE],
        }
    )
    elev = {1: 100.0, 2: 130.0, 3: 100.0}
    got = edge_cost_array(edges_df=edges_df, elev_by_osmid=elev, params=DEFAULT_PARAMS)
    assert got[0] > got[1]  # 1→2 climbs, 2→1 descends → uphill costs more
    assert got[2] > 800.0  # gravel + main road → penalised above raw length
    assert (got >= 800.0).all()  # never below raw length


def test_from_arrays_drops_edges_with_missing_endpoint():
    # An edge referencing an osmid outside the node set dangles off the window → dropped.
    rg = RouteGraph.from_arrays(
        osmids=np.array([1, 2], dtype="int64"),
        lat=np.array([48.0, 48.0]),
        lon=np.array([8.0, 8.01]),
        node_type=np.array([NodeType.BIKE, NodeType.BIKE], dtype=object),
        from_osmid=np.array([1, 2], dtype="int64"),
        to_osmid=np.array([2, 99], dtype="int64"),  # 99 not present
        cost=np.array([800.0, 800.0]),
    )
    assert rg.n_edges == 1  # only the 1→2 edge survives
