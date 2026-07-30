"""RouteGraph (CSR) tests — the ONLY router: compact matrix, parallel-min, snapping, Dijkstra.

Structured as TestRouteGraph (the dataclass: from_arrays build, parallel-edge min-collapse,
bike-only snapping, dangling-edge drop) + exact-name tests for the module functions
_min_cost_matrix and shortest_path. Every original assertion is preserved.
"""

import numpy as np
import pytest

from bike_router.core.constants import NodeType
from bike_router.core.errors import NoRouteError
from bike_router.core.route_graph import RouteGraph, _min_cost_matrix, shortest_path
from tests.conftest import (
    ZERO_PARAMS,
    make_choice_edges,
    make_cutthrough_edges,
    make_line_edges,
    zero_params,
)


class TestRouteGraph:
    """The compact CSR routing graph: build from flat arrays, collapse parallels, snap bike-only."""

    def test_from_arrays_builds_and_drops_dangling_edges(self):
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

    def test_parallel_edges_collapse_to_minimum(self):
        # Two parallel 1→2 edges of different cost; the matrix keeps the cheaper.
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

    def test_snap_bike_node_ignores_rail(self):
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


def test_min_cost_matrix():
    # Collapses parallel (u,v) edges to their MINIMUM cost (csr_matrix would otherwise SUM dups),
    # matching A*'s cheapest-parallel-edge choice; an empty edge set yields an all-zero n×n matrix.
    matrix = _min_cost_matrix(
        u=np.array([0, 0, 1], dtype="int64"),
        v=np.array([1, 1, 2], dtype="int64"),
        cost=np.array([800.0, 500.0, 300.0]),
        n=3,
    )
    assert matrix[0, 1] == pytest.approx(500.0)  # cheaper of the two parallels kept, not summed
    assert matrix[1, 2] == pytest.approx(300.0)
    empty = _min_cost_matrix(u=np.array([], dtype="int64"), v=np.array([], dtype="int64"), cost=np.array([]), n=3)
    assert empty.nnz == 0 and empty.shape == (3, 3)


def test_shortest_path():
    # Dijkstra over the CSR matrix returns the optimal osmid path: the line 1→2→3 straight through;
    # on the choice graph it avoids the steep main road when penalised but takes it distance-only;
    # the boarding penalty gates cutting THROUGH a station vs a long bike detour; every station edge
    # costs >= its length; and an unreachable target raises NoRouteError (the project type, not nx).
    line = RouteGraph.from_arrays(**make_line_edges().route_graph_args(params=ZERO_PARAMS))
    assert shortest_path(route_graph=line, source_osmid=1, target_osmid=3) == [1, 2, 3]

    penalise = zero_params(extra_km_per_uphill_100m=5.0, extra_km_per_main_road_km=1.0)
    avoid = RouteGraph.from_arrays(**make_choice_edges().route_graph_args(params=penalise))
    assert shortest_path(route_graph=avoid, source_osmid=1, target_osmid=5) == [1, 3, 5]  # detour hill+road
    direct = RouteGraph.from_arrays(**make_choice_edges().route_graph_args(params=ZERO_PARAMS))
    assert shortest_path(route_graph=direct, source_osmid=1, target_osmid=5) == [1, 2, 5]  # distance-only

    # cut-through: L and R are entrances to one station S; the only bike alternative is a 10 km
    # detour. Boarding-free → the near-free station edges win (cut through L→S→R); expensive boarding
    # → a full boarding charge makes the long bike detour win instead.
    cheap = RouteGraph.from_arrays(
        **make_cutthrough_edges(detour_m=10_000.0).route_graph_args(params=zero_params(extra_km_per_boarding=0.0))
    )
    assert shortest_path(route_graph=cheap, source_osmid=1, target_osmid=2) == [1, -1, 2]
    dear = RouteGraph.from_arrays(
        **make_cutthrough_edges(detour_m=10_000.0).route_graph_args(params=zero_params(extra_km_per_boarding=50.0))
    )
    assert shortest_path(route_graph=dear, source_osmid=1, target_osmid=2) == [1, 3, 2]

    arr = make_line_edges()
    arr.add_node(99, lon=20.0, lat=60.0, elevation=0.0)  # isolated, unreachable
    unreachable = RouteGraph.from_arrays(**arr.route_graph_args(params=ZERO_PARAMS))
    with pytest.raises(NoRouteError):
        shortest_path(route_graph=unreachable, source_osmid=1, target_osmid=99)


def test_station_edge_cost_floor():
    # Admissibility floor: EVERY station edge's cost is >= its straight-line length (boarding only
    # ADDS). Equal at boarding 0, strictly greater at boarding 50 — keeps Dijkstra costs non-negative.
    for boarding in (0.0, 50.0):
        params = zero_params(extra_km_per_boarding=boarding)
        arr = make_cutthrough_edges(detour_m=10_000.0)
        costs = [(arr.edge_cost_of(u, v, params=params), 100.0) for u, v in [(1, -1), (-1, 1), (2, -1), (-1, 2)]]
        assert costs, "expected station edges in the cut-through graph"
        for cost, length in costs:
            assert cost >= length  # cost floor = geometric length
        exact = all(cost == length for cost, length in costs)
        assert exact if boarding == 0.0 else not exact
