"""Routing tests — optimal path over the tiny graphs (CSR Dijkstra) + param effects."""

import pytest

from bike_router.core.errors import NoRouteError
from bike_router.core.route_graph import RouteGraph, shortest_path
from tests.conftest import ZERO_PARAMS, make_choice_edges, make_cutthrough_edges, make_line_edges, zero_params


def _route(arr, *, params, source, target):
    """Build a CSR RouteGraph from the edge arrays and return the optimal node path."""
    rg = RouteGraph.from_arrays(**arr.route_graph_args(params=params))
    return shortest_path(route_graph=rg, source_osmid=source, target_osmid=target)


def test_shortest_route_traverses_line():
    assert _route(make_line_edges(), params=ZERO_PARAMS, source=1, target=3) == [1, 2, 3]


def test_no_path_raises():
    arr = make_line_edges()
    arr.add_node(99, lon=20.0, lat=60.0, elevation=0.0)  # isolated
    rg = RouteGraph.from_arrays(**arr.route_graph_args(params=ZERO_PARAMS))
    with pytest.raises(NoRouteError):
        shortest_path(route_graph=rg, source_osmid=1, target_osmid=99)


def test_params_change_the_chosen_path():
    """With penalties on, the router avoids the steep paved main road (node 2) and
    takes the flat detour (node 3); distance-only picks the short node-2 path.
    """
    penalise = zero_params(extra_km_per_uphill_100m=5.0, extra_km_per_main_road_km=1.0)
    avoid = _route(make_choice_edges(), params=penalise, source=1, target=5)
    direct = _route(make_choice_edges(), params=ZERO_PARAMS, source=1, target=5)
    assert avoid == [1, 3, 5]  # detours around hill + main road
    assert direct == [1, 2, 5]  # shortest ignores hill/road


def test_cutthrough_taken_when_detour_is_long_and_boarding_free():
    # L and R are entrances to one station S; the only pedalled alternative is a 10 km
    # detour L→M→R. With boarding 0 the two station edges are nearly free, so the router
    # cuts THROUGH the station (L→S→R) rather than ride the long way — the accepted tradeoff.
    arr = make_cutthrough_edges(detour_m=10_000.0)
    assert _route(arr, params=zero_params(extra_km_per_boarding=0.0), source=1, target=2) == [1, -1, 2]


def test_cutthrough_avoided_when_boarding_is_expensive():
    # Same geometry, but a high boarding penalty makes passing through S cost a full boarding
    # (½ + ½), so the long bike detour wins — no cut-through.
    arr = make_cutthrough_edges(detour_m=10_000.0)
    assert _route(arr, params=zero_params(extra_km_per_boarding=50.0), source=1, target=2) == [1, 3, 2]


def test_station_edge_cost_never_below_length_keeps_costs_nonnegative():
    # Admissibility floor: EVERY station edge's cost must be >= its straight-line length (the
    # boarding term only ADDS). True at boarding 0 (equal) and boarding 50 (strictly greater).
    for boarding in (0.0, 50.0):
        params = zero_params(extra_km_per_boarding=boarding)
        arr = make_cutthrough_edges()
        costs = [(arr.edge_cost_of(u, v, params=params), 100.0) for u, v in [(1, -1), (-1, 1), (2, -1), (-1, 2)]]
        assert costs, "expected station edges in the cut-through graph"
        for cost, length in costs:
            assert cost >= length  # cost floor = geometric length
        exact = all(cost == length for cost, length in costs)
        assert exact if boarding == 0.0 else not exact
