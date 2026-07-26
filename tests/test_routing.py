"""Routing tests — A* over the tiny graphs + admissible heuristic + param effects."""

import networkx as nx
import pytest

from bike_router.constants import CostConfig, Mode
from bike_router.routing import make_heuristic, shortest_route
from tests.conftest import ZERO_PARAMS, make_choice_graph, make_cutthrough_graph, make_line_graph, zero_params


def test_shortest_route_traverses_line():
    graph = make_line_graph()
    assert shortest_route(graph=graph, source=1, target=3) == [1, 2, 3]


def test_heuristic_non_negative_and_zero_at_target():
    graph = make_line_graph()
    heuristic = make_heuristic(graph=graph)
    # node 1 (8.0,48) → node 3 (8.02,48): the heuristic IS the great-circle distance.
    from bike_router.geo import haversine_distance_m

    expected = haversine_distance_m(lat_a=48.0, lon_a=8.0, lat_b=48.0, lon_b=8.02)
    assert heuristic(1, 3) == pytest.approx(expected)
    assert heuristic(3, 3) == 0.0


def test_no_path_raises():
    graph = make_line_graph()
    graph.add_node(99, x=20.0, y=60.0, elevation=0.0)  # isolated
    with pytest.raises(nx.NetworkXNoPath):
        shortest_route(graph=graph, source=1, target=99)


def test_params_change_the_chosen_path():
    """With penalties on, the router avoids the steep paved main road (node 2) and
    takes the flat detour (node 3); distance-only picks the short node-2 path.
    """
    penalise = zero_params(extra_km_per_uphill_100m=5.0, extra_km_per_main_road_km=1.0)
    distance_only = ZERO_PARAMS

    avoid = shortest_route(graph=make_choice_graph(params=penalise), source=1, target=5)
    direct = shortest_route(graph=make_choice_graph(params=distance_only), source=1, target=5)

    assert avoid == [1, 3, 5]  # detours around hill + main road
    assert direct == [1, 2, 5]  # shortest ignores hill/road


def test_cutthrough_taken_when_detour_is_long_and_boarding_free():
    # L and R are entrances to one station S; the only pedalled alternative is a 10 km
    # detour L→M→R. With boarding 0 the two station edges are nearly free, so the router
    # cuts THROUGH the station (L→S→R) rather than ride the long way — the accepted tradeoff.
    graph = make_cutthrough_graph(params=zero_params(extra_km_per_boarding=0.0), detour_m=10_000.0)
    assert shortest_route(graph=graph, source=1, target=2) == [1, -1, 2]  # through station node -1


def test_cutthrough_avoided_when_boarding_is_expensive():
    # Same geometry, but a high boarding penalty makes passing through S cost a full boarding
    # (½ + ½), so the long bike detour wins — no cut-through.
    graph = make_cutthrough_graph(params=zero_params(extra_km_per_boarding=50.0), detour_m=10_000.0)
    assert shortest_route(graph=graph, source=1, target=2) == [1, 3, 2]  # the L→M→R detour


def test_station_edge_cost_never_below_length_keeps_heuristic_admissible():
    # Admissibility floor: EVERY station edge's stored cost must be >= its straight-line
    # length (the great-circle heuristic never overestimates). True even at boarding 0,
    # and the boarding term only ADDS. Checked on both boarding settings.
    for boarding in (0.0, 50.0):
        graph = make_cutthrough_graph(params=zero_params(extra_km_per_boarding=boarding))
        station_edges = [d for _u, _v, d in graph.edges(data=True) if d["mode"] == Mode.STATION]
        assert station_edges, "expected station edges in the cut-through graph"
        for data in station_edges:
            assert data[CostConfig.EDGE_COST] >= data["length"]  # cost floor = geometric length
        # boarding 0 → cost equals length exactly; boarding 50 → strictly greater.
        exact = all(d[CostConfig.EDGE_COST] == d["length"] for d in station_edges)
        assert exact if boarding == 0.0 else not exact
