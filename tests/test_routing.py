"""Routing tests — A* over the tiny graphs + admissible heuristic + param effects."""

import networkx as nx
import pytest

from bike_router.constants import RoutingParams
from bike_router.routing import make_heuristic, shortest_route
from tests.conftest import make_choice_graph, make_line_graph


def test_shortest_route_traverses_line():
    graph = make_line_graph()
    assert shortest_route(graph=graph, source=1, target=3) == [1, 2, 3]


def test_heuristic_non_negative_and_zero_at_target():
    graph = make_line_graph()
    heuristic = make_heuristic(graph=graph)
    assert heuristic(1, 3) >= 0
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
    penalise = RoutingParams(extra_km_per_uphill_100m=5.0, extra_km_per_unpaved_km=0.0, extra_km_per_main_road_km=1.0)
    distance_only = RoutingParams(
        extra_km_per_uphill_100m=0.0, extra_km_per_unpaved_km=0.0, extra_km_per_main_road_km=0.0
    )

    avoid = shortest_route(graph=make_choice_graph(params=penalise), source=1, target=5)
    direct = shortest_route(graph=make_choice_graph(params=distance_only), source=1, target=5)

    assert avoid == [1, 3, 5]  # detours around hill + main road
    assert direct == [1, 2, 5]  # shortest ignores hill/road
