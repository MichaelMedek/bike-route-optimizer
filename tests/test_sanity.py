"""Sanity-check tests (spec §5) + assign_edge_costs on a real graph."""

import networkx as nx
import pytest

from bike_router.constants import CostConfig, Mode, RoutingParams
from bike_router.cost import assign_edge_costs
from bike_router.sanity import (
    check_simplify_shrunk,
    check_strongly_connected,
    check_uphill_costlier,
    find_steepest_bidirectional_edge,
)
from tests.conftest import DEFAULT_PARAMS, make_line_graph


def test_assign_edge_costs_writes_cost_and_is_asymmetric():
    graph = make_line_graph()
    for _a, _b, data in graph.edges(data=True):
        assert CostConfig.EDGE_COST in data
    # 1→2 climbs (100→130), 2→1 descends → uphill costs more (default params penalise uphill)
    assert graph.get_edge_data(1, 2)[0][CostConfig.EDGE_COST] > graph.get_edge_data(2, 1)[0][CostConfig.EDGE_COST]


def test_check_simplify_shrunk_pass_and_fail():
    check_simplify_shrunk(nodes_before=100, nodes_after=40)  # >50% shrink → OK
    with pytest.raises(AssertionError):
        check_simplify_shrunk(nodes_before=100, nodes_after=80)


def test_check_simplify_shrunk_skips_tiny_graph():
    check_simplify_shrunk(nodes_before=10, nodes_after=10)  # too small → no assertion


def test_check_strongly_connected_pass_and_fail():
    graph = make_line_graph()
    check_strongly_connected(graph=graph)
    graph.add_node(99, x=20.0, y=60.0, elevation=0.0)  # unreachable
    with pytest.raises(AssertionError):
        check_strongly_connected(graph=graph)


def test_find_steepest_and_uphill_costlier():
    graph = make_line_graph()
    steepest = find_steepest_bidirectional_edge(graph=graph)
    assert steepest is not None and set(steepest) == {1, 2}  # 1↔2 is the only non-flat edge (100↔130 m)
    check_uphill_costlier(graph=graph, node_lower=steepest[0], node_upper=steepest[1], params=DEFAULT_PARAMS)


def test_uphill_check_skipped_when_penalty_disabled():
    # uphill penalty 0 → both directions cost the same → check must skip, not fail
    params = RoutingParams(
        extra_km_per_uphill_100m=0.0,
        extra_km_per_unpaved_km=1.0,
        extra_km_per_main_road_km=1.0,
        extra_km_per_rail_km=0.0,
        extra_km_per_boarding=0.0,
    )
    graph = make_line_graph(params=params)
    steepest = find_steepest_bidirectional_edge(graph=graph)
    assert steepest is not None
    check_uphill_costlier(graph=graph, node_lower=steepest[0], node_upper=steepest[1], params=params)  # no raise


def test_find_steepest_returns_none_when_no_bidirectional_edge():
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=8.0, y=48.0, elevation=100.0)
    graph.add_node(2, x=8.1, y=48.0, elevation=200.0)
    graph.add_edge(1, 2, key=0, length=100.0)  # one-way only
    assert find_steepest_bidirectional_edge(graph=graph) is None


def test_check_uphill_costlier_rejects_flat_edge():
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=8.0, y=48.0, elevation=100.0)
    graph.add_node(2, x=8.1, y=48.0, elevation=100.0)  # flat
    for node_a, node_b in [(1, 2), (2, 1)]:
        graph.add_edge(node_a, node_b, key=0, length=100.0, surface="asphalt", highway="residential", mode=Mode.BIKE)
    assign_edge_costs(graph=graph, params=DEFAULT_PARAMS)
    with pytest.raises(AssertionError):
        check_uphill_costlier(graph=graph, node_lower=1, node_upper=2, params=DEFAULT_PARAMS)
