"""Runtime sanity checks (spec section 5) — invariants the pipeline asserts.

These are intentionally cheap and fail loud, catching a broken graph or cost
model before we emit outputs.
"""

import logging

import networkx as nx

from bike_router.constants import CostConfig, RoutingParams, SanityConfig
from bike_router.track import cheapest_edge

logger = logging.getLogger(__name__)


def check_simplify_shrunk(nodes_before: int, nodes_after: int) -> None:
    """Sanity 1: simplified node count should shrink by >50% vs the raw graph.

    Guarded: only meaningful when the raw graph is non-trivial (>= 20 nodes).
    """
    if nodes_before < SanityConfig.MIN_MEANINGFUL_NODES:
        logger.info(f"Sanity 1 skipped (raw graph too small: {nodes_before} nodes)")
        return
    assert nodes_after < 0.5 * nodes_before, (
        f"Sanity 1 failed: simplified graph did not shrink >50% ({nodes_before} → {nodes_after} nodes)"
    )
    logger.info(f"Sanity 1 OK: {nodes_before} → {nodes_after} nodes (>50% shrink)")


def _cheapest_cost(edges: dict[int, dict[str, object]]) -> float:
    """Stored cost of the cheapest parallel edge (via the canonical selector)."""
    return float(cheapest_edge(edges=edges)[CostConfig.EDGE_COST])


def check_uphill_costlier(graph: nx.MultiDiGraph, node_lower: int, node_upper: int, params: RoutingParams) -> None:
    """Sanity 2: when the rider penalises uphill, the uphill direction of a non-flat
    edge costs more than downhill. If the uphill penalty is 0 the rider does not
    care, so the two directions are (correctly) equal — the check is skipped.

    Caller must pass a genuinely bidirectional, non-flat edge (see
    find_steepest_bidirectional_edge) — edges/elevations accessed strictly.
    """
    if params.extra_km_per_uphill_100m == 0:
        logger.info("Sanity 2 skipped (uphill penalty disabled by user)")
        return
    cost_up = _cheapest_cost(edges=graph.get_edge_data(node_lower, node_upper))
    cost_down = _cheapest_cost(edges=graph.get_edge_data(node_upper, node_lower))
    elev_lower = graph.nodes[node_lower]["elevation"]
    elev_upper = graph.nodes[node_upper]["elevation"]
    if elev_upper > elev_lower:  # node_lower→node_upper is uphill
        assert cost_up > cost_down, f"Sanity 2 failed: uphill {cost_up} !> downhill {cost_down}"
    elif elev_lower > elev_upper:  # node_upper→node_lower is uphill
        assert cost_down > cost_up, f"Sanity 2 failed: uphill {cost_down} !> downhill {cost_up}"
    else:
        raise AssertionError("Sanity 2 misused: edge is flat (caller must pass a non-flat edge)")
    logger.info(f"Sanity 2 OK: uphill {cost_up:.1f} > downhill {cost_down:.1f}")


def check_strongly_connected(graph: nx.MultiDiGraph) -> None:
    """Sanity 3: the routable core must be strongly connected (a route exists)."""
    assert nx.is_strongly_connected(graph), "Sanity 3 failed: graph is not strongly connected"
    logger.info("Sanity 3 OK: graph is strongly connected")


def find_steepest_bidirectional_edge(graph: nx.MultiDiGraph) -> tuple[int, int] | None:
    """Return (node_a, node_b) of the bidirectional edge with the largest |Δelevation|.

    Used to feed check_uphill_costlier a genuinely non-flat edge.
    """
    steepest = None
    steepest_delta = 0.0
    for node_a, node_b in graph.edges():
        if not graph.has_edge(node_b, node_a):
            continue
        delta = abs(graph.nodes[node_b]["elevation"] - graph.nodes[node_a]["elevation"])
        if delta > steepest_delta:
            steepest_delta, steepest = delta, (node_a, node_b)
    return steepest
