"""Runtime sanity checks (spec section 5) — invariants the pipeline asserts.

These are intentionally cheap and fail loud, catching a broken graph or cost
model before we emit outputs.
"""

import logging

import numpy as np

from bike_router.core.constants import Mode, RoutingParams, SanityConfig

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


def check_cost_model(
    *,
    from_osmid: np.ndarray,
    to_osmid: np.ndarray,
    mode: np.ndarray,
    cost: np.ndarray,
    elev_by_osmid: dict[int, float],
    params: RoutingParams,
) -> None:
    """Sanity 2: on the steepest bidirectional BIKE edge, uphill must cost more than downhill.

    Operates on the corridor's flat edge arrays (the CSR router's own inputs) — no networkx.
    Only BIKE edges qualify (the terrain penalty is bike-only). If the uphill penalty is 0 the
    rider doesn't care, so the two directions are equal and the check is skipped; if there is no
    bidirectional non-flat bike edge the check is skipped (legitimate, not a bug).
    """
    if params.extra_km_per_uphill_100m == 0:
        logger.info("Sanity 2 skipped (uphill penalty disabled by user)")
        return
    # Cheapest cost per directed (u, v) among BIKE edges (matches the router's min-collapse).
    cost_by_pair: dict[tuple[int, int], float] = {}
    for u, v, m, c in zip(from_osmid, to_osmid, mode, cost, strict=True):
        if m != Mode.BIKE:
            continue
        key = (int(u), int(v))
        if key not in cost_by_pair or c < cost_by_pair[key]:
            cost_by_pair[key] = float(c)

    steepest: tuple[int, int] | None = None
    steepest_delta = 0.0
    for u, v in cost_by_pair:
        if (v, u) not in cost_by_pair:
            continue
        delta = abs(elev_by_osmid[v] - elev_by_osmid[u])
        if delta > steepest_delta:
            steepest_delta, steepest = delta, (u, v)
    if steepest is None:
        logger.info("Sanity 2 skipped (no bidirectional non-flat bike edge)")
        return

    lower, upper = steepest
    cost_lu, cost_ul = cost_by_pair[(lower, upper)], cost_by_pair[(upper, lower)]
    if elev_by_osmid[upper] > elev_by_osmid[lower]:  # lower→upper is uphill
        assert cost_lu > cost_ul, f"Sanity 2 failed: uphill {cost_lu} !> downhill {cost_ul}"
    else:  # upper→lower is uphill
        assert cost_ul > cost_lu, f"Sanity 2 failed: uphill {cost_ul} !> downhill {cost_lu}"
    logger.info(f"Sanity 2 OK: uphill > downhill on steepest bike edge (|Δ|={steepest_delta:.1f} m)")
