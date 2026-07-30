"""Sanity-check tests (spec §5) — cost-model + simplify-shrink checks on edge arrays."""

import numpy as np
import pytest

from bike_router.core.constants import Mode, RoutingParams
from bike_router.core.sanity import check_cost_model, check_simplify_shrunk
from tests.conftest import DEFAULT_PARAMS, make_line_edges


def _line_bike_edges(*, params):
    """Flat (from, to, mode, cost) arrays for the costed bike line graph's edges."""
    args = make_line_edges().route_graph_args(params=params)
    return args["from_osmid"], args["to_osmid"], np.array([Mode.BIKE] * len(args["cost"]), dtype=object), args["cost"]


def test_edge_cost_array_is_asymmetric_uphill_costlier():
    # 1→2 climbs (100→130), 2→1 descends → uphill costs more (default params penalise uphill).
    frm, to, _mode, cost = _line_bike_edges(params=DEFAULT_PARAMS)
    by_pair = {(int(u), int(v)): float(c) for u, v, c in zip(frm, to, cost, strict=True)}
    assert by_pair[(1, 2)] > by_pair[(2, 1)]


def test_check_simplify_shrunk_pass_and_fail():
    check_simplify_shrunk(nodes_before=100, nodes_after=40)  # >50% shrink → OK
    with pytest.raises(AssertionError):
        check_simplify_shrunk(nodes_before=100, nodes_after=80)


def test_check_simplify_shrunk_skips_tiny_graph():
    check_simplify_shrunk(nodes_before=10, nodes_after=10)  # too small → no assertion


def test_check_cost_model_passes_on_uphill_costlier():
    frm, to, mode, cost = _line_bike_edges(params=DEFAULT_PARAMS)
    # steepest bidirectional bike edge is 1↔2 (100↔130); uphill 1→2 costs more → passes.
    check_cost_model(
        from_osmid=frm,
        to_osmid=to,
        mode=mode,
        cost=cost,
        elev_by_osmid={1: 100.0, 2: 130.0, 3: 100.0},
        params=DEFAULT_PARAMS,
    )


def test_check_cost_model_skipped_when_penalty_disabled():
    params = RoutingParams(
        extra_km_per_uphill_100m=0.0,
        extra_km_per_unpaved_km=1.0,
        extra_km_per_main_road_km=1.0,
        extra_km_per_rail_km=0.0,
        extra_km_per_boarding=0.0,
    )
    frm, to, mode, cost = _line_bike_edges(params=params)
    # uphill penalty 0 → both directions equal → check must skip, not fail.
    check_cost_model(
        from_osmid=frm,
        to_osmid=to,
        mode=mode,
        cost=cost,
        elev_by_osmid={1: 100.0, 2: 130.0, 3: 100.0},
        params=params,
    )


def test_check_cost_model_ignores_rail_uses_bike_edge():
    """Regression: the steepest bidirectional edge may be RAIL (terrain-blind cost → up==down),
    which would spuriously fail. The check must consider only BIKE edges.
    """
    # 1↔2 bike (40 m climb, asymmetric cost); 2↔3 rail (900 m Δelev but symmetric terrain-blind cost).
    frm = np.array([1, 2, 2, 3], dtype="int64")
    to = np.array([2, 1, 3, 2], dtype="int64")
    mode = np.array([Mode.BIKE, Mode.BIKE, Mode.RAIL, Mode.RAIL], dtype=object)
    cost = np.array([150.0, 100.0, 100.0, 100.0], dtype=float)
    elev = {1: 100.0, 2: 140.0, 3: 1000.0}
    check_cost_model(from_osmid=frm, to_osmid=to, mode=mode, cost=cost, elev_by_osmid=elev, params=DEFAULT_PARAMS)


def test_check_cost_model_skipped_when_no_bidirectional_bike_edge():
    # Only a one-way bike edge → no bidirectional pair → skip (legitimate, not a failure).
    check_cost_model(
        from_osmid=np.array([1], dtype="int64"),
        to_osmid=np.array([2], dtype="int64"),
        mode=np.array([Mode.BIKE], dtype=object),
        cost=np.array([100.0], dtype=float),
        elev_by_osmid={1: 100.0, 2: 200.0},
        params=DEFAULT_PARAMS,
    )


def test_check_cost_model_raises_when_uphill_not_costlier():
    # A broken cost model where uphill (1→2) is CHEAPER than downhill (2→1) must fail loud.
    with pytest.raises(AssertionError, match="Sanity 2 failed"):
        check_cost_model(
            from_osmid=np.array([1, 2], dtype="int64"),
            to_osmid=np.array([2, 1], dtype="int64"),
            mode=np.array([Mode.BIKE, Mode.BIKE], dtype=object),
            cost=np.array([100.0, 150.0], dtype=float),
            elev_by_osmid={1: 100.0, 2: 200.0},
            params=DEFAULT_PARAMS,
        )
