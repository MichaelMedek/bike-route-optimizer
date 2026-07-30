"""Sanity-check tests (spec §5) — cost-model + simplify-shrink checks on edge arrays.

One test_<fn> per production function; each folds ALL its scenarios (pass, the skip cases,
and the raise case) into a single exact-name test so every original assertion is preserved.
"""

import numpy as np
import pytest

from bike_router.core.constants import Mode, RoutingParams
from bike_router.core.sanity import check_cost_model, check_simplify_shrunk
from tests.conftest import DEFAULT_PARAMS, make_line_edges


def _line_bike_edges(*, params):
    """Flat (from, to, mode, cost) arrays for the costed bike line graph's edges."""
    args = make_line_edges().route_graph_args(params=params)
    return args["from_osmid"], args["to_osmid"], np.array([Mode.BIKE] * len(args["cost"]), dtype=object), args["cost"]


def test_check_simplify_shrunk():
    # >50% shrink passes; a <50% shrink fails loud; a below-threshold raw graph is too small to
    # meaningfully judge, so the check is skipped (no assertion) rather than failing.
    check_simplify_shrunk(nodes_before=100, nodes_after=40)  # >50% shrink → OK
    with pytest.raises(AssertionError):
        check_simplify_shrunk(nodes_before=100, nodes_after=80)  # <50% → fail loud
    check_simplify_shrunk(nodes_before=10, nodes_after=10)  # tiny raw graph → skipped, no assertion


def test_check_cost_model():
    # Sanity 2: on the steepest bidirectional BIKE edge, uphill must cost more than downhill.
    # This one exact-name test walks every scenario the check must handle.

    # (a) real costed line graph: 1↔2 is the steepest pair and its cost array is asymmetric
    # (uphill 1→2 dearer than downhill 2→1), so the check passes.
    frm, to, mode, cost = _line_bike_edges(params=DEFAULT_PARAMS)
    by_pair = {(int(u), int(v)): float(c) for u, v, c in zip(frm, to, cost, strict=True)}
    assert by_pair[(1, 2)] > by_pair[(2, 1)]  # the asymmetry the check relies on
    check_cost_model(
        from_osmid=frm,
        to_osmid=to,
        mode=mode,
        cost=cost,
        elev_by_osmid={1: 100.0, 2: 130.0, 3: 100.0},
        params=DEFAULT_PARAMS,
    )

    # (b) uphill penalty disabled → both directions cost the same → the check is skipped, not failed.
    no_uphill = RoutingParams(
        extra_km_per_uphill_100m=0.0,
        extra_km_per_unpaved_km=1.0,
        extra_km_per_main_road_km=1.0,
        extra_km_per_rail_km=0.0,
        extra_km_per_boarding=0.0,
    )
    frm0, to0, mode0, cost0 = _line_bike_edges(params=no_uphill)
    check_cost_model(
        from_osmid=frm0,
        to_osmid=to0,
        mode=mode0,
        cost=cost0,
        elev_by_osmid={1: 100.0, 2: 130.0, 3: 100.0},
        params=no_uphill,
    )

    # (c) REGRESSION: the steepest bidirectional edge here is RAIL (900 m Δelev, terrain-blind
    # symmetric cost) — the check must consider only BIKE edges, so it passes on the bike 1↔2.
    check_cost_model(
        from_osmid=np.array([1, 2, 2, 3], dtype="int64"),
        to_osmid=np.array([2, 1, 3, 2], dtype="int64"),
        mode=np.array([Mode.BIKE, Mode.BIKE, Mode.RAIL, Mode.RAIL], dtype=object),
        cost=np.array([150.0, 100.0, 100.0, 100.0], dtype=float),
        elev_by_osmid={1: 100.0, 2: 140.0, 3: 1000.0},
        params=DEFAULT_PARAMS,
    )

    # (d) only a one-way bike edge → no bidirectional pair to compare → skipped (legitimate).
    check_cost_model(
        from_osmid=np.array([1], dtype="int64"),
        to_osmid=np.array([2], dtype="int64"),
        mode=np.array([Mode.BIKE], dtype=object),
        cost=np.array([100.0], dtype=float),
        elev_by_osmid={1: 100.0, 2: 200.0},
        params=DEFAULT_PARAMS,
    )

    # (e) a broken cost model where uphill (1→2) is CHEAPER than downhill (2→1) must fail loud.
    with pytest.raises(AssertionError, match="Sanity 2 failed"):
        check_cost_model(
            from_osmid=np.array([1, 2], dtype="int64"),
            to_osmid=np.array([2, 1], dtype="int64"),
            mode=np.array([Mode.BIKE, Mode.BIKE], dtype=object),
            cost=np.array([100.0, 150.0], dtype=float),
            elev_by_osmid={1: 100.0, 2: 200.0},
            params=DEFAULT_PARAMS,
        )
