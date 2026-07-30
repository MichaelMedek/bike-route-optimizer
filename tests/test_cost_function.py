"""Cost-model tests: surface tiers, main-road detection, extra-km penalties, and modes.

Cost is the ONE vectorized ``edge_cost_array``; ``_cost`` wraps a single edge in a one-row
table so the penalty contracts read one line each.
"""

import numpy as np
import pandas as pd
import pytest

from bike_router.core.constants import Mode
from bike_router.core.cost import _as_values, edge_cost_array, road_included, road_tier, surface_included, surface_tier
from tests.conftest import ZERO_PARAMS, zero_params


def _cost(*, mode, length, surface=None, highway=None, from_elev=0.0, to_elev=0.0, params=ZERO_PARAMS) -> float:
    """The vectorized cost of ONE edge (one-row table), for the single-edge penalty contracts."""
    edges_df = pd.DataFrame(
        [{"from_node": 1, "to_node": 2, "mode": mode, "length_m": length, "surface": surface, "highway": highway}]
    )
    return float(edge_cost_array(edges_df=edges_df, elev_by_osmid={1: from_elev, 2: to_elev}, params=params)[0])


# --- tag parsing / tiers -----------------------------------------------------


def test_as_values_normalizes_types():
    assert _as_values(tag=None) == []
    assert _as_values(tag="Asphalt") == ["asphalt"]
    assert _as_values(tag=["Gravel", "DIRT"]) == ["gravel", "dirt"]
    assert _as_values(tag=42) == ["42"]


def test_as_values_treats_nan_as_missing_not_string():
    # REGRESSION: pandas/pyrosm encode a missing OSM tag as float nan (NOT None), and
    # to_graph(simplify=True) puts nan INSIDE merged lists (['asphalt', nan]). Stringifying nan to
    # "nan" made tag_included reject every untagged/mixed road → 54% of austria's network dropped.
    nan = float("nan")
    assert _as_values(tag=nan) == []  # scalar nan → missing, not ["nan"]
    assert _as_values(tag=[nan, "Asphalt"]) == ["asphalt"]  # nan dropped, asphalt kept
    assert _as_values(tag=[nan, nan]) == []  # all-nan → missing


def test_nan_and_none_kept_for_both_surface_and_highway():
    # REGRESSION: the missing-tag keeper must apply to BOTH surface and highway.
    nan = float("nan")
    for val in (None, nan):
        assert surface_included(surface=val) is True
        assert road_included(highway=val) is True
        assert surface_tier(surface=val) == 1  # untagged → DEFAULT_TIER
        assert road_tier(highway=val) == 1
    assert surface_included(surface=[nan, "asphalt"]) is True  # nan + allowed → kept
    assert road_included(highway=[nan, "residential"]) is True
    assert surface_included(surface=[nan, "sand"]) is False  # nan doesn't rescue a disallowed value
    assert road_included(highway=[nan, "motorway"]) is False


def test_surface_tier_mapping_and_worst_wins():
    assert surface_tier(surface="asphalt") == 0
    assert surface_tier(surface="concrete:plates") == 0  # paved variant → good
    assert surface_tier(surface="gravel") == 1  # loose
    assert surface_tier(surface="ground") == 2  # natural/rough but rideable
    assert surface_tier(surface=["asphalt", "gravel"]) == 1  # worst wins (0 vs 1 → 1)
    assert surface_tier(surface=["gravel", "ground"]) == 2  # worst wins (1 vs 2 → 2)
    assert surface_tier(surface="spacedust") == 1  # unknown → DEFAULT_TIER (loose)
    assert surface_tier(surface=None) == 1


def test_surface_included_allowlist():
    assert all(surface_included(surface=s) for s in ("asphalt", "gravel", "ground", "dirt", None))
    assert not surface_included(surface="sand")  # genuinely impassable → excluded
    assert not surface_included(surface="mud")
    assert not surface_included(surface="gravel;mud")  # any disallowed value → excluded


def test_road_tier_mapping_and_worst_wins():
    assert all(road_tier(highway=h) == 1 for h in ("secondary", "primary", "unclassified", "trunk", "primary_link"))
    assert all(road_tier(highway=h) == 0 for h in ("residential", "cycleway", "tertiary"))
    assert road_tier(highway=["residential", "secondary"]) == 1  # worst wins → main
    assert road_tier(highway=None) == 1  # untagged → DEFAULT_TIER (main, pessimistic)


def test_road_included_allowlist():
    assert all(road_included(highway=h) for h in ("residential", "secondary", None))
    assert not road_included(highway="motorway")  # not in allowlist → excluded (no bikes)
    assert not road_included(highway="raceway")
    assert not road_included(highway=["residential", "motorway"])  # any disallowed → excluded


# --- edge cost (the vectorized formula, one edge at a time) -------------------


def test_distance_only_cost_is_length():
    assert _cost(mode=Mode.BIKE, length=1000.0, surface="mud", highway="primary", to_elev=50.0) == 1000.0


def test_uphill_penalty_matches_extra_km_contract():
    # 100 m climb, 5 extra km per 100 m → +5000 m on top of length.
    cost = _cost(
        mode=Mode.BIKE,
        length=1000.0,
        surface="asphalt",
        highway="residential",
        to_elev=100.0,
        params=zero_params(extra_km_per_uphill_100m=5.0),
    )
    assert cost == pytest.approx(1000.0 + 5000.0)


def test_uphill_only_downhill_is_free():
    cost = _cost(
        mode=Mode.BIKE,
        length=1000.0,
        surface="asphalt",
        highway="residential",
        from_elev=100.0,
        to_elev=0.0,
        params=zero_params(extra_km_per_uphill_100m=5.0),
    )
    assert cost == 1000.0  # descending adds no penalty


def test_unpaved_penalty_scales_with_tier():
    params = zero_params(extra_km_per_unpaved_km=1.0)
    assert _cost(mode=Mode.BIKE, length=1000.0, surface="asphalt", highway="residential", params=params) == 1000.0
    assert _cost(
        mode=Mode.BIKE, length=1000.0, surface="gravel", highway="residential", params=params
    ) == pytest.approx(2000.0)


def test_unpaved_tier2_doubles_the_penalty():
    # The tier is a literal multiplier: 1 km, 1 extra km/km → tier 1 +1000 m, tier 2 +2000 m.
    params = zero_params(extra_km_per_unpaved_km=1.0)
    assert _cost(
        mode=Mode.BIKE, length=1000.0, surface="gravel", highway="residential", params=params
    ) == pytest.approx(2000.0)
    assert _cost(
        mode=Mode.BIKE, length=1000.0, surface="ground", highway="residential", params=params
    ) == pytest.approx(3000.0)


def test_main_road_penalty():
    params = zero_params(extra_km_per_main_road_km=2.0)
    assert _cost(mode=Mode.BIKE, length=1000.0, surface="asphalt", highway="secondary", params=params) == pytest.approx(
        3000.0
    )
    assert _cost(mode=Mode.BIKE, length=1000.0, surface="asphalt", highway="residential", params=params) == 1000.0


def test_penalties_are_additive():
    params = zero_params(extra_km_per_uphill_100m=5.0, extra_km_per_unpaved_km=1.0, extra_km_per_main_road_km=1.0)
    cost = _cost(mode=Mode.BIKE, length=1000.0, surface="gravel", highway="secondary", to_elev=100.0, params=params)
    assert cost == pytest.approx(1000.0 + 5000.0 + 1000.0 + 1000.0)


def test_cost_never_below_length():
    params = zero_params(extra_km_per_uphill_100m=10.0, extra_km_per_unpaved_km=3.0, extra_km_per_main_road_km=3.0)
    cost = _cost(mode=Mode.BIKE, length=500.0, surface="gravel", highway="primary", to_elev=100.0, params=params)
    assert cost == pytest.approx(500.0 + 10000.0 + 1500.0 + 1500.0)
    assert cost > 500.0  # never below raw length


def test_rail_cost_uses_per_km_only_no_boarding_no_terrain_penalty():
    # 10 km rail, 2 extra km/km → length + 20000. Boarding lives on station edges; terrain ignored.
    params = zero_params(extra_km_per_rail_km=2.0, extra_km_per_boarding=15.0)
    cost = _cost(mode=Mode.RAIL, length=10_000.0, surface="mud", highway="primary", to_elev=500.0, params=params)
    assert cost == pytest.approx(10_000.0 + 20_000.0)


def test_rail_sliders_scale_cost_so_high_values_deter_rail():
    cheap = zero_params(extra_km_per_rail_km=0.1, extra_km_per_boarding=1.0)
    dear = zero_params(extra_km_per_rail_km=5.0, extra_km_per_boarding=80.0)
    assert _cost(mode=Mode.RAIL, length=5_000.0, params=dear) > _cost(mode=Mode.RAIL, length=5_000.0, params=cheap)


def test_station_cost_is_length_plus_half_boarding():
    # Station edge = straight-line length + half the boarding charge (board + alight = full).
    assert _cost(mode=Mode.STATION, length=150.0, to_elev=100.0) == 150.0  # boarding 0 → pure length
    cost = _cost(mode=Mode.STATION, length=150.0, params=zero_params(extra_km_per_boarding=10.0))
    assert cost == pytest.approx(150.0 + 5000.0)


def test_unknown_mode_raises():
    with pytest.raises(AssertionError, match="unknown edge mode"):
        _cost(mode="teleport", length=1.0)


def test_edge_cost_array_is_vectorized_over_many_rows():
    # The array path costs a whole table at once (the ONE cost formula) — mixed modes in one call.
    edges_df = pd.DataFrame(
        {
            "from_node": [1, 2, 3],
            "to_node": [2, 3, 4],
            "mode": [Mode.BIKE, Mode.RAIL, Mode.STATION],
            "length_m": [1000.0, 10_000.0, 150.0],
            "surface": ["gravel", None, None],
            "highway": ["secondary", None, None],
        }
    )
    params = zero_params(
        extra_km_per_unpaved_km=1.0, extra_km_per_main_road_km=1.0, extra_km_per_rail_km=2.0, extra_km_per_boarding=10.0
    )
    elev = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    got = edge_cost_array(edges_df=edges_df, elev_by_osmid=elev, params=params)
    assert np.allclose(got, [1000.0 + 1000.0 + 1000.0, 10_000.0 + 20_000.0, 150.0 + 5000.0])
