"""Cost-model tests: surface tiers, main-road detection, extra-km penalties, and modes.

One test_<fn> per production symbol (the exact-name mirror), each folding every scenario for
that function. Cost is the ONE vectorized ``edge_cost_array``; ``_cost`` wraps a single edge
in a one-row table so the per-mode penalty contracts read compactly.
"""

import numpy as np
import pandas as pd
import pytest

from bike_router.core.constants import Mode, RoadConfig, SurfaceConfig
from bike_router.core.cost import (
    _as_values,
    _is_missing,
    edge_cost_array,
    road_included,
    road_tier,
    road_weight,
    surface_included,
    surface_tier,
    surface_weight,
    tag_included,
    tag_tier,
    tag_weight,
)
from tests.conftest import ZERO_PARAMS, zero_params


def _cost(*, mode, length, surface=None, highway=None, from_elev=0.0, to_elev=0.0, params=ZERO_PARAMS) -> float:
    """The vectorized cost of ONE edge (one-row table), for the single-edge penalty contracts."""
    row = {"from_node": 1, "to_node": 2, "mode": mode, "length_m": length, "surface": surface, "highway": highway}
    edges_df = pd.DataFrame([row])
    return float(edge_cost_array(edges_df=edges_df, elev_by_osmid={1: from_elev, 2: to_elev}, params=params)[0])


# --- tag parsing / tiers -----------------------------------------------------


def test_as_values():
    # Normalizes an OSM tag to a list of lowercased strings; a scalar/str/int maps directly.
    assert _as_values(tag=None) == []
    assert _as_values(tag="Asphalt") == ["asphalt"]
    assert _as_values(tag=["Gravel", "DIRT"]) == ["gravel", "dirt"]
    assert _as_values(tag=42) == ["42"]
    # REGRESSION: pandas/pyrosm encode a missing OSM tag as float nan (NOT None), and
    # to_graph(simplify=True) puts nan INSIDE merged lists (['asphalt', nan]). Stringifying nan to
    # "nan" made tag_included reject every untagged/mixed road → 54% of austria's network dropped.
    nan = float("nan")
    assert _as_values(tag=nan) == []  # scalar nan → missing, not ["nan"]
    assert _as_values(tag=[nan, "Asphalt"]) == ["asphalt"]  # nan dropped, asphalt kept
    assert _as_values(tag=[nan, nan]) == []  # all-nan → missing


def test_is_missing():
    # True only for the two ways pandas encodes an absent OSM tag: None and float nan.
    assert _is_missing(value=None) is True
    assert _is_missing(value=float("nan")) is True
    assert _is_missing(value="asphalt") is False
    assert _is_missing(value=0) is False  # a real 0 is NOT missing


def test_tag_tier():
    # Shared tier resolver: worst (highest) known value wins; all-unknown/missing → default_tier.
    tiers = {"asphalt": 0, "gravel": 1, "ground": 2}
    assert tag_tier(tag="asphalt", tier_map=tiers, default_tier=1) == 0
    assert tag_tier(tag=["asphalt", "gravel"], tier_map=tiers, default_tier=1) == 1  # worst wins (0 vs 1)
    assert tag_tier(tag=["gravel", "ground"], tier_map=tiers, default_tier=1) == 2  # worst wins (1 vs 2)
    assert tag_tier(tag="spacedust", tier_map=tiers, default_tier=1) == 1  # unknown → default
    assert tag_tier(tag=None, tier_map=tiers, default_tier=2) == 2  # missing → default


def test_tag_weight():
    # Shared continuous-weight resolver: worst (highest) known value wins; all-unknown/missing → default.
    weights = {"asphalt": 0.0, "gravel": 0.9, "ground": 1.0}
    assert tag_weight(tag="asphalt", weight_map=weights, default_weight=1.0) == 0.0
    assert tag_weight(tag=["asphalt", "gravel"], weight_map=weights, default_weight=1.0) == 0.9  # worst wins
    assert tag_weight(tag="spacedust", weight_map=weights, default_weight=0.5) == 0.5  # unknown → default
    assert tag_weight(tag=None, weight_map=weights, default_weight=0.5) == 0.5  # missing → default


def test_tag_included():
    # Shared allowlist gate: False iff a tag names a category outside the map; missing → True.
    tiers = {"asphalt": 0, "gravel": 1}
    assert tag_included(tag="asphalt", tier_map=tiers) is True
    assert tag_included(tag=None, tier_map=tiers) is True  # untagged kept
    assert tag_included(tag="sand", tier_map=tiers) is False
    assert tag_included(tag=["asphalt", "sand"], tier_map=tiers) is False  # any disallowed → excluded


def test_surface_tier():
    # Tier is the capped colour bucket color_tier(rolling-share-scaled weight): asphalt/gravel round to 0
    # (gravel Crr only ~2× asphalt), genuinely rough sett/grass/ground round to 1. Never a third colour.
    assert surface_tier(surface="asphalt") == 0
    assert surface_tier(surface="compacted") == 0  # firm, low Crr
    assert surface_tier(surface="gravel") == 0  # Crr ~2× asphalt × ~0.4 share → weight 0.40 → good
    assert surface_tier(surface="grass") == 1  # soft turf, high Crr → bad
    assert surface_tier(surface="sett") == 1  # impedance-rough → bad
    assert surface_tier(surface=["asphalt", "grass"]) == 1  # worst wins (0 vs 1 → 1)
    assert surface_tier(surface="spacedust") == 1  # unknown → DEFAULT_TIER
    assert surface_tier(surface=None) == 1  # untagged → DEFAULT_TIER
    assert surface_tier(surface=float("nan")) == 1  # nan is missing → DEFAULT_TIER


def test_surface_weight():
    # Continuous Crr-ordered weight: asphalt 0.0, gravel worse than compacted, worst wins, untagged → default.
    assert surface_weight(surface="asphalt") == 0.0
    assert surface_weight(surface="compacted") < surface_weight(surface="gravel")  # firm cheaper than loose
    assert surface_weight(surface=["asphalt", "gravel"]) == SurfaceConfig.SURFACE_WEIGHT["gravel"]  # worst wins
    assert surface_weight(surface="spacedust") == SurfaceConfig.DEFAULT_WEIGHT  # unknown → default
    assert surface_weight(surface=None) == SurfaceConfig.DEFAULT_WEIGHT  # untagged → default


def test_surface_included():
    assert all(surface_included(surface=s) for s in ("asphalt", "gravel", "ground", "dirt", None))
    assert not surface_included(surface="sand")  # genuinely impassable → excluded
    assert not surface_included(surface="mud")
    assert not surface_included(surface="gravel;mud")  # any disallowed value → excluded
    # REGRESSION: nan merged with an allowed value is kept; nan doesn't rescue a disallowed one.
    nan = float("nan")
    assert surface_included(surface=[nan, "asphalt"]) is True
    assert surface_included(surface=[nan, "sand"]) is False


def test_road_tier():
    # Tier is the capped colour bucket color_tier(RP weight): arterials (secondary/primary/trunk, w≥0.65)
    # → 1 (main/red); quiet + minor through-roads (residential/tertiary/unclassified/track, w≤0.45) → 0.
    assert all(road_tier(highway=h) == 1 for h in ("secondary", "primary", "trunk", "primary_link"))
    assert all(road_tier(highway=h) == 0 for h in ("residential", "cycleway", "tertiary", "unclassified", "track"))
    assert road_tier(highway=["residential", "secondary"]) == 1  # worst wins → main
    assert road_tier(highway=None) == 1  # untagged → DEFAULT_TIER (main, pessimistic)
    assert road_tier(highway=float("nan")) == 1  # nan is missing → DEFAULT_TIER


def test_road_weight():
    # Revealed-preference detour weight (Broach): cycleway 0.0, quiet local ~0.15, minor through ~0.45,
    # arterial ~0.65, trunk 1.0. Worst wins; untagged → default (worst, pessimistic).
    assert road_weight(highway="cycleway") == 0.0
    assert road_weight(highway="trunk") == 1.0
    # Monotone by measured avoidance: local < minor through-road < arterial < major arterial.
    assert road_weight(highway="residential") < road_weight(highway="tertiary") < road_weight(highway="secondary")
    assert road_weight(highway="secondary") < road_weight(highway="primary") < road_weight(highway="trunk")
    assert road_weight(highway=["residential", "secondary"]) == RoadConfig.ROAD_WEIGHT["secondary"]  # worst wins
    assert road_weight(highway=None) == RoadConfig.DEFAULT_WEIGHT  # untagged → default


def test_road_included():
    assert all(road_included(highway=h) for h in ("residential", "secondary", None))
    assert not road_included(highway="motorway")  # not in allowlist → excluded (no bikes)
    assert not road_included(highway="raceway")
    assert not road_included(highway=["residential", "motorway"])  # any disallowed → excluded
    nan = float("nan")
    assert road_included(highway=[nan, "residential"]) is True  # nan + allowed → kept
    assert road_included(highway=[nan, "motorway"]) is False  # nan doesn't rescue a disallowed value


# --- edge_cost_array (the ONE vectorized cost formula) -----------------------


def test_edge_cost_array():
    # Every mode + penalty contract of the single vectorized cost, one edge at a time via _cost,
    # then a whole mixed-mode table in one call. bike = length + uphill(uphill-only) + unpaved(tier×)
    # + main-road; rail = per-km only (terrain ignored, boarding lives on station edges); station =
    # length + half the boarding charge; cost is never below raw length; unknown mode fails loud.

    # distance-only: every penalty disabled → pure length (mud/primary/climb all ignored)
    assert _cost(mode=Mode.BIKE, length=1000.0, surface="mud", highway="primary", to_elev=50.0) == 1000.0

    # uphill: 100 m climb × 5 extra-km/100m → +5000 m; descending the same edge adds nothing
    up = zero_params(extra_km_per_uphill_100m=5.0)
    assert _cost(
        mode=Mode.BIKE, length=1000.0, surface="asphalt", highway="residential", to_elev=100.0, params=up
    ) == pytest.approx(6000.0)
    assert (
        _cost(
            mode=Mode.BIKE,
            length=1000.0,
            surface="asphalt",
            highway="residential",
            from_elev=100.0,
            to_elev=0.0,
            params=up,
        )
        == 1000.0
    )  # downhill free

    # unpaved: the continuous WEIGHT is the multiplier — asphalt 0.0 (free), gravel 0.9, ground 1.0
    # (at 1 extra-km/km) → +weight·length. Weights come from SurfaceConfig.SURFACE_WEIGHT (Crr-ordered).
    unp = zero_params(extra_km_per_unpaved_km=1.0)
    assert _cost(mode=Mode.BIKE, length=1000.0, surface="asphalt", highway="residential", params=unp) == 1000.0
    assert _cost(mode=Mode.BIKE, length=1000.0, surface="gravel", highway="residential", params=unp) == pytest.approx(
        1000.0 + 1000.0 * SurfaceConfig.SURFACE_WEIGHT["gravel"]
    )
    assert _cost(mode=Mode.BIKE, length=1000.0, surface="ground", highway="residential", params=unp) == pytest.approx(
        1000.0 + 1000.0 * SurfaceConfig.SURFACE_WEIGHT["ground"]
    )

    # main road: +weight·extra-km/km on a main road (secondary weight 0.55), nothing on a quiet way
    main = zero_params(extra_km_per_main_road_km=2.0)
    assert _cost(mode=Mode.BIKE, length=1000.0, surface="asphalt", highway="secondary", params=main) == pytest.approx(
        1000.0 + 2.0 * 1000.0 * RoadConfig.ROAD_WEIGHT["secondary"]
    )
    assert _cost(mode=Mode.BIKE, length=1000.0, surface="asphalt", highway="residential", params=main) == pytest.approx(
        1000.0 + 2.0 * 1000.0 * RoadConfig.ROAD_WEIGHT["residential"]
    )

    # additive: uphill + unpaved + main-road all stack on top of length
    add = zero_params(extra_km_per_uphill_100m=5.0, extra_km_per_unpaved_km=1.0, extra_km_per_main_road_km=1.0)
    assert _cost(
        mode=Mode.BIKE, length=1000.0, surface="gravel", highway="secondary", to_elev=100.0, params=add
    ) == pytest.approx(
        1000.0 + 5000.0 + 1000.0 * SurfaceConfig.SURFACE_WEIGHT["gravel"] + 1000.0 * RoadConfig.ROAD_WEIGHT["secondary"]
    )

    # never below raw length (penalties fire, total > length)
    floor = zero_params(extra_km_per_uphill_100m=10.0, extra_km_per_unpaved_km=3.0, extra_km_per_main_road_km=3.0)
    c = _cost(mode=Mode.BIKE, length=500.0, surface="gravel", highway="primary", to_elev=100.0, params=floor)
    assert (
        c
        == pytest.approx(
            500.0
            + 10000.0
            + 3.0 * 500.0 * SurfaceConfig.SURFACE_WEIGHT["gravel"]
            + 3.0 * 500.0 * RoadConfig.ROAD_WEIGHT["primary"]
        )
        and c > 500.0
    )

    # rail: per-km rail charge only; terrain/surface/road ignored; boarding NOT here
    assert _cost(
        mode=Mode.RAIL,
        length=10_000.0,
        surface="mud",
        highway="primary",
        to_elev=500.0,
        params=zero_params(extra_km_per_rail_km=2.0, extra_km_per_boarding=15.0),
    ) == pytest.approx(30_000.0)
    cheap = zero_params(extra_km_per_rail_km=0.1, extra_km_per_boarding=1.0)
    dear = zero_params(extra_km_per_rail_km=5.0, extra_km_per_boarding=80.0)
    assert _cost(mode=Mode.RAIL, length=5_000.0, params=dear) > _cost(mode=Mode.RAIL, length=5_000.0, params=cheap)

    # station: length + half the boarding charge (board + alight = one full boarding)
    assert _cost(mode=Mode.STATION, length=150.0, to_elev=100.0) == 150.0  # boarding 0 → pure length
    assert _cost(mode=Mode.STATION, length=150.0, params=zero_params(extra_km_per_boarding=10.0)) == pytest.approx(
        5150.0
    )

    # unknown mode fails loud
    with pytest.raises(AssertionError, match="unknown edge mode"):
        _cost(mode="teleport", length=1.0)

    # a whole mixed-mode table costed in ONE vectorized call
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
    got = edge_cost_array(edges_df=edges_df, elev_by_osmid=dict.fromkeys([1, 2, 3, 4], 0.0), params=params)
    bike_expected = (
        1000.0 + 1000.0 * SurfaceConfig.SURFACE_WEIGHT["gravel"] + 1000.0 * RoadConfig.ROAD_WEIGHT["secondary"]
    )
    assert np.allclose(got, [bike_expected, 10_000.0 + 20_000.0, 150.0 + 5000.0])


def test_untagged_default_weight_applies_in_cost():
    # REGRESSION: an untagged surface/highway edge is costed at the flat DEFAULT_WEIGHT (1.0) — the
    # pessimistic prior for the ~44% of length with no surface tag (paper §4; class-conditional prior
    # is a follow-up needing an artifact rebuild). Both penalties fire at exactly weight·length.
    unp = zero_params(extra_km_per_unpaved_km=1.0)
    assert _cost(mode=Mode.BIKE, length=1000.0, surface=None, highway="residential", params=unp) == pytest.approx(
        1000.0 + 1000.0 * SurfaceConfig.DEFAULT_WEIGHT
    )
    main = zero_params(extra_km_per_main_road_km=1.0)
    assert _cost(mode=Mode.BIKE, length=1000.0, surface="asphalt", highway=None, params=main) == pytest.approx(
        1000.0 + 1000.0 * RoadConfig.DEFAULT_WEIGHT
    )


def test_continuous_cost_ordering_beyond_colour_bucket():
    # REGRESSION (continuous tiers): identical-length bike edges cost STRICTLY more as the surface
    # roughens (asphalt < compacted < gravel < grass, by Crr) and the road stress rises by LTS level
    # (residential LTS2 < tertiary LTS3 < secondary LTS4) — orderings the old integer buckets collapsed.
    surf = zero_params(extra_km_per_unpaved_km=1.0)
    surfaces = ["asphalt", "compacted", "gravel", "grass"]
    surf_costs = [_cost(mode=Mode.BIKE, length=1000.0, surface=s, highway="residential", params=surf) for s in surfaces]
    assert all(surf_costs[i] < surf_costs[i + 1] for i in range(len(surf_costs) - 1))

    road = zero_params(extra_km_per_main_road_km=1.0)
    roads = ["residential", "tertiary", "secondary"]  # LTS 2→3→4 → weight 0.0 < 0.5 < 1.0
    road_costs = [_cost(mode=Mode.BIKE, length=1000.0, surface="asphalt", highway=h, params=road) for h in roads]
    assert all(road_costs[i] < road_costs[i + 1] for i in range(len(road_costs) - 1))

    # Colour ≠ cost: 'compacted' and 'fine_gravel' both COLOUR paved (tier 0) yet cost differently —
    # proving the continuous weight carries information the binary colour bucket throws away.
    assert surface_tier(surface="compacted") == surface_tier(surface="fine_gravel") == 0
    assert _cost(mode=Mode.BIKE, length=1000.0, surface="compacted", highway="residential", params=surf) < _cost(
        mode=Mode.BIKE, length=1000.0, surface="fine_gravel", highway="residential", params=surf
    )
