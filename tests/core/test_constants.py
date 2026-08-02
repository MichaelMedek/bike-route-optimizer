"""constants tests — config values + the load-time invariants they must satisfy.

One TestFoo per config/enum/dataclass: each asserts the values other modules depend on and the
relationships the module's own module-level asserts guarantee (so a bad edit fails loud here too).
"""

import pytest

from bike_router.core.constants import (
    PARAM_SPECS,
    Condition,
    CorridorConfig,
    CostConfig,
    DEMConfig,
    GeoConfig,
    GmapsConfig,
    GpxConfig,
    Grade,
    GradeConfig,
    GraphConfig,
    Mode,
    NodeType,
    NominatimConfig,
    OutputConfig,
    Palette,
    PhotonConfig,
    PlotConfig,
    RailConfig,
    RoadConfig,
    RoutingDefaults,
    RoutingParams,
    RoutingParamSpec,
    SanityConfig,
    Schema,
    SessionKey,
    SpeedConfig,
    SurfaceConfig,
    SurfaceLabel,
    WebMapConfig,
    color_tier,
    road_weight_from_lts,
    surface_weight_from_crr,
)
from bike_router.core.errors import ParamOutOfRangeError
from tests.conftest import params as make_params


class TestOutputConfig:
    def test_output_dir_under_project_root(self):
        assert OutputConfig.OUTPUT_DIR.name == "output"


class TestDEMConfig:
    def test_dem_path_is_a_tif_in_data(self):
        assert DEMConfig.EURODEM_PATH.suffix == ".tif" and DEMConfig.EURODEM_PATH.parent.name == "data"


class TestMode:
    def test_members_are_strings(self):
        assert Mode.BIKE == "bike" and Mode.RAIL == "rail" and Mode.STATION == "station"
        assert set(Mode) == {Mode.BIKE, Mode.RAIL, Mode.STATION}


class TestNodeType:
    def test_two_kinds_are_strings(self):
        assert NodeType.BIKE == "bike" and NodeType.RAIL == "rail"
        assert set(NodeType) == {NodeType.BIKE, NodeType.RAIL}


class TestRailConfig:
    def test_speed_wait_and_tags(self):
        assert RailConfig.RAIL_SPEED_KMH > 0 and RailConfig.BOARDING_WAIT_S > 0
        assert RailConfig.STATION_RADIUS_M > 0 and RailConfig.STATION_MAX_ENTRANCES >= 1
        assert SpeedConfig.BASE_KMH_AT_WEIGHT0 < RailConfig.RAIL_SPEED_KMH  # rail beats any bike leg
        assert "rail" in RailConfig.RAIL_TAGS and "station" in RailConfig.STATION_TAGS


class TestGraphConfig:
    def test_tiling_and_tolerances_sane(self):
        assert GraphConfig.TILE_DEG > 0 and GraphConfig.CONSOLIDATION_TOLERANCE_M >= 0
        assert len(GraphConfig.DACH_BBOX_DEG) == 4 and GraphConfig.HEIGHT_DIFF_TOLERANCE_M > 0


class TestGeoConfig:
    def test_earth_constants(self):
        assert GeoConfig.EARTH_RADIUS_M == 6_371_000.0 and GeoConfig.METERS_PER_DEGREE_EQUATOR == 111_320.0


class TestCorridorConfig:
    def test_rail_tube_wider_and_trip_bounds_ordered(self):
        assert CorridorConfig.RAIL_HALF_WIDTH_KM > CorridorConfig.BIKE_HALF_WIDTH_KM
        assert CorridorConfig.BIKE_HALF_WIDTH_KM > 0 and CorridorConfig.RAIL_HALF_WIDTH_KM > 0
        assert CorridorConfig.BIKE_EXTEND_KM >= 0 and CorridorConfig.RAIL_EXTEND_KM >= 0
        assert 0 < CorridorConfig.MIN_TRIP_KM < CorridorConfig.MAX_TRIP_KM
        assert CorridorConfig.MAX_ROUTE_EDGES > 0


class TestPalette:
    def test_hex_to_rgb_and_condition_grade_scales(self):
        assert Palette.hex_to_rgb(hex_color="#1565c0") == (0x15, 0x65, 0xC0)
        # the two 3-colour scales share swatches; a train stays purple on BOTH.
        assert Palette.CONDITION_COLORS["good"] == Palette.BLUE
        assert Palette.CONDITION_COLORS["main road + unpaved"] == Palette.RED  # both → main-road red
        assert Palette.GRADE_COLORS["train"] == Palette.RAIL == Palette.CONDITION_COLORS["train"]
        assert Palette.GRADE_COLORS["downhill"] == Palette.GREEN


class TestSurfaceConfig:
    def test_tiers_labels_and_default(self):
        assert SurfaceConfig.SURFACE_TIER  # not empty
        assert set(SurfaceConfig.SURFACE_TIER.values()) <= {0, 1}  # capped binary colour bucket
        assert SurfaceConfig.SURFACE_TIER["asphalt"] == 0 and SurfaceConfig.SURFACE_TIER["gravel"] == 1
        assert SurfaceConfig.SURFACE_TIER["grass"] == 1  # roughest rideable, capped at 1
        assert set(SurfaceConfig.TIER_LABEL_COLORS) == set(SurfaceConfig.SURFACE_TIER.values())
        assert SurfaceConfig.DEFAULT_TIER in SurfaceConfig.SURFACE_TIER.values()

    def test_weight_derived_from_cited_crr_and_tier_capped(self):
        # Chain: raw cited Crr → weight = surface_weight_from_crr → tier = color_tier(weight) (capped 0/1).
        # No hand-set weight/tier: both are FUNCTIONS of SURFACE_CRR, so nothing can drift.
        assert set(SurfaceConfig.SURFACE_CRR) == set(SurfaceConfig.SURFACE_WEIGHT) == set(SurfaceConfig.SURFACE_TIER)
        assert all(
            SurfaceConfig.SURFACE_WEIGHT[k] == surface_weight_from_crr(crr=SurfaceConfig.SURFACE_CRR[k])
            for k in SurfaceConfig.SURFACE_CRR
        )
        assert all(
            SurfaceConfig.SURFACE_TIER[k] == color_tier(weight=w) for k, w in SurfaceConfig.SURFACE_WEIGHT.items()
        )
        assert color_tier(weight=SurfaceConfig.DEFAULT_WEIGHT) == SurfaceConfig.DEFAULT_TIER
        # Crr-ordered ⇒ weight-ordered: asphalt cheapest (0.0), firm < loose < rough.
        assert SurfaceConfig.SURFACE_WEIGHT["asphalt"] == 0.0
        assert SurfaceConfig.SURFACE_WEIGHT["compacted"] < SurfaceConfig.SURFACE_WEIGHT["gravel"]
        assert SurfaceConfig.SURFACE_WEIGHT["gravel"] < SurfaceConfig.SURFACE_WEIGHT["grass"]


class TestRoadConfig:
    def test_quiet_vs_main_and_labels(self):
        assert RoadConfig.ROAD_TIER  # not empty
        assert set(RoadConfig.ROAD_TIER.values()) <= {0, 1}
        assert RoadConfig.ROAD_TIER["residential"] == 0 and RoadConfig.ROAD_TIER["primary"] == 1
        assert RoadConfig.DEFAULT_TIER in {0, 1}
        assert set(RoadConfig.TIER_LABEL_COLORS) == set(RoadConfig.ROAD_TIER.values())

    def test_weight_derived_from_cited_lts_and_fixes_inversion(self):
        # Chain: cited LTS → weight = road_weight_from_lts → tier = color_tier(weight). No hand-set values.
        assert set(RoadConfig.ROAD_LTS) == set(RoadConfig.ROAD_WEIGHT) == set(RoadConfig.ROAD_TIER)
        assert all(
            RoadConfig.ROAD_WEIGHT[k] == road_weight_from_lts(lts=RoadConfig.ROAD_LTS[k]) for k in RoadConfig.ROAD_LTS
        )
        assert all(RoadConfig.ROAD_TIER[k] == color_tier(weight=w) for k, w in RoadConfig.ROAD_WEIGHT.items())
        assert color_tier(weight=RoadConfig.DEFAULT_WEIGHT) == RoadConfig.DEFAULT_TIER
        assert RoadConfig.ROAD_WEIGHT["cycleway"] == 0.0 and RoadConfig.ROAD_WEIGHT["trunk"] == 1.0
        # Inversion fix: 'unclassified' (LTS 3) is no longer WORSE than 'tertiary' — the cited LTS makes
        # them EQUAL (both 50 km/h non-residential), and both strictly below secondary/primary (LTS 4).
        assert RoadConfig.ROAD_WEIGHT["unclassified"] == RoadConfig.ROAD_WEIGHT["tertiary"]
        assert (
            RoadConfig.ROAD_WEIGHT["tertiary"]
            < RoadConfig.ROAD_WEIGHT["secondary"]
            == RoadConfig.ROAD_WEIGHT["primary"]
        )
        assert all(0.0 <= w <= 1.0 for w in RoadConfig.ROAD_WEIGHT.values())


def test_color_tier():
    # Binary colour cap: 0 iff round(weight) == 0, else 1 — even a very bad surface (weight 4, 7) stays 1.
    assert color_tier(weight=0.0) == 0 and color_tier(weight=0.4) == 0  # rounds to 0 → good
    assert color_tier(weight=0.6) == 1 and color_tier(weight=1.0) == 1  # rounds to ≥1 → bad
    assert color_tier(weight=7.33) == 1  # capped: never a third colour class


def test_surface_weight_from_crr():
    # Weight = extra equivalent-km per km = crr/Crr_asphalt − 1, floored at 0.
    assert surface_weight_from_crr(crr=SurfaceConfig.CRR_ASPHALT) == 0.0  # asphalt anchor
    assert surface_weight_from_crr(crr=2 * SurfaceConfig.CRR_ASPHALT) == pytest.approx(1.0)  # 2× Crr → weight 1
    assert surface_weight_from_crr(crr=0.5 * SurfaceConfig.CRR_ASPHALT) == 0.0  # smoother → floored at 0
    assert surface_weight_from_crr(crr=0.030) == pytest.approx(0.030 / SurfaceConfig.CRR_ASPHALT - 1.0)


def test_road_weight_from_lts():
    # LTS≤2 is the low-stress network → free; above it normalise (lts−2)/(LTS_MAX−2): 3→0.5, 4→1.0.
    assert road_weight_from_lts(lts=1) == 0.0
    assert road_weight_from_lts(lts=2) == 0.0
    assert road_weight_from_lts(lts=3) == pytest.approx(0.5)
    assert road_weight_from_lts(lts=RoadConfig.LTS_MAX) == pytest.approx(1.0)


class TestRoutingDefaults:
    def test_max_extra_km_positive(self):
        assert RoutingDefaults.MAX_EXTRA_KM > 0


class TestRoutingParamSpec:
    def test_spec_fields_present(self):
        assert PARAM_SPECS  # not empty
        assert all(0 <= s.default <= RoutingDefaults.MAX_EXTRA_KM for s in PARAM_SPECS)
        spec = PARAM_SPECS[0]
        assert isinstance(spec, RoutingParamSpec)
        assert spec.field and spec.label and spec.help and 0 <= spec.default <= RoutingDefaults.MAX_EXTRA_KM


class TestRoutingParams:
    def test_specs_match_fields_and_out_of_range_raises(self):
        assert {s.field for s in PARAM_SPECS} == set(RoutingParams.__dataclass_fields__)
        with pytest.raises(ParamOutOfRangeError):
            RoutingParams(**{s.field: (999.0 if i == 0 else s.default) for i, s in enumerate(PARAM_SPECS)})

    def test_valid_values_construct(self):
        p = RoutingParams(**{s.field: s.default for s in PARAM_SPECS})
        assert p.extra_km_per_uphill_100m == PARAM_SPECS[0].default
        # a value exactly AT the max bound is accepted (no silent clamp, no raise)
        at_max = make_params(extra_km_per_main_road_km=RoutingDefaults.MAX_EXTRA_KM)
        assert at_max.extra_km_per_main_road_km == RoutingDefaults.MAX_EXTRA_KM

    def test_negative_or_above_max_raises_including_rail_params(self):
        # Every field is range-validated, terrain AND rail sliders: below 0 or above the max raises.
        with pytest.raises(ParamOutOfRangeError):
            make_params(extra_km_per_uphill_100m=-1.0)
        with pytest.raises(ParamOutOfRangeError):
            make_params(extra_km_per_uphill_100m=RoutingDefaults.MAX_EXTRA_KM + 1)
        with pytest.raises(ParamOutOfRangeError):
            make_params(extra_km_per_boarding=-5.0)
        with pytest.raises(ParamOutOfRangeError):
            make_params(extra_km_per_rail_km=RoutingDefaults.MAX_EXTRA_KM + 1)


class TestCostConfig:
    def test_uphill_reference(self):
        assert CostConfig.UPHILL_REFERENCE_M == 100.0


class TestGradeConfig:
    def test_margin_is_a_meaningful_positive_grade(self):
        assert 0 < GradeConfig.MARGIN < 1


class TestSpeedConfig:
    def test_weight_anchors_ordered_and_exceed_walk(self):
        # Continuous base-speed anchors: paved (weight 0) fastest, rough floor slower, both above walk;
        # the weight span is positive so the interpolation is well-defined.
        assert SpeedConfig.BASE_KMH_AT_WEIGHT0 > SpeedConfig.BASE_KMH_AT_WEIGHT_MAX > SpeedConfig.WALK_KMH
        assert SpeedConfig.SURFACE_WEIGHT_MAX > 0
        assert SpeedConfig.WALK_GRADE > 0


class TestGmapsConfig:
    def test_waypoint_and_url_config(self):
        assert GmapsConfig.N_WAYPOINTS >= 2 and GmapsConfig.MAX_INTERMEDIATE_WAYPOINTS == 9
        assert GmapsConfig.TRAVEL_MODE == "bicycling" and GmapsConfig.BASE_URL.startswith("https://")


class TestPlotConfig:
    def test_dpi_and_dimensions_positive(self):
        assert PlotConfig.DPI > 0 and PlotConfig.MAP_LONG_IN > PlotConfig.MAP_SHORT_MIN_IN > 0


class TestNominatimConfig:
    def test_user_agent_and_rate_limit(self):
        assert NominatimConfig.RATE_LIMIT_S >= 1.0 and "bike-route-optimizer" in NominatimConfig.USER_AGENT


class TestPhotonConfig:
    def test_limit_timeout_and_place_tag(self):
        assert PhotonConfig.LIMIT > 0 and PhotonConfig.TIMEOUT_S > 0 and PhotonConfig.PLACE_OSM_TAG == "place"


class TestGpxConfig:
    def test_unit_conversions(self):
        assert GpxConfig.METERS_PER_KM == 1000.0 and GpxConfig.SECONDS_PER_HOUR == 3600.0
        assert GpxConfig.MINUTES_PER_HOUR == 60.0


class TestSanityConfig:
    def test_min_meaningful_nodes(self):
        assert SanityConfig.MIN_MEANINGFUL_NODES > 0


class TestWebMapConfig:
    def test_mode_labels_and_rail_colour_derive_from_palette(self):
        # The label→colour map now lives in core/composition (MODE_COLORS); here just labels + colours.
        assert set(WebMapConfig.MODE_DONUT_LABELS.values()) == {"bike route", "train path"}
        assert Palette.hex_to_rgb(hex_color=Palette.RAIL) == WebMapConfig.RAIL_COLOR
        assert WebMapConfig.MAP_HEIGHT_PX > 0


class TestSchema:
    def test_on_disk_column_names(self):
        # The single-source parquet column names both the writer and reader share.
        assert Schema.OSMID == "osmid" and Schema.NODE_TYPE == "node_type"
        assert {Schema.FROM_NODE, Schema.TO_NODE, Schema.MODE} == {"from_node", "to_node", "mode"}

    def test_names_are_unique(self):
        cols = [Schema.OSMID, Schema.LAT, Schema.LON, Schema.ELEVATION_M, Schema.NODE_TYPE, Schema.STATION_NAME]
        assert len(cols) == len(set(cols))


class TestCondition:
    def test_labels_are_the_condition_colour_keys(self):
        # Every Condition label must key into the QUALITY colour scale (no drift).
        assert Condition.GOOD in Palette.CONDITION_COLORS and Condition.MAIN_ROAD_UNPAVED in Palette.CONDITION_COLORS
        assert Condition.TRAIN == "train"


class TestGrade:
    def test_labels_are_the_grade_colour_keys(self):
        assert {Grade.FLAT, Grade.UPHILL, Grade.DOWNHILL, Grade.TRAIN} <= set(Palette.GRADE_COLORS)


class TestSurfaceLabel:
    def test_human_surface_and_road_words(self):
        # Internal DISPLAY words — deliberately distinct from the raw OSM "paved"/"unpaved" tag values.
        assert SurfaceLabel.PAVED == "paved road" and SurfaceLabel.UNPAVED == "unpaved path"
        assert SurfaceLabel.QUIET_WAY == "quiet way" and SurfaceLabel.MAIN_ROAD == "main road"


class TestSessionKey:
    def test_keys_are_distinct(self):
        keys = [
            SessionKey.START_BOX,
            SessionKey.END_BOX,
            SessionKey.START_BOX_RESOLVED,
            SessionKey.END_BOX_RESOLVED,
            SessionKey.START_LATLON,
            SessionKey.END_LATLON,
            SessionKey.RESULT,
        ]
        assert len(keys) == len(set(keys))

    def test_box_keys_match_their_widget_names(self):
        # The start/end box keys ARE the text_input widget keys; resolved keys extend them by suffix.
        assert SessionKey.START_BOX == "start_box" and SessionKey.END_BOX == "end_box"
        assert f"{SessionKey.START_BOX}_resolved" == SessionKey.START_BOX_RESOLVED
        assert f"{SessionKey.END_BOX}_resolved" == SessionKey.END_BOX_RESOLVED


def test_condition_and_grade_labels_are_exactly_the_palette_scale_keys():
    # Cross-check: the label constants ARE the Palette scale keys, so classify/colour never drift.
    assert set(Palette.CONDITION_COLORS) == {
        Condition.TRAIN,
        Condition.GOOD,
        Condition.UNPAVED,
        Condition.MAIN_ROAD,
        Condition.MAIN_ROAD_UNPAVED,
    }
    assert set(Palette.GRADE_COLORS) == {Grade.TRAIN, Grade.FLAT, Grade.UPHILL, Grade.DOWNHILL}
