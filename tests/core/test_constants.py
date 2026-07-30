"""constants tests — config values + the load-time invariants they must satisfy.

One TestFoo per config/enum/dataclass: each asserts the values other modules depend on and the
relationships the module's own module-level asserts guarantee (so a bad edit fails loud here too).
"""

import pytest

from bike_router.core.constants import (
    PARAM_SPECS,
    CorridorConfig,
    CostConfig,
    DEMConfig,
    GeoConfig,
    GmapsConfig,
    GpxConfig,
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
    SpeedConfig,
    SurfaceConfig,
    WebMapConfig,
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
        assert SurfaceConfig.SURFACE_TIER["asphalt"] == 0 and SurfaceConfig.SURFACE_TIER["gravel"] == 1
        assert SurfaceConfig.SURFACE_TIER["ground"] == 2
        assert set(SurfaceConfig.TIER_LABEL_COLORS) == set(SurfaceConfig.SURFACE_TIER.values())
        assert SurfaceConfig.DEFAULT_TIER in SurfaceConfig.SURFACE_TIER.values()


class TestRoadConfig:
    def test_quiet_vs_main_and_labels(self):
        assert RoadConfig.ROAD_TIER["residential"] == 0 and RoadConfig.ROAD_TIER["primary"] == 1
        assert set(RoadConfig.TIER_LABEL_COLORS) == set(RoadConfig.ROAD_TIER.values())


class TestRoutingDefaults:
    def test_max_extra_km_positive(self):
        assert RoutingDefaults.MAX_EXTRA_KM > 0


class TestRoutingParamSpec:
    def test_spec_fields_present(self):
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
    def test_base_speeds_cover_tiers_and_exceed_walk(self):
        assert set(SpeedConfig.BASE_KMH_BY_TIER) == set(SurfaceConfig.SURFACE_TIER.values())
        assert all(v > SpeedConfig.WALK_KMH for v in SpeedConfig.BASE_KMH_BY_TIER.values())
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
    def test_mode_donut_keys_match_labels_and_colours_derive_from_palette(self):
        assert set(WebMapConfig.MODE_DONUT_COLORS) == set(WebMapConfig.MODE_DONUT_LABELS.values())
        assert Palette.hex_to_rgb(hex_color=Palette.RAIL) == WebMapConfig.RAIL_COLOR
        assert WebMapConfig.MAP_HEIGHT_PX > 0
