"""Light smoke tests for the Streamlit map helpers (not held to the coverage gate).

Confidence checks only: ribbon points are lifted 100 m and finite; the view states carry
the expected centre/zoom; the pydeck builders return the right objects; the two colour
scales (quality + grade) map as expected. The app shell itself (app_webmap.py) is not tested.
"""

import math

import pydeck as pdk
import pytest

from bike_router.core.constants import Mode, Palette, WebMapConfig
from bike_router.core.geo import haversine_distance_m
from bike_router.core.track import build_track, grade_color, segment_color
from bike_router.ui.webmap import (
    GRADE_SCALE,
    MODE_DONUT_COLORS,
    QUALITY_SCALE,
    RibbonSegment,
    ViewState,
    default_view_state,
    elevation_profile_chart,
    profile_markers,
    ribbon_width_m,
    route_ribbon_segments,
    route_view_state,
    zoom_for_span_m,
)
from bike_router.ui.webmap_layers import (
    build_deck,
    create_endpoint_layer,
    create_route_ribbon_layers,
    create_station_layer,
    create_terrain_layer,
)
from tests.conftest import make_line_route


def _rgb(hex_color: str) -> list[int]:
    """Palette hex → RGB list, for comparing against segment_color output."""
    return list(Palette.hex_to_rgb(hex_color=hex_color))


def _line_track():
    """A computed Track over the flat line route (all-bike, elevations 100–130 m)."""
    return build_track(route=make_line_route())


def test_segment_color_is_quality_scale_three_colours_plus_rail():
    """Rail → purple; good → blue; unpaved → orange; main road (or both) → red."""
    assert segment_color(mode=str(Mode.RAIL), surface_bad=False, road_bad=False) == _rgb(Palette.RAIL)
    assert segment_color(mode=str(Mode.RAIL), surface_bad=True, road_bad=True) == _rgb(Palette.RAIL)  # rail ignores
    assert segment_color(mode=str(Mode.BIKE), surface_bad=False, road_bad=False) == _rgb(Palette.BLUE)  # good
    assert segment_color(mode=str(Mode.BIKE), surface_bad=True, road_bad=False) == _rgb(Palette.ORANGE)  # unpaved
    assert segment_color(mode=str(Mode.BIKE), surface_bad=False, road_bad=True) == _rgb(Palette.RED)  # main road
    assert segment_color(mode=str(Mode.BIKE), surface_bad=True, road_bad=True) == _rgb(
        Palette.RED
    )  # main+unpaved → red
    # exactly three bike colours on the quality scale (good/unpaved/main), main+unpaved folds into red
    distinct = {
        tuple(segment_color(mode=str(Mode.BIKE), surface_bad=s, road_bad=r))
        for s in (False, True)
        for r in (False, True)
    }
    assert len(distinct) == 3


def test_grade_color_is_grade_scale_three_colours():
    """Flat → blue; uphill (steep +) → red; downhill (steep −) → green; rail keeps purple."""
    assert grade_color(mode=str(Mode.BIKE), grade=0.0) == _rgb(Palette.BLUE)  # flat
    assert grade_color(mode=str(Mode.BIKE), grade=0.10) == _rgb(Palette.RED)  # steep uphill
    assert grade_color(mode=str(Mode.BIKE), grade=-0.10) == _rgb(Palette.GREEN)  # steep downhill
    assert grade_color(mode=str(Mode.RAIL), grade=0.10) == _rgb(Palette.RAIL)  # rail → purple (train)


def test_route_ribbon_segments_quality_scale_blue_lifted_and_width_inverse_speed():
    """All-bike asphalt/residential track → every run blue (good); z lifted; width ∝ 1/speed."""
    track = _line_track()
    segments = route_ribbon_segments(track=track, color_scale=QUALITY_SCALE)

    assert segments, "expected at least one run"
    for seg in segments:
        assert seg.color == _rgb(Palette.BLUE)  # good surface + quiet road → blue
        assert seg.width_m > 0
        for lon, lat, _z in seg.points:
            assert math.isfinite(lon) and math.isfinite(lat)
    # Pipe-flow width (∝ 1/√speed): the slow uphill run is WIDER than the fast flat/downhill.
    widths = sorted(seg.width_m for seg in segments)
    assert len(set(widths)) >= 2, "uphill and flat/downhill legs must differ in width"
    fastest = max(p.speed_kmh for p in track.points)  # 25 km/h (tier-0 base, flat/downhill)
    slowest = min(p.speed_kmh for p in track.points)  # the climb
    assert slowest < fastest
    assert min(widths) == pytest.approx(ribbon_width_m(speed_kmh=fastest))
    assert max(widths) == pytest.approx(ribbon_width_m(speed_kmh=slowest))


def test_route_ribbon_grade_scale_colours_uphill_and_downhill_distinctly():
    # The line route climbs 1→2 then descends 2→3, so the grade scale must show a red (uphill)
    # run and a green (downhill) run — distinct from the quality scale's all-blue.
    track = _line_track()
    colors = {tuple(seg.color) for seg in route_ribbon_segments(track=track, color_scale=GRADE_SCALE)}
    assert tuple(_rgb(Palette.RED)) in colors  # the uphill leg
    assert tuple(_rgb(Palette.GREEN)) in colors  # the downhill leg


def test_ribbon_width_m_pipe_flow_quarter_speed_doubles_width():
    # Water-in-a-pipe: area×speed conserved, width is the diameter (area ∝ width²), so
    # width ∝ 1/√speed → 4× slower = 2× wider (4× area). Anchored at REF_SPEED → REF_WIDTH.
    ref_speed, ref_width = WebMapConfig.RIBBON_REF_SPEED_KMH, WebMapConfig.RIBBON_REF_WIDTH_M
    assert ribbon_width_m(speed_kmh=ref_speed) == pytest.approx(ref_width)
    assert ribbon_width_m(speed_kmh=ref_speed / 4) == pytest.approx(ref_width * 2)  # 4× slower → 2× wide
    assert ribbon_width_m(speed_kmh=ref_speed * 4) == pytest.approx(ref_width / 2)  # 4× faster → half


def test_rail_and_station_segments_use_fixed_width():
    # make_rail_route: bike node → station hop (walk pace) → rail ride (80 km/h). Neither the
    # station nor the rail segment should get the inverted-speed width — both draw the fixed
    # RIBBON_REF_WIDTH_M (so a 5 km/h station hop is NOT the map's fattest ribbon).
    from tests.conftest import make_rail_route

    track = build_track(route=make_rail_route())
    segments = route_ribbon_segments(track=track, color_scale=QUALITY_SCALE)
    non_bike = [seg for seg in segments if seg.color != _rgb(Palette.BLUE)]
    assert non_bike, "expected rail/station runs on this train route"
    assert all(seg.width_m == WebMapConfig.RIBBON_REF_WIDTH_M for seg in non_bike)


def test_ribbon_segment_tooltip_describes_bike_condition():
    # A pedalled segment's tooltip names surface, road, gradient sign, and est. speed.
    track = _line_track()
    tips = " | ".join(seg.tooltip for seg in route_ribbon_segments(track=track))
    assert "paved" in tips and "quiet way" in tips
    assert "km/h" in tips
    assert "uphill" in tips or "downhill" in tips or "flat" in tips


def test_route_ribbon_segments_custom_float():
    """The float offset is honoured on every segment point."""
    track = _line_track()
    points = route_ribbon_segments(track=track, float_above_m=250.0)[0].points
    assert all(math.isfinite(z) and z > 0 for *_, z in points)
    first_point = track.points[0]
    assert points[0][2] == pytest.approx(first_point.elevation_m + 250.0)


def test_elevation_profile_chart_distance_elevation_line_coloured_by_mode():
    # Plotly profile: distance × elevation, line tinted by mode with the SAME colours as the
    # "By mode" donut. The all-bike line graph rises 100→130→100 m → one bike-blue trace.
    track = _line_track()
    fig = elevation_profile_chart(track=track)
    assert len(fig.data) == 1  # single contiguous bike run
    trace = fig.data[0]
    assert trace.name == WebMapConfig.MODE_DONUT_LABELS[Mode.BIKE]
    assert trace.line.color == MODE_DONUT_COLORS[WebMapConfig.MODE_DONUT_LABELS[Mode.BIKE]]  # matches the donut
    assert max(trace.y) == pytest.approx(130.0) and min(trace.y) == pytest.approx(100.0)
    assert trace.x[0] == pytest.approx(0.0) and trace.x[-1] > 0  # cumulative distance
    lo, hi = fig.layout.yaxis.range
    assert lo < 100.0 and hi > 130.0  # padded elevation range (not a flat/auto axis)


def test_elevation_profile_overlays_markers():
    """Given projected markers, the profile adds a labelled scatter trace at their dist/elev."""
    track = _line_track()
    fig = elevation_profile_chart(track=track, markers=[(0.0, 100.0, "Start"), (1.6, 100.0, "End")])
    marker_trace = next(t for t in fig.data if t.mode and "markers" in t.mode)
    assert list(marker_trace.text) == ["Start", "End"]
    assert marker_trace.x[0] == pytest.approx(0.0) and marker_trace.x[-1] == pytest.approx(1.6)


def test_profile_markers_names_endpoints_stations_and_villages():
    """profile_markers labels endpoints from the typed text and each waypoint via village_of."""
    from types import SimpleNamespace

    result = SimpleNamespace(track=_line_track(), rail_legs=[], waypoints=[(48.0, 8.01)])
    placed = profile_markers(
        result=result,
        start_latlon=(48.0, 8.0, 100.0),
        end_latlon=(48.0, 8.02, 100.0),
        start_name="Freudenstadt",
        end_name="Pforzheim",
        village_of=lambda lat, lon: "Baiersbronn",
    )
    assert [lab for _d, _e, lab in placed] == ["Freudenstadt", "Pforzheim", "Baiersbronn"]


def test_profile_markers_drops_waypoint_with_no_village():
    """A waypoint whose reverse lookup returns None is dropped (no unnamed marker)."""
    from types import SimpleNamespace

    result = SimpleNamespace(track=_line_track(), rail_legs=[], waypoints=[(48.0, 8.01)])
    placed = profile_markers(
        result=result,
        start_latlon=(48.0, 8.0, 100.0),
        end_latlon=(48.0, 8.02, 100.0),
        start_name="A",
        end_name="B",
        village_of=lambda lat, lon: None,
    )
    assert [lab for _d, _e, lab in placed] == ["A", "B"]  # waypoint dropped


def test_default_view_state_is_dach_overview():
    view = default_view_state()
    assert view == ViewState(
        latitude=WebMapConfig.DEFAULT_LAT,
        longitude=WebMapConfig.DEFAULT_LON,
        zoom=WebMapConfig.DEFAULT_ZOOM,  # far-out DACH overview, not the closer route zoom
        pitch=WebMapConfig.DEFAULT_PITCH,
        bearing=WebMapConfig.DEFAULT_BEARING,
    )


def test_route_view_state_centres_on_midpoint():
    start, end = (48.0, 8.0), (48.4, 8.6)
    view = route_view_state(start_latlon=start, end_latlon=end)
    assert view.latitude == pytest.approx(48.2)
    assert view.longitude == pytest.approx(8.3)
    assert view.bearing == WebMapConfig.DEFAULT_BEARING
    assert view.pitch == WebMapConfig.DEFAULT_PITCH
    span_m = haversine_distance_m(lat_a=start[0], lon_a=start[1], lat_b=end[0], lon_b=end[1])
    assert view.zoom == pytest.approx(zoom_for_span_m(span_m=span_m))


def test_zoom_for_span_anchor_and_clamps():
    assert zoom_for_span_m(span_m=WebMapConfig.ZOOM_SPAN_ANCHOR_M) == pytest.approx(WebMapConfig.VIEWING_ZOOM)
    assert zoom_for_span_m(span_m=WebMapConfig.ZOOM_SPAN_ANCHOR_M / 2) == pytest.approx(WebMapConfig.VIEWING_ZOOM + 1)
    assert zoom_for_span_m(span_m=1.0) == WebMapConfig.VIEWING_ZOOM + WebMapConfig.ZOOM_STEPS_IN
    assert zoom_for_span_m(span_m=1e9) == WebMapConfig.VIEWING_ZOOM - WebMapConfig.ZOOM_STEPS_OUT


def _seg(color: list[int], width: float, points: list[list[float]], tooltip: str = "t") -> RibbonSegment:
    return RibbonSegment(color=color, width_m=width, points=points, tooltip=tooltip)


def test_layer_builders_return_expected_pydeck_layers():
    terrain = create_terrain_layer(mesh_max_error=1.0)
    assert isinstance(terrain, pdk.Layer)
    assert terrain.type == "TerrainLayer" and terrain.id == "terrain_3d"

    segments = [
        _seg(_rgb(Palette.BLUE), 20.0, [[8.0, 48.0, 1100.0], [8.01, 48.0, 1100.0]], tooltip="paved · quiet way"),
        _seg(list(WebMapConfig.RAIL_COLOR), 8.0, [[8.01, 48.0, 1100.0], [8.02, 48.0, 1100.0]], tooltip="Train: A → B"),
    ]
    ribbons = create_route_ribbon_layers(segments=segments)
    assert len(ribbons) == 1  # ONE PathLayer holding all runs (uniform picking across the ribbon)
    layer = ribbons[0]
    assert layer.type == "PathLayer" and layer.pickable and layer.id == "route_ribbon"
    assert len(layer.data) == 2  # both runs are rows in the single layer
    assert layer.data[0]["tooltip"] == "paved · quiet way"  # per-segment tooltip carried
    assert layer.data[1]["tooltip"] == "Train: A → B"


def test_endpoint_layer_marks_start_and_end_with_labels():
    layer = create_endpoint_layer(
        start=(48.0, 8.0, 300.0), end=(48.4, 8.6, 500.0), start_label="A (300 m)", end_label="B (500 m)"
    )
    assert layer.type == "ScatterplotLayer" and layer.id == "route_endpoints" and layer.pickable
    lift = WebMapConfig.RIBBON_FLOAT_ABOVE_M
    positions = [row["position"] for row in layer.data]
    assert positions == [[8.0, 48.0, 300.0 + lift], [8.6, 48.4, 500.0 + lift]]  # [lon, lat, elev+lift]
    assert [row["color"] for row in layer.data] == [list(WebMapConfig.START_COLOR), list(WebMapConfig.END_COLOR)]
    assert [row["tooltip"] for row in layer.data] == ["A (300 m)", "B (500 m)"]
    assert layer.parameters == {"depthTest": False}  # drawn on top of terrain, never buried


def test_station_layer_marks_hops_rail_coloured_and_smaller():
    layer = create_station_layer(stations=[(48.5, 8.4, 700.0, "Freudenstadt Stadt (700 m)")])
    assert layer.type == "ScatterplotLayer" and layer.id == "route_stations" and layer.pickable
    assert layer.get_radius == WebMapConfig.STATION_MARKER_RADIUS_M < WebMapConfig.MARKER_RADIUS_M  # smaller
    assert layer.data[0]["color"] == list(WebMapConfig.RAIL_COLOR)  # rail purple
    assert layer.data[0]["tooltip"] == "Freudenstadt Stadt (700 m)"


def test_build_deck_layer_count_and_camera():
    view = default_view_state()
    deck_terrain_only = build_deck(view=view, ribbon_segments=None)
    assert len(deck_terrain_only.layers) == 1  # terrain only
    assert deck_terrain_only.initial_view_state.latitude == pytest.approx(WebMapConfig.DEFAULT_LAT)

    deck_endpoints = build_deck(view=view, ribbon_segments=None, endpoints=((48.0, 8.0, 300.0), (48.4, 8.6, 500.0)))
    assert len(deck_endpoints.layers) == 2

    two_run = [
        _seg(_rgb(Palette.BLUE), 20.0, [[8.0, 48.0, 1100.0], [8.01, 48.0, 1100.0]]),
        _seg(list(WebMapConfig.RAIL_COLOR), 8.0, [[8.01, 48.0, 1100.0], [8.02, 48.0, 1100.0]]),
    ]
    deck_full = build_deck(
        view=view,
        ribbon_segments=two_run,
        endpoints=((48.0, 8.0, 300.0), (48.4, 8.6, 500.0)),
        stations=[(48.01, 8.01, 500.0, "S (500 m)")],
    )
    ids = [layer.id for layer in deck_full.layers]
    assert ids == ["terrain_3d", "route_stations", "route_endpoints", "route_ribbon"]
