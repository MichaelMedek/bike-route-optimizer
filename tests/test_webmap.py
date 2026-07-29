"""Light smoke tests for the Streamlit map helpers (not held to the coverage gate).

Confidence checks only: ribbon points are lifted 100 m and finite; the view
states carry the expected centre/zoom; the pydeck builders return the right
objects. The app shell itself (app_webmap.py) is not tested.
"""

import math

import pydeck as pdk
import pytest

from bike_router.constants import Mode, Palette, WebMapConfig
from bike_router.routing import shortest_route
from bike_router.track import build_track
from bike_router.webmap import (
    RibbonSegment,
    ViewState,
    default_view_state,
    ribbon_width_m,
    route_ribbon_segments,
    route_view_state,
    segment_color,
    zoom_for_span_m,
)
from bike_router.webmap_layers import (
    build_deck,
    create_endpoint_layer,
    create_route_ribbon_layers,
    create_station_layer,
    create_terrain_layer,
)
from tests.conftest import make_line_graph


def _rgb(hex_color: str) -> list[int]:
    """Palette hex → RGB list, for comparing against segment_color output."""
    return list(Palette.hex_to_rgb(hex_color=hex_color))


def _line_track():
    """A computed Track over the flat line graph (all-bike, elevations 100–130 m)."""
    graph = make_line_graph()
    node_path = shortest_route(graph=graph, source=1, target=3)
    return build_track(graph=graph, node_path=node_path)


def test_segment_color_distinguishes_surface_road_and_both():
    """Rail → purple; good → green; bad surface / bad road / both → three distinct reds."""
    assert segment_color(mode=str(Mode.RAIL), surface_bad=False, road_bad=False) == _rgb(Palette.RAIL)
    assert segment_color(mode=str(Mode.RAIL), surface_bad=True, road_bad=True) == _rgb(Palette.RAIL)  # rail ignores
    assert segment_color(mode=str(Mode.BIKE), surface_bad=False, road_bad=False) == _rgb(Palette.GOOD)
    assert segment_color(mode=str(Mode.BIKE), surface_bad=True, road_bad=False) == _rgb(Palette.BAD_SURFACE)
    assert segment_color(mode=str(Mode.BIKE), surface_bad=False, road_bad=True) == _rgb(Palette.BAD_ROAD)
    assert segment_color(mode=str(Mode.BIKE), surface_bad=True, road_bad=True) == _rgb(Palette.BAD_BOTH)
    # the four pedalled reds/greens are all distinct so conditions are tellable apart
    distinct = {
        tuple(segment_color(mode=str(Mode.BIKE), surface_bad=s, road_bad=r))
        for s in (False, True)
        for r in (False, True)
    }
    assert len(distinct) == 4


def test_route_ribbon_segments_green_lifted_and_width_inverse_speed():
    """All-bike asphalt/residential track → every run green; z lifted; width ∝ 1/speed."""
    track = _line_track()
    segments = route_ribbon_segments(track=track)

    assert segments, "expected at least one run"
    for seg in segments:
        assert seg.color == _rgb(Palette.GOOD)  # good surface + quiet road → green
        assert seg.width_m > 0
        for lon, lat, _z in seg.points:
            assert math.isfinite(lon) and math.isfinite(lat)
    # Pipe-flow width (∝ 1/√speed): the slow uphill run is WIDER than the fast flat/downhill.
    widths = sorted(seg.width_m for seg in segments)
    assert len(set(widths)) >= 2, "uphill and flat/downhill legs must differ in width"
    fastest = max(p.speed_kmh for p in track.points)  # 25 km/h (tier-0 base, flat/downhill)
    slowest = min(p.speed_kmh for p in track.points)  # the climb
    assert slowest < fastest
    # fastest → narrowest, slowest → widest (∝ 1/√speed)
    assert min(widths) == pytest.approx(ribbon_width_m(speed_kmh=fastest))
    assert max(widths) == pytest.approx(ribbon_width_m(speed_kmh=slowest))


def test_ribbon_width_m_pipe_flow_quarter_speed_doubles_width():
    # Water-in-a-pipe: area×speed conserved, width is the diameter (area ∝ width²), so
    # width ∝ 1/√speed → 4× slower = 2× wider (4× area). Anchored at REF_SPEED → REF_WIDTH.
    ref_speed, ref_width = WebMapConfig.RIBBON_REF_SPEED_KMH, WebMapConfig.RIBBON_REF_WIDTH_M
    assert ribbon_width_m(speed_kmh=ref_speed) == pytest.approx(ref_width)
    assert ribbon_width_m(speed_kmh=ref_speed / 4) == pytest.approx(ref_width * 2)  # 4× slower → 2× wide
    assert ribbon_width_m(speed_kmh=ref_speed * 4) == pytest.approx(ref_width / 2)  # 4× faster → half


def test_rail_and_station_segments_use_fixed_width():
    # make_rail_graph: bike node → station hop (walk pace) → rail ride (80 km/h). Neither the
    # station nor the rail segment should get the inverted-speed width — both draw the fixed
    # RIBBON_REF_WIDTH_M (so a 5 km/h station hop is NOT the map's fattest ribbon).
    from bike_router.track import build_track
    from tests.conftest import make_rail_graph

    track = build_track(graph=make_rail_graph(), node_path=[1, 2, 3])
    segments = route_ribbon_segments(track=track)
    non_bike = [seg for seg in segments if seg.color != _rgb(Palette.GOOD)]
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


def test_default_view_state_is_freudenstadt():
    view = default_view_state()
    assert view == ViewState(
        latitude=WebMapConfig.DEFAULT_LAT,
        longitude=WebMapConfig.DEFAULT_LON,
        zoom=WebMapConfig.VIEWING_ZOOM,
        pitch=WebMapConfig.DEFAULT_PITCH,
        bearing=WebMapConfig.DEFAULT_BEARING,
    )


def test_route_view_state_centres_on_midpoint():
    view = route_view_state(start_latlon=(48.0, 8.0), end_latlon=(48.4, 8.6))
    assert view.latitude == pytest.approx(48.2)
    assert view.longitude == pytest.approx(8.3)
    assert view.bearing == WebMapConfig.DEFAULT_BEARING
    assert view.pitch == WebMapConfig.DEFAULT_PITCH


def test_zoom_for_span_anchor_and_clamps():
    assert zoom_for_span_m(span_m=WebMapConfig.ZOOM_SPAN_ANCHOR_M) == pytest.approx(WebMapConfig.VIEWING_ZOOM)
    # Half the span → one zoom level closer.
    assert zoom_for_span_m(span_m=WebMapConfig.ZOOM_SPAN_ANCHOR_M / 2) == pytest.approx(WebMapConfig.VIEWING_ZOOM + 1)
    # Tiny span clamps at the zoom-in ceiling; huge span at the zoom-out floor.
    assert zoom_for_span_m(span_m=1.0) == WebMapConfig.VIEWING_ZOOM + WebMapConfig.ZOOM_STEPS_IN
    assert zoom_for_span_m(span_m=1e9) == WebMapConfig.VIEWING_ZOOM - WebMapConfig.ZOOM_STEPS_OUT


def _seg(color: list[int], width: float, points: list[list[float]], tooltip: str = "t") -> RibbonSegment:
    return RibbonSegment(color=color, width_m=width, points=points, tooltip=tooltip)


def test_layer_builders_return_expected_pydeck_layers():
    terrain = create_terrain_layer(mesh_max_error=1.0)
    assert isinstance(terrain, pdk.Layer)
    assert terrain.type == "TerrainLayer" and terrain.id == "terrain_3d"

    segments = [
        _seg(_rgb(Palette.GOOD), 20.0, [[8.0, 48.0, 1100.0], [8.01, 48.0, 1100.0]], tooltip="paved · quiet way"),
        _seg(list(WebMapConfig.RAIL_COLOR), 8.0, [[8.01, 48.0, 1100.0], [8.02, 48.0, 1100.0]], tooltip="Train: A → B"),
    ]
    ribbons = create_route_ribbon_layers(segments=segments)
    assert len(ribbons) == 2  # one PathLayer per run
    assert all(layer.type == "PathLayer" and layer.pickable for layer in ribbons)  # pickable → tooltip
    assert ribbons[0].id == "route_ribbon_0" and ribbons[1].id == "route_ribbon_1"
    assert ribbons[0].data[0]["tooltip"] == "paved · quiet way"  # per-segment tooltip carried


def test_endpoint_layer_marks_start_and_end_with_labels():
    # endpoints are (lat, lon, elevation_m); markers hover RIBBON_FLOAT_ABOVE_M above, each
    # carrying its name+elev tooltip.
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

    # Endpoints set but no route yet → terrain + endpoint markers.
    deck_endpoints = build_deck(view=view, ribbon_segments=None, endpoints=((48.0, 8.0, 300.0), (48.4, 8.6, 500.0)))
    assert len(deck_endpoints.layers) == 2

    # Endpoints + stations + a two-run route → terrain + stations + markers + one ribbon per run.
    two_run = [
        _seg(_rgb(Palette.GOOD), 20.0, [[8.0, 48.0, 1100.0], [8.01, 48.0, 1100.0]]),
        _seg(list(WebMapConfig.RAIL_COLOR), 8.0, [[8.01, 48.0, 1100.0], [8.02, 48.0, 1100.0]]),
    ]
    deck_full = build_deck(
        view=view,
        ribbon_segments=two_run,
        endpoints=((48.0, 8.0, 300.0), (48.4, 8.6, 500.0)),
        stations=[(48.01, 8.01, 500.0, "S (500 m)")],
    )
    ids = [layer.id for layer in deck_full.layers]
    assert ids == ["terrain_3d", "route_stations", "route_endpoints", "route_ribbon_0", "route_ribbon_1"]
