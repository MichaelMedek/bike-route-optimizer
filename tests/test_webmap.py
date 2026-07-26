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
    ViewState,
    default_view_state,
    route_ribbon_segments,
    route_view_state,
    segment_color,
    zoom_for_span_m,
)
from bike_router.webmap_layers import (
    build_deck,
    create_endpoint_layer,
    create_route_ribbon_layers,
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


def test_route_ribbon_segments_green_lifted_and_width_from_speed():
    """All-bike asphalt/residential track → every run green; z lifted; width ∝ speed."""
    track = _line_track()
    segments = route_ribbon_segments(track=track)

    assert segments, "expected at least one run"
    for color, width, points in segments:
        assert color == _rgb(Palette.GOOD)  # good surface + quiet road → green
        assert width > 0
        for lon, lat, _z in points:
            assert math.isfinite(lon) and math.isfinite(lat)
    # Width tracks segment speed: the flat/downhill run rides at the tier-0 base (25 km/h)
    # → the widest run; the uphill climb is slower → strictly narrower.
    widths = sorted(width for _c, width, _p in segments)
    assert len(set(widths)) >= 2, "uphill and flat/downhill legs must differ in width"
    fastest = max(p.speed_kmh for p in track.points)
    assert fastest == 25.0  # tier-0 base on the flat/downhill leg
    assert max(widths) == pytest.approx(fastest * WebMapConfig.RIBBON_WIDTH_PER_KMH_M)
    assert min(widths) < max(widths)  # the climb is genuinely thinner


def test_route_ribbon_segments_custom_float():
    """The float offset is honoured on every segment point."""
    track = _line_track()
    _color, _width, points = route_ribbon_segments(track=track, float_above_m=250.0)[0]
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


def test_layer_builders_return_expected_pydeck_layers():
    terrain = create_terrain_layer()
    assert isinstance(terrain, pdk.Layer)
    assert terrain.type == "TerrainLayer" and terrain.id == "terrain_3d"

    segments = [
        (_rgb(Palette.GOOD), 20.0, [[8.0, 48.0, 1100.0], [8.01, 48.0, 1100.0]]),
        (list(WebMapConfig.RAIL_COLOR), 80.0, [[8.01, 48.0, 1100.0], [8.02, 48.0, 1100.0]]),
    ]
    ribbons = create_route_ribbon_layers(segments=segments)
    assert len(ribbons) == 2  # one PathLayer per run
    assert all(layer.type == "PathLayer" for layer in ribbons)
    assert ribbons[0].id == "route_ribbon_0" and ribbons[1].id == "route_ribbon_1"


def test_endpoint_layer_marks_start_and_end():
    # endpoints are (lat, lon, elevation_m); markers hover RIBBON_FLOAT_ABOVE_M above.
    layer = create_endpoint_layer(start=(48.0, 8.0, 300.0), end=(48.4, 8.6, 500.0))
    assert layer.type == "ScatterplotLayer" and layer.id == "route_endpoints"
    lift = WebMapConfig.RIBBON_FLOAT_ABOVE_M
    positions = [row["position"] for row in layer.data]
    assert positions == [[8.0, 48.0, 300.0 + lift], [8.6, 48.4, 500.0 + lift]]  # [lon, lat, elev+lift]
    colors = [row["color"] for row in layer.data]
    assert colors == [list(WebMapConfig.START_COLOR), list(WebMapConfig.END_COLOR)]
    assert layer.parameters == {"depthTest": False}  # drawn on top of terrain, never buried


def test_build_deck_layer_count_and_camera():
    view = default_view_state()
    deck_terrain_only = build_deck(view=view, ribbon_segments=None)
    assert len(deck_terrain_only.layers) == 1  # terrain only
    assert deck_terrain_only.initial_view_state.latitude == pytest.approx(WebMapConfig.DEFAULT_LAT)

    # Endpoints set but no route yet → terrain + endpoint markers.
    deck_endpoints = build_deck(view=view, ribbon_segments=None, endpoints=((48.0, 8.0, 300.0), (48.4, 8.6, 500.0)))
    assert len(deck_endpoints.layers) == 2

    # Endpoints + a two-run route → terrain + markers + one ribbon layer per run.
    two_run = [
        (_rgb(Palette.GOOD), 20.0, [[8.0, 48.0, 1100.0], [8.01, 48.0, 1100.0]]),
        (list(WebMapConfig.RAIL_COLOR), 80.0, [[8.01, 48.0, 1100.0], [8.02, 48.0, 1100.0]]),
    ]
    deck_full = build_deck(view=view, ribbon_segments=two_run, endpoints=((48.0, 8.0, 300.0), (48.4, 8.6, 500.0)))
    assert len(deck_full.layers) == 4  # terrain + markers + 2 ribbon runs
