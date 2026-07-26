"""Light smoke tests for the Streamlit map helpers (not held to the coverage gate).

Confidence checks only: ribbon points are lifted 100 m and finite; the view
states carry the expected centre/zoom; the pydeck builders return the right
objects. The app shell itself (app_webmap.py) is not tested.
"""

import math

import pydeck as pdk
import pytest

from bike_router.constants import WebMapConfig
from bike_router.routing import shortest_route
from bike_router.track import build_track
from bike_router.webmap import (
    ViewState,
    default_view_state,
    route_ribbon_segments,
    route_view_state,
    zoom_for_span_m,
)
from bike_router.webmap_layers import (
    build_deck,
    create_endpoint_layer,
    create_route_ribbon_layers,
    create_terrain_layer,
)
from tests.conftest import make_line_graph


def _line_track():
    """A computed Track over the flat line graph (all-bike, elevations 100–130 m)."""
    graph = make_line_graph()
    node_path = shortest_route(graph=graph, source=1, target=3)
    return build_track(graph=graph, node_path=node_path)


def test_route_ribbon_segments_lift_and_single_bike_color():
    """An all-bike track → one segment, bike color, every z lifted by the float."""
    track = _line_track()
    segments = route_ribbon_segments(track=track)

    assert len(segments) == 1  # all one mode → a single run
    color, points = segments[0]
    assert color == list(WebMapConfig.BIKE_COLOR)
    for (lon, lat, z), point in zip(points, track.points, strict=True):
        assert math.isfinite(lon) and math.isfinite(lat)
        assert z == pytest.approx(point.elevation_m + WebMapConfig.RIBBON_FLOAT_ABOVE_M)


def test_route_ribbon_segments_custom_float():
    """The float offset is honoured on every segment point."""
    track = _line_track()
    _color, points = route_ribbon_segments(track=track, float_above_m=250.0)[0]
    assert all(z == pytest.approx(point.elevation_m + 250.0) for (*_, z), point in zip(points, track.points, strict=True))


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
        (list(WebMapConfig.BIKE_COLOR), [[8.0, 48.0, 1100.0], [8.01, 48.0, 1100.0]]),
        (list(WebMapConfig.RAIL_COLOR), [[8.01, 48.0, 1100.0], [8.02, 48.0, 1100.0]]),
    ]
    ribbons = create_route_ribbon_layers(segments=segments)
    assert len(ribbons) == 2  # one PathLayer per mode run
    assert all(layer.type == "PathLayer" for layer in ribbons)
    assert ribbons[0].id == "route_ribbon_0" and ribbons[1].id == "route_ribbon_1"


def test_endpoint_layer_marks_start_and_end():
    layer = create_endpoint_layer(start_latlon=(48.0, 8.0), end_latlon=(48.4, 8.6))
    assert layer.type == "ScatterplotLayer" and layer.id == "route_endpoints"
    positions = [row["position"] for row in layer.data]
    assert positions == [[8.0, 48.0], [8.6, 48.4]]  # [lon, lat] start then end
    colors = [row["color"] for row in layer.data]
    assert colors == [list(WebMapConfig.START_COLOR), list(WebMapConfig.END_COLOR)]


def test_build_deck_layer_count_and_camera():
    view = default_view_state()
    deck_terrain_only = build_deck(view=view, ribbon_segments=None)
    assert len(deck_terrain_only.layers) == 1  # terrain only
    assert deck_terrain_only.initial_view_state.latitude == pytest.approx(WebMapConfig.DEFAULT_LAT)

    # Endpoints set but no route yet → terrain + endpoint markers.
    deck_endpoints = build_deck(view=view, ribbon_segments=None, endpoints=((48.0, 8.0), (48.4, 8.6)))
    assert len(deck_endpoints.layers) == 2

    # Endpoints + a two-mode route → terrain + markers + one ribbon layer per mode run.
    two_mode = [
        (list(WebMapConfig.BIKE_COLOR), [[8.0, 48.0, 1100.0], [8.01, 48.0, 1100.0]]),
        (list(WebMapConfig.RAIL_COLOR), [[8.01, 48.0, 1100.0], [8.02, 48.0, 1100.0]]),
    ]
    deck_full = build_deck(view=view, ribbon_segments=two_mode, endpoints=((48.0, 8.0), (48.4, 8.6)))
    assert len(deck_full.layers) == 4  # terrain + markers + 2 ribbon runs
