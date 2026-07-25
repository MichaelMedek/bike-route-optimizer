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
    route_ribbon_points,
    route_view_state,
    zoom_for_span_m,
)
from bike_router.webmap_layers import build_deck, create_route_ribbon_layer, create_terrain_layer
from tests.conftest import make_line_graph


def _line_track():
    """A computed Track over the flat line graph (elevations = 100 m at every node)."""
    graph = make_line_graph()
    node_path = shortest_route(graph=graph, source=1, target=3)
    return build_track(graph=graph, node_path=node_path)


def test_route_ribbon_points_lifts_float_above_terrain():
    """Every ribbon z equals its node elevation lifted by the float offset, all finite."""
    track = _line_track()
    points = route_ribbon_points(track=track)

    assert len(points) == len(track.points) >= 2
    for (lon, lat, z), point in zip(points, track.points, strict=True):
        assert math.isfinite(lon) and math.isfinite(lat)
        assert z == pytest.approx(point.elevation_m + WebMapConfig.RIBBON_FLOAT_ABOVE_M)


def test_route_ribbon_points_custom_float():
    """The float offset is honoured."""
    track = _line_track()
    points = route_ribbon_points(track=track, float_above_m=250.0)
    assert all(
        z == pytest.approx(point.elevation_m + 250.0) for (*_, z), point in zip(points, track.points, strict=True)
    )


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

    ribbon = create_route_ribbon_layer(points=[[8.0, 48.0, 1100.0], [8.01, 48.0, 1100.0]])
    assert ribbon.type == "PathLayer" and ribbon.id == "route_ribbon"


def test_build_deck_layer_count_and_camera():
    view = default_view_state()
    deck_no_route = build_deck(view=view, ribbon_points=None)
    assert len(deck_no_route.layers) == 1  # terrain only
    assert deck_no_route.initial_view_state.latitude == pytest.approx(WebMapConfig.DEFAULT_LAT)

    deck_with_route = build_deck(view=view, ribbon_points=[[8.0, 48.0, 1100.0], [8.01, 48.0, 1100.0]])
    assert len(deck_with_route.layers) == 2  # terrain + ribbon
