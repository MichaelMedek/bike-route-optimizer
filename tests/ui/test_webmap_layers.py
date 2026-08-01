"""webmap_layers tests — the pydeck layer builders for the 3D map.

One test_<fn> per production symbol (exact-name mirror). Asserts the built pydeck Layer/Deck
types, ids, and per-row data (position lifted above terrain, colours, tooltips) without a browser.
"""

import pydeck as pdk

from bike_router.core.constants import Palette, WebMapConfig
from bike_router.ui.webmap import RibbonSegment, default_view_state
from bike_router.ui.webmap_layers import (
    _marker_layer,
    _marker_row,
    build_deck,
    create_endpoint_layer,
    create_route_ribbon_layers,
    create_terrain_layer,
    create_top_station_layer,
    create_waypoint_layer,
)


def _rgb(hex_color: str) -> list[int]:
    """Palette hex → RGB list."""
    return list(Palette.hex_to_rgb(hex_color=hex_color))


def _seg(color: list[int], width: float, points: list[list[float]], tooltip: str = "t") -> RibbonSegment:
    """One RibbonSegment run for the ribbon/deck builders."""
    return RibbonSegment(color=color, width_m=width, points=points, tooltip=tooltip)


def test_create_terrain_layer():
    # 3D TerrainLayer with the fixed id; never pickable (it's the basemap).
    terrain = create_terrain_layer(mesh_max_error=1.0)
    assert isinstance(terrain, pdk.Layer)
    assert terrain.type == "TerrainLayer" and terrain.id == "terrain_3d"


def test_create_route_ribbon_layers():
    # ONE pickable PathLayer holding every run as a data row (uniform picking across the ribbon);
    # per-row path/color/width/tooltip carried through.
    segments = [
        _seg(_rgb(Palette.BLUE), 20.0, [[8.0, 48.0, 1100.0], [8.01, 48.0, 1100.0]], tooltip="paved · quiet way"),
        _seg(list(WebMapConfig.RAIL_COLOR), 8.0, [[8.01, 48.0, 1100.0], [8.02, 48.0, 1100.0]], tooltip="Train: A → B"),
    ]
    ribbons = create_route_ribbon_layers(segments=segments)
    assert len(ribbons) == 1
    layer = ribbons[0]
    assert layer.type == "PathLayer" and layer.pickable and layer.id == "route_ribbon"
    assert len(layer.data) == 2  # both runs are rows in the single layer
    assert layer.data[0]["tooltip"] == "paved · quiet way"
    assert layer.data[1]["tooltip"] == "Train: A → B"


def test_marker_layer():
    # A pickable ScatterplotLayer drawn ON TOP of terrain (depthTest off), with the given id/radius.
    rows = [_marker_row(lat=48.0, lon=8.0, elev=300.0, color=[1, 2, 3], tooltip="x")]
    layer = _marker_layer(layer_id="probe", markers=rows, radius_m=50.0, min_pixels=4)
    assert layer.type == "ScatterplotLayer" and layer.id == "probe" and layer.pickable
    assert layer.get_radius == 50.0
    assert layer.parameters == {"depthTest": False}


def test_marker_row():
    # One marker datum lifted RIBBON_FLOAT_ABOVE_M above the terrain — the single source of the shape.
    lift = WebMapConfig.RIBBON_FLOAT_ABOVE_M
    row = _marker_row(lat=48.0, lon=8.0, elev=300.0, color=[10, 20, 30], tooltip="hi")
    assert row == {"position": [8.0, 48.0, 300.0 + lift], "color": [10, 20, 30], "tooltip": "hi"}


def test_create_endpoint_layer():
    # Start + end markers, BOTH the one blue MARKER_COLOR, each [lon, lat, elev+lift], name+elev
    # tooltips, bigger than waypoints, drawn on top of terrain.
    layer = create_endpoint_layer(
        start=(48.0, 8.0, 300.0), end=(48.4, 8.6, 500.0), start_label="A (300 m)", end_label="B (500 m)"
    )
    assert layer.type == "ScatterplotLayer" and layer.id == "route_endpoints" and layer.pickable
    lift = WebMapConfig.RIBBON_FLOAT_ABOVE_M
    assert [row["position"] for row in layer.data] == [[8.0, 48.0, 300.0 + lift], [8.6, 48.4, 500.0 + lift]]
    blue = list(WebMapConfig.MARKER_COLOR)
    assert [row["color"] for row in layer.data] == [blue, blue]  # ONE colour for both endpoints
    assert [row["tooltip"] for row in layer.data] == ["A (300 m)", "B (500 m)"]
    assert layer.get_radius == WebMapConfig.ENDPOINT_RADIUS_M > WebMapConfig.WAYPOINT_RADIUS_M  # bigger than waypoints
    assert layer.parameters == {"depthTest": False}  # drawn on top of terrain, never buried


def test_create_waypoint_layer():
    # Intermediate markers (stations + gmaps waypoints): the SAME blue as the endpoints but SMALLER,
    # round, on top — told apart from endpoints by size, never colour (no more rail purple).
    layer = create_waypoint_layer(waypoints=[(48.5, 8.4, 700.0, "Freudenstadt Stadt (700 m)")])
    assert layer.type == "ScatterplotLayer" and layer.id == "route_waypoints" and layer.pickable
    assert layer.get_radius == WebMapConfig.WAYPOINT_RADIUS_M < WebMapConfig.ENDPOINT_RADIUS_M  # smaller
    assert layer.data[0]["color"] == list(WebMapConfig.MARKER_COLOR)  # same blue as endpoints
    assert layer.data[0]["tooltip"] == "Freudenstadt Stadt (700 m)"


def test_create_top_station_layer():
    # Rail-purple clickable markers; each datum carries a ``name`` (for click-to-fill) + hover tooltip.
    layer = create_top_station_layer(top_stations=[(48.47, 8.41, 739.0, "Freudenstadt Stadt")])
    assert layer.type == "ScatterplotLayer" and layer.id == "top_stations" and layer.pickable
    assert layer.data[0]["color"] == list(WebMapConfig.RAIL_COLOR)  # rail purple
    assert layer.data[0]["name"] == "Freudenstadt Stadt"  # picked datum → fills Start box
    assert layer.data[0]["tooltip"] == "Freudenstadt Stadt (739 m)"


def test_build_deck():
    # Assembles the Deck bottom→top: terrain only; +endpoints; then terrain/waypoints/endpoints/ribbon
    # in that draw order, carrying the camera pose from the ViewState.
    view = default_view_state()
    terrain_only = build_deck(
        view=view, ribbon_segments=None, endpoints=None, endpoint_labels=None, waypoints=None, top_stations=None
    )
    assert len(terrain_only.layers) == 1
    assert terrain_only.initial_view_state.latitude == WebMapConfig.DEFAULT_LAT

    with_endpoints = build_deck(
        view=view,
        ribbon_segments=None,
        endpoints=((48.0, 8.0, 300.0), (48.4, 8.6, 500.0)),
        endpoint_labels=("Start (300 m)", "End (500 m)"),
        waypoints=None,
        top_stations=None,
    )
    assert len(with_endpoints.layers) == 2

    two_run = [
        _seg(_rgb(Palette.BLUE), 20.0, [[8.0, 48.0, 1100.0], [8.01, 48.0, 1100.0]]),
        _seg(list(WebMapConfig.RAIL_COLOR), 8.0, [[8.01, 48.0, 1100.0], [8.02, 48.0, 1100.0]]),
    ]
    full = build_deck(
        view=view,
        ribbon_segments=two_run,
        endpoints=((48.0, 8.0, 300.0), (48.4, 8.6, 500.0)),
        endpoint_labels=("Start (300 m)", "End (500 m)"),
        waypoints=[(48.01, 8.01, 500.0, "S (500 m)")],
        top_stations=None,
    )
    assert [layer.id for layer in full.layers] == ["terrain_3d", "route_waypoints", "route_endpoints", "route_ribbon"]


def test_build_deck_with_top_stations():
    # Top-station markers layer is inserted right above the terrain when supplied.
    deck = build_deck(
        view=default_view_state(),
        ribbon_segments=None,
        endpoints=None,
        endpoint_labels=None,
        waypoints=None,
        top_stations=[(48.47, 8.41, 739.0, "Freudenstadt Stadt")],
    )
    assert [layer.id for layer in deck.layers] == ["terrain_3d", "top_stations"]
