"""Pydeck layer builders for the 3D bike-route map (app_webmap.py).

Radically simplified from ski-resort-designer's `ui/terrain_layer.py`: 3D only.
A `TerrainLayer` (AWS Terrarium elevation tiles + OpenTopoMap texture, same as the
ski map) plus an optional route ribbon `PathLayer`. No 2D style, no clicks.
"""

from dataclasses import asdict

import pydeck as pdk

from bike_router.constants import WebMapConfig
from bike_router.webmap import ViewState

# Free, no-API-key tiles — identical to the ski-resort 3D map.
AWS_TERRAIN_TILES = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
AWS_ELEVATION_DECODER = {"rScaler": 256, "gScaler": 1, "bScaler": 1 / 256, "offset": -32768}
OPENTOPOMAP_TILES = "https://a.tile.opentopomap.org/{z}/{x}/{y}.png"


def create_terrain_layer(mesh_max_error: float = 1.0) -> pdk.Layer:
    """3D TerrainLayer: AWS Terrarium elevation meshed, OpenTopoMap texture draped."""
    return pdk.Layer(
        "TerrainLayer",
        elevation_data=AWS_TERRAIN_TILES,
        elevation_decoder=AWS_ELEVATION_DECODER,
        texture=OPENTOPOMAP_TILES,
        mesh_max_error=mesh_max_error,
        id="terrain_3d",
        pickable=False,
    )


def create_route_ribbon_layers(segments: list[tuple[list[int], list[list[float]]]]) -> list[pdk.Layer]:
    """One PathLayer per contiguous same-mode run, each in its mode's color.

    Args:
        segments: ``(color, points)`` runs from webmap.route_ribbon_segments; color
            is RGB, points are ``[[lon, lat, z], ...]`` (z already lifted).
    """
    return [
        pdk.Layer(
            "PathLayer",
            [{"path": points, "color": color}],
            get_path="path",
            get_color="color",
            get_width=WebMapConfig.RIBBON_WIDTH_M,
            width_min_pixels=WebMapConfig.RIBBON_MIN_PIXELS,
            cap_rounded=True,
            joint_rounded=True,
            id=f"route_ribbon_{index}",
            pickable=False,
        )
        for index, (color, points) in enumerate(segments)
    ]


def create_endpoint_layer(start_latlon: tuple[float, float], end_latlon: tuple[float, float]) -> pdk.Layer:
    """ScatterplotLayer with the start (green) and end (red) endpoint markers.

    Colors match the debug PNG (WebMapConfig.START_COLOR / END_COLOR) so the PNG
    and the 3D map speak one visual language.

    Args:
        start_latlon: (lat, lon) of the start.
        end_latlon: (lat, lon) of the destination.
    """
    start_lat, start_lon = start_latlon
    end_lat, end_lon = end_latlon
    data = [
        {"position": [start_lon, start_lat], "color": list(WebMapConfig.START_COLOR)},
        {"position": [end_lon, end_lat], "color": list(WebMapConfig.END_COLOR)},
    ]
    return pdk.Layer(
        "ScatterplotLayer",
        data,
        get_position="position",
        get_fill_color="color",
        get_radius=WebMapConfig.MARKER_RADIUS_M,
        radius_min_pixels=WebMapConfig.MARKER_MIN_PIXELS,
        stroked=True,
        get_line_color=[0, 0, 0],
        line_width_min_pixels=1,
        id="route_endpoints",
        pickable=False,
    )


def build_deck(
    view: ViewState,
    ribbon_segments: list[tuple[list[int], list[list[float]]]] | None,
    endpoints: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> pdk.Deck:
    """Assemble the Deck: terrain always, endpoint markers + per-mode ribbon when set."""
    layers = [create_terrain_layer()]
    if endpoints is not None:
        layers.append(create_endpoint_layer(start_latlon=endpoints[0], end_latlon=endpoints[1]))
    if ribbon_segments is not None:
        layers.extend(create_route_ribbon_layers(segments=ribbon_segments))
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(**asdict(view)),
        map_provider=None,  # TerrainLayer is the basemap; no Mapbox style needed
    )
