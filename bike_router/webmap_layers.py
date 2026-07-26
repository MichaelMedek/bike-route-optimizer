"""Pydeck layer builders for the 3D bike-route map (app_webmap.py).

Radically simplified from ski-resort-designer's `ui/terrain_layer.py`: 3D only.
A `TerrainLayer` (AWS Terrarium elevation tiles + OpenTopoMap texture, same as the
ski map) plus an optional route ribbon `PathLayer`. No 2D style, no clicks.
"""

from dataclasses import asdict

import pydeck as pdk

from bike_router.constants import WebMapConfig
from bike_router.webmap import ViewState


def create_terrain_layer(mesh_max_error: float = 1.0) -> pdk.Layer:
    """3D TerrainLayer: AWS Terrarium elevation meshed, OpenTopoMap texture draped."""
    return pdk.Layer(
        "TerrainLayer",
        elevation_data=WebMapConfig.TERRAIN_TILES_URL,
        elevation_decoder=WebMapConfig.TERRAIN_ELEVATION_DECODER,
        texture=WebMapConfig.TEXTURE_TILES_URL,
        mesh_max_error=mesh_max_error,
        id="terrain_3d",
        pickable=False,
    )


def create_route_ribbon_layers(segments: list[tuple[list[int], float, list[list[float]]]]) -> list[pdk.Layer]:
    """One PathLayer per contiguous run, in its condition colour and speed-scaled width.

    Args:
        segments: ``(color, width_m, points)`` runs from webmap.route_ribbon_segments;
            color is RGB, width_m scales with segment speed, points are
            ``[[lon, lat, z], ...]`` (z already lifted).
    """
    return [
        pdk.Layer(
            "PathLayer",
            [{"path": points, "color": color}],
            get_path="path",
            get_color="color",
            get_width=width,
            width_min_pixels=WebMapConfig.RIBBON_MIN_PIXELS,
            cap_rounded=True,
            joint_rounded=True,
            id=f"route_ribbon_{index}",
            pickable=False,
        )
        for index, (color, width, points) in enumerate(segments)
    ]


def create_endpoint_layer(start: tuple[float, float, float], end: tuple[float, float, float]) -> pdk.Layer:
    """ScatterplotLayer with the start (green) and end (red) endpoint markers.

    Each endpoint is ``(lat, lon, elevation_m)`` (snapped to its graph node); the
    marker hovers RIBBON_FLOAT_ABOVE_M above that elevation, matching the ribbon.
    Colors match the debug PNG so the PNG and 3D map speak one visual language.
    """
    lift = WebMapConfig.RIBBON_FLOAT_ABOVE_M
    data = [
        {"position": [start[1], start[0], start[2] + lift], "color": list(WebMapConfig.START_COLOR)},
        {"position": [end[1], end[0], end[2] + lift], "color": list(WebMapConfig.END_COLOR)},
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
        billboard=True,  # face the camera in the 3D tilted view
        parameters={"depthTest": False},  # draw ON TOP of the terrain mesh, never buried
        id="route_endpoints",
        pickable=False,
    )


def build_deck(
    view: ViewState,
    ribbon_segments: list[tuple[list[int], float, list[list[float]]]] | None,
    endpoints: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
) -> pdk.Deck:
    """Assemble the Deck: terrain always, endpoint markers + per-mode ribbon when set."""
    layers = [create_terrain_layer()]
    if endpoints is not None:
        layers.append(create_endpoint_layer(start=endpoints[0], end=endpoints[1]))
    if ribbon_segments is not None:
        layers.extend(create_route_ribbon_layers(segments=ribbon_segments))
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(**asdict(view)),
        map_provider=None,  # TerrainLayer is the basemap; no Mapbox style needed
    )
