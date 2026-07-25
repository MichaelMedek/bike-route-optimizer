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


def create_route_ribbon_layer(points: list[list[float]]) -> pdk.Layer:
    """PathLayer ribbon over `[[lon, lat, z], ...]` points (z already lifted)."""
    return pdk.Layer(
        "PathLayer",
        [{"path": points}],
        get_path="path",
        get_color=list(WebMapConfig.RIBBON_COLOR),
        get_width=WebMapConfig.RIBBON_WIDTH_M,
        width_min_pixels=WebMapConfig.RIBBON_MIN_PIXELS,
        cap_rounded=True,
        joint_rounded=True,
        id="route_ribbon",
        pickable=False,
    )


def build_deck(view: ViewState, ribbon_points: list[list[float]] | None) -> pdk.Deck:
    """Assemble the Deck: terrain always, ribbon on top when a route exists."""
    layers = [create_terrain_layer()]
    if ribbon_points is not None:
        layers.append(create_route_ribbon_layer(points=ribbon_points))
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(**asdict(view)),
        map_provider=None,  # TerrainLayer is the basemap; no Mapbox style needed
    )
