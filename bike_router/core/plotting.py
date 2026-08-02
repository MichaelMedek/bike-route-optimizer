"""Debug visualization: the route drawn on a plain map, coloured for quick verification.

DEBUG only — uses what inference already has (edge list + track), no ox. The line uses routing/condition
colours; only the special points (waypoints + stations + start/end) get elevation-coloured dots.
"""

import logging
import math

import matplotlib

matplotlib.use("Agg")  # headless — must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import cm  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from bike_router.core.composition import RouteComposition, format_composition  # noqa: E402
from bike_router.core.constants import ELEVATION_AXIS_LABEL, PLOT_BG, Palette, PlotConfig, RoutingParams  # noqa: E402
from bike_router.core.geo import nearest_index  # noqa: E402
from bike_router.core.route_path import RoutePath  # noqa: E402
from bike_router.core.track import (  # noqa: E402
    Track,
    classify_condition,
    edge_condition_speed,
    edge_display_unreliable,
    edge_vertices_3d,
    segment_color,
)

logger = logging.getLogger(__name__)


def _prefs_text(*, params: RoutingParams, track: Track) -> str:
    """Left stats column: rider preferences, then bike-only and total (bike+train) route stats."""
    return (
        "PREFERENCES (extra km)\n"
        f"  +{params.extra_km_per_uphill_100m:<4g} / 100 m uphill\n"
        f"  +{params.extra_km_per_unpaved_km:<4g} / km unpaved\n"
        f"  +{params.extra_km_per_main_road_km:<4g} / km main road\n"
        f"  +{params.extra_km_per_rail_km:<4g} / km rail\n"
        f"  +{params.extra_km_per_boarding:<4g} / boarding\n\n"
        f"ROUTE (bike + train)\n  {track.total.oneline}\n"
        f"ROUTE (bike only)\n  {track.bike.oneline}"
    )


def _figsize_for_route(*, route_lons: list[float], route_lats: list[float]) -> tuple[float, float, float]:
    """Figure (width, height) + map-axis height, matched to the route's geographic aspect.

    Fixes the map's long side and derives the short one from the lon/lat span so the
    equal-aspect map fills its axis for any orientation (no dead gap); short side floored.
    """
    lon_span = max(route_lons) - min(route_lons)
    lat_span = max(route_lats) - min(route_lats)
    mean_lat_rad = math.radians((max(route_lats) + min(route_lats)) / 2.0)
    aspect = (lon_span * math.cos(mean_lat_rad)) / lat_span if lat_span > 0 else 1.0
    long_in, short_min = PlotConfig.MAP_LONG_IN, PlotConfig.MAP_SHORT_MIN_IN
    if aspect >= 1.0:  # wide (E-W) route: width is the long side, height derived
        map_w, map_h = long_in, max(short_min, long_in / aspect)
    else:  # tall (N-S) route: height is the long side, width derived
        map_w, map_h = max(short_min, long_in * aspect), long_in
    return map_w + PlotConfig.SIDE_MARGIN_IN, map_h + PlotConfig.STATS_HEIGHT_IN, map_h


def _draw_route_overlay(*, axes: Axes, route: RoutePath) -> None:
    """Draw each route edge along its real polyline, coloured by condition (one legend entry each).

    A bike edge whose baked terrain strays far from the coarse node-to-node elevation the router used is
    drawn gray, matching the app's map warning, so the debug PNG flags the same questionable stretches.
    """
    seen_labels: set[str] = set()
    for node_a, node_b, edge in route.iter_edges():
        surface_bad, road_bad, _speed = edge_condition_speed(
            edge=edge, elev_source=node_a.elevation_m, elev_target=node_b.elevation_m
        )
        if edge_display_unreliable(node_a=node_a, node_b=node_b, edge=edge):
            rgb, edge_label = list(Palette.hex_to_rgb(hex_color=Palette.GRAY)), "unreliable elevation"
        else:
            rgb = segment_color(mode=edge.mode, surface_bad=surface_bad, road_bad=road_bad)
            edge_label = classify_condition(mode=edge.mode, surface_bad=surface_bad, road_bad=road_bad)
        color = (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
        label = edge_label if edge_label not in seen_labels else None
        seen_labels.add(edge_label)
        verts = edge_vertices_3d(node_a=node_a, node_b=node_b, edge=edge)
        axes.plot(
            [lon for lon, _lat, _elev in verts],
            [lat for _lon, lat, _elev in verts],
            color=color,
            linewidth=3.0,
            alpha=0.95,
            zorder=5,
            label=label,
        )


def _marker_node_indices(*, route: RoutePath, marker_points: list[tuple[float, float]]) -> list[int]:
    """Nearest route-node index for each (lat, lon) marker — so its dot takes that node's elevation.

    Always includes the start (0) and end (last) node; the given points are the interior
    waypoints + train stations. Nearest by great-circle distance (the shared haversine).
    """
    plats = np.array([n.lat for n in route.nodes], dtype=np.float64)
    plons = np.array([n.lon for n in route.nodes], dtype=np.float64)
    idxs = {0, len(route.nodes) - 1}
    for lat, lon in marker_points:
        idxs.add(nearest_index(lat=lat, lon=lon, lats=plats, lons=plons))
    return sorted(idxs)


def plot_route_debug(
    *,
    route: RoutePath,
    track: Track,
    params: RoutingParams,
    out_path: str,
    marker_points: list[tuple[float, float]] | None,
    origin: str,
    destination: str,
    composition: RouteComposition | None,
    cmap_name: str,
    dpi: int,
) -> None:
    """Save a debug PNG: the route line in ROUTING/condition colours (blue good / orange unpaved /
    red main / purple rail, same as the Streamlit ribbon); ONLY ``marker_points`` (waypoints + train
    board/alight) plus start/end get elevation-colormapped dots, over the whole route's elevation range.
    """
    lons = np.array([node.lon for node in route.nodes], dtype=float)
    lats = np.array([node.lat for node in route.nodes], dtype=float)
    elevations = np.array([node.elevation_m for node in route.nodes], dtype=float)
    assert np.all(np.isfinite(elevations)), "node elevations must be finite"

    elev_min, elev_max = float(elevations.min()), float(elevations.max())
    if elev_min == elev_max:
        elev_max = elev_min + 1.0
    norm = Normalize(vmin=elev_min, vmax=elev_max)
    cmap = matplotlib.colormaps[cmap_name]

    fig_w, fig_h, map_h = _figsize_for_route(route_lons=lons.tolist(), route_lats=lats.tolist())
    figure, mosaic = plt.subplot_mosaic(
        [["map", "cbar"], ["stats", "stats"]],
        gridspec_kw={"width_ratios": [1.0, 0.04], "height_ratios": [map_h, PlotConfig.STATS_HEIGHT_IN]},
        figsize=(fig_w, fig_h),
        layout="constrained",
    )
    figure.set_facecolor(PLOT_BG)
    axes, cbar_ax, stats_ax = mosaic["map"], mosaic["cbar"], mosaic["stats"]
    # adjustable="box" keeps our explicit x/y limits (set below) AND equal aspect by fitting the
    # axes box — "datalim" would instead override those limits (matplotlib warns + ignores them).
    axes.set_aspect("equal", adjustable="box")

    # The route line itself is drawn in routing/condition colours (same source as the 3D ribbon).
    _draw_route_overlay(axes=axes, route=route)

    # ONLY the special points get an elevation-coloured dot: the interior waypoints + the train
    # board/alight stations, projected to their nearest route node for the elevation value.
    pts = _marker_node_indices(route=route, marker_points=marker_points or [])
    if pts:
        axes.scatter(
            lons[pts],
            lats[pts],
            c=elevations[pts],
            cmap=cmap,
            norm=norm,
            s=60,
            zorder=6,
            edgecolors="black",
            linewidths=0.6,
        )

    pad = max(lons.max() - lons.min(), lats.max() - lats.min()) * PlotConfig.ROUTE_ZOOM_MARGIN or 0.001
    axes.set_xlim(lons.min() - pad, lons.max() + pad)
    axes.set_ylim(lats.min() - pad, lats.max() + pad)

    start_rgb = tuple(c / 255 for c in Palette.hex_to_rgb(hex_color=Palette.START))
    end_rgb = tuple(c / 255 for c in Palette.hex_to_rgb(hex_color=Palette.END))
    axes.scatter(lons[0], lats[0], s=150, c=[start_rgb], edgecolors="black", linewidths=1.2, zorder=7, label="start")
    axes.scatter(
        lons[-1], lats[-1], s=200, c=[end_rgb], edgecolors="black", linewidths=1.2, zorder=7, marker="*", label="end"
    )
    axes.set_title(f"Bike route: {origin} → {destination}", fontsize=13, weight="bold")

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    colorbar = figure.colorbar(mappable, cax=cbar_ax)
    colorbar.set_label(ELEVATION_AXIS_LABEL, fontsize=11, weight="bold", labelpad=10)
    colorbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=8))

    stats_ax.axis("off")
    stats = _prefs_text(params=params, track=track)
    if composition is not None:
        stats += "\n\n" + format_composition(comp=composition)
    stats_ax.text(
        0.0,
        1.0,
        stats,
        transform=stats_ax.transAxes,
        family="monospace",
        fontsize=9,
        va="top",
        ha="left",
        linespacing=1.5,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f5f5f5", "edgecolor": "#cccccc"},
    )
    handles, labels = axes.get_legend_handles_labels()
    stats_ax.legend(
        handles, labels, loc="upper right", fontsize=9, framealpha=0.95, facecolor=PLOT_BG, edgecolor="#999999"
    )

    figure.savefig(out_path, dpi=dpi, facecolor=PLOT_BG, bbox_inches="tight", pad_inches=0.3)
    plt.close(figure)
    logger.info(f"Wrote debug route PNG to {out_path}")
