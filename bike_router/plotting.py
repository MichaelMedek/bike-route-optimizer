"""Debug visualization: graph nodes colored by elevation + the route overlaid.

Renders the corridor's bike graph with intersection nodes color-mapped by
elevation (plasma/viridis) and the computed route drawn as a thick, high-contrast
foreground line. Uses the Agg backend so it works headless.
"""

import logging
import math

import matplotlib

matplotlib.use("Agg")  # headless — must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import osmnx as ox  # noqa: E402
from matplotlib import cm  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from bike_router.composition import RouteComposition, format_composition  # noqa: E402
from bike_router.constants import Mode, PlotConfig, RoutingParams, WebMapConfig  # noqa: E402
from bike_router.cost import road_tier, surface_tier  # noqa: E402
from bike_router.track import Track, cheapest_edge, edge_vertices_3d  # noqa: E402
from bike_router.webmap import segment_color  # noqa: E402

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


def _draw_route_overlay(
    *, axes: Axes, graph: nx.MultiDiGraph, route_nodes: list[int], route_lons: list[float], route_lats: list[float]
) -> None:
    """Draw each route edge coloured by condition/mode along its baked polyline, then zoom."""
    seen_labels: set[str] = set()
    for node_a, node_b in zip(route_nodes[:-1], route_nodes[1:], strict=True):
        data = cheapest_edge(edges=graph.get_edge_data(node_a, node_b))
        mode = str(data["mode"])
        is_bad = surface_tier(surface=data.get("surface")) != 0 or road_tier(highway=data.get("highway")) != 0
        rgb = segment_color(mode=mode, is_bad=is_bad)
        color = (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
        # Legend lists each condition once: "train", or "bad"/"good" for pedalled legs.
        edge_label = "train" if mode == str(Mode.RAIL) else ("bad" if is_bad else "good")
        label = edge_label if edge_label not in seen_labels else None
        seen_labels.add(edge_label)
        verts = edge_vertices_3d(graph=graph, node_a=node_a, node_b=node_b, data=data)
        axes.plot(
            [lon for lon, _lat, _elev in verts],
            [lat for _lon, lat, _elev in verts],
            color=color,
            linewidth=3.0,
            alpha=0.95,
            zorder=5,
            label=label,
        )

    # Zoom to the route bounds. Pad BOTH axes by a fraction of the route's larger extent —
    # always positive for a valid route (>= 2 distinct nodes), so an axis-aligned route
    # (span 0 on one axis) still gets a real margin.
    pad = max(max(route_lons) - min(route_lons), max(route_lats) - min(route_lats)) * PlotConfig.ROUTE_ZOOM_MARGIN
    axes.set_xlim(min(route_lons) - pad, max(route_lons) + pad)
    axes.set_ylim(min(route_lats) - pad, max(route_lats) + pad)


def plot_elevation_heatmap(
    *,
    graph: nx.MultiDiGraph,
    route_nodes: list[int],
    track: Track,
    params: RoutingParams,
    out_path: str,
    origin: str = "Start",
    destination: str = "End",
    composition: RouteComposition | None = None,
    cmap_name: str = PlotConfig.CMAP,
    dpi: int = PlotConfig.DPI,
) -> None:
    """Save a PNG heatmap of the graph (nodes by elevation) with the route on top.

    Layout is one solid-white page: a large map (nodes colour-mapped by elevation, the
    route overlaid and its ends marked), an elevation colorbar down the right, and a
    condensed stats panel (preferences · route totals · km breakdown) across the bottom.
    """
    nodes = list(graph.nodes)
    assert nodes, "graph must have nodes to plot"
    assert len(route_nodes) >= 2, "route must have >= 2 nodes to overlay"
    # elevation is an invariant (enrich_elevations fills nodata) → strict access.
    elevations = np.array([graph.nodes[node]["elevation"] for node in nodes], dtype=float)
    assert np.all(np.isfinite(elevations)), "node elevations must be finite"

    elev_min, elev_max = float(elevations.min()), float(elevations.max())
    if elev_min == elev_max:
        elev_max = elev_min + 1.0
    norm = Normalize(vmin=elev_min, vmax=elev_max)
    cmap = matplotlib.colormaps[cmap_name]  # cm.get_cmap removed in matplotlib 3.9+
    node_colors = [cmap(norm(elevation)) for elevation in elevations]

    # Size the page to the route's geographic aspect BEFORE drawing, so the equal-aspect
    # map (OSMnx keeps it accurate) fills its axis and the colorbar hugs it — no dead gap.
    route_lons = [graph.nodes[node]["x"] for node in route_nodes]
    route_lats = [graph.nodes[node]["y"] for node in route_nodes]
    fig_w, fig_h, map_h = _figsize_for_route(route_lons=route_lons, route_lats=route_lats)

    # One page: map (large) + colorbar (thin, right of the map) + stats (full-width, bottom).
    # Row heights come straight from the computed inches (one source of truth, no drift).
    figure, mosaic = plt.subplot_mosaic(
        [["map", "cbar"], ["stats", "stats"]],
        gridspec_kw={
            "width_ratios": [1.0, 0.04],
            "height_ratios": [map_h, PlotConfig.STATS_HEIGHT_IN],
        },
        figsize=(fig_w, fig_h),
        layout="constrained",
    )
    figure.set_facecolor("white")  # solid white page, not the default transparent canvas
    axes, cbar_ax, stats_ax = mosaic["map"], mosaic["cbar"], mosaic["stats"]

    ox.plot_graph(
        graph,
        ax=axes,
        node_color=node_colors,
        node_size=6,
        edge_color="#cccccc",
        edge_linewidth=0.4,
        bgcolor="white",
        show=False,
        close=False,
        save=False,
    )

    # Route overlay: colour each edge by CONDITION (green good / red bad) or purple for
    # trains — the same segment_color the 3D map uses (one source of truth). Each edge
    # follows its BAKED OSM polyline, so rail/bike curves render as the real path.
    _draw_route_overlay(axes=axes, graph=graph, route_nodes=route_nodes, route_lons=route_lons, route_lats=route_lats)

    # Start / end markers, named so the reader knows which end is which. Colors
    # read from WebMapConfig so the PNG and 3D map share one source of truth.
    start_rgb = tuple(c / 255 for c in WebMapConfig.START_COLOR)
    end_rgb = tuple(c / 255 for c in WebMapConfig.END_COLOR)
    axes.scatter(
        route_lons[0],
        route_lats[0],
        s=150,
        c=[start_rgb],
        edgecolors="black",
        linewidths=1.2,
        zorder=7,
        marker="o",
        label="start",
    )
    axes.scatter(
        route_lons[-1],
        route_lats[-1],
        s=200,
        c=[end_rgb],
        edgecolors="black",
        linewidths=1.2,
        zorder=7,
        marker="*",
        label="end",
    )
    axes.set_title(f"Bike route: {origin} → {destination}", fontsize=13, weight="bold")

    # Elevation colorbar down the right, its own axis (crisp label + evenly-spaced ticks).
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    colorbar = figure.colorbar(mappable, cax=cbar_ax)
    colorbar.set_label("Elevation (m)", fontsize=11, weight="bold", labelpad=10)
    colorbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=8))

    # Bottom panel (own axis, no frame): a single combined stats box on the LEFT (prefs +
    # route + km composition, stacked so it fits any page width) and the route/marker legend
    # on the RIGHT — kept HERE, never over the map, so the map stays clear.
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
        handles,
        labels,
        loc="upper right",
        fontsize=9,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#999999",
    )

    figure.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.3)
    plt.close(figure)
    logger.info("Wrote debug heatmap to %s", out_path)
