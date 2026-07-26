"""Debug visualization: graph nodes colored by elevation + the route overlaid.

Renders the corridor's bike graph with intersection nodes color-mapped by
elevation (plasma/viridis) and the computed route drawn as a thick, high-contrast
foreground line. Uses the Agg backend so it works headless.
"""

import logging

import matplotlib

matplotlib.use("Agg")  # headless — must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import osmnx as ox  # noqa: E402
from matplotlib import cm  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from bike_router.composition import RouteComposition, format_composition  # noqa: E402
from bike_router.constants import Mode, PlotConfig, RoutingParams, WebMapConfig  # noqa: E402
from bike_router.cost import road_tier, surface_tier  # noqa: E402
from bike_router.track import Track, cheapest_edge  # noqa: E402
from bike_router.webmap import segment_color  # noqa: E402

logger = logging.getLogger(__name__)


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

    Start/end points are marked and named in the legend; a colorbar labels the
    elevation scale; an overlay box reports the rider's preferences, the route
    stats, and (when given) the surface/road/mode km breakdown.
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

    figure, axes = ox.plot_graph(
        graph,
        node_color=node_colors,
        node_size=8,
        edge_color="#444444",
        edge_linewidth=0.4,
        bgcolor="white",
        show=False,
        close=False,
        save=False,
    )

    # Route overlay: colour each edge by CONDITION (green good / red bad) or purple
    # for trains — the same segment_color the 3D map uses (one source of truth).
    route_lons = [graph.nodes[node]["x"] for node in route_nodes]
    route_lats = [graph.nodes[node]["y"] for node in route_nodes]
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
        axes.plot(
            [graph.nodes[node_a]["x"], graph.nodes[node_b]["x"]],
            [graph.nodes[node_a]["y"], graph.nodes[node_b]["y"]],
            color=color,
            linewidth=4.0,
            alpha=0.95,
            zorder=5,
            label=label,
        )

    # Zoom to the route bounds. Pad BOTH axes by a fraction of the route's larger
    # extent — always positive for a valid route (>= 2 distinct nodes, asserted above),
    # so an axis-aligned route (span 0 on one axis) still gets a real margin.
    pad = max(max(route_lons) - min(route_lons), max(route_lats) - min(route_lats)) * PlotConfig.ROUTE_ZOOM_MARGIN
    axes.set_xlim(min(route_lons) - pad, max(route_lons) + pad)
    axes.set_ylim(min(route_lats) - pad, max(route_lats) + pad)

    # Start / end markers, named so the reader knows which end is which. Colors
    # read from WebMapConfig so the PNG and 3D map share one source of truth.
    start_rgb = tuple(c / 255 for c in WebMapConfig.START_COLOR)
    end_rgb = tuple(c / 255 for c in WebMapConfig.END_COLOR)
    axes.scatter(
        route_lons[0],
        route_lats[0],
        s=140,
        c=[start_rgb],
        edgecolors="black",
        linewidths=1.2,
        zorder=7,
        marker="o",
        label=f"start: {origin}",
    )
    axes.scatter(
        route_lons[-1],
        route_lats[-1],
        s=180,
        c=[end_rgb],
        edgecolors="black",
        linewidths=1.2,
        zorder=7,
        marker="*",
        label=f"end: {destination}",
    )
    # ONE combined legend below the plot: the route/marker swatches, with the full
    # preferences + route stats + composition as the legend title (same single box,
    # never overlaying the map). bbox_inches="tight" expands the canvas to fit it.
    stats = (
        "Preferences (extra km):\n"
        f"  +{params.extra_km_per_uphill_100m:g} / 100 m uphill\n"
        f"  +{params.extra_km_per_unpaved_km:g} / km unpaved\n"
        f"  +{params.extra_km_per_main_road_km:g} / km main road\n"
        f"  +{params.extra_km_per_rail_km:g} / km rail\n"
        f"  +{params.extra_km_per_boarding:g} / boarding\n"
        "Route:\n"
        f"  {track.distance_km:.1f} km · {track.duration_min:.0f} min\n"
        f"  +{track.ascent_m:.0f} m / -{track.descent_m:.0f} m"
    )
    if composition is not None:
        stats += "\n" + format_composition(comp=composition)
    legend = axes.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=2,
        fontsize=7,
        framealpha=0.9,
        title=stats,
        title_fontproperties={"family": "monospace", "size": 7},
    )
    legend.get_title().set_horizontalalignment("left")

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    colorbar = figure.colorbar(mappable, ax=axes, fraction=0.03, pad=0.02)
    colorbar.set_label("Elevation (m)")
    axes.set_title(f"Bike route {origin} → {destination} — nodes colored by elevation")

    figure.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    logger.info("Wrote debug heatmap to %s", out_path)
