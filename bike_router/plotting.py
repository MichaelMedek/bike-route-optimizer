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

from bike_router.constants import RoutingParams  # noqa: E402
from bike_router.track import Track  # noqa: E402

logger = logging.getLogger(__name__)


def plot_elevation_heatmap(
    graph: nx.MultiDiGraph,
    route_nodes: list[int],
    track: Track,
    params: RoutingParams,
    out_path: str,
    cmap_name: str = "plasma",
    dpi: int = 200,
) -> None:
    """Save a PNG heatmap of the graph (nodes by elevation) with the route on top.

    An overlay text box reports the rider's three preferences and the resulting
    route stats (distance, ride time, total ascent/descent).
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

    # Route overlay: thick high-contrast foreground line.
    route_lons = [graph.nodes[node]["x"] for node in route_nodes]
    route_lats = [graph.nodes[node]["y"] for node in route_nodes]
    axes.plot(route_lons, route_lats, color="#00d0ff", linewidth=4.0, alpha=0.95, zorder=5, label="route")

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    colorbar = figure.colorbar(mappable, ax=axes, fraction=0.03, pad=0.02)
    colorbar.set_label("Elevation (m)")
    axes.set_title("Bike route — nodes colored by elevation")

    overlay = (
        "Preferences (extra km):\n"
        f"  +{params.extra_km_per_uphill_100m:g} / 100 m uphill\n"
        f"  +{params.extra_km_per_unpaved_km:g} / km unpaved\n"
        f"  +{params.extra_km_per_main_road_km:g} / km main road\n"
        "Route:\n"
        f"  {track.distance_km:.1f} km · {track.duration_min:.0f} min\n"
        f"  +{track.ascent_m:.0f} m / -{track.descent_m:.0f} m"
    )
    axes.text(
        0.02,
        0.02,
        overlay,
        transform=axes.transAxes,
        va="bottom",
        ha="left",
        fontsize=7,
        family="monospace",
        color="black",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75, "edgecolor": "#888888"},
        zorder=6,
    )

    figure.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    logger.info("Wrote debug heatmap to %s", out_path)
