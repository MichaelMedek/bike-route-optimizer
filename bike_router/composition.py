"""Route composition — how many km fall on each surface tier, road class, and mode.

Walks the chosen A* path once (same cheapest-edge walk as build_track) and tallies km
by surface tier, road class, and travel mode for the CLI/web stats and plot legend.
"""

from dataclasses import dataclass

import networkx as nx

from bike_router.constants import GpxConfig, Mode
from bike_router.cost import is_main_road, surface_tier
from bike_router.track import iter_route_edges

# Human labels for the three surface tiers (SurfaceConfig tier ints → text).
_SURFACE_LABELS = {0: "paved", 1: "gravel/unpaved", 2: "rough"}


@dataclass(frozen=True)
class RouteComposition:
    """Kilometre breakdown of a route, three independent ways to slice the bike legs.

    ``by_mode`` covers the WHOLE route (bike + rail + transfer km); the surface and
    road tallies describe the pedalled (bike) portion only.
    """

    by_surface_km: dict[str, float]  # bike km per surface label
    by_road_km: dict[str, float]  # bike km: "main road" vs "quiet way"
    by_mode_km: dict[str, float]  # all km per travel mode


def route_composition(graph: nx.MultiDiGraph, node_path: list[int]) -> RouteComposition:
    """Tally route km by surface tier, road class, and travel mode."""
    assert len(node_path) >= 2, "route must have >= 2 nodes"
    by_surface: dict[str, float] = {}
    by_road: dict[str, float] = {}
    by_mode: dict[str, float] = {}

    for _node_a, _node_b, data in iter_route_edges(graph=graph, node_path=node_path):
        km = float(data["length"]) / GpxConfig.METERS_PER_KM
        mode = data["mode"]
        by_mode[mode] = by_mode.get(mode, 0.0) + km
        if mode != Mode.BIKE:  # surface/road only meaningful for pedalled legs
            continue
        label = _SURFACE_LABELS[surface_tier(surface=data.get("surface"))]
        by_surface[label] = by_surface.get(label, 0.0) + km
        road = "main road" if is_main_road(highway=data.get("highway")) else "quiet way"
        by_road[road] = by_road.get(road, 0.0) + km

    return RouteComposition(by_surface_km=by_surface, by_road_km=by_road, by_mode_km=by_mode)


def format_composition(comp: RouteComposition) -> str:
    """One-line-per-category human summary (used by the CLI and the plot overlay)."""
    lines = ["Surface:"]
    lines += [f"  {label}: {km:.1f} km" for label, km in sorted(comp.by_surface_km.items(), key=lambda kv: -kv[1])]
    lines.append("Roads:")
    lines += [f"  {label}: {km:.1f} km" for label, km in sorted(comp.by_road_km.items(), key=lambda kv: -kv[1])]
    # Only show the mode split when rail/transfer is actually used (else it's all bike).
    if set(comp.by_mode_km) - {Mode.BIKE}:
        lines.append("Mode:")
        # Display order derived from the Mode enum (single source; StrEnum keys are str).
        order: dict[str, int] = {str(mode): index for index, mode in enumerate(Mode)}
        lines += [f"  {m}: {km:.1f} km" for m, km in sorted(comp.by_mode_km.items(), key=lambda kv: order[kv[0]])]
    return "\n".join(lines)
