"""Route composition — how many km fall on each surface tier, road class, and mode.

Walks the chosen A* path once (same cheapest-edge walk as build_track) and tallies km
by surface tier, road class, and travel mode for the CLI/web stats and plot legend.
"""

from dataclasses import dataclass

import networkx as nx

from bike_router.constants import GpxConfig, Mode, RoadConfig, SurfaceConfig
from bike_router.cost import is_main_road, surface_tier
from bike_router.track import iter_route_edges


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
        label, _color = SurfaceConfig.TIER_LABEL_COLORS[surface_tier(surface=data.get("surface"))]
        by_surface[label] = by_surface.get(label, 0.0) + km
        road, _road_color = RoadConfig.LABEL_COLORS[is_main_road(highway=data.get("highway"))]
        by_road[road] = by_road.get(road, 0.0) + km

    return RouteComposition(by_surface_km=by_surface, by_road_km=by_road, by_mode_km=by_mode)


def _percent_lines(by_km: dict[str, float], order: dict[str, int] | None = None) -> list[str]:
    """Format a km breakdown as ``  label: NN%`` lines (percent of the category total).

    Args:
        by_km: label → km for one category.
        order: optional label → sort index; defaults to descending km.
    """
    total = sum(by_km.values())
    key = (lambda kv: order[kv[0]]) if order is not None else (lambda kv: -kv[1])
    return [f"  {label}: {km / total * 100:.0f}%" for label, km in sorted(by_km.items(), key=key)]


def format_composition(comp: RouteComposition) -> str:
    """Per-category percentage summary (used by the CLI and the plot overlay).

    Surface/road are percent of the pedalled distance; mode is percent of the whole
    route (bike vs train), always shown so the train ratio is never hidden.
    """
    # Mode display order from the Mode enum (single source; StrEnum keys are str).
    mode_order: dict[str, int] = {str(mode): index for index, mode in enumerate(Mode)}
    lines = ["Surface:"]
    lines += _percent_lines(by_km=comp.by_surface_km)
    lines.append("Roads:")
    lines += _percent_lines(by_km=comp.by_road_km)
    lines.append("Mode:")
    lines += _percent_lines(by_km=comp.by_mode_km, order=mode_order)
    return "\n".join(lines)
