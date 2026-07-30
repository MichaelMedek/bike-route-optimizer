"""Route composition — how many km fall on each surface tier, road class, and mode.

Walks the route's ordered edge list once and tallies km by surface tier, road class, and
travel mode for the CLI/web stats and plot legend.
"""

from dataclasses import dataclass

from bike_router.constants import GpxConfig, Mode, RoadConfig, SurfaceConfig, WebMapConfig
from bike_router.cost import road_tier, surface_tier
from bike_router.route_path import RoutePath


@dataclass(frozen=True)
class RouteComposition:
    """Kilometre breakdown of a route, three independent ways to slice the bike legs.

    ``by_mode`` covers the WHOLE route as two display buckets ("bike route" / "train path");
    station access-hops are negligible and fold into "bike route". The surface and road
    tallies describe the pedalled (bike) portion only.
    """

    by_surface_km: dict[str, float]  # bike km per surface label ("paved road" / "unpaved path")
    by_road_km: dict[str, float]  # bike km ("quiet way" / "main road")
    by_mode_km: dict[str, float]  # whole-route km ("bike route" / "train path")


def route_composition(route: RoutePath) -> RouteComposition:
    """Tally route km by surface tier, road class, and travel mode."""
    by_surface: dict[str, float] = {}
    by_road: dict[str, float] = {}
    by_mode: dict[str, float] = {}

    for edge in route.edges:
        km = edge.length_m / GpxConfig.METERS_PER_KM
        # Two display buckets only: a train ride is "train path"; everything pedalled OR a
        # negligible station access-hop counts as "bike route".
        mode_label = WebMapConfig.MODE_DONUT_LABELS[Mode.RAIL if edge.mode == Mode.RAIL else Mode.BIKE]
        by_mode[mode_label] = by_mode.get(mode_label, 0.0) + km
        if edge.mode != Mode.BIKE:  # surface/road only meaningful for pedalled legs
            continue
        label, _color = SurfaceConfig.TIER_LABEL_COLORS[surface_tier(surface=edge.surface)]
        by_surface[label] = by_surface.get(label, 0.0) + km
        road, _road_color = RoadConfig.TIER_LABEL_COLORS[road_tier(highway=edge.highway)]
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
    # Mode display order from the label map (single source; "bike route" then "train path").
    mode_order = {label: index for index, label in enumerate(WebMapConfig.MODE_DONUT_LABELS.values())}
    lines = ["Surface:"]
    lines += _percent_lines(by_km=comp.by_surface_km)
    lines.append("Roads:")
    lines += _percent_lines(by_km=comp.by_road_km)
    lines.append("Mode:")
    lines += _percent_lines(by_km=comp.by_mode_km, order=mode_order)
    return "\n".join(lines)
