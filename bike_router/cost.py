"""Asymmetric edge-cost computation for the directed bike graph.

Each directed edge carries three ADDITIVE cost components (all in "metre-equivalent"
units so they combine cleanly):

    dist    = length
    surface = length * (SurfaceFactor*RoadFactor - 1)   # extra cost of rough/arterial
    elev    = uphill elevation penalty

The router minimizes a weighted sum ``w_dist*dist + w_surface*surface + w_elev*elev``.
With weights (1,1,1) this collapses to the original ``length*SF*RF + EP`` model.
Different RouteProfiles boost one weight to lean toward that category.

Because the graph is directed, the uphill and downhill directions of one street
get different `elev` components — uphill costs more (flat-preferring).
"""

import networkx as nx

from bike_router.constants import CostConfig, RoadConfig, RouteProfile, SurfaceConfig


def _as_values(tag: object) -> list[str]:
    """Normalize an OSM tag (str | list | None) to a list of lowercased strings."""
    if tag is None:
        return []
    if isinstance(tag, (list, tuple, set)):
        return [str(v).lower() for v in tag]
    return [str(tag).lower()]


def surface_factor(surface: object, highway: object) -> float:
    """Surface-quality multiplier.

    Known surfaces map through SURFACE_FACTORS (worst element wins for lists).
    Unknown/missing surface falls back by road class: rougher (track/path) → 2.5,
    else → 1.1.
    """
    factors = [SurfaceConfig.SURFACE_FACTORS[s] for s in _as_values(tag=surface) if s in SurfaceConfig.SURFACE_FACTORS]
    if factors:
        return max(factors)  # worst (highest-penalty) known surface
    # Unknown surface → depend on highway class.
    hw = set(_as_values(tag=highway))
    if hw & SurfaceConfig.ROUGH_HIGHWAYS:
        return SurfaceConfig.UNKNOWN_TRACK_PATH
    return SurfaceConfig.UNKNOWN_OTHER


def road_factor(highway: object) -> float:
    """Road-type multiplier (cycleways rewarded, arterials deterred).

    Worst (highest) factor wins for list-valued highway tags, so a way tagged
    both `residential` and `secondary` is treated as the arterial.
    """
    factors = [RoadConfig.ROAD_FACTORS[h] for h in _as_values(tag=highway) if h in RoadConfig.ROAD_FACTORS]
    return max(factors) if factors else RoadConfig.DEFAULT


def elevation_penalty(elev_source: float, elev_target: float, length: float) -> float:
    """Uphill-only penalty that punishes steep segments super-linearly.

    dh <= 0 (flat/downhill) → 0. Otherwise dh * ELEV_COEFF * (1 + grade*GRADE_COEFF),
    where grade = dh / length. Elevations are finite here (enrich_elevations fills
    nodata), so no NaN guard.
    """
    dh = elev_target - elev_source
    if dh <= 0 or length <= 0:
        return 0.0
    grade = dh / length
    return dh * CostConfig.ELEV_COEFF * (1.0 + grade * CostConfig.GRADE_COEFF)


def edge_components(
    length: float, surface: object, highway: object, elev_source: float, elev_target: float
) -> tuple[float, float, float]:
    """The (dist, surface, elev) additive cost components of one directed edge."""
    sf_rf = surface_factor(surface=surface, highway=highway) * road_factor(highway=highway)
    dist = length
    surface_extra = length * (sf_rf - 1.0)
    elev = elevation_penalty(elev_source=elev_source, elev_target=elev_target, length=length)
    return dist, surface_extra, elev


def combine(components: tuple[float, float, float], profile: RouteProfile) -> float:
    """Weighted sum of (dist, surface, elev) components for a routing profile."""
    dist, surface_extra, elev = components
    return profile.w_dist * dist + profile.w_surface * surface_extra + profile.w_elev * elev


def edge_stored_components(data: dict[str, object]) -> tuple[float, float, float]:
    """Read the (dist, surface, elev) components stored on an edge by assign_edge_costs."""
    return (
        float(data[CostConfig.COMP_DIST]),  # type: ignore[arg-type]
        float(data[CostConfig.COMP_SURFACE]),  # type: ignore[arg-type]
        float(data[CostConfig.COMP_ELEV]),  # type: ignore[arg-type]
    )


def assign_edge_costs(graph: nx.MultiDiGraph) -> None:
    """Store the three cost components on every directed edge of the graph, in place.

    Requires node ``elevation`` (enrich_elevations) and edge ``length`` (OSMnx, in
    metres) — both internal invariants, accessed strictly so a gap fails loud.
    Per-profile weighted costs are derived on demand (see routing.weighted_cost_fn).
    """
    for node_a, node_b, _key, data in graph.edges(keys=True, data=True):
        dist, surface_extra, elev = edge_components(
            length=float(data["length"]),
            surface=data.get("surface"),  # OSM surface tag is genuinely optional
            highway=data.get("highway"),  # ditto highway (external OSM data)
            elev_source=float(graph.nodes[node_a]["elevation"]),
            elev_target=float(graph.nodes[node_b]["elevation"]),
        )
        data[CostConfig.COMP_DIST] = dist
        data[CostConfig.COMP_SURFACE] = surface_extra
        data[CostConfig.COMP_ELEV] = elev
