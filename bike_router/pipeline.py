"""Route-planning pipeline orchestration.

Builds the bike graph once, then computes FOUR routes over it — one per
RouteConfig profile (flattest / shortest / smoothest / balanced). Each profile
weights the three cost components (distance, surface, elevation) differently but
keeps all active. Every route is written to its own place+profile-stamped GPX and
debug PNG, and its distance/time/ascent/descent is logged.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from bike_router.constants import CorridorConfig, GmapsConfig, GpxConfig, RouteConfig, RouteProfile
from bike_router.corridor import build_corridor, corridor_within_dem
from bike_router.cost import assign_edge_costs
from bike_router.elevation import DEMService
from bike_router.geo import haversine_distance_m
from bike_router.geocoding import geocode, make_geocode_fn
from bike_router.gmaps import build_gmaps_url
from bike_router.gpx_export import build_gpx
from bike_router.graph import build_bike_graph, enrich_elevations, raw_node_count, snap_endpoints
from bike_router.naming import route_output_paths
from bike_router.plotting import plot_elevation_heatmap
from bike_router.route_stats import RouteStats, route_stats
from bike_router.routing import shortest_route
from bike_router.sanity import (
    check_simplify_shrunk,
    check_strongly_connected,
    check_uphill_costlier,
    find_steepest_bidirectional_edge,
)
from bike_router.simplify import route_to_linestring, select_waypoints, simplify_track

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouteResult:
    """One computed route variant: its profile, stats, outputs, and Maps URL."""

    profile: RouteProfile
    stats: RouteStats
    gpx_path: Path
    png_path: Path
    gmaps_url: str


def plan_routes(*, origin: str, destination: str, dem_path: Path) -> list[RouteResult]:
    """Compute all RouteConfig profiles between origin and destination.

    Args:
        origin: Start place string (geocoded via Nominatim).
        destination: Destination place string.
        dem_path: DEM GeoTIFF path (already ensured to exist).
    """
    geocode_fn = make_geocode_fn()  # one rate-limited fn spans both calls (1 req/s)
    start_latlon = geocode(place=origin, geocode_fn=geocode_fn)
    dest_latlon = geocode(place=destination, geocode_fn=geocode_fn)
    logger.info("Geocoded: %s=%s, %s=%s", origin, start_latlon, destination, dest_latlon)

    trip_km = (
        haversine_distance_m(lat_a=start_latlon[0], lon_a=start_latlon[1], lat_b=dest_latlon[0], lon_b=dest_latlon[1])
        / GpxConfig.METERS_PER_KM
    )
    if trip_km < CorridorConfig.MIN_TRIP_KM:
        raise SystemExit(
            f"Start and destination are only {trip_km:.1f} km apart "
            f"(minimum {CorridorConfig.MIN_TRIP_KM:.0f} km) — too short to plan."
        )

    terrain = DEMService(dem_path=dem_path)
    corridor = build_corridor(start_latlon=start_latlon, dest_latlon=dest_latlon)
    if not corridor_within_dem(polygon=corridor, dem_bounds=terrain.bounds):
        logger.warning("Corridor %s not fully within DEM %s — may hit nodata.", corridor.bounds, terrain.bounds)

    raw_count = raw_node_count(polygon=corridor)
    graph = build_bike_graph(polygon=corridor)
    check_strongly_connected(graph=graph)

    source, target = snap_endpoints(graph=graph, start_latlon=start_latlon, dest_latlon=dest_latlon)
    enrich_elevations(graph=graph, dem=terrain)

    assign_edge_costs(graph=graph)
    check_simplify_shrunk(nodes_before=raw_count, nodes_after=graph.number_of_nodes())
    steepest = find_steepest_bidirectional_edge(graph=graph)
    if steepest is not None:  # None = no bidirectional edge (legitimate, not a bug)
        # Sanity 2 must hold for EVERY user-facing profile's weighting.
        for profile in RouteConfig.PROFILES:
            check_uphill_costlier(graph=graph, node_lower=steepest[0], node_upper=steepest[1], profile=profile)

    results = []
    for profile in RouteConfig.PROFILES:
        results.append(
            _plan_single(
                graph=graph,
                terrain=terrain,
                origin=origin,
                destination=destination,
                source=source,
                target=target,
                profile=profile,
            )
        )
    return results


def _plan_single(
    *,
    graph: nx.MultiDiGraph,
    terrain: DEMService,
    origin: str,
    destination: str,
    source: int,
    target: int,
    profile: RouteProfile,
) -> RouteResult:
    """Route + write GPX/PNG + compute stats for one profile."""
    try:
        node_path = shortest_route(graph=graph, source=source, target=target, profile=profile)
    except nx.NetworkXNoPath as exc:
        raise SystemExit("No bike route found between the two places within the corridor.") from exc

    stats = route_stats(graph=graph, node_path=node_path, profile=profile)
    logger.info(
        "[%s] %.1f km, %.0f min, +%.0f m / -%.0f m",
        profile.name,
        stats.distance_km,
        stats.duration_min,
        stats.ascent_m,
        stats.descent_m,
    )

    geometry = route_to_linestring(graph=graph, node_path=node_path, profile=profile)
    waypoints = select_waypoints(line=geometry, count=GmapsConfig.N_WAYPOINTS)
    gpx_path, png_path = route_output_paths(origin=origin, destination=destination, profile=profile)
    _write_gpx(terrain=terrain, geometry=geometry, gpx_path=gpx_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plot_elevation_heatmap(graph=graph, route_nodes=node_path, out_path=str(png_path))

    return RouteResult(
        profile=profile,
        stats=stats,
        gpx_path=gpx_path,
        png_path=png_path,
        gmaps_url=build_gmaps_url(waypoints_latlon=waypoints),
    )


def _write_gpx(terrain: DEMService, geometry: object, gpx_path: Path) -> None:
    """Douglas-Peucker-simplify the full route track, then write GPX."""
    track_lonlat = simplify_track(line=geometry)  # type: ignore[arg-type]
    full_latlon = [(lat, lon) for lon, lat in track_lonlat]
    elevations = terrain.get_elevations(
        lons=[lon for _lat, lon in full_latlon], lats=[lat for lat, _lon in full_latlon]
    ).tolist()
    gpx_xml = build_gpx(coords_latlon=full_latlon, elevations_m=elevations)
    gpx_path.parent.mkdir(parents=True, exist_ok=True)
    gpx_path.write_text(gpx_xml)
    logger.info("Wrote %s (%d trackpoints)", gpx_path, len(full_latlon))
