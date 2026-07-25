"""Route-planning pipeline orchestration.

Builds (or loads the cached) bike graph and computes ONE route for the rider's
RoutingParams (the three "extra km" preferences). A single Track (surface/
grade-adaptive timing) is the source for the GPX, the printed stats, and the debug
PNG — so every number agrees. The route is written to a place-stamped GPX + PNG.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from bike_router.constants import CorridorConfig, GmapsConfig, GpxConfig, RoutingParams
from bike_router.corridor import build_corridor, corridor_within_dem
from bike_router.cost import assign_edge_costs
from bike_router.elevation import DEMService
from bike_router.geo import haversine_distance_m
from bike_router.geocoding import geocode_endpoint, make_geocode_fn
from bike_router.gmaps import build_gmaps_url
from bike_router.gpx_export import build_gpx
from bike_router.graph import build_bike_graph, enrich_elevations, snap_endpoints
from bike_router.naming import route_output_paths
from bike_router.plotting import plot_elevation_heatmap
from bike_router.progress import ProgressFn, null_progress
from bike_router.routing import shortest_route
from bike_router.sanity import (
    check_simplify_shrunk,
    check_strongly_connected,
    check_uphill_costlier,
    find_steepest_bidirectional_edge,
)
from bike_router.simplify import route_to_linestring, select_waypoints
from bike_router.track import Track, build_track

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouteResult:
    """The computed route: its track (stats + points), output paths, and Maps URL."""

    track: Track
    gpx_path: Path
    png_path: Path
    gmaps_url: str


def plan_route(
    *, origin: str, destination: str, dem_path: Path, params: RoutingParams, progress: ProgressFn = null_progress
) -> RouteResult:
    """Compute a single route between origin and destination for ``params``.

    Args:
        origin: Start place string (geocoded via Nominatim).
        destination: Destination place string.
        dem_path: DEM GeoTIFF path (already ensured to exist).
        params: The rider's three "extra km" routing preferences.
        progress: Real per-item sink (nodes done, total) for the graph-build loop;
            the CLI wires tqdm, Streamlit st.progress. Defaults to no-op.
    """
    geocode_fn = make_geocode_fn()  # one rate-limited fn spans both calls (1 req/s)
    # Fail-fast: a bad Start raises before Destination is ever looked up.
    start_latlon = geocode_endpoint(place=origin, label="Start", geocode_fn=geocode_fn)
    dest_latlon = geocode_endpoint(place=destination, label="Destination", geocode_fn=geocode_fn)
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
    if trip_km > CorridorConfig.MAX_TRIP_KM:
        raise SystemExit(
            f"Start and destination are {trip_km:.0f} km apart "
            f"(maximum {CorridorConfig.MAX_TRIP_KM:.0f} km) — too far to plan."
        )

    terrain = DEMService(dem_path=dem_path)
    corridor = build_corridor(start_latlon=start_latlon, dest_latlon=dest_latlon)
    if not corridor_within_dem(polygon=corridor, dem_bounds=terrain.bounds):
        logger.warning("Corridor %s not fully within DEM %s — may hit nodata.", corridor.bounds, terrain.bounds)

    graph, raw_count = build_bike_graph(polygon=corridor, progress=progress)
    check_strongly_connected(graph=graph)

    source, target = snap_endpoints(graph=graph, start_latlon=start_latlon, dest_latlon=dest_latlon)
    enrich_elevations(graph=graph, dem=terrain)

    assign_edge_costs(graph=graph, params=params)
    check_simplify_shrunk(nodes_before=raw_count, nodes_after=graph.number_of_nodes())
    steepest = find_steepest_bidirectional_edge(graph=graph)
    if steepest is not None:  # None = no bidirectional edge (legitimate, not a bug)
        check_uphill_costlier(graph=graph, node_lower=steepest[0], node_upper=steepest[1], params=params)

    try:
        node_path = shortest_route(graph=graph, source=source, target=target)
    except nx.NetworkXNoPath as exc:
        raise SystemExit("No bike route found between the two places within the corridor.") from exc
    logger.info("Route: %d nodes", len(node_path))

    track = build_track(graph=graph, node_path=node_path)
    logger.info(
        "%.1f km, %.0f min, +%.0f m / -%.0f m",
        track.distance_km,
        track.duration_min,
        track.ascent_m,
        track.descent_m,
    )

    gpx_path, png_path = route_output_paths(origin=origin, destination=destination)
    gpx_path.parent.mkdir(parents=True, exist_ok=True)
    gpx_path.write_text(build_gpx(track=track))
    logger.info("Wrote %s (%d trackpoints)", gpx_path, len(track.points))

    png_path.parent.mkdir(parents=True, exist_ok=True)
    plot_elevation_heatmap(graph=graph, route_nodes=node_path, track=track, params=params, out_path=str(png_path))

    # Google Maps waypoints: the N most significant points of the full geometry.
    geometry = route_to_linestring(graph=graph, node_path=node_path)
    waypoints = select_waypoints(line=geometry, count=GmapsConfig.N_WAYPOINTS)
    assert len(waypoints) == GmapsConfig.N_WAYPOINTS, "must produce exactly N waypoints"
    assert gpx_path.exists() and png_path.exists(), "GPX and PNG must be written"

    return RouteResult(
        track=track,
        gpx_path=gpx_path,
        png_path=png_path,
        gmaps_url=build_gmaps_url(waypoints_latlon=waypoints),
    )
