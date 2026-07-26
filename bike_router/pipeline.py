"""Route-planning pipeline orchestration.

Loads the corridor subset of the prebuilt DACH bike+rail graph (elevations + baked 3D
geometry already carry all terrain) and computes ONE route per RoutingParams. A single
Track feeds the GPX, stats, and debug PNG so every number agrees; no DEM at inference.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from bike_router.composition import RouteComposition, route_composition
from bike_router.constants import CorridorConfig, GmapsConfig, GpxConfig, GraphConfig, RoutingParams
from bike_router.corridor import build_corridor
from bike_router.cost import assign_edge_costs
from bike_router.errors import NoRouteError, OutOfCoverageError, TripTooLongError, TripTooShortError
from bike_router.geo import haversine_distance_m
from bike_router.geocoding import geocode_endpoint, make_geocode_fn
from bike_router.gmaps import build_gmaps_url
from bike_router.gpx_export import build_gpx
from bike_router.graph_ops import snap_endpoints
from bike_router.graph_store import load_corridor_graph, load_meta, snap_to_node
from bike_router.naming import route_output_paths
from bike_router.plotting import plot_elevation_heatmap
from bike_router.routing import shortest_route
from bike_router.sanity import (
    check_strongly_connected,
    check_uphill_costlier,
    find_steepest_bidirectional_edge,
)
from bike_router.simplify import (
    BikeLeg,
    RailLeg,
    bike_leg_endpoints,
    route_to_linestring,
    select_waypoints,
    split_bike_legs,
    split_rail_legs,
)
from bike_router.track import Track, build_track, densify_track

logger = logging.getLogger(__name__)


def _assert_within_coverage(
    start_latlon: tuple[float, float], dest_latlon: tuple[float, float], graph_dir: Path
) -> None:
    """Raise OutOfCoverageError if either endpoint falls outside the prebuilt graph's bbox."""
    meta = load_meta(graph_dir=graph_dir)
    west, south, east, north = meta["bbox"]
    for lat, lon in (start_latlon, dest_latlon):
        if not (west <= lon <= east and south <= lat <= north):
            raise OutOfCoverageError(
                "Route is outside the prebuilt graph coverage "
                f"(covered bbox W,S,E,N = {west:.2f},{south:.2f},{east:.2f},{north:.2f})."
            )


@dataclass(frozen=True)
class RouteResult:
    """The computed route: its track (stats + points), output paths, and Maps URL."""

    track: Track
    gpx_path: Path
    png_path: Path
    bike_legs: list[BikeLeg]  # one pedalled leg (Maps URL + from/to place names); trains split the route
    rail_legs: list[RailLeg]  # boarding + alighting station per train ride (empty = no train)
    composition: RouteComposition


def resolve_endpoints(
    *, origin: str, destination: str, graph_dir: Path = GraphConfig.GRAPH_DIR
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Geocode two place strings and snap each to the nearest graph node.

    Whatever text is passed is what's geocoded — the web app hands the box text here,
    so a picked suggestion (its label fills the box) and free-typed text share ONE path.
    Returns two (lat, lon, elevation_m) tuples; a bad place raises GeocodeError.

    Args:
        origin: Start place string.
        destination: Destination place string.
        graph_dir: Prebuilt-graph dir (tests override; defaults to the shipped artifact).
    """
    geocode_fn = make_geocode_fn()  # one rate-limited fn spans both calls (1 req/s)
    start_ll = geocode_endpoint(place=origin, label="Start", geocode_fn=geocode_fn)
    end_ll = geocode_endpoint(place=destination, label="End", geocode_fn=geocode_fn)
    return (
        snap_to_node(lat=start_ll[0], lon=start_ll[1], graph_dir=graph_dir),
        snap_to_node(lat=end_ll[0], lon=end_ll[1], graph_dir=graph_dir),
    )


def plan_route(
    *,
    origin: str,
    destination: str,
    params: RoutingParams,
    graph_dir: Path = GraphConfig.GRAPH_DIR,
) -> RouteResult:
    """Compute a single route between origin and destination for ``params``.

    Reads ONLY the prebuilt graph artifact — node + baked 3D edge geometry already
    carry elevation, so no DEM is loaded at inference (the DEM is a build-time input).

    Args:
        origin: Start place string (geocoded via Nominatim).
        destination: Destination place string.
        params: The rider's five "extra km" routing preferences (incl. rail sliders).
        graph_dir: Prebuilt-graph dir (tests override; defaults to the shipped artifact).
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
        raise TripTooShortError(
            f"Start and destination are only {trip_km:.1f} km apart "
            f"(minimum {CorridorConfig.MIN_TRIP_KM:.0f} km) — too short to plan."
        )
    if trip_km > CorridorConfig.MAX_TRIP_KM:
        raise TripTooLongError(
            f"Start and destination are {trip_km:.0f} km apart "
            f"(maximum {CorridorConfig.MAX_TRIP_KM:.0f} km) — too far to plan."
        )

    _assert_within_coverage(start_latlon=start_latlon, dest_latlon=dest_latlon, graph_dir=graph_dir)
    corridor = build_corridor(start_latlon=start_latlon, dest_latlon=dest_latlon)

    graph = load_corridor_graph(corridor=corridor, graph_dir=graph_dir)
    check_strongly_connected(graph=graph)

    source, target = snap_endpoints(graph=graph, start_latlon=start_latlon, dest_latlon=dest_latlon)
    assign_edge_costs(graph=graph, params=params)
    steepest = find_steepest_bidirectional_edge(graph=graph)
    if steepest is not None:  # None = no bidirectional edge (legitimate, not a bug)
        check_uphill_costlier(graph=graph, node_lower=steepest[0], node_upper=steepest[1], params=params)

    try:
        node_path = shortest_route(graph=graph, source=source, target=target)
    except nx.NetworkXNoPath as exc:
        raise NoRouteError("No bike route found between the two places within the corridor.") from exc
    logger.info("Route: %d nodes", len(node_path))

    track = build_track(graph=graph, node_path=node_path)
    # Expand to the full baked 3D road polyline (elevation already in the artifact).
    track = densify_track(graph=graph, node_path=node_path, track=track)
    composition = route_composition(graph=graph, node_path=node_path)
    logger.info(
        "total %.1f km / %.0f min, bike %.1f km, +%.0f m / -%.0f m",
        track.total.distance_km,
        track.total.duration_min,
        track.bike.distance_km,
        track.bike.ascent_m,
        track.bike.descent_m,
    )

    gpx_path, png_path = route_output_paths(origin=origin, destination=destination, params=params)
    gpx_path.parent.mkdir(parents=True, exist_ok=True)
    gpx_path.write_text(build_gpx(track=track))
    logger.info("Wrote %s (%d trackpoints)", gpx_path, len(track.points))

    png_path.parent.mkdir(parents=True, exist_ok=True)
    plot_elevation_heatmap(
        graph=graph,
        route_nodes=node_path,
        track=track,
        params=params,
        out_path=str(png_path),
        origin=origin,
        destination=destination,
        composition=composition,
    )

    # Train rides first (boarding + alighting station per ride) — they both label the bike
    # legs and let the rider look the actual train up in a railway app. Empty for pure bike.
    rail_legs = split_rail_legs(graph=graph, node_path=node_path)

    # One Google Maps bicycling URL per pedalled leg: a train ride splits the route, so a
    # pure-bike trip yields one link and a one-train trip yields two. Each leg is labelled
    # by its real endpoints (origin/destination at the ends, station names where a train abuts).
    leg_paths = split_bike_legs(graph=graph, node_path=node_path)
    endpoints = bike_leg_endpoints(
        graph=graph, node_path=node_path, leg_paths=leg_paths, origin=origin, destination=destination
    )
    bike_legs = [
        BikeLeg(
            url=build_gmaps_url(
                waypoints_latlon=select_waypoints(
                    line=route_to_linestring(graph=graph, node_path=leg), count=GmapsConfig.N_WAYPOINTS
                )
            ),
            from_place=from_place,
            to_place=to_place,
        )
        for leg, (from_place, to_place) in zip(leg_paths, endpoints, strict=True)
    ]
    assert gpx_path.exists() and png_path.exists(), "GPX and PNG must be written"

    return RouteResult(
        track=track,
        gpx_path=gpx_path,
        png_path=png_path,
        bike_legs=bike_legs,
        rail_legs=rail_legs,
        composition=composition,
    )
