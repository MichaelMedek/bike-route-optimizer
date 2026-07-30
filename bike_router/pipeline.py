"""Route-planning pipeline orchestration.

Loads the corridor subset of the prebuilt DACH bike+rail graph (elevations + baked 3D
geometry already carry all terrain) and computes ONE route per RoutingParams. A single
Track feeds the GPX, stats, and debug PNG so every number agrees; no DEM at inference.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
from shapely.geometry import Polygon

from bike_router.composition import RouteComposition, route_composition
from bike_router.constants import CorridorConfig, GmapsConfig, GpxConfig, GraphConfig, RoutingParams
from bike_router.corridor import build_corridor
from bike_router.cost import edge_cost_array
from bike_router.errors import NoRouteError, OutOfCoverageError, RouteTooLargeError, TripTooLongError, TripTooShortError
from bike_router.geo import haversine_distance_m
from bike_router.geocoding import geocode_endpoint, make_geocode_fn
from bike_router.gmaps import build_gmaps_url
from bike_router.gpx_export import build_gpx
from bike_router.graph_store import load_meta, load_path_edges, load_route_tables, snap_to_node
from bike_router.naming import route_output_paths
from bike_router.plotting import plot_route_debug
from bike_router.progress import log_rss
from bike_router.route_graph import RouteGraph, shortest_path
from bike_router.sanity import check_cost_model
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


def _geocode_both(*, origin: str, destination: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """Geocode origin + destination to (lat, lon) via ONE rate-limited fn (1 req/s spans both).

    Fail-fast: a bad Start raises before Destination is looked up.
    """
    geocode_fn = make_geocode_fn()
    start_ll = geocode_endpoint(place=origin, label="Start", geocode_fn=geocode_fn)
    dest_ll = geocode_endpoint(place=destination, label="Destination", geocode_fn=geocode_fn)
    return start_ll, dest_ll


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
    start_ll, end_ll = _geocode_both(origin=origin, destination=destination)
    return (
        snap_to_node(lat=start_ll[0], lon=start_ll[1], graph_dir=graph_dir),
        snap_to_node(lat=end_ll[0], lon=end_ll[1], graph_dir=graph_dir),
    )


def _route_node_path(
    *,
    bike_corridor: Polygon,
    rail_corridor: Polygon,
    graph_dir: Path,
    start_latlon: tuple[float, float],
    dest_latlon: tuple[float, float],
    params: RoutingParams,
) -> list[tuple[int, float, float]]:
    """Load the corridor, route on a compact CSR graph, return the path as (osmid, lat, lon).

    Reads ONLY the minimal routing columns (no geometry_wkt — 73% of the edge table) so the
    whole-corridor load stays memory-lean, then routes on a scipy CSR matrix (~12 bytes/edge vs
    ~2.8 KB/edge for networkx). Frees the corridor tables before returning; geometry for the tiny
    chosen path is re-read by the caller. Raises RouteTooLargeError past the memory cap.
    """
    nodes_df, edges_df = load_route_tables(
        bike_corridor=bike_corridor, rail_corridor=rail_corridor, graph_dir=graph_dir
    )
    log_rss(label=f"corridor tables loaded ({len(edges_df)} edges)")
    if len(edges_df) > CorridorConfig.MAX_ROUTE_EDGES:
        raise RouteTooLargeError(
            f"Route corridor is too large for this server ({len(edges_df):,} edges > "
            f"{CorridorConfig.MAX_ROUTE_EDGES:,}) — try a shorter trip."
        )
    assert not edges_df.empty, "corridor graph has no edges"

    elev_by_osmid = {int(o): float(e) for o, e in zip(nodes_df["osmid"], nodes_df["elevation_m"], strict=True)}
    from_osmid = edges_df["from_node"].to_numpy(dtype="int64")
    to_osmid = edges_df["to_node"].to_numpy(dtype="int64")
    cost = edge_cost_array(edges_df=edges_df, elev_by_osmid=elev_by_osmid, params=params)
    check_cost_model(
        from_osmid=from_osmid,
        to_osmid=to_osmid,
        mode=edges_df["mode"].to_numpy(),
        cost=cost,
        elev_by_osmid=elev_by_osmid,
        params=params,
    )
    route_graph = RouteGraph.from_arrays(
        osmids=nodes_df["osmid"].to_numpy(dtype="int64"),
        lat=nodes_df["lat"].to_numpy(dtype=float),
        lon=nodes_df["lon"].to_numpy(dtype=float),
        node_type=nodes_df["node_type"].to_numpy(),
        from_osmid=from_osmid,
        to_osmid=to_osmid,
        cost=cost,
    )
    log_rss(label=f"CSR route graph built ({route_graph.n_edges} edges)")
    source = route_graph.snap_bike_node(lat=start_latlon[0], lon=start_latlon[1])
    target = route_graph.snap_bike_node(lat=dest_latlon[0], lon=dest_latlon[1])
    try:
        node_path = shortest_path(route_graph=route_graph, source_osmid=source, target_osmid=target)
    except nx.NetworkXNoPath as exc:
        raise NoRouteError("No bike route found between the two places within the corridor.") from exc
    return [
        (int(n), float(route_graph.lat[route_graph.index[n]]), float(route_graph.lon[route_graph.index[n]]))
        for n in node_path
    ]


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
    start_latlon, dest_latlon = _geocode_both(origin=origin, destination=destination)
    logger.info(f"Geocoded: {origin}={start_latlon}, {destination}={dest_latlon}")

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
    bike_corridor = build_corridor(
        start_latlon=start_latlon,
        dest_latlon=dest_latlon,
        half_width_km=CorridorConfig.BIKE_HALF_WIDTH_KM,
        extend_km=CorridorConfig.BIKE_EXTEND_KM,
    )
    rail_corridor = build_corridor(
        start_latlon=start_latlon,
        dest_latlon=dest_latlon,
        half_width_km=CorridorConfig.RAIL_HALF_WIDTH_KM,
        extend_km=CorridorConfig.RAIL_EXTEND_KM,
    )

    # Load the corridor, route on a compact CSR graph, resolve the node path (memory-lean:
    # no geometry, no networkx). Returns the path as (osmid, lat, lon) for the geometry re-read.
    path_coords = _route_node_path(
        bike_corridor=bike_corridor,
        rail_corridor=rail_corridor,
        graph_dir=graph_dir,
        start_latlon=start_latlon,
        dest_latlon=dest_latlon,
        params=params,
    )
    logger.info(f"Route: {len(path_coords)} nodes")

    # The route is an ultra-small subset (hundreds of edges); re-read those edges WITH geometry
    # into an ordered RoutePath — no networkx. The big corridor tables were already freed inside
    # _route_node_path. The re-read picks the SAME cheapest parallel edge per hop (same params).
    route = load_path_edges(path_nodes=path_coords, params=params, graph_dir=graph_dir)
    log_rss(label="path edges loaded (corridor freed)")

    track = build_track(route=route)
    # Expand to the full real 2D polyline; elevation stays LINEAR node-to-node (edge_vertices_3d),
    # so the GPX, 3D ribbon, and elevation profile all read the SAME elevation the optimiser + stats use.
    track = densify_track(route=route, track=track)
    composition = route_composition(route=route)
    logger.info(
        f"total {track.total.distance_km:.1f} km / {track.total.duration_min:.0f} min, "
        f"bike {track.bike.distance_km:.1f} km, +{track.bike.ascent_m:.0f} m / -{track.bike.descent_m:.0f} m"
    )

    gpx_path, png_path = route_output_paths(origin=origin, destination=destination, params=params)
    gpx_path.parent.mkdir(parents=True, exist_ok=True)
    gpx_path.write_text(build_gpx(track=track))
    logger.info(f"Wrote {gpx_path} ({len(track.points)} trackpoints)")

    png_path.parent.mkdir(parents=True, exist_ok=True)
    plot_route_debug(
        route=route,
        track=track,
        params=params,
        out_path=str(png_path),
        origin=origin,
        destination=destination,
        composition=composition,
    )

    # Train rides first (boarding + alighting station per ride) — they both label the bike
    # legs and let the rider look the actual train up in a railway app. Empty for pure bike.
    rail_legs = split_rail_legs(route=route)

    # One Google Maps bicycling URL per pedalled leg: a train ride splits the route, so a
    # pure-bike trip yields one link and a one-train trip yields two. Each leg is labelled
    # by its real endpoints (origin/destination at the ends, station names where a train abuts).
    leg_paths = split_bike_legs(route=route)
    endpoints = bike_leg_endpoints(route=route, leg_paths=leg_paths, origin=origin, destination=destination)
    position = {osmid: index for index, osmid in enumerate(route.osmids)}
    bike_legs = [
        BikeLeg(
            url=build_gmaps_url(
                waypoints_latlon=select_waypoints(
                    line=route_to_linestring(
                        route=route.subpath(start_index=position[leg[0]], end_index=position[leg[-1]])
                    ),
                    count=GmapsConfig.N_WAYPOINTS,
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
