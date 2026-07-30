"""Route geometry helpers.

``route_to_linestring`` stitches the node path into the full OSM geometry;
``select_waypoints`` reduces it to N significant points (Visvalingam-Whyatt) for
the Google Maps URL.
"""

import logging
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString

from bike_router.core.constants import GmapsConfig, GpxConfig, Mode, NodeType
from bike_router.core.geo import haversine_distance_m
from bike_router.core.route_path import RouteNode, RoutePath

logger = logging.getLogger(__name__)

_UNNAMED_STOP = "(unnamed stop)"  # placeholder when OSM gives a rail node no station_name


def place_label(*, name: str, elevation_m: float) -> str:
    """``Name (739 m)`` — the ONE name+elevation label (start/end + station markers/tooltips)."""
    return f"{name} ({elevation_m:.0f} m)"


def _split_mode_runs(route: RoutePath, mode: str) -> list[list[RouteNode]]:
    """Maximal runs of consecutive nodes whose connecting edges are ALL ``mode``.

    Each run is the node list ``[n0, n1, …]`` (>= 2 nodes) for one contiguous stretch;
    edges of other modes break the route. Shared by split_bike_legs and split_rail_legs.
    """
    runs: list[list[RouteNode]] = []
    current: list[RouteNode] = []
    for node_a, node_b, edge in route.iter_edges():
        if edge.mode == mode:
            if not current:
                current = [node_a]
            current.append(node_b)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def split_bike_legs(route: RoutePath) -> list[list[int]]:
    """Split a route into maximal runs of consecutive BIKE edges (rail/station cut).

    Each returned sub-path is one pedalled leg the rider actually cycles (as osmids); train
    rides and station-access hops break the route so a pure-bike trip yields one leg and a trip
    with one train ride yields two. Sub-paths have >= 2 nodes (one usable linestring).
    """
    return [[node.osmid for node in run] for run in _split_mode_runs(route=route, mode=Mode.BIKE)]


@dataclass(frozen=True)
class Station:
    """One boarded/alighted rail stop: its name (may be None for an unnamed halt) + position."""

    name: str | None
    lat: float
    lon: float
    elevation_m: float

    @property
    def name_or_placeholder(self) -> str:
        """The station name, or a readable placeholder for an unnamed OSM stop."""
        return self.name or _UNNAMED_STOP

    @property
    def label(self) -> str:
        """``Name (739 m)`` for map markers/tooltips — the shared name+elevation format."""
        return place_label(name=self.name_or_placeholder, elevation_m=self.elevation_m)


@dataclass(frozen=True)
class RailLeg:
    """One train ride the rider takes: the station boarded at and the one alighted at.

    Each ``Station`` carries name + position (from the rail node), so the CLI/web lists, the
    map's station markers, and the ribbon's train-leg tooltip all read from this one structure.
    """

    board: Station
    alight: Station


def split_rail_legs(route: RoutePath) -> list[RailLeg]:
    """Boarding + alighting station (name + position) for each train ride on the route.

    A ride is a maximal run of consecutive RAIL edges (a junction change stays one ride):
    board the first rail node, alight the last. Separate rides yield separate RailLegs.
    """
    return [
        RailLeg(board=_station(node=run[0]), alight=_station(node=run[-1]))
        for run in _split_mode_runs(route=route, mode=Mode.RAIL)
    ]


def _station(node: RouteNode) -> Station:
    """Build a Station from a rail node's baked attributes."""
    return Station(name=node.station_name, lat=node.lat, lon=node.lon, elevation_m=node.elevation_m)


def route_station_markers(rail_legs: list[RailLeg]) -> list[tuple[float, float, float, str]]:
    """(lat, lon, elevation_m, label) for every boarded/alighted station across all train rides.

    Feeds the map's station markers; the label is the shared "Name (elev m)" text.
    """
    markers: list[tuple[float, float, float, str]] = []
    for leg in rail_legs:
        for stop in (leg.board, leg.alight):
            markers.append((stop.lat, stop.lon, stop.elevation_m, stop.label))
    return markers


def rail_leg_tooltips(rail_legs: list[RailLeg]) -> list[str]:
    """One "Train: A (e) → B (e)" hover label per train ride, in order (board → alight)."""
    return [f"Train: {leg.board.label} → {leg.alight.label}" for leg in rail_legs]


def format_rail_legs(rail_legs: list[RailLeg]) -> list[str]:
    """One "Train N: board → alight" line per ride (shared by CLI + web output).

    An unnamed stop (no OSM station name) renders via the shared placeholder so the line
    is still readable rather than crashing on a None the external data legitimately holds.
    """
    return [
        f"Train {index}: {leg.board.name_or_placeholder} → {leg.alight.name_or_placeholder}"
        for index, leg in enumerate(rail_legs, start=1)
    ]


@dataclass(frozen=True)
class BikeLeg:
    """One pedalled leg's Google Maps URL plus the place names of its two ends.

    ``from_place``/``to_place`` are the human endpoints of THIS pedalled leg: the trip
    origin/destination at the outer ends, and the boarding/alighting station names where a
    train ride abuts the leg — so the link can be labelled "Route N: from → to".
    """

    url: str
    from_place: str
    to_place: str


def bike_leg_endpoints(
    *, route: RoutePath, leg_paths: list[list[int]], origin: str, destination: str
) -> list[tuple[str, str]]:
    """(from_place, to_place) for each pedalled leg, derived STRUCTURALLY from the node path.

    A leg's endpoint is ``origin``/``destination`` at the very ends of the whole route, else the
    adjacent station it alighted-from / boards-at (the neighbouring node on the full path, which
    is a rail-station node). Robust to a route that starts/ends on a train or chains two trains.
    """
    nodes = route.nodes
    position = {node.osmid: index for index, node in enumerate(nodes)}  # osmid → its index on the path
    ends: list[tuple[str, str]] = []
    for leg in leg_paths:
        head, tail = position[leg[0]], position[leg[-1]]
        from_place = origin if head == 0 else _neighbour_station_name(node=nodes[head - 1])
        to_place = destination if tail == len(nodes) - 1 else _neighbour_station_name(node=nodes[tail + 1])
        ends.append((from_place, to_place))
    return ends


def _neighbour_station_name(node: RouteNode) -> str:
    """Station name of a path-neighbour that MUST be a rail node (unnamed halt → placeholder).

    A pedalled leg only ends before the route start/end when the adjacent node is the station
    it boards/alights at — so the neighbour is invariantly a rail node. We assert that (fail
    loud on a broken path) rather than let a stray bike node silently read as an unnamed stop.
    """
    assert node.node_type == NodeType.RAIL, (
        f"bike-leg neighbour {node.osmid} must be a rail station, got {node.node_type}"
    )
    return node.station_name or _UNNAMED_STOP


def format_bike_legs(bike_legs: list[BikeLeg]) -> list[str]:
    """One "Bike Route N: from → to" label per pedalled leg (shared by CLI + web)."""
    return [f"Bike Route {index}: {leg.from_place} → {leg.to_place}" for index, leg in enumerate(bike_legs, start=1)]


def route_to_linestring(route: RoutePath) -> LineString:
    """Stitch the route's edges into the full lon/lat OSM geometry (x=lon, y=lat).

    Each edge uses its real oriented ``geometry`` if present, else a straight segment
    between node coordinates. Shared vertices between consecutive edges are de-duplicated.
    """
    coords: list[tuple[float, float]] = []
    for node_a, node_b, edge in route.iter_edges():
        if edge.geometry is not None:
            segment = list(edge.geometry)  # already oriented a→b, 2D lon/lat
        else:
            segment = [(node_a.lon, node_a.lat), (node_b.lon, node_b.lat)]
        if coords and segment and coords[-1] == segment[0]:
            segment = segment[1:]  # avoid duplicating the shared vertex
        coords.extend(segment)
    return LineString(coords)


def select_waypoints(line: LineString, count: int = 10) -> list[tuple[float, float]]:
    """Reduce ``line`` to at most ``count`` significant points, returned (lat, lon).

    Visvalingam-Whyatt picks the ``count`` most significant points (endpoints kept), then
    over-close interior points are thinned so a short leg isn't cluttered with near-identical
    waypoints. Origin + destination are always kept, so the result has 2..count points.
    """
    coords: list[tuple[float, float]] = [(c[0], c[1]) for c in line.coords]  # (lon, lat); ignore any z
    assert count >= 2, "need at least origin + destination waypoints"
    if len(coords) <= count:
        coords = _interpolate_to_n(coords=coords, count=count)
    else:
        coords = _visvalingam(coords=coords, count=count)
    latlon = [(lat, lon) for lon, lat in coords]
    return _thin_close_points(points=latlon, min_spacing_km=GmapsConfig.MIN_WAYPOINT_SPACING_KM)


def _thin_close_points(points: list[tuple[float, float]], min_spacing_km: float) -> list[tuple[float, float]]:
    """Drop interior (lat, lon) points closer than ``min_spacing_km`` to the last kept one.

    Origin (first) and destination (last) are ALWAYS kept; only in-between points are thinned,
    so a short leg collapses toward just its two endpoints. Distance is real great-circle km.
    """
    assert len(points) >= 2, "need origin + destination to thin between"
    min_m = min_spacing_km * GpxConfig.METERS_PER_KM
    kept = [points[0]]
    for point in points[1:-1]:
        last_lat, last_lon = kept[-1]
        if haversine_distance_m(lat_a=last_lat, lon_a=last_lon, lat_b=point[0], lon_b=point[1]) >= min_m:
            kept.append(point)
    kept.append(points[-1])
    return kept


def _triangle_area(point_a: tuple[float, float], point_b: tuple[float, float], point_c: tuple[float, float]) -> float:
    """Absolute area of the triangle a-b-c (planar; degrees are fine for ranking)."""
    return (
        abs(
            (point_b[0] - point_a[0]) * (point_c[1] - point_a[1])
            - (point_c[0] - point_a[0]) * (point_b[1] - point_a[1])
        )
        / 2.0
    )


def _visvalingam(coords: list[tuple[float, float]], count: int) -> list[tuple[float, float]]:
    """Drop the smallest-effective-area interior point until ``count`` remain."""
    assert len(coords) >= count, "cannot reduce below the requested count"
    survivors = list(coords)
    while len(survivors) > count:
        smallest_index = 1
        smallest_area = float("inf")
        for index in range(1, len(survivors) - 1):
            area = _triangle_area(point_a=survivors[index - 1], point_b=survivors[index], point_c=survivors[index + 1])
            if area < smallest_area:
                smallest_area, smallest_index = area, index
        survivors.pop(smallest_index)
    return survivors


def _interpolate_to_n(coords: list[tuple[float, float]], count: int) -> list[tuple[float, float]]:
    """Resample the polyline to exactly ``count`` points evenly spaced by arc length."""
    if len(coords) == 1:
        return [coords[0]] * count
    points = np.asarray(coords, dtype=np.float64)
    seg_lengths = np.sqrt(((points[1:] - points[:-1]) ** 2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cumulative[-1]
    if total == 0:
        return [tuple(points[0])] * count
    targets = np.linspace(0.0, total, count)
    interp_lons = np.interp(targets, cumulative, points[:, 0])
    interp_lats = np.interp(targets, cumulative, points[:, 1])
    return [(float(lon), float(lat)) for lon, lat in zip(interp_lons, interp_lats, strict=True)]
