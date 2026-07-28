"""Shared test fixtures.

Provides a MockDEMService (linear synthetic elevation, no real raster) adapted
from ski-resort-designer's conftest, plus a tiny MultiDiGraph builder for cost /
routing tests. No network, no GeoTIFF — CI-safe.
"""

import pathlib
from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from bike_router.constants import PARAM_SPECS, GeoConfig, Mode, NodeType, RoutingParams
from bike_router.elevation import DEMService

# The committed real Schwarzwald artifact — the ONLY data source e2e tests may use.
FIXTURE_GRAPH_DIR = pathlib.Path(__file__).parent / "fixtures" / "dach_graph"


class MockDEMService(DEMService):
    """Synthetic linear DEM: elevation = base + lat·k·ns% − lon·k·ew%.

    Near lat/lon = 0 where 1° ≈ 111,320 m, so slope percentages are exact.
    Bypasses the singleton so each test gets an independent instance.
    """

    _instance = None

    def __new__(cls, *args: object, **kwargs: object) -> "MockDEMService":
        return object.__new__(cls)

    def __init__(self, base_elevation: float = 1000.0, slope_ns_pct: float = 0.0, slope_ew_pct: float = 0.0) -> None:
        self.base_elevation = base_elevation
        self.slope_ns_pct = slope_ns_pct
        self.slope_ew_pct = slope_ew_pct
        self._bounds = (-2.0, -2.0, 2.0, 2.0)

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self._bounds

    def get_elevations(self, lons, lats):  # noqa: ANN001, ANN201
        """Vectorized synthetic elevation (the only query path)."""
        lons = np.asarray(lons, dtype=float)
        lats = np.asarray(lats, dtype=float)
        return (
            self.base_elevation
            + lats * GeoConfig.METERS_PER_DEGREE_EQUATOR * (self.slope_ns_pct / 100)
            - lons * GeoConfig.METERS_PER_DEGREE_EQUATOR * (self.slope_ew_pct / 100)
        )


@pytest.fixture
def flat_dem() -> MockDEMService:
    """Perfectly flat DEM at 1000 m."""
    return MockDEMService(base_elevation=1000.0, slope_ns_pct=0.0, slope_ew_pct=0.0)


DEFAULT_PARAMS = RoutingParams(**{spec.field: spec.default for spec in PARAM_SPECS})
# All-penalties-off params — the distance-only baseline for routing/cost tests.
ZERO_PARAMS = RoutingParams(**{spec.field: 0.0 for spec in PARAM_SPECS})


def params(**overrides: float) -> RoutingParams:
    """RoutingParams at the spec DEFAULTS, with named fields overridden."""
    return replace(DEFAULT_PARAMS, **overrides)


def zero_params(**overrides: float) -> RoutingParams:
    """RoutingParams with ALL penalties zeroed, with named fields overridden."""
    return replace(ZERO_PARAMS, **overrides)


def make_line_graph(params: RoutingParams = DEFAULT_PARAMS) -> nx.MultiDiGraph:
    """A tiny bidirectional 3-node line graph with elevations + edge costs.

    Nodes 1→2→3 west→east, node 2 higher so an uphill/downhill asymmetry exists.
    Costs use the real assign_edge_costs so tests exercise production code.
    """
    from bike_router.cost import assign_edge_costs

    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "EPSG:4326"  # OSMnx plotting reads this
    coords = {1: (8.0, 48.0, 100.0), 2: (8.01, 48.0, 130.0), 3: (8.02, 48.0, 100.0)}
    for node_id, (lon, lat, elevation) in coords.items():
        graph.add_node(node_id, x=lon, y=lat, elevation=elevation, node_type=NodeType.BIKE)
    for node_a, node_b in [(1, 2), (2, 1), (2, 3), (3, 2)]:
        graph.add_edge(node_a, node_b, key=0, length=800.0, surface="asphalt", highway="residential", mode=Mode.BIKE)
    assign_edge_costs(graph=graph, params=params)
    return graph


def make_choice_graph(params: RoutingParams) -> nx.MultiDiGraph:
    """S→T via TWO alternatives, so routing choice depends on ``params``.

    Node 2 is short/steep/paved; node 3 is a longer flat unpaved detour. Penalising
    hills/main-roads picks node 3; zeroing them picks node 2. Lengths are haversine.
    """
    from bike_router.cost import assign_edge_costs
    from bike_router.geo import haversine_distance_m

    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "EPSG:4326"
    nodes = {
        1: (8.000, 48.00, 100.0),  # S
        2: (8.025, 48.00, 130.0),  # direct, steep, paved main road (secondary)
        3: (8.025, 48.03, 100.0),  # north detour, flat, unpaved residential
        5: (8.050, 48.00, 100.0),  # T
    }
    for node_id, (lon, lat, elevation) in nodes.items():
        graph.add_node(node_id, x=lon, y=lat, elevation=elevation, node_type=NodeType.BIKE)
    mids = {2: ("secondary", "asphalt"), 3: ("residential", "gravel")}
    for mid, (highway, surface) in mids.items():
        for node_a, node_b in [(1, mid), (mid, 1), (mid, 5), (5, mid)]:
            length = haversine_distance_m(
                lat_a=nodes[node_a][1], lon_a=nodes[node_a][0], lat_b=nodes[node_b][1], lon_b=nodes[node_b][0]
            )
            graph.add_edge(node_a, node_b, key=0, length=length, highway=highway, surface=surface, mode=Mode.BIKE)
    assign_edge_costs(graph=graph, params=params)
    return graph


def _mode_edge(mode: str, length: float, surface: object = None, highway: object = None) -> dict:
    """Edge attr dict with custom_cost pre-baked to length (mode-walk tests skip costing)."""
    return {"mode": mode, "length": length, "surface": surface, "highway": highway, "custom_cost": length}


def make_mixed_mode_graph(sequence: list[tuple[int, int, str]]) -> nx.MultiDiGraph:
    """A line graph whose consecutive edges carry the GIVEN modes (for leg-splitting tests).

    A node touched by any RAIL edge is typed RAIL, the rest BIKE. Every edge gets
    custom_cost == length so cheapest-edge walks are deterministic.
    """
    rail_nodes = {node for u, v, m in sequence if m == Mode.RAIL for node in (u, v)}
    nodes = {node for u, v, _m in sequence for node in (u, v)}
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    for node in nodes:
        ntype = NodeType.RAIL if node in rail_nodes else NodeType.BIKE
        # Rail nodes carry a station name (as in the real graph); bike nodes carry None.
        name = f"Station {node}" if node in rail_nodes else None
        graph.add_node(node, x=8.0 + node * 0.01, y=48.0, elevation=100.0, node_type=ntype, station_name=name)
    for u, v, mode in sequence:
        graph.add_edge(u, v, key=0, **_mode_edge(mode=mode, length=800.0))
    return graph


def make_composition_graph() -> nx.MultiDiGraph:
    """Route 1→2→3→4→5: paved-quiet bike (1km), gravel-main bike (2km), station (0.1km), rail (10km).

    Nodes 1-3 bike, 4-5 rail — matching the node-type invariant. Used by composition tests.
    """
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    for n in (1, 2, 3):
        graph.add_node(n, x=float(n), y=48.0, elevation=100.0, node_type=NodeType.BIKE)
    for n in (4, 5):
        graph.add_node(n, x=float(n), y=48.0, elevation=100.0, node_type=NodeType.RAIL)
    graph.add_edge(1, 2, key=0, **_mode_edge(mode=Mode.BIKE, length=1000.0, surface="asphalt", highway="residential"))
    graph.add_edge(2, 3, key=0, **_mode_edge(mode=Mode.BIKE, length=2000.0, surface="gravel", highway="secondary"))
    graph.add_edge(3, 4, key=0, **_mode_edge(mode=Mode.STATION, length=100.0))
    graph.add_edge(4, 5, key=0, **_mode_edge(mode=Mode.RAIL, length=10000.0))
    return graph


def make_rail_graph() -> nx.MultiDiGraph:
    """S(bike) → station A → station B: one station-access hop then a rail ride.

    Node 1 bike; nodes 2/3 rail stations. Station edge 1→2 enters a rail node (boarding wait);
    rail 2→3 rides at RAIL_SPEED_KMH. Leg times are DERIVED in build_track from length +
    node_type — nothing time-related is stored on the edges. Used by track rail-timing tests.
    """
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    graph.add_node(1, x=8.00, y=48.0, elevation=200.0, node_type=NodeType.BIKE)
    graph.add_node(2, x=8.001, y=48.0, elevation=205.0, node_type=NodeType.RAIL)  # station A
    graph.add_node(3, x=8.10, y=48.0, elevation=600.0, node_type=NodeType.RAIL)  # station B
    graph.add_edge(1, 2, key=0, **_mode_edge(mode=Mode.STATION, length=80.0))
    graph.add_edge(2, 3, key=0, **_mode_edge(mode=Mode.RAIL, length=7000.0))
    return graph


def make_exchange_rail_graph() -> nx.MultiDiGraph:
    """Bike → A → B(exchange, degree-3 rail junction) → C → bike: one train trip, one change.

    Boarding is charged ONLY on the two station edges (½ each: board at A, alight at C). The
    exchange at B is a rail→rail hop through a degree-3 junction — NO station edge, so NO extra
    boarding. Confirms an exchange trip pays the boarding wait exactly once. Rail lengths differ
    so ride time is unambiguous. Used by the boarding-once exchange test.
    """
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    graph.add_node(1, x=8.000, y=48.0, elevation=100.0, node_type=NodeType.BIKE)  # start (bike)
    graph.add_node(2, x=8.200, y=48.0, elevation=100.0, node_type=NodeType.BIKE)  # end (bike)
    graph.add_node(-1, x=8.001, y=48.0, elevation=100.0, node_type=NodeType.RAIL, station_name="A")
    graph.add_node(-2, x=8.050, y=48.0, elevation=100.0, node_type=NodeType.RAIL, station_name="B")  # exchange
    graph.add_node(-3, x=8.199, y=48.0, elevation=100.0, node_type=NodeType.RAIL, station_name="C")
    graph.add_node(-4, x=8.050, y=48.1, elevation=100.0, node_type=NodeType.RAIL, station_name="D")  # 3rd branch at B
    # station-access hops (board at A, alight at C)
    graph.add_edge(1, -1, key=0, **_mode_edge(mode=Mode.STATION, length=90.0))
    graph.add_edge(-1, 1, key=0, **_mode_edge(mode=Mode.STATION, length=90.0))
    graph.add_edge(-3, 2, key=0, **_mode_edge(mode=Mode.STATION, length=90.0))
    graph.add_edge(2, -3, key=0, **_mode_edge(mode=Mode.STATION, length=90.0))
    # rail edges: A↔B, B↔C, B↔D (B is a degree-3 junction), both directions
    for a, b, length in [(-1, -2, 4000.0), (-2, -3, 3000.0), (-2, -4, 2000.0)]:
        graph.add_edge(a, b, key=0, **_mode_edge(mode=Mode.RAIL, length=length))
        graph.add_edge(b, a, key=0, **_mode_edge(mode=Mode.RAIL, length=length))
    return graph


def make_cutthrough_graph(params: RoutingParams, detour_m: float = 10_000.0) -> nx.MultiDiGraph:
    """L(bike) and R(bike) joined either by a LONG bike detour or THROUGH a shared station.

    L and R are both entrances (≈100 m) to one rail station S; the only pedalled alternative is
    a long detour L→M→R (``detour_m`` total). With boarding 0 the station edges are nearly free,
    so cutting THROUGH S (L→S→R) beats the detour — the accepted trade-off. With a high boarding
    penalty the full-boarding cost makes the detour win. Costs use the real assign_edge_costs.
    """
    from bike_router.cost import assign_edge_costs

    graph = nx.MultiDiGraph(crs="EPSG:4326")
    graph.add_node(1, x=8.000, y=48.000, elevation=100.0, node_type=NodeType.BIKE)  # L
    graph.add_node(2, x=8.010, y=48.000, elevation=100.0, node_type=NodeType.BIKE)  # R
    graph.add_node(3, x=8.005, y=48.050, elevation=100.0, node_type=NodeType.BIKE)  # M (detour midpoint)
    graph.add_node(-1, x=8.005, y=48.000, elevation=100.0, node_type=NodeType.RAIL, station_name="S")
    half = detour_m / 2.0
    for a, b in [(1, 3), (3, 1), (3, 2), (2, 3)]:  # long bike detour L↔M↔R
        graph.add_edge(a, b, key=0, length=half, surface="asphalt", highway="residential", mode=Mode.BIKE)
    for a, b in [(1, -1), (-1, 1), (2, -1), (-1, 2)]:  # short station edges L↔S, R↔S
        graph.add_edge(a, b, key=0, length=100.0, surface=None, highway=None, mode=Mode.STATION)
    assign_edge_costs(graph=graph, params=params)
    return graph


def make_store_roundtrip_graph() -> nx.MultiDiGraph:
    """A 4-node bike square + two rail nodes joined by rail, each linked to a bike node.

    Exercises the full node-type/mode matrix for graph_store round-trips: bike↔bike ring,
    rail↔rail edge, and bike↔rail station edges — matching the real type invariants.
    """
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    pts = {1: (8.00, 48.00, 100.0), 2: (8.01, 48.00, 110.0), 3: (8.01, 48.01, 130.0), 4: (8.00, 48.01, 120.0)}
    for nid, (lon, lat, elev) in pts.items():
        graph.add_node(nid, x=lon, y=lat, elevation=elev, node_type=NodeType.BIKE, station_name=None)
    ring = [(1, 2), (2, 3), (3, 4), (4, 1), (2, 1), (3, 2), (4, 3), (1, 4)]
    for a, b in ring:
        graph.add_edge(a, b, key=0, length=800.0, surface="asphalt", highway="residential", mode=Mode.BIKE)
    graph.add_node(-1, x=8.001, y=48.001, elevation=100.0, node_type=NodeType.RAIL, station_name="A")
    graph.add_node(-2, x=8.009, y=48.009, elevation=130.0, node_type=NodeType.RAIL, station_name="B")
    graph.add_edge(-1, -2, key=0, length=1500.0, surface=None, highway=None, mode=Mode.RAIL)  # rail↔rail
    graph.add_edge(-2, -1, key=0, length=1500.0, surface=None, highway=None, mode=Mode.RAIL)
    graph.add_edge(1, -1, key=0, length=120.0, surface=None, highway=None, mode=Mode.STATION)  # bike↔rail
    graph.add_edge(-1, 1, key=0, length=120.0, surface=None, highway=None, mode=Mode.STATION)
    graph.add_edge(3, -2, key=0, length=120.0, surface=None, highway=None, mode=Mode.STATION)
    graph.add_edge(-2, 3, key=0, length=120.0, surface=None, highway=None, mode=Mode.STATION)
    return graph


def make_surface_mix_graph() -> nx.MultiDiGraph:
    """A 6-node line spanning both the surface AND highway allowlist boundaries (drop tests).

    1→2 allowlisted (asphalt/residential), 2→3 untagged surface (kept), 3→4 disallowed
    surface (sand), 4→5 a list naming a disallowed surface (gravel;dirt), 5→6 a disallowed
    highway (motorway — no bikes). Edges 3→4, 4→5, 5→6 must be dropped. Pre-cost.
    """
    graph = nx.MultiDiGraph()
    for node in (1, 2, 3, 4, 5, 6):
        graph.add_node(node, x=float(node), y=0.0)
    graph.add_edge(1, 2, key=0, length=100.0, surface="asphalt", highway="residential")
    graph.add_edge(2, 3, key=0, length=100.0, surface=None, highway="path")
    graph.add_edge(3, 4, key=0, length=100.0, surface="sand", highway="path")
    graph.add_edge(4, 5, key=0, length=100.0, surface="gravel;dirt", highway="track")
    graph.add_edge(5, 6, key=0, length=100.0, surface="asphalt", highway="motorway")  # bad highway
    return graph


def make_two_cluster_graph() -> nx.MultiDiGraph:
    """Two tight knots (~5 m nodes) ~2 km apart, linked by one long edge (for consolidation).

    Each cluster's near-identical nodes merge under a 25 m tolerance; the 1.5 km link
    between clusters survives. Pre-cost (consolidation runs before edge costing).
    """
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    coords = {
        1: (8.0000, 48.0),
        2: (8.00005, 48.0),
        3: (8.0001, 48.0),  # cluster A
        4: (8.0200, 48.0),
        5: (8.02005, 48.0),
        6: (8.0201, 48.0),  # cluster B
    }
    for nid, (lon, lat) in coords.items():
        graph.add_node(nid, x=lon, y=lat)
    for a, b in [(3, 4), (4, 3)]:  # link the clusters
        graph.add_edge(a, b, key=0, length=1500.0, surface="asphalt", highway="residential")
    for a, b in [(1, 2), (2, 3), (2, 1), (3, 2), (4, 5), (5, 6), (5, 4), (6, 5)]:
        graph.add_edge(a, b, key=0, length=5.0, surface="asphalt", highway="residential")
    return graph


def make_condition_route_graph() -> nx.MultiDiGraph:
    """Flat 1→2→3 bike route: leg 1→2 good/quiet, leg 2→3 a main road (primary).

    So build_track marks point 2 not-bad and point 3 road_bad=True — the single
    graph the condition/colour tests diff against. custom_cost==length (deterministic).
    """
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    for node in (1, 2, 3):
        graph.add_node(node, x=8.00 + node * 0.01, y=48.0, elevation=100.0, node_type=NodeType.BIKE)
    graph.add_edge(1, 2, key=0, **_mode_edge(mode=Mode.BIKE, length=800.0, surface="asphalt", highway="residential"))
    graph.add_edge(2, 3, key=0, **_mode_edge(mode=Mode.BIKE, length=800.0, surface="asphalt", highway="primary"))
    return graph


def make_densify_detour_graph() -> nx.MultiDiGraph:
    """A single 1→2 bike edge whose baked 3D geometry bulges EAST and rises to 200 m.

    Endpoints are due-north (100→140 m) but the polyline detours east through a 200 m
    apex, so densify tests can assert it follows the real 3D vertices. custom_cost set.
    """
    from shapely.geometry import LineString

    graph = nx.MultiDiGraph(crs="EPSG:4326")
    graph.add_node(1, x=8.00, y=48.00, elevation=100.0, node_type=NodeType.BIKE)
    graph.add_node(2, x=8.00, y=48.02, elevation=140.0, node_type=NodeType.BIKE)
    detour = LineString([(8.00, 48.00, 100.0), (8.03, 48.01, 200.0), (8.00, 48.02, 140.0)])
    graph.add_edge(
        1,
        2,
        key=0,
        **_mode_edge(mode=Mode.BIKE, length=3000.0, surface="asphalt", highway="residential"),
        geometry=detour,
    )
    return graph


def make_hill_vs_rail_graph(
    *, params: RoutingParams, climb_m: float, rail_alternative: bool, bike_km: float = 6.0
) -> nx.MultiDiGraph:
    """Start→End joined by a paved bike path climbing ``climb_m``, optionally shadowed by rail.

    Node 1 = start (elevation 100 m), node 2 = end (100 + ``climb_m``); the direct BIKE edge
    spans ``bike_km`` at that grade. When ``rail_alternative`` is set, a station at each end
    (nodes -1/-2) is joined by a RAIL edge of the same length — so the router picks bike vs
    train purely on the tuned penalties. Reversing source/target makes the same hill a descent
    (uphill penalty 0 downhill), so the train should NOT be chosen going down. Real costing.
    """
    from bike_router.cost import assign_edge_costs

    bike_m = bike_km * 1000.0
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    graph.add_node(1, x=8.000, y=48.000, elevation=100.0, node_type=NodeType.BIKE)  # start
    graph.add_node(2, x=8.080, y=48.000, elevation=100.0 + climb_m, node_type=NodeType.BIKE)  # end (higher)
    for a, b in [(1, 2), (2, 1)]:  # the pedalled climb both ways (uphill dir carries the penalty)
        graph.add_edge(a, b, key=0, length=bike_m, surface="asphalt", highway="residential", mode=Mode.BIKE)
    if rail_alternative:
        graph.add_node(-1, x=8.001, y=48.000, elevation=100.0, node_type=NodeType.RAIL, station_name="Start Bf")
        graph.add_node(-2, x=8.079, y=48.000, elevation=100.0 + climb_m, node_type=NodeType.RAIL, station_name="End Bf")
        for a, b in [(1, -1), (-1, 1), (2, -2), (-2, 2)]:  # short station-access hops
            graph.add_edge(a, b, key=0, length=100.0, surface=None, highway=None, mode=Mode.STATION)
        for a, b in [(-1, -2), (-2, -1)]:  # the train ride, both directions
            graph.add_edge(a, b, key=0, length=bike_m, surface=None, highway=None, mode=Mode.RAIL)
    assign_edge_costs(graph=graph, params=params)
    return graph
