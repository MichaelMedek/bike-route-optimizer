"""Shared test fixtures.

Provides a MockDEMService (linear synthetic elevation, no real raster) plus tiny
RoutePath / edge-array builders for the cost, routing, track, and composition tests.
No network, no GeoTIFF — CI-safe.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

# Import the two entry-point shells ONCE so coverage MEASURES them
import app_webmap  # noqa: E402, F401
import bike_route  # noqa: E402, F401
from bike_router.core.constants import PARAM_SPECS, PROJECT_ROOT, GeoConfig, Mode, NodeType, RoutingParams
from bike_router.core.cost import edge_cost_array
from bike_router.core.geo import haversine_distance_m
from bike_router.core.route_path import RouteEdge, RouteNode, RoutePath

# The committed real Schwarzwald artifact — the ONLY data source e2e tests may use.
FIXTURE_GRAPH_DIR = PROJECT_ROOT / "tests" / "fixtures" / "dach_graph"
# Tiny committed tiled store (built offline from make_store_roundtrip_graph).
FIXTURE_ROUNDTRIP_STORE = PROJECT_ROOT / "tests" / "fixtures" / "roundtrip_store"


class MockDEMService:
    """Synthetic linear DEM: elevation = base + lat·k·ns% − lon·k·ew%.

    Duck-typed to the real ``preprocessing.elevation.DEMService`` interface (is_loaded, bounds,
    get_elevations) WITHOUT subclassing it — so the shared test session never imports rasterio.
    Near lat/lon = 0 where 1° ≈ 111,320 m, so slope percentages are exact.
    """

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


@pytest.fixture
def roundtrip_store() -> "object":
    """The committed tiny tiled store — CORE graph_store tests READ it, no build stack needed."""
    return FIXTURE_ROUNDTRIP_STORE


DEFAULT_PARAMS = RoutingParams(**{spec.field: spec.default for spec in PARAM_SPECS})
# All-penalties-off params — the distance-only baseline for routing/cost tests.
ZERO_PARAMS = RoutingParams(**{spec.field: 0.0 for spec in PARAM_SPECS})


def params(**overrides: float) -> RoutingParams:
    """RoutingParams at the spec DEFAULTS, with named fields overridden."""
    return replace(DEFAULT_PARAMS, **overrides)


def zero_params(**overrides: float) -> RoutingParams:
    """RoutingParams with ALL penalties zeroed, with named fields overridden."""
    return replace(ZERO_PARAMS, **overrides)


# --- edge-array builders (for the CSR RouteGraph.from_arrays routing tests) ------------------


class EdgeArrays:
    """Accumulates nodes + directed edges into the flat arrays RouteGraph.from_arrays wants.

    A tiny test-only graph builder: add_node/add_edge, then route_graph_args() costs every
    edge via the real edge_cost_array (production formula) so routing tests exercise the real cost.
    """

    def __init__(self) -> None:
        self._nodes: dict[int, tuple[float, float, float, str]] = {}  # osmid → (lat, lon, elev, node_type)
        self._edges: list[tuple[int, int, str, float, object, object]] = []

    def add_node(self, osmid: int, *, lon: float, lat: float, elevation: float, node_type: str = NodeType.BIKE) -> None:
        self._nodes[osmid] = (lat, lon, elevation, node_type)

    def add_edge(
        self, u: int, v: int, *, mode: str, length: float, surface: object = None, highway: object = None
    ) -> None:
        self._edges.append((u, v, mode, length, surface, highway))

    def _edges_df(self) -> pd.DataFrame:
        """The accumulated edges as the edge table edge_cost_array expects."""
        return pd.DataFrame(self._edges, columns=["from_node", "to_node", "mode", "length_m", "surface", "highway"])

    def _costs(self, *, params: RoutingParams) -> np.ndarray:
        """Per-edge cost for every accumulated edge, via the ONE vectorized cost formula."""
        elev = {o: self._nodes[o][2] for o in self._nodes}
        return edge_cost_array(edges_df=self._edges_df(), elev_by_osmid=elev, params=params)

    def route_graph_args(self, *, params: RoutingParams) -> dict[str, np.ndarray]:
        """Flat arrays for RouteGraph.from_arrays — edges costed by the real edge_cost_array."""
        osmids = list(self._nodes)
        return {
            "osmids": np.array(osmids, dtype="int64"),
            "lat": np.array([self._nodes[o][0] for o in osmids], dtype=float),
            "lon": np.array([self._nodes[o][1] for o in osmids], dtype=float),
            "node_type": np.array([self._nodes[o][3] for o in osmids], dtype=object),
            "from_osmid": np.array([e[0] for e in self._edges], dtype="int64"),
            "to_osmid": np.array([e[1] for e in self._edges], dtype="int64"),
            "cost": self._costs(params=params),
        }

    def edge_cost_of(self, u: int, v: int, *, params: RoutingParams) -> float:
        """The real cost of the (u, v) edge — for cost-floor assertions."""
        costs = self._costs(params=params)
        for i, (eu, ev, *_rest) in enumerate(self._edges):
            if (eu, ev) == (u, v):
                return float(costs[i])
        raise KeyError(f"no edge {u}->{v}")


def make_line_edges() -> EdgeArrays:
    """A tiny bidirectional 3-node line: 1→2→3 west→east, node 2 higher (uphill/downhill)."""
    arr = EdgeArrays()
    for osmid, (lon, lat, elev) in {1: (8.0, 48.0, 100.0), 2: (8.01, 48.0, 130.0), 3: (8.02, 48.0, 100.0)}.items():
        arr.add_node(osmid, lon=lon, lat=lat, elevation=elev)
    for u, v in [(1, 2), (2, 1), (2, 3), (3, 2)]:
        arr.add_edge(u, v, mode=Mode.BIKE, length=800.0, surface="asphalt", highway="residential")
    return arr


def make_choice_edges() -> EdgeArrays:
    """S→T via TWO alternatives: node 2 short/steep/paved-main; node 3 long/flat/unpaved-quiet."""
    arr = EdgeArrays()
    nodes = {1: (8.000, 48.00, 100.0), 2: (8.025, 48.00, 130.0), 3: (8.025, 48.03, 100.0), 5: (8.050, 48.00, 100.0)}
    for osmid, (lon, lat, elev) in nodes.items():
        arr.add_node(osmid, lon=lon, lat=lat, elevation=elev)
    for mid, (highway, surface) in {2: ("secondary", "asphalt"), 3: ("residential", "gravel")}.items():
        for u, v in [(1, mid), (mid, 1), (mid, 5), (5, mid)]:
            length = haversine_distance_m(lat_a=nodes[u][1], lon_a=nodes[u][0], lat_b=nodes[v][1], lon_b=nodes[v][0])
            arr.add_edge(u, v, mode=Mode.BIKE, length=length, surface=surface, highway=highway)
    return arr


def make_cutthrough_edges(detour_m: float = 10_000.0) -> EdgeArrays:
    """L and R joined either by a long bike detour L→M→R or through a shared station S."""
    arr = EdgeArrays()
    arr.add_node(1, lon=8.000, lat=48.000, elevation=100.0)  # L
    arr.add_node(2, lon=8.010, lat=48.000, elevation=100.0)  # R
    arr.add_node(3, lon=8.005, lat=48.050, elevation=100.0)  # M (detour midpoint)
    arr.add_node(-1, lon=8.005, lat=48.000, elevation=100.0, node_type=NodeType.RAIL)
    half = detour_m / 2.0
    for u, v in [(1, 3), (3, 1), (3, 2), (2, 3)]:
        arr.add_edge(u, v, mode=Mode.BIKE, length=half, surface="asphalt", highway="residential")
    for u, v in [(1, -1), (-1, 1), (2, -1), (-1, 2)]:
        arr.add_edge(u, v, mode=Mode.STATION, length=100.0)
    return arr


def make_hill_vs_rail_edges(*, climb_m: float, rail_alternative: bool, bike_km: float = 6.0) -> EdgeArrays:
    """Start→End on a paved bike path climbing ``climb_m``, optionally shadowed by a rail line.

    Node 1 = start (100 m), node 2 = end (100 + climb_m); the direct BIKE edge spans ``bike_km``
    both ways (uphill dir carries the penalty). With ``rail_alternative`` a station at each end
    (-1/-2) joined by a same-length RAIL edge lets the router pick bike vs train on the penalties.
    """
    arr = EdgeArrays()
    bike_m = bike_km * 1000.0
    arr.add_node(1, lon=8.000, lat=48.000, elevation=100.0)
    arr.add_node(2, lon=8.080, lat=48.000, elevation=100.0 + climb_m)
    for u, v in [(1, 2), (2, 1)]:
        arr.add_edge(u, v, mode=Mode.BIKE, length=bike_m, surface="asphalt", highway="residential")
    if rail_alternative:
        arr.add_node(-1, lon=8.001, lat=48.000, elevation=100.0, node_type=NodeType.RAIL)
        arr.add_node(-2, lon=8.079, lat=48.000, elevation=100.0 + climb_m, node_type=NodeType.RAIL)
        for u, v in [(1, -1), (-1, 1), (2, -2), (-2, 2)]:
            arr.add_edge(u, v, mode=Mode.STATION, length=100.0)
        for u, v in [(-1, -2), (-2, -1)]:
            arr.add_edge(u, v, mode=Mode.RAIL, length=bike_m)
    return arr


def route_node_types(arr: EdgeArrays, path: list[int]) -> list[str]:
    """node_type per osmid on a path — for the 'does the route board a train?' assertions."""
    return [arr._nodes[osmid][3] for osmid in path]  # noqa: SLF001 — test helper reads its own builder


# --- RoutePath builders (for track / composition / simplify / plotting tests) ----------------


def _node(osmid: int, *, lon: float, lat: float, elev: float, node_type: str, name: str | None) -> RouteNode:
    return RouteNode(osmid=osmid, lat=lat, lon=lon, elevation_m=elev, node_type=node_type, station_name=name)


def make_line_route() -> RoutePath:
    """RoutePath 1→2→3: two 800 m asphalt/residential bike edges; 100→130→100 m."""
    nodes = [
        _node(1, lon=8.00, lat=48.0, elev=100.0, node_type=NodeType.BIKE, name=None),
        _node(2, lon=8.01, lat=48.0, elev=130.0, node_type=NodeType.BIKE, name=None),
        _node(3, lon=8.02, lat=48.0, elev=100.0, node_type=NodeType.BIKE, name=None),
    ]
    edges = [
        RouteEdge(
            from_node=1,
            to_node=2,
            mode=Mode.BIKE,
            length_m=800.0,
            surface="asphalt",
            highway="residential",
            geometry=None,
        ),
        RouteEdge(
            from_node=2,
            to_node=3,
            mode=Mode.BIKE,
            length_m=800.0,
            surface="asphalt",
            highway="residential",
            geometry=None,
        ),
    ]
    return RoutePath(nodes=nodes, edges=edges)


def _bike(osmid: int) -> RouteNode:
    return _node(osmid, lon=8.0 + osmid * 0.01, lat=48.0, elev=100.0, node_type=NodeType.BIKE, name=None)


def _rail(osmid: int, *, name: str | None = None) -> RouteNode:
    return _node(
        osmid,
        lon=8.0 + osmid * 0.01,
        lat=48.0,
        elev=100.0,
        node_type=NodeType.RAIL,
        name=name if name is not None else f"Station {osmid}",
    )


def make_mixed_mode_route(sequence: list[tuple[int, int, str]]) -> RoutePath:
    """A line RoutePath whose consecutive edges carry the GIVEN modes (leg-splitting tests).

    A node touched by any RAIL/STATION-rail edge is typed RAIL (with a station name), else BIKE.
    """
    rail_nodes = {node for u, v, m in sequence if m == Mode.RAIL for node in (u, v)}
    order: list[int] = []
    for u, v, _m in sequence:
        for node in (u, v):
            if node not in order:
                order.append(node)
    node_by_id = {n: (_rail(n) if n in rail_nodes else _bike(n)) for n in order}
    nodes = [node_by_id[sequence[0][0]]] + [node_by_id[v] for _u, v, _m in sequence]
    edges = [
        RouteEdge(from_node=u, to_node=v, mode=mode, length_m=800.0, surface=None, highway=None, geometry=None)
        for u, v, mode in sequence
    ]
    return RoutePath(nodes=nodes, edges=edges)


def make_composition_route() -> RoutePath:
    """1→2→3→4→5: paved-quiet bike (1km), gravel-main bike (2km), station (0.1km), rail (10km)."""
    nodes = [
        _node(1, lon=1.0, lat=48.0, elev=100.0, node_type=NodeType.BIKE, name=None),
        _node(2, lon=2.0, lat=48.0, elev=100.0, node_type=NodeType.BIKE, name=None),
        _node(3, lon=3.0, lat=48.0, elev=100.0, node_type=NodeType.BIKE, name=None),
        _node(4, lon=4.0, lat=48.0, elev=100.0, node_type=NodeType.RAIL, name="A"),
        _node(5, lon=5.0, lat=48.0, elev=100.0, node_type=NodeType.RAIL, name="B"),
    ]
    edges = [
        RouteEdge(
            from_node=1,
            to_node=2,
            mode=Mode.BIKE,
            length_m=1000.0,
            surface="asphalt",
            highway="residential",
            geometry=None,
        ),
        RouteEdge(
            from_node=2,
            to_node=3,
            mode=Mode.BIKE,
            length_m=2000.0,
            surface="gravel",
            highway="secondary",
            geometry=None,
        ),
        RouteEdge(from_node=3, to_node=4, mode=Mode.STATION, length_m=100.0, surface=None, highway=None, geometry=None),
        RouteEdge(from_node=4, to_node=5, mode=Mode.RAIL, length_m=10000.0, surface=None, highway=None, geometry=None),
    ]
    return RoutePath(nodes=nodes, edges=edges)


def make_rail_route() -> RoutePath:
    """S(bike) → station A → station B: one station-access hop then a rail ride (200→205→600 m)."""
    nodes = [
        _node(1, lon=8.00, lat=48.0, elev=200.0, node_type=NodeType.BIKE, name=None),
        _node(2, lon=8.001, lat=48.0, elev=205.0, node_type=NodeType.RAIL, name="A"),
        _node(3, lon=8.10, lat=48.0, elev=600.0, node_type=NodeType.RAIL, name="B"),
    ]
    edges = [
        RouteEdge(from_node=1, to_node=2, mode=Mode.STATION, length_m=80.0, surface=None, highway=None, geometry=None),
        RouteEdge(from_node=2, to_node=3, mode=Mode.RAIL, length_m=7000.0, surface=None, highway=None, geometry=None),
    ]
    return RoutePath(nodes=nodes, edges=edges)


def make_exchange_rail_route() -> RoutePath:
    """Bike → A → B(exchange) → C → bike: one train trip, one on-train change, flat (all 100 m)."""
    nodes = [
        _node(1, lon=8.000, lat=48.0, elev=100.0, node_type=NodeType.BIKE, name=None),
        _node(-1, lon=8.001, lat=48.0, elev=100.0, node_type=NodeType.RAIL, name="A"),
        _node(-2, lon=8.050, lat=48.0, elev=100.0, node_type=NodeType.RAIL, name="B"),
        _node(-3, lon=8.199, lat=48.0, elev=100.0, node_type=NodeType.RAIL, name="C"),
        _node(2, lon=8.200, lat=48.0, elev=100.0, node_type=NodeType.BIKE, name=None),
    ]
    edges = [
        RouteEdge(from_node=1, to_node=-1, mode=Mode.STATION, length_m=90.0, surface=None, highway=None, geometry=None),
        RouteEdge(from_node=-1, to_node=-2, mode=Mode.RAIL, length_m=4000.0, surface=None, highway=None, geometry=None),
        RouteEdge(from_node=-2, to_node=-3, mode=Mode.RAIL, length_m=3000.0, surface=None, highway=None, geometry=None),
        RouteEdge(from_node=-3, to_node=2, mode=Mode.STATION, length_m=90.0, surface=None, highway=None, geometry=None),
    ]
    return RoutePath(nodes=nodes, edges=edges)


def make_condition_route() -> RoutePath:
    """Flat 1→2→3 bike: leg 1→2 good/quiet, leg 2→3 a main road (primary)."""
    nodes = [
        _node(1, lon=8.01, lat=48.0, elev=100.0, node_type=NodeType.BIKE, name=None),
        _node(2, lon=8.02, lat=48.0, elev=100.0, node_type=NodeType.BIKE, name=None),
        _node(3, lon=8.03, lat=48.0, elev=100.0, node_type=NodeType.BIKE, name=None),
    ]
    edges = [
        RouteEdge(
            from_node=1,
            to_node=2,
            mode=Mode.BIKE,
            length_m=800.0,
            surface="asphalt",
            highway="residential",
            geometry=None,
        ),
        RouteEdge(
            from_node=2, to_node=3, mode=Mode.BIKE, length_m=800.0, surface="asphalt", highway="primary", geometry=None
        ),
    ]
    return RoutePath(nodes=nodes, edges=edges)


def make_densify_detour_route() -> RoutePath:
    """A single 1→2 bike edge whose 2D geometry bulges EAST; nodes 100→140 m (no baked apex)."""
    nodes = [
        _node(1, lon=8.00, lat=48.00, elev=100.0, node_type=NodeType.BIKE, name=None),
        _node(2, lon=8.00, lat=48.02, elev=140.0, node_type=NodeType.BIKE, name=None),
    ]
    geometry = [(8.00, 48.00), (8.03, 48.01), (8.00, 48.02)]  # eastward bulge, 2D only
    edges = [
        RouteEdge(
            from_node=1,
            to_node=2,
            mode=Mode.BIKE,
            length_m=3000.0,
            surface="asphalt",
            highway="residential",
            geometry=geometry,
        ),
    ]
    return RoutePath(nodes=nodes, edges=edges)


# --- Build-time networkx fixtures (preprocessing tests: graph_ops / graph_store / builder) ----
# These legitimately use networkx — the offline dataset build is the ONLY place networkx lives.


def make_store_roundtrip_graph() -> "object":
    """A 4-node bike square + two rail nodes joined by rail, each linked to a bike node.

    Exercises the full node-type/mode matrix for graph_store round-trips: bike↔bike ring,
    rail↔rail edge, and bike↔rail station edges — matching the real type invariants.
    """
    import networkx as nx

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


def write_store_roundtrip_fixture(out_dir: "object") -> "object":
    """Write the round-trip graph as a tiled parquet store under ``out_dir`` (returns it).

    Wraps the preprocessing writer here in shared test infra so LAYER tests (e.g. core/graph_store)
    can obtain a real on-disk store WITHOUT importing preprocessing themselves (import-boundary safe).
    """
    from bike_router.preprocessing.graph_writer import graph_to_tables, write_graph_parquet

    nodes_df, edges_df = graph_to_tables(graph=make_store_roundtrip_graph())
    meta = {"bbox": [7.9, 47.9, 8.2, 48.1], "tile_deg": 0.5, "tolerance_m": 25.0}
    write_graph_parquet(nodes_df=nodes_df, edges_df=edges_df, meta=meta, out_dir=out_dir)
    return out_dir


def make_surface_mix_graph() -> "object":
    """A 6-node line spanning both the surface AND highway allowlist boundaries (drop tests).

    1→2 allowlisted (asphalt/residential), 2→3 untagged surface (kept), 3→4 disallowed
    surface (sand), 4→5 a list naming a disallowed surface (gravel;dirt), 5→6 a disallowed
    highway (motorway — no bikes). Edges 3→4, 4→5, 5→6 must be dropped. Pre-cost.
    """
    import networkx as nx

    graph = nx.MultiDiGraph()
    for node in (1, 2, 3, 4, 5, 6):
        graph.add_node(node, x=float(node), y=0.0)
    graph.add_edge(1, 2, key=0, length=100.0, surface="asphalt", highway="residential")
    graph.add_edge(2, 3, key=0, length=100.0, surface=None, highway="path")
    graph.add_edge(3, 4, key=0, length=100.0, surface="sand", highway="path")
    graph.add_edge(4, 5, key=0, length=100.0, surface="gravel;dirt", highway="track")
    graph.add_edge(5, 6, key=0, length=100.0, surface="asphalt", highway="motorway")  # bad highway
    return graph


def make_two_cluster_graph() -> "object":
    """Two tight knots (~5 m nodes) ~2 km apart, linked by one long edge (for consolidation).

    Each cluster's near-identical nodes merge under a 25 m tolerance; the 1.5 km link
    between clusters survives. Pre-cost (consolidation runs before edge costing).
    """
    import networkx as nx

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
