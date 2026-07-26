"""Shared test fixtures.

Provides a MockDEMService (linear synthetic elevation, no real raster) adapted
from ski-resort-designer's conftest, plus a tiny MultiDiGraph builder for cost /
routing tests. No network, no GeoTIFF — CI-safe.
"""

import networkx as nx
import numpy as np
import pytest

from bike_router.constants import PARAM_SPECS, GeoConfig, Mode, RoutingParams
from bike_router.elevation import DEMService

METERS_PER_DEGREE = GeoConfig.METERS_PER_DEGREE_EQUATOR


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
            + lats * METERS_PER_DEGREE * (self.slope_ns_pct / 100)
            - lons * METERS_PER_DEGREE * (self.slope_ew_pct / 100)
        )


@pytest.fixture
def flat_dem() -> MockDEMService:
    """Perfectly flat DEM at 1000 m."""
    return MockDEMService(base_elevation=1000.0, slope_ns_pct=0.0, slope_ew_pct=0.0)


DEFAULT_PARAMS = RoutingParams(**{spec.field: spec.default for spec in PARAM_SPECS})


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
        graph.add_node(node_id, x=lon, y=lat, elevation=elevation)
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
        graph.add_node(node_id, x=lon, y=lat, elevation=elevation)
    mids = {2: ("secondary", "asphalt"), 3: ("residential", "gravel")}
    for mid, (highway, surface) in mids.items():
        for node_a, node_b in [(1, mid), (mid, 1), (mid, 5), (5, mid)]:
            length = haversine_distance_m(
                lat_a=nodes[node_a][1], lon_a=nodes[node_a][0], lat_b=nodes[node_b][1], lon_b=nodes[node_b][0]
            )
            graph.add_edge(node_a, node_b, key=0, length=length, highway=highway, surface=surface, mode=Mode.BIKE)
    assign_edge_costs(graph=graph, params=params)
    return graph
