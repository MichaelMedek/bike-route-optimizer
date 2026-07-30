"""route_path tests — the ordered edge-list inference representation (no networkx)."""

import pytest

from bike_router.core.constants import Mode, NodeType
from bike_router.core.route_path import RouteEdge, RouteNode, RoutePath


def _node(osmid: int, lon: float) -> RouteNode:
    """A bike RouteNode at (lon, 48.0), 100 m — one builder for the tests below."""
    return RouteNode(osmid=osmid, lat=48.0, lon=lon, elevation_m=100.0, node_type=NodeType.BIKE, station_name=None)


def _edge(u: int, v: int) -> RouteEdge:
    """A straight bike RouteEdge u→v (no geometry)."""
    return RouteEdge(
        from_node=u, to_node=v, mode=Mode.BIKE, length_m=800.0, surface="asphalt", highway="residential", geometry=None
    )


class TestRouteNode:
    def test_carries_position_elevation_type_and_name(self):
        node = RouteNode(osmid=7, lat=48.0, lon=8.1, elevation_m=305.0, node_type=NodeType.RAIL, station_name="Bf")
        assert (node.osmid, node.lat, node.lon, node.elevation_m) == (7, 48.0, 8.1, 305.0)
        assert node.node_type == NodeType.RAIL and node.station_name == "Bf"

    def test_is_frozen(self):
        node = _node(1, 8.0)
        with pytest.raises(AttributeError):
            node.elevation_m = 200.0  # type: ignore[misc]  # frozen dataclass rejects assignment


class TestRouteEdge:
    def test_geometry_optional_and_tags_kept(self):
        geom = [(8.0, 48.0), (8.01, 48.0)]
        edge = RouteEdge(
            from_node=1, to_node=2, mode=Mode.BIKE, length_m=800.0, surface="gravel", highway="track", geometry=geom
        )
        assert edge.geometry == geom and edge.surface == "gravel" and edge.highway == "track"
        assert _edge(1, 2).geometry is None  # a straight hop carries no polyline

    def test_is_frozen(self):
        with pytest.raises(AttributeError):
            _edge(1, 2).length_m = 1.0  # type: ignore[misc]


class TestRoutePath:
    def test_iter_edges_yields_node_node_edge_in_order(self):
        nodes = [_node(1, 8.0), _node(2, 8.01), _node(3, 8.02)]
        edges = [_edge(1, 2), _edge(2, 3)]
        route = RoutePath(nodes=nodes, edges=edges)
        hops = list(route.iter_edges())
        assert [(a.osmid, b.osmid, e.from_node) for a, b, e in hops] == [(1, 2, 1), (2, 3, 2)]
        assert route.osmids == [1, 2, 3]

    def test_subpath_slices_inclusive(self):
        nodes = [_node(1, 8.0), _node(2, 8.01), _node(3, 8.02)]
        route = RoutePath(nodes=nodes, edges=[_edge(1, 2), _edge(2, 3)])
        sub = route.subpath(start_index=1, end_index=2)
        assert sub.osmids == [2, 3] and len(sub.edges) == 1

    def test_rejects_edge_not_joining_its_nodes(self):
        # edges[i] MUST join nodes[i]→nodes[i+1]; a mismatch fails loud at construction.
        with pytest.raises(AssertionError, match="does not join"):
            RoutePath(nodes=[_node(1, 8.0), _node(2, 8.01)], edges=[_edge(1, 9)])

    def test_rejects_wrong_edge_count(self):
        with pytest.raises(AssertionError, match="one edge per consecutive"):
            RoutePath(nodes=[_node(1, 8.0), _node(2, 8.01)], edges=[])
