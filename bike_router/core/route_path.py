"""The route as an ordered edge list — the single inference-side representation.

A computed route IS its ordered nodes plus the ordered edges between them; there is no
networkx graph at inference. ``RouteEdge`` carries only what drawing/stats/legs need
(mode, length, tags, oriented 2D geometry); ``RouteNode`` carries coords + baked
elevation + type. ``edges[i]`` joins ``nodes[i] → nodes[i+1]`` — asserted at construction.
"""

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteNode:
    """One node on the route: position, baked elevation, kind, and station name (rail only)."""

    osmid: int
    lat: float
    lon: float
    elevation_m: float
    node_type: str
    station_name: str | None


@dataclass(frozen=True)
class RouteEdge:
    """One hop on the route, oriented from_node → to_node.

    ``geometry`` is the real 2D polyline as ``[(lon, lat), ...]`` oriented along travel, or
    None for a straight rail/station hop (drawn node-to-node). surface/highway stay optional
    external OSM tags; length_m is the baked edge length.
    """

    from_node: int
    to_node: int
    mode: str
    length_m: float
    surface: object
    highway: object
    geometry: list[tuple[float, float]] | None


@dataclass(frozen=True)
class RoutePath:
    """The ordered nodes + the edges between them (``edges[i]`` joins node i → i+1)."""

    nodes: list[RouteNode]
    edges: list[RouteEdge]

    def __post_init__(self) -> None:
        assert len(self.nodes) >= 2, "route must have >= 2 nodes"
        assert len(self.edges) == len(self.nodes) - 1, "one edge per consecutive node pair"
        for edge, node_a, node_b in zip(self.edges, self.nodes[:-1], self.nodes[1:], strict=True):
            assert edge.from_node == node_a.osmid and edge.to_node == node_b.osmid, (
                f"edge {edge.from_node}->{edge.to_node} does not join {node_a.osmid}->{node_b.osmid}"
            )

    @property
    def osmids(self) -> list[int]:
        """The ordered node osmids of the route."""
        return [node.osmid for node in self.nodes]

    def iter_edges(self) -> Iterator[tuple[RouteNode, RouteNode, RouteEdge]]:
        """Yield ``(node_a, node_b, edge)`` for each hop, in route order."""
        yield from zip(self.nodes[:-1], self.nodes[1:], self.edges, strict=True)

    def subpath(self, *, start_index: int, end_index: int) -> "RoutePath":
        """The contiguous sub-route from node ``start_index`` to ``end_index`` (inclusive).

        Used to isolate one pedalled leg for its own linestring; indices are positions on
        this path (0-based), end > start.
        """
        assert 0 <= start_index < end_index < len(self.nodes), "subpath indices out of range"
        return RoutePath(nodes=self.nodes[start_index : end_index + 1], edges=self.edges[start_index:end_index])
