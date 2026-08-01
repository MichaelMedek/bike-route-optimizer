"""Compact CSR routing graph + optimal path — the memory-lean inference engine.

A corridor's directed edges become one scipy.sparse cost matrix (~12 bytes/edge vs ~2.8 KB/edge for
networkx); Dijkstra over non-negative costs returns the SAME optimal path A* did. Geometry never enters here.
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from bike_router.core.constants import NodeType
from bike_router.core.errors import NoRouteError
from bike_router.core.geo import haversine_vec

logger = logging.getLogger(__name__)

_NO_PRED = -9999  # scipy predecessor sentinel: "unreachable / no predecessor"


@dataclass(frozen=True)
class RouteGraph:
    """A corridor as a CSR cost matrix plus parallel node arrays (index ↔ osmid). ``matrix`` holds
    the MIN cost of each directed (u, v) (parallel edges collapsed); ``osmids``/``lat``/``lon``/
    ``node_type`` are row-aligned and ``index`` maps osmid → CSR row (no networkx at inference).
    """

    matrix: csr_matrix
    osmids: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    node_type: np.ndarray
    index: dict[int, int]

    @property
    def n_edges(self) -> int:
        """Directed edge count after parallel-edge min-collapse (matrix nonzeros)."""
        return int(self.matrix.nnz)

    @classmethod
    def from_arrays(
        cls,
        *,
        osmids: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        node_type: np.ndarray,
        from_osmid: np.ndarray,
        to_osmid: np.ndarray,
        cost: np.ndarray,
    ) -> "RouteGraph":
        """Build from flat node arrays + parallel edge arrays (osmid endpoints + cost).

        Edges whose endpoint isn't in the node set are dropped (dangle off the corridor
        window); parallel (u, v) edges collapse to their minimum cost.
        """
        index = {int(o): i for i, o in enumerate(osmids)}
        u = np.array([index.get(int(a), -1) for a in from_osmid], dtype=np.int64)
        v = np.array([index.get(int(b), -1) for b in to_osmid], dtype=np.int64)
        keep = (u >= 0) & (v >= 0)
        u, v, cost = u[keep], v[keep], np.asarray(cost, dtype=np.float64)[keep]
        matrix = _min_cost_matrix(u=u, v=v, cost=cost, n=len(osmids))
        return cls(
            matrix=matrix,
            osmids=np.asarray(osmids),
            lat=np.asarray(lat, dtype=np.float64),
            lon=np.asarray(lon, dtype=np.float64),
            node_type=np.asarray(node_type),
            index=index,
        )

    def snap_bike_node(self, *, lat: float, lon: float) -> int:
        """Nearest BIKE node's osmid to (lat, lon) — a route must start/end pedalling.

        Rail nodes are excluded: reaching a platform always crosses a station edge, so an
        endpoint never snaps onto one (mirrors the old bike-subgraph snap).
        """
        bike = self.node_type == NodeType.BIKE
        assert bool(bike.any()), "no bike node to snap endpoint to"
        dists = haversine_vec(lat_a=lat, lon_a=lon, lat_b=self.lat[bike], lon_b=self.lon[bike])
        return int(self.osmids[bike][int(dists.argmin())])


def _min_cost_matrix(*, u: np.ndarray, v: np.ndarray, cost: np.ndarray, n: int) -> csr_matrix:
    """CSR matrix keeping the MIN cost per directed (u, v) — parallel edges collapse. csr_matrix
    SUMS duplicate coords, so first reduce to the min per (u, v) (lexsort, keep first of each group):
    the cheapest parallel edge A* traversed. All costs are > 0 (length floor), so none read as "no edge".
    """
    if len(u) == 0:
        return csr_matrix((n, n), dtype=np.float64)
    order = np.lexsort((cost, v, u))
    us, vs, cs = u[order], v[order], cost[order]
    first = np.ones(len(us), dtype=bool)
    first[1:] = (us[1:] != us[:-1]) | (vs[1:] != vs[:-1])
    return csr_matrix((cs[first], (us[first], vs[first])), shape=(n, n))


def shortest_path(*, route_graph: RouteGraph, source_osmid: int, target_osmid: int) -> list[int]:
    """Optimal source→target osmid path under the stored cost (scipy Dijkstra).

    Dijkstra on non-negative weights is provably optimal — identical path to the old A*
    (the heuristic only pruned the frontier). Raises NoRouteError if unreachable.
    """
    assert source_osmid in route_graph.index, "source node must be in the graph"
    assert target_osmid in route_graph.index, "target node must be in the graph"
    src, tgt = route_graph.index[source_osmid], route_graph.index[target_osmid]
    dist, pred = dijkstra(route_graph.matrix, directed=True, indices=[src], return_predecessors=True)
    if not np.isfinite(dist[0][tgt]):
        raise NoRouteError(f"no path from {source_osmid} to {target_osmid} in corridor")
    rows: list[int] = []
    cur = tgt
    while cur != src:
        rows.append(cur)
        cur = int(pred[0][cur])
        assert cur != _NO_PRED, "predecessor chain broke despite finite distance"
    rows.append(src)
    rows.reverse()
    return [int(route_graph.osmids[r]) for r in rows]
