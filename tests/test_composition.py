"""Route-composition tests — km tallied by surface, road class, and travel mode."""

import networkx as nx

from bike_router.composition import format_composition, route_composition
from bike_router.constants import Mode


def _edge(mode: str, length: float, surface=None, highway=None, cost=None):  # noqa: ANN001, ANN202
    return {
        "mode": mode,
        "length": length,
        "surface": surface,
        "highway": highway,
        "custom_cost": cost if cost is not None else length,
    }


def _graph() -> nx.MultiDiGraph:
    """1→2 paved quiet bike (1 km), 2→3 gravel main-road bike (2 km),
    3→4 transfer (0.1 km), 4→5 rail (10 km).
    """
    g = nx.MultiDiGraph()
    for n in (1, 2, 3, 4, 5):
        g.add_node(n, x=float(n), y=48.0, elevation=100.0)
    g.add_edge(1, 2, key=0, **_edge(Mode.BIKE, 1000.0, surface="asphalt", highway="residential"))
    g.add_edge(2, 3, key=0, **_edge(Mode.BIKE, 2000.0, surface="gravel", highway="secondary"))
    g.add_edge(3, 4, key=0, **_edge(Mode.TRANSFER, 100.0))
    g.add_edge(4, 5, key=0, **_edge(Mode.RAIL, 10000.0))
    return g


def test_composition_tallies_surface_road_and_mode():
    comp = route_composition(graph=_graph(), node_path=[1, 2, 3, 4, 5])
    # surface / road describe bike legs only (1 km paved + 2 km gravel)
    assert comp.by_surface_km["paved"] == 1.0
    assert comp.by_surface_km["gravel/unpaved"] == 2.0
    assert comp.by_road_km["quiet way"] == 1.0
    assert comp.by_road_km["main road"] == 2.0
    # mode covers the whole route
    assert comp.by_mode_km[Mode.BIKE] == 3.0
    assert comp.by_mode_km[Mode.TRANSFER] == 0.1
    assert comp.by_mode_km[Mode.RAIL] == 10.0


def test_format_composition_includes_mode_split_only_when_rail_used():
    with_rail = format_composition(comp=route_composition(graph=_graph(), node_path=[1, 2, 3, 4, 5]))
    assert "Surface:" in with_rail and "Roads:" in with_rail and "Mode:" in with_rail
    assert "rail: 10.0 km" in with_rail

    # bike-only route → no "Mode:" section (it would be all-bike, redundant)
    g = nx.MultiDiGraph()
    for n in (1, 2):
        g.add_node(n, x=float(n), y=48.0, elevation=100.0)
    g.add_edge(1, 2, key=0, **_edge(Mode.BIKE, 1500.0, surface="asphalt", highway="residential"))
    bike_only = format_composition(comp=route_composition(graph=g, node_path=[1, 2]))
    assert "Mode:" not in bike_only
    assert "paved: 1.5 km" in bike_only
