"""Route-composition tests — km tallied by surface, road class, and travel mode."""

from bike_router.composition import format_composition, route_composition
from bike_router.constants import Mode
from tests.conftest import make_composition_graph, make_line_graph


def test_composition_tallies_surface_road_and_mode():
    comp = route_composition(graph=make_composition_graph(), node_path=[1, 2, 3, 4, 5])
    # surface / road describe bike legs only (1 km paved + 2 km gravel)
    assert comp.by_surface_km["paved"] == 1.0
    assert comp.by_surface_km["gravel/unpaved"] == 2.0
    assert comp.by_road_km["quiet way"] == 1.0
    assert comp.by_road_km["main road"] == 2.0
    # mode covers the whole route
    assert comp.by_mode_km[Mode.BIKE] == 3.0
    assert comp.by_mode_km[Mode.STATION] == 0.1
    assert comp.by_mode_km[Mode.RAIL] == 10.0


def test_format_composition_is_percentages_with_mode_always_shown():
    # 3 km bike + 0.1 km station + 10 km rail = 13.1 km total → rail ≈ 76%.
    with_rail = format_composition(comp=route_composition(graph=make_composition_graph(), node_path=[1, 2, 3, 4, 5]))
    assert "Surface:" in with_rail and "Roads:" in with_rail and "Mode:" in with_rail
    assert "rail: 76%" in with_rail  # train ratio must be shown, as a percent
    assert "paved: 33%" in with_rail and "gravel/unpaved: 67%" in with_rail
    assert "km" not in with_rail  # percentages only, never raw km

    # bike-only route (the shared all-paved/quiet line graph) → Mode: still shown (bike: 100%).
    bike_only = format_composition(comp=route_composition(graph=make_line_graph(), node_path=[1, 2, 3]))
    assert "Mode:" in bike_only and "bike: 100%" in bike_only
    assert "paved: 100%" in bike_only
