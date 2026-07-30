"""Route-composition tests — km tallied by surface, road class, and travel mode."""

from bike_router.core.composition import format_composition, route_composition
from tests.conftest import make_composition_route, make_line_route


def test_composition_tallies_surface_road_and_mode():
    comp = route_composition(route=make_composition_route())
    # surface / road describe bike legs only (1 km paved + 2 km gravel)
    assert comp.by_surface_km["paved road"] == 1.0
    assert comp.by_surface_km["unpaved path"] == 2.0
    assert comp.by_road_km["quiet way"] == 1.0
    assert comp.by_road_km["main road"] == 2.0
    # mode = two display buckets over the whole route; the 0.1 km station hop folds into bike.
    assert comp.by_mode_km["bike route"] == 3.1  # 3 km pedalled + 0.1 km station access
    assert comp.by_mode_km["train path"] == 10.0
    assert "station" not in comp.by_mode_km  # station is never its own bucket


def test_format_composition_is_percentages_with_mode_always_shown():
    # 3.1 km bike-bucket + 10 km train = 13.1 km total → train ≈ 76%.
    with_rail = format_composition(comp=route_composition(route=make_composition_route()))
    assert "Surface:" in with_rail and "Roads:" in with_rail and "Mode:" in with_rail
    assert "train path: 76%" in with_rail  # train ratio must be shown, as a percent
    assert "paved road: 33%" in with_rail and "unpaved path: 67%" in with_rail
    assert "km" not in with_rail  # percentages only, never raw km

    # bike-only route (the shared all-paved/quiet line graph) → Mode: still shown.
    bike_only = format_composition(comp=route_composition(route=make_line_route()))
    assert "Mode:" in bike_only and "bike route: 100%" in bike_only
    assert "paved road: 100%" in bike_only
