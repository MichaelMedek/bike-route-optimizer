"""Route-composition tests — km tallied by surface, road class, and travel mode.

TestRouteComposition covers the dataclass; test_<fn> per module function (route_composition,
_percent_lines, format_composition), each folding its full contract.
"""

from bike_router.core.composition import RouteComposition, _percent_lines, format_composition, route_composition
from tests.conftest import make_composition_route, make_line_route


class TestRouteComposition:
    def test_holds_three_independent_km_breakdowns(self):
        # by_surface/by_road slice the pedalled legs; by_mode covers the whole route (bike vs train).
        comp = RouteComposition(
            by_surface_km={"paved road": 1.0}, by_road_km={"quiet way": 1.0}, by_mode_km={"bike route": 1.0}
        )
        assert comp.by_surface_km["paved road"] == 1.0
        assert comp.by_road_km["quiet way"] == 1.0
        assert comp.by_mode_km["bike route"] == 1.0


def test_route_composition():
    # Tallies bike km by surface tier + road class (pedalled legs only) and whole-route km by mode;
    # the negligible station-access hop folds into "bike route", never its own bucket.
    comp = route_composition(route=make_composition_route())
    assert comp.by_surface_km["paved road"] == 1.0 and comp.by_surface_km["unpaved path"] == 2.0
    assert comp.by_road_km["quiet way"] == 1.0 and comp.by_road_km["main road"] == 2.0
    assert comp.by_mode_km["bike route"] == 3.1  # 3 km pedalled + 0.1 km station access
    assert comp.by_mode_km["train path"] == 10.0
    assert "station" not in comp.by_mode_km


def test_percent_lines():
    # Formats a km breakdown as "  label: NN%" of the category total; default order is descending km.
    lines = _percent_lines(by_km={"a": 3.0, "b": 1.0})
    assert lines == ["  a: 75%", "  b: 25%"]  # 3/4 and 1/4, biggest first
    # an explicit order map overrides the descending-km default
    ordered = _percent_lines(by_km={"a": 3.0, "b": 1.0}, order={"b": 0, "a": 1})
    assert ordered == ["  b: 25%", "  a: 75%"]


def test_format_composition():
    # Percentage summary with Surface/Roads/Mode headers; mode is ALWAYS shown; never raw km.
    with_rail = format_composition(comp=route_composition(route=make_composition_route()))
    assert "Surface:" in with_rail and "Roads:" in with_rail and "Mode:" in with_rail
    assert "train path: 76%" in with_rail  # 10 / 13.1 ≈ 76%
    assert "paved road: 33%" in with_rail and "unpaved path: 67%" in with_rail
    assert "km" not in with_rail  # percentages only
    bike_only = format_composition(comp=route_composition(route=make_line_route()))
    assert "Mode:" in bike_only and "bike route: 100%" in bike_only and "paved road: 100%" in bike_only
