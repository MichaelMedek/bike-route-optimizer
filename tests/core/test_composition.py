"""composition tests — the THREE donut breakdowns (quality/grade/mode), the ONE shared source.

Built from the Track and consumed identically by the donuts, the PNG overlay, and the CLI text.
One test_<fn> per module symbol; each folds its full contract. Km is great-circle over the track
legs (the same basis the donuts always used), so tests assert labels + proportions, not raw km.
"""

from bike_router.core.composition import (
    MODE_COLORS,
    QUALITY_COLORS,
    RouteComposition,
    _leg_km_by,
    _mode_bucket,
    _ordered,
    _percent_lines,
    _quality_bucket,
    composition_rows,
    format_composition,
    route_composition,
)
from bike_router.core.track import build_track
from tests.conftest import make_composition_route, make_line_route


def _pct(by_km: dict[str, float], label: str) -> float:
    """Percent of a breakdown's total that ``label`` holds (for proportion asserts)."""
    return by_km[label] / sum(by_km.values()) * 100


class TestRouteComposition:
    def test_holds_three_donut_breakdowns(self):
        # by_quality/by_grade slice the pedalled legs; by_mode covers the whole route (bike vs train).
        comp = RouteComposition(by_quality_km={"good": 1.0}, by_grade_km={"flat": 1.0}, by_mode_km={"bike route": 1.0})
        assert comp.by_quality_km["good"] == 1.0
        assert comp.by_grade_km["flat"] == 1.0
        assert comp.by_mode_km["bike route"] == 1.0


def test_leg_km_by():
    # Great-circle km per bucket over the track legs; bike_only skips rail/station points.
    track = build_track(route=make_composition_route())
    bike_only = _leg_km_by(track=track, bucket_of=lambda p: "all", bike_only=True)
    everything = _leg_km_by(track=track, bucket_of=lambda p: "all", bike_only=False)
    assert set(bike_only) == {"all"} and set(everything) == {"all"}
    assert everything["all"] > bike_only["all"]  # the whole route (incl. rail/station) is longer


def test_quality_bucket():
    # A pedalled point's quality label; main-road (± unpaved) folds into the dominant "main road".
    track = build_track(route=make_composition_route())
    good_pt = track.points[1]  # arrived via the paved/quiet bike edge
    main_pt = track.points[2]  # arrived via the gravel/secondary edge (unpaved + main → main road)
    assert _quality_bucket(good_pt) == "good"
    assert _quality_bucket(main_pt) == "main road"


def test_mode_bucket():
    # Any point → "train path" for a rail leg, "bike route" otherwise (station folds into bike route).
    track = build_track(route=make_composition_route())
    assert _mode_bucket(track.points[1]) == "bike route"  # bike leg
    assert _mode_bucket(track.points[3]) == "bike route"  # station access hop
    assert _mode_bucket(track.points[4]) == "train path"  # rail leg


def test_route_composition():
    # The three breakdowns from one walk: quality good+main (bike legs), grade all flat (no climb),
    # mode bike-route + train-path (station folds into bike route, never its own bucket).
    comp = route_composition(track=build_track(route=make_composition_route()))
    assert set(comp.by_quality_km) == {"good", "main road"}  # the two pedalled leg qualities
    assert set(comp.by_grade_km) == {"flat"}  # elevations all 100 m → flat
    assert set(comp.by_mode_km) == {"bike route", "train path"}
    assert "station" not in comp.by_mode_km
    assert _pct(comp.by_mode_km, "train path") > 0  # the rail leg is counted


def test_composition_rows():
    # The THREE rows (title, km, hex colours) — the one shared source for donuts + PNG + CLI. Fixed
    # titles Quality/Grade/Mode; colours come from the Palette-derived maps; only present labels shown.
    rows = composition_rows(comp=route_composition(track=build_track(route=make_composition_route())))
    assert [title for title, _km, _colors in rows] == ["Quality", "Grade", "Mode"]
    quality_km, quality_colors = rows[0][1], rows[0][2]
    assert quality_colors is QUALITY_COLORS and rows[2][2] is MODE_COLORS
    # quality labels appear in the fixed best→worst order (good before main road)
    assert list(quality_km) == ["good", "main road"]


def test_ordered():
    # Re-keys a breakdown into a fixed label order; labels absent from the route are dropped.
    got = _ordered({"main road": 2.0, "good": 1.0}, {"good": 0, "unpaved": 1, "main road": 2})
    assert list(got) == ["good", "main road"]  # sorted by the order map; "unpaved" absent → dropped


def test_percent_lines():
    # Formats a km breakdown as "  label: NN%" of the total, in the dict's existing order.
    assert _percent_lines(by_km={"good": 3.0, "main road": 1.0}) == ["  good: 75%", "  main road: 25%"]


def test_format_composition():
    # The three donut rows as percentage text — SAME Quality/Grade/Mode titles as the donuts, never
    # raw km; mode always present so the train ratio is never hidden.
    comp = route_composition(track=build_track(route=make_composition_route()))
    text = format_composition(comp=comp)
    assert "Quality:" in text and "Grade:" in text and "Mode:" in text
    assert "Surface:" not in text and "Roads:" not in text  # the OLD drifted headers are gone
    assert "train path:" in text and "%" in text and "km" not in text

    bike_only = format_composition(comp=route_composition(track=build_track(route=make_line_route())))
    assert "bike route: 100%" in bike_only and "good: 100%" in bike_only  # all-good pure-bike route
