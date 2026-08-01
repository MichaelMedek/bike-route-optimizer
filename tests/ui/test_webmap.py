"""webmap tests — pure Streamlit-map helpers PLUS the app_webmap shell driven by AppTest.

One test_<fn> per production symbol (exact-name mirror) and a TestFoo per dataclass; the pure
helpers build Altair/Plotly/pydeck objects directly (no browser). The app_webmap integration
tests run the REAL app headlessly via streamlit's AppTest against the committed fixture graph.
"""

import math
from types import SimpleNamespace

import pytest

from bike_router.core.composition import MODE_COLORS
from bike_router.core.constants import Mode, Palette, WebMapConfig
from bike_router.core.geo import haversine_distance_m
from bike_router.core.track import build_track
from bike_router.ui.webmap import (
    GRADE_SCALE,
    QUALITY_SCALE,
    RibbonSegment,
    ViewState,
    _hex,
    _named_waypoints,
    _point_color,
    _segment_tooltip,
    _station_marker_points,
    composition_donut,
    compute_gate,
    default_view_state,
    elevation_profile_chart,
    endpoint_labels,
    flattened_view,
    map_remount_key,
    map_waypoint_markers,
    output_donuts,
    output_stat_rows,
    picked_station_name,
    profile_markers,
    ribbon_width_m,
    route_ribbon_segments,
    route_view_state,
    scale_label,
    station_click_pending,
    swapped_endpoint_state,
    zoom_for_span_m,
)
from tests.conftest import make_line_route, make_rail_route


def _rgb(hex_color: str) -> list[int]:
    """Palette hex → RGB list, for comparing against segment_color output."""
    return list(Palette.hex_to_rgb(hex_color=hex_color))


def _line_track():
    """A computed Track over the flat line route (all-bike, elevations 100–130 m)."""
    return build_track(route=make_line_route())


# --- colour / km breakdowns --------------------------------------------------


def test_hex():
    # RGB tuple → #rrggbb, zero-padded.
    assert _hex(rgb=(0x15, 0x65, 0xC0)) == "#1565c0"
    assert _hex(rgb=(0, 0, 0)) == "#000000"


def test_composition_donut():
    # A small Altair donut of a km breakdown; the colour domain is exactly the given labels; an
    # empty breakdown fails loud.
    chart = composition_donut(
        title="By quality", by_km={"good": 3.0, "unpaved": 1.0}, colors={"good": "#00f", "unpaved": "#f80"}
    )
    spec = chart.to_dict()
    assert spec["title"] == "By quality"
    assert set(spec["encoding"]["color"]["scale"]["domain"]) == {"good", "unpaved"}
    with pytest.raises(AssertionError, match="non-empty"):
        composition_donut(title="x", by_km={}, colors={})


# --- elevation profile -------------------------------------------------------


def test_elevation_profile_chart():
    # distance × elevation, line tinted by mode with the SAME colours as the "By mode" donut; the
    # all-bike line rises 100→130→100 → one bike-blue trace; given markers add a labelled scatter.
    track = _line_track()
    fig = elevation_profile_chart(track=track, markers=None)
    assert len(fig.data) == 1  # single contiguous bike run
    trace = fig.data[0]
    assert trace.name == WebMapConfig.MODE_DONUT_LABELS[Mode.BIKE]
    assert trace.line.color == MODE_COLORS[WebMapConfig.MODE_DONUT_LABELS[Mode.BIKE]]  # SAME as the "Mode" donut
    assert max(trace.y) == pytest.approx(130.0) and min(trace.y) == pytest.approx(100.0)
    assert trace.x[0] == pytest.approx(0.0) and trace.x[-1] > 0  # cumulative distance
    lo, hi = fig.layout.yaxis.range
    assert lo < 100.0 and hi > 130.0  # padded range, not a flat/auto axis

    marked = elevation_profile_chart(track=track, markers=[(0.0, 100.0, "Start"), (1.6, 100.0, "End")])
    marker_trace = next(t for t in marked.data if t.mode and "markers" in t.mode)
    assert list(marker_trace.text) == ["Start", "End"]
    assert marker_trace.x[0] == pytest.approx(0.0) and marker_trace.x[-1] == pytest.approx(1.6)


# --- dataclasses -------------------------------------------------------------


class TestRibbonSegment:
    def test_holds_color_width_points_tooltip(self):
        seg = RibbonSegment(color=[1, 2, 3], width_m=20.0, points=[[8.0, 48.0, 1.0]], tooltip="t")
        assert seg.color == [1, 2, 3] and seg.width_m == 20.0 and seg.tooltip == "t"

    def test_is_frozen(self):
        with pytest.raises(AttributeError):
            RibbonSegment(color=[1, 2, 3], width_m=1.0, points=[], tooltip="t").width_m = 2.0  # type: ignore[misc]


class TestViewState:
    def test_holds_camera_pose(self):
        view = ViewState(latitude=48.0, longitude=8.0, zoom=10.0, pitch=45.0, bearing=0.0)
        assert (view.latitude, view.longitude, view.zoom, view.pitch, view.bearing) == (48.0, 8.0, 10.0, 45.0, 0.0)

    def test_is_frozen(self):
        with pytest.raises(AttributeError):
            ViewState(latitude=48.0, longitude=8.0, zoom=1.0, pitch=0.0, bearing=0.0).zoom = 2.0  # type: ignore[misc]


# --- ribbon ------------------------------------------------------------------


def test_point_color():
    # QUALITY scale → segment_color; GRADE scale → grade_color; an unknown scale raises.
    from bike_router.core.track import grade_color, segment_color

    pt = _line_track().points[1]
    assert _point_color(point=pt, scale=QUALITY_SCALE) == segment_color(
        mode=pt.mode, surface_bad=pt.surface_bad, road_bad=pt.road_bad
    )
    assert _point_color(point=pt, scale=GRADE_SCALE) == grade_color(mode=pt.mode, grade=pt.grade)
    with pytest.raises(ValueError, match="unknown route colour scale"):
        _point_color(point=pt, scale="rainbow")


def test_segment_tooltip():
    # Names surface · road · signed gradient direction · est. speed · elevation for a pedalled segment.
    uphill = _line_track().points[1]  # climbs 1→2
    tip = _segment_tooltip(point=uphill)
    assert "paved" in tip and "quiet way" in tip and "km/h" in tip
    assert "uphill" in tip or "downhill" in tip or "flat" in tip
    # the trailing elevation, same "NNN m" format the markers hover — the edge's height
    assert tip.rstrip().endswith(f"{uphill.elevation_m:.0f} m")


def test_ribbon_width_m():
    # Water-in-a-pipe: width ∝ 1/√speed, anchored REF_SPEED → REF_WIDTH; 4× slower → 2× wider.
    ref_speed, ref_width = WebMapConfig.RIBBON_REF_SPEED_KMH, WebMapConfig.RIBBON_REF_WIDTH_M
    assert ribbon_width_m(speed_kmh=ref_speed) == pytest.approx(ref_width)
    assert ribbon_width_m(speed_kmh=ref_speed / 4) == pytest.approx(ref_width * 2)
    assert ribbon_width_m(speed_kmh=ref_speed * 4) == pytest.approx(ref_width / 2)
    with pytest.raises(AssertionError, match="speed must be positive"):
        ribbon_width_m(speed_kmh=0.0)


def test_route_ribbon_segments():
    # QUALITY: all-bike asphalt/quiet → every run blue, z lifted + finite, width ∝ 1/√speed (slow
    # uphill wider). GRADE: the climb-then-descend shows distinct red (up) + green (down) runs.
    # Rail/station runs draw the FIXED reference width. A custom float offset lifts every point.
    track = _line_track()
    segments = route_ribbon_segments(
        track=track, float_above_m=WebMapConfig.RIBBON_FLOAT_ABOVE_M, rail_tooltips=None, color_scale=QUALITY_SCALE
    )
    assert segments
    for seg in segments:
        assert seg.color == _rgb(Palette.BLUE) and seg.width_m > 0
        for lon, lat, _z in seg.points:
            assert math.isfinite(lon) and math.isfinite(lat)
    widths = sorted(seg.width_m for seg in segments)
    assert len(set(widths)) >= 2  # uphill vs flat/downhill differ
    fastest, slowest = max(p.speed_kmh for p in track.points), min(p.speed_kmh for p in track.points)
    assert slowest < fastest
    assert min(widths) == pytest.approx(ribbon_width_m(speed_kmh=fastest))
    assert max(widths) == pytest.approx(ribbon_width_m(speed_kmh=slowest))

    grade_colors = {
        tuple(seg.color)
        for seg in route_ribbon_segments(
            track=track, float_above_m=WebMapConfig.RIBBON_FLOAT_ABOVE_M, rail_tooltips=None, color_scale=GRADE_SCALE
        )
    }
    assert tuple(_rgb(Palette.RED)) in grade_colors and tuple(_rgb(Palette.GREEN)) in grade_colors

    tips = " | ".join(
        seg.tooltip
        for seg in route_ribbon_segments(
            track=track, float_above_m=WebMapConfig.RIBBON_FLOAT_ABOVE_M, rail_tooltips=None, color_scale=QUALITY_SCALE
        )
    )
    assert "paved" in tips and "quiet way" in tips and "km/h" in tips

    rail = build_track(route=make_rail_route())
    non_bike = [
        seg
        for seg in route_ribbon_segments(
            track=rail, float_above_m=WebMapConfig.RIBBON_FLOAT_ABOVE_M, rail_tooltips=None, color_scale=QUALITY_SCALE
        )
        if seg.color != _rgb(Palette.BLUE)
    ]
    assert non_bike and all(seg.width_m == WebMapConfig.RIBBON_REF_WIDTH_M for seg in non_bike)

    lifted = route_ribbon_segments(track=track, float_above_m=250.0, rail_tooltips=None, color_scale=QUALITY_SCALE)[
        0
    ].points
    assert all(math.isfinite(z) and z > 0 for *_, z in lifted)
    assert lifted[0][2] == pytest.approx(track.points[0].elevation_m + 250.0)


# --- camera ------------------------------------------------------------------


def test_zoom_for_span_m():
    # Log2 anchor: halving the span adds one zoom level; clamped to the ±steps window; span > 0.
    assert zoom_for_span_m(span_m=WebMapConfig.ZOOM_SPAN_ANCHOR_M) == pytest.approx(WebMapConfig.VIEWING_ZOOM)
    assert zoom_for_span_m(span_m=WebMapConfig.ZOOM_SPAN_ANCHOR_M / 2) == pytest.approx(WebMapConfig.VIEWING_ZOOM + 1)
    assert zoom_for_span_m(span_m=1.0) == WebMapConfig.VIEWING_ZOOM + WebMapConfig.ZOOM_STEPS_IN
    assert zoom_for_span_m(span_m=1e9) == WebMapConfig.VIEWING_ZOOM - WebMapConfig.ZOOM_STEPS_OUT
    with pytest.raises(AssertionError, match="span must be positive"):
        zoom_for_span_m(span_m=0.0)


def test_default_view_state():
    # The opening camera: the far-out DACH overview (not the closer route zoom), north-up, tilted.
    assert default_view_state() == ViewState(
        latitude=WebMapConfig.DEFAULT_LAT,
        longitude=WebMapConfig.DEFAULT_LON,
        zoom=WebMapConfig.DEFAULT_ZOOM,
        pitch=WebMapConfig.DEFAULT_PITCH,
        bearing=WebMapConfig.DEFAULT_BEARING,
    )


def test_route_view_state():
    # Centres on the start→end midpoint; zoom derives from the direct-line span; north-up, tilted.
    start, end = (48.0, 8.0), (48.4, 8.6)
    view = route_view_state(start_latlon=start, end_latlon=end)
    assert view.latitude == pytest.approx(48.2) and view.longitude == pytest.approx(8.3)
    assert view.bearing == WebMapConfig.DEFAULT_BEARING and view.pitch == WebMapConfig.DEFAULT_PITCH
    span_m = haversine_distance_m(lat_a=start[0], lon_a=start[1], lat_b=end[0], lon_b=end[1])
    assert view.zoom == pytest.approx(zoom_for_span_m(span_m=span_m))


# --- shell-decision logic ----------------------------------------------------


def test_compute_gate():
    # Compute enabled ONLY when endpoints are set AND both boxes still hold the resolved text;
    # the three states map to distinct help strings.
    unset, msg = compute_gate(start_latlon=None, origin="A", destination="B", start_resolved="A", end_resolved="B")
    assert unset is False and "Set a start" in msg
    changed, msg = compute_gate(
        start_latlon=(48.0, 8.0), origin="A2", destination="B", start_resolved="A", end_resolved="B"
    )
    assert changed is False and "again" in msg
    ready, msg = compute_gate(
        start_latlon=(48.0, 8.0), origin="A", destination="B", start_resolved="A", end_resolved="B"
    )
    assert ready is True and "Plan the route" in msg


def test_endpoint_labels():
    # (start, end) "Name (elev m)" labels; None when either endpoint is unset.
    labels = endpoint_labels(
        start_latlon=(48.0, 8.0, 300.0), end_latlon=(48.4, 8.6, 500.0), origin="Freudenstadt", destination="Pforzheim"
    )
    assert labels == ("Freudenstadt (300 m)", "Pforzheim (500 m)")
    assert endpoint_labels(start_latlon=None, end_latlon=(48.4, 8.6, 500.0), origin="A", destination="B") is None


def test_map_remount_key():
    # Keyed on camera_epoch (bumped by Set start & end) AND the top-down flag (top-stations toggle),
    # so the map remounts to move the camera or flip pitch; a colour-scale toggle repaints in place.
    assert map_remount_key(camera_epoch=3, top_down=False) == "bike_map_3_tilted"
    assert map_remount_key(camera_epoch=0, top_down=True) == "bike_map_0_topdown"
    assert map_remount_key(camera_epoch=1, top_down=False) != map_remount_key(camera_epoch=2, top_down=False)
    # flipping top-down (pitch change) must remount so st_deckgl applies the new pose
    assert map_remount_key(camera_epoch=1, top_down=False) != map_remount_key(camera_epoch=1, top_down=True)


def test_flattened_view():
    # Forces pitch to 0 (reliable deck.gl picking) while keeping every other camera field.
    view = ViewState(latitude=48.0, longitude=8.0, zoom=10.0, pitch=45.0, bearing=20.0)
    flat = flattened_view(view)
    assert flat.pitch == 0.0
    assert (flat.latitude, flat.longitude, flat.zoom, flat.bearing) == (48.0, 8.0, 10.0, 20.0)


def test_flattened_view_is_idempotent():
    # An already-top-down camera is returned unchanged (equal), so re-flattening never remounts.
    flat = ViewState(latitude=47.0, longitude=9.0, zoom=8.0, pitch=0.0, bearing=0.0)
    assert flattened_view(flat) == flat


def test_scale_label():
    # Human label for each route colour scale (the radio's format_func).
    assert "quality" in scale_label(scale=QUALITY_SCALE).lower()
    assert "grade" in scale_label(scale=GRADE_SCALE).lower()


def test_output_stat_rows():
    # A train route → bike-vs-total split (two rows); a pure-bike route → one "Route".
    bike_only = SimpleNamespace(track=_line_track(), rail_legs=[])
    rows = output_stat_rows(result=bike_only)
    assert len(rows) == 1 and rows[0][0] == "**Route**"

    with_train = SimpleNamespace(track=build_track(route=make_rail_route()), rail_legs=["a-ride"])
    rows = output_stat_rows(result=with_train)
    assert [r[0] for r in rows] == ["**Total** (bike + train)", "**Bike only**"]


def test_output_donuts():
    # Delegates to core composition_rows → the THREE rows (Quality/Grade/Mode) the PNG + CLI also use.
    from bike_router.core.composition import route_composition

    result = SimpleNamespace(composition=route_composition(track=_line_track()))
    donuts = output_donuts(result=result)
    assert [title for title, _km, _colors in donuts] == ["Quality", "Grade", "Mode"]
    assert "bike route" in donuts[2][1]  # the all-bike line route's mode bucket


def test_profile_markers():
    # Labels endpoints from the typed names + stations; interior waypoints appear ONLY when named.
    result = SimpleNamespace(track=_line_track(), rail_legs=[], waypoints=[(48.0, 8.01)])
    named = profile_markers(
        result=result,
        start_latlon=(48.0, 8.0, 100.0),
        end_latlon=(48.0, 8.02, 100.0),
        start_name="Freudenstadt",
        end_name="Pforzheim",
        village_of=lambda lat, lon: "Baiersbronn",
    )
    assert [lab for _d, _e, lab in named] == ["Freudenstadt", "Pforzheim", "Baiersbronn"]

    dropped = profile_markers(
        result=result,
        start_latlon=(48.0, 8.0, 100.0),
        end_latlon=(48.0, 8.02, 100.0),
        start_name="A",
        end_name="B",
        village_of=lambda lat, lon: None,
    )
    assert [lab for _d, _e, lab in dropped] == ["A", "B"]  # unnamed waypoint DROPPED from the profile


def test_station_marker_points():
    # Delegates to the core single source: (lat, lon, elev, label) per boarded/alighted station.
    from bike_router.core.simplify import RailLeg, Station

    leg = RailLeg(
        board=Station(name="A", lat=48.0, lon=8.0, elevation_m=100.0),
        alight=Station(name="B", lat=48.1, lon=8.1, elevation_m=200.0),
    )
    result = SimpleNamespace(rail_legs=[leg])
    points = _station_marker_points(result=result)
    assert [lab for _lat, _lon, _e, lab in points] == ["A (100 m)", "B (200 m)"]
    assert _station_marker_points(result=SimpleNamespace(rail_legs=[])) == []


def test_map_waypoint_markers():
    # The map's intermediate markers = stations + gmaps waypoints as (lat, lon, elev, label). A
    # waypoint's elevation snaps to its nearest track point; ONLY NAMED waypoints appear (unnamed
    # ones are dropped); each label is the shared "Name (elev m)".
    track = _line_track()  # nodes at lon 8.00/8.01/8.02, elevations 100/130/100 m
    result = SimpleNamespace(track=track, rail_legs=[], waypoints=[(48.0, 8.01), (48.0, 8.02)])
    markers = map_waypoint_markers(result=result, village_of=lambda lat, lon: "Baiersbronn" if lon == 8.01 else None)
    assert len(markers) == 1  # only the NAMED 8.01 waypoint; the unnamed 8.02 is dropped
    assert markers[0][:2] == (48.0, 8.01) and markers[0][2] == pytest.approx(130.0)  # snapped to node 2
    assert markers[0][3] == "Baiersbronn (130 m)"  # named village

    # stations come first (from _station_marker_points), then the named waypoints
    from bike_router.core.simplify import RailLeg, Station

    leg = RailLeg(
        board=Station(name="A", lat=48.0, lon=8.0, elevation_m=100.0),
        alight=Station(name="B", lat=48.0, lon=8.02, elevation_m=100.0),
    )
    with_rail = SimpleNamespace(track=track, rail_legs=[leg], waypoints=[(48.0, 8.01)])
    labels = [lab for _lat, _lon, _e, lab in map_waypoint_markers(result=with_rail, village_of=lambda lat, lon: "V")]
    assert labels == ["A (100 m)", "B (100 m)", "V (130 m)"]  # stations, then the named waypoint


def test_named_waypoints():
    # (lat, lon, name) for ONLY the waypoints that reverse-geocode to a real name; unnamed dropped.
    wps = [(48.0, 8.01), (48.0, 8.02), (48.0, 8.03)]
    named = _named_waypoints(waypoints=wps, village_of=lambda lat, lon: "Baiersbronn" if lon != 8.02 else None)
    assert named == [(48.0, 8.01, "Baiersbronn"), (48.0, 8.03, "Baiersbronn")]  # 8.02 (None) dropped
    assert _named_waypoints(waypoints=wps, village_of=lambda lat, lon: None) == []  # none named → empty
    assert _named_waypoints(waypoints=[], village_of=lambda lat, lon: "X") == []  # no waypoints → empty


def test_picked_station_name():
    # st_deckgl spreads the picked datum at TOP LEVEL with deck.gl's OWN eventType
    # ("deck-click-event", NOT "click"); a top-station click with a name yields it, everything
    # else (other event, no name, blank, non-dict) yields None.
    click = WebMapConfig.DECK_CLICK_EVENT
    assert picked_station_name({"name": "Freudenstadt Stadt", "eventType": click}) == "Freudenstadt Stadt"
    assert picked_station_name({"tooltip": "route seg", "eventType": click}) is None  # no name (route/waypoint)
    assert picked_station_name({"name": "X", "eventType": "click"}) is None  # raw "click" is NOT the deck event
    assert picked_station_name({"name": "X", "eventType": "deck-hover-event"}) is None  # not a click
    assert picked_station_name({"eventType": click}) is None  # empty-terrain click, no datum
    assert picked_station_name(None) is None  # no event at all
    assert picked_station_name({"name": "", "eventType": click}) is None  # blank name ignored


def test_station_click_pending():
    # A top-station click fills the "<name> Bahnhof" form (bare name → town centre, not the platform);
    # st_deckgl re-returns the last event every rerun, so the SAME pending value dedups to None.
    click = WebMapConfig.DECK_CLICK_EVENT
    event = {"name": "Sauldorf", "eventType": click}
    assert station_click_pending(event=event, last_applied=None) == "Sauldorf Bahnhof"  # Bahnhof appended
    assert station_click_pending(event=event, last_applied="Sauldorf Bahnhof") is None  # re-returned → dedup
    assert station_click_pending(event=event, last_applied="Konstanz Bahnhof") == "Sauldorf Bahnhof"  # a NEW station
    already = {"name": "Sauldorf Bahnhof", "eventType": click}
    assert station_click_pending(event=already, last_applied=None) == "Sauldorf Bahnhof"  # not doubled
    assert station_click_pending(event=None, last_applied=None) is None  # no click at all


def test_swapped_endpoint_state():
    # Swaps box texts, resolved names, and snapped latlons; clears the (now-reversed) route.
    state = {
        "start_box": "A",
        "end_box": "B",
        "start_box_resolved": "A",
        "end_box_resolved": "B",
        "start_latlon": (48.0, 8.0, 100.0),
        "end_latlon": (47.0, 9.0, 200.0),
        "result": "old",
    }
    swapped = swapped_endpoint_state(state)
    assert swapped["start_box"] == "B" and swapped["end_box"] == "A"
    assert swapped["start_box_resolved"] == "B" and swapped["end_box_resolved"] == "A"
    assert swapped["start_latlon"] == (47.0, 9.0, 200.0) and swapped["end_latlon"] == (48.0, 8.0, 100.0)
    assert swapped["result"] is None  # the previous route ran the other direction
