"""plotting tests — the debug route PNG: figure sizing, overlay, marker projection, full render.

One test_<fn> per production symbol (exact-name mirror). Renders headlessly (Agg) to a temp
PNG; the overlay/marker helpers are checked directly on a throwaway Matplotlib axis so no logic
is exercised only through the end-to-end render.
"""

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt

from bike_router.core.constants import Mode, PlotConfig
from bike_router.core.plotting import (
    _draw_route_overlay,
    _figsize_for_route,
    _marker_node_indices,
    _prefs_text,
    plot_route_debug,
)
from bike_router.core.track import build_track
from tests.conftest import DEFAULT_PARAMS, make_line_route, make_mixed_mode_route, zero_params


def test_prefs_text():
    # Left stats column: the five preference lines + bike-only and total (bike+train) route stats.
    track = build_track(route=make_line_route())
    text = _prefs_text(params=zero_params(extra_km_per_uphill_100m=3.0), track=track)
    assert "PREFERENCES (extra km)" in text
    assert "+3    / 100 m uphill" in text  # the value renders in the uphill line
    assert "ROUTE (bike + train)" in text and "ROUTE (bike only)" in text
    assert track.total.oneline in text and track.bike.oneline in text


def test_figsize_for_route():
    # Orientation drives the page shape: E-W → wide, N-S → tall; short side floored, long side fixed;
    # a degenerate (zero-latitude-span) route must not divide by zero.
    ew_w, ew_h, ew_map = _figsize_for_route(route_lons=[8.0, 8.6], route_lats=[48.5, 48.5001])
    ns_w, ns_h, ns_map = _figsize_for_route(route_lons=[8.5, 8.5001], route_lats=[48.2, 48.9])
    assert ew_w > ew_h and ns_h > ns_w
    assert min(ew_w, ew_h, ns_w, ns_h) > 0
    assert ew_map >= PlotConfig.MAP_SHORT_MIN_IN and ns_map == PlotConfig.MAP_LONG_IN

    w, h, map_h = _figsize_for_route(route_lons=[8.0, 8.2], route_lats=[48.5, 48.5])  # lat_span 0
    assert w > 0 and h > 0 and map_h > 0


def test_draw_route_overlay():
    # Draws one coloured polyline per edge; each condition label appears at most once (one legend
    # entry each) — a mixed bike+rail route yields distinct "good"/"train" legend labels.
    route = make_mixed_mode_route([(1, 2, Mode.BIKE), (2, 3, Mode.STATION), (3, 4, Mode.RAIL)])
    _figure, axes = plt.subplots()
    try:
        _draw_route_overlay(axes=axes, route=route)
        labels = [t for t in axes.get_legend_handles_labels()[1]]
        assert len(labels) == len(set(labels))  # each condition labelled once, no duplicate legend rows
        assert len(axes.lines) == 3  # one polyline per edge
    finally:
        plt.close(_figure)


def test_marker_node_indices():
    # Nearest route-node index per (lat, lon) marker, ALWAYS including start (0) and end (last),
    # sorted and de-duplicated.
    route = make_line_route()  # nodes 1,2,3 at lon 8.00/8.01/8.02, lat 48.0
    idxs = _marker_node_indices(route=route, marker_points=[(48.0, 8.01)])  # nearest is node 2 (index 1)
    assert idxs == [0, 1, 2]  # start, the projected middle marker, end — sorted, unique
    assert _marker_node_indices(route=route, marker_points=[]) == [0, 2]  # always start + end


def test_plot_route_debug(tmp_path: Path):
    # Renders the debug PNG headlessly; also covers the uniform-elevation (vmin == vmax) branch.
    route = make_line_route()
    out_path = tmp_path / "route.png"
    plot_route_debug(route=route, track=build_track(route=route), params=DEFAULT_PARAMS, out_path=str(out_path), dpi=50)
    assert out_path.exists() and out_path.stat().st_size > 0

    flat = replace(route, nodes=[replace(node, elevation_m=500.0) for node in route.nodes])  # vmin == vmax
    flat_path = tmp_path / "flat.png"
    plot_route_debug(route=flat, track=build_track(route=flat), params=DEFAULT_PARAMS, out_path=str(flat_path), dpi=50)
    assert flat_path.exists()
