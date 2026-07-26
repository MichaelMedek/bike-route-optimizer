"""Debug-heatmap plotting test — renders to a temp PNG headlessly."""

from pathlib import Path

from bike_router.constants import PlotConfig
from bike_router.plotting import _figsize_for_route, plot_elevation_heatmap
from bike_router.track import build_track
from tests.conftest import DEFAULT_PARAMS, make_line_graph


def test_figsize_matches_route_aspect_for_any_orientation():
    """E-W routes give a wide page, N-S a tall page; both stay positive and floored."""
    ew_w, ew_h, ew_map = _figsize_for_route(route_lons=[8.0, 8.6], route_lats=[48.5, 48.5001])
    ns_w, ns_h, ns_map = _figsize_for_route(route_lons=[8.5, 8.5001], route_lats=[48.2, 48.9])
    assert ew_w > ew_h and ns_h > ns_w  # orientation drives the page shape
    assert min(ew_w, ew_h, ns_w, ns_h) > 0  # every dimension usable
    assert ew_map >= PlotConfig.MAP_SHORT_MIN_IN and ns_map == PlotConfig.MAP_LONG_IN  # floor + long side


def test_figsize_degenerate_span_does_not_divide_by_zero():
    """A single-latitude route (lat_span 0) must not crash on the aspect division."""
    w, h, map_h = _figsize_for_route(route_lons=[8.0, 8.2], route_lats=[48.5, 48.5])
    assert w > 0 and h > 0 and map_h > 0


def test_plot_writes_png(tmp_path: Path):
    graph = make_line_graph()
    track = build_track(graph=graph, node_path=[1, 2, 3])
    out_path = tmp_path / "heatmap.png"
    plot_elevation_heatmap(
        graph=graph, route_nodes=[1, 2, 3], track=track, params=DEFAULT_PARAMS, out_path=str(out_path), dpi=50
    )
    assert out_path.exists() and out_path.stat().st_size > 0


def test_plot_handles_uniform_elevation(tmp_path: Path):
    graph = make_line_graph()
    track = build_track(graph=graph, node_path=[1, 2, 3])
    for node in graph.nodes:
        graph.nodes[node]["elevation"] = 500.0  # vmin == vmax branch
    out_path = tmp_path / "flat.png"
    plot_elevation_heatmap(
        graph=graph, route_nodes=[1, 2, 3], track=track, params=DEFAULT_PARAMS, out_path=str(out_path), dpi=50
    )
    assert out_path.exists()
