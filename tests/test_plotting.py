"""Debug-heatmap plotting test — renders to a temp PNG headlessly."""

from pathlib import Path

from bike_router.plotting import plot_elevation_heatmap
from bike_router.track import build_track
from tests.conftest import DEFAULT_PARAMS, make_line_graph


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
