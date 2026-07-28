"""Coverage for HF download, multi-region id-shift, bbox open, and coverage guard."""

import pandas as pd
import pytest

from bike_router import graph_store, pipeline
from bike_router.builder import _open_osm, reindex_region, remap_contiguous
from bike_router.errors import OutOfCoverageError


def test_download_graph_skips_when_present(tmp_path):
    (tmp_path / "meta.json").write_text("{}")
    # No snapshot_download call should happen when meta.json already exists.
    assert graph_store.download_graph_from_hf(target_dir=tmp_path) == tmp_path


def test_download_graph_pulls_when_missing(tmp_path, monkeypatch):
    called = {}

    def _fake_list(repo_id, repo_type):  # noqa: ANN001, ANN202
        called["repo"] = repo_id
        return ["meta.json", "nodes/tile_0_0.parquet"]

    def _fake_download(repo_id, repo_type, filename, local_dir):  # noqa: ANN001, ANN202
        (tmp_path / "meta.json").write_text("{}")  # simulate meta landing

    monkeypatch.setattr(graph_store, "list_repo_files", _fake_list)
    monkeypatch.setattr(graph_store, "hf_hub_download", _fake_download)
    result = graph_store.download_graph_from_hf(target_dir=tmp_path)
    assert result == tmp_path
    assert called["repo"] == graph_store.GraphConfig.HF_REPO_ID


def test_remap_contiguous_renumbers_gapped_and_negative_ids():
    # osmnx leaves gapped 0-based ids + our station code adds negatives; remap → dense 0..N-1.
    nodes = pd.DataFrame(
        {
            "osmid": [-2, 0, 5, 40],  # gapped + negative
            "lat": [48.0, 48.1, 48.2, 48.3],
            "lon": [8.0, 8.1, 8.2, 8.3],
            "elevation_m": [0.0, 0.0, 0.0, 0.0],
            "node_type": ["rail", "bike", "bike", "bike"],
            "station_name": ["S", None, None, None],
        }
    )
    edges = pd.DataFrame({"from_node": [-2, 5], "to_node": [0, 40]})
    n2, e2 = remap_contiguous(nodes_df=nodes, edges_df=edges)
    assert sorted(n2["osmid"]) == [0, 1, 2, 3]  # dense, contiguous, no negatives
    assert n2["osmid"].max() == len(n2) - 1  # n_nodes == max_id + 1
    # edges follow the same mapping: -2→0, 0→1, 5→2, 40→3
    assert list(e2["from_node"]) == [0, 2] and list(e2["to_node"]) == [1, 3]


def test_reindex_region_pure_offset():
    nodes = pd.DataFrame({"osmid": [0, 1, 2]})
    edges = pd.DataFrame({"from_node": [0, 1], "to_node": [1, 2]})
    n2, e2 = reindex_region(nodes_df=nodes, edges_df=edges, offset=100)
    assert list(n2["osmid"]) == [100, 101, 102]
    assert list(e2["from_node"]) == [100, 101] and list(e2["to_node"]) == [101, 102]


def test_open_osm_returns_osm():
    import pyrosm

    osm = _open_osm(pbf_path=pyrosm.get_data("test_pbf"))
    assert osm is not None  # bbox clipping moved upstream to osmium (stage_pbf); _open_osm just opens


def test_assert_within_coverage_passes_and_fails(tmp_path):
    (tmp_path / "meta.json").write_text('{"bbox": [7.9, 47.9, 8.2, 48.2], "tile_deg": 0.5}')
    # (lat, lon) endpoints inside the bbox → no raise
    pipeline._assert_within_coverage(start_latlon=(48.0, 8.0), dest_latlon=(48.1, 8.1), graph_dir=tmp_path)
    with pytest.raises(OutOfCoverageError, match="coverage"):
        pipeline._assert_within_coverage(start_latlon=(48.0, 8.0), dest_latlon=(60.0, 20.0), graph_dir=tmp_path)
