"""Coverage for HF download, multi-region id-shift, bbox open, and coverage guard."""

import pandas as pd
import pytest

from bike_router import graph_store, pipeline
from bike_router.builder import _open_osm, _shift_station_ids
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


def test_shift_station_ids_offsets_only_negatives():
    nodes = pd.DataFrame({"osmid": [10, -1, -2]})
    edges = pd.DataFrame({"from_node": [10, -1], "to_node": [-1, -2]})
    n2, e2 = _shift_station_ids(nodes_df=nodes, edges_df=edges, offset=5)
    assert list(n2["osmid"]) == [10, -6, -7]  # positives untouched, negatives shifted
    assert list(e2["from_node"]) == [10, -6]
    assert list(e2["to_node"]) == [-6, -7]


def test_open_osm_without_bbox_returns_osm():
    import pyrosm

    osm = _open_osm(pbf_path=pyrosm.get_data("test_pbf"), bbox=None)
    assert osm is not None


def test_open_osm_with_bbox_clips():
    import pyrosm

    osm = _open_osm(pbf_path=pyrosm.get_data("test_pbf"), bbox=(26.9, 60.5, 27.0, 60.6))
    assert osm is not None  # constructed with a bounding box without error


def test_assert_within_coverage_passes_and_fails(tmp_path):
    (tmp_path / "meta.json").write_text('{"bbox": [7.9, 47.9, 8.2, 48.2], "tile_deg": 0.5}')
    # (lat, lon) endpoints inside the bbox → no raise
    pipeline._assert_within_coverage(start_latlon=(48.0, 8.0), dest_latlon=(48.1, 8.1), graph_dir=tmp_path)
    with pytest.raises(OutOfCoverageError, match="coverage"):
        pipeline._assert_within_coverage(start_latlon=(48.0, 8.0), dest_latlon=(60.0, 20.0), graph_dir=tmp_path)
