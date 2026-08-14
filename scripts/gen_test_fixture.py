"""Regenerate the committed test fixture (tests/fixtures/dach_graph) from the built per-region artifacts,
and plot its overview PNG so the fixture's rail geometry can be eyeballed exactly like the full-DACH one.

1:1 PARITY with the production pipeline: this clips ONE built region to a small Schwarzwald/Gäubahn window
(Freudenstadt summit + Horb valley + the Nagold/Murg lines), writes it as a temp region artifact, then runs
the SAME ``combine_regions`` Phase-3 (dedup → consolidate_rail → prune → remap) the full build uses.

    python scripts/gen_test_fixture.py
"""

import logging
import shutil
import tempfile
from pathlib import Path

from bike_router.core.constants import GraphConfig, Schema
from bike_router.preprocessing.graph_writer import (
    plot_graph_overview,
    read_region_tables,
    write_graph_parquet,
)
from bike_router.preprocessing.regions import base_meta, combine_regions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("gen_test_fixture")

_SOURCE_REGION = "karlsruhe-regbez"  # holds Freudenstadt, Horb, Eutingen, the Nagold + Murg lines
_BBOX = (8.30, 48.42, 8.80, 48.62)  # W, S, E, N — one connected Schwarzwald/Gäubahn window
_REGIONS_DIR = GraphConfig.GRAPH_DIR.parent / "dach_build" / "dach_graph_per_region"
_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "dach_graph"


def _clip(nodes_df, edges_df, bbox):  # noqa: ANN001, ANN202
    """Keep nodes inside the bbox and the edges whose both endpoints survive."""
    w, s, e, n = bbox
    inside = (nodes_df[Schema.LON] >= w) & (nodes_df[Schema.LON] <= e)
    inside &= (nodes_df[Schema.LAT] >= s) & (nodes_df[Schema.LAT] <= n)
    keep = set(nodes_df.loc[inside, Schema.OSMID].astype(int))
    nodes_df = nodes_df[inside].reset_index(drop=True)
    mask = edges_df[Schema.FROM_NODE].isin(keep) & edges_df[Schema.TO_NODE].isin(keep)
    return nodes_df, edges_df[mask].reset_index(drop=True)


def main() -> int:
    """Clip a built region, then run the PRODUCTION combine_regions Phase-3 on it → write fixture + plot."""
    nodes_df, edges_df = read_region_tables(region_dir=_REGIONS_DIR / _SOURCE_REGION)
    nodes_df, edges_df = _clip(nodes_df, edges_df, _BBOX)
    logger.info(f"clipped {_SOURCE_REGION} to {_BBOX}: {len(nodes_df)} nodes / {len(edges_df)} edges")

    # Stage the clip as a one-region artifact, then run the EXACT production Phase-3 (1:1 parity).
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp)
        stage_meta = {"tile_deg": GraphConfig.TILE_DEG, "confirmed_complete": True}
        write_graph_parquet(
            nodes_df=nodes_df, edges_df=edges_df, meta=stage_meta, out_dir=staged / "clip", compression="snappy"
        )
        nodes_df, edges_df = combine_regions(regions_dir=staged, regions=["clip"])

    if _FIXTURE_DIR.exists():
        shutil.rmtree(_FIXTURE_DIR)
    meta = base_meta(nodes_df=nodes_df, edges_df=edges_df, tolerance_m=GraphConfig.CONSOLIDATION_TOLERANCE_M)
    write_graph_parquet(nodes_df=nodes_df, edges_df=edges_df, meta=meta, out_dir=_FIXTURE_DIR, compression="snappy")
    plot_graph_overview(
        nodes_df=nodes_df,
        edges_df=edges_df,
        out_path=_FIXTURE_DIR / "fixture_overview.png",
        title="Test-fixture overview — bike (thin blue) · rail (thick purple)",
        figsize=(12, 10),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
