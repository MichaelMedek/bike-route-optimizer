"""Upload the prebuilt DACH bike+rail graph artifact to Hugging Face Hub (zstd-recompressed).

Recompresses every tile parquet snappy→zstd into a temp staging dir before upload (lossless,
~35% smaller → faster first-run download on Streamlit Cloud). The local build artifact is left
untouched; readers auto-detect the codec, so download_graph_from_hf needs no change.

Usage:
    1. pip install huggingface_hub
    2. python scripts/upload_graph_to_huggingface.py
    3. On first run you'll be prompted to login (creates ~/.huggingface/token).

Note: create the dataset repo first if it does not exist, e.g.
    huggingface-cli repo create dach_bike_graph --type dataset
"""

import shutil
import tempfile
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, login

from bike_router.constants import GraphConfig


def _stage_zstd_copy(*, artifact_dir: Path, staging_dir: Path) -> None:
    """Mirror the artifact into ``staging_dir``, recompressing each parquet tile to zstd (lossless).

    Parquet stores its codec in the file, so re-reading needs no codec hint; meta.json is copied as-is.
    """
    for src in artifact_dir.rglob("*"):
        if not src.is_file():
            continue
        dest = staging_dir / src.relative_to(artifact_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".parquet":
            pd.read_parquet(src).to_parquet(dest, compression="zstd", index=False)
        else:
            shutil.copy(src, dest)  # contents only — mtime is "now" like the rewritten tiles


def upload_graph_to_hf() -> None:
    """Recompress the tiled graph artifact to zstd, then upload it to Hugging Face Hub."""
    artifact_dir = GraphConfig.GRAPH_DIR
    meta = artifact_dir / GraphConfig.META_FILENAME
    if not meta.exists():
        raise FileNotFoundError(f"No artifact at {artifact_dir} (run scripts/build_dach_graph.py first)")

    src_mb = sum(f.stat().st_size for f in artifact_dir.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"Artifact dir: {artifact_dir}")
    print(f"Source size (snappy): {src_mb:.1f} MB")

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "dach_graph"
        print("Recompressing tiles snappy → zstd …")
        _stage_zstd_copy(artifact_dir=artifact_dir, staging_dir=staging)
        staged_mb = sum(f.stat().st_size for f in staging.rglob("*") if f.is_file()) / 1024 / 1024
        print(f"Staged size (zstd): {staged_mb:.1f} MB ({(1 - staged_mb / src_mb) * 100:.0f}% smaller)")
        print(f"Target repo: {GraphConfig.HF_REPO_ID}")

        login()
        api = HfApi()
        api.upload_folder(folder_path=str(staging), repo_id=GraphConfig.HF_REPO_ID, repo_type="dataset")
    print("\nUploaded successfully!")


if __name__ == "__main__":
    upload_graph_to_hf()
