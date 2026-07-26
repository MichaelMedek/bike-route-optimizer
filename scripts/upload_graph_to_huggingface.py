"""Upload the prebuilt DACH bike+rail graph artifact to Hugging Face Hub.

Uploads the whole tiled artifact dir (nodes/, edges/, meta.json) built by
scripts/build_dach_graph.py. Mirrors upload_dem_to_huggingface.py.

Usage:
    1. pip install huggingface_hub
    2. python scripts/upload_graph_to_huggingface.py
    3. On first run you'll be prompted to login (creates ~/.huggingface/token).

Note: create the dataset repo first if it does not exist, e.g.
    huggingface-cli repo create dach_bike_graph --type dataset
"""

from huggingface_hub import HfApi, login

from bike_router.constants import GraphConfig


def upload_graph_to_hf() -> None:
    """Upload the tiled graph artifact directory to Hugging Face Hub."""
    artifact_dir = GraphConfig.GRAPH_DIR
    meta = artifact_dir / GraphConfig.META_FILENAME
    if not meta.exists():
        raise FileNotFoundError(f"No artifact at {artifact_dir} (run scripts/build_dach_graph.py first)")

    total_mb = sum(f.stat().st_size for f in artifact_dir.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"Artifact dir: {artifact_dir}")
    print(f"Total size: {total_mb:.1f} MB")
    print(f"Target repo: {GraphConfig.HF_REPO_ID}")

    login()
    api = HfApi()
    api.upload_folder(
        folder_path=str(artifact_dir),
        repo_id=GraphConfig.HF_REPO_ID,
        repo_type="dataset",
    )
    print("\nUploaded successfully!")


if __name__ == "__main__":
    upload_graph_to_hf()
