"""Upload the prebuilt DACH bike+rail graph artifact to Hugging Face Hub.

Uploads the final tiled artifact dir (nodes/, edges/, meta.json, dach_graph_overview.png) built by
scripts/build_dach_graph.py, then pushes huggingface_dataset_card.md as the repo README.md (the dataset
card). Phase 3 writes the graph zstd-compressed; Phase 4 saves the overview plot into the same dir so the
card's relative reference resolves on the dataset page.

Usage:
    1. pip install huggingface_hub
    2. python scripts/upload_graph_to_huggingface.py
    3. On first run you'll be prompted to login (creates ~/.huggingface/token).

The dataset repo is created on first run if it does not exist.
"""

from huggingface_hub import HfApi, login

from bike_router.core.constants import PROJECT_ROOT, GraphConfig


def upload_graph_to_hf() -> None:
    """Upload the final tiled graph artifact directory + dataset card to Hugging Face Hub (as-is)."""
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
    api.create_repo(repo_id=GraphConfig.HF_REPO_ID, repo_type="dataset", exist_ok=True)
    # delete_patterns=["*"] makes the upload a true MIRROR: remote files absent locally are deleted
    # README.md is re-pushed just below.
    api.upload_folder(
        folder_path=str(artifact_dir), repo_id=GraphConfig.HF_REPO_ID, repo_type="dataset", delete_patterns=["*"]
    )
    # Push the repo-root dataset card directly AS README.md so the Hub renders it.
    api.upload_file(
        path_or_fileobj=str(PROJECT_ROOT / "huggingface_dataset_card.md"),
        path_in_repo="README.md",
        repo_id=GraphConfig.HF_REPO_ID,
        repo_type="dataset",
    )
    print("\nUploaded successfully!")


if __name__ == "__main__":
    upload_graph_to_hf()
