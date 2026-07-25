"""Upload the cropped Central Europe DEM to Hugging Face Hub.

Adapted verbatim from ski-resort-designer's upload script; points at the new
DEMConfig (repo MichaelMedek/central_europe_eurodem, file region_dem.tif).

Usage:
1. pip install huggingface_hub
2. python scripts/upload_dem_to_huggingface.py
3. On first run you'll be prompted to login (creates ~/.huggingface/token).

Note: create the dataset repo first if it does not exist, e.g.
    huggingface-cli repo create central_europe_eurodem --type dataset
"""

from huggingface_hub import HfApi, login

from bike_router.constants import DEMConfig


def upload_dem_to_hf() -> None:
    """Upload the region DEM file to Hugging Face Hub."""
    dem_file = DEMConfig.EURODEM_PATH
    repo_id = DEMConfig.HF_REPO_ID
    filename = DEMConfig.HF_FILENAME

    if not dem_file.exists():
        raise FileNotFoundError(f"DEM file not found: {dem_file}")

    print(f"File to upload: {dem_file}")
    print(f"File size: {dem_file.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"Target repo: {repo_id}")

    login()

    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(dem_file),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="dataset",
    )

    print("\nUploaded successfully!")
    print(f"Download URL: {DEMConfig.HF_DOWNLOAD_URL}")


if __name__ == "__main__":
    upload_dem_to_hf()
