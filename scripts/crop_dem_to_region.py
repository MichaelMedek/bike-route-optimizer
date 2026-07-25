"""Crop the full Euro DEM to the Central Europe region for the bike router.

Adapted verbatim from ski-resort-designer's `scripts/crop_dem_to_alps.py`; only
the bounding box (and output name) changed.

The EuroDEM's native CRS is ETRS89 with ARCSECOND units, so the degree bbox is
multiplied by 3600 before building the crop geometry (rasterio.mask needs shapes
in the dataset's native CRS).

To create the cropped DEM:
1. Download full EuroDEM from https://www.mapsforeurope.org/datasets/euro-dem
2. Ensure INPUT_FILE points to eurodem.tif
3. Run: python scripts/crop_dem_to_region.py
"""

from pathlib import Path

import rasterio
from rasterio.mask import mask
from shapely.geometry import box

from bike_router.constants import DEMConfig

# Paths - update INPUT_FILE to point to your downloaded full EuroDEM
INPUT_FILE = Path.home() / "Downloads" / "euro-dem-tif" / "data" / "eurodem" / "eurodem.tif"
OUTPUT_FILE = DEMConfig.EURODEM_PATH

# Central Europe bounding box in degrees (WGS84):
# Alps + all Bayern & Baden-Württemberg + Switzerland + Austria + eastern France
# + northern Italy / Slovenia.
REGION_WEST_DEG = 4.0
REGION_EAST_DEG = 17.5
REGION_SOUTH_DEG = 43.0
REGION_NORTH_DEG = 50.7

# The EuroDEM uses arcseconds as units (1 degree = 3600 arcseconds).
ARCSECONDS_PER_DEGREE = 3600


def crop_dem_to_region() -> None:
    """Load full Euro DEM, crop to the Central Europe region, save as compressed GeoTIFF."""
    west = REGION_WEST_DEG * ARCSECONDS_PER_DEGREE
    east = REGION_EAST_DEG * ARCSECONDS_PER_DEGREE
    south = REGION_SOUTH_DEG * ARCSECONDS_PER_DEGREE
    north = REGION_NORTH_DEG * ARCSECONDS_PER_DEGREE

    print(f"Region bbox in arcseconds: W={west}, E={east}, S={south}, N={north}")

    region_bbox = box(west, south, east, north)
    geo = [region_bbox.__geo_interface__]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(INPUT_FILE) as src:
        print(f"Input CRS: {src.crs}")
        print(f"Input bounds: {src.bounds}")
        print(f"Input shape: {src.width} x {src.height}")

        out_image, out_transform = mask(dataset=src, shapes=geo, crop=True)
        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "lzw",
                "predictor": 2,  # horizontal differencing → better int16 LZW ratio
            }
        )

        print(f"Output shape: {out_meta['width']} x {out_meta['height']}")

        with rasterio.open(OUTPUT_FILE, "w", **out_meta) as dest:
            dest.write(out_image)

    size_mb = OUTPUT_FILE.stat().st_size / 1024 / 1024
    print(f"Saved cropped DEM to {OUTPUT_FILE} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    crop_dem_to_region()
