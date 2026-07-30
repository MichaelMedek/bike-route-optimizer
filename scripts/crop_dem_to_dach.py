"""Crop the full EuroDEM to the DACH region — the build-time elevation input.

The offline graph build bakes node/edge elevations from this DEM, so it must cover
all of Germany + Austria + Switzerland. Cropping the ~1.9 GB Europe-wide EuroDEM to
DACH keeps the build's DEM small and fast. Ported from ski-resort-designer's
crop_dem_to_alps.py (same EuroDEM, different region).

To (re)create the DACH DEM:
    1. Download the full EuroDEM from https://www.mapsforeurope.org/datasets/euro-dem
    2. Run: python scripts/crop_dem_to_dach.py --input /path/to/eurodem.tif
       (default --input is ~/Downloads/euro-dem-tif/data/eurodem/eurodem.tif)
"""

import argparse
from pathlib import Path

import rasterio
from rasterio.mask import mask
from shapely.geometry import box

from bike_router.core.constants import DEMConfig, GraphConfig

_DEFAULT_INPUT = Path.home() / "Downloads" / "euro-dem-tif" / "data" / "eurodem" / "eurodem.tif"

# EuroDEM's native CRS is ETRS89 in ARCSECONDS (not degrees): 1° = 3600″.
ARCSECONDS_PER_DEGREE = 3600


def crop_dem_to_dach(*, input_file: Path) -> None:
    """Load the full EuroDEM, crop to the DACH bbox (+ margin), save as an LZW GeoTIFF.

    The crop box is GraphConfig.DACH_BBOX_DEG grown by DEM_CROP_MARGIN_DEG (single source,
    so DEM coverage always contains the built region). Always writes to
    DEMConfig.EURODEM_PATH — the one fixed location the builder reads.

    Args:
        input_file: Full Europe-wide EuroDEM GeoTIFF.
    """
    output_file = DEMConfig.EURODEM_PATH
    w, s, e, n = GraphConfig.DACH_BBOX_DEG
    m = GraphConfig.DEM_CROP_MARGIN_DEG
    # The bbox must be expressed in the DEM's native arcsecond CRS.
    west = (w - m) * ARCSECONDS_PER_DEGREE
    east = (e + m) * ARCSECONDS_PER_DEGREE
    south = (s - m) * ARCSECONDS_PER_DEGREE
    north = (n + m) * ARCSECONDS_PER_DEGREE
    print(f"DACH bbox in arcseconds: W={west}, E={east}, S={south}, N={north}")

    geo = [box(west, south, east, north).__geo_interface__]
    with rasterio.open(input_file) as src:
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
                "compress": "lzw",  # lossless
            }
        )
        print(f"Output shape: {out_meta['width']} x {out_meta['height']}")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_file, "w", **out_meta) as dest:
            dest.write(out_image)

    print(f"Saved DACH DEM to {output_file}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Crop the full EuroDEM to the DACH region.")
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT, help="Full EuroDEM GeoTIFF.")
    args = parser.parse_args(argv)
    if not args.input.exists():
        raise FileNotFoundError(
            f"Full EuroDEM not found: {args.input}\nDownload it from https://www.mapsforeurope.org/datasets/euro-dem"
        )
    crop_dem_to_dach(input_file=args.input)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
