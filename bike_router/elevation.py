"""DEM elevation sampling — vectorized, CRS-aware, nodata-safe.

Copied in refined form from ski-resort-designer's `core/dem_service.py`. The
sibling implementation is production-grade and solves the load-bearing gotcha:
the EuroDEM is **ETRS89 in ARCSECONDS**, not EPSG:4326 degrees. We therefore
reproject WGS84 lon/lat → the DEM's native CRS with a cached pyproj transformer
before the inverse-affine array gather. (This is why we do NOT use
`osmnx.elevation.add_node_elevations_raster`, which assumes graph coords already
match the raster grid and would silently mis-sample the arcsecond DEM.)
"""

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import rasterio
import requests
from pyproj import Transformer
from rasterio.io import DatasetReader

from bike_router.constants import DEMConfig

logger = logging.getLogger(__name__)


def download_dem_from_huggingface(
    target_path: Path = DEMConfig.EURODEM_PATH,
    progress_callback: Callable[[float], None] | None = None,
) -> Path:
    """Download the region DEM from Hugging Face if not already present."""
    if target_path.exists():
        logger.debug("DEM already exists at %s", target_path)
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    url = DEMConfig.HF_DOWNLOAD_URL
    logger.info("Downloading region DEM from %s ...", url)

    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0
    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if progress_callback and total_size > 0:
                progress_callback(downloaded / total_size)

    logger.info("DEM downloaded to %s", target_path)
    return target_path


def ensure_dem(dem_path: Path) -> Path:
    """Return a ready-to-use DEM path, downloading the default from HF if missing.

    Only the bundled default path (DEMConfig.EURODEM_PATH) is auto-fetched; a
    missing explicit override is a hard error.

    Args:
        dem_path: Requested DEM file path.
    """
    if dem_path.exists():
        return dem_path
    if dem_path != DEMConfig.EURODEM_PATH:
        raise FileNotFoundError(f"DEM file not found: {dem_path}")
    logger.info("DEM not found at %s — downloading from Hugging Face…", dem_path)
    return download_dem_from_huggingface(target_path=dem_path, progress_callback=None)


class DEMService:
    """Singleton elevation sampler over a DEM GeoTIFF.

    The DEM is loaded once (thread-safe) into an in-memory NumPy array; queries
    then reproject to the raster CRS and gather via the inverse affine transform.

    Example:
        dem = DEMService(Path("region_dem.tif"))
        elevations = dem.get_elevations(lons=[8.41], lats=[48.46])
    """

    _instance: Optional["DEMService"] = None
    _load_lock = threading.Lock()
    _dem_path: Path
    _dem: DatasetReader | None = None
    _dem_crs: str | None = None
    _dem_array: npt.NDArray[np.float64] | None = None
    _dem_transform: object = None
    _dem_nodata: object = None
    # Cached WGS84→DEM-CRS transformer, built once at load. None when the DEM is
    # already EPSG:4326 (identity, no reprojection).
    _to_dem: Transformer | None = None
    # WGS84 (west, south, east, north), computed once at load.
    _wgs84_bounds: tuple[float, float, float, float] | None = None

    def __new__(cls, dem_path: Path | None = None) -> "DEMService":
        """Create or return the singleton instance (one DEM per process)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._dem_path = dem_path or DEMConfig.EURODEM_PATH
        elif dem_path is not None and cls._instance._dem_path != Path(dem_path):
            # A different DEM path was requested — reset so the new file loads.
            cls._instance = super().__new__(cls)
            cls._instance._dem_path = Path(dem_path)
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        """True once the DEM array + transform are in memory."""
        return self._dem_transform is not None

    def _ensure_loaded(self) -> None:
        """Load the DEM into memory on first access (thread-safe)."""
        if self.is_loaded:
            return
        with self._load_lock:
            if self.is_loaded:
                return  # type: ignore[unreachable]
            dem_path = self._dem_path
            if not Path(dem_path).exists():
                raise FileNotFoundError(
                    f"DEM file not found at {dem_path}. Crop it with "
                    "scripts/crop_dem_to_region.py or download from Hugging Face."
                )
            logger.info("Loading DEM from %s ...", dem_path)
            start = time.time()
            dataset = rasterio.open(dem_path)
            dem_array = dataset.read(1)
            self._dem = dataset
            self._dem_crs = dataset.crs.to_string() if dataset.crs else "EPSG:4326"
            self._dem_array = dem_array
            self._dem_nodata = dataset.nodata
            if self._dem_crs != "EPSG:4326":
                self._to_dem = Transformer.from_crs("EPSG:4326", self._dem_crs, always_xy=True)
            self._wgs84_bounds = self._compute_wgs84_bounds(native_bounds=dataset.bounds)
            self._dem_transform = dataset.transform  # set LAST — is_loaded checks this
            logger.info(
                "DEM loaded in %.2fs (shape=%s, CRS=%s)",
                time.time() - start,
                dem_array.shape,
                self._dem_crs,
            )

    def _compute_wgs84_bounds(self, native_bounds: object) -> tuple[float, float, float, float]:
        """WGS84 (west, south, east, north) from the DEM's native-CRS bounds."""
        box = native_bounds
        if self._to_dem is not None:
            corners_x = [box.left, box.right, box.left, box.right]  # type: ignore[attr-defined]
            corners_y = [box.bottom, box.bottom, box.top, box.top]  # type: ignore[attr-defined]
            corner_lons, corner_lats = self._to_dem.transform(corners_x, corners_y, direction="INVERSE")
            return min(corner_lons), min(corner_lats), max(corner_lons), max(corner_lats)
        return box.left, box.bottom, box.right, box.top  # type: ignore[attr-defined]

    def get_elevations(
        self,
        lons: "npt.NDArray[np.float64] | list[float]",
        lats: "npt.NDArray[np.float64] | list[float]",
    ) -> "npt.NDArray[np.float64]":
        """Batch elevation lookup — one reprojection + one vectorized array gather.

        Out-of-coverage / nodata cells come back as ``np.nan``. This is the only
        query path: the pipeline always samples node/vertex arrays in bulk.
        """
        self._ensure_loaded()
        assert self._dem_transform is not None
        assert self._dem_array is not None
        lons = np.asarray(lons, dtype=np.float64)
        lats = np.asarray(lats, dtype=np.float64)

        # WGS84 → DEM CRS (one batched pyproj transform) when reprojection needed.
        if self._to_dem is not None:
            proj_x, proj_y = self._to_dem.transform(lons, lats)
        else:
            proj_x, proj_y = lons, lats

        # Inverse affine → array indices (vectorized ~transform * (x, y)).
        inverse = ~self._dem_transform  # type: ignore[operator]
        cols = (inverse.a * proj_x + inverse.b * proj_y + inverse.c).astype(np.int64)
        rows = (inverse.d * proj_x + inverse.e * proj_y + inverse.f).astype(np.int64)

        height, width = self._dem_array.shape
        in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        result = np.full(np.shape(lons), np.nan, dtype=np.float64)
        result[in_bounds] = self._dem_array[rows[in_bounds], cols[in_bounds]]
        if self._dem_nodata is not None:
            result[result == self._dem_nodata] = np.nan
        return result

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(west, south, east, north) WGS84 bounds — computed once at load."""
        self._ensure_loaded()
        assert self._wgs84_bounds is not None
        return self._wgs84_bounds


# --- Spec-compatible module-level function ----------------------------------


def get_elevations_from_raster(
    lats: "npt.NDArray[np.float64] | list[float]",
    lngs: "npt.NDArray[np.float64] | list[float]",
    raster_path: "str | Path",
) -> "npt.NDArray[np.float64]":
    """Vectorized elevation lookup for parallel (lats, lngs) arrays; NaN out of coverage."""
    return DEMService(dem_path=Path(raster_path)).get_elevations(lons=lngs, lats=lats)
