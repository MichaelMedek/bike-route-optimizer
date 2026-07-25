"""DEMService tests against a SYNTHETIC arcsecond-CRS GeoTIFF.

Proves the load-bearing behavior: the DEM's native CRS is ETRS89 in arcseconds,
so a WGS84 lon/lat query must be reprojected (×3600) before the array gather.
We build a tiny in-CRS raster with a known elevation ramp and assert both the
scalar and vectorized paths, plus the spec-signature wrapper and nodata → NaN.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS

from bike_router.elevation import (
    DEMService,
    download_dem_from_huggingface,
    ensure_dem,
    get_elevations_from_raster,
)

# ETRS89 in arcseconds — the exact WKT family of the real EuroDEM.
ARCSEC_WKT = (
    'GEOGCS["ETRS89",DATUM["European_Terrestrial_Reference_System_1989",'
    'SPHEROID["GRS 1980",6378137,298.257222101]],PRIMEM["Greenwich",0],'
    'UNIT["Arcsecond",4.84813681109536E-06],AXIS["Latitude",NORTH],AXIS["Longitude",EAST]]'
)
NODATA = 32767


@pytest.fixture
def arcsecond_dem(tmp_path: Path) -> Path:
    """A 100×100 int16 DEM in arcsecond CRS covering lon 8–9°, lat 48–49°.

    Elevation ramps with latitude: elev = round((lat - 48) * 1000) metres, so a
    known lat maps to a known value only if reprojection (deg→arcsec) is correct.
    One corner cell is nodata.
    """
    # 1° = 3600 arcsec. Pixel = 36 arcsec (0.01°). Origin at (8°, 49°) top-left.
    west_as, north_as = 8 * 3600, 49 * 3600
    px = 36.0  # arcsec per pixel
    transform = Affine.translation(west_as, north_as) * Affine.scale(px, -px)

    h = w = 100
    lats = 49.0 - (np.arange(h) + 0.5) * 0.01  # row center latitudes
    data = np.round((lats - 48.0) * 1000).astype(np.int16)[:, None].repeat(w, axis=1)
    data[0, 0] = NODATA

    path = tmp_path / "arcsec_dem.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="int16",
        crs=CRS.from_wkt(ARCSEC_WKT),
        transform=transform,
        nodata=NODATA,
    ) as dst:
        dst.write(data, 1)

    DEMService._instance = None  # reset singleton between tests
    return path


def test_reprojection_samples_correct_elevation(arcsecond_dem: Path):
    dem = DEMService(dem_path=arcsecond_dem)
    # lat 48.5 → (48.5-48)*1000 = 500 m (± one pixel). Only correct if deg→arcsec
    # reprojection happened before the array gather.
    elev = float(dem.get_elevations(lons=[8.5], lats=[48.5])[0])
    assert abs(elev - 500) <= 12


def test_vectorized_batch_ramp(arcsecond_dem: Path):
    dem = DEMService(dem_path=arcsecond_dem)
    batch = dem.get_elevations(lons=[8.2, 8.5, 8.8], lats=[48.2, 48.5, 48.8])
    # elevation ramps with latitude → strictly increasing across the batch
    assert batch[0] < batch[1] < batch[2]
    np.testing.assert_allclose(batch, [200, 500, 800], atol=12)


def test_bounds_reprojected_to_wgs84(arcsecond_dem: Path):
    dem = DEMService(dem_path=arcsecond_dem)
    west, south, east, north = dem.bounds
    assert 7.9 < west < 8.1 and 8.9 < east < 9.1
    assert 47.9 < south < 48.1 and 48.9 < north < 49.1


def test_nodata_and_out_of_bounds_are_nan(arcsecond_dem: Path):
    dem = DEMService(dem_path=arcsecond_dem)
    out = dem.get_elevations(lons=[8.0, 20.0], lats=[49.0, 60.0])
    # top-left cell is nodata; second point is far outside coverage → both NaN
    assert np.all(np.isnan(out))


def test_spec_vectorized_wrapper(arcsecond_dem: Path):
    arr = get_elevations_from_raster(lats=[48.2, 48.8], lngs=[8.2, 8.8], raster_path=str(arcsecond_dem))
    assert arr.shape == (2,)
    assert np.all(np.isfinite(arr))


def test_ensure_dem_returns_existing(arcsecond_dem: Path):
    # an existing path is returned unchanged, no download
    assert ensure_dem(dem_path=arcsecond_dem) == arcsecond_dem


def test_ensure_dem_missing_override_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ensure_dem(dem_path=tmp_path / "nope.tif")


def test_download_dem_skips_when_present(arcsecond_dem: Path):
    # already-present target short-circuits (no network call)
    assert download_dem_from_huggingface(target_path=arcsecond_dem) == arcsecond_dem


def test_download_dem_streams_to_disk(tmp_path: Path, monkeypatch):
    """Missing target → streamed download written chunk-by-chunk; progress fires."""
    from bike_router import elevation

    class _FakeResponse:
        headers = {"content-length": "6"}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"abc"
            yield b"def"

    monkeypatch.setattr(elevation.requests, "get", lambda url, stream, timeout: _FakeResponse())
    seen: list[float] = []
    target = tmp_path / "dl.tif"
    out = download_dem_from_huggingface(target_path=target, progress_callback=seen.append)
    assert out == target
    assert target.read_bytes() == b"abcdef"
    assert seen[-1] == 1.0  # reached 100%


def test_missing_dem_raises_filenotfound(tmp_path: Path):
    DEMService._instance = None
    dem = DEMService(dem_path=tmp_path / "absent.tif")
    with pytest.raises(FileNotFoundError):
        dem.get_elevations(lons=[8.0], lats=[48.0])


def test_wgs84_dem_needs_no_reprojection(tmp_path: Path):
    """A plain EPSG:4326 DEM skips the pyproj transform (identity branch)."""
    from affine import Affine

    transform = Affine.translation(8.0, 49.0) * Affine.scale(0.01, -0.01)
    data = np.arange(100, dtype=np.int16).reshape(10, 10)
    path = tmp_path / "wgs84.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype="int16",
        crs=CRS.from_epsg(4326),
        transform=transform,
        nodata=-1,
    ) as dst:
        dst.write(data, 1)

    DEMService._instance = None
    dem = DEMService(dem_path=path)
    west, south, east, north = dem.bounds
    assert abs(west - 8.0) < 1e-9 and abs(north - 49.0) < 1e-9  # identity bounds
    value = float(dem.get_elevations(lons=[8.05], lats=[48.95])[0])
    assert np.isfinite(value)


def test_singleton_resets_on_new_path(arcsecond_dem: Path, tmp_path: Path):
    DEMService._instance = None
    first = DEMService(dem_path=arcsecond_dem)
    second = DEMService(dem_path=tmp_path / "other.tif")
    assert first is not second  # a different path forces a fresh instance
