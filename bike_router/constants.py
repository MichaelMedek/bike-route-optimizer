"""Central configuration for the bicycle route optimizer."""

import os
from dataclasses import dataclass
from pathlib import Path

import platformdirs

# Package + project roots
PACKAGE_DIR = Path(__file__).parent
PROJECT_ROOT = PACKAGE_DIR.parent


def _user_data_root() -> Path:
    """Writable per-user OS app-data dir for the DEM download.

    `BIKEROUTER_DATA_ROOT` overrides it (CI / power users) — an external
    condition, not an internal invariant.
    """
    override = os.environ.get("BIKEROUTER_DATA_ROOT")
    if override:
        return Path(override)
    return Path(platformdirs.user_data_dir("BikeRouteOptimizer", appauthor=False))


DATA_DIR = _user_data_root() / "data"


class OutputConfig:
    """Where generated artifacts and the OSM/Overpass HTTP cache live.

    Both directories are version-controlled as empty dirs (via .gitkeep) but
    their *contents* are gitignored — routes and cached Overpass responses are
    reproducible, not source.
    """

    OUTPUT_DIR = PROJECT_ROOT / "output"
    CACHE_DIR = PROJECT_ROOT / "cache"


class DEMConfig:
    """Elevation-model file path and Hugging Face hosting.

    Mirrors the naming convention of the Alps dataset
    (`MichaelMedek/alps_eurodem`) for the wider Central Europe region.
    """

    EURODEM_PATH = DATA_DIR / "region_dem.tif"

    HF_REPO_ID = "MichaelMedek/central_europe_eurodem"
    HF_FILENAME = "region_dem.tif"
    HF_DOWNLOAD_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{HF_FILENAME}"


class GeoConfig:
    """Spherical-Earth constants for great-circle distance."""

    EARTH_RADIUS_M = 6_371_000.0
    # metres per degree of latitude at the equator
    METERS_PER_DEGREE_EQUATOR = 111_320.0


class CorridorConfig:
    """The "Schlauch" corridor around the straight start→dest line."""

    # buffer radius in DEGREES (~0.2° ≈ 22 km each side → ~40 km-wide corridor)
    BUFFER_DEG = 0.2
    # Fail fast if start/dest are closer than this straight-line — too short to plan.
    MIN_TRIP_KM = 5.0


class SurfaceConfig:
    """Surface-quality multipliers (asphalt cheap, dirt very expensive)."""

    SURFACE_FACTORS = {
        "asphalt": 1.0,
        "concrete": 1.0,
        "paving_stones": 1.0,
        "gravel": 3.5,
        "fine_gravel": 3.5,
        "unpaved": 3.5,
        "compacted": 3.5,
        "dirt": 8.0,
        "earth": 8.0,
        "sand": 8.0,
        "mud": 8.0,
    }
    # Fallbacks when `surface` is missing/unknown, keyed by highway class.
    UNKNOWN_TRACK_PATH = 2.5  # highway in {track, path}
    UNKNOWN_OTHER = 1.1
    # highway values that trigger the harsher unknown-surface fallback
    ROUGH_HIGHWAYS = frozenset({"track", "path"})


class RoadConfig:
    """Road-type multipliers (cycleways rewarded, arterials deterred)."""

    ROAD_FACTORS = {
        "cycleway": 0.85,
        "living_street": 0.85,
        "residential": 1.2,
        "tertiary": 1.2,
        "secondary": 6.0,
        "primary": 6.0,
    }
    DEFAULT = 1.0


class CostConfig:
    """Elevation-penalty coefficients and per-edge cost components."""

    ELEV_COEFF = 50.0  # metres-of-climb weight
    GRADE_COEFF = 10.0  # extra punishment proportional to gradient
    # Per-edge additive cost COMPONENTS (see cost.py). A weighted sum of these
    # three, with per-profile weights, is what A* minimizes.
    COMP_DIST = "cost_dist"  # length in metres
    COMP_SURFACE = "cost_surface"  # length * (SurfaceFactor*RoadFactor - 1)
    COMP_ELEV = "cost_elev"  # uphill elevation penalty
    # Cheapest possible SurfaceFactor*RoadFactor (perfect asphalt cycleway),
    # DERIVED from the tables (no drift). The surface component's per-metre floor
    # is (this - 1); used to keep each profile's A* heuristic admissible.
    MIN_SF_RF = min(SurfaceConfig.SURFACE_FACTORS.values()) * min(RoadConfig.ROAD_FACTORS.values())


@dataclass(frozen=True)
class RouteProfile:
    """One routing preset: a display name, filename postfix, and the three
    component weights. All factors stay active; one is simply weighted heavier.
    """

    name: str
    postfix: str
    w_dist: float
    w_surface: float
    w_elev: float


# The routes computed every query. Base weight 1.0 on all components, with one
# boosted (BOOST) so that route leans toward its category without collapsing the
# others. The 4th, `balanced` (1,1,1), is the best all-round compromise and also
# drives the sanity checks (it reproduces the plain cost model).
class RouteConfig:
    BOOST = 2.5  # tuning knob: how strongly the favoured component is weighted
    PROFILES = (
        RouteProfile(name="least uphill", postfix="flattest", w_dist=1.0, w_surface=1.0, w_elev=BOOST),
        RouteProfile(name="least distance", postfix="shortest", w_dist=BOOST, w_surface=1.0, w_elev=1.0),
        RouteProfile(name="best surface", postfix="smoothest", w_dist=1.0, w_surface=BOOST, w_elev=1.0),
        RouteProfile(name="best all-round", postfix="balanced", w_dist=1.0, w_surface=1.0, w_elev=1.0),
    )
    BALANCED = PROFILES[-1]  # the (1,1,1) profile; also the default for sanity checks


class GmapsConfig:
    """Google Maps directions-URL output."""

    # exactly N downsampled points → origin + (N-2) waypoints + destination.
    # Maps `api=1` allows max 9 intermediate waypoints; N=10 → 8 waypoints (OK).
    N_WAYPOINTS = 10
    BASE_URL = "https://www.google.com/maps/dir/?api=1"
    TRAVEL_MODE = "bicycling"


class NominatimConfig:
    """Geocoding via OpenStreetMap Nominatim."""

    USER_AGENT = "bike-route-optimizer/0.1 (https://github.com/MichaelMedek/bike-route-optimizer)"
    RATE_LIMIT_S = 1.0  # Nominatim usage policy: max 1 request / second


class GpxConfig:
    """Synthesized-timestamp settings for the GPX track.

    One constant cycling speed across the whole route — timestamps are purely
    synthetic (no real ride), so a single average is all that's meaningful.
    """

    SPEED_KMH = 20.0
    METERS_PER_KM = 1000.0
    SECONDS_PER_HOUR = 3600.0
    MINUTES_PER_HOUR = 60.0


class SanityConfig:
    """Thresholds for the runtime sanity checks."""

    # Below this many raw nodes the >50% simplify-shrink check is not meaningful.
    MIN_MEANINGFUL_NODES = 20


class SimplifyConfig:
    """GPX track simplification + Google-Maps waypoint reduction.

    The GPX/PNG follow the FULL OSM route; we only Douglas-Peucker-simplify the
    track to drop redundant near-collinear points on straights while keeping sharp
    turns (points far from the chord). ONE tolerance sets the smoothness. The Maps
    URL takes the N most significant points (Visvalingam-Whyatt effective area).
    """

    # RDP tolerance in metres: track points within this of the chord are dropped.
    TRACK_TOLERANCE_M = 15.0


# --- Load-time invariants: fail loud on a bad edit ---------------------------
assert SurfaceConfig.SURFACE_FACTORS, "SURFACE_FACTORS must not be empty"
assert RoadConfig.ROAD_FACTORS, "ROAD_FACTORS must not be empty"
assert CostConfig.ELEV_COEFF > 0 and CostConfig.GRADE_COEFF > 0, "cost coeffs must be positive"
assert GmapsConfig.N_WAYPOINTS >= 2, "need at least origin + destination"

# Unknown/untagged surface fallbacks must sit BETWEEN the best and worst known surface factors
_SURFACE_MIN = min(SurfaceConfig.SURFACE_FACTORS.values())
_SURFACE_MAX = max(SurfaceConfig.SURFACE_FACTORS.values())
assert _SURFACE_MIN <= SurfaceConfig.UNKNOWN_OTHER < _SURFACE_MAX, "UNKNOWN_OTHER must be in [min, max) surface"
assert _SURFACE_MIN <= SurfaceConfig.UNKNOWN_TRACK_PATH < _SURFACE_MAX, "UNKNOWN_TRACK_PATH must be in [min, max)"
assert SurfaceConfig.UNKNOWN_OTHER <= SurfaceConfig.UNKNOWN_TRACK_PATH, "track/path fallback must be >= plain"
