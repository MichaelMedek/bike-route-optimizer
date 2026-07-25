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
    """Surface → penalty TIER (0 = good/paved, 1 = moderate/gravel, 2 = heavy/soft).

    Covers the surfaces actually seen on the Freudenstadt→Pforzheim corridor
    (measured). The tier multiplies the user's --extra_km_per_unpaved_km penalty:
    good adds nothing, moderate adds it once, heavy adds it twice.
    """

    SURFACE_TIER = {
        # good / paved → no unpaved penalty
        "asphalt": 0,
        "concrete": 0,
        "paved": 0,
        "paving_stones": 0,
        "sett": 0,
        "cobblestone": 0,
        # moderate / compacted-loose → penalty ×1
        "compacted": 1,
        "fine_gravel": 1,
        "gravel": 1,
        "pebblestone": 1,
        "unpaved": 1,
        # heavy / soft-natural → penalty ×2
        "ground": 2,
        "grass": 2,
        "dirt": 2,
        "earth": 2,
        "sand": 2,
        "mud": 2,
    }
    # Untagged surface (~37% of ways) → assume moderate.
    DEFAULT_TIER = 1


class RoadConfig:
    """Which highway classes count as a "main road" (attract the main-road penalty).

    Everything not listed here (cycleway, living_street, residential, tertiary,
    service, track, path, …) is a standard bike-friendly way with no penalty.
    Unknown/untagged highway (~0% in practice) is treated as a main road.
    """

    MAIN_ROADS = frozenset({"secondary", "primary", "unclassified"})


class RoutingDefaults:
    """Default values + safety limit for the three routing parameters."""

    EXTRA_KM_PER_UPHILL_100M = 5.0
    EXTRA_KM_PER_UNPAVED_KM = 1.0
    EXTRA_KM_PER_MAIN_ROAD_KM = 1.0
    MAX_EXTRA_KM = 1000.0  # hard ceiling for numerical stability (values above → error)


@dataclass(frozen=True)
class RoutingParams:
    """User-facing routing preferences, expressed as intuitive "extra km".

    Each is how many extra kilometres of virtual distance the router adds:
      * per 100 m of climb            (0 = ignore hills, high = force flat)
      * per km ridden on unpaved      (×1 moderate, ×2 heavy surfaces)
      * per km ridden on a main road  (secondary/primary/unclassified)
    0 is valid ("don't care"). Out-of-range values raise loudly (no silent clamp).
    """

    extra_km_per_uphill_100m: float
    extra_km_per_unpaved_km: float
    extra_km_per_main_road_km: float

    def __post_init__(self) -> None:
        for name, value in (
            ("extra_km_per_uphill_100m", self.extra_km_per_uphill_100m),
            ("extra_km_per_unpaved_km", self.extra_km_per_unpaved_km),
            ("extra_km_per_main_road_km", self.extra_km_per_main_road_km),
        ):
            if not 0.0 <= value <= RoutingDefaults.MAX_EXTRA_KM:
                raise ValueError(f"{name}={value} out of range [0, {RoutingDefaults.MAX_EXTRA_KM}]")


class CostConfig:
    """Per-edge cost = length + uphill + unpaved + main-road penalties (all in metres).

    All penalties are >= 0, so the cheapest possible edge is pure distance — which
    keeps the A* great-circle heuristic admissible with scale 1.0.
    """

    EDGE_COST = "custom_cost"  # stored per directed edge by assign_edge_costs
    UPHILL_REFERENCE_M = 100.0  # the "per 100 m of climb" reference rise


class SpeedConfig:
    """Surface- and grade-adaptive cycling speed (km/h) for time estimation.

    Two anchors, linearly interpolated: at 0 % grade the rider does the surface's
    base speed; at WALK_GRADE (a steep 12 %) they slow to WALK_KMH (pushing pace).
    Flat/downhill hold the base speed; above WALK_GRADE stay at WALK_KMH. Applied
    per edge, treating each edge as a single linear grade.
    """

    BASE_KMH_BY_TIER = {0: 25.0, 1: 20.0, 2: 15.0}  # good / moderate / heavy surface
    WALK_KMH = 5.0  # speed at WALK_GRADE and steeper (pushing the bike)
    WALK_GRADE = 0.12  # rise/run at which the rider drops to walking pace


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
    """Unit conversions for GPX timestamp synthesis (speed model lives in SpeedConfig)."""

    METERS_PER_KM = 1000.0
    SECONDS_PER_HOUR = 3600.0
    MINUTES_PER_HOUR = 60.0


class SanityConfig:
    """Thresholds for the runtime sanity checks."""

    # Below this many raw nodes the >50% simplify-shrink check is not meaningful.
    MIN_MEANINGFUL_NODES = 20


# --- Load-time invariants: fail loud on a bad edit ---------------------------
assert SurfaceConfig.SURFACE_TIER, "SURFACE_TIER must not be empty"
assert set(SurfaceConfig.SURFACE_TIER.values()) <= {0, 1, 2}, "surface tiers must be 0, 1, or 2"
assert SurfaceConfig.DEFAULT_TIER in {0, 1, 2}, "DEFAULT_TIER must be 0, 1, or 2"
assert RoadConfig.MAIN_ROADS, "MAIN_ROADS must not be empty"
assert GmapsConfig.N_WAYPOINTS >= 2, "need at least origin + destination"
assert RoutingDefaults.MAX_EXTRA_KM > 0, "MAX_EXTRA_KM must be positive"
assert all(
    0 <= value <= RoutingDefaults.MAX_EXTRA_KM
    for value in (
        RoutingDefaults.EXTRA_KM_PER_UPHILL_100M,
        RoutingDefaults.EXTRA_KM_PER_UNPAVED_KM,
        RoutingDefaults.EXTRA_KM_PER_MAIN_ROAD_KM,
    )
), "default routing params must be within [0, MAX_EXTRA_KM]"
assert set(SpeedConfig.BASE_KMH_BY_TIER) == {0, 1, 2}, "need a base speed for every surface tier"
assert all(speed > SpeedConfig.WALK_KMH for speed in SpeedConfig.BASE_KMH_BY_TIER.values()), "base speeds > walk"
assert SpeedConfig.WALK_GRADE > 0, "WALK_GRADE must be a positive uphill grade"
