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
    """The "Schlauch" search corridor around the straight start→dest line.

    A tube of half-width HALF_WIDTH_KM each side of the direct line (isotropic in
    real distance — N-S and E-W get the same km width; the km→degree conversion
    corrects for longitude shrink at the route's latitude). Trips shorter than
    MIN_TRIP_KM or longer than MAX_TRIP_KM are rejected up front.
    """

    HALF_WIDTH_KM = 22.0  # search this far each side of the direct line
    MIN_TRIP_KM = 5.0  # too short to bother planning
    MAX_TRIP_KM = 100.0  # beyond this the corridor graph is too big / out of scope


class SurfaceConfig:
    """Surface → penalty TIER (0 = good/paved, 1 = moderate/gravel, 2 = heavy/soft).

    Covers the surfaces actually seen on the Freudenstadt→Pforzheim corridor
    (measured). Tier 0 rides free; tier 1 adds the user's --extra_km_per_unpaved_km
    once. Tier 2 (soft natural ground) is EXCLUDED from the graph entirely — those
    ways are dropped before routing so a route never runs over mud/sand/grass.
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
        # heavy / soft-natural → EXCLUDED from the graph (see EXCLUDED_TIER)
        "ground": 2,
        "grass": 2,
        "dirt": 2,
        "earth": 2,
        "sand": 2,
        "mud": 2,
    }
    # Untagged surface (~37% of ways) → assume moderate (kept, penalised).
    DEFAULT_TIER = 1
    # Edges whose surface tier is this high are removed from the routable graph.
    EXCLUDED_TIER = 2


class RoadConfig:
    """Which highway classes count as a "main road" (attract the main-road penalty).

    Everything not listed here (cycleway, living_street, residential, tertiary,
    service, track, path, …) is a standard bike-friendly way with no penalty.
    Unknown/untagged highway (~0% in practice) is treated as a main road.
    """

    MAIN_ROADS = frozenset({"secondary", "primary", "unclassified"})


class RoutingDefaults:
    """Safety limit for the routing parameters (per-param defaults live in PARAM_SPECS)."""

    MAX_EXTRA_KM = 100.0  # hard ceiling for numerical stability (values above → error)


@dataclass(frozen=True)
class RoutingParamSpec:
    """One user-facing routing knob — the SINGLE source both CLI and web read.

    ``field`` is the RoutingParams attribute / CLI flag name; ``label`` + ``help``
    drive the web slider and CLI --help; ``default`` seeds both.
    """

    field: str
    label: str
    help: str
    default: float


# The three "extra km" preferences, defined ONCE. CLI args and web sliders both
# iterate this — no duplicated label/default/help text anywhere.
PARAM_SPECS = (
    RoutingParamSpec(
        field="extra_km_per_uphill_100m",
        label="Uphill penalty",
        help="Extra km you'd ride to avoid every 100 m of climbing (0 = ignore hills).",
        default=5.0,
    ),
    RoutingParamSpec(
        field="extra_km_per_unpaved_km",
        label="Unpaved penalty",
        help="Extra km you'd ride to avoid 1 km of unpaved surface (0 = don't mind gravel).",
        default=1.0,
    ),
    RoutingParamSpec(
        field="extra_km_per_main_road_km",
        label="Main road penalty",
        help="Extra km you'd ride to avoid 1 km on a busy main road (0 = don't mind them).",
        default=1.0,
    ),
)


@dataclass(frozen=True)
class RoutingParams:
    """User-facing routing preferences, expressed as intuitive "extra km".

    Each is how many extra kilometres of virtual distance the router adds per unit
    of the bad thing (see PARAM_SPECS). 0 is valid ("don't care"); out-of-range
    values raise loudly (no silent clamp).
    """

    extra_km_per_uphill_100m: float
    extra_km_per_unpaved_km: float
    extra_km_per_main_road_km: float

    def __post_init__(self) -> None:
        for spec in PARAM_SPECS:
            value = getattr(self, spec.field)
            if not 0.0 <= value <= RoutingDefaults.MAX_EXTRA_KM:
                raise ValueError(f"{spec.field}={value} out of range [0, {RoutingDefaults.MAX_EXTRA_KM}]")


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

    # exactly N significant points → origin + (N-2) waypoints + destination.
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


class WebMapConfig:
    """Defaults for the Streamlit 3D map viewer (app_webmap.py).

    Default camera looks north over Freudenstadt with a pitch; after a route it
    reframes to the start/end midpoint, zoom derived from the direct-line span.
    """

    # Default camera (Freudenstadt, Germany), north-up with a tilt.
    DEFAULT_LAT = 48.4634
    DEFAULT_LON = 8.4111
    DEFAULT_PITCH = 50.0
    DEFAULT_BEARING = 0.0
    # Rendered map height in the browser, pixels.
    MAP_HEIGHT_PX = 600
    # Route ribbon rendered as a PathLayer floating above the terrain mesh.
    RIBBON_FLOAT_ABOVE_M = 100.0
    RIBBON_WIDTH_M = 20.0
    RIBBON_MIN_PIXELS = 3
    RIBBON_COLOR = (255, 90, 0)
    # Zoom, same log formula as ski-resort's MapConfig.zoom_for_span_m:
    VIEWING_ZOOM = 12.0
    ZOOM_SPAN_ANCHOR_M = 8000.0  # a route this long fits at VIEWING_ZOOM
    ZOOM_STEPS_OUT = 4.0
    ZOOM_STEPS_IN = 3.0


# --- Load-time invariants: fail loud on a bad edit ---------------------------
assert SurfaceConfig.SURFACE_TIER, "SURFACE_TIER must not be empty"
assert set(SurfaceConfig.SURFACE_TIER.values()) <= {0, 1, 2}, "surface tiers must be 0, 1, or 2"
assert SurfaceConfig.DEFAULT_TIER in {0, 1, 2}, "DEFAULT_TIER must be 0, 1, or 2"
assert RoadConfig.MAIN_ROADS, "MAIN_ROADS must not be empty"
assert GmapsConfig.N_WAYPOINTS >= 2, "need at least origin + destination"
assert RoutingDefaults.MAX_EXTRA_KM > 0, "MAX_EXTRA_KM must be positive"
assert PARAM_SPECS, "PARAM_SPECS must not be empty"
assert all(0 <= spec.default <= RoutingDefaults.MAX_EXTRA_KM for spec in PARAM_SPECS), (
    "every param default must be within [0, MAX_EXTRA_KM]"
)
assert {spec.field for spec in PARAM_SPECS} == set(RoutingParams.__dataclass_fields__), (
    "PARAM_SPECS fields must match RoutingParams attributes exactly"
)
assert set(SpeedConfig.BASE_KMH_BY_TIER) == {0, 1, 2}, "need a base speed for every surface tier"
assert all(speed > SpeedConfig.WALK_KMH for speed in SpeedConfig.BASE_KMH_BY_TIER.values()), "base speeds > walk"
assert SpeedConfig.WALK_GRADE > 0, "WALK_GRADE must be a positive uphill grade"
assert 0 < CorridorConfig.MIN_TRIP_KM < CorridorConfig.MAX_TRIP_KM, "trip bounds must be 0 < min < max"
assert CorridorConfig.HALF_WIDTH_KM > 0, "corridor half-width must be positive"
