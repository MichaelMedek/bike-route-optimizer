"""Central configuration for the bicycle route optimizer."""

import os
from dataclasses import dataclass
from enum import StrEnum
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
    """Elevation-model file path.

    The DEM is a build-time input (elevation is baked into the prebuilt graph), so
    only the offline builder reads it — there is no inference-time DEM download.
    """

    EURODEM_PATH = DATA_DIR / "region_dem.tif"


# Edge travel modes. `transfer` = the bike↔station link (walk to/from the platform).
class Mode(StrEnum):
    """The three edge travel modes (StrEnum → members ARE strings on the graph/parquet)."""

    BIKE = "bike"
    RAIL = "rail"
    TRANSFER = "transfer"


class RailConfig:
    """Railway integration: how a train leg is built and timed.

    Rail edges are bidirectional and always present; sliders decide if A* uses
    them. Boarding wait hits TIME only, not cost.
    """

    RAIL_SPEED_KMH = 80.0  # average train speed for the ride-time estimate
    BOARDING_WAIT_S = 1800.0  # 30 min wait added once per boarding (time only)
    STATION_TRANSFER_RADIUS_M = 500.0  # bike node ↔ station link distance
    RAIL_TAGS = ("rail",)  # OSM railway= values kept as routable track
    STATION_TAGS = ("station", "halt")  # OSM railway= values treated as boardable stops


class GraphConfig:
    """Prebuilt DACH bike+rail graph: on-disk layout and Hugging Face hosting.

    Built offline from Geofabrik .osm.pbf, DEM elevations baked in, intersections
    consolidated, stored as lat/lon-tiled GeoParquet read per corridor tile.
    Mirrors the DEMConfig HF pattern.
    """

    # Pre-saved (bundled) in the repo — small enough to ship, so no HF download at
    # inference. The offline builder writes here; the app reads it directly.
    GRAPH_DIR = PROJECT_ROOT / "data" / "dach_graph"
    NODES_SUBDIR = "nodes"
    EDGES_SUBDIR = "edges"
    META_FILENAME = "meta.json"

    HF_REPO_ID = "MichaelMedek/dach_bike_graph"
    HF_FILENAME = META_FILENAME  # the whole snapshot is pulled; meta anchors the download

    # Merge nodes within this radius (metres, UTM-projected). Benchmarked on the
    # Freudenstadt→Pforzheim corridor: 25 m is the largest tolerance keeping route
    # distance under ~2% error (50 m → 2.6%, 100 m → 12%) while 3.2× faster A*.
    CONSOLIDATION_TOLERANCE_M = 25.0
    # Coarse grid for tiling the parquet so a corridor reads only a few tiles.
    TILE_DEG = 0.5
    # Lat/lon decimal places kept when serializing geometry (6 dp ≈ 0.1 m).
    COORD_PRECISION = 6
    # Reader tolerance: stored height_diff_m must match to_elev − from_elev within this
    # (metres), else the artifact is corrupt/stale and the load hard-fails.
    HEIGHT_DIFF_TOLERANCE_M = 0.5


class GeoConfig:
    """Spherical-Earth constants for great-circle distance."""

    EARTH_RADIUS_M = 6_371_000.0
    # metres per degree of latitude at the equator
    METERS_PER_DEGREE_EQUATOR = 111_320.0


class CorridorConfig:
    """The "Schlauch" search corridor around the straight start→dest line.

    A tube HALF_WIDTH_KM each side of the direct line (isotropic in real km; the
    km→degree conversion corrects longitude shrink). Trips outside MIN/MAX_TRIP_KM
    are rejected up front.
    """

    HALF_WIDTH_KM = 22.0  # search this far each side of the direct line
    MIN_TRIP_KM = 5.0  # too short to bother planning
    MAX_TRIP_KM = 100.0  # beyond this the corridor graph is too big / out of scope


# Quality colours — green good, red bad.
GOOD_COLOR = "#2e7d32"  # green
BAD_COLOR = "#c62828"  # red


class SurfaceConfig:
    """Surface → penalty TIER (0 = good/paved, 1 = moderate/gravel, 2 = heavy/soft).

    Tier 0 rides free; tier 1 adds --extra_km_per_unpaved_km once; tier 2 (soft
    natural ground) is EXCLUDED from the graph so a route never runs over mud/sand.
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
    # Per-tier human label + swatch: only tier 0 (paved) is good/green, the rest bad/red.
    TIER_LABEL_COLORS = {0: ("paved", GOOD_COLOR), 1: ("gravel/unpaved", BAD_COLOR), 2: ("rough", BAD_COLOR)}


class RoadConfig:
    """Which highway classes count as a "main road" (attract the main-road penalty).

    Everything not listed here (cycleway, living_street, residential, tertiary,
    service, track, path, …) is a standard bike-friendly way with no penalty.
    Unknown/untagged highway (~0% in practice) is treated as a main road.
    """

    MAIN_ROADS = frozenset({"secondary", "primary", "unclassified"})
    # is-main-road bool → human label + swatch (quiet good/green, main bad/red).
    LABEL_COLORS = {False: ("quiet way", GOOD_COLOR), True: ("main road", BAD_COLOR)}


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
    RoutingParamSpec(
        field="extra_km_per_rail_km",
        label="Rail distance penalty",
        help="Extra km you'd bike to avoid 1 km carried by train (≈ ticket cost per km; high = avoid trains).",
        default=1.0,
    ),
    RoutingParamSpec(
        field="extra_km_per_boarding",
        label="Train boarding penalty",
        help="Extra km you'd bike rather than board a train once (the wait/hassle; high = avoid boarding).",
        default=10.0,
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
    extra_km_per_rail_km: float
    extra_km_per_boarding: float

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

    # N significant points → origin + (N-2) intermediate + destination.
    N_WAYPOINTS = 10
    MAX_INTERMEDIATE_WAYPOINTS = 9  # hard Google Maps api=1 limit
    BASE_URL = "https://www.google.com/maps/dir/?api=1"
    TRAVEL_MODE = "bicycling"


class PlotConfig:
    """Debug elevation-heatmap PNG rendering."""

    CMAP = "plasma"
    DPI = 200


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
    # ONE blue is the whole route's colour — start marker, bike + transfer ribbon,
    # and the debug-PNG route all use START_COLOR (single source). Only trains differ.
    START_COLOR = (0, 150, 255)  # blue
    END_COLOR = (0, 229, 255)  # cyan (destination marker)
    RAIL_COLOR = (150, 0, 200)  # purple — trains only
    MODE_COLORS: dict[str, tuple[int, int, int]] = {
        Mode.BIKE: START_COLOR,
        Mode.RAIL: RAIL_COLOR,
        Mode.TRANSFER: START_COLOR,
    }
    MARKER_RADIUS_M = 60.0
    MARKER_MIN_PIXELS = 8
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
assert max(SpeedConfig.BASE_KMH_BY_TIER.values()) < RailConfig.RAIL_SPEED_KMH, "rail must be faster than any bike leg"
assert RailConfig.BOARDING_WAIT_S > 0 and RailConfig.STATION_TRANSFER_RADIUS_M > 0, "rail waits/radius must be positive"
assert GraphConfig.CONSOLIDATION_TOLERANCE_M >= 0 and GraphConfig.TILE_DEG > 0, "graph tolerance/tile must be sane"
assert set(WebMapConfig.MODE_COLORS) == set(Mode), "MODE_COLORS must have a color for every Mode"
