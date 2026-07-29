"""Central configuration for the bicycle route optimizer."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from bike_router.errors import ParamOutOfRangeError

# Package + project roots
PACKAGE_DIR = Path(__file__).parent
PROJECT_ROOT = PACKAGE_DIR.parent

# The one build-data dir: the repo's gitignored data/ folder. The DEM and the prebuilt
# graph both live here — written by the offline scripts, read from EXACTLY here.
DATA_DIR = PROJECT_ROOT / "data"


class OutputConfig:
    """Where generated route artifacts live.

    Contents are gitignored (reproducible, not source); the empty dir is kept via .gitkeep.
    """

    OUTPUT_DIR = PROJECT_ROOT / "output"


class DEMConfig:
    """Elevation-model file path — the ONE fixed location, in the gitignored data/ dir.

    The DEM is a build-time input (elevation is baked into the prebuilt graph): the
    crop script writes here and the builder reads from here, nowhere else.
    """

    EURODEM_PATH = DATA_DIR / "region_dem.tif"


# Edge travel modes. `station` = the bike↔station-node access link (board/alight).
class Mode(StrEnum):
    """The three edge travel modes (StrEnum → members ARE strings on the graph/parquet)."""

    BIKE = "bike"
    RAIL = "rail"
    STATION = "station"


# Node kinds. Every node is EXACTLY one: a cycling node or a rail-station node. A
# station is a SEPARATE node from any bike node so a bike route can never pass through
# it — reaching it always crosses a station edge (which carries the boarding hassle).
class NodeType(StrEnum):
    """The two node kinds (StrEnum → members ARE strings on the graph/parquet)."""

    BIKE = "bike"
    RAIL = "rail"


class RailConfig:
    """Railway integration: how a train leg is built and timed.

    Rail edges are bidirectional and always present; sliders decide if A* uses
    them. Boarding wait hits TIME only, not cost.
    """

    RAIL_SPEED_KMH = 80.0  # average train speed for the ride-time estimate
    BOARDING_WAIT_S = 1800.0  # 30 min wait added once per boarding (time only)
    STATION_RADIUS_M = 200.0  # bike node ↔ station-node access-link distance
    STATION_MAX_ENTRANCES = 5  # declare up to this many nearest bike nodes as entrances
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
    GRAPH_DIR = DATA_DIR / "dach_graph"
    NODES_SUBDIR = "nodes"
    EDGES_SUBDIR = "edges"
    META_FILENAME = "meta.json"
    OVERVIEW_FILENAME = "dach_graph_overview.png"  # whole-network preview, written into the artifact dir

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
    # Min total bike-edge length (km) for an isolated bike component to be KEPT. A region is a clip:
    # bike roads legitimately split into pieces that connect only through neighbours, so we do NOT force
    # one component. Small strays (<this) are dropped as noise; a large island (e.g. lake/ocean island
    # roads, usable once you're there) is kept. Rail is exempt — it must always be ONE component.
    MIN_BIKE_COMPONENT_KM = 50.0
    # Reader tolerance: stored height_diff_m must match to_elev − from_elev within this
    # (metres), else the artifact is corrupt/stale and the load hard-fails.
    HEIGHT_DIFF_TOLERANCE_M = 0.5
    # Each region's synthetic station ids (-1, -2, …) are shifted into a private block by
    # (region_index * this) so regions never collide, regardless of build order.
    STATION_ID_BLOCK = 100_000_000

    # The DACH region actually built (WGS84 W,S,E,N degrees) — the single source both the
    # DEM crop and the build's coverage preflight use.
    DACH_BBOX_DEG = (5.9, 45.8, 17.2, 55.1)
    # The DEM is cropped this much wider than DACH_BBOX_DEG so nodes near the border always
    # have elevation coverage (consolidation can nudge a node slightly past the region edge).
    DEM_CROP_MARGIN_DEG = 0.4


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
    MAX_TRIP_KM = 200.0  # beyond this the corridor graph is too big / out of scope


class Palette:
    """All display colours, defined ONCE as hex; use ``hex_to_rgb`` where RGB is needed.

    Bad surface and bad road are DISTINCT red tones so the two conditions are tellable
    apart on the map/PNG (both still red to read as "avoid"); an edge that is BOTH gets a
    near-black red. Green = good, purple = trains, blue/cyan = start/end markers.
    """

    GOOD = "#2e7d32"  # green — good surface AND quiet road
    BAD_SURFACE = "#d63f15"  # red variant — unpaved/loose surface on a quiet road
    BAD_ROAD = "#c80d29"  # red variant — main road with a good surface
    BAD_BOTH = "#280303"  # near-black red — main road AND unpaved (worst)
    RAIL = "#9600c8"  # purple — trains only
    START = "#0096ff"  # blue — start marker
    END = "#00e5ff"  # cyan — destination marker

    # Route-segment CONDITION → hex.
    CONDITION_COLORS = {
        "train": RAIL,
        "good": GOOD,
        "unpaved": BAD_SURFACE,
        "main road": BAD_ROAD,
        "main road + unpaved": BAD_BOTH,
    }

    @staticmethod
    def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        """(r, g, b) for a '#rrggbb' string — the one hex→RGB conversion for the whole app."""
        h = hex_color.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


class SurfaceConfig:
    """Surface → penalty TIER (0 = paved/good, 1 = loose/moderate, 2 = rough/natural).

    ALLOWLIST: only the categories listed here enter the graph. Any OTHER named surface
    (mud/sand/rock/impassable/…) is dropped at build time; a missing/untagged surface is
    assumed DEFAULT_TIER and kept. The tier is a literal multiplier on the unpaved penalty:
    tier 0 rides free, tier 1 adds --extra_km_per_unpaved_km once, tier 2 doubles it (natural
    ground/dirt/grass — rideable but rough, per the OSM surface wiki).
    """

    SURFACE_TIER = {
        # tier 0 — paved / good (no unpaved penalty, colored green)
        "asphalt": 0,
        "concrete": 0,
        "concrete:plates": 0,
        "concrete:lanes": 0,
        "asphalt:lanes": 0,
        "paved": 0,
        "paving_stones": 0,
        "sett": 0,
        "cobblestone": 0,
        "unhewn_cobblestone": 0,
        "chipseal": 0,
        "bricks": 0,
        "wood": 0,
        "metal": 0,
        # tier 1 — loose / compacted-gravel (penalty ×1, colored red)
        "compacted": 1,
        "fine_gravel": 1,
        "gravel": 1,
        "pebblestone": 1,
        "unpaved": 1,
        "grass_paver": 1,
        "stone": 1,
        "metal_grid": 1,
        "shells": 1,
        # tier 2 — natural / rough but rideable (penalty ×2 — the tier IS the multiplier)
        "ground": 2,
        "dirt": 2,
        "earth": 2,
        "grass": 2,
        "woodchips": 2,
    }
    # Untagged surface (~35% of raw ways) → assume tier 1 (loose, pessimistic but rideable).
    DEFAULT_TIER = 1
    # Per-tier human label + swatch. Tier 2 is a COMPUTE-only split.
    TIER_LABEL_COLORS = {
        0: ("paved road", Palette.GOOD),
        1: ("unpaved path", Palette.BAD_SURFACE),
        2: ("unpaved path", Palette.BAD_SURFACE),  # COMPUTE-only, same optic
    }


class RoadConfig:
    """Highway class → penalty TIER (0 = quiet/bike-friendly, 1 = main road).

    Symmetric with SurfaceConfig: an ALLOWLIST where only listed highway classes enter
    the graph. Any OTHER named highway (motorway/raceway/…) is dropped at build time; a
    missing/untagged highway is assumed DEFAULT_TIER (main, pessimistic) and kept. Tier 0
    rides free; tier 1 adds --extra_km_per_main_road_km once.
    """

    ROAD_TIER = {
        # tier 0 — quiet / bike-friendly (no main-road penalty, colored green)
        "cycleway": 0,
        "path": 0,
        "footway": 0,
        "bridleway": 0,
        "steps": 0,
        "pedestrian": 0,
        "living_street": 0,
        "residential": 0,
        "service": 0,
        "track": 0,
        "tertiary": 0,
        "tertiary_link": 0,
        "road": 0,
        # tier 1 — main road (penalty ×1, kept, colored red)
        "trunk": 1,
        "primary": 1,
        "secondary": 1,
        "unclassified": 1,
        "trunk_link": 1,
        "primary_link": 1,
        "secondary_link": 1,
    }
    # Untagged highway (~0% in practice) → assume main road (kept, penalised).
    DEFAULT_TIER = 1
    # Per-tier human label + swatch (donut colours): tier 0 (quiet) green, tier 1 (main) crimson.
    TIER_LABEL_COLORS = {0: ("quiet way", Palette.GOOD), 1: ("main road", Palette.BAD_ROAD)}


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
        label="Uphill penalty (extra km per 100 m climb)",
        help="Extra km you'd ride to avoid every 100 m of climbing (0 = ignore hills; high = long detours to stay flat).",
        default=12.0,
    ),
    RoutingParamSpec(
        field="extra_km_per_unpaved_km",
        label="Unpaved penalty (extra km per unpaved km)",
        help="Extra km you'd ride to avoid 1 km of unpaved surface (0 = don't mind gravel; high = detour far to stay paved).",
        default=1.0,
    ),
    RoutingParamSpec(
        field="extra_km_per_main_road_km",
        label="Main road penalty (extra km per main-road km)",
        help="Extra km you'd ride to avoid 1 km on a busy main road (0 = don't mind them; high = detour far to avoid main roads).",
        default=1.0,
    ),
    RoutingParamSpec(
        field="extra_km_per_rail_km",
        label="Rail distance penalty (extra km per rail km)",
        help="Extra km you'd bike to avoid 1 km carried by train (0 = train distance is free; high = avoid long train legs).",
        default=1.0,
    ),
    RoutingParamSpec(
        field="extra_km_per_boarding",
        label="Train boarding penalty (extra km per boarding)",
        help="Extra km you'd bike to avoid boarding a train once (0 = board freely; high = avoid catching trains).",
        default=20.0,
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
                raise ParamOutOfRangeError(f"{spec.field}={value} out of range [0, {RoutingDefaults.MAX_EXTRA_KM}]")


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

    BASE_KMH_BY_TIER = {0: 25.0, 1: 20.0, 2: 15.0}  # paved / loose / natural-rough surface
    WALK_KMH = 5.0  # speed at WALK_GRADE and steeper (pushing the bike)
    WALK_GRADE = 0.12  # rise/run at which the rider drops to walking pace


class GmapsConfig:
    """Google Maps directions-URL output."""

    # N significant points → origin + (N-2) intermediate + destination.
    N_WAYPOINTS = 10
    MAX_INTERMEDIATE_WAYPOINTS = 9  # hard Google Maps api=1 limit
    # Interior waypoints closer than this to the previous kept point are dropped, so a
    # short leg isn't cluttered with near-identical points (origin + destination always kept).
    MIN_WAYPOINT_SPACING_KM = 1.0
    BASE_URL = "https://www.google.com/maps/dir/?api=1"
    TRAVEL_MODE = "bicycling"


class PlotConfig:
    """Debug elevation-heatmap PNG rendering."""

    # Elevation colormap: cividis (blue→yellow, colourblind-safe) — deliberately avoids
    # green/red/purple so it never clashes with the route's condition + rail colours.
    CMAP = "cividis"
    DPI = 200
    ROUTE_ZOOM_MARGIN = 0.08  # pad the debug plot's route bounds by this fraction of the span
    # Figure sized to each route's geographic aspect (OSMnx keeps the map equal-aspect), so
    # the map fills its axis in both dimensions for any route shape and the colorbar hugs it.
    MAP_LONG_IN = 8.5  # the map's LONG side (inches); the short side is derived from the aspect
    MAP_SHORT_MIN_IN = 4.5  # floor for the derived short side so an extreme route stays printable
    SIDE_MARGIN_IN = 1.5  # extra width for the colorbar + its label
    STATS_HEIGHT_IN = 3.0  # bottom stats-panel height (inches)


class NominatimConfig:
    """Geocoding via OpenStreetMap Nominatim."""

    USER_AGENT = "bike-route-optimizer/0.1 (https://github.com/MichaelMedek/bike-route-optimizer)"
    RATE_LIMIT_S = 1.0  # Nominatim usage policy: max 1 request / second


class PhotonConfig:
    """Search-as-you-type geocoding via Photon (photon.komoot.io).

    Photon is the OSM project built for autocomplete — Nominatim's policy forbids
    client-side typeahead, so it powers only the one-shot "Set start & end" geocode.
    Suggestions are biased/limited to the prebuilt graph's coverage bbox (from
    load_meta), so this holds no bbox of its own.
    """

    BASE_URL = "https://photon.komoot.io/api"
    LANG = "de"
    LIMIT = 5
    PLACE_OSM_TAG = "place"  # settlements, not POIs
    TIMEOUT_S = 3.0


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
    # Route ribbon floats above the terrain mesh. BIKE segments size their WIDTH INVERSELY to
    # speed (fluid-dynamics: slow spots flow through fatter pipes → wider ribbon): a segment at
    # RIBBON_REF_SPEED_KMH draws RIBBON_REF_WIDTH_M, half that speed → double the width. RAIL and
    # STATION segments draw the fixed RIBBON_REF_WIDTH_M (a train's pace isn't rider "effort").
    RIBBON_FLOAT_ABOVE_M = 100.0
    RIBBON_REF_SPEED_KMH = 20.0
    RIBBON_REF_WIDTH_M = 20.0
    RIBBON_MIN_PIXELS = 3
    # Endpoint markers keep their own blue/cyan; the ribbon itself is coloured by
    # CONDITION (green good / graded reds for bad) and purple for trains. All colours come
    # from the single Palette (hex), converted here via hex_to_rgb for the RGB pydeck/PNG APIs.
    START_COLOR = Palette.hex_to_rgb(hex_color=Palette.START)  # blue (start marker)
    END_COLOR = Palette.hex_to_rgb(hex_color=Palette.END)  # cyan (destination marker)
    RAIL_COLOR = Palette.hex_to_rgb(hex_color=Palette.RAIL)  # purple — trains only
    # Composition-donut mode display: two buckets only — pedalled vs train. Station
    # access-hops are negligible and fold into "bike route".
    # The ribbon/PNG use segment_color (condition-based), NOT this map.
    MODE_DONUT_LABELS = {Mode.BIKE: "bike route", Mode.RAIL: "train path"}
    MODE_DONUT_COLORS = {
        "bike route": Palette.hex_to_rgb(hex_color=Palette.START),  # blue
        "train path": RAIL_COLOR,  # purple
    }
    MARKER_RADIUS_M = 60.0
    MARKER_MIN_PIXELS = 8
    # Station hop-on/hop-off markers: smaller than the start/end markers, rail-coloured.
    STATION_MARKER_RADIUS_M = 35.0
    STATION_MARKER_MIN_PIXELS = 5
    # Zoom, same log formula as ski-resort's MapConfig.zoom_for_span_m:
    VIEWING_ZOOM = 12.0
    ZOOM_SPAN_ANCHOR_M = 8000.0  # a route this long fits at VIEWING_ZOOM
    ZOOM_STEPS_OUT = 4.0
    ZOOM_STEPS_IN = 3.0
    # Free, no-API-key basemap tiles (identical to the ski-resort 3D map): AWS Terrarium
    # elevation meshed by the decoder, OpenTopoMap draped as texture.
    TERRAIN_TILES_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
    TERRAIN_ELEVATION_DECODER = {"rScaler": 256, "gScaler": 1, "bScaler": 1 / 256, "offset": -32768}
    TEXTURE_TILES_URL = "https://a.tile.opentopomap.org/{z}/{x}/{y}.png"


# --- Load-time invariants: fail loud on a bad edit ---------------------------
assert SurfaceConfig.SURFACE_TIER, "SURFACE_TIER must not be empty"
assert set(SurfaceConfig.SURFACE_TIER.values()) <= {0, 1, 2}, "surface tiers must be 0, 1, or 2"
assert SurfaceConfig.DEFAULT_TIER in set(SurfaceConfig.SURFACE_TIER.values()), (
    "surface DEFAULT_TIER must be a real tier"
)
assert RoadConfig.ROAD_TIER, "ROAD_TIER must not be empty"
assert set(RoadConfig.ROAD_TIER.values()) <= {0, 1}, "road tiers must be 0 or 1"
assert RoadConfig.DEFAULT_TIER in {0, 1}, "road DEFAULT_TIER must be 0 or 1"
# labels + speeds must cover EVERY surface tier that exists (single source: the tier map)
assert set(SurfaceConfig.TIER_LABEL_COLORS) == set(SurfaceConfig.SURFACE_TIER.values()), (
    "surface labels must cover exactly the surface tiers in use"
)
assert set(RoadConfig.TIER_LABEL_COLORS) == set(RoadConfig.ROAD_TIER.values()), (
    "road labels must cover exactly the road tiers in use"
)
assert GmapsConfig.N_WAYPOINTS >= 2, "need at least origin + destination"
assert RoutingDefaults.MAX_EXTRA_KM > 0, "MAX_EXTRA_KM must be positive"
assert PARAM_SPECS, "PARAM_SPECS must not be empty"
assert all(0 <= spec.default <= RoutingDefaults.MAX_EXTRA_KM for spec in PARAM_SPECS), (
    "every param default must be within [0, MAX_EXTRA_KM]"
)
assert {spec.field for spec in PARAM_SPECS} == set(RoutingParams.__dataclass_fields__), (
    "PARAM_SPECS fields must match RoutingParams attributes exactly"
)
assert set(SpeedConfig.BASE_KMH_BY_TIER) == set(SurfaceConfig.SURFACE_TIER.values()), (
    "need a base speed for every surface tier in use"
)
assert all(speed > SpeedConfig.WALK_KMH for speed in SpeedConfig.BASE_KMH_BY_TIER.values()), "base speeds > walk"
assert SpeedConfig.WALK_GRADE > 0, "WALK_GRADE must be a positive uphill grade"
assert 0 < CorridorConfig.MIN_TRIP_KM < CorridorConfig.MAX_TRIP_KM, "trip bounds must be 0 < min < max"
assert CorridorConfig.HALF_WIDTH_KM > 0, "corridor half-width must be positive"
assert max(SpeedConfig.BASE_KMH_BY_TIER.values()) < RailConfig.RAIL_SPEED_KMH, "rail must be faster than any bike leg"
assert RailConfig.BOARDING_WAIT_S > 0 and RailConfig.STATION_RADIUS_M > 0, "rail waits/radius must be positive"
assert RailConfig.STATION_MAX_ENTRANCES >= 1, "must declare at least one entrance per station"
assert GraphConfig.CONSOLIDATION_TOLERANCE_M >= 0 and GraphConfig.TILE_DEG > 0, "graph tolerance/tile must be sane"
assert set(WebMapConfig.MODE_DONUT_COLORS) == set(WebMapConfig.MODE_DONUT_LABELS.values()), (
    "mode donut colours must key on exactly the display labels"
)
assert PhotonConfig.LIMIT > 0 and PhotonConfig.TIMEOUT_S > 0, "Photon limit/timeout must be positive"
