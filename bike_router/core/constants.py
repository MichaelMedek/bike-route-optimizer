"""Central configuration for the bicycle route optimizer."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from bike_router.core.errors import ParamOutOfRangeError

# Package + project roots (this file lives at bike_router/core/, so root is two levels up).
PACKAGE_DIR = Path(__file__).parent.parent
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
    RAIL_TAGS = ("rail", "light_rail", "narrow_gauge")  # OSM railway= values kept as routable track
    STATION_TAGS = ("station", "halt")  # OSM railway= values treated as boardable stops


class GraphConfig:
    """Prebuilt DACH bike+rail graph: on-disk layout and Hugging Face hosting. Built offline
    from Geofabrik .osm.pbf with DEM elevations baked in, intersections consolidated, stored
    as lat/lon-tiled GeoParquet read per corridor tile (mirrors the DEMConfig HF pattern).
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
    HF_MAX_WORKERS = 8  # snapshot_download concurrent file downloads (its own default)

    # Merge nodes within this radius (metres, UTM-projected). Benchmarked on the
    # Freudenstadt→Pforzheim corridor: 25 m is the largest tolerance keeping route
    # distance under ~2% error (50 m → 2.6%, 100 m → 12%) while 3.2× faster A*.
    CONSOLIDATION_TOLERANCE_M = 25.0
    # Coarse grid for tiling the parquet so a corridor reads only a few tiles.
    TILE_DEG = 0.5
    # Lat/lon decimal places kept when serializing geometry (6 dp ≈ 0.1 m).
    COORD_PRECISION = 6
    # Min bike-edge length (km) to KEEP an isolated bike component. A region is a clip, so bike roads
    # legitimately split; small strays (<this) are pruned as noise, large islands kept. Rail is exempt.
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
    """The "Schlauch" search corridor: a tight bike tube (~98% of the graph) + a wide rail
    tube (~1%, sparse so generous is cheap). Isotropic in km; trips outside MIN/MAX rejected.
    """

    BIKE_HALF_WIDTH_KM = 30.0  # bike tube half-width each side of the direct line
    BIKE_EXTEND_KM = 5.0  # extend the bike tube this far past each endpoint
    RAIL_HALF_WIDTH_KM = 80.0  # rail tube half-width (wide: rail is sparse, ~1% of edges)
    RAIL_EXTEND_KM = 50.0  # extend the rail tube this far past each endpoint
    MIN_TRIP_KM = 5.0  # too short to bother planning
    MAX_TRIP_KM = 300.0  # beyond this the corridor graph is too big / out of scope
    # Hard memory guard: ~669k edges ≈ 1.35 GB peak, well under the ~2.7 GB deploy ceiling.
    MAX_ROUTE_EDGES = 1_100_000


class Palette:
    """All display colours, defined ONCE as hex (use ``hex_to_rgb`` where RGB is needed). Two
    3-colour scales share these swatches: road-QUALITY (blue good / orange unpaved / red main)
    and road-GRADE (blue flat / red uphill / green downhill); purple = trains, blue/cyan = markers.
    """

    BLUE = "#1565c0"  # good surface + quiet road, and flat grade (high-contrast on terrain)
    ORANGE = "#ef7f18"  # unpaved (but not a main road)
    RED = "#c80d29"  # main road (or main road AND unpaved), and uphill grade
    GREEN = "#1b9e3f"  # downhill grade
    RAIL = "#9600c8"  # purple — trains only
    START = "#0096ff"  # blue — start marker
    END = "#00e5ff"  # cyan — destination marker

    # Route-segment CONDITION → hex, the road-QUALITY scale (3 bike colours + train purple).
    # "main road + unpaved" folds into the main-road red (a main road is the dominant hazard).
    CONDITION_COLORS = {
        "train": RAIL,
        "good": BLUE,
        "unpaved": ORANGE,
        "main road": RED,
        "main road + unpaved": RED,
    }
    # Route-segment GRADE → hex, the road-GRADE scale (flat blue / uphill red / downhill green);
    # a train has no rider-felt grade so it keeps its purple, same as the quality scale.
    GRADE_COLORS = {"train": RAIL, "flat": BLUE, "uphill": RED, "downhill": GREEN}

    @staticmethod
    def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        """(r, g, b) for a '#rrggbb' string — the one hex→RGB conversion for the whole app."""
        h = hex_color.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


class SurfaceConfig:
    """Surface → penalty TIER (0 = paved/good, 1 = loose/moderate, 2 = rough/natural).

    ALLOWLIST: only listed categories enter the graph (others dropped at build; untagged →
    DEFAULT_TIER). The tier is a literal multiplier on --extra_km_per_unpaved_km (0 free, 1 ×1, 2 ×2).
    """

    SURFACE_TIER = {
        # tier 0 — paved / good (no unpaved penalty, colored blue)
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
        0: ("paved road", Palette.BLUE),
        1: ("unpaved path", Palette.ORANGE),
        2: ("unpaved path", Palette.ORANGE),  # COMPUTE-only, same optic
    }


class RoadConfig:
    """Highway class → penalty TIER (0 = quiet/bike-friendly, 1 = main road). Symmetric with
    SurfaceConfig: an ALLOWLIST — listed classes enter the graph, others (motorway/raceway/…)
    are dropped, missing/untagged → DEFAULT_TIER (main, pessimistic). Tier 1 adds one main-road penalty.
    """

    ROAD_TIER = {
        # tier 0 — quiet / bike-friendly (no main-road penalty, colored blue)
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
    # Per-tier human label + swatch (donut colours): tier 0 (quiet) blue, tier 1 (main) red.
    TIER_LABEL_COLORS = {0: ("quiet way", Palette.BLUE), 1: ("main road", Palette.RED)}


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


# The five "extra km" preferences, defined ONCE. CLI args and web sliders both iterate
# this — no duplicated label/default/help anywhere. Each value is a DETOUR willingness:
# extra km you'd add on top of the real distance to avoid one unit of the bad thing; 0
# never means "free" (the thing still costs its real distance) — it means "don't detour".
PARAM_SPECS = (
    RoutingParamSpec(
        field="extra_km_per_uphill_100m",
        label="Hill avoidance (extra km per 100 m climb)",
        help="How far out of your way you'd ride to dodge 100 m of climbing. 0 = shortest route, ignore hills; higher = detour to stay flat.",
        default=12.0,
    ),
    RoutingParamSpec(
        field="extra_km_per_unpaved_km",
        label="Unpaved avoidance (extra km per unpaved km)",
        help="Extra km you'd ride to swap 1 km of gravel/dirt for pavement. 0 = don't avoid unpaved; higher = detour to stay paved.",
        default=1.0,
    ),
    RoutingParamSpec(
        field="extra_km_per_main_road_km",
        label="Main-road avoidance (extra km per km)",
        help="Extra km you'd ride to swap 1 km of busy road for a quiet one. 0 = don't avoid main roads; higher = detour for quiet ways.",
        default=1.0,
    ),
    RoutingParamSpec(
        field="extra_km_per_rail_km",
        label="Train-distance cost (extra km per rail km)",
        help="Per-km cost of riding the train, like a fare. 0 = don't mind trains over similar-length biking; higher = avoid long train legs, bike instead.",
        default=1.0,
    ),
    RoutingParamSpec(
        field="extra_km_per_boarding",
        label="Train-boarding cost (extra km per boarding)",
        help="Flat cost of getting on a train once (a transfer boards again). 0 = board freely at any station, even multiple train legs; higher = avoid trains however long.",
        default=15.0,
    ),
)


@dataclass(frozen=True)
class RoutingParams:
    """User-facing routing preferences, expressed as intuitive "extra km": how many extra
    virtual kilometres the router adds per unit of the bad thing (see PARAM_SPECS). 0 is valid
    ("don't care"); out-of-range values raise loudly (no silent clamp).
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

    All penalties are >= 0, so the cheapest possible edge is pure distance — Dijkstra on
    these non-negative costs is provably optimal (the CSR routing engine).
    """

    UPHILL_REFERENCE_M = 100.0  # the "per 100 m of climb" reference rise


class GradeConfig:
    """Grade classification for the road-GRADE colour scale + grade donut.

    An edge at/above +MARGIN uphill is red, at/below −MARGIN downhill is green, and everything
    strictly between is flat blue. One threshold both the colour scale and the donut read.
    """

    MARGIN = 0.02  # rise/run: |grade| BELOW this reads as flat (so only ~-1/0/+1% is flat, ≥2% slopes)


class SkiResortConfig:
    """Ski-resort extraction (build-time; scripts/build_ski_resort.py). Slopes use ONLY GradeConfig.MARGIN."""

    LIFT_MIN_GAIN_M = 200.0  # a lift must gain at least this much bottom→top, else it is invalid/dropped
    SLOPE_MAX_ASCENT_FRACTION = 0.10  # a slope's total climb may be at most this fraction of its total drop


class SpeedConfig:
    """Surface- and grade-adaptive cycling speed (km/h), two anchors linearly interpolated: at
    0 % grade the rider does the surface's base speed; at WALK_GRADE (a steep 12 %) they slow to
    WALK_KMH. Flat/downhill hold base; above WALK_GRADE stay WALK_KMH. Applied per edge as one grade.
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
    # blue/red/purple so it never clashes with the route's condition + rail colours.
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
    """Search-as-you-type geocoding via Photon (photon.komoot.io), the OSM autocomplete project.
    Nominatim's policy forbids client typeahead, so Photon powers only the one-shot geocode.
    Suggestions are biased to the graph's coverage bbox (from load_meta), so it holds no bbox.
    """

    BASE_URL = "https://photon.komoot.io/api"
    LANG = "de"
    LIMIT = 3  # max autocomplete suggestions shown under a place box
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

    # Opening camera: high above the Bodensee (Lake Constance, the DE/AT/CH tripoint ≈ DACH
    # centre), zoomed far out so the whole coverage region is in view before a route is set.
    DEFAULT_LAT = 47.6
    DEFAULT_LON = 9.4
    DEFAULT_ZOOM = 6.0  # far out (whole DACH); route framing uses VIEWING_ZOOM, not this
    DEFAULT_PITCH = 30.0  # deck.gl pitch: 0 = top-down, 90 = horizon
    DEFAULT_BEARING = 0.0
    # Rendered map height in the browser, pixels.
    MAP_HEIGHT_PX = 600
    # Route ribbon floats above the terrain mesh. BIKE segments size their WIDTH by pipe-flow
    # (water in a pipe: area×speed conserved, width = diameter so area ∝ width² → width ∝ 1/√speed):
    # a segment at RIBBON_REF_SPEED_KMH draws RIBBON_REF_WIDTH_M, 4× slower → 2× wider. RAIL and
    # STATION segments draw the fixed RIBBON_REF_WIDTH_M (a train's pace isn't rider "effort").
    RIBBON_FLOAT_ABOVE_M = 100.0
    RIBBON_REF_SPEED_KMH = 20.0
    RIBBON_REF_WIDTH_M = 40.0  # wide so the route reads clearly against the terrain texture
    RIBBON_MIN_PIXELS = 6
    # Endpoint markers keep their own blue/cyan; the ribbon itself is coloured by
    # CONDITION (blue good / graded reds for bad) and purple for trains. All colours come
    # from the single Palette (hex), converted here via hex_to_rgb for the RGB pydeck/PNG APIs.
    START_COLOR = Palette.hex_to_rgb(hex_color=Palette.START)  # blue (start marker)
    END_COLOR = Palette.hex_to_rgb(hex_color=Palette.END)  # cyan (destination marker)
    RAIL_COLOR = Palette.hex_to_rgb(hex_color=Palette.RAIL)  # purple — the train ribbon + donut only
    # Composition "by mode" display labels: two buckets only — pedalled vs train (station access-hops
    # fold into "bike route"). The label→colour map lives in core/composition (MODE_COLORS, one source).
    MODE_DONUT_LABELS = {Mode.BIKE: "bike route", Mode.RAIL: "train path"}
    # ONE colour for EVERY map marker (start, end, stations, waypoints) — the start/end blue.
    MARKER_COLOR = Palette.hex_to_rgb(hex_color=Palette.START)  # blue — all markers
    ENDPOINT_RADIUS_M = 70.0  # start/end: slightly bigger than the intermediate markers
    ENDPOINT_MIN_PIXELS = 9
    # Intermediate markers (board/alight stations AND gmaps waypoints)
    WAYPOINT_RADIUS_M = 50.0
    WAYPOINT_MIN_PIXELS = 7
    # Zoom, same log formula as ski-resort's MapConfig.zoom_for_span_m. Set one level below the
    # raw span fit so the framed route — which usually extends past the straight start→end span.
    VIEWING_ZOOM = 11.0
    ZOOM_SPAN_ANCHOR_M = 8000.0  # a route this long fits at VIEWING_ZOOM
    ZOOM_STEPS_OUT = 4.0
    ZOOM_STEPS_IN = 3.0
    # Free, no-API-key basemap tiles (identical to the ski-resort 3D map): AWS Terrarium
    # elevation meshed by the decoder, OpenTopoMap draped as texture.
    TERRAIN_TILES_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
    TERRAIN_ELEVATION_DECODER = {"rScaler": 256, "gScaler": 1, "bScaler": 1 / 256, "offset": -32768}
    TEXTURE_TILES_URL = "https://a.tile.opentopomap.org/{z}/{x}/{y}.png"
