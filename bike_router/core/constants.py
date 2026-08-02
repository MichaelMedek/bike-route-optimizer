"""Central configuration for the bicycle route optimizer."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

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


class Schema:
    """On-disk GeoParquet column names — the SINGLE source for both the writer (build) and the
    reader (inference), so a rename can never drift between graph_writer and graph_store.
    """

    OSMID = "osmid"
    LAT = "lat"
    LON = "lon"
    ELEVATION_M = "elevation_m"
    NODE_TYPE = "node_type"
    STATION_NAME = "station_name"
    FROM_NODE = "from_node"
    TO_NODE = "to_node"
    KEY = "key"
    LENGTH_M = "length_m"
    HEIGHT_DIFF_M = "height_diff_m"
    SURFACE = "surface"
    HIGHWAY = "highway"
    MODE = "mode"
    GEOMETRY_WKT = "geometry_wkt"
    # OSMnx in-memory node/edge attrs (x=lon, y=lat) + the polyline attr; distinct from the on-disk names.
    GEOMETRY = "geometry"


class Condition:
    """Route-segment CONDITION labels (road-QUALITY scale) — the ONE source for classify/colour/donut.

    "main road + unpaved" folds into "main road" for display; keys of Palette.CONDITION_COLORS.
    """

    TRAIN = "train"
    GOOD = "good"
    UNPAVED = "unpaved"
    MAIN_ROAD = "main road"
    MAIN_ROAD_UNPAVED = "main road + unpaved"


class Grade:
    """Route-segment GRADE labels (road-GRADE scale) — keys of Palette.GRADE_COLORS."""

    TRAIN = "train"
    FLAT = "flat"
    UPHILL = "uphill"
    DOWNHILL = "downhill"


class SurfaceLabel:
    """Our INTERNAL display words (tooltips + tier-label swatches) — NOT OSM tag values.

    Deliberately distinct from the raw OSM ``surface`` values ("paved"/"unpaved") that key the tier maps.
    """

    PAVED = "paved road"
    UNPAVED = "unpaved path"
    QUIET_WAY = "quiet way"
    MAIN_ROAD = "main road"


class SessionKey:
    """Streamlit session_state keys shared by the app shell + the pure swap helper (one source)."""

    START_BOX = "start_box"
    END_BOX = "end_box"
    START_BOX_RESOLVED = "start_box_resolved"
    END_BOX_RESOLVED = "end_box_resolved"
    START_LATLON = "start_latlon"
    END_LATLON = "end_latlon"
    RESULT = "result"


# EuroDEM + OSMnx graphs are WGS84 lon/lat; the ONE CRS string, shared by build steps.
WGS84_CRS = "EPSG:4326"
# Python logging format shared by the CLI entry + the web app's one-time setup.
LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"
# The "name" field/tag — the OSM name column (builder), Photon feature name (geocoding), and the
# deck.gl picked-datum / marker key (ui). One string across the app's several "name" touch-points.
NAME_KEY = "name"
# Endpoint human labels — the origin/destination, shown as the geocode-error field name (pipeline)
# and the Start input box label (web app). Destination has no cross-file dup but pairs here for clarity.
START_LABEL = "Start"
DESTINATION_LABEL = "Destination"
# Streamlit button ``type`` for the red primary action (the Bahnhof suggestion pick). Typed as the
# exact Literal streamlit's API expects so mypy accepts it whether or not streamlit stubs are present.
ST_PRIMARY: Literal["primary"] = "primary"
# Coordinate-range assertion messages, shared by the scalar (gmaps) + vectorized (geo) guards.
LAT_OUT_OF_RANGE = "latitude out of range"
LON_OUT_OF_RANGE = "longitude out of range"
# Shared plot/chart display strings (the elevation axis label + the figure background), used by
# both the matplotlib debug PNG and the Plotly web profile.
ELEVATION_AXIS_LABEL = "Elevation (m)"
PLOT_BG = "white"


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
    # A "top" station is a local high point graded by the TWO standard mountaineering measures
    # Dominanz (topographic isolation): it must be the highest station within this radius.
    # Schartenhöhe (prominence): it must rise this far above the LOWEST station in that radius.
    TOP_STATION_DOMINANCE_KM = 10.0
    TOP_STATION_PROMINENCE_M = 100.0


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
    TILE_SUFFIX = ".parquet"  # per-tile file extension (shared by the writer + the reader glob)
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
        Condition.TRAIN: RAIL,
        Condition.GOOD: BLUE,
        Condition.UNPAVED: ORANGE,
        Condition.MAIN_ROAD: RED,
        Condition.MAIN_ROAD_UNPAVED: RED,
    }
    # Route-segment GRADE → hex, the road-GRADE scale (flat blue / uphill red / downhill green);
    # a train has no rider-felt grade so it keeps its purple, same as the quality scale.
    GRADE_COLORS = {Grade.TRAIN: RAIL, Grade.FLAT: BLUE, Grade.UPHILL: RED, Grade.DOWNHILL: GREEN}

    @staticmethod
    def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        """(r, g, b) for a '#rrggbb' string — the one hex→RGB conversion for the whole app."""
        h = hex_color.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def color_tier(weight: float) -> int:
    """Binary colour bucket for a continuous cost weight: 0 (good) iff round(weight) == 0, else 1 (bad).

    Caps the physically-ordered weight to two optical classes — even a very bad surface (weight rounding
    to 2, 3, …) still shows as the single "bad" colour, so colour never gains a third meaning.
    """
    return 0 if round(weight) == 0 else 1


def surface_weight_from_crr(crr: float) -> float:
    """Cost weight (extra equivalent-km per km) from a raw Crr: ROLLING_SHARE·(crr/Crr_asphalt − 1), floored at 0.

    Rolling energy is Crr·m·g·d, but rolling is only ~40% of total propulsive power at touring speed (rest
    is aero+drivetrain), so the Crr excess is scaled by ROLLING_SHARE — not applied full (Wikipedia power model).
    """
    return max(0.0, _ROLLING_SHARE * (crr / _CRR_ASPHALT - 1.0))


# Reference anchors for the surface weight formula (module scope so the class-body comprehension sees them).
_CRR_ASPHALT = 0.005  # DIRECTLY-MEASURED touring-tyre Crr on real asphalt (de.wikipedia Rollwiderstand, Tour mag)
# Rolling resistance is ~42% of total propulsive power at 20 km/h touring (Wikipedia power model / gribble.org);
# the Crr excess is scaled by this so a rough surface's penalty reflects rolling's SHARE of effort, not 100%.
_ROLLING_SHARE = 0.4


class SurfaceConfig:
    """Surface → FROZEN raw Crr → COST weight (rolling-share-scaled formula) → COLOUR tier (capped).

    SURFACE_CRR holds frozen Crr per OSM surface, each tagged measured vs interp (only asphalt/gravel are
    real bike tests); SURFACE_WEIGHT is DERIVED via surface_weight_from_crr; SURFACE_TIER caps to good/bad.
    """

    # Reference: DIRECTLY-MEASURED touring-tyre Crr on real asphalt — the 0-weight anchor.
    CRR_ASPHALT = _CRR_ASPHALT
    # Raw rolling-resistance coefficient per OSM surface (touring tyre). FROZEN. Each line's ≤5-word tag is
    # HONEST: "measured" = a real bicycle test on THAT surface; "interp"/"est" = engineering estimate, NOT cited.
    SURFACE_CRR = {
        "asphalt": 0.005,  # measured: de.wiki Tour-mag 40mm
        "concrete": 0.0055,  # interp: just above asphalt
        "concrete:plates": 0.007,  # interp: jointed concrete
        "concrete:lanes": 0.007,  # interp: jointed concrete
        "asphalt:lanes": 0.005,  # measured: = asphalt
        "paved": 0.0055,  # interp: generic sealed
        "paving_stones": 0.011,  # interp: ~2x asphalt (joints)
        "sett": 0.020,  # interp: impedance, no bike test
        "cobblestone": 0.030,  # interp: high impedance est.
        "unhewn_cobblestone": 0.040,  # interp: roughest, extrapolated
        "chipseal": 0.008,  # interp: rough asphalt
        "bricks": 0.012,  # interp: ~paving_stones
        "wood": 0.006,  # interp: boardwalk gaps est.
        "metal": 0.006,  # interp: grating est.
        "compacted": 0.008,  # interp: packed-gravel low end
        "fine_gravel": 0.009,  # interp: compacted↔gravel
        "gravel": 0.010,  # measured: de.wiki gravel-bike Schotter
        "pebblestone": 0.017,  # interp: loose stone est.
        "unpaved": 0.013,  # interp: generic unpaved
        "grass_paver": 0.016,  # interp: pavers↔grass
        "stone": 0.020,  # interp: rough stone est.
        "metal_grid": 0.008,  # interp: open grating est.
        "shells": 0.013,  # interp: loose shell est.
        "ground": 0.015,  # interp: off-road proxy
        "dirt": 0.017,  # interp: off-road proxy
        "earth": 0.017,  # interp: = dirt/ground
        "grass": 0.035,  # interp: soft-turf est. (Omni 0.007 firm)
        "woodchips": 0.050,  # interp: loose substrate, extrapolated
    }
    # Cost weight = extra equivalent-km per km from rolling resistance (surface_weight_from_crr); this is
    # what the cost + speed read. A surface smoother than asphalt earns no detour credit (floored at 0).
    SURFACE_WEIGHT = {value: surface_weight_from_crr(crr=crr) for value, crr in SURFACE_CRR.items()}
    # Binary colour bucket, DERIVED via color_tier so it can never drift from the weight (0 paved/blue,
    # 1 unpaved/orange — a bad surface caps at 1 even if its weight rounds to 2+).
    SURFACE_TIER = {value: color_tier(weight=weight) for value, weight in SURFACE_WEIGHT.items()}
    # Untagged surface (~44% of length) → assume Crr 0.013 (generic unpaved) → pessimistic default weight.
    # (Class-conditional prior is a follow-up needing an artifact rebuild — see the paper's rollout.)
    DEFAULT_WEIGHT = surface_weight_from_crr(crr=0.013)  # generic "unpaved"
    DEFAULT_TIER = color_tier(weight=DEFAULT_WEIGHT)
    # Per-tier human label + swatch: 0 paved/blue, 1 unpaved/orange (the two capped colour classes).
    TIER_LABEL_COLORS = {
        0: (SurfaceLabel.PAVED, Palette.BLUE),
        1: (SurfaceLabel.UNPAVED, Palette.ORANGE),
    }


class RoadConfig:
    """Highway class → FROZEN cited revealed-preference detour weight (single source) → COLOUR tier (capped).

    ROAD_WEIGHT is the extra equivalent-km per km cyclists actually detour to avoid each class, from
    revealed-preference GPS route-choice (Broach/Dill/Gliebe 2012), NOT ordinal LTS; ROAD_TIER caps to quiet/main.
    """

    # Cost weight = extra equivalent-km per km, from REVEALED-PREFERENCE detour data (Broach/Dill/Gliebe 2012
    # "equivalent %-distance", bounded to [0,1] so no edge swamps surface+grade). FROZEN — inline source each.
    ROAD_WEIGHT = {
        "cycleway": 0.0,  # Broach: off-street path, best facility
        "path": 0.05,  # Broach: near-cycleway separated path
        "footway": 0.05,  # Broach: separated, walk-pace
        "bridleway": 0.1,  # Broach: off-road, unpaved-ish
        "steps": 0.1,  # off-road, dismount (no traffic)
        "pedestrian": 0.1,  # low-traffic shared zone
        "living_street": 0.1,  # Broach: bike-boulevard ~−11% discount
        "residential": 0.15,  # Broach: local-street baseline (+13.5%)
        "service": 0.2,  # Broach: baseline-plus, driveways/parking
        "track": 0.2,  # low-ADT rural, baseline-plus
        "unclassified": 0.2,  # Broach: low-ADT non-residential
        "road": 0.2,  # unknown-class, baseline-plus
        "tertiary": 0.45,  # Broach: ~10-20k ADT no-lane (+0.37)
        "tertiary_link": 0.45,  # inherits tertiary
        "secondary": 0.65,  # Broach: ~20k ADT arterial (climbs to +1.6)
        "secondary_link": 0.65,  # inherits secondary
        "primary": 0.85,  # Broach: 20-30k ADT no-facility (near-top)
        "primary_link": 0.85,  # inherits primary
        "trunk": 1.0,  # Broach: >30k ADT, largest avoidance
        "trunk_link": 1.0,  # inherits trunk
    }
    # Binary colour bucket, DERIVED via color_tier (never drifts): 0 quiet/blue, 1 main/red.
    ROAD_TIER = {value: color_tier(weight=weight) for value, weight in ROAD_WEIGHT.items()}
    # Untagged highway (~0% in practice) → assume worst (trunk-like) → weight 1.0, tier 1 (red, pessimistic).
    DEFAULT_WEIGHT = 1.0
    DEFAULT_TIER = color_tier(weight=DEFAULT_WEIGHT)
    # Per-tier human label + swatch (donut colours): tier 0 (quiet) blue, tier 1 (main) red.
    TIER_LABEL_COLORS = {0: (SurfaceLabel.QUIET_WAY, Palette.BLUE), 1: (SurfaceLabel.MAIN_ROAD, Palette.RED)}


class RoutingDefaults:
    """Safety limit for the routing parameters (per-param defaults live in PARAM_SPECS)."""

    MAX_EXTRA_KM = 100.0  # hard ceiling for numerical stability (values above → error)


@dataclass(frozen=True)
class RoutingParamSpec:
    """One user-facing routing knob — the SINGLE source CLI, web, and filename naming read.

    ``field`` is the RoutingParams attribute / CLI flag name; ``label`` + ``help`` drive the web
    slider and CLI --help; ``default`` seeds both; ``abbrev`` is the short filename token.
    """

    field: str
    label: str
    help: str
    default: float
    abbrev: str


# The five "extra km" preferences, defined ONCE — CLI args and web sliders both iterate this (no
# duplicated label/default/help). Each value is a DETOUR willingness: extra km you'd add to avoid one
# unit of the bad thing; 0 means "don't detour" (the thing still costs its real distance), not "free".
PARAM_SPECS = (
    RoutingParamSpec(
        field="extra_km_per_uphill_100m",
        label="Hill avoidance (extra km per 100 m climb)",
        help="How far out of your way you'd ride to dodge 100 m of climbing. 0 = shortest route, ignore hills; higher = detour to stay flat.",
        default=14.0,
        abbrev=Grade.UPHILL,
    ),
    RoutingParamSpec(
        field="extra_km_per_unpaved_km",
        label="Unpaved avoidance (extra km per unpaved km)",
        help="Extra km you'd ride to swap 1 km of gravel/dirt for pavement. 0 = don't avoid unpaved; higher = detour to stay paved.",
        default=1.0,
        abbrev="unpaved",
    ),
    RoutingParamSpec(
        field="extra_km_per_main_road_km",
        label="Main-road avoidance (extra km per km)",
        help="Extra km you'd ride to swap 1 km of busy road for a quiet one. 0 = don't avoid main roads; higher = detour for quiet ways.",
        default=1.0,
        abbrev="main",
    ),
    RoutingParamSpec(
        field="extra_km_per_rail_km",
        label="Train-distance cost (extra km per rail km)",
        help="Per-km cost of riding the train, like a fare. 0 = don't mind trains over similar-length biking; higher = avoid long train legs, bike instead.",
        default=1.0,
        abbrev=Mode.RAIL,
    ),
    RoutingParamSpec(
        field="extra_km_per_boarding",
        label="Train-boarding cost (extra km per boarding)",
        help="Flat cost of getting on a train once (a transfer boards again). 0 = board freely at any station, even multiple train legs; higher = avoid trains however long.",
        default=20.0,
        abbrev="boarding",
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
    # Rolling-window length (m) the displayed ascent/descent stats resample the REAL bike terrain onto,
    # to shed DEM coastline-paradox terracing without erasing real hills. ~7× the ~30 m EuroDEM posting;
    # corroborated by GraphHopper's 150 m avg / 60 m resample + BRouter ~100 m (see elevation-ascent-research.md).
    ASCENT_RESAMPLE_WINDOW_M = 200.0


class BuildValidationConfig:
    """STRICT build-time invariants on bike-edge geometry — a violation fails the build LOUD.

    Guards against the two corruption classes that shipped bad graphs: sparse polylines that shortcut
    across streets, and baked z that leaves the [endpoint-elevation] band (a tunnel/dip a bike can't take).
    """

    MAX_VERTEX_SPACING_M = 100.0  # no two consecutive bike-edge vertices may be farther apart than this
    ELEV_BAND_MARGIN_M = 30.0  # bike-edge z must stay within [min,max endpoint elev] ± this (DEM noise)


class SpeedConfig:
    """Surface- and grade-adaptive cycling speed (km/h). Base speed interpolates linearly with the
    continuous surface WEIGHT (0.0 → BASE_KMH_AT_WEIGHT0, SURFACE_WEIGHT_MAX → BASE_KMH_AT_WEIGHT_MAX);
    then a second linear ramp drops it to WALK_KMH at WALK_GRADE. Flat/downhill hold the surface base.
    """

    BASE_KMH_AT_WEIGHT0 = 25.0  # paved (weight 0.0) base speed
    BASE_KMH_AT_WEIGHT_MAX = 15.0  # roughest rideable surface base speed
    # Weight at which the base speed bottoms out: 1.0 ≈ dirt/rough natural (Crr ~0.017); smoother
    # surfaces interpolate up to paved, rougher (sett, grass, woodchips) clamp to BASE_KMH_AT_WEIGHT_MAX.
    SURFACE_WEIGHT_MAX = 1.0
    WALK_KMH = 5.0  # speed at WALK_GRADE and steeper (pushing the bike)
    WALK_GRADE = 0.12  # rise/run at which the rider drops to walking pace


class GmapsConfig:
    """Google Maps directions-URL output."""

    # N significant points → origin + (N-2) intermediate + destination.
    N_WAYPOINTS = 10
    MAX_INTERMEDIATE_WAYPOINTS = 9  # hard Google Maps api=1 limit
    # Interior waypoints closer than this to the previous kept point are dropped, so a
    # short leg isn't cluttered with near-identical points (origin + destination always kept).
    MIN_WAYPOINT_SPACING_KM = 5.0
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
    # Reverse geocoding restricted to real populated places (via repeated osm_tag).
    REVERSE_PLACE_TAGS = ("place:city", "place:town", "place:village", "place:suburb")
    STATION_OSM_TAG = "railway:station"  # railway stations (Bahnhof) — proposed first as they're the top picks
    TIMEOUT_S = 3.0
    # Reverse-geocode search radius (km, Photon allows 0–5000).
    REVERSE_RADIUS_KM = 10.0


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
    DEFAULT_PITCH = 0.0  # start top-down (still 3D-draggable); pitch>0 breaks deck.gl terrain-click coords
    DEFAULT_BEARING = 0.0
    # Rendered map height in the browser, pixels.
    MAP_HEIGHT_PX = 600
    # Route ribbon floats above the terrain mesh. BIKE segments size WIDTH by pipe-flow (area×speed
    # conserved, width ∝ 1/√speed): RIBBON_REF_SPEED_KMH draws RIBBON_REF_WIDTH_M, 4× slower → 2× wider.
    # RAIL + STATION segments draw the fixed RIBBON_REF_WIDTH_M (a train's pace isn't rider effort).
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
    # st_deckgl stamps a picked-datum event with deck.gl's OWN name — "deck-click-event", NOT
    # "click" (the frontend maps events=["click"] → "deck-click-event"). picked_station gates on it.
    DECK_CLICK_EVENT = "deck-click-event"
