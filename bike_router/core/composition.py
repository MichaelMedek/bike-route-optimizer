"""Route composition — the THREE donut breakdowns (quality / grade / mode), one source.

Built once from the Track and consumed identically in all three places: the Streamlit donuts,
the PNG overlay text, and the CLI stdout. ``composition_rows`` is the ONE helper — donuts render
its (title, km, colour) rows, and ``format_composition`` prints the exact same rows as text, so
the three can never drift (they used to: donuts showed quality/grade/mode, text showed surface/road/mode).
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from bike_router.core.constants import Mode, Palette, WebMapConfig
from bike_router.core.geo import haversine_vec
from bike_router.core.track import Track, TrackPoint, classify_condition, classify_grade

# The three donut label→hex-colour maps, each from a single Palette/config source (no re-typed labels).
QUALITY_COLORS = {label: Palette.CONDITION_COLORS[label] for label in ("good", "unpaved", "main road")}
GRADE_COLORS = {label: Palette.GRADE_COLORS[label] for label in ("flat", "uphill", "downhill")}
MODE_COLORS = {
    WebMapConfig.MODE_DONUT_LABELS[Mode.BIKE]: Palette.START,  # bike route → blue
    WebMapConfig.MODE_DONUT_LABELS[Mode.RAIL]: Palette.RAIL,  # train path → purple
}
# Fixed display order per donut (mirrors best→worst / bike→train); labels absent from a route are skipped.
_QUALITY_ORDER = {label: i for i, label in enumerate(("good", "unpaved", "main road"))}
_GRADE_ORDER = {label: i for i, label in enumerate(("flat", "uphill", "downhill"))}
_MODE_ORDER = {label: i for i, label in enumerate(WebMapConfig.MODE_DONUT_LABELS.values())}


@dataclass(frozen=True)
class RouteComposition:
    """The three km breakdowns behind the three donuts (and the PNG/CLI text), the single source.

    ``by_quality_km``/``by_grade_km`` slice the pedalled (bike) legs; ``by_mode_km`` covers the whole
    route (bike route vs train path, station access-hops folding into bike route).
    """

    by_quality_km: dict[str, float]  # bike km per quality bucket (good / unpaved / main road)
    by_grade_km: dict[str, float]  # bike km per grade bucket (flat / uphill / downhill)
    by_mode_km: dict[str, float]  # whole-route km (bike route / train path)


def _leg_km_by(*, track: Track, bucket_of: Callable[[TrackPoint], str], bike_only: bool) -> dict[str, float]:
    """Great-circle km per bucket over the track's legs; ``bike_only`` skips non-bike points.

    Each point (after the first) is one edge's far end; its leg is the great-circle gap from the
    previous point (vectorized). One walk serves the quality, grade AND mode breakdowns (no duplication).
    """
    lats = np.array([p.lat for p in track.points], dtype=np.float64)
    lons = np.array([p.lon for p in track.points], dtype=np.float64)
    leg_km = haversine_vec(lat_a=lats[:-1], lon_a=lons[:-1], lat_b=lats[1:], lon_b=lons[1:]) / 1000.0
    by_km: dict[str, float] = {}
    for point, km in zip(track.points[1:], leg_km, strict=True):
        if bike_only and point.mode != str(Mode.BIKE):
            continue
        bucket = bucket_of(point)
        by_km[bucket] = by_km.get(bucket, 0.0) + float(km)
    return by_km


def _quality_bucket(point: TrackPoint) -> str:
    """Quality label for a pedalled point; main-road (± unpaved) folds into the dominant "main road"."""
    condition = classify_condition(mode=point.mode, surface_bad=point.surface_bad, road_bad=point.road_bad)
    return "main road" if condition in ("main road", "main road + unpaved") else condition


def _mode_bucket(point: TrackPoint) -> str:
    """Mode label for any point: a rail leg is "train path", everything else "bike route"."""
    return WebMapConfig.MODE_DONUT_LABELS[Mode.RAIL if point.mode == str(Mode.RAIL) else Mode.BIKE]


def route_composition(track: Track) -> RouteComposition:
    """The three km breakdowns (quality, grade, mode) from ONE walk of the track's legs."""
    return RouteComposition(
        by_quality_km=_leg_km_by(track=track, bucket_of=_quality_bucket, bike_only=True),
        by_grade_km=_leg_km_by(
            track=track, bucket_of=lambda p: classify_grade(mode=p.mode, grade=p.grade), bike_only=True
        ),
        by_mode_km=_leg_km_by(track=track, bucket_of=_mode_bucket, bike_only=False),
    )


def composition_rows(comp: RouteComposition) -> tuple[tuple[str, dict[str, float], dict[str, str]], ...]:
    """The THREE donut rows — (title, km-breakdown, label→hex colours) — the ONE shared source.

    Donuts render each row as a chart; the PNG overlay and the CLI print the same rows as text
    (format_composition). Order is fixed per donut so quality/grade read best→worst, mode bike→train.
    """
    return (
        ("Quality", _ordered(comp.by_quality_km, _QUALITY_ORDER), QUALITY_COLORS),
        ("Grade", _ordered(comp.by_grade_km, _GRADE_ORDER), GRADE_COLORS),
        ("Mode", _ordered(comp.by_mode_km, _MODE_ORDER), MODE_COLORS),
    )


def _ordered(by_km: dict[str, float], order: dict[str, int]) -> dict[str, float]:
    """Re-key a km breakdown into the donut's fixed label order (labels absent from the route drop)."""
    return {label: by_km[label] for label in sorted(by_km, key=lambda label: order[label])}


def _percent_lines(by_km: dict[str, float]) -> list[str]:
    """Format a km breakdown as ``  label: NN%`` lines (percent of the category total), in dict order."""
    total = sum(by_km.values())
    return [f"  {label}: {km / total * 100:.0f}%" for label, km in by_km.items()]


def format_composition(comp: RouteComposition) -> str:
    """The three donut rows as percentage text for the CLI + PNG overlay — SAME titles/km as the donuts."""
    lines: list[str] = []
    for title, by_km, _colors in composition_rows(comp=comp):
        lines.append(f"{title}:")
        lines += _percent_lines(by_km=by_km)
    return "\n".join(lines)
