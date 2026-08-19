"""Cross-session comparison and longitudinal trends.

Guardrails implemented here (not in templates, so they cannot be skipped):

* Sessions compare within one activity comparison-group by default; mixing
  groups requires the explicit ``mixed`` override and always attaches the
  posture warning.
* SDNN is strongly recording-length dependent — comparing it across
  materially different durations attaches a warning to the SDNN row.
* Longitudinal RMSSD is tracked as ln(RMSSD) (raw RMSSD is right-skewed);
  the rolling baseline band only appears once five sessions exist, and is
  labelled provisional before that.

All per-minute series come from the ``.npz`` caches written at processing
time — comparison never reruns the pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.activities.metrics import lowest_sustained_hr
from app.activities.registry import resolve_profile
from app.models import Session
from app.pipeline.rr import RRSeries
from app.processing import cache_path_for

#: SDNN grows with recording length (Task Force 1996; Shaffer & Ginsberg,
#: Front Public Health 2017;5:258). Durations whose ratio exceeds this are
#: "materially different" and the SDNN comparison gets a warning.
SDNN_DURATION_RATIO_GUARD = 1.25

#: Sessions needed before a rolling baseline band is drawn. Below this the
#: baseline is provisional — three points do not deserve confident bands.
BASELINE_MIN_SESSIONS = 5

#: Rolling window (in sessions) for the baseline band.
BASELINE_WINDOW = 5

MIXED_ACTIVITY_WARNING = (
    "You are comparing sessions of different activity types. Posture and motion "
    "dominate HRV far more than fitness does — a supine RMSSD and a standing RMSSD "
    "are not the same measurement. Differences below are expected physiology of the "
    "activities themselves and say little about change over time."
)


def comparison_key_of(session: Session) -> str | None:
    if session.activity_type is None:
        return None
    return resolve_profile(session.activity_type).comparison_key


def load_cached_rr(session: Session) -> RRSeries | None:
    """Rebuild the RR series from the session's .npz cache, if present."""
    if not session.stored_path:
        return None
    path = cache_path_for(session)
    if not Path(path).exists():
        return None
    data = np.load(path)
    return RRSeries(
        rr_ms=data["rr_ms"],
        t_s=data["rr_t_s"],
        discontinuity=data["rr_discontinuity"],
        n_dropped_excluded=0,
        n_dropped_ceiling=0,
        n_dropped_floor=0,
    )


#: Lowest sustained (rolling 60 s mean) HR — shared with the processing layer.
resting_hr_bpm = lowest_sustained_hr


# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------


@dataclass
class MetricRow:
    label: str
    unit: str
    values: list[float | None]
    deltas: list[float | None]  # vs the first (earliest) session; None in col 0
    fmt: str = "{:.1f}"
    note: str | None = None

    def formatted(self, i: int) -> str:
        v = self.values[i]
        return self.fmt.format(v) if v is not None else "—"

    def formatted_delta(self, i: int) -> str | None:
        d = self.deltas[i]
        if d is None:
            return None
        return ("+" if d >= 0 else "") + self.fmt.format(d)


@dataclass
class ComparisonData:
    sessions: list[Session]
    rows: list[MetricRow]
    warnings: list[str] = field(default_factory=list)
    mixed: bool = False


class ComparisonError(Exception):
    """User-visible reason a comparison cannot be built."""


def build_comparison(sessions: list[Session], mixed: bool = False) -> ComparisonData:
    if len(sessions) < 2:
        raise ComparisonError("Pick at least two sessions to compare.")
    if any(s.processing_status != "done" for s in sessions):
        raise ComparisonError("All compared sessions must have finished processing.")

    sessions = sorted(sessions, key=lambda s: s.recorded_at or s.created_at)
    keys = {comparison_key_of(s) for s in sessions}
    warnings: list[str] = []

    if len(keys) > 1:
        if not mixed:
            raise ComparisonError(
                "These sessions belong to different activity types. Comparison "
                "defaults to same-activity groups because posture and motion dominate "
                "HRV; tick the cross-activity override if you really want to mix them."
            )
        warnings.append(MIXED_ACTIVITY_WARNING)

    durations = [s.analysed_s or s.duration_s or 0.0 for s in sessions]
    sdnn_note = None
    positive = [d for d in durations if d > 0]
    if positive and max(positive) / max(min(positive), 1.0) > SDNN_DURATION_RATIO_GUARD:
        mins = [f"{d / 60.0:.0f}" for d in durations]
        sdnn_note = (
            f"SDNN is strongly recording-length dependent and these durations differ "
            f"materially ({' vs '.join(mins)} min) — this row is not a fair comparison."
        )
        warnings.append(sdnn_note)

    resting = [
        resting_hr_bpm(rr) if (rr := load_cached_rr(s)) is not None else None
        for s in sessions
    ]

    def metric(attr: str) -> list[float | None]:
        return [getattr(s.metrics, attr, None) if s.metrics else None for s in sessions]

    def row(
        label: str,
        unit: str,
        values: list[float | None],
        fmt: str = "{:.1f}",
        note: str | None = None,
    ) -> MetricRow:
        base = values[0]
        deltas: list[float | None] = [None]
        for v in values[1:]:
            deltas.append(v - base if v is not None and base is not None else None)
        return MetricRow(label, unit, values, deltas, fmt, note)

    rmssd = metric("rmssd_ms")
    ln_rmssd = [math.log(v) if v and v > 0 else None for v in rmssd]

    rows = [
        row("Duration analysed", "min", [d / 60.0 if d else None for d in durations]),
        row("Excluded time", "s", [s.excluded_s for s in sessions], "{:.0f}"),
        row("Beats corrected", "%", [s.beats_corrected_pct for s in sessions], "{:.2f}"),
        row("Mean HR", "bpm", metric("mean_hr_bpm"), "{:.0f}"),
        row(
            "Ectopic beats",
            "/h",
            metric("ectopy_per_hour"),
            "{:.2f}",
            "confirmed count over analysed time; single-session differences "
            "sit inside normal day-to-day variability",
        ),
        row("Resting HR (lowest sustained)", "bpm", resting, "{:.0f}"),
        row("RMSSD", "ms", rmssd),
        row(
            "ln(RMSSD)",
            "ln ms",
            ln_rmssd,
            "{:.2f}",
            "the trend-tracking form — raw RMSSD is right-skewed",
        ),
        row("SDNN (whole record)", "ms", metric("sdnn_ms"), "{:.1f}", sdnn_note),
        row("pNN50", "%", metric("pnn50_pct")),
        row("SD1", "ms", metric("sd1_ms")),
        row("SD2", "ms", metric("sd2_ms")),
        row("SD1/SD2", "", metric("sd1_sd2_ratio"), "{:.2f}"),
        row("Sample entropy", "", metric("sample_entropy"), "{:.2f}"),
        row("DFA α1", "", metric("dfa_alpha1"), "{:.2f}"),
        row(
            "LF/HF",
            "",
            metric("lf_hf_ratio"),
            "{:.2f}",
            "engine-dependent, descriptive only",
        ),
    ]

    reduced = [s for s in sessions if s.reduced_confidence]
    if reduced:
        warnings.append(
            f"Session(s) {', '.join(str(s.id) for s in reduced)} carry the "
            "reduced-confidence badge (>5% of beats corrected) — read their values "
            "accordingly."
        )

    return ComparisonData(sessions=sessions, rows=rows, warnings=warnings, mixed=len(keys) > 1)


# ---------------------------------------------------------------------------
# Longitudinal trends
# ---------------------------------------------------------------------------


@dataclass
class TrendPoint:
    session_id: int
    recorded_at: object  # datetime
    resting_hr_bpm: float | None
    rmssd_ms: float | None
    ln_rmssd: float | None
    #: Environment context at recording time (opt-in lookup; often None).
    env_temp_c: float | None = None
    env_pm25_ugm3: float | None = None


@dataclass
class TrendSeries:
    """One tracked quantity with an optional rolling baseline band."""

    values: list[float | None]
    band_mean: list[float | None] = field(default_factory=list)
    band_lo: list[float | None] = field(default_factory=list)
    band_hi: list[float | None] = field(default_factory=list)


@dataclass
class TrendData:
    comparison_key: str
    points: list[TrendPoint]
    resting: TrendSeries | None = None
    ln_rmssd: TrendSeries | None = None
    provisional: bool = True


def build_trend(sessions: list[Session], comparison_key: str) -> TrendData:
    done = sorted(
        (s for s in sessions if s.processing_status == "done"),
        key=lambda s: s.recorded_at or s.created_at,
    )
    points: list[TrendPoint] = []
    for s in done:
        rr = load_cached_rr(s)
        rmssd = s.metrics.rmssd_ms if s.metrics else None
        points.append(
            TrendPoint(
                session_id=s.id,
                recorded_at=s.recorded_at or s.created_at,
                resting_hr_bpm=resting_hr_bpm(rr) if rr is not None else None,
                rmssd_ms=rmssd,
                ln_rmssd=math.log(rmssd) if rmssd and rmssd > 0 else None,
                env_temp_c=s.env_temp_c,
                env_pm25_ugm3=s.env_pm25_ugm3,
            )
        )

    provisional = len(points) < BASELINE_MIN_SESSIONS
    data = TrendData(comparison_key=comparison_key, points=points, provisional=provisional)
    data.resting = _series([p.resting_hr_bpm for p in points], provisional)
    data.ln_rmssd = _series([p.ln_rmssd for p in points], provisional)
    return data


def _series(values: list[float | None], provisional: bool) -> TrendSeries:
    series = TrendSeries(values=values)
    if provisional:
        return series
    for i in range(len(values)):
        window = [v for v in values[max(0, i - BASELINE_WINDOW + 1) : i + 1] if v is not None]
        if i + 1 >= BASELINE_WINDOW and len(window) >= 3:
            mean = float(np.mean(window))
            sd = float(np.std(window, ddof=1))
            series.band_mean.append(mean)
            series.band_lo.append(mean - 1.96 * sd)
            series.band_hi.append(mean + 1.96 * sd)
        else:
            series.band_mean.append(None)
            series.band_lo.append(None)
            series.band_hi.append(None)
    return series
