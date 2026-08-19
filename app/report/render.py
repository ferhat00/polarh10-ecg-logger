"""Server-side report rendering: a self-contained HTML page.

Figures are inlined as base64 PNGs and the IBM Plex fonts as base64 woff2, so
the rendered file is a single artifact that works fully offline and can be
downloaded as-is. No JS charting, no external requests of any kind.
"""

from __future__ import annotations

import base64
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.activities import metrics as activity_metrics
from app.activities.base import ActivityAnalysis
from app.activities.registry import ResolvedActivity
from app.ingest.loader import LoadedRecording
from app.pipeline.hrv import HRVResult
from app.pipeline.process import PipelineResult
from app.report import figures as fig
from app.report import sleep_figures
from app.screening.flags import NO_FLAGS_STATEMENT, ScreeningFlag
from app.sleep.orchestrator import SleepAnalysis

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_FONT_DIR = Path(__file__).resolve().parent.parent / "static" / "vendor" / "fonts"

#: Why interval measurements are absent — used in the report limits section
#: and wherever a user asks for them.
INTERVAL_METRICS_EXPLANATION = (
    "QRS width, QT/QTc, PR interval, and ECG axis are not computed anywhere in this "
    "tool. The H10 samples at ~130 Hz, so one sample spans 7.7 ms and a QRS complex "
    "is only ~12 samples wide; delineation algorithms will still return numbers at "
    "this resolution, but they are quantisation artifacts, not physiology. Interval "
    "measurement needs 500–1000 Hz and multiple leads."
)

#: Unconditional disclaimer for the sleep section — it renders whenever any
#: staging output does, in the same spirit as the screening disclaimer that
#: travels with every flag.
SLEEP_DISCLAIMER = (
    "Sleep stages here are estimated from heartbeat patterns (and movement, when "
    "an accelerometer file is attached), not from brain activity. Even the best "
    "published heart-beat-based model validated on this strap reaches ~80% "
    "epoch agreement with laboratory polysomnography (Topalidis et al. 2023, "
    "Sensors 23(5):2390) — treat every number as an estimate with real error. "
    "This analysis cannot detect sleep apnea, periodic limb movements, or the "
    "difference between quiet wakefulness and sleep misperception, and none of "
    "its output is a diagnosis or a substitute for a sleep study."
)

#: Above this duration the per-minute table aggregates to 5-minute rows.
MINUTE_TABLE_AGGREGATE_AFTER_MIN = 180.0
MINUTE_TABLE_BUCKET_MIN = 5


@dataclass
class ReportMeta:
    """Session metadata shown in the report header."""

    person_name: str = ""
    activity_name: str = ""
    recorded_at: dt.datetime | None = None
    context_note: str | None = None
    original_filename: str = ""
    file_sha256: str | None = None
    reduced_confidence: bool = False
    # Structured context (docs/CONTEXT_METRICS.md); all optional.
    body_position: str | None = None
    body_position_source: str | None = None
    alcohol_drinks_24h: int | None = None
    sleep_quality_1_5: int | None = None
    #: One-line environment summary ("21.3 °C · RH 46% · …"), pre-formatted
    #: by the caller so the template stays dumb; None when nothing fetched.
    environment_line: str | None = None


@dataclass
class MinuteRow:
    minute: int
    mean_hr_bpm: float | None = None
    sdnn_ms: float | None = None
    rmssd_ms: float | None = None
    excluded_s: float = 0.0
    sqi_mean: float | None = None
    #: Row span in minutes (>1 when a long recording aggregates the table).
    span_min: int = 1


@dataclass
class ReportData:
    """Everything the template needs, assembled by :func:`build_report_html`."""

    meta: ReportMeta
    result: PipelineResult
    hrv: HRVResult  # censored (post-suppression) version
    analysis: ActivityAnalysis
    activity: ResolvedActivity | None
    flags: list[ScreeningFlag]
    figures: dict[str, fig.Figure | None] = field(default_factory=dict)
    strips: list[fig.Figure] = field(default_factory=list)
    minute_rows: list[MinuteRow] = field(default_factory=list)
    kpis: list[tuple[str, str, str]] = field(default_factory=list)  # (label, value, note)
    extras_rows: list[tuple[str, str]] = field(default_factory=list)
    # --- sleep section (sleep sessions only) ------------------------------
    sleep: SleepAnalysis | None = None
    sleep_figures: dict[str, fig.Figure | None] = field(default_factory=dict)
    sleep_kpis: list[tuple[str, str, str]] = field(default_factory=list)
    sleep_disclaimer: str = SLEEP_DISCLAIMER
    fonts_css: str = ""
    no_flags_statement: str = NO_FLAGS_STATEMENT
    interval_explanation: str = INTERVAL_METRICS_EXPLANATION
    generated_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))


def build_report_html(
    rec: LoadedRecording,
    result: PipelineResult,
    hrv_censored: HRVResult,
    analysis: ActivityAnalysis,
    activity: ResolvedActivity | None,
    flags: list[ScreeningFlag],
    meta: ReportMeta,
    sleep: SleepAnalysis | None = None,
) -> str:
    """Assemble figures + tables and render the standalone report page."""
    data = ReportData(
        meta=meta,
        result=result,
        hrv=hrv_censored,
        analysis=analysis,
        activity=activity,
        flags=flags,
        sleep=sleep,
    )
    data.strips = _build_strips(rec, result)
    data.figures = {
        "hr": fig.hr_timeseries(
            result.rr, result.quality.excluded_segments, rec.duration_s
        ),
        "poincare": fig.poincare(result.rr, hrv_censored),
        "rr_hist": fig.rr_histogram(result.rr),
        "sdnn_windows": fig.sdnn_per_window(hrv_censored),
        "spectrum": fig.spectrum(hrv_censored),
        "template": fig.beat_template(
            result.morphology,
            result.detection.ecg_clean,
            result.sampling_rate_hz,
            np.clip(result.correction.peaks_corrected, 0, len(rec.time_s) - 1),
        ),
        "quality": fig.quality_traces(result.quality),
    }
    data.minute_rows = _minute_table(result)
    data.kpis = _kpis(rec, result, hrv_censored, analysis)
    data.extras_rows = _format_extras(analysis.extras)
    data.fonts_css = _fonts_css()

    if sleep is not None:
        data.sleep_figures = {
            "hypnogram": sleep_figures.hypnogram_figure(
                sleep.hypnograms, sleep.override_epochs_n
            ),
            "stage_distribution": sleep_figures.stage_distribution(sleep.summaries),
            "movement": (
                sleep_figures.movement_trace(sleep.acc_epochs)
                if sleep.acc_epochs is not None
                else None
            ),
        }
        data.sleep_kpis = _sleep_kpis(sleep)

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    env.filters["num"] = _fmt_num
    return env.get_template("report/report.html").render(d=data)


# ---------------------------------------------------------------------------
# Assembly helpers
# ---------------------------------------------------------------------------


def _build_strips(rec: LoadedRecording, result: PipelineResult) -> list[fig.Figure]:
    """First / middle / last 10 s strips, plus one centred on the worst
    morphology outlier when outliers exist."""
    duration = rec.duration_s
    strip_len = 10.0
    outlier_times = rec.time_s[
        np.clip(result.correction.peaks_corrected, 0, len(rec.time_s) - 1)
    ][result.morphology.outlier_mask[: len(result.correction.peaks_corrected)]]

    wanted: list[tuple[float, str]] = [(0.0, "Opening strip")]
    if duration > 3 * strip_len:
        wanted.append(((duration - strip_len) / 2.0, "Mid-session strip"))
    if duration > 2 * strip_len:
        wanted.append((duration - strip_len, "Closing strip"))

    strips: list[fig.Figure] = []
    for start, label in wanted:
        s = fig.rhythm_strip(
            rec.time_s,
            result.detection.ecg_clean,
            start_s=start,
            duration_s=min(strip_len, duration),
            peak_times_s=result.peak_times_s,
            outlier_times_s=outlier_times,
            label=label,
            with_calibration=label == "Opening strip",
        )
        if s is not None:
            strips.append(s)

    if len(outlier_times) > 0:
        peaks = np.clip(result.correction.peaks_corrected, 0, len(rec.time_s) - 1)
        corr = result.morphology.correlations[: len(peaks)]
        masked = np.where(result.morphology.outlier_mask[: len(corr)], corr, np.nan)
        worst_beat = int(np.nanargmin(masked))
        t_worst = float(rec.time_s[int(peaks[worst_beat])])
        start = max(0.0, min(t_worst - strip_len / 2.0, duration - strip_len))
        s = fig.rhythm_strip(
            rec.time_s,
            result.detection.ecg_clean,
            start_s=start,
            duration_s=min(strip_len, duration),
            peak_times_s=result.peak_times_s,
            outlier_times_s=outlier_times,
            label="Worst morphology outlier",
            with_calibration=False,
        )
        if s is not None:
            strips.append(s)
    return strips


def _minute_table(result: PipelineResult) -> list[MinuteRow]:
    """Per-minute rows; long recordings aggregate to 5-minute rows.

    Bucket membership is resolved with ``searchsorted``/pre-binning: an 8 h
    night has ~480 rows over ~35 k intervals and ~5,800 quality windows,
    where per-row scans are quadratic.
    """
    rr = result.rr
    if len(rr) == 0:
        return []
    end_min = int(float(rr.t_s[-1]) // 60) + 1
    bucket = (
        MINUTE_TABLE_BUCKET_MIN if end_min > MINUTE_TABLE_AGGREGATE_AFTER_MIN else 1
    )
    bucket_s = bucket * 60.0
    n_rows = int(np.ceil(end_min / bucket))

    rmssd_t, rmssd_v = activity_metrics.per_minute_rmssd(rr)
    rmssd_rows: dict[int, list[float]] = {}
    for t, v in zip(rmssd_t, rmssd_v, strict=True):
        rmssd_rows.setdefault(int(t // bucket_s), []).append(v)
    sdnn_rows: dict[int, list[float]] = {}
    for t, v in zip(
        result.hrv.sdnn_window_t_s, result.hrv.sdnn_per_window_ms, strict=True
    ):
        sdnn_rows.setdefault(int(t // bucket_s), []).append(v)
    sqi_rows: dict[int, list[float]] = {}
    for w in result.quality.windows:
        if not np.isnan(w.sqi_mean):
            sqi_rows.setdefault(int(w.start_s // bucket_s), []).append(w.sqi_mean)

    edges = np.arange(n_rows + 1) * bucket_s
    lo = np.searchsorted(rr.t_s, edges[:-1], side="left")
    hi = np.searchsorted(rr.t_s, edges[1:], side="left")
    csum = np.concatenate(([0.0], np.cumsum(rr.rr_ms)))

    rows: list[MinuteRow] = []
    for b in range(n_rows):
        w0, w1 = edges[b], edges[b + 1]
        row = MinuteRow(minute=b * bucket, span_min=bucket)
        n = int(hi[b] - lo[b])
        if n >= 5 * bucket:
            row.mean_hr_bpm = float(60000.0 / ((csum[hi[b]] - csum[lo[b]]) / n))
        if b in sdnn_rows:
            row.sdnn_ms = float(np.mean(sdnn_rows[b]))
        if b in rmssd_rows:
            row.rmssd_ms = float(np.mean(rmssd_rows[b]))
        row.excluded_s = sum(
            max(0.0, min(e, w1) - max(s, w0))
            for s, e, _ in result.quality.excluded_segments
        )
        if b in sqi_rows:
            row.sqi_mean = float(np.mean(sqi_rows[b]))
        rows.append(row)
    return rows


def _sleep_kpis(sleep: SleepAnalysis) -> list[tuple[str, str, str]]:
    """(label, value, note) cards for the primary engine's summary."""
    summary = sleep.primary_summary()
    if summary is None:
        return []
    engine_note = f"engine: {sleep.primary_engine}"

    def minutes(v: float | None) -> str:
        if v is None:
            return "—"
        return f"{int(v // 60)} h {v % 60:02.0f} min" if v >= 60 else f"{v:.0f} min"

    kpis = [
        ("Time in bed", minutes(summary.tib_min), "recording span (≈ lights-off proxy)"),
        ("Total sleep time", minutes(summary.tst_min), engine_note),
        (
            "Sleep efficiency",
            f"{summary.sleep_efficiency_pct:.0f}%",
            "TST / time in bed",
        ),
        (
            "Sleep onset",
            minutes(summary.sol_min) if summary.sol_min is not None else "—",
            "first sleep epoch",
        ),
        (
            "WASO",
            minutes(summary.waso_min) if summary.waso_min is not None else "—",
            "wake after sleep onset",
        ),
        (
            "Awakenings",
            str(summary.awakenings_n) if summary.awakenings_n is not None else "—",
            "wake bouts ≥ 30 s",
        ),
    ]
    if summary.deep_min is not None:
        kpis.append(("Deep sleep", minutes(summary.deep_min), engine_note))
    if summary.rem_min is not None:
        kpis.append(("REM sleep", minutes(summary.rem_min), engine_note))
    if summary.rem_latency_min is not None:
        kpis.append(("REM latency", minutes(summary.rem_latency_min), "onset → first REM"))
    if summary.unscored_min > 0:
        kpis.append(("Unscored", minutes(summary.unscored_min), "insufficient signal"))
    return kpis


def _kpis(
    rec: LoadedRecording,
    result: PipelineResult,
    hrv: HRVResult,
    analysis: ActivityAnalysis,
) -> list[tuple[str, str, str]]:
    q = result.quality
    kpis: list[tuple[str, str, str]] = [
        (
            "Excluded time",
            f"{q.excluded_total_s:.0f} s",
            f"of {q.excluded_total_s + q.analysed_total_s:.0f} s — "
            f"{q.analysed_total_s:.0f} s analysed",
        ),
        ("Beats analysed", f"{hrv.n_beats}", ""),
        (
            "Beats corrected",
            f"{result.correction.pct_corrected:.2f}%",
            "reduced confidence" if result.correction.reduced_confidence else "",
        ),
    ]
    if hrv.mean_hr_bpm:
        kpis.append(("Mean HR", f"{hrv.mean_hr_bpm:.0f} bpm", ""))
    if hrv.rmssd_ms is not None:
        kpis.append(("RMSSD", f"{hrv.rmssd_ms:.1f} ms", ""))
    if hrv.sdnn_ms is not None:
        note = "trend-inclusive"
        if hrv.sdnn_per_window_ms:
            note += (
                f"; per-minute {min(hrv.sdnn_per_window_ms):.0f}–"
                f"{max(hrv.sdnn_per_window_ms):.0f} ms"
            )
        kpis.append(("SDNN (whole record)", f"{hrv.sdnn_ms:.1f} ms", note))
    if result.events is not None:
        ev = result.events
        rate = f" ({ev.per_hour:.2f}/h)" if ev.per_hour is not None else ""
        kpis.append(
            (
                "Ectopic beats",
                f"{ev.n_confirmed}{rate}",
                "confirmed by correction class + prematurity + motion veto",
            )
        )
    kpis.append(("Engine", result.engine_used, result.fallback_reason or ""))
    return kpis


def _format_extras(extras: dict[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for key, value in extras.items():
        label = key.replace("_", " ")
        if value is None:
            rows.append((label, "—"))
        elif isinstance(value, dict):
            inner = " · ".join(f"{k}: {_fmt_num(v)}" for k, v in value.items())
            rows.append((label, inner))
        elif isinstance(value, list):
            if len(value) <= 8:
                rows.append((label, ", ".join(_fmt_num(v) for v in value)))
            # long series are plotted, not tabulated
        else:
            rows.append((label, _fmt_num(value)))
    return rows


def _fmt_num(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}" if abs(value) < 1000 else f"{value:.0f}"
    return str(value)


def _fonts_css() -> str:
    """@font-face rules with the vendored woff2 files inlined as base64."""
    faces = (
        ("IBM Plex Sans", 400, "IBMPlexSans-Regular.woff2"),
        ("IBM Plex Sans", 600, "IBMPlexSans-SemiBold.woff2"),
        ("IBM Plex Mono", 400, "IBMPlexMono-Regular.woff2"),
        ("IBM Plex Mono", 500, "IBMPlexMono-Medium.woff2"),
    )
    rules = []
    for family, weight, filename in faces:
        path = _FONT_DIR / filename
        if not path.exists():
            continue
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        rules.append(
            f"@font-face {{ font-family: '{family}'; font-weight: {weight}; "
            f"font-style: normal; font-display: swap; "
            f"src: url(data:font/woff2;base64,{b64}) format('woff2'); }}"
        )
    return "\n".join(rules)
