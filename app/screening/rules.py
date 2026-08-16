"""The screening rules. Rule-based only; every threshold cited in
:mod:`app.screening.thresholds`.

Each rule is a pure function over pipeline outputs so it can be tested with
constructed series. :func:`run_screening` composes them for a full pipeline
result.
"""

from __future__ import annotations

import numpy as np

from app.pipeline.process import PipelineResult
from app.pipeline.rr import RRSeries
from app.pipeline.template import PREMATURITY_THRESHOLD_PCT
from app.screening import thresholds as th
from app.screening.cosen import cosen
from app.screening.flags import ScreeningFlag
from app.screening.thresholds import HRLimits


def run_screening(
    result: PipelineResult,
    limits: HRLimits | None = None,
    athlete_baseline: bool = False,
) -> list[ScreeningFlag]:
    """Run all screening rules over a pipeline result."""
    limits = limits or th.default_limits(athlete_baseline)
    flags: list[ScreeningFlag] = []
    flags += flag_sustained_hr(result.rr, limits)
    flags += flag_ectopy(
        n_beats=len(result.peak_times_s),
        ectopic_beat_indices=result.correction.ectopic_beat_indices,
        prematurity_pct=result.morphology.prematurity_pct,
        motion_explained=result.morphology.motion_explained,
    )
    flags += flag_irregularity(result.rr)
    return flags


# ---------------------------------------------------------------------------
# Sustained tachycardia / bradycardia
# ---------------------------------------------------------------------------


def flag_sustained_hr(rr: RRSeries, limits: HRLimits) -> list[ScreeningFlag]:
    """Rolling ~45 s mean-HR windows against activity-aware limits.

    Single-beat excursions never flag; a window must *average* across the
    threshold. Both thresholds come from the activity profile (140 bpm during
    a run is expected, not a tachycardia observation).
    """
    if len(rr) < th.SUSTAINED_MIN_BEATS:
        return []

    windows = _hr_windows(rr)
    if not windows:
        return []

    flags: list[ScreeningFlag] = []
    high = [(s, e, hr) for s, e, hr in windows if hr > limits.tachy_bpm]
    low = [(s, e, hr) for s, e, hr in windows if hr < limits.brady_bpm]

    if high:
        worst = max(hr for _, _, hr in high)
        spans = _merge_spans([(s, e) for s, e, _ in high])
        flags.append(
            ScreeningFlag(
                kind="sustained_high_hr",
                severity="review",
                description=(
                    f"Sustained heart rate above the {limits.context} screening threshold "
                    f"of {limits.tachy_bpm:.0f} bpm: {len(high)} rolling "
                    f"{th.SUSTAINED_WINDOW_S:.0f}-second window(s) averaged above it, "
                    f"peaking at {worst:.0f} bpm. For the {limits.context} activity "
                    "context this is a pattern worth noting."
                ),
                evidence={
                    "worst_window_mean_bpm": round(worst, 1),
                    "n_windows_crossed": len(high),
                    "sustained_spans_s": [[round(s, 1), round(e, 1)] for s, e in spans],
                    "activity_context": limits.context,
                },
                threshold={
                    "rule": "rolling-window mean HR",
                    "window_s": th.SUSTAINED_WINDOW_S,
                    "tachy_bpm": limits.tachy_bpm,
                },
            )
        )

    if low:
        worst = min(hr for _, _, hr in low)
        spans = _merge_spans([(s, e) for s, e, _ in low])
        note = f" {limits.brady_note}" if limits.brady_note else ""
        flags.append(
            ScreeningFlag(
                kind="sustained_low_hr",
                severity="review",
                description=(
                    f"Sustained heart rate below the {limits.context} screening threshold "
                    f"of {limits.brady_bpm:.0f} bpm: {len(low)} rolling "
                    f"{th.SUSTAINED_WINDOW_S:.0f}-second window(s) averaged below it, "
                    f"reaching {worst:.0f} bpm. This is a low-specificity screening "
                    f"observation.{note}"
                ),
                evidence={
                    "lowest_window_mean_bpm": round(worst, 1),
                    "n_windows_crossed": len(low),
                    "sustained_spans_s": [[round(s, 1), round(e, 1)] for s, e in spans],
                    "activity_context": limits.context,
                    "athlete_note_applied": limits.brady_note is not None,
                },
                threshold={
                    "rule": "rolling-window mean HR",
                    "window_s": th.SUSTAINED_WINDOW_S,
                    "brady_bpm": limits.brady_bpm,
                },
            )
        )
    return flags


def _hr_windows(rr: RRSeries) -> list[tuple[float, float, float]]:
    """(start_s, end_s, mean HR bpm) for rolling sustained-HR windows."""
    out: list[tuple[float, float, float]] = []
    t0, t1 = float(rr.t_s[0]), float(rr.t_s[-1])
    w = t0
    while w + th.SUSTAINED_WINDOW_S <= t1 + th.SUSTAINED_STEP_S:
        mask = (rr.t_s >= w) & (rr.t_s < w + th.SUSTAINED_WINDOW_S)
        if int(np.sum(mask)) >= th.SUSTAINED_MIN_BEATS:
            mean_rr = float(np.mean(rr.rr_ms[mask]))
            out.append((w, w + th.SUSTAINED_WINDOW_S, 60000.0 / mean_rr))
        w += th.SUSTAINED_STEP_S
    return out


def _merge_spans(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(e, merged[-1][1]))
        else:
            merged.append((s, e))
    return merged


# ---------------------------------------------------------------------------
# Ectopy
# ---------------------------------------------------------------------------


def flag_ectopy(
    n_beats: int,
    ectopic_beat_indices: np.ndarray,
    prematurity_pct: np.ndarray,
    motion_explained: np.ndarray,
) -> list[ScreeningFlag]:
    """Ectopic beats: correction-classified AND premature AND not motion.

    Only the correction step's *ectopic* class enters — 'missed' and 'extra'
    are detector errors, never ectopy. Each candidate must then be confirmed
    premature against the local median RR and must not be explained by the
    motion proxy. Wording is "ectopic beats" throughout: a single lead cannot
    distinguish atrial from ventricular origin.
    """
    if n_beats == 0 or len(ectopic_beat_indices) == 0:
        return []

    confirmed: list[int] = []
    for i in np.asarray(ectopic_beat_indices, dtype=np.int64):
        if i < 0 or i >= len(prematurity_pct):
            continue
        premature = (
            not np.isnan(prematurity_pct[i])
            and prematurity_pct[i] <= PREMATURITY_THRESHOLD_PCT
        )
        motion = bool(motion_explained[i]) if i < len(motion_explained) else False
        if premature and not motion:
            confirmed.append(int(i))

    if not confirmed:
        return []

    pct = 100.0 * len(confirmed) / n_beats
    bigeminy = _bigeminy_run(confirmed) >= th.BIGEMINY_MIN_RUN
    if pct <= th.ECTOPY_PCT_THRESHOLD and not bigeminy:
        return []

    pattern_txt = (
        " An alternating (bigeminy-like) pattern was present, which is why this is "
        "flagged despite the burden being below the percentage threshold."
        if bigeminy and pct <= th.ECTOPY_PCT_THRESHOLD
        else ""
    )
    return [
        ScreeningFlag(
            kind="ectopy",
            severity="review",
            description=(
                f"{len(confirmed)} ectopic beat(s) ({pct:.2f}% of {n_beats} beats) were "
                "confirmed by three independent checks: classified ectopic by artifact "
                "correction, premature against the local rhythm, and not explained by "
                "motion. A single lead cannot distinguish atrial from ventricular "
                f"origin, so these are reported only as ectopic beats.{pattern_txt}"
            ),
            evidence={
                "n_confirmed_ectopic": len(confirmed),
                "pct_of_beats": round(pct, 2),
                "n_beats": n_beats,
                "beat_indices": confirmed[:100],
                "bigeminy_pattern": bigeminy,
                "n_correction_ectopic_class": int(len(ectopic_beat_indices)),
            },
            threshold={
                "rule": "confirmed ectopy percentage or repeating pattern",
                "pct_threshold": th.ECTOPY_PCT_THRESHOLD,
                "bigeminy_min_run": th.BIGEMINY_MIN_RUN,
            },
        )
    ]


def _bigeminy_run(confirmed_sorted: list[int]) -> int:
    """Longest run of confirmed ectopics spaced exactly two beats apart."""
    best, run = 1, 1
    for a, b in zip(confirmed_sorted, confirmed_sorted[1:], strict=False):
        run = run + 1 if b - a == 2 else 1
        best = max(best, run)
    return best if len(confirmed_sorted) else 0


# ---------------------------------------------------------------------------
# RR irregularity (possible AF)
# ---------------------------------------------------------------------------


def flag_irregularity(rr: RRSeries) -> list[ScreeningFlag]:
    """COSEn + RR-CV dual gate over rolling 30-beat windows.

    Both indices must cross in the same window, and at least
    IRREGULARITY_MIN_CONSECUTIVE consecutive windows must agree. Requiring
    agreement keeps healthy high-HRV sinus rhythm (high CV, low COSEn) and
    metronomic rhythm (low both) from flagging. Single-lead data cannot
    resolve P-waves, so this is an irregularity index only.
    """
    n = len(rr)
    if n < th.IRREGULARITY_WINDOW_BEATS + th.IRREGULARITY_STEP_BEATS:
        return []

    window_hits: list[tuple[int, float, float, float]] = []  # (start_beat, t, cosen, cv)
    dual_positive: list[bool] = []
    for start in range(0, n - th.IRREGULARITY_WINDOW_BEATS + 1, th.IRREGULARITY_STEP_BEATS):
        w = rr.rr_ms[start : start + th.IRREGULARITY_WINDOW_BEATS]
        # Windows that span a discontinuity mix non-adjacent beats; skip them.
        if np.any(rr.discontinuity[start + 1 : start + th.IRREGULARITY_WINDOW_BEATS]):
            dual_positive.append(False)
            continue
        c = cosen(w)
        cv = float(np.std(w, ddof=1) / np.mean(w))
        hit = c is not None and c > th.COSEN_THRESHOLD and cv > th.RR_CV_THRESHOLD
        dual_positive.append(hit)
        if hit:
            window_hits.append((start, float(rr.t_s[start]), float(c), cv))

    longest = _longest_true_run(dual_positive)
    if longest < th.IRREGULARITY_MIN_CONSECUTIVE:
        return []

    worst_cosen = max(c for _, _, c, _ in window_hits)
    worst_cv = max(cv for _, _, _, cv in window_hits)
    first_t = min(t for _, t, _, _ in window_hits)
    return [
        ScreeningFlag(
            kind="rr_irregularity",
            severity="review",
            description=(
                f"Sustained RR irregularity: {len(window_hits)} rolling "
                f"{th.IRREGULARITY_WINDOW_BEATS}-beat window(s) crossed both the COSEn "
                f"and the RR coefficient-of-variation screening thresholds, with "
                f"{longest} consecutive windows agreeing. This irregularity pattern is "
                "consistent with possible atrial fibrillation, but a single lead cannot "
                "resolve P-waves, so it is reported only as an irregularity index worth "
                "reviewing."
            ),
            evidence={
                "n_dual_positive_windows": len(window_hits),
                "longest_consecutive_run": longest,
                "max_cosen": round(worst_cosen, 3),
                "max_rr_cv": round(worst_cv, 3),
                "first_window_t_s": round(first_t, 1),
            },
            threshold={
                "rule": "COSEn AND RR-CV dual gate, consecutive windows",
                "cosen_threshold": th.COSEN_THRESHOLD,
                "rr_cv_threshold": th.RR_CV_THRESHOLD,
                "window_beats": th.IRREGULARITY_WINDOW_BEATS,
                "min_consecutive": th.IRREGULARITY_MIN_CONSECUTIVE,
            },
        )
    ]


def _longest_true_run(values: list[bool]) -> int:
    best = run = 0
    for v in values:
        run = run + 1 if v else 0
        best = max(best, run)
    return best
