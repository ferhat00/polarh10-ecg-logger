"""Derived-metric functions shared by activity profiles.

All operate on the RR series (and quality windows), respect discontinuities,
and return None rather than a fabricated number when the data can't support
the metric.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize as sp_optimize

from app.pipeline.quality import QualityResult
from app.pipeline.rr import RRSeries

#: Standard %HRmax training zones (ACSM's Guidelines for Exercise Testing and
#: Prescription, 11th ed.): Z1 50-60, Z2 60-70, Z3 70-80, Z4 80-90, Z5 90+.
ZONE_BOUNDS_PCT_MAX = (50.0, 60.0, 70.0, 80.0, 90.0)


def instantaneous_hr(rr: RRSeries) -> tuple[np.ndarray, np.ndarray]:
    """(t_s, bpm) per interval."""
    return rr.t_s, 60000.0 / rr.rr_ms


def per_minute_hr(rr: RRSeries) -> tuple[list[float], list[float]]:
    """(minute_start_s, mean bpm) for whole minutes with enough beats."""
    starts: list[float] = []
    means: list[float] = []
    if len(rr) == 0:
        return starts, means
    w = float(rr.t_s[0])
    end = float(rr.t_s[-1])
    while w < end:
        mask = (rr.t_s >= w) & (rr.t_s < w + 60.0)
        if int(np.sum(mask)) >= 10:
            starts.append(w)
            means.append(float(60000.0 / np.mean(rr.rr_ms[mask])))
        w += 60.0
    return starts, means


def hr_trend_bpm_per_min(rr: RRSeries) -> float | None:
    """Least-squares slope of instantaneous HR over the session, bpm/min."""
    if len(rr) < 30:
        return None
    t, hr = instantaneous_hr(rr)
    slope_per_s = float(np.polyfit(t, hr, 1)[0])
    return slope_per_s * 60.0


def hr_settling(
    rr: RRSeries, tol_bpm_per_min: float = 0.5
) -> tuple[float, float] | None:
    """(settled_at_min, settled_hr_bpm): first minute after which the
    remaining per-minute trend stays within ±tol. None if it never settles."""
    starts, means = per_minute_hr(rr)
    if len(means) < 4:
        return None
    t0 = float(rr.t_s[0])
    for k in range(len(means) - 2):
        tail_t = np.array(starts[k:]) / 60.0
        tail_hr = np.array(means[k:])
        slope = float(np.polyfit(tail_t, tail_hr, 1)[0])
        if abs(slope) <= tol_bpm_per_min:
            return (starts[k] - t0) / 60.0, float(np.mean(tail_hr))
    return None


def hr_zones(rr: RRSeries, max_hr_bpm: float) -> dict[str, float] | None:
    """Fraction of *analysed beat time* in each %HRmax zone."""
    if len(rr) == 0 or max_hr_bpm <= 0:
        return None
    hr = 60000.0 / rr.rr_ms
    pct = 100.0 * hr / max_hr_bpm
    weights = rr.rr_ms / float(np.sum(rr.rr_ms))
    zones: dict[str, float] = {}
    edges = (0.0, *ZONE_BOUNDS_PCT_MAX, np.inf)
    labels = ("below_z1", "z1", "z2", "z3", "z4", "z5")
    for label, lo, hi in zip(labels, edges[:-1], edges[1:], strict=True):
        zones[label] = float(np.sum(weights[(pct >= lo) & (pct < hi)]))
    return zones


def rolling_peak_hr(rr: RRSeries, window_s: float = 30.0) -> tuple[float, float] | None:
    """(end_of_peak_t_s, peak bpm) from `window_s` rolling mean HR.

    The time returned is the END of the last window whose mean sits within
    1 bpm of the maximum — i.e. the end of the peak plateau. That is the
    right anchor for recovery metrics: HRR is defined from the cessation of
    peak effort, not from the first moment the peak was reached.
    """
    if len(rr) < 10:
        return None
    windows: list[tuple[float, float]] = []  # (window_end_s, mean bpm)
    for i in range(len(rr)):
        w_start = float(rr.t_s[i])
        mask = (rr.t_s >= w_start) & (rr.t_s < w_start + window_s)
        if int(np.sum(mask)) < 5:
            continue
        w_end = float(rr.t_s[mask][-1])
        windows.append((w_end, float(60000.0 / np.mean(rr.rr_ms[mask]))))
    if not windows:
        return None
    peak_hr = max(hr for _, hr in windows)
    anchor = max(end for end, hr in windows if hr >= peak_hr - 1.0)
    return anchor, peak_hr


def hr_recovery(rr: RRSeries, after_s: float) -> float | None:
    """HRR: drop (bpm) from peak rolling HR to the HR `after_s` later.

    HRR60 (one-minute heart-rate recovery) per Cole et al., NEJM
    1999;341:1351-1357. Returns None when the recording doesn't extend far
    enough past the peak.
    """
    peak = rolling_peak_hr(rr)
    if peak is None:
        return None
    t_peak, hr_peak = peak
    target = t_peak + after_s
    mask = (rr.t_s >= target - 10.0) & (rr.t_s < target + 10.0)
    if int(np.sum(mask)) < 3:
        return None
    hr_after = float(60000.0 / np.mean(rr.rr_ms[mask]))
    return hr_peak - hr_after


def recovery_time_constant(rr: RRSeries) -> float | None:
    """τ (s) of an exponential fit HR(t) = HR∞ + A·e^(−t/τ) after the peak.

    Mono-exponential post-exercise recovery model (Imai et al., JACC
    1994;24:1529-1535). None when the fit fails or the tail is too short.
    """
    peak = rolling_peak_hr(rr)
    if peak is None:
        return None
    t_peak, _ = peak
    mask = rr.t_s >= t_peak
    if int(np.sum(mask)) < 30:
        return None
    t = rr.t_s[mask] - t_peak
    hr = 60000.0 / rr.rr_ms[mask]
    if float(t[-1]) < 90.0:
        return None
    try:
        popt, _ = sp_optimize.curve_fit(
            lambda x, hr_inf, amp, tau: hr_inf + amp * np.exp(-x / tau),
            t,
            hr,
            p0=(float(np.min(hr)), float(np.max(hr) - np.min(hr)), 60.0),
            bounds=([20.0, 0.0, 5.0], [220.0, 150.0, 600.0]),
            maxfev=2000,
        )
    except (RuntimeError, ValueError):
        return None
    return float(popt[2])


def per_minute_rmssd(rr: RRSeries) -> tuple[list[float], list[float]]:
    """(minute_start_s, RMSSD ms) per whole minute, discontinuity-aware."""
    starts: list[float] = []
    values: list[float] = []
    if len(rr) < 2:
        return starts, values
    diffs = np.diff(rr.rr_ms)
    contiguous = ~rr.discontinuity[1:]
    t_diff = rr.t_s[1:]
    w = float(rr.t_s[0])
    end = float(rr.t_s[-1])
    while w < end:
        mask = (t_diff >= w) & (t_diff < w + 60.0) & contiguous
        if int(np.sum(mask)) >= 10:
            starts.append(w)
            values.append(float(np.sqrt(np.mean(diffs[mask] ** 2))))
        w += 60.0
    return starts, values


def motion_hr_coupling(rr: RRSeries, quality: QualityResult) -> float | None:
    """Pearson r between per-window baseline wander and per-window mean HR.

    A high positive coupling says HR elevation tracks motion — useful for
    walking sessions where effort and artifact rise together.
    """
    wander: list[float] = []
    hr: list[float] = []
    for w in quality.windows:
        mask = (rr.t_s >= w.start_s) & (rr.t_s < w.end_s)
        if int(np.sum(mask)) < 3:
            continue
        wander.append(w.wander_rms_mv)
        hr.append(float(60000.0 / np.mean(rr.rr_ms[mask])))
    if len(hr) < 6 or np.std(wander) == 0 or np.std(hr) == 0:
        return None
    return float(np.corrcoef(wander, hr)[0, 1])


def cardiac_drift_bpm_per_min(rr: RRSeries, skip_first_s: float = 300.0) -> float | None:
    """HR slope after the first `skip_first_s` — a cardiac-drift proxy.

    True cardiovascular drift is HR rise at *constant effort*; effort isn't
    observable here, so this is labelled a proxy wherever it is shown.
    """
    mask = rr.t_s >= rr.t_s[0] + skip_first_s
    if int(np.sum(mask)) < 60:
        return None
    sub = RRSeries(
        rr_ms=rr.rr_ms[mask],
        t_s=rr.t_s[mask],
        discontinuity=rr.discontinuity[mask],
        n_dropped_excluded=0,
        n_dropped_ceiling=0,
        n_dropped_floor=0,
    )
    return hr_trend_bpm_per_min(sub)


def orthostatic_response(
    rr: RRSeries, markers: list[tuple[float, str]]
) -> dict[str, float | str] | None:
    """ΔHR across a supine→standing transition, if one is captured.

    A marker containing "stand" pins the transition; otherwise the largest
    sustained upward HR step is used and labelled as inferred. Normal
    orthostatic response is a 10-20 bpm transient rise (Wieling et al.,
    Clin Sci 2007;112:157-165).
    """
    if len(rr) < 60:
        return None

    marker_t = next(
        (t for t, text in markers if "stand" in text.lower()), None
    )
    if marker_t is not None:
        before = _mean_hr_between(rr, marker_t - 60.0, marker_t)
        after = _mean_hr_between(rr, marker_t + 5.0, marker_t + 65.0)
        if before is None or after is None:
            return None
        return {
            "delta_hr_bpm": after - before,
            "transition_t_s": marker_t,
            "source": "marker",
        }

    # Inferred: biggest rise between consecutive minutes, if it persists.
    starts, means = per_minute_hr(rr)
    if len(means) < 3:
        return None
    rises = np.diff(means)
    k = int(np.argmax(rises))
    if rises[k] < 8.0:  # no convincing step
        return None
    persists = k + 2 < len(means) and means[k + 2] > means[k] + 0.5 * rises[k]
    if not persists:
        return None
    return {
        "delta_hr_bpm": float(rises[k]),
        "transition_t_s": float(starts[k + 1]),
        "source": "inferred from HR step (no marker present)",
    }


def _mean_hr_between(rr: RRSeries, t0: float, t1: float) -> float | None:
    mask = (rr.t_s >= t0) & (rr.t_s < t1)
    if int(np.sum(mask)) < 5:
        return None
    return float(60000.0 / np.mean(rr.rr_ms[mask]))
