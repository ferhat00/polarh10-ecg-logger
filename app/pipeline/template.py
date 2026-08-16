"""Beat-template correlation: separating motion artifact from genuine ectopy.

A median beat template is built and every beat correlated against it.
Morphology alone is *not* ectopy: in real chest-strap data, low-correlation
beats routinely coincide with motion (degraded quality, elevated baseline
wander) while showing no prematurity. A beat is only a candidate ectopic if it
is a morphology outlier AND premature against the local median RR AND not
explained by the motion proxy — that cross-referencing happens here; the
screening rules consume the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.pipeline.quality import QualityResult

#: Beat window around the R-peak. At 130 Hz this is ~33 samples before and
#: ~52 after — enough to span P through T without touching neighbours at
#: normal rates.
PRE_MS = 250.0
POST_MS = 400.0
#: Beats correlating below this against the median template are morphology
#: outliers (r = 0.90, the conventional template-matching cut).
MORPHOLOGY_R_THRESHOLD = 0.90
#: A beat is premature when its preceding RR is ≥20 % shorter than the local
#: median — the usual premature-beat criterion for RR-based screening.
PREMATURITY_THRESHOLD_PCT = -20.0
#: Number of surrounding intervals used for the local median RR.
LOCAL_WINDOW_BEATS = 10
#: A beat's window counts as motion-degraded only when SQI is *severely*
#: depressed. The bar is deliberately low: averageQRS quality drops on any
#: odd-shaped beat, so a stricter SQI bar would let a genuine ectopic veto
#: itself as "motion". Wander is the primary motion evidence.
MOTION_SQI_THRESHOLD = 0.5
#: A window's baseline-wander RMS above this multiple of the session median
#: is elevated — the primary motion-proxy signal.
MOTION_WANDER_FACTOR = 3.0


@dataclass
class BeatMorphology:
    #: Median beat template (mV) and its time axis in ms relative to R.
    template: np.ndarray
    template_t_ms: np.ndarray
    #: Per-beat Pearson correlation against the template (NaN at edges).
    correlations: np.ndarray
    #: Per-beat prematurity of the preceding RR vs the local median, in %
    #: (negative = early). NaN where undefined.
    prematurity_pct: np.ndarray
    #: correlations < MORPHOLOGY_R_THRESHOLD.
    outlier_mask: np.ndarray
    #: Outlier beats whose window shows degraded quality / elevated wander.
    motion_explained: np.ndarray
    #: Outlier beats that are also premature and NOT motion-explained —
    #: the only beats screening may consider ectopy candidates.
    ectopy_candidate: np.ndarray

    @property
    def n_outliers(self) -> int:
        return int(np.sum(self.outlier_mask))


def beat_morphology(
    ecg_clean: np.ndarray,
    fs_hz: float,
    peak_indices: np.ndarray,
    time_s: np.ndarray,
    quality: QualityResult,
) -> BeatMorphology:
    """Correlate every beat against the median template and cross-reference."""
    pre = int(round(PRE_MS / 1000.0 * fs_hz))
    post = int(round(POST_MS / 1000.0 * fs_hz))
    n_beats = len(peak_indices)

    beats = np.full((n_beats, pre + post), np.nan)
    for i, p in enumerate(peak_indices):
        lo, hi = p - pre, p + post
        if lo >= 0 and hi <= len(ecg_clean):
            beats[i] = ecg_clean[lo:hi]

    valid = ~np.isnan(beats).any(axis=1)
    if valid.sum() < 3:
        nan = np.full(n_beats, np.nan)
        false = np.zeros(n_beats, dtype=bool)
        return BeatMorphology(
            template=np.array([]),
            template_t_ms=np.array([]),
            correlations=nan,
            prematurity_pct=nan,
            outlier_mask=false,
            motion_explained=false.copy(),
            ectopy_candidate=false.copy(),
        )

    template = np.median(beats[valid], axis=0)
    template_t_ms = (np.arange(pre + post) - pre) / fs_hz * 1000.0

    correlations = np.full(n_beats, np.nan)
    t_centered = template - template.mean()
    t_norm = np.sqrt(np.sum(t_centered**2))
    for i in np.flatnonzero(valid):
        b = beats[i] - beats[i].mean()
        denom = np.sqrt(np.sum(b**2)) * t_norm
        if denom > 0:
            correlations[i] = float(np.dot(b, t_centered) / denom)

    prematurity = _prematurity_pct(peak_indices, fs_hz)
    outliers = np.nan_to_num(correlations, nan=1.0) < MORPHOLOGY_R_THRESHOLD
    motion = _motion_explained(peak_indices, time_s, quality) & outliers
    premature = np.nan_to_num(prematurity, nan=0.0) <= PREMATURITY_THRESHOLD_PCT
    candidates = outliers & premature & ~motion

    return BeatMorphology(
        template=template,
        template_t_ms=template_t_ms,
        correlations=correlations,
        prematurity_pct=prematurity,
        outlier_mask=outliers,
        motion_explained=motion,
        ectopy_candidate=candidates,
    )


def _prematurity_pct(peak_indices: np.ndarray, fs_hz: float) -> np.ndarray:
    """Preceding-RR deviation from the local median RR, as a percentage."""
    n = len(peak_indices)
    prematurity = np.full(n, np.nan)
    if n < 3:
        return prematurity
    rr = np.diff(peak_indices) / fs_hz * 1000.0  # rr[i] precedes beat i+1
    half = LOCAL_WINDOW_BEATS // 2
    for beat in range(1, n):
        i = beat - 1  # index of the preceding interval
        lo, hi = max(0, i - half), min(len(rr), i + half + 1)
        neighbours = np.delete(rr[lo:hi], i - lo)
        if len(neighbours) >= 3:
            local_med = float(np.median(neighbours))
            if local_med > 0:
                prematurity[beat] = 100.0 * (rr[i] - local_med) / local_med
    return prematurity


def _motion_explained(
    peak_indices: np.ndarray, time_s: np.ndarray, quality: QualityResult
) -> np.ndarray:
    """True per beat when its quality window shows motion degradation."""
    wander_values = [w.wander_rms_mv for w in quality.windows]
    wander_median = float(np.median(wander_values)) if wander_values else 0.0
    wander_limit = MOTION_WANDER_FACTOR * max(wander_median, 1e-6)

    out = np.zeros(len(peak_indices), dtype=bool)
    for i, p in enumerate(peak_indices):
        t = float(time_s[p])
        w = quality.window_at(t)
        if w is None:
            continue
        degraded = (
            w.excluded
            or (not np.isnan(w.sqi_mean) and w.sqi_mean < MOTION_SQI_THRESHOLD)
            or w.wander_rms_mv > wander_limit
        )
        out[i] = degraded
    return out
