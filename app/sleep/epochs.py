"""Per-epoch cardiac features for the built-in heuristic staging engine.

Each 30 s epoch gets: mean HR, a windowed RMSSD, a windowed LF/HF power
ratio, HR relative to the night baseline, a movement value, and a coverage
fraction. Feature windows are 5 min centred on the epoch (±5 epochs) —
30 s of RR data is too little for a stable RMSSD or any LF estimate (the
LF band's longest period is 25 s), and 5 min is the Task Force short-term
standard the rest of the app already uses.

Band powers are computed by band-pass filtering the 4 Hz-resampled
tachogram per contiguous run (the same cubic-spline resampling as
:mod:`app.pipeline.hrv`, and the same rule: nothing is ever estimated
across a dropped-interval gap) and averaging the squared filtered signal —
by Parseval this converges to the band's integrated PSD, at O(n) cost over
an 8-hour night.

Everything is vectorised with ``bincount``/convolution: an overnight RR
series is ~35 k intervals × ~960 epochs, where per-epoch boolean scans are
the known performance trap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import interpolate as sp_interpolate
from scipy import signal as sp_signal

from app.pipeline.hrv import HF_BAND, LF_BAND, RESAMPLE_HZ, _contiguous_runs
from app.pipeline.quality import QualityResult
from app.pipeline.rr import RRSeries
from app.sleep.actigraphy import THRESHOLD_MADS, AccEpochs
from app.sleep.stages import EPOCH_LEN_S, epoch_indices, make_epoch_grid

#: Fewer usable intervals than this in an epoch → the epoch has no HR value.
MIN_BEATS_PER_EPOCH = 10

#: Epochs with less RR coverage than this are unstageable (UNSCORED).
MIN_EPOCH_COVERAGE = 0.5

#: Half-width of the centred feature window, in epochs (±5 → 5 minutes).
FEATURE_HALF_WIDTH_EPOCHS = 5

#: Shortest contiguous run worth band-filtering (two LF periods).
MIN_BAND_RUN_S = 50.0

#: The night's HR baseline is this percentile of per-epoch mean HR — the
#: sustained nocturnal minimum, robust to brief arousals.
BASELINE_HR_PERCENTILE = 10.0


@dataclass
class EpochFeatures:
    """Per-epoch features on the canonical grid."""

    epoch_start_s: np.ndarray
    mean_hr_bpm: np.ndarray
    #: RMSSD over the centred 5-min window (ms).
    rmssd_ms: np.ndarray
    #: LF/HF band-power ratio over the centred 5-min window.
    lf_hf: np.ndarray
    #: Mean HR minus the night baseline (bpm).
    hr_vs_baseline: np.ndarray
    #: Movement per epoch: ACC activity counts when available, else the mean
    #: baseline-wander RMS of the ECG quality windows (a coarse proxy).
    movement: np.ndarray
    #: Robust per-night movement threshold matching ``movement``'s source.
    movement_threshold: float
    movement_source: str  # "acc" | "wander_proxy"
    #: Fraction of the epoch covered by usable RR intervals.
    coverage: np.ndarray
    notes: list[str] = field(default_factory=list)

    @property
    def n_epochs(self) -> int:
        return len(self.epoch_start_s)

    def stageable(self) -> np.ndarray:
        return (self.coverage >= MIN_EPOCH_COVERAGE) & ~np.isnan(self.mean_hr_bpm)


def _centered_rolling_sum(values: np.ndarray, half_width: int) -> np.ndarray:
    """Sum over a centred window, truncated at the edges (any input length)."""
    n = len(values)
    if n == 0:
        return values.copy()
    c = np.concatenate(([0.0], np.cumsum(values)))
    lo = np.clip(np.arange(n) - half_width, 0, n)
    hi = np.clip(np.arange(n) + half_width + 1, 0, n)
    return c[hi] - c[lo]


def _rolling_nansum(values: np.ndarray, half_width: int) -> tuple[np.ndarray, np.ndarray]:
    """(windowed nan-sum, windowed count of non-nan values)."""
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    sums = _centered_rolling_sum(filled, half_width)
    counts = _centered_rolling_sum(finite.astype(float), half_width)
    return sums, counts


def _band_power_per_epoch(
    rr: RRSeries, n_epochs: int, band: tuple[float, float], notes: list[str]
) -> np.ndarray:
    """Mean squared band-passed tachogram per epoch (ms²), NaN where absent."""
    sums = np.zeros(n_epochs)
    counts = np.zeros(n_epochs)
    nyq = RESAMPLE_HZ / 2.0
    sos = sp_signal.butter(
        4, [band[0] / nyq, band[1] / nyq], btype="bandpass", output="sos"
    )
    n_used = 0
    for t, x in _contiguous_runs(rr):
        if t[-1] - t[0] < MIN_BAND_RUN_S or np.any(np.diff(t) <= 0):
            continue
        grid = np.arange(t[0], t[-1], 1.0 / RESAMPLE_HZ)
        if len(grid) < 40:
            continue
        resampled = sp_interpolate.CubicSpline(t, x)(grid)
        filtered = sp_signal.sosfiltfilt(sos, resampled - np.mean(resampled))
        power = filtered**2
        idx = np.floor(grid / EPOCH_LEN_S).astype(np.int64)
        keep = (idx >= 0) & (idx < n_epochs)
        sums += np.bincount(idx[keep], weights=power[keep], minlength=n_epochs)
        counts += np.bincount(idx[keep], minlength=n_epochs)
        n_used += 1
    if n_used == 0:
        notes.append(
            f"No contiguous run of ≥{MIN_BAND_RUN_S:.0f} s — spectral features "
            "unavailable."
        )
    out = np.full(n_epochs, np.nan)
    has = counts > 0
    out[has] = sums[has] / counts[has]
    return out


def compute_epoch_features(
    rr: RRSeries,
    quality: QualityResult,
    duration_s: float,
    acc_epochs: AccEpochs | None,
) -> EpochFeatures:
    """Build the per-epoch feature set for one night."""
    grid = make_epoch_grid(duration_s)
    n_epochs = len(grid)
    notes: list[str] = []
    nan = np.full(n_epochs, np.nan)
    if n_epochs == 0 or len(rr) == 0:
        return EpochFeatures(
            epoch_start_s=grid,
            mean_hr_bpm=nan.copy(),
            rmssd_ms=nan.copy(),
            lf_hf=nan.copy(),
            hr_vs_baseline=nan.copy(),
            movement=nan.copy(),
            movement_threshold=np.nan,
            movement_source="wander_proxy",
            coverage=np.zeros(n_epochs),
            notes=["No usable RR intervals — nothing to stage."],
        )

    epoch_of = epoch_indices(rr.t_s, n_epochs)
    valid = epoch_of >= 0
    idx = epoch_of[valid]

    n_beats = np.bincount(idx, minlength=n_epochs).astype(float)
    sum_rr = np.bincount(idx, weights=rr.rr_ms[valid], minlength=n_epochs)
    coverage = np.clip(sum_rr / 1000.0 / EPOCH_LEN_S, 0.0, 1.0)

    mean_hr = np.full(n_epochs, np.nan)
    enough = n_beats >= MIN_BEATS_PER_EPOCH
    mean_hr[enough] = 60000.0 / (sum_rr[enough] / n_beats[enough])

    # Windowed RMSSD: per-epoch sums of squared successive differences
    # (contiguous pairs only), then a centred rolling sum.
    diffs_sq = np.diff(rr.rr_ms) ** 2
    pair_ok = (~rr.discontinuity[1:]) & valid[1:] & valid[:-1] & (
        epoch_of[1:] == epoch_of[:-1]
    )
    d_idx = epoch_of[1:][pair_ok]
    d_sums = np.bincount(d_idx, weights=diffs_sq[pair_ok], minlength=n_epochs)
    d_counts = np.bincount(d_idx, minlength=n_epochs).astype(float)
    win_sums = _centered_rolling_sum(d_sums, FEATURE_HALF_WIDTH_EPOCHS)
    win_counts = _centered_rolling_sum(d_counts, FEATURE_HALF_WIDTH_EPOCHS)
    rmssd = np.full(n_epochs, np.nan)
    has_diffs = win_counts >= MIN_BEATS_PER_EPOCH
    rmssd[has_diffs] = np.sqrt(win_sums[has_diffs] / win_counts[has_diffs])

    # Windowed LF/HF from band-passed tachogram power.
    lf_pow = _band_power_per_epoch(rr, n_epochs, LF_BAND, notes)
    hf_pow = _band_power_per_epoch(rr, n_epochs, HF_BAND, notes[:0])
    lf_sums, lf_counts = _rolling_nansum(lf_pow, FEATURE_HALF_WIDTH_EPOCHS)
    hf_sums, hf_counts = _rolling_nansum(hf_pow, FEATURE_HALF_WIDTH_EPOCHS)
    lf_hf = np.full(n_epochs, np.nan)
    ok = (lf_counts > 0) & (hf_counts > 0) & (hf_sums > 0)
    lf_hf[ok] = (lf_sums[ok] / lf_counts[ok]) / (hf_sums[ok] / hf_counts[ok])

    # HR relative to the night's sustained minimum.
    finite_hr = mean_hr[np.isfinite(mean_hr)]
    if len(finite_hr):
        baseline = float(np.percentile(finite_hr, BASELINE_HR_PERCENTILE))
        hr_delta = mean_hr - baseline
    else:
        hr_delta = np.full(n_epochs, np.nan)

    # Movement: ACC counts, else the quality-window wander proxy.
    if acc_epochs is not None and len(acc_epochs.counts) == n_epochs:
        movement = acc_epochs.counts
        movement_threshold = acc_epochs.threshold
        movement_source = "acc"
    else:
        movement = np.full(n_epochs, np.nan)
        w_idx = np.array(
            [int(w.start_s // EPOCH_LEN_S) for w in quality.windows], dtype=np.int64
        )
        w_val = np.array([w.wander_rms_mv for w in quality.windows])
        keep = (w_idx >= 0) & (w_idx < n_epochs) & np.isfinite(w_val)
        if np.any(keep):
            sums = np.bincount(w_idx[keep], weights=w_val[keep], minlength=n_epochs)
            counts = np.bincount(w_idx[keep], minlength=n_epochs)
            has = counts > 0
            movement[has] = sums[has] / counts[has]
        covered = movement[np.isfinite(movement)]
        if len(covered):
            med = float(np.median(covered))
            mad = float(np.median(np.abs(covered - med)))
            movement_threshold = (
                med + THRESHOLD_MADS * mad if mad > 0 else 2.0 * med + 1e-9
            )
        else:
            movement_threshold = np.nan
        movement_source = "wander_proxy"
        notes.append(
            "No accelerometer file — movement is approximated by ECG baseline "
            "wander, a coarse proxy that misses still-body wakefulness."
        )

    return EpochFeatures(
        epoch_start_s=grid,
        mean_hr_bpm=mean_hr,
        rmssd_ms=rmssd,
        lf_hf=lf_hf,
        hr_vs_baseline=hr_delta,
        movement=movement,
        movement_threshold=movement_threshold,
        movement_source=movement_source,
        coverage=coverage,
        notes=notes,
    )
