"""HRV metrics per the 1996 Task Force standard.

Task Force of the European Society of Cardiology and NASPE, "Heart rate
variability: standards of measurement, physiological interpretation and
clinical use," Circulation 1996;93:1043-1065.

Deliberate design points (each one a known naive-implementation bug):

* Whole-record SDNN on long recordings is trend-inclusive — a slow HR drift
  inflates it far beyond true beat-to-beat variability. Per-minute SDNN is
  always computed alongside and the whole-record figure is labelled.
* Spectral estimation never bridges dropped intervals: the RR series is split
  at discontinuities and Welch runs per contiguous run, averaged by duration.
* LF/HF is engine-dependent (a 44 % spread was measured between two engines on
  identical input); it is reported as descriptive only, tagged with the PSD
  method, and nothing is scored or flagged on it.
* No respiration rate is derived: near-0.1 Hz spectral power is the baroreflex
  Mayer wave, and an "ECG-derived respiration" figure off this data is that
  wave in disguise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import interpolate as sp_interpolate
from scipy import signal as sp_signal

from app.pipeline.rr import RRSeries

# Task Force frequency bands (Hz).
VLF_BAND = (0.0033, 0.04)
LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.40)

#: Resampling rate for spectral estimation of the RR tachogram.
RESAMPLE_HZ = 4.0
#: Minimum contiguous run length worth including in the spectrum. The Task
#: Force short-term standard is 5 min; 120 s is the floor below which LF
#: estimates lose meaning entirely.
MIN_SPECTRAL_RUN_S = 120.0
#: Minimum beats for sample entropy / DFA to be meaningful.
MIN_NONLINEAR_BEATS = 100
#: Window length for per-window SDNN.
SDNN_WINDOW_S = 60.0


@dataclass
class HRVResult:
    n_beats: int = 0

    # Time domain
    mean_rr_ms: float | None = None
    mean_hr_bpm: float | None = None
    #: Whole-record SDNN — trend-inclusive and length-dependent; always read
    #: alongside sdnn_per_window_ms.
    sdnn_ms: float | None = None
    rmssd_ms: float | None = None
    pnn50_pct: float | None = None
    #: Per-window SDNN (window starts in sdnn_window_t_s).
    sdnn_per_window_ms: list[float] = field(default_factory=list)
    sdnn_window_t_s: list[float] = field(default_factory=list)

    # Frequency domain (descriptive; see notes)
    vlf_power_ms2: float | None = None
    lf_power_ms2: float | None = None
    hf_power_ms2: float | None = None
    lf_hf_ratio: float | None = None
    #: Frequency of the dominant spectral peak in the LF band, if any.
    lf_peak_hz: float | None = None
    psd_method: str | None = None
    #: The averaged Welch PSD itself (for the report's spectrum figure —
    #: not persisted to the metrics table).
    psd_freq_hz: list[float] = field(default_factory=list)
    psd_ms2_per_hz: list[float] = field(default_factory=list)

    # Nonlinear
    sd1_ms: float | None = None
    sd2_ms: float | None = None
    sd1_sd2_ratio: float | None = None
    sample_entropy: float | None = None
    dfa_alpha1: float | None = None

    notes: list[str] = field(default_factory=list)


def compute_hrv(rr: RRSeries) -> HRVResult:
    """Compute time-, frequency-, and nonlinear-domain HRV from an RR series."""
    result = HRVResult(n_beats=len(rr))
    x = rr.rr_ms
    if len(x) < 10:
        result.notes.append("Fewer than 10 usable intervals — HRV not computed.")
        return result

    _time_domain(result, rr)
    _frequency_domain(result, rr)
    _nonlinear(result, rr)
    return result


def _time_domain(result: HRVResult, rr: RRSeries) -> None:
    x = rr.rr_ms
    result.mean_rr_ms = float(np.mean(x))
    result.mean_hr_bpm = 60000.0 / result.mean_rr_ms
    result.sdnn_ms = float(np.std(x, ddof=1))

    # Successive differences only within contiguous runs — a diff across a
    # dropped interval is not a beat-to-beat difference.
    diffs = np.diff(x)
    contiguous = ~rr.discontinuity[1:]
    diffs = diffs[contiguous]
    if len(diffs):
        result.rmssd_ms = float(np.sqrt(np.mean(diffs**2)))
        result.pnn50_pct = float(100.0 * np.mean(np.abs(diffs) > 50.0))

    # Per-window SDNN: the trend-free counterpart to the whole-record figure.
    t = rr.t_s
    start, end = float(t[0]), float(t[-1])
    w = start
    while w < end:
        mask = (t >= w) & (t < w + SDNN_WINDOW_S)
        if np.sum(mask) >= 10:
            result.sdnn_per_window_ms.append(float(np.std(x[mask], ddof=1)))
            result.sdnn_window_t_s.append(w)
        w += SDNN_WINDOW_S

    result.notes.append(
        "Whole-record SDNN is trend-inclusive and length-dependent; compare only "
        "against recordings of similar duration and read it alongside per-window SDNN."
    )


def _contiguous_runs(rr: RRSeries) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split the RR series at discontinuities into (t_s, rr_ms) runs."""
    if len(rr) == 0:
        return []
    boundaries = np.flatnonzero(rr.discontinuity)
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(rr.rr_ms)]))
    return [
        (rr.t_s[s:e], rr.rr_ms[s:e]) for s, e in zip(starts, ends, strict=True) if e - s >= 4
    ]


def _frequency_domain(result: HRVResult, rr: RRSeries) -> None:
    runs = [
        (t, x) for t, x in _contiguous_runs(rr) if t[-1] - t[0] >= MIN_SPECTRAL_RUN_S
    ]
    if not runs:
        result.notes.append(
            f"No contiguous run of at least {MIN_SPECTRAL_RUN_S:.0f} s — frequency-domain "
            "metrics not computed (spectra are never estimated across excluded gaps)."
        )
        return

    # One nperseg for every run so all Welch estimates share a frequency grid.
    shortest = min(int((t[-1] - t[0]) * RESAMPLE_HZ) for t, _ in runs)
    nperseg = min(shortest, 1024)

    psds = []
    weights = []
    freqs = np.array([])
    for t, x in runs:
        # Cubic-spline resample of the tachogram at 4 Hz *within* the run.
        grid = np.arange(t[0], t[-1], 1.0 / RESAMPLE_HZ)
        spline = sp_interpolate.CubicSpline(t, x)
        resampled = spline(grid)
        resampled = resampled - np.mean(resampled)
        freqs, pxx = sp_signal.welch(resampled, fs=RESAMPLE_HZ, nperseg=nperseg)
        psds.append(pxx)
        weights.append(t[-1] - t[0])

    psd = np.average(psds, axis=0, weights=weights)
    result.psd_freq_hz = [float(f) for f in freqs]
    result.psd_ms2_per_hz = [float(p) for p in psd]

    def band_power(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.trapezoid(psd[mask], freqs[mask])) if np.any(mask) else 0.0

    result.vlf_power_ms2 = band_power(*VLF_BAND)
    result.lf_power_ms2 = band_power(*LF_BAND)
    result.hf_power_ms2 = band_power(*HF_BAND)
    if result.hf_power_ms2 and result.hf_power_ms2 > 0:
        result.lf_hf_ratio = result.lf_power_ms2 / result.hf_power_ms2

    lf_mask = (freqs >= LF_BAND[0]) & (freqs < LF_BAND[1])
    if np.any(lf_mask) and np.max(psd[lf_mask]) > 0:
        result.lf_peak_hz = float(freqs[lf_mask][np.argmax(psd[lf_mask])])

    result.psd_method = f"welch@{RESAMPLE_HZ:.0f}Hz(cubic-spline tachogram)"
    result.notes.append(
        f"Frequency-domain metrics are descriptive only ({result.psd_method}); LF/HF is "
        "engine-dependent and nothing is scored or flagged on it. A peak near 0.1 Hz is "
        "the baroreflex Mayer wave, not respiration."
    )
    total_run_s = sum(weights)
    if total_run_s < 300.0:
        result.notes.append(
            f"Spectral estimate uses {total_run_s:.0f} s of contiguous data — below the "
            "Task Force 5-minute short-term standard; interpret with caution."
        )


def _nonlinear(result: HRVResult, rr: RRSeries) -> None:
    x = rr.rr_ms
    diffs = np.diff(x)
    contiguous = ~rr.discontinuity[1:]
    diffs = diffs[contiguous]
    if len(diffs) < 10 or result.sdnn_ms is None:
        return

    # Poincaré (Brennan et al. 2001, IEEE Trans Biomed Eng 48:1342-7):
    # SD1² = ½·var(ΔRR); SD2² = 2·SDNN² − ½·var(ΔRR).
    var_diff = float(np.var(diffs, ddof=1))
    sd1_sq = 0.5 * var_diff
    sd2_sq = max(0.0, 2.0 * result.sdnn_ms**2 - 0.5 * var_diff)
    result.sd1_ms = float(np.sqrt(sd1_sq))
    result.sd2_ms = float(np.sqrt(sd2_sq))
    if result.sd2_ms > 0:
        result.sd1_sd2_ratio = result.sd1_ms / result.sd2_ms

    if len(x) < MIN_NONLINEAR_BEATS:
        result.notes.append(
            f"Fewer than {MIN_NONLINEAR_BEATS} beats — sample entropy and DFA α1 "
            "not computed."
        )
        return

    import neurokit2 as nk

    try:
        sampen, _ = nk.entropy_sample(x, delay=1, dimension=2, tolerance=0.2 * np.std(x, ddof=1))
        if np.isfinite(sampen):
            result.sample_entropy = float(sampen)
    except Exception:  # noqa: BLE001 - optional metric, never fatal
        result.notes.append("Sample entropy could not be computed.")

    try:
        alpha1 = nk.fractal_dfa(x, scale=np.arange(4, 17), multifractal=False)
        if isinstance(alpha1, tuple):
            alpha1 = alpha1[0]
        if np.isfinite(float(alpha1)):
            result.dfa_alpha1 = float(alpha1)
    except Exception:  # noqa: BLE001
        result.notes.append("DFA α1 could not be computed.")
