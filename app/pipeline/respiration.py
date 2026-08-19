"""ECG-derived respiration (EDR): respiratory-rate estimation, never airflow.

Two independent modulation channels are extracted per recording —

* **rate channel**: instantaneous heart rate (respiratory sinus arrhythmia
  modulates RR intervals), and
* **amplitude channel**: R-wave amplitude (chest-geometry changes with
  breathing modulate the projected QRS amplitude),

each resampled to a uniform 4 Hz grid, band-limited with neurokit2's
``ecg_rsp`` **charlton2016** filter (0.066–1 Hz ⇒ 4–60 brpm; the default
vangent2019 band starts at 0.1 Hz and would be blind to slow paced breathing
at ~6 brpm), then reduced to a dominant frequency per 60 s window (Welch).
A window is kept only when the two channels agree — the fusion idea of
Charlton et al. 2016 (Physiol Meas 37:610) — and when quality gates pass.

Validated on this exact hardware: Schaffarczyk et al. 2022 (Sensors 22:7156)
report r = 0.85 vs gas-exchange respiratory frequency with the Polar H10,
degrading at high exercise intensity — hence the mean-HR gate below, and the
"estimate" labelling on every surface.

Sustained rates in the slow-breathing band flag ``paced_breathing``: slow
breathing (~6/min) mechanically inflates RMSSD and HF power (Laborde et al.
2022, doi:10.1016/j.neubiorev.2022.104711), so such sessions must not be
read as parasympathetic baselines. Processing turns the flag into a caution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sp_signal

from app.pipeline.quality import QualityResult
from app.pipeline.rr import RRSeries

#: Uniform resampling rate for both channels (Hz).
GRID_HZ = 4.0

#: Analysis window and hop (s). 60 s at 4 Hz gives ~1 brpm resolution.
WINDOW_S = 60.0
HOP_S = 30.0

#: Searched respiratory band (Hz): 3.6–54 brpm.
RESP_BAND_HZ = (0.06, 0.9)

#: The two channels must agree within this many brpm to keep a window.
AGREEMENT_BRPM = 4.0

#: Windows overlapping excluded segments by more than this are dropped.
MAX_EXCLUDED_FRACTION = 0.25

#: EDR degrades with intensity (Schaffarczyk 2022): gate on mean window HR.
MAX_WINDOW_HR_BPM = 110.0

#: Minimum kept windows before any summary number is reported.
MIN_WINDOWS = 5

#: Sustained rates in this band (≥ MIN_PACED_WINDOWS consecutive kept
#: windows) flag paced/slow breathing.
PACED_BAND_BRPM = (4.5, 7.5)
MIN_PACED_WINDOWS = 3

#: Minimum usable record length.
MIN_DURATION_S = 120.0

RESPIRATION_CAUTION = (
    "Sustained slow breathing (~{rate:.1f} brpm) detected across the "
    "recording — slow or paced breathing mechanically inflates RMSSD and HF "
    "power via respiratory sinus arrhythmia, so this session's "
    "parasympathetic indices reflect the breathing pattern, not a resting "
    "baseline. Compare it only against other paced-breathing sessions."
)


@dataclass
class RespirationResult:
    """Windowed respiratory-rate estimates for one recording."""

    median_brpm: float | None = None
    p5_brpm: float | None = None
    p95_brpm: float | None = None
    #: Kept windows: center time (s) and fused rate (brpm).
    window_t_s: list[float] = field(default_factory=list)
    window_brpm: list[float] = field(default_factory=list)
    n_windows_used: int = 0
    n_windows_total: int = 0
    paced_breathing: bool = False
    #: Median rate across the paced stretch, when flagged.
    paced_rate_brpm: float | None = None
    method: str = "EDR: HR + R-amplitude channels, charlton2016 band, Welch fusion"
    notes: list[str] = field(default_factory=list)

    def as_extras(self) -> dict:
        return {
            "median_brpm": self.median_brpm,
            "p5_brpm": self.p5_brpm,
            "p95_brpm": self.p95_brpm,
            "window_t_s": [round(t, 1) for t in self.window_t_s],
            "window_brpm": [round(b, 2) for b in self.window_brpm],
            "n_windows_used": self.n_windows_used,
            "n_windows_total": self.n_windows_total,
            "paced_breathing": self.paced_breathing,
            "paced_rate_brpm": self.paced_rate_brpm,
            "method": self.method,
            "notes": list(self.notes),
        }


def estimate_respiration(
    rr: RRSeries,
    ecg_clean: np.ndarray,
    peaks: np.ndarray,
    peak_times_s: np.ndarray,
    quality: QualityResult,
) -> RespirationResult:
    """Estimate the respiratory rate; empty result (with notes) when unusable."""
    out = RespirationResult()
    if len(rr) < 30 or len(peak_times_s) < 30:
        out.notes.append("Too few beats for respiration estimation.")
        return out
    t0, t1 = float(rr.t_s[0]), float(rr.t_s[-1])
    if t1 - t0 < MIN_DURATION_S:
        out.notes.append(
            f"Record shorter than {MIN_DURATION_S:.0f} s of usable RR — "
            "respiration not estimated."
        )
        return out

    grid = np.arange(t0, t1, 1.0 / GRID_HZ)

    # Rate channel: instantaneous HR interpolated to the grid.
    hr_bpm = 60000.0 / rr.rr_ms
    hr_grid = np.interp(grid, rr.t_s, hr_bpm)

    # Amplitude channel: R-peak amplitude interpolated to the grid.
    amp = ecg_clean[np.clip(peaks, 0, len(ecg_clean) - 1)]
    n = min(len(amp), len(peak_times_s))
    amp_grid = np.interp(grid, peak_times_s[:n], amp[:n])

    filt_rate = _edr_filter(hr_grid)
    filt_amp = _edr_filter(amp_grid)

    win_n = int(WINDOW_S * GRID_HZ)
    hop_n = int(HOP_S * GRID_HZ)
    starts = range(0, max(len(grid) - win_n + 1, 0), hop_n)

    kept_t: list[float] = []
    kept_rate: list[float] = []
    n_total = 0
    for s in starts:
        n_total += 1
        seg = slice(s, s + win_n)
        w_start, w_end = float(grid[s]), float(grid[s + win_n - 1])
        if _excluded_fraction(quality, w_start, w_end) > MAX_EXCLUDED_FRACTION:
            continue
        if float(np.mean(hr_grid[seg])) > MAX_WINDOW_HR_BPM:
            continue
        f_rate = _dominant_freq(filt_rate[seg])
        f_amp = _dominant_freq(filt_amp[seg])
        if f_rate is None or f_amp is None:
            continue
        brpm_rate, brpm_amp = 60.0 * f_rate, 60.0 * f_amp
        if abs(brpm_rate - brpm_amp) > AGREEMENT_BRPM:
            continue
        kept_t.append((w_start + w_end) / 2.0)
        kept_rate.append((brpm_rate + brpm_amp) / 2.0)

    out.n_windows_total = n_total
    out.n_windows_used = len(kept_rate)
    out.window_t_s = kept_t
    out.window_brpm = kept_rate

    if len(kept_rate) < MIN_WINDOWS:
        out.notes.append(
            f"Only {len(kept_rate)} of {n_total} window(s) passed the quality "
            "and channel-agreement gates — no respiratory rate reported."
        )
        return out

    rates = np.array(kept_rate)
    out.median_brpm = float(np.median(rates))
    out.p5_brpm = float(np.percentile(rates, 5))
    out.p95_brpm = float(np.percentile(rates, 95))

    # Paced/slow-breathing detection over consecutive kept windows.
    in_band = (rates >= PACED_BAND_BRPM[0]) & (rates <= PACED_BAND_BRPM[1])
    run = best_start = 0
    best_run = 0
    start = 0
    for i, flag in enumerate(in_band):
        if flag:
            if run == 0:
                start = i
            run += 1
            if run > best_run:
                best_run, best_start = run, start
        else:
            run = 0
    if best_run >= MIN_PACED_WINDOWS:
        out.paced_breathing = True
        stretch = rates[best_start : best_start + best_run]
        out.paced_rate_brpm = float(np.median(stretch))

    return out


def _edr_filter(channel: np.ndarray) -> np.ndarray:
    """Band-limit one channel to the respiratory band.

    neurokit2's ``ecg_rsp`` with ``method="charlton2016"`` is a 0.066–1 Hz
    band-pass — wide enough to include slow paced breathing, which the
    default band would erase (verified against the vendored 0.2.13 source).
    """
    import neurokit2 as nk

    return np.asarray(
        nk.ecg_rsp(channel, sampling_rate=int(GRID_HZ), method="charlton2016")
    )


def _dominant_freq(segment: np.ndarray) -> float | None:
    """Dominant frequency (Hz) of one window within the respiratory band."""
    if len(segment) < 16 or not np.all(np.isfinite(segment)):
        return None
    freqs, psd = sp_signal.welch(
        segment - np.mean(segment), fs=GRID_HZ, nperseg=len(segment)
    )
    band = (freqs >= RESP_BAND_HZ[0]) & (freqs <= RESP_BAND_HZ[1])
    if not np.any(band) or not np.any(psd[band] > 0):
        return None
    return float(freqs[band][int(np.argmax(psd[band]))])


def _excluded_fraction(quality: QualityResult, w_start: float, w_end: float) -> float:
    """Fraction of [w_start, w_end] overlapped by excluded segments."""
    length = max(w_end - w_start, 1e-9)
    overlap = 0.0
    for seg_start, seg_end, _reason in quality.excluded_segments:
        overlap += max(0.0, min(seg_end, w_end) - max(seg_start, w_start))
    return overlap / length
