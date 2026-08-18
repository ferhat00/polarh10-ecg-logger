"""Signal-quality scoring over 5-second windows.

Noisy or lead-off stretches are **excluded and reported** — never silently
interpolated. Each window is checked for clipping, implausible amplitude,
flat/dead signal, sample gaps, and low signal-quality index; a 0.7 Hz low-pass
RMS of the *raw* signal serves as the baseline-wander / motion proxy (the
cleaned signal has its baseline removed, so wander must be measured pre-clean).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sp_signal

#: Window length for quality scoring.
WINDOW_S = 5.0
#: Chunk length for the local SQI pass (see _signal_quality_index): long
#: enough for a stable average-QRS template, short enough that heart rate is
#: near-stationary within it.
SQI_CHUNK_S = 300.0
#: Plausible chest-strap ECG range. R-waves run roughly 0.5–3 mV; sustained
#: excursions beyond ±4 mV indicate electrode artifact, not physiology.
PLAUSIBLE_ABS_MV = 4.0
#: A window whose standard deviation falls below this is flat / lead-off.
FLAT_STD_MV = 0.01
#: Windows with mean averageQRS quality index below this are excluded.
SQI_THRESHOLD = 0.5
#: A sample interval more than this multiple of the median is a gap.
GAP_FACTOR = 2.0
#: Cutoff for the baseline-wander (motion proxy) low-pass filter.
WANDER_CUTOFF_HZ = 0.7
#: Minimum consecutive identical extreme samples that count as clipping.
CLIP_RUN = 3
#: Rails below this are treated as genuine signal, not converter limits.
CLIP_MIN_RAIL_MV = 2.0


@dataclass
class QualityWindow:
    start_s: float
    end_s: float
    sqi_mean: float
    wander_rms_mv: float
    excluded: bool
    reasons: tuple[str, ...] = field(default=())


@dataclass
class QualityResult:
    windows: list[QualityWindow]
    #: Merged consecutive excluded windows: (start_s, end_s, reason).
    excluded_segments: list[tuple[float, float, str]]
    #: Per-sample signal quality index (averageQRS method).
    sqi: np.ndarray
    #: Per-sample baseline wander (0.7 Hz low-pass of the raw signal), mV.
    wander_mv: np.ndarray
    excluded_total_s: float
    analysed_total_s: float

    def window_at(self, t_s: float) -> QualityWindow | None:
        for w in self.windows:
            if w.start_s <= t_s < w.end_s:
                return w
        return None

    def is_excluded_at(self, t_s: float) -> bool:
        return any(s <= t_s < e for s, e, _ in self.excluded_segments)


def assess_quality(
    time_s: np.ndarray,
    ecg_raw_mv: np.ndarray,
    ecg_clean_mv: np.ndarray,
    fs_hz: float,
) -> QualityResult:
    """Score 5 s windows and return exclusions with start, end, and reason."""
    sqi = _signal_quality_index(ecg_clean_mv, fs_hz)
    wander = _baseline_wander(ecg_raw_mv, fs_hz)
    dt_all = np.diff(time_s)
    median_dt = float(np.median(dt_all))
    # Gaps found globally so one spanning a window boundary is not missed.
    gap_idx = np.flatnonzero(dt_all > GAP_FACTOR * median_dt)
    gaps = [(float(time_s[i]), float(time_s[i + 1])) for i in gap_idx]
    rail = _clipping_rail(ecg_raw_mv)

    duration = float(time_s[-1] - time_s[0])
    n_windows = max(1, int(np.ceil(duration / WINDOW_S)))
    windows: list[QualityWindow] = []

    for k in range(n_windows):
        w_start = float(time_s[0]) + k * WINDOW_S
        w_end = min(w_start + WINDOW_S, float(time_s[-1]))
        mask = (time_s >= w_start) & (time_s < w_end) if k < n_windows - 1 else (
            (time_s >= w_start) & (time_s <= w_end)
        )
        x_raw = ecg_raw_mv[mask]
        reasons: list[str] = []

        if len(x_raw) < 2:
            reasons.append("sample gap (no data in window)")
            windows.append(
                QualityWindow(w_start, w_end, 0.0, 0.0, True, tuple(reasons))
            )
            continue

        if any(g_start < w_end and g_end > w_start for g_start, g_end in gaps):
            reasons.append("sample gap")
        if rail is not None and _has_clipping_run(x_raw, rail):
            reasons.append("clipping")
        p99_abs = float(np.percentile(np.abs(x_raw), 99))
        if p99_abs > PLAUSIBLE_ABS_MV:
            reasons.append("amplitude outside plausible chest-strap range")
        if float(np.std(x_raw)) < FLAT_STD_MV:
            reasons.append("flat/dead signal")
        sqi_mean = float(np.mean(sqi[mask])) if np.any(mask) else 0.0
        if sqi_mean < SQI_THRESHOLD and "flat/dead signal" not in reasons:
            reasons.append(f"quality index below threshold ({sqi_mean:.2f})")

        wander_rms = float(np.sqrt(np.mean(wander[mask] ** 2)))
        windows.append(
            QualityWindow(
                start_s=w_start,
                end_s=w_end,
                sqi_mean=sqi_mean,
                wander_rms_mv=wander_rms,
                excluded=bool(reasons),
                reasons=tuple(reasons),
            )
        )

    segments = _merge_excluded(windows)
    excluded_total = sum(e - s for s, e, _ in segments)
    return QualityResult(
        windows=windows,
        excluded_segments=segments,
        sqi=sqi,
        wander_mv=wander,
        excluded_total_s=float(excluded_total),
        analysed_total_s=float(max(0.0, duration - excluded_total)),
    )


def _signal_quality_index(ecg_clean_mv: np.ndarray, fs_hz: float) -> np.ndarray:
    """Per-sample averageQRS quality index, robust to slow heart-rate drift.

    The averageQRS method correlates each beat's neighbourhood against the
    recording-wide mean template, so its window content shifts with heart
    rate: on a multi-hour recording whose rate legitimately drifts (an
    overnight session moves between sleep stages 15+ bpm apart), clean
    stretches at a rate atypical for the recording score as "bad signal".

    Fix: also compute the index per ~5-minute chunk (rate is near-stationary
    within a chunk, so the local template fits the local rate) and take the
    per-sample maximum. Genuine noise correlates with *neither* the global
    nor the local template and stays excluded; clean rate-deviant signal is
    rescued by its local template. Recordings shorter than one chunk are
    unchanged (local == global).
    """
    global_sqi = _average_qrs(ecg_clean_mv, fs_hz)
    chunk = int(SQI_CHUNK_S * fs_hz)
    n = len(ecg_clean_mv)
    if n <= chunk + chunk // 2:
        return global_sqi

    local = np.full(n, np.nan)
    start = 0
    while start < n:
        end = start + chunk
        if n - end < chunk // 2:  # fold a short tail into the last chunk
            end = n
        local[start:end] = _average_qrs(ecg_clean_mv[start:end], fs_hz)
        start = end
    return np.fmax(global_sqi, local)  # fmax: NaN loses to a real value


def _average_qrs(ecg_clean_mv: np.ndarray, fs_hz: float) -> np.ndarray:
    import neurokit2 as nk

    try:
        q = nk.ecg_quality(ecg_clean_mv, sampling_rate=int(round(fs_hz)), method="averageQRS")
        return np.asarray(q, dtype=np.float64)
    except Exception:  # noqa: BLE001 - quality index failing must not kill the run
        # Without an SQI, windows rely on the other checks; be explicit.
        return np.full(len(ecg_clean_mv), np.nan)


def _baseline_wander(ecg_raw_mv: np.ndarray, fs_hz: float) -> np.ndarray:
    """0.7 Hz low-pass of the raw signal — baseline wander as a motion proxy."""
    sos = sp_signal.butter(2, WANDER_CUTOFF_HZ, btype="low", fs=fs_hz, output="sos")
    return sp_signal.sosfiltfilt(sos, ecg_raw_mv)


def _clipping_rail(ecg_raw_mv: np.ndarray) -> float | None:
    """The apparent converter rail, if the signal ever gets near one."""
    rail = float(np.max(np.abs(ecg_raw_mv)))
    return rail if rail >= CLIP_MIN_RAIL_MV else None


def _has_clipping_run(x: np.ndarray, rail: float) -> bool:
    """True if ≥CLIP_RUN consecutive samples sit pinned at the rail."""
    pinned = np.abs(x) >= 0.98 * rail
    if not np.any(pinned):
        return False
    run = 0
    for p in pinned:
        run = run + 1 if p else 0
        if run >= CLIP_RUN:
            return True
    return False


def _merge_excluded(windows: list[QualityWindow]) -> list[tuple[float, float, str]]:
    segments: list[tuple[float, float, str]] = []
    for w in windows:
        if not w.excluded:
            continue
        reason = "; ".join(dict.fromkeys(w.reasons))
        if segments and abs(segments[-1][1] - w.start_s) < 1e-9:
            prev_s, _, prev_reason = segments[-1]
            merged = "; ".join(dict.fromkeys(prev_reason.split("; ") + list(w.reasons)))
            segments[-1] = (prev_s, w.end_s, merged)
        else:
            segments.append((w.start_s, w.end_s, reason))
    return segments
