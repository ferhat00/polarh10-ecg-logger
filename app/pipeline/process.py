"""Pipeline orchestration: LoadedRecording → PipelineResult.

Order of operations, each stage visible in the result:

1. Clean + detect R-peaks (engine chain, provenance recorded).
2. Score 5 s quality windows; build the excluded-segment list.
3. Kubios artifact correction (classes and rate recorded).
4. Build the RR series — dropping intervals that straddle exclusions or
   breach physiological bounds.
5. Beat-template correlation with prematurity/motion cross-referencing.
6. HRV metrics.
7. Ectopy confirmation on the *raw detected* train and event grouping
   (singles/couplets/runs — see :mod:`app.pipeline.events`).
8. Cross-check our RR series against the device's own rr stream.

QRS width, QT, QTc, PR interval, and ECG axis are never computed: at ~130 Hz
one sample is 7.7 ms and delineation output is quantisation artifact, not
physiology. Interval measurement needs 500–1000 Hz and multiple leads.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.ingest.loader import LoadedRecording
from app.pipeline.correction import CorrectionResult, correct_peaks
from app.pipeline.engines import RPeakDetection, detect_rpeaks
from app.pipeline.events import EctopyEvents, confirmed_ectopic_indices, extract_events
from app.pipeline.hrv import HRVResult, compute_hrv
from app.pipeline.quality import QualityResult, assess_quality
from app.pipeline.respiration import RespirationResult, estimate_respiration
from app.pipeline.rr import RRSeries, build_rr
from app.pipeline.template import (
    BeatMorphology,
    beat_morphology,
    motion_degraded,
    prematurity_series,
)

#: Max time distance when matching our beats to device rr entries (s).
DEVICE_MATCH_TOLERANCE_S = 0.5


@dataclass
class PipelineResult:
    engine_used: str
    fallback_reason: str | None
    sampling_rate_hz: float

    detection: RPeakDetection
    quality: QualityResult
    correction: CorrectionResult
    rr: RRSeries
    morphology: BeatMorphology
    hrv: HRVResult

    #: Times (s) of the corrected R-peaks.
    peak_times_s: np.ndarray = field(default_factory=lambda: np.array([]))
    #: Median |our RR − device RR| in ms, when the device stream exists.
    device_rr_median_abs_diff_ms: float | None = None
    notes: list[str] = field(default_factory=list)
    #: Confirmed-ectopy events, grouped from the masks above (derived only).
    events: EctopyEvents | None = None
    #: ECG-derived respiratory-rate estimate (never measured airflow).
    respiration: RespirationResult | None = None
    #: Per *raw detected* beat: prematurity and motion state, the inputs to
    #: ectopy confirmation. The raw train is used deliberately — the Kubios
    #: iterative pass repositions the beats it classifies ectopic, so the
    #: corrected train no longer carries the prematurity being confirmed.
    ectopy_prematurity_pct: np.ndarray = field(default_factory=lambda: np.array([]))
    ectopy_motion_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))


def run_pipeline(rec: LoadedRecording) -> PipelineResult:
    """Run the full signal pipeline on a loaded recording."""
    notes: list[str] = list(rec.notes)

    detection = detect_rpeaks(rec.ecg_mv, rec.sampling_rate_hz)
    if detection.fallback_reason:
        notes.append(
            f"Engine fallback: {detection.fallback_reason} — biosppy results used."
        )

    quality = assess_quality(
        rec.time_s, rec.ecg_mv, detection.ecg_clean, rec.sampling_rate_hz
    )
    if quality.excluded_total_s > 0:
        notes.append(
            f"{quality.excluded_total_s:.0f} s excluded across "
            f"{len(quality.excluded_segments)} segment(s); "
            f"{quality.analysed_total_s:.0f} s analysed."
        )

    correction = correct_peaks(detection.rpeak_indices, rec.sampling_rate_hz)
    notes.append(
        f"Artifact correction (Lipponen-Tarvainen/Kubios): {correction.pct_corrected:.2f}% "
        f"of beats corrected ({correction.counts})."
        + (" Session marked reduced-confidence (>5%)." if correction.reduced_confidence else "")
    )

    peaks = np.clip(correction.peaks_corrected, 0, len(rec.time_s) - 1)
    peak_times = rec.time_s[peaks]
    rr = build_rr(peak_times, quality.excluded_segments)
    dropped = rr.n_dropped_excluded + rr.n_dropped_ceiling + rr.n_dropped_floor
    if dropped:
        notes.append(
            f"RR filtering dropped {rr.n_dropped_excluded} interval(s) overlapping "
            f"excluded segments, {rr.n_dropped_ceiling} above the physiological ceiling, "
            f"{rr.n_dropped_floor} below the floor."
        )

    morphology = beat_morphology(
        detection.ecg_clean, rec.sampling_rate_hz, peaks, rec.time_s, quality
    )
    if morphology.n_outliers:
        n_motion = int(np.sum(morphology.motion_explained))
        n_candidates = int(np.sum(morphology.ectopy_candidate))
        notes.append(
            f"{morphology.n_outliers} morphology outlier(s) (r < 0.90): "
            f"{n_motion} coincide with motion/degraded quality, "
            f"{n_candidates} are premature ectopy candidate(s)."
        )

    hrv = compute_hrv(rr)

    try:
        respiration = estimate_respiration(
            rr, detection.ecg_clean, peaks, peak_times, quality
        )
        if respiration.median_brpm is not None:
            notes.append(
                f"EDR respiratory-rate estimate: median {respiration.median_brpm:.1f} "
                f"brpm over {respiration.n_windows_used} quality-gated window(s)."
            )
    except Exception:  # noqa: BLE001 - an optional estimate must not kill the run
        respiration = None
        notes.append("Respiration estimation failed; no respiratory rate reported.")

    raw_peaks = np.clip(detection.rpeak_indices, 0, len(rec.time_s) - 1)
    raw_times = rec.time_s[raw_peaks]
    ectopy_prematurity = prematurity_series(raw_peaks, rec.sampling_rate_hz)
    ectopy_motion = motion_degraded(raw_peaks, rec.time_s, quality)
    confirmed = confirmed_ectopic_indices(
        len(raw_peaks),
        correction.ectopic_beat_indices,
        ectopy_prematurity,
        ectopy_motion,
    )
    events = extract_events(
        confirmed, raw_times, quality.analysed_total_s, quality.excluded_segments
    )
    if events.n_confirmed:
        notes.append(
            f"{events.n_confirmed} confirmed ectopic beat(s): "
            f"{events.n_singles} single(s), {events.n_couplets} couplet(s), "
            f"{events.n_runs} run(s) of 3+."
        )

    device_diff = _device_rr_agreement(rec, peak_times)
    if device_diff is not None:
        notes.append(
            f"Cross-check vs device rr stream: median |ΔRR| = {device_diff:.0f} ms "
            f"across {len(rec.device_rr_ms)} device intervals (two independent "
            "detectors at 7.7 ms sample quantisation)."
        )

    return PipelineResult(
        engine_used=detection.engine,
        fallback_reason=detection.fallback_reason,
        sampling_rate_hz=rec.sampling_rate_hz,
        detection=detection,
        quality=quality,
        correction=correction,
        rr=rr,
        morphology=morphology,
        hrv=hrv,
        peak_times_s=peak_times,
        device_rr_median_abs_diff_ms=device_diff,
        notes=notes,
        events=events,
        respiration=respiration,
        ectopy_prematurity_pct=ectopy_prematurity,
        ectopy_motion_mask=ectopy_motion,
    )


def _device_rr_agreement(
    rec: LoadedRecording, peak_times_s: np.ndarray
) -> float | None:
    """Median |our RR − device RR| by nearest-in-time matching.

    The device stream is a cross-check on our detection, never the primary
    source. Differences of a few tens of ms are expected: two different
    detectors, each quantised at ~7.7 ms.
    """
    if len(rec.device_rr_ms) < 5 or len(peak_times_s) < 3:
        return None
    our_rr_ms = np.diff(peak_times_s) * 1000.0
    our_t = peak_times_s[1:]

    diffs = []
    for t_dev, rr_dev in zip(rec.device_rr_time_s, rec.device_rr_ms, strict=True):
        i = int(np.argmin(np.abs(our_t - t_dev)))
        if abs(our_t[i] - t_dev) <= DEVICE_MATCH_TOLERANCE_S:
            diffs.append(abs(our_rr_ms[i] - rr_dev))
    if len(diffs) < 5:
        return None
    return float(np.median(diffs))
