"""R-peak detection engines: neurokit2 primary, biosppy fallback.

The fallback chain is honest: the result records which engine actually ran and
why the primary failed, and that provenance is stored on the session and shown
in the report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class PipelineError(Exception):
    """The signal could not be processed by any engine."""


@dataclass
class RPeakDetection:
    """Cleaned signal and R-peak sample indices, with engine provenance."""

    engine: str  # "neurokit2" | "biosppy"
    ecg_clean: np.ndarray
    rpeak_indices: np.ndarray
    fallback_reason: str | None = None


def detect_rpeaks(ecg_mv: np.ndarray, fs_hz: float) -> RPeakDetection:
    """Clean the ECG and detect R-peaks, falling back to biosppy if needed.

    Artifact correction is deliberately *not* applied here: it runs as a
    separate, visible step (:mod:`app.pipeline.correction`) so the correction
    rate and artifact classes can be recorded rather than hidden inside the
    detector.
    """
    try:
        return _neurokit_detect(ecg_mv, fs_hz)
    except Exception as primary_exc:  # noqa: BLE001 - any engine failure falls through
        try:
            result = _biosppy_detect(ecg_mv, fs_hz)
        except Exception as fallback_exc:
            raise PipelineError(
                "Both engines failed to detect R-peaks — "
                f"neurokit2: {primary_exc}; biosppy: {fallback_exc}"
            ) from fallback_exc
        result.fallback_reason = f"neurokit2 failed: {primary_exc}"
        return result


def _neurokit_detect(ecg_mv: np.ndarray, fs_hz: float) -> RPeakDetection:
    import neurokit2 as nk

    clean = nk.ecg_clean(ecg_mv, sampling_rate=fs_hz)
    _, info = nk.ecg_peaks(clean, sampling_rate=fs_hz, correct_artifacts=False)
    peaks = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
    if len(peaks) < 10:
        raise PipelineError(f"neurokit2 found only {len(peaks)} R-peaks.")
    return RPeakDetection(engine="neurokit2", ecg_clean=clean, rpeak_indices=peaks)


def _biosppy_detect(ecg_mv: np.ndarray, fs_hz: float) -> RPeakDetection:
    from biosppy.signals import ecg as bs_ecg

    out = bs_ecg.ecg(signal=ecg_mv, sampling_rate=fs_hz, show=False)
    peaks = np.asarray(out["rpeaks"], dtype=np.int64)
    if len(peaks) < 10:
        raise PipelineError(f"biosppy found only {len(peaks)} R-peaks.")
    return RPeakDetection(
        engine="biosppy",
        ecg_clean=np.asarray(out["filtered"], dtype=np.float64),
        rpeak_indices=peaks,
    )
