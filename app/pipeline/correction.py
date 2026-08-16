"""Artifact correction via the Lipponen & Tarvainen (2019) method.

Reference: Lipponen JA, Tarvainen MP. "A robust algorithm for heart rate
variability time series artefact correction using novel beat classification."
J Med Eng Technol. 2019;43(3):173-181 (neurokit2 ``signal_fixpeaks``
``method="Kubios"``). Never a fixed percentage threshold.

The artifact classes are kept separate: 'missed' and 'extra' are detector
errors, not physiology, and downstream ectopy screening uses the 'ectopic'
class only — never the summed correction count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Above this percentage of corrected beats the whole session is marked
#: reduced-confidence, per Kubios HRV guidance (Tarvainen et al., 2014,
#: Comput Methods Programs Biomed 113:210-220).
REDUCED_CONFIDENCE_PCT = 5.0


@dataclass
class CorrectionResult:
    #: Corrected R-peak sample indices.
    peaks_corrected: np.ndarray
    #: Per-class corrected-beat counts: ectopic / missed / extra / longshort.
    counts: dict[str, int]
    #: Beat indices (into the original peak array) classified as ectopic.
    ectopic_beat_indices: np.ndarray
    #: Beat indices classified long/short (rhythm-timing outliers).
    longshort_beat_indices: np.ndarray
    pct_corrected: float
    reduced_confidence: bool


def correct_peaks(rpeak_indices: np.ndarray, fs_hz: float) -> CorrectionResult:
    """Run Kubios-method artifact correction and record what it changed.

    Two passes are deliberate: with ``iterative=True`` neurokit2 returns the
    artifact dict of the *last* iteration — i.e. after correction, when counts
    are near zero — which would silently hide the correction rate. The first,
    non-iterative pass classifies the raw series honestly; the second,
    iterative pass produces the corrected peaks.
    """
    import neurokit2 as nk

    artifacts, _ = nk.signal_fixpeaks(
        rpeak_indices, sampling_rate=fs_hz, iterative=False, method="Kubios"
    )
    _, peaks_clean = nk.signal_fixpeaks(
        rpeak_indices, sampling_rate=fs_hz, iterative=True, method="Kubios"
    )

    class_indices = {
        key: np.atleast_1d(np.asarray(artifacts.get(key, []), dtype=np.int64))
        for key in ("ectopic", "missed", "extra", "longshort")
    }
    counts = {key: len(v) for key, v in class_indices.items()}
    n_beats = max(1, len(rpeak_indices))
    # A single bad beat can land in more than one class (a displaced beat is
    # both ectopic-patterned and long/short); count distinct beats.
    distinct = np.unique(np.concatenate(list(class_indices.values())))
    pct = 100.0 * len(distinct) / n_beats

    return CorrectionResult(
        peaks_corrected=np.asarray(peaks_clean, dtype=np.int64),
        counts=counts,
        ectopic_beat_indices=class_indices["ectopic"],
        longshort_beat_indices=class_indices["longshort"],
        pct_corrected=pct,
        reduced_confidence=pct > REDUCED_CONFIDENCE_PCT,
    )
