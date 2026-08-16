"""Artifact-correction tests (Lipponen-Tarvainen / Kubios method)."""

from __future__ import annotations

import numpy as np

from app.pipeline.correction import REDUCED_CONFIDENCE_PCT, correct_peaks
from tests.synth_util import FS_HZ


def _regular_peaks(n: int = 300, rr_s: float = 0.8) -> np.ndarray:
    """A physiological clean series: LF-modulated RR plus small jitter.

    Pure white jitter would be unrealistic here: combined with the 7.7 ms
    sample quantisation it makes the dRR quartile deviation degenerate, which
    collapses the Kubios adaptive thresholds and flags a 'clean' series.
    Real RR always carries smooth autonomic modulation.
    """
    rng = np.random.default_rng(9)
    t = np.arange(n) * rr_s
    rr = rr_s + 0.03 * np.sin(2 * np.pi * 0.1 * t) + rng.normal(0, 0.008, n)
    times = np.cumsum(rr)
    return np.round(times * FS_HZ).astype(np.int64)


class TestCleanSeries:
    def test_low_correction_rate(self) -> None:
        result = correct_peaks(_regular_peaks(), FS_HZ)
        assert result.pct_corrected < 2.0
        assert result.reduced_confidence is False
        assert set(result.counts) == {"ectopic", "missed", "extra", "longshort"}


class TestCorruptedSeries:
    def test_heavy_artifact_marks_reduced_confidence(self) -> None:
        peaks = _regular_peaks().astype(np.float64)
        # Displace every 12th peak by 250 ms — ~8 % of beats badly mistimed.
        idx = np.arange(6, len(peaks), 12)
        peaks[idx] += 0.25 * FS_HZ
        result = correct_peaks(np.round(peaks).astype(np.int64), FS_HZ)
        assert result.pct_corrected > REDUCED_CONFIDENCE_PCT
        assert result.reduced_confidence is True

    def test_counts_are_per_class_and_pct_counts_distinct_beats(self) -> None:
        peaks = _regular_peaks().astype(np.float64)
        idx = np.arange(6, len(peaks), 12)
        peaks[idx] += 0.25 * FS_HZ
        result = correct_peaks(np.round(peaks).astype(np.int64), FS_HZ)
        # Screening must see the ectopic class separately from detector
        # errors (missed/extra) — the classes are never lumped.
        assert sum(result.counts.values()) >= len(idx)
        assert len(result.ectopic_beat_indices) > 0
        # pct counts distinct corrected beats: a displaced beat that lands in
        # two classes is one bad beat, not two.
        assert result.pct_corrected <= 100.0 * sum(result.counts.values()) / len(peaks)
        assert result.pct_corrected >= 100.0 * len(idx) / len(peaks)
