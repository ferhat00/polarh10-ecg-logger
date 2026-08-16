"""Beat-template tests: morphology alone is never ectopy.

The classification cross-references three signals — template correlation,
prematurity vs the local median RR, and the motion proxy — because in real
chest-strap data most morphology outliers are motion artifact, not ectopy.
"""

from __future__ import annotations

import numpy as np

from app.pipeline.quality import assess_quality
from app.pipeline.template import beat_morphology
from tests.synth_util import FS_HZ, build_ecg_from_rr, peak_indices_for


def _steady_rr(n: int = 120, rr_ms: float = 800.0) -> np.ndarray:
    return np.full(n, rr_ms)


def _analyse(rr_ms: np.ndarray, beat_amps: dict[int, float] | None = None,
             raw_extra=None):
    t, ecg, beat_times = build_ecg_from_rr(rr_ms, beat_amps=beat_amps)
    raw = ecg if raw_extra is None else ecg + raw_extra(t)
    quality = assess_quality(t, raw, ecg, FS_HZ)
    peaks = peak_indices_for(beat_times)
    return beat_morphology(ecg, FS_HZ, peaks, t, quality), quality


class TestNormalBeats:
    def test_uniform_train_has_no_outliers(self) -> None:
        morph, _ = _analyse(_steady_rr())
        assert morph.n_outliers == 0
        assert not np.any(morph.ectopy_candidate)
        # Interior beats correlate tightly with the template (the σ=10 ms
        # noise floor costs a little correlation on individual beats).
        interior = morph.correlations[2:-2]
        assert np.nanmin(interior) > 0.95


class TestEctopyCandidate:
    def test_premature_distorted_beat_is_candidate(self) -> None:
        rr = _steady_rr()
        # Beat 61 arrives 300 ms early (500 ms coupling interval, -37.5 %)
        # with a compensatory pause, and its morphology is inverted-ish.
        rr[60] = 500.0
        rr[61] = 1100.0
        morph, _ = _analyse(rr, beat_amps={61: -0.45})
        assert morph.outlier_mask[61]
        assert morph.prematurity_pct[61] < -20.0
        assert not morph.motion_explained[61]
        assert morph.ectopy_candidate[61]
        # And nothing else got swept up.
        assert int(np.sum(morph.ectopy_candidate)) == 1


class TestMotionNotEctopy:
    def test_ontime_distorted_beat_is_not_candidate(self) -> None:
        """A morphology outlier that is not premature is not ectopy."""
        morph, _ = _analyse(_steady_rr(), beat_amps={60: -0.45})
        assert morph.outlier_mask[60]
        assert abs(morph.prematurity_pct[60]) < 10.0
        assert not morph.ectopy_candidate[60]

    def test_motion_explained_outlier_is_not_candidate(self) -> None:
        """Even a premature-looking outlier in a motion-degraded window
        must not be counted as ectopy."""
        rr = _steady_rr()
        rr[60] = 500.0
        rr[61] = 1100.0
        beat_time_61 = 0.5 + float(np.cumsum(rr)[60]) / 1000.0

        def sway(t: np.ndarray) -> np.ndarray:
            # Strong 0.4 Hz baseline sway around beat 61 only (motion burst).
            burst = np.abs(t - beat_time_61) < 6.0
            return 0.5 * np.sin(2 * np.pi * 0.4 * t) * burst

        morph, quality = _analyse(rr, beat_amps={61: -0.45}, raw_extra=sway)
        assert morph.outlier_mask[61]
        assert morph.motion_explained[61]
        assert not morph.ectopy_candidate[61]


class TestTemplateShape:
    def test_template_spans_pqrst(self) -> None:
        morph, _ = _analyse(_steady_rr())
        assert len(morph.template) > 0
        # R peak at t≈0 in template coordinates.
        r_idx = int(np.argmax(morph.template))
        assert abs(morph.template_t_ms[r_idx]) < 20.0
        # T wave present after R.
        after = morph.template[morph.template_t_ms > 100.0]
        assert np.max(after) > 0.1
