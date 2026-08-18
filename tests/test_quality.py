"""Quality-windowing tests: every exclusion reason, and the no-exclusion path."""

from __future__ import annotations

import numpy as np

from app.pipeline.quality import assess_quality
from tests.synth_util import FS_HZ, build_ecg_from_rr


def _steady_rr(duration_s: float = 60.0, rr_ms: float = 800.0) -> np.ndarray:
    return np.full(int(duration_s * 1000 / rr_ms), rr_ms)


def _run(t: np.ndarray, ecg: np.ndarray):
    return assess_quality(t, ecg, ecg, FS_HZ)


class TestCleanSignal:
    def test_no_exclusions_on_clean_ecg(self) -> None:
        t, ecg, _ = build_ecg_from_rr(_steady_rr())
        q = _run(t, ecg)
        assert q.excluded_segments == []
        assert q.excluded_total_s == 0.0
        assert q.analysed_total_s > 55.0
        assert all(not w.excluded for w in q.windows)

    def test_wander_tracks_motion(self) -> None:
        t, ecg, _ = build_ecg_from_rr(_steady_rr())
        # 0.4 Hz, 0.4 mV baseline sway from 20 s onward — motion, sub-cutoff.
        sway = 0.4 * np.sin(2 * np.pi * 0.4 * t) * (t >= 20.0)
        q = _run(t, ecg + sway)
        early = [w.wander_rms_mv for w in q.windows if w.end_s <= 15.0]
        late = [w.wander_rms_mv for w in q.windows if w.start_s >= 25.0]
        # The clean ECG itself has some sub-0.7 Hz content (T-wave energy), so
        # the ratio is bounded but must clearly separate motion from rest.
        assert np.mean(late) > 3 * np.mean(early)


class TestExclusionReasons:
    def test_flat_signal_excluded_with_bounds(self) -> None:
        t, ecg, _ = build_ecg_from_rr(_steady_rr())
        ecg = ecg.copy()
        flat = (t >= 20.0) & (t < 30.0)
        ecg[flat] = 0.0
        q = _run(t, ecg)
        assert any("flat/dead" in r for _, _, r in q.excluded_segments)
        (start, end, _reason) = next(
            (s, e, r) for s, e, r in q.excluded_segments if "flat/dead" in r
        )
        assert start <= 20.0 + 5.0 and end >= 30.0 - 5.0
        assert q.excluded_total_s >= 5.0

    def test_implausible_amplitude_excluded(self) -> None:
        t, ecg, _ = build_ecg_from_rr(_steady_rr())
        ecg = ecg.copy()
        burst = (t >= 25.0) & (t < 30.0)
        ecg[burst] += 6.0 * np.sin(2 * np.pi * 8.0 * t[burst])
        q = _run(t, ecg)
        assert any("amplitude" in r for _, _, r in q.excluded_segments)

    def test_clipping_excluded(self) -> None:
        t, ecg, _ = build_ecg_from_rr(_steady_rr())
        ecg = ecg.copy()
        pinned = (t >= 40.0) & (t < 44.0)
        ecg[pinned] = 3.2  # pinned at an apparent rail
        q = _run(t, ecg)
        assert any("clipping" in r for _, _, r in q.excluded_segments)

    def test_sample_gap_excluded(self) -> None:
        t, ecg, _ = build_ecg_from_rr(_steady_rr())
        keep = (t < 35.0) | (t >= 38.0)  # 3 s of samples missing entirely
        q = _run(t[keep], ecg[keep])
        assert any("gap" in r for _, _, r in q.excluded_segments)

    def test_excluded_time_is_reported_not_interpolated(self) -> None:
        """Excluded + analysed must account for the whole recording."""
        t, ecg, _ = build_ecg_from_rr(_steady_rr())
        ecg = ecg.copy()
        ecg[(t >= 20.0) & (t < 30.0)] = 0.0
        q = _run(t, ecg)
        duration = float(t[-1] - t[0])
        np.testing.assert_allclose(
            q.excluded_total_s + q.analysed_total_s, duration, atol=0.01
        )
        assert q.excluded_total_s >= 5.0


class TestWindowLookup:
    def test_window_at_and_is_excluded_at(self) -> None:
        t, ecg, _ = build_ecg_from_rr(_steady_rr())
        ecg = ecg.copy()
        ecg[(t >= 20.0) & (t < 30.0)] = 0.0
        q = _run(t, ecg)
        assert q.is_excluded_at(25.0)
        assert not q.is_excluded_at(5.0)
        w = q.window_at(25.0)
        assert w is not None and w.excluded


class TestRateDriftRobustness:
    """The SQI must not exclude clean signal whose rate drifts with sleep
    stage (the averageQRS global template penalises rate-atypical beats;
    the chunked local-template pass rescues them)."""

    def _two_rate_ecg(self, with_noise_burst: bool = False):
        # 12 min at 55 bpm then 4 min at 72 bpm — clean throughout.
        rr = np.concatenate(
            [_steady_rr(duration_s=720.0, rr_ms=1090.0),
             _steady_rr(duration_s=240.0, rr_ms=833.0)]
        )
        t, ecg, _ = build_ecg_from_rr(rr)
        ecg = ecg.copy()
        if with_noise_burst:
            rng = np.random.default_rng(9)
            burst = (t >= 800.0) & (t < 830.0)
            # Unambiguous electrode garbage: amplitude beyond the plausible
            # chest-strap range plus template-free shape. (Moderate noise
            # near the SQI threshold has always been a partial exclusion —
            # that behaviour predates the chunked-SQI change.)
            ecg[burst] = rng.normal(0.0, 2.0, int(np.sum(burst)))
        return t, ecg

    def test_clean_rate_change_not_excluded(self) -> None:
        t, ecg = self._two_rate_ecg()
        q = _run(t, ecg)
        # The faster block (720 s onward) must not be thrown away.
        late_excluded = sum(
            e - s for s, e, _ in q.excluded_segments if s >= 700.0
        )
        assert late_excluded <= 15.0, q.excluded_segments

    def test_noise_burst_still_excluded(self) -> None:
        t, ecg = self._two_rate_ecg(with_noise_burst=True)
        q = _run(t, ecg)
        assert q.is_excluded_at(815.0)
