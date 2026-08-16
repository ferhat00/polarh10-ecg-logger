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
