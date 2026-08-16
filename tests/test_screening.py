"""Screening-rule tests over constructed RR series and beat classifications."""

from __future__ import annotations

import numpy as np

from app.pipeline.rr import RRSeries
from app.screening import thresholds as th
from app.screening.rules import (
    flag_ectopy,
    flag_irregularity,
    flag_sustained_hr,
)
from app.screening.thresholds import HRLimits, default_limits


def _series(rr_ms: np.ndarray) -> RRSeries:
    t = np.cumsum(rr_ms) / 1000.0
    return RRSeries(
        rr_ms=rr_ms,
        t_s=t,
        discontinuity=np.zeros(len(rr_ms), dtype=bool),
        n_dropped_excluded=0,
        n_dropped_ceiling=0,
        n_dropped_floor=0,
    )


def _steady(bpm: float, duration_s: float = 300.0) -> RRSeries:
    rr = 60000.0 / bpm
    n = int(duration_s * 1000 / rr)
    rng = np.random.default_rng(5)
    t = np.arange(n) * rr / 1000.0
    jitter = 0.02 * rr * np.sin(2 * np.pi * 0.1 * t) + rng.normal(0, 5.0, n)
    return _series(np.full(n, rr) + jitter)


class TestSustainedHR:
    def test_sustained_high_hr_flags(self) -> None:
        flags = flag_sustained_hr(_steady(110.0), default_limits(False))
        assert [f.kind for f in flags] == ["sustained_high_hr"]
        f = flags[0]
        assert f.evidence["worst_window_mean_bpm"] > 100
        assert f.threshold["tachy_bpm"] == th.TACHY_SUSTAINED_BPM
        assert f.disclaimer

    def test_normal_hr_no_flags(self) -> None:
        assert flag_sustained_hr(_steady(72.0), default_limits(False)) == []

    def test_brief_spike_does_not_flag(self) -> None:
        # 20 s at 130 bpm inside 5 min at 75 bpm: no 45 s window averages >100.
        base = _steady(75.0)
        rr = base.rr_ms.copy()
        t = base.t_s
        spike = (t >= 100.0) & (t < 120.0)
        rr[spike] = 60000.0 / 130.0
        assert flag_sustained_hr(_series(rr), default_limits(False)) == []

    def test_sustained_low_hr_flags_with_note(self) -> None:
        flags = flag_sustained_hr(_steady(52.0), default_limits(False))
        assert [f.kind for f in flags] == ["sustained_low_hr"]
        assert "low-specificity" in flags[0].description

    def test_athlete_baseline_raises_specificity(self) -> None:
        athlete = default_limits(True)
        # 52 bpm: crosses the default 60 threshold but not the athlete 45.
        assert flag_sustained_hr(_steady(52.0), athlete) == []
        # 40 bpm: crosses even the athlete threshold, and the text says the
        # athlete baseline was taken into account.
        flags = flag_sustained_hr(_steady(40.0), athlete)
        assert [f.kind for f in flags] == ["sustained_low_hr"]
        assert "athlete-baseline" in flags[0].description
        assert flags[0].evidence["athlete_note_applied"] is True

    def test_activity_limits_respected(self) -> None:
        running = HRLimits(tachy_bpm=180.0, brady_bpm=40.0, context="running")
        assert flag_sustained_hr(_steady(140.0), running) == []
        flags = flag_sustained_hr(_steady(190.0), running)
        assert [f.kind for f in flags] == ["sustained_high_hr"]
        assert "running" in flags[0].description


class TestEctopy:
    def _arrays(self, n: int = 500) -> tuple[np.ndarray, np.ndarray]:
        prematurity = np.zeros(n)
        motion = np.zeros(n, dtype=bool)
        return prematurity, motion

    def test_confirmed_ectopy_above_1pct_flags(self) -> None:
        prematurity, motion = self._arrays()
        ectopic = np.arange(30, 530, 40)[:12]  # 12 beats = 2.4 %
        prematurity[ectopic] = -30.0
        flags = flag_ectopy(500, ectopic, prematurity, motion)
        assert [f.kind for f in flags] == ["ectopy"]
        f = flags[0]
        assert f.evidence["n_confirmed_ectopic"] == 12
        assert "ectopic beat" in f.description
        assert "single lead" in f.description.lower()

    def test_motion_explained_beats_do_not_flag(self) -> None:
        prematurity, motion = self._arrays()
        ectopic = np.arange(30, 530, 40)[:12]
        prematurity[ectopic] = -30.0
        motion[ectopic] = True  # every one coincides with motion
        assert flag_ectopy(500, ectopic, prematurity, motion) == []

    def test_non_premature_correction_class_does_not_flag(self) -> None:
        # The correction step flagged beats, but none are premature — exactly
        # the real-data case (20 morphology outliers, none premature, all
        # motion): must NOT be reported as ectopy.
        prematurity, motion = self._arrays()
        ectopic = np.arange(30, 530, 40)[:12]
        prematurity[ectopic] = +5.0  # on-time or late
        assert flag_ectopy(500, ectopic, prematurity, motion) == []

    def test_below_1pct_without_pattern_does_not_flag(self) -> None:
        prematurity, motion = self._arrays()
        ectopic = np.array([100, 250, 400])  # 0.6 %, scattered
        prematurity[ectopic] = -30.0
        assert flag_ectopy(500, ectopic, prematurity, motion) == []

    def test_bigeminy_flags_below_1pct(self) -> None:
        prematurity, motion = self._arrays(1000)
        ectopic = np.array([100, 102, 104, 106])  # 0.4 % but alternating
        prematurity[ectopic] = -30.0
        flags = flag_ectopy(1000, ectopic, prematurity, motion)
        assert [f.kind for f in flags] == ["ectopy"]
        assert flags[0].evidence["bigeminy_pattern"] is True

    def test_never_says_pvc(self) -> None:
        prematurity, motion = self._arrays()
        ectopic = np.arange(30, 530, 40)[:12]
        prematurity[ectopic] = -30.0
        (flag,) = flag_ectopy(500, ectopic, prematurity, motion)
        assert "pvc" not in flag.description.lower()
        assert "ventricular" not in flag.description.lower().replace(
            "atrial from ventricular origin", ""
        )


class TestIrregularity:
    def test_af_like_series_flags(self) -> None:
        rng = np.random.default_rng(11)
        rr = rng.uniform(450.0, 1250.0, 240)
        flags = flag_irregularity(_series(rr))
        assert [f.kind for f in flags] == ["rr_irregularity"]
        f = flags[0]
        assert f.evidence["max_cosen"] > th.COSEN_THRESHOLD
        assert f.evidence["max_rr_cv"] > th.RR_CV_THRESHOLD
        assert "consistent with" in f.description
        assert "cannot resolve p-waves" in f.description.lower()

    def test_metronome_does_not_flag(self) -> None:
        rng = np.random.default_rng(12)
        rr = 800.0 + rng.normal(0, 2.0, 240)
        assert flag_irregularity(_series(rr)) == []

    def test_healthy_high_hrv_sinus_does_not_flag(self) -> None:
        # High vagal tone: big smooth respiratory swings (CV ≈ 7.5 %) but
        # structured — COSEn stays low. The dual gate must hold.
        t = np.cumsum(np.full(240, 850.0)) / 1000.0
        rng = np.random.default_rng(13)
        rr = 850.0 + 60.0 * np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, 10.0, 240)
        assert flag_irregularity(_series(rr)) == []

    def test_brief_irregular_run_does_not_flag(self) -> None:
        rng = np.random.default_rng(14)
        rr = 800.0 + rng.normal(0, 15.0, 300)
        rr[100:115] = rng.uniform(450.0, 1250.0, 15)  # < window length
        assert flag_irregularity(_series(rr)) == []

    def test_windows_never_span_discontinuities(self) -> None:
        rng = np.random.default_rng(15)
        # Two clean halves; the seam would look irregular if bridged.
        rr = np.concatenate([np.full(120, 600.0), np.full(120, 1100.0)])
        rr += rng.normal(0, 5.0, 240)
        series = _series(rr)
        series.discontinuity[120] = True
        assert flag_irregularity(series) == []


class TestEveryFlagCarriesEverything:
    """Spec: every flag carries kind, severity, description, evidence,
    threshold, and the disclaimer."""

    def _all_flags(self) -> list:
        flags = []
        flags += flag_sustained_hr(_steady(110.0), default_limits(False))
        flags += flag_sustained_hr(_steady(50.0), default_limits(False))
        prematurity = np.zeros(500)
        motion = np.zeros(500, dtype=bool)
        ectopic = np.arange(30, 530, 40)[:12]
        prematurity[ectopic] = -30.0
        flags += flag_ectopy(500, ectopic, prematurity, motion)
        rng = np.random.default_rng(11)
        flags += flag_irregularity(_series(rng.uniform(450.0, 1250.0, 240)))
        return flags

    def test_all_fields_populated(self) -> None:
        flags = self._all_flags()
        assert len(flags) == 4  # one of each kind
        for f in flags:
            assert f.kind and f.severity and f.description
            assert f.evidence and f.threshold
            assert f.disclaimer and f.disclaimer.strip()
