"""Ectopy event extraction: grouping, pause ratios, patterns, pipeline wiring."""

from __future__ import annotations

import datetime as dt

import numpy as np

from app.ingest.loader import load_polar_csv
from app.pipeline.events import (
    confirmed_ectopic_indices,
    extract_events,
)
from app.pipeline.process import run_pipeline
from app.screening import thresholds as th
from app.screening.rules import flag_ectopy
from tests.synth_util import build_ecg_from_rr

NOW = dt.datetime(2026, 8, 16, 12, 0, tzinfo=dt.UTC)
START = dt.datetime(2026, 8, 10, 9, 0, tzinfo=dt.UTC)


def _peak_times(rr_ms: list[float], t0: float = 0.5) -> np.ndarray:
    """Beat times (s) from an RR series in ms."""
    return np.concatenate(([t0], t0 + np.cumsum(np.asarray(rr_ms)) / 1000.0))


def _steady_with_premature(
    n: int = 40, premature_at: tuple[int, ...] = (20,), post_ms: float = 1050.0
) -> np.ndarray:
    """Steady 800 ms rhythm with premature beats (550 ms coupling)."""
    rr = [800.0] * n
    for k in premature_at:
        rr[k - 1] = 550.0
        rr[k] = post_ms
    return _peak_times(rr)


class TestGrouping:
    def test_single_couplet_run_classification(self) -> None:
        times = _peak_times([800.0] * 60)
        idx = np.array([5, 20, 21, 40, 41, 42, 43])
        ev = extract_events(idx, times, analysed_s=48.0, excluded_segments=[])
        assert ev.n_confirmed == 7
        assert ev.n_singles == 1
        assert ev.n_couplets == 1
        assert ev.n_runs == 1
        assert ev.longest_run_beats == 4
        kinds = [e.kind for e in ev.events]
        assert kinds == ["single", "couplet", "run"]

    def test_event_times_and_mask(self) -> None:
        times = _steady_with_premature()
        ev = extract_events(np.array([20]), times, 32.0, [])
        (e,) = ev.events
        assert e.start_beat == e.end_beat == 20
        assert e.t_start_s == times[20]
        assert ev.confirmed_mask[20]
        assert ev.confirmed_mask.sum() == 1

    def test_no_events(self) -> None:
        times = _peak_times([800.0] * 30)
        ev = extract_events(np.array([], dtype=np.int64), times, 24.0, [])
        assert ev.n_confirmed == 0
        assert ev.events == []
        assert ev.per_hour == 0.0
        assert ev.longest_run_beats == 0

    def test_out_of_range_indices_ignored(self) -> None:
        times = _peak_times([800.0] * 10)
        ev = extract_events(np.array([-3, 5, 99]), times, 8.0, [])
        assert ev.n_confirmed == 1

    def test_rates(self) -> None:
        times = _peak_times([800.0] * 40)  # 41 beats
        ev = extract_events(np.array([10, 30]), times, analysed_s=1800.0, excluded_segments=[])
        assert ev.per_hour == 4.0  # 2 in half an hour
        assert ev.pct_of_beats == 100.0 * 2 / 41


class TestPauseRatio:
    def test_full_compensatory_pause(self) -> None:
        # 550 + 1050 = 2 x 800: the classical full pause.
        times = _steady_with_premature(post_ms=1050.0)
        ev = extract_events(np.array([20]), times, 32.0, [])
        (e,) = ev.events
        assert e.pause_ratio is not None
        assert abs(e.pause_ratio - 1.0) < 0.01
        assert e.pause_complete is True

    def test_incomplete_pause(self) -> None:
        # 550 + 850 = 1400 < 0.95 x 1600: sinus-reset pattern.
        times = _steady_with_premature(post_ms=850.0)
        ev = extract_events(np.array([20]), times, 32.0, [])
        (e,) = ev.events
        assert e.pause_ratio is not None
        assert e.pause_ratio < th.PAUSE_COMPLETE_RATIO
        assert e.pause_complete is False

    def test_none_at_recording_edges(self) -> None:
        times = _peak_times([800.0] * 10)
        ev = extract_events(np.array([0, 10]), times, 8.0, [])
        assert all(e.pause_ratio is None for e in ev.events)
        assert ev.n_pause_complete == 0
        assert ev.n_pause_incomplete == 0

    def test_none_when_neighbour_interval_not_physiological(self) -> None:
        rr = [800.0] * 40
        rr[19] = 550.0
        rr[20] = 2500.0  # detector dropout after the event, not a pause
        times = _peak_times(rr)
        ev = extract_events(np.array([20]), times, 32.0, [])
        (e,) = ev.events
        assert e.pause_ratio is None
        assert e.pause_complete is None

    def test_none_when_neighbour_interval_excluded(self) -> None:
        times = _steady_with_premature()
        t_pre = float(times[19]), float(times[20])
        excluded = [(t_pre[0] + 0.1, t_pre[1] - 0.1, "test segment")]
        ev = extract_events(np.array([20]), times, 32.0, excluded)
        (e,) = ev.events
        assert e.pause_ratio is None


class TestPatterns:
    def test_bigeminy_episode_counted(self) -> None:
        times = _peak_times([800.0] * 60)
        idx = np.array([10, 12, 14, 16])  # every other beat, 4 pattern beats
        ev = extract_events(idx, times, 48.0, [])
        assert ev.bigeminy_episodes == 1
        assert ev.trigeminy_episodes == 0

    def test_trigeminy_episode_counted(self) -> None:
        times = _peak_times([800.0] * 60)
        idx = np.array([10, 13, 16])  # every third beat
        ev = extract_events(idx, times, 48.0, [])
        assert ev.trigeminy_episodes == 1
        assert ev.bigeminy_episodes == 0

    def test_short_chain_is_no_episode(self) -> None:
        times = _peak_times([800.0] * 60)
        idx = np.array([10, 12])  # only 2 pattern beats < BIGEMINY_MIN_RUN
        ev = extract_events(idx, times, 48.0, [])
        assert ev.bigeminy_episodes == 0

    def test_two_separate_bigeminy_episodes(self) -> None:
        times = _peak_times([800.0] * 80)
        idx = np.array([10, 12, 14, 40, 42, 44])
        ev = extract_events(idx, times, 64.0, [])
        assert ev.bigeminy_episodes == 2


class TestConfirmationGate:
    """The shared triple gate, including the Kubios index-offset resolution."""

    def _arrays(self, n: int = 500) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(n), np.zeros(n, dtype=bool)

    def test_matches_flag_ectopy_evidence(self) -> None:
        # Anti-drift: the helper and the screening flag must agree exactly.
        prematurity, motion = self._arrays()
        ectopic = np.arange(30, 530, 40)[:12]
        prematurity[ectopic] = -30.0
        confirmed = confirmed_ectopic_indices(500, ectopic, prematurity, motion)
        (flag,) = flag_ectopy(500, ectopic, prematurity, motion)
        assert flag.evidence["beat_indices"] == [int(i) for i in confirmed[:100]]
        assert flag.evidence["n_confirmed_ectopic"] == len(confirmed)

    def test_offset_index_resolves_to_premature_beat(self) -> None:
        # neurokit's Kubios classification can index the pause beat; the gate
        # must confirm the premature neighbour, by its own index.
        prematurity, motion = self._arrays()
        prematurity[60] = -31.0
        prematurity[61] = +33.0  # the compensatory pause
        confirmed = confirmed_ectopic_indices(500, np.array([61]), prematurity, motion)
        assert list(confirmed) == [60]

    def test_adjacent_candidates_deduplicate(self) -> None:
        prematurity, motion = self._arrays()
        prematurity[60] = -31.0
        confirmed = confirmed_ectopic_indices(
            500, np.array([60, 61]), prematurity, motion
        )
        assert list(confirmed) == [60]

    def test_motion_vetoes(self) -> None:
        prematurity, motion = self._arrays()
        prematurity[60] = -31.0
        motion[60] = True
        assert len(confirmed_ectopic_indices(500, np.array([61]), prematurity, motion)) == 0

    def test_not_premature_not_confirmed(self) -> None:
        prematurity, motion = self._arrays()
        prematurity[60] = -10.0  # above the -20 % gate
        assert len(confirmed_ectopic_indices(500, np.array([60]), prematurity, motion)) == 0


class TestPipelineIntegration:
    def _write_csv(self, tmp_path, rr_ms: np.ndarray, beat_amps=None) -> str:
        t, ecg, _beats = build_ecg_from_rr(rr_ms, seed=21, beat_amps=beat_amps)
        start_ns = int(START.timestamp() * 1e9)
        lines = ["time,ecg,hr,rr,marker"]
        for ti, v in zip(t, ecg, strict=True):
            lines.append(f"{start_ns + round(ti * 1e9)},{v:.4f}")
        path = tmp_path / "ecg_2026-08-10.csv"
        path.write_text("\n".join(lines) + "\n")
        return str(path)

    def test_injected_premature_beats_become_events(self, tmp_path) -> None:
        rng = np.random.default_rng(21)
        n_beats = 190
        t_beat = np.arange(n_beats) * 0.8
        rr = 800.0 + 25.0 * np.sin(2 * np.pi * 0.1 * t_beat) + rng.normal(0, 8.0, n_beats)
        for k in (60, 120):  # premature beat + full compensatory pause
            rr[k - 1] = 550.0
            rr[k] = 1050.0
        path = self._write_csv(tmp_path, rr, beat_amps={60: 0.25, 120: 0.25})

        rec = load_polar_csv(path, now=NOW)
        result = run_pipeline(rec)

        ev = result.events
        assert ev is not None
        assert ev.n_confirmed == 2
        assert ev.n_singles == 2
        assert ev.n_pause_complete == 2
        assert ev.per_hour is not None and ev.per_hour > 0
        # The event times sit near the injected beats (raw-train clock).
        event_times = sorted(e.t_start_s for e in ev.events)
        assert abs(event_times[0] - (0.5 + np.sum(rr[:60]) / 1000.0)) < 0.2
        assert any("confirmed ectopic beat(s)" in n for n in result.notes)

    def test_clean_recording_has_no_confirmed_events(self, tmp_path) -> None:
        rng = np.random.default_rng(7)
        n_beats = 190
        t_beat = np.arange(n_beats) * 0.8
        rr = 800.0 + 25.0 * np.sin(2 * np.pi * 0.1 * t_beat) + rng.normal(0, 8.0, n_beats)
        path = self._write_csv(tmp_path, rr)

        rec = load_polar_csv(path, now=NOW)
        result = run_pipeline(rec)

        assert result.events is not None
        assert result.events.n_confirmed == 0
        assert not any("confirmed ectopic beat(s)" in n for n in result.notes)
