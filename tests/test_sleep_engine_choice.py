"""Choosing which staging engine leads a night.

The contract under test: the choice decides which hypnogram becomes primary
(and so fills the queryable Metrics columns), never which engines run — the
pairwise agreement table has to survive, because it is how a reader sees
that two algorithms disagree about the same night. A choice that produced
no staging falls back to the precedence order and says so.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from app.activities.base import PersonContext
from app.sleep import orchestrator
from app.sleep.engines import EngineStatus, sleepecg_engine
from app.sleep.stages import EPOCH_LEN_S, Hypnogram, StageVocab, make_epoch_grid
from tests.synth_util import overnight_rr, rr_series_from

START = dt.datetime(2026, 8, 10, 22, 30, tzinfo=dt.UTC)


def _night():
    """A stage-programmed night as the orchestrator's minimal inputs."""
    rr_ms, t_s, _truth = overnight_rr(
        stage_plan=[("wake", 3), ("light", 14), ("deep", 10), ("rem", 8)], seed=11
    )
    rr = rr_series_from(rr_ms, t_s)

    class FakeRec:
        duration_s = float(t_s[-1])
        start_time = START
        ecg_mv = np.array([])
        sampling_rate_hz = 130.0

    class FakeQuality:
        windows = []
        excluded_segments = []
        excluded_total_s = 0.0
        analysed_total_s = float(t_s[-1])

    class FakeResult:
        pass

    result = FakeResult()
    result.rr = rr
    result.quality = FakeQuality()
    result.peak_times_s = t_s
    return FakeRec(), result


def _enable_fake_sleepecg(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make sleepecg 'available' and return a deterministic 3-class night."""
    monkeypatch.setattr(
        sleepecg_engine,
        "status",
        lambda: EngineStatus(
            sleepecg_engine.ENGINE_KEY, sleepecg_engine.ENGINE_LABEL, True
        ),
    )

    def fake_stage(peak_times_s, start_time, duration_s, ctx=None, sex=None):
        grid = make_epoch_grid(duration_s)
        # Wake for the first three minutes, then NREM — coarse but real
        # staging, so summarize() and the agreement table have something
        # to work with.
        stages = np.where(grid < 180.0, 0, 1).astype(np.int8)
        return Hypnogram(
            engine=sleepecg_engine.ENGINE_KEY,
            engine_label=sleepecg_engine.ENGINE_LABEL,
            vocab=StageVocab.WAKE_REM_NREM,
            epoch_len_s=EPOCH_LEN_S,
            epoch_start_s=grid,
            stages=stages,
            probabilities=None,
            accuracy_note="fake engine for tests",
            notes=[],
        )

    monkeypatch.setattr(sleepecg_engine, "stage_sleepecg", fake_stage)


def _run(engine_pref=None):
    rec, result = _night()
    return orchestrator.run_sleep_analysis(
        rec, result, None, PersonContext(), {}, engine_pref=engine_pref
    )


class TestPreferenceHonoured:
    def test_chosen_engine_becomes_primary(self, monkeypatch) -> None:
        _enable_fake_sleepecg(monkeypatch)
        analysis = _run(engine_pref="sleepecg")
        assert analysis.primary_engine == "sleepecg"
        assert analysis.engine_pref == "sleepecg"

    def test_default_still_follows_the_precedence_order(self, monkeypatch) -> None:
        _enable_fake_sleepecg(monkeypatch)
        analysis = _run()
        # sleepecg outranks heuristic when no preference is expressed.
        assert analysis.primary_engine == "sleepecg"
        assert analysis.engine_pref is None

    def test_choosing_the_lower_ranked_engine_overrides_precedence(
        self, monkeypatch
    ) -> None:
        _enable_fake_sleepecg(monkeypatch)
        analysis = _run(engine_pref="heuristic")
        assert analysis.primary_engine == "heuristic"

    def test_every_engine_still_runs_and_agreement_survives(self, monkeypatch) -> None:
        """The whole point of (b): the choice must not cost the kappa table."""
        _enable_fake_sleepecg(monkeypatch)
        analysis = _run(engine_pref="sleepecg")
        assert {h.engine for h in analysis.hypnograms} == {"heuristic", "sleepecg"}
        assert set(analysis.summaries) == {"heuristic", "sleepecg"}
        assert analysis.agreement, "pairwise agreement must survive an engine choice"
        pair = {analysis.agreement[0].engine_a, analysis.agreement[0].engine_b}
        assert pair == {"heuristic", "sleepecg"}

    def test_preference_is_echoed_into_extras(self, monkeypatch) -> None:
        _enable_fake_sleepecg(monkeypatch)
        extras = _run(engine_pref="sleepecg").as_extras()
        assert extras["engine_pref"] == "sleepecg"
        assert extras["primary_engine"] == "sleepecg"


class TestFallbackIsNeverSilent:
    def test_unavailable_choice_falls_back_with_a_note(self, monkeypatch) -> None:
        monkeypatch.setattr(
            sleepecg_engine,
            "status",
            lambda: EngineStatus(
                sleepecg_engine.ENGINE_KEY,
                sleepecg_engine.ENGINE_LABEL,
                False,
                "extras not installed",
            ),
        )
        analysis = _run(engine_pref="sleepecg")
        assert analysis.primary_engine == "heuristic"
        note = " ".join(analysis.notes)
        assert sleepecg_engine.ENGINE_LABEL in note
        assert "extras not installed" in note
        assert "Built-in rules" in note  # names what actually filled the columns

    def test_failed_choice_falls_back_with_a_note(self, monkeypatch) -> None:
        monkeypatch.setattr(
            sleepecg_engine,
            "status",
            lambda: EngineStatus(
                sleepecg_engine.ENGINE_KEY, sleepecg_engine.ENGINE_LABEL, True
            ),
        )

        def boom(*args, **kwargs):
            raise RuntimeError("classifier exploded")

        monkeypatch.setattr(sleepecg_engine, "stage_sleepecg", boom)
        analysis = _run(engine_pref="sleepecg")
        assert analysis.primary_engine == "heuristic"
        assert any("produced no staging" in n for n in analysis.notes)

    def test_unknown_engine_key_does_not_raise(self) -> None:
        """A key from a build that no longer exists degrades, never errors."""
        analysis = _run(engine_pref="engine-from-the-future")
        assert analysis.primary_engine == "heuristic"
        assert any("engine-from-the-future" in n for n in analysis.notes)

    def test_no_preference_adds_no_note(self, monkeypatch) -> None:
        _enable_fake_sleepecg(monkeypatch)
        analysis = _run()
        assert not any("produced no staging" in n for n in analysis.notes)


class TestVocabularyHonesty:
    def test_three_class_primary_reports_no_light_or_deep(self, monkeypatch) -> None:
        """Switching to a coarser engine blanks those minutes — never invents."""
        _enable_fake_sleepecg(monkeypatch)
        analysis = _run(engine_pref="sleepecg")
        summary = analysis.primary_summary()
        assert summary.light_min is None
        assert summary.deep_min is None
        # The four-class engine still knows them, in its own summary.
        assert analysis.summaries["heuristic"].deep_min is not None
