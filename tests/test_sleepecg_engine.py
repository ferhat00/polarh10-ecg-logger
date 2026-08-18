"""SleepECG engine: mocked-module tests (run everywhere, no TensorFlow) plus
a real-inference smoke test that runs only where the optional extras are
installed."""

from __future__ import annotations

import datetime as dt
import sys
import types

import numpy as np
import pytest

from app.activities.base import PersonContext
from app.sleep.engines import sleepecg_engine
from app.sleep.stages import UNSCORED, StageVocab

START = dt.datetime(2026, 8, 10, 22, 30, tzinfo=dt.UTC)


class _FakeGender:
    FEMALE = 0
    MALE = 1


def _fake_sleepecg(probs: np.ndarray, captured: dict) -> types.ModuleType:
    """A stand-in sleepecg module returning a fixed probability matrix."""
    fake = types.ModuleType("sleepecg")

    class SubjectData:
        def __init__(self, gender=None, age=None, weight=None):
            captured["gender"] = gender
            captured["age"] = age

    class SleepRecord:
        def __init__(self, **kwargs):
            captured["record_kwargs"] = kwargs

    def load_classifier(name, classifiers_dir=None, silence_tf_messages=True):
        captured["classifier"] = (name, classifiers_dir)
        return object()

    def stage(clf, record, return_mode="int"):
        assert return_mode == "prob"
        return probs

    fake.SubjectData = SubjectData
    fake.SleepRecord = SleepRecord
    fake.load_classifier = load_classifier
    fake.stage = stage
    fake.Gender = _FakeGender
    return fake


def _probs_for_ints(ints: list[int]) -> np.ndarray:
    """(n, 4) matrix whose argmax reproduces sleepecg's integer labels
    (columns: UNDEFINED, NREM, REM, WAKE)."""
    probs = np.full((len(ints), 4), 0.1)
    for i, label in enumerate(ints):
        probs[i, label] = 0.7
    return probs


class TestMappingAndPadding:
    def _run(self, ints, duration_s, monkeypatch, ctx=None, sex=None):
        captured: dict = {}
        monkeypatch.setitem(
            sys.modules, "sleepecg", _fake_sleepecg(_probs_for_ints(ints), captured)
        )
        hyp = sleepecg_engine.stage_sleepecg(
            np.arange(0.5, duration_s, 1.0), START, duration_s, ctx, sex
        )
        return hyp, captured

    def test_integer_mapping_verified(self, monkeypatch) -> None:
        # sleepecg ints: 0=UNDEFINED 1=NREM 2=REM 3=WAKE
        hyp, _cap = self._run([3, 1, 2, 0], 120.0, monkeypatch)
        assert hyp.vocab == StageVocab.WAKE_REM_NREM
        assert hyp.stages.tolist() == [0, 1, 2, UNSCORED]

    def test_probability_columns_reordered(self, monkeypatch) -> None:
        hyp, _cap = self._run([3, 1], 60.0, monkeypatch)
        # Row 0: WAKE dominant -> our column 0 (WAKE) carries the 0.7.
        assert hyp.probabilities[0].tolist() == [0.7, 0.1, 0.1]
        # Row 1: NREM dominant -> our column 1 (NREM).
        assert hyp.probabilities[1].tolist() == [0.1, 0.7, 0.1]

    def test_short_output_padded_unscored(self, monkeypatch) -> None:
        hyp, _cap = self._run([1, 1, 1], 150.0, monkeypatch)  # 5-epoch grid
        assert hyp.n_epochs == 5
        assert hyp.stages.tolist() == [1, 1, 1, UNSCORED, UNSCORED]
        assert any("unscored" in n for n in hyp.notes)

    def test_long_output_trimmed_to_grid(self, monkeypatch) -> None:
        hyp, _cap = self._run([1] * 6, 120.0, monkeypatch)  # 4-epoch grid
        assert hyp.n_epochs == 4
        assert hyp.stages.tolist() == [1, 1, 1, 1]

    def test_subject_data_from_person(self, monkeypatch) -> None:
        ctx = PersonContext(age_years=41.7, sex="female")
        _hyp, cap = self._run([1], 30.0, monkeypatch, ctx=ctx, sex="female")
        assert cap["age"] == 41
        assert cap["gender"] == _FakeGender.FEMALE
        assert cap["classifier"] == ("wrn-gru-mesa-weighted", "SleepECG")
        assert cap["record_kwargs"]["sleep_stage_duration"] == 30

    def test_missing_demographics_noted(self, monkeypatch) -> None:
        hyp, cap = self._run([1], 30.0, monkeypatch)
        assert cap["age"] is None and cap["gender"] is None
        assert any("person profile" in n for n in hyp.notes)


class TestStatus:
    def test_unavailable_reason_names_requirements_file(self, monkeypatch) -> None:
        import importlib.util

        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name, *args, **kwargs):
            if name in ("sleepecg", "tensorflow"):
                return None
            return real_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        st = sleepecg_engine.status()
        assert st.available is False
        assert "requirements-sleep.txt" in st.unavailable_reason

    def test_available_when_modules_present(self) -> None:
        pytest.importorskip("sleepecg")
        pytest.importorskip("tensorflow")
        assert sleepecg_engine.status().available is True


class TestOrchestratorDegradation:
    def test_engine_failure_keeps_heuristic(self, monkeypatch) -> None:
        from app.sleep import orchestrator
        from app.sleep.engines import EngineStatus

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

        from app.ingest.loader import LoadedRecording
        from tests.synth_util import overnight_rr, rr_series_from

        rr_ms, t_s, _ = overnight_rr(stage_plan=[("light", 40)], seed=6)
        rr = rr_series_from(rr_ms, t_s)

        class FakeRec:
            duration_s = float(t_s[-1])
            start_time = START

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

        analysis = orchestrator.run_sleep_analysis(
            FakeRec(), result, None, PersonContext(), {}
        )
        assert {h.engine for h in analysis.hypnograms} == {"heuristic"}
        assert analysis.primary_engine == "heuristic"
        failed = [s for s in analysis.engines if s.key == "sleepecg"]
        assert failed and failed[0].available is False
        assert "classifier exploded" in failed[0].unavailable_reason
        assert LoadedRecording is not None  # keep the import honest


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestRealInference:
    def test_real_classifier_smoke(self) -> None:
        pytest.importorskip("sleepecg")
        pytest.importorskip("tensorflow")
        from tests.synth_util import overnight_rr

        _rr_ms, t_s, _truth = overnight_rr(
            stage_plan=[("wake", 5), ("light", 20), ("deep", 15), ("rem", 10)],
            seed=2,
        )
        duration = float(t_s[-1])
        hyp = sleepecg_engine.stage_sleepecg(t_s, START, duration)
        assert hyp.vocab == StageVocab.WAKE_REM_NREM
        assert hyp.n_epochs == len(np.arange(0, duration, 30.0))
        scored = hyp.stages[hyp.stages != UNSCORED]
        assert len(scored) > 0
        assert set(np.unique(scored)) <= {0, 1, 2}
        assert hyp.probabilities.shape == (hyp.n_epochs, 3)
