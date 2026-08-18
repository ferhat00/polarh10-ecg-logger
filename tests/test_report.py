"""Report-rendering tests: self-containment, disclaimers, honest labelling."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pytest

from app.activities import get_profile
from app.activities.base import ActivityInputs, PersonContext, apply_suppressions
from app.ingest.loader import load_polar_csv
from app.pipeline.process import run_pipeline
from app.report.render import ReportMeta, build_report_html
from app.screening.flags import NO_FLAGS_STATEMENT, ScreeningFlag
from app.screening.rules import run_screening

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_supine.csv"
NOW = dt.datetime(2026, 8, 16, 12, 0, tzinfo=dt.UTC)


@pytest.fixture(scope="module")
def rendered() -> dict:
    rec = load_polar_csv(FIXTURE, now=NOW)
    result = run_pipeline(rec)
    profile = get_profile("supine")
    ctx = PersonContext()
    inputs = ActivityInputs.from_pipeline(result, markers=rec.markers)
    analysis = profile.analyze(inputs, ctx)
    hrv_censored, _ = apply_suppressions(result.hrv, analysis.suppressions)
    flags = run_screening(result)
    meta = ReportMeta(
        person_name="Test Person",
        activity_name="Lying down (supine)",
        recorded_at=rec.start_time,
        original_filename=FIXTURE.name,
        context_note="unit-test render",
    )
    html = build_report_html(rec, result, hrv_censored, analysis, None, flags, meta)
    return {
        "html": html,
        "rec": rec,
        "result": result,
        "analysis": analysis,
        "meta": meta,
        "hrv": hrv_censored,
    }


class TestSelfContained:
    """All health data stays local: the report makes zero external requests."""

    def test_no_external_urls(self, rendered: dict) -> None:
        html = rendered["html"]
        assert not re.search(r'(src|href)\s*=\s*["\']https?://', html)
        assert "cdn." not in html
        assert "googleapis" not in html

    def test_fonts_embedded(self, rendered: dict) -> None:
        assert rendered["html"].count("data:font/woff2;base64,") == 4
        assert "IBM Plex Mono" in rendered["html"]

    def test_figures_inlined(self, rendered: dict) -> None:
        # Strips (≥3) + hr + poincaré + histogram + sdnn + spectrum +
        # template + quality.
        assert rendered["html"].count("data:image/png;base64,") >= 9


class TestTrueScaleStrips:
    def test_scale_stated_and_css_width_set(self, rendered: dict) -> None:
        html = rendered["html"]
        assert "25 mm/s, 10 mm/mV" in html
        assert "one small square = 40 ms" in html
        assert re.search(r'style="width: \d+mm', html)

    def test_calibration_pulse_in_masthead(self, rendered: dict) -> None:
        assert "1&nbsp;mV" in rendered["html"]


class TestHeadlinesAndHonesty:
    def test_excluded_time_is_first_kpi(self, rendered: dict) -> None:
        html = rendered["html"]
        first_kpi = html.split('class="kpi headline"')[1][:300]
        assert "Excluded time" in first_kpi

    def test_no_flags_statement_for_clean_fixture(self, rendered: dict) -> None:
        assert NO_FLAGS_STATEMENT in rendered["html"]
        assert "not clinical clearance" in rendered["html"]

    def test_sdnn_labelled_trend_inclusive(self, rendered: dict) -> None:
        assert "trend-inclusive" in rendered["html"]

    def test_interval_metrics_absent_and_explained(self, rendered: dict) -> None:
        html = rendered["html"]
        # Explained exactly once, in the limits section — never as a metric.
        assert html.count("QRS width, QT/QTc, PR interval, and ECG axis") == 1
        assert "quantisation artifacts" in html

    def test_engine_and_unit_provenance_shown(self, rendered: dict) -> None:
        html = rendered["html"]
        assert "neurokit2" in html
        assert "millivolts" in html  # loader unit-detection note surfaced
        assert "130.030 Hz" in html or "130.03" in html

    def test_mayer_wave_labelled_not_respiration(self, rendered: dict) -> None:
        html = rendered["html"]
        assert "Mayer" in html
        assert "not a respiratory rate" in html
        assert "breaths/min" not in html

    def test_per_minute_table_present(self, rendered: dict) -> None:
        assert "Per-minute detail" in rendered["html"]
        assert rendered["html"].count("<tr>") >= 5


class TestFlagsRendering:
    def test_every_flag_renders_with_adjacent_disclaimer(self, rendered: dict) -> None:
        flags = [
            ScreeningFlag(
                kind="sustained_high_hr",
                severity="review",
                description="Sustained heart rate above the screening threshold.",
                evidence={"worst_window_mean_bpm": 112.0},
                threshold={"tachy_bpm": 100.0},
            ),
            ScreeningFlag(
                kind="rr_irregularity",
                severity="review",
                description="Irregularity pattern consistent with possible atrial "
                "fibrillation; single-lead data cannot resolve P-waves.",
                evidence={"max_cosen": 0.1},
                threshold={"cosen_threshold": -1.0},
            ),
        ]
        html = build_report_html(
            rendered["rec"],
            rendered["result"],
            rendered["hrv"],
            rendered["analysis"],
            None,
            flags,
            rendered["meta"],
        )
        # Each flag card carries its own disclaimer div — 2 flags, 2 disclaimers.
        assert html.count('class="disclaimer"') == 2
        assert html.count(flags[0].disclaimer) == 2
        for f in flags:
            assert f.description in html
        assert NO_FLAGS_STATEMENT not in html


class TestSuppressedReport:
    def test_running_at_intensity_shows_suppression_not_values(self) -> None:
        rec = load_polar_csv(FIXTURE, now=NOW)
        result = run_pipeline(rec)
        # Force the intensity gate: fixture mean HR is 75 bpm, and with
        # max 80 / rest 50 that is 83 % of heart-rate reserve.
        ctx = PersonContext(max_hr_bpm=80, resting_hr_bpm=50)
        profile = get_profile("running")
        inputs = ActivityInputs.from_pipeline(result)
        analysis = profile.analyze(inputs, ctx)
        assert analysis.suppressions  # 75 bpm mean = 50% ... ensure suppressed
        hrv_censored, _ = apply_suppressions(result.hrv, analysis.suppressions)
        html = build_report_html(
            rec, result, hrv_censored, analysis, None, [],
            ReportMeta(person_name="T", activity_name="Running",
                       original_filename="f.csv"),
        )
        assert "Suppressed for this activity" in html
        assert "heart-rate" in html and "reserve" in html


class TestSleepSection:
    def _sleep_analysis(self):
        import numpy as np

        from app.sleep.actigraphy import AccEpochs
        from app.sleep.agreement import pairwise_agreement
        from app.sleep.orchestrator import SleepAnalysis
        from app.sleep.stages import EPOCH_LEN_S, Hypnogram, StageVocab
        from app.sleep.summary import summarize

        n = 80
        stages = np.array([0] * 4 + [1] * 30 + [2] * 20 + [3] * 20 + [1] * 6, dtype=np.int8)
        grid = np.arange(n) * EPOCH_LEN_S
        hyps = [
            Hypnogram(
                engine="heuristic",
                engine_label="Built-in rules (cardio-actigraphy)",
                vocab=StageVocab.WAKE_LIGHT_DEEP_REM,
                epoch_len_s=EPOCH_LEN_S,
                epoch_start_s=grid,
                stages=stages,
                probabilities=None,
                accuracy_note="Rule-based estimate: roughly 65-75% epoch agreement.",
            ),
            Hypnogram(
                engine="sleepecg",
                engine_label="SleepECG GRU (wrn-gru-mesa)",
                vocab=StageVocab.WAKE_REM_NREM,
                epoch_len_s=EPOCH_LEN_S,
                epoch_start_s=grid,
                stages=np.array([0] * 4 + [1] * 50 + [2] * 20 + [1] * 6, dtype=np.int8),
                probabilities=None,
                accuracy_note="GRU trained on MESA.",
            ),
        ]
        analysis = SleepAnalysis(hypnograms=hyps)
        analysis.summaries = {h.engine: summarize(h) for h in hyps}
        analysis.primary_engine = "sleepecg"
        analysis.agreement = pairwise_agreement(hyps)
        analysis.acc_epochs = AccEpochs(
            epoch_start_s=grid,
            counts=np.abs(np.sin(np.arange(n))) * 10.0,
            coverage=np.ones(n),
            threshold=8.0,
        )
        analysis.override_epochs_n = {"heuristic": 2, "sleepecg": 2}
        from app.sleep.engines import EngineStatus

        analysis.engines = [
            EngineStatus("heuristic", "Built-in rules", True),
            EngineStatus("sleepecg", "SleepECG GRU", True),
            EngineStatus(
                "external-5class", "External deep net", False,
                "ECGLOG_SLEEP_EXTERNAL_DIR is not set",
            ),
        ]
        return analysis

    def test_sleep_section_rendered(self, rendered: dict) -> None:
        html = build_report_html(
            rendered["rec"],
            rendered["result"],
            rendered["hrv"],
            rendered["analysis"],
            None,
            [],
            rendered["meta"],
            sleep=self._sleep_analysis(),
        )
        assert "Hypnogram" in html
        assert "substitute for a sleep study" in html
        assert "Engine agreement" in html
        assert "heuristic vs sleepecg" in html
        assert "ECGLOG_SLEEP_EXTERNAL_DIR is not set" in html
        assert "Sleep efficiency" in html
        # Movement figure present (ACC provided).
        assert "wake-override threshold" in html or "Accelerometer activity" in html

    def test_no_sleep_section_without_sleep(self, rendered: dict) -> None:
        assert "Hypnogram" not in rendered["html"]
        assert "Sleep efficiency" not in rendered["html"]


class TestFigureDecimation:
    def test_minmax_decimation_preserves_envelope(self) -> None:
        import numpy as np

        from app.report.figures import _minmax_decimate

        t = np.arange(50_000, dtype=float)
        y = np.sin(t / 100.0)
        y[12_345] = 99.0  # a single extreme beat must survive decimation
        y[40_000] = -55.0
        t2, y2 = _minmax_decimate(t, y, 4000)
        assert len(t2) <= 4000
        assert 99.0 in y2
        assert -55.0 in y2
        assert np.all(np.diff(t2) >= 0)

    def test_short_series_untouched(self) -> None:
        import numpy as np

        from app.report.figures import _minmax_decimate

        t = np.arange(100, dtype=float)
        y = t * 2.0
        t2, y2 = _minmax_decimate(t, y, 4000)
        assert np.array_equal(t, t2)
        assert np.array_equal(y, y2)
