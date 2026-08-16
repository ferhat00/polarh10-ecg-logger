"""Golden-file test: the full pipeline over the committed synthetic fixture.

The fixture (tests/fixtures/synthetic_supine.csv, built by
scripts/make_fixture.py) encodes a *known* RR series — 800 ms with 0.1 Hz
±30 ms modulation and σ=10 ms jitter — so results are checked two ways:

1. pinned golden numbers to two decimals (regression detection), and
2. sanity bounds against the analytically known series (correctness).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from app.ingest.loader import load_polar_csv
from app.pipeline import engines
from app.pipeline.engines import detect_rpeaks
from app.pipeline.process import PipelineResult, run_pipeline

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_supine.csv"
NOW = dt.datetime(2026, 8, 16, 12, 0, tzinfo=dt.UTC)

# Ground truth of the generator (scripts/make_fixture.py, seed 42).
TRUE_MEAN_RR_MS = 799.53
TRUE_SDNN_MS = 23.50
TRUE_RMSSD_MS = 16.74


@pytest.fixture(scope="module")
def result() -> PipelineResult:
    rec = load_polar_csv(FIXTURE, now=NOW)
    return run_pipeline(rec)


class TestGoldenNumbers:
    """Pinned to two decimals — any drift here is a pipeline change."""

    def test_headline_metrics(self, result: PipelineResult) -> None:
        h = result.hrv
        assert h.n_beats == 376
        assert h.mean_rr_ms == pytest.approx(799.53, abs=0.005)
        assert h.mean_hr_bpm == pytest.approx(75.04, abs=0.005)
        assert h.sdnn_ms == pytest.approx(23.78, abs=0.005)
        assert h.rmssd_ms == pytest.approx(17.74, abs=0.005)
        assert h.pnn50_pct == pytest.approx(0.53, abs=0.005)
        assert h.sd1_ms == pytest.approx(12.56, abs=0.005)
        assert h.sd2_ms == pytest.approx(31.20, abs=0.005)

    def test_frequency_domain(self, result: PipelineResult) -> None:
        h = result.hrv
        assert h.lf_power_ms2 == pytest.approx(520.18, abs=0.01)
        assert h.hf_power_ms2 == pytest.approx(47.57, abs=0.01)
        assert h.lf_hf_ratio == pytest.approx(10.93, abs=0.005)
        assert h.lf_peak_hz == pytest.approx(0.10, abs=0.005)

    def test_nonlinear(self, result: PipelineResult) -> None:
        h = result.hrv
        assert h.sample_entropy == pytest.approx(2.07, abs=0.005)
        assert h.dfa_alpha1 == pytest.approx(1.34, abs=0.005)

    def test_quality_and_correction(self, result: PipelineResult) -> None:
        assert result.quality.excluded_total_s == 0.0
        assert result.quality.excluded_segments == []
        assert result.correction.pct_corrected == pytest.approx(0.27, abs=0.01)
        assert result.correction.reduced_confidence is False
        assert result.correction.counts["ectopic"] == 0

    def test_engine_provenance(self, result: PipelineResult) -> None:
        assert result.engine_used == "neurokit2"
        assert result.fallback_reason is None


class TestAgainstKnownSeries:
    """Correctness against ground truth, at detection-quantisation tolerance."""

    def test_mean_rr_matches_truth(self, result: PipelineResult) -> None:
        assert result.hrv.mean_rr_ms == pytest.approx(TRUE_MEAN_RR_MS, abs=1.0)

    def test_sdnn_close_to_truth(self, result: PipelineResult) -> None:
        # R-peak time quantisation (7.7 ms samples) adds a little variance.
        assert result.hrv.sdnn_ms == pytest.approx(TRUE_SDNN_MS, abs=1.5)

    def test_rmssd_close_to_truth(self, result: PipelineResult) -> None:
        assert result.hrv.rmssd_ms == pytest.approx(TRUE_RMSSD_MS, abs=2.0)

    def test_lf_peak_at_modulation_frequency(self, result: PipelineResult) -> None:
        assert result.hrv.lf_peak_hz == pytest.approx(0.1, abs=0.01)

    def test_device_rr_agreement(self, result: PipelineResult) -> None:
        # Our detection vs the device's own rr stream: within quantisation.
        assert result.device_rr_median_abs_diff_ms is not None
        assert result.device_rr_median_abs_diff_ms < 8.0

    def test_no_false_ectopy_on_clean_signal(self, result: PipelineResult) -> None:
        assert int(np.sum(result.morphology.ectopy_candidate)) == 0

    def test_per_window_sdnn_present(self, result: PipelineResult) -> None:
        assert len(result.hrv.sdnn_per_window_ms) == 5  # five whole minutes
        assert all(15.0 < v < 30.0 for v in result.hrv.sdnn_per_window_ms)


class TestEngineFallback:
    """The fallback chain is honest: engine_used reports what actually ran."""

    def test_biosppy_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated neurokit2 failure")

        monkeypatch.setattr(engines, "_neurokit_detect", boom)
        rec = load_polar_csv(FIXTURE, now=NOW)
        detection = detect_rpeaks(rec.ecg_mv, rec.sampling_rate_hz)
        assert detection.engine == "biosppy"
        assert "simulated neurokit2 failure" in (detection.fallback_reason or "")
        # biosppy finds essentially the same beats.
        assert abs(len(detection.rpeak_indices) - 377) <= 5

    def test_both_engines_failing_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("down")

        monkeypatch.setattr(engines, "_neurokit_detect", boom)
        monkeypatch.setattr(engines, "_biosppy_detect", boom)
        with pytest.raises(engines.PipelineError, match="Both engines"):
            detect_rpeaks(np.zeros(1000), 130.03)
