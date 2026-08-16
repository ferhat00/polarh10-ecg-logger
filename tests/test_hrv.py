"""HRV-metric tests against analytically known RR series."""

from __future__ import annotations

import numpy as np
import pytest

from app.pipeline.hrv import compute_hrv
from app.pipeline.rr import RRSeries


def _series(rr_ms: np.ndarray, discontinuity_at: list[int] | None = None) -> RRSeries:
    t = np.cumsum(rr_ms) / 1000.0
    disc = np.zeros(len(rr_ms), dtype=bool)
    for i in discontinuity_at or []:
        disc[i] = True
    return RRSeries(
        rr_ms=rr_ms,
        t_s=t,
        discontinuity=disc,
        n_dropped_excluded=0,
        n_dropped_ceiling=0,
        n_dropped_floor=0,
    )


class TestTimeDomain:
    def test_alternating_series_analytic(self) -> None:
        # 800 ± 20 alternating: SDNN ≈ 20, RMSSD = 40 exactly, pNN50 = 0.
        rr = np.array([800.0 + (20.0 if i % 2 else -20.0) for i in range(300)])
        h = compute_hrv(_series(rr))
        assert h.mean_rr_ms == pytest.approx(800.0, abs=0.01)
        assert h.mean_hr_bpm == pytest.approx(75.0, abs=0.01)
        assert h.sdnn_ms == pytest.approx(20.0, abs=0.1)
        assert h.rmssd_ms == pytest.approx(40.0, abs=0.01)
        assert h.pnn50_pct == 0.0

    def test_pnn50_counts_only_diffs_over_50(self) -> None:
        rr = np.array([800.0 + (30.0 if i % 2 else -30.0) for i in range(300)])
        h = compute_hrv(_series(rr))  # diffs are 60 ms > 50 ms
        assert h.pnn50_pct == pytest.approx(100.0, abs=0.5)

    def test_per_window_sdnn_separates_trend_from_variability(self) -> None:
        # A pure linear trend: whole-record SDNN large, per-window SDNN small.
        n = 600
        rr = 700.0 + np.linspace(0.0, 200.0, n)  # slow drift, no jitter
        h = compute_hrv(_series(rr))
        assert h.sdnn_ms > 50.0
        assert h.sdnn_per_window_ms, "per-window SDNN must always be computed"
        assert max(h.sdnn_per_window_ms) < 10.0
        assert any("trend-inclusive" in note for note in h.notes)

    def test_rmssd_skips_discontinuities(self) -> None:
        # Two flat runs at different levels; the 200 ms level jump happens at
        # a discontinuity, so RMSSD must be 0, not polluted by the seam.
        rr = np.array([800.0] * 150 + [1000.0] * 150)
        h_with = compute_hrv(_series(rr, discontinuity_at=[150]))
        assert h_with.rmssd_ms == pytest.approx(0.0, abs=1e-9)
        h_without = compute_hrv(_series(rr))
        assert h_without.rmssd_ms > 5.0


class TestFrequencyDomain:
    def test_lf_modulation_lands_in_lf_band(self) -> None:
        # 0.1 Hz ±30 ms modulation: LF power ≈ 30²/2 = 450 ms², peak ≈ 0.1 Hz.
        t = np.cumsum(np.full(400, 800.0)) / 1000.0
        rr = 800.0 + 30.0 * np.sin(2 * np.pi * 0.1 * t)
        h = compute_hrv(_series(rr))
        assert h.lf_peak_hz == pytest.approx(0.1, abs=0.015)
        assert h.lf_power_ms2 == pytest.approx(450.0, rel=0.3)
        assert h.hf_power_ms2 < h.lf_power_ms2 / 5
        assert h.psd_method is not None
        assert any("Mayer wave" in note for note in h.notes)

    def test_short_record_has_no_frequency_metrics(self) -> None:
        rr = np.full(60, 800.0)  # 48 s
        h = compute_hrv(_series(rr))
        assert h.lf_power_ms2 is None
        assert h.lf_hf_ratio is None
        assert any("frequency-domain" in note.lower() for note in h.notes)

    def test_spectrum_never_bridges_discontinuities(self) -> None:
        # Two 100 s runs (each below the 120 s spectral minimum) separated by
        # a discontinuity: no frequency metrics may be produced.
        rr = np.full(250, 800.0)  # 200 s total
        h = compute_hrv(_series(rr, discontinuity_at=[125]))
        assert h.lf_power_ms2 is None


class TestNonlinear:
    def test_poincare_identities(self) -> None:
        rng = np.random.default_rng(3)
        rr = 800.0 + rng.normal(0.0, 25.0, 500)
        h = compute_hrv(_series(rr))
        diffs = np.diff(rr)
        sd1_expected = np.sqrt(0.5 * np.var(diffs, ddof=1))
        assert h.sd1_ms == pytest.approx(sd1_expected, rel=0.01)
        assert h.sd2_ms is not None and h.sd2_ms > 0
        assert h.sd1_sd2_ratio == pytest.approx(h.sd1_ms / h.sd2_ms, rel=1e-6)
        # For white noise SD1 ≈ SD2.
        assert h.sd1_sd2_ratio == pytest.approx(1.0, abs=0.15)

    def test_nonlinear_skipped_when_too_few_beats(self) -> None:
        rr = np.full(50, 800.0)
        h = compute_hrv(_series(rr))
        assert h.sample_entropy is None
        assert h.dfa_alpha1 is None


class TestDegenerate:
    def test_tiny_series_reports_not_computed(self) -> None:
        h = compute_hrv(_series(np.full(5, 800.0)))
        assert h.mean_rr_ms is None
        assert any("not computed" in note for note in h.notes)
