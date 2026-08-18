"""Per-epoch feature extraction pinned on constructed RR series."""

from __future__ import annotations

import numpy as np

from app.pipeline.quality import QualityResult, QualityWindow
from app.sleep.epochs import MIN_EPOCH_COVERAGE, compute_epoch_features
from tests.synth_util import overnight_rr, rr_series_from


def _quality(duration_s: float, wander: float = 0.01) -> QualityResult:
    windows = [
        QualityWindow(
            start_s=float(w * 5.0),
            end_s=float(w * 5.0 + 5.0),
            sqi_mean=0.99,
            wander_rms_mv=wander,
            excluded=False,
        )
        for w in range(int(duration_s // 5))
    ]
    return QualityResult(
        windows=windows,
        excluded_segments=[],
        sqi=np.array([]),
        wander_mv=np.array([]),
        excluded_total_s=0.0,
        analysed_total_s=duration_s,
    )


class TestFeatureGrid:
    def test_constant_rr_exact_means(self) -> None:
        # 1000 ms beats for 5 min: HR exactly 60 everywhere, full coverage.
        t = np.arange(1.0, 300.0, 1.0)
        rr = rr_series_from(np.full(len(t), 1000.0), t)
        f = compute_epoch_features(rr, _quality(300.0), 300.0, None)
        assert f.n_epochs == 10
        assert np.allclose(f.mean_hr_bpm, 60.0)
        assert np.all(f.coverage > 0.9)
        assert np.allclose(f.rmssd_ms[~np.isnan(f.rmssd_ms)], 0.0)
        assert np.allclose(f.hr_vs_baseline, 0.0)

    def test_empty_rr_yields_unstageable(self) -> None:
        rr = rr_series_from(np.array([]), np.array([]))
        f = compute_epoch_features(rr, _quality(60.0), 60.0, None)
        assert f.n_epochs == 2
        assert not np.any(f.stageable())

    def test_gap_epoch_has_low_coverage(self) -> None:
        # Beats in epochs 0 and 2, nothing in epoch 1.
        t = np.concatenate([np.arange(1.0, 30.0, 1.0), np.arange(61.0, 90.0, 1.0)])
        rr_ms = np.full(len(t), 1000.0)
        rr = rr_series_from(rr_ms, t)
        rr.discontinuity[len(np.arange(1.0, 30.0, 1.0))] = True
        f = compute_epoch_features(rr, _quality(90.0), 90.0, None)
        assert f.coverage[1] < MIN_EPOCH_COVERAGE
        assert f.stageable().tolist() == [True, False, True]

    def test_hr_vs_baseline_tracks_sustained_minimum(self) -> None:
        # First half 1200 ms (50 bpm), second half 1000 ms (60 bpm).
        t1 = np.arange(1.2, 299.0, 1.2)
        t2 = np.arange(300.0, 600.0, 1.0)
        rr = rr_series_from(
            np.concatenate([np.full(len(t1), 1200.0), np.full(len(t2), 1000.0)]),
            np.concatenate([t1, t2]),
        )
        f = compute_epoch_features(rr, _quality(600.0), 600.0, None)
        # Baseline is the 10th-percentile epoch HR ≈ 50 bpm.
        assert abs(np.nanmin(f.hr_vs_baseline)) < 1.0
        assert np.nanmax(f.hr_vs_baseline) > 8.0

    def test_lf_hf_separates_programmed_bands(self) -> None:
        # Two long blocks: HF-dominant (deep-like) then LF-dominant (REM-like).
        rr_ms, t_s, _truth = overnight_rr(
            stage_plan=[("deep", 10), ("rem", 10)], seed=5
        )
        rr = rr_series_from(rr_ms, t_s)
        f = compute_epoch_features(rr, _quality(t_s[-1]), t_s[-1], None)
        deep_mid = f.lf_hf[6:12]  # well inside the deep block
        rem_mid = f.lf_hf[26:32]  # well inside the rem block
        assert np.nanmedian(deep_mid) < 0.5
        assert np.nanmedian(rem_mid) > 3.0

    def test_wander_proxy_used_without_acc(self) -> None:
        t = np.arange(1.0, 120.0, 1.0)
        rr = rr_series_from(np.full(len(t), 1000.0), t)
        f = compute_epoch_features(rr, _quality(120.0, wander=0.02), 120.0, None)
        assert f.movement_source == "wander_proxy"
        assert np.all(np.isfinite(f.movement))
        assert any("movement" in n.lower() for n in f.notes)
