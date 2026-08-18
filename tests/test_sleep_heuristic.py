"""Heuristic staging engine on stage-programmed synthetic nights.

The agreement bars in here are synthetic-only sanity checks — the generator
produces textbook autonomic signatures with no pathology, so recovery of
its macro-structure says nothing about agreement with polysomnography
(which the accuracy note honestly bounds at ~65-75%).
"""

from __future__ import annotations

import numpy as np

from app.pipeline.quality import QualityResult
from app.sleep.engines.heuristic import stage_heuristic
from app.sleep.epochs import compute_epoch_features
from app.sleep.stages import UNSCORED, StageVocab
from tests.synth_util import overnight_rr, rr_series_from


def _quality(duration_s: float) -> QualityResult:
    # No quality windows: movement falls back to the (empty) wander proxy —
    # these tests exercise the cardiac features, not movement.
    return QualityResult(
        windows=[],
        excluded_segments=[],
        sqi=np.array([]),
        wander_mv=np.array([]),
        excluded_total_s=0.0,
        analysed_total_s=duration_s,
    )


def _stage(stage_plan=None, seed=11):
    rr_ms, t_s, truth = overnight_rr(stage_plan=stage_plan, seed=seed)
    rr = rr_series_from(rr_ms, t_s)
    duration = float(t_s[-1])
    features = compute_epoch_features(rr, _quality(duration), duration, None)
    hyp = stage_heuristic(features)
    n = min(len(truth), hyp.n_epochs)
    return hyp, truth[:n], hyp.stages[:n]


class TestMacroStructureRecovery:
    def test_overall_agreement_on_default_night(self) -> None:
        hyp, truth, staged = _stage()
        scored = staged != UNSCORED
        agreement = float(np.mean(staged[scored] == truth[scored]))
        # Generous bar: macro-structure recovery on a textbook synthetic.
        assert agreement >= 0.70, f"synthetic agreement only {agreement:.2f}"
        assert hyp.vocab == StageVocab.WAKE_LIGHT_DEEP_REM

    def test_deep_block_found_where_programmed(self) -> None:
        _hyp, truth, staged = _stage()
        deep_truth = truth == 2
        # Most of the programmed deep epochs stage as deep.
        assert np.mean(staged[deep_truth] == 2) > 0.6
        # And deep is not hallucinated during programmed REM.
        rem_truth = truth == 3
        assert np.mean(staged[rem_truth] == 2) < 0.1

    def test_rem_block_found_where_programmed(self) -> None:
        _hyp, truth, staged = _stage()
        rem_truth = truth == 3
        assert np.mean(staged[rem_truth] == 3) > 0.6

    def test_determinism(self) -> None:
        _h1, _t1, s1 = _stage(seed=11)
        _h2, _t2, s2 = _stage(seed=11)
        assert np.array_equal(s1, s2)

    def test_movement_forces_wake(self) -> None:
        # Same night, but with movement injected into a mid-night span via
        # the features' movement channel (as an ACC file would provide).
        rr_ms, t_s, truth = overnight_rr(seed=11)
        rr = rr_series_from(rr_ms, t_s)
        duration = float(t_s[-1])
        features = compute_epoch_features(rr, _quality(duration), duration, None)
        features.movement = np.zeros(features.n_epochs)
        features.movement[100:106] = 10.0
        object.__setattr__(features, "movement_threshold", 5.0)
        hyp = stage_heuristic(features)
        assert np.all(hyp.stages[101:105] == 0)


class TestSmoothing:
    def test_single_epoch_flips_are_removed(self) -> None:
        # A long light block: without smoothing, per-epoch noise flips a few
        # epochs; the Viterbi pass must return one contiguous stage.
        hyp, truth, staged = _stage(stage_plan=[("light", 45)], seed=3)
        scored = staged[staged != UNSCORED]
        # No isolated single-epoch stage islands.
        changes = np.flatnonzero(np.diff(scored) != 0)
        if len(changes) >= 2:
            run_lengths = np.diff(changes)
            assert np.min(run_lengths) >= 2

    def test_unscored_epochs_survive_smoothing(self) -> None:
        rr_ms, t_s, _truth = overnight_rr(stage_plan=[("light", 20)], seed=4)
        rr = rr_series_from(rr_ms, t_s)
        duration = float(t_s[-1])
        features = compute_epoch_features(rr, _quality(duration), duration, None)
        features.coverage[10:12] = 0.0  # simulate an excluded stretch
        hyp = stage_heuristic(features)
        assert np.all(hyp.stages[10:12] == UNSCORED)
        assert hyp.stages[9] != UNSCORED
        assert hyp.stages[12] != UNSCORED
