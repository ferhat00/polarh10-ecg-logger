"""Sleep-architecture math pinned to hand-computed values."""

from __future__ import annotations

import numpy as np

from app.pipeline.rr import RRSeries
from app.sleep.stages import EPOCH_LEN_S, UNSCORED, Hypnogram, StageVocab
from app.sleep.summary import stage_stats, summarize


def _hyp(stages, vocab=StageVocab.WAKE_LIGHT_DEEP_REM) -> Hypnogram:
    stages = np.asarray(stages, dtype=np.int8)
    return Hypnogram(
        engine="test",
        engine_label="Test engine",
        vocab=vocab,
        epoch_len_s=EPOCH_LEN_S,
        epoch_start_s=np.arange(len(stages)) * EPOCH_LEN_S,
        stages=stages,
        probabilities=None,
        accuracy_note="test",
    )


class TestSummaryHandBuilt:
    def test_textbook_night(self) -> None:
        # W W L L D D R R W L W W   (12 epochs = 6 min TIB)
        s = summarize(_hyp([0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 0, 0]))
        assert s.tib_min == 6.0
        assert s.tst_min == 3.5  # 7 sleep epochs
        assert s.sleep_efficiency_pct == 3.5 / 6.0 * 100.0
        assert s.sol_min == 1.0  # onset at epoch 2
        # Between onset (2) and last sleep epoch (9): one wake epoch (8).
        assert s.waso_min == 0.5
        assert s.awakenings_n == 1
        assert s.rem_latency_min == (6 - 2) * 0.5  # first REM at epoch 6
        assert s.light_min == 1.5
        assert s.deep_min == 1.0
        assert s.rem_min == 1.0
        assert s.wake_min == 2.5
        assert s.unscored_min == 0.0
        # Percentages of TST.
        assert abs(s.stage_pct_of_tst["Light"] - (1.5 / 3.5 * 100)) < 1e-9
        assert abs(s.stage_pct_of_tst["Deep"] - (1.0 / 3.5 * 100)) < 1e-9

    def test_two_awakenings_counted_as_runs(self) -> None:
        # onset epoch1; wake runs: epochs 3-4 (one run) and epoch 6 (one run).
        s = summarize(_hyp([0, 1, 1, 0, 0, 1, 0, 1]))
        assert s.awakenings_n == 2
        assert s.waso_min == 1.5  # three wake epochs inside the sleep span

    def test_all_wake(self) -> None:
        s = summarize(_hyp([0, 0, 0, 0]))
        assert s.tst_min == 0.0
        assert s.sleep_efficiency_pct == 0.0
        assert s.sol_min is None
        assert s.waso_min is None
        assert s.awakenings_n is None
        assert s.rem_latency_min is None
        assert any("No epoch was scored as sleep" in n for n in s.notes)

    def test_no_rem_night(self) -> None:
        s = summarize(_hyp([0, 1, 2, 1]))
        assert s.rem_latency_min is None
        assert s.rem_min == 0.0

    def test_terminal_wake_is_not_waso(self) -> None:
        s = summarize(_hyp([1, 1, 0, 0, 0]))
        assert s.waso_min == 0.0
        assert s.awakenings_n == 0

    def test_unscored_counts_toward_tib_only(self) -> None:
        s = summarize(_hyp([UNSCORED, 1, 1, UNSCORED]))
        assert s.tib_min == 2.0
        assert s.tst_min == 1.0
        assert s.unscored_min == 1.0
        # Unscored epochs are not wake: no awakening from them.
        assert s.awakenings_n == 0

    def test_two_class_vocab_reports_only_what_it_knows(self) -> None:
        s = summarize(_hyp([0, 1, 1, 0], vocab=StageVocab.WAKE_SLEEP))
        assert s.sleep_min == 1.0
        assert s.light_min is None
        assert s.deep_min is None
        assert s.rem_min is None
        assert s.rem_latency_min is None

    def test_aasm5_n3_lands_in_deep_column(self) -> None:
        s = summarize(_hyp([0, 1, 2, 3, 4], vocab=StageVocab.AASM_5))
        assert s.n1_min == 0.5
        assert s.n2_min == 0.5
        assert s.deep_min == 0.5  # N3 is deep sleep
        assert s.rem_min == 0.5


class TestStageStats:
    def _rr_alternating(self) -> RRSeries:
        """Two epochs: 1000 ms beats in epoch 0, 600 ms beats in epoch 1."""
        t0 = np.arange(1.0, 30.0, 1.0)  # ends in epoch 0
        t1 = np.arange(30.6, 59.9, 0.6)  # ends in epoch 1
        t = np.concatenate([t0, t1])
        rr_ms = np.concatenate([np.full(len(t0), 1000.0), np.full(len(t1), 600.0)])
        return RRSeries(
            rr_ms=rr_ms,
            t_s=t,
            discontinuity=np.zeros(len(t), dtype=bool),
            n_dropped_excluded=0,
            n_dropped_ceiling=0,
            n_dropped_floor=0,
        )

    def test_per_stage_hr(self) -> None:
        hyp = _hyp([0, 2])  # epoch 0 wake, epoch 1 deep
        rows = {r["stage"]: r for r in stage_stats(hyp, self._rr_alternating())}
        assert abs(rows["Wake"]["mean_hr_bpm"] - 60.0) < 1e-6
        assert abs(rows["Deep"]["mean_hr_bpm"] - 100.0) < 1e-6
        assert rows["Light"]["mean_hr_bpm"] is None
        assert rows["Wake"]["minutes"] == 0.5

    def test_constant_rr_gives_zero_rmssd(self) -> None:
        hyp = _hyp([0, 2])
        rows = {r["stage"]: r for r in stage_stats(hyp, self._rr_alternating())}
        # Within each stage the series is constant -> RMSSD 0; the cross-stage
        # 1000->600 jump must NOT leak into either stage's RMSSD.
        assert rows["Wake"]["mean_rmssd_ms"] == 0.0
        assert rows["Deep"]["mean_rmssd_ms"] == 0.0

    def test_empty_inputs(self) -> None:
        empty = RRSeries(
            np.array([]), np.array([]), np.array([], dtype=bool), 0, 0, 0
        )
        assert stage_stats(_hyp([0, 1]), empty) == []
