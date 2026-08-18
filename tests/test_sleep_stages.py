"""Stage-vocabulary tests: every collapse table exact, no invented detail."""

from __future__ import annotations

import numpy as np
import pytest

from app.sleep.stages import (
    EPOCH_LEN_S,
    REM_CODE,
    UNSCORED,
    WAKE_CODE,
    Hypnogram,
    StageVocab,
    collapse,
    common_vocab,
    epoch_indices,
    make_epoch_grid,
    n_classes,
    stage_labels,
)


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
        accuracy_note="test note",
    )


class TestVocabInvariants:
    def test_wake_is_zero_everywhere(self) -> None:
        for vocab in StageVocab:
            assert stage_labels(vocab)[WAKE_CODE] == "Wake"

    def test_labels_match_class_counts(self) -> None:
        assert n_classes(StageVocab.WAKE_SLEEP) == 2
        assert n_classes(StageVocab.WAKE_REM_NREM) == 3
        assert n_classes(StageVocab.WAKE_LIGHT_DEEP_REM) == 4
        assert n_classes(StageVocab.AASM_5) == 5

    def test_rem_codes_point_at_rem_labels(self) -> None:
        for vocab, code in REM_CODE.items():
            if code is not None:
                assert stage_labels(vocab)[code] == "REM"


class TestCollapse:
    def test_aasm5_to_wldr(self) -> None:
        # W N1 N2 N3 REM -> W Light Light Deep REM
        out = collapse(
            np.array([0, 1, 2, 3, 4]), StageVocab.AASM_5, StageVocab.WAKE_LIGHT_DEEP_REM
        )
        assert out.tolist() == [0, 1, 1, 2, 3]

    def test_aasm5_to_wrn(self) -> None:
        out = collapse(
            np.array([0, 1, 2, 3, 4]), StageVocab.AASM_5, StageVocab.WAKE_REM_NREM
        )
        assert out.tolist() == [0, 1, 1, 1, 2]

    def test_aasm5_to_wake_sleep(self) -> None:
        out = collapse(
            np.array([0, 1, 2, 3, 4]), StageVocab.AASM_5, StageVocab.WAKE_SLEEP
        )
        assert out.tolist() == [0, 1, 1, 1, 1]

    def test_wldr_to_wrn(self) -> None:
        # W Light Deep REM -> W NREM NREM REM
        out = collapse(
            np.array([0, 1, 2, 3]),
            StageVocab.WAKE_LIGHT_DEEP_REM,
            StageVocab.WAKE_REM_NREM,
        )
        assert out.tolist() == [0, 1, 1, 2]

    def test_wldr_to_wake_sleep(self) -> None:
        out = collapse(
            np.array([0, 1, 2, 3]),
            StageVocab.WAKE_LIGHT_DEEP_REM,
            StageVocab.WAKE_SLEEP,
        )
        assert out.tolist() == [0, 1, 1, 1]

    def test_wrn_to_wake_sleep(self) -> None:
        out = collapse(
            np.array([0, 1, 2]), StageVocab.WAKE_REM_NREM, StageVocab.WAKE_SLEEP
        )
        assert out.tolist() == [0, 1, 1]

    def test_unscored_passes_through_every_table(self) -> None:
        for (src, dst) in [
            (StageVocab.AASM_5, StageVocab.WAKE_LIGHT_DEEP_REM),
            (StageVocab.WAKE_LIGHT_DEEP_REM, StageVocab.WAKE_SLEEP),
            (StageVocab.WAKE_REM_NREM, StageVocab.WAKE_SLEEP),
        ]:
            out = collapse(np.array([UNSCORED, 0]), src, dst)
            assert out[0] == UNSCORED
            assert out[1] == 0

    def test_identity_returns_copy(self) -> None:
        stages = np.array([0, 1, 2], dtype=np.int8)
        out = collapse(stages, StageVocab.WAKE_REM_NREM, StageVocab.WAKE_REM_NREM)
        assert out.tolist() == stages.tolist()
        out[0] = 2
        assert stages[0] == 0  # the input was not mutated

    def test_toward_finer_raises(self) -> None:
        with pytest.raises(ValueError, match="never invented"):
            collapse(
                np.array([0, 1]), StageVocab.WAKE_SLEEP, StageVocab.AASM_5
            )

    def test_common_vocab_is_the_coarser(self) -> None:
        assert (
            common_vocab(StageVocab.AASM_5, StageVocab.WAKE_REM_NREM)
            == StageVocab.WAKE_REM_NREM
        )
        assert (
            common_vocab(StageVocab.WAKE_SLEEP, StageVocab.WAKE_LIGHT_DEEP_REM)
            == StageVocab.WAKE_SLEEP
        )


class TestEpochGrid:
    def test_grid_covers_partial_final_epoch(self) -> None:
        grid = make_epoch_grid(95.0)  # 3 full epochs + 5 s
        assert grid.tolist() == [0.0, 30.0, 60.0, 90.0]

    def test_exact_multiple_has_no_extra_epoch(self) -> None:
        assert make_epoch_grid(90.0).tolist() == [0.0, 30.0, 60.0]

    def test_zero_duration_empty(self) -> None:
        assert len(make_epoch_grid(0.0)) == 0

    def test_epoch_indices_boundaries(self) -> None:
        idx = epoch_indices(np.array([0.0, 29.99, 30.0, 59.99, 60.0]), n_epochs=2)
        # Epoch k covers [k*30, (k+1)*30); t=60 is outside a 2-epoch grid.
        assert idx.tolist() == [0, 0, 1, 1, -1]

    def test_epoch_indices_negative_times_excluded(self) -> None:
        idx = epoch_indices(np.array([-0.5, 5.0]), n_epochs=1)
        assert idx.tolist() == [-1, 0]


class TestHypnogram:
    def test_masks(self) -> None:
        hyp = _hyp([0, 1, 3, UNSCORED])
        assert hyp.sleep_mask().tolist() == [False, True, True, False]
        assert hyp.scored_mask().tolist() == [True, True, True, False]

    def test_collapsed_keeps_grid_and_note(self) -> None:
        hyp = _hyp([0, 1, 2, 3])
        out = hyp.collapsed(StageVocab.WAKE_REM_NREM)
        assert out.stages.tolist() == [0, 1, 1, 2]
        assert out.epoch_start_s.tolist() == hyp.epoch_start_s.tolist()
        assert out.accuracy_note == hyp.accuracy_note
        assert out.probabilities is None
