"""Activity counts and the movement wake-override."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from app.ingest.acc_loader import load_polar_acc_csv
from app.sleep.actigraphy import (
    activity_counts,
    apply_wake_override,
    wake_override_mask,
)
from app.sleep.stages import EPOCH_LEN_S, UNSCORED, Hypnogram, StageVocab
from tests.synth_util import make_acc_csv_bytes

START = dt.datetime(2026, 8, 14, 10, 0, tzinfo=dt.UTC)
NOW = dt.datetime(2026, 8, 15, 12, 0, tzinfo=dt.UTC)


def _acc(tmp_path: Path, **kwargs):
    path = tmp_path / "acc.csv"
    path.write_bytes(make_acc_csv_bytes(**kwargs))
    return load_polar_acc_csv(path, now=NOW)


def _hyp(stages) -> Hypnogram:
    stages = np.asarray(stages, dtype=np.int8)
    return Hypnogram(
        engine="test",
        engine_label="Test",
        vocab=StageVocab.WAKE_LIGHT_DEEP_REM,
        epoch_len_s=EPOCH_LEN_S,
        epoch_start_s=np.arange(len(stages)) * EPOCH_LEN_S,
        stages=stages,
        probabilities=np.ones((len(stages), 4)) / 4.0,
        accuracy_note="test",
    )


class TestActivityCounts:
    def test_still_night_low_counts_everywhere(self, tmp_path: Path) -> None:
        acc = _acc(tmp_path, duration_s=300.0, fs_hz=50.0, start=START)
        epochs = activity_counts(acc, START, n_epochs=10)
        assert len(epochs.counts) == 10
        assert not np.any(np.isnan(epochs.counts))
        # Gravity is filtered out: counts reflect only the ~2 mg noise floor.
        assert float(np.nanmax(epochs.counts)) < 200.0

    def test_bursts_land_in_programmed_epochs(self, tmp_path: Path) -> None:
        acc = _acc(
            tmp_path,
            duration_s=300.0,
            fs_hz=50.0,
            start=START,
            movement_bursts=[(60.0, 90.0, 300.0), (240.0, 270.0, 300.0)],
        )
        epochs = activity_counts(acc, START, n_epochs=10)
        moving = {2, 8}  # epochs [60,90) and [240,270)
        for k in range(10):
            if k in moving:
                assert epochs.counts[k] > epochs.threshold
            else:
                assert epochs.counts[k] <= epochs.threshold

    def test_offset_alignment_shifts_epochs(self, tmp_path: Path) -> None:
        # ACC starts 60 s after the ECG: a burst at ACC t=0..30 is ECG epoch 2.
        acc = _acc(
            tmp_path,
            duration_s=120.0,
            fs_hz=50.0,
            start=START + dt.timedelta(seconds=60),
            movement_bursts=[(0.0, 30.0, 300.0)],
        )
        epochs = activity_counts(acc, START, n_epochs=6)
        # Epochs 0-1 have no ACC samples at all -> NaN, not zero.
        assert np.isnan(epochs.counts[0]) and np.isnan(epochs.counts[1])
        assert epochs.counts[2] > epochs.threshold
        assert epochs.counts[3] <= epochs.threshold

    def test_partial_coverage_is_nan(self, tmp_path: Path) -> None:
        acc = _acc(tmp_path, duration_s=45.0, fs_hz=50.0, start=START)
        epochs = activity_counts(acc, START, n_epochs=2)
        assert not np.isnan(epochs.counts[0])
        assert epochs.coverage[1] == pytest.approx(0.5, abs=0.02)


class TestWakeOverride:
    def _epochs(self, counts, threshold=100.0):
        from app.sleep.actigraphy import AccEpochs

        counts = np.asarray(counts, dtype=float)
        return AccEpochs(
            epoch_start_s=np.arange(len(counts)) * EPOCH_LEN_S,
            counts=counts,
            coverage=np.ones(len(counts)),
            threshold=threshold,
        )

    def test_single_epoch_burst_not_enough(self) -> None:
        mask = wake_override_mask(self._epochs([0, 500, 0, 0]))
        assert mask.tolist() == [False, False, False, False]

    def test_two_consecutive_epochs_fire(self) -> None:
        mask = wake_override_mask(self._epochs([0, 500, 500, 0]))
        assert mask.tolist() == [False, True, True, False]

    def test_run_at_end_fires(self) -> None:
        mask = wake_override_mask(self._epochs([0, 0, 500, 500]))
        assert mask.tolist() == [False, False, True, True]

    def test_nan_counts_never_fire(self) -> None:
        mask = wake_override_mask(self._epochs([np.nan, np.nan, 500, 500]))
        assert mask.tolist() == [False, False, True, True]

    def test_nan_threshold_disables_override(self) -> None:
        mask = wake_override_mask(self._epochs([500, 500], threshold=np.nan))
        assert mask.tolist() == [False, False]

    def test_apply_override_counts_and_preserves_unscored(self) -> None:
        hyp = _hyp([1, 2, UNSCORED, 0])
        mask = np.array([True, True, True, True])
        out, n = apply_wake_override(hyp, mask)
        assert n == 2  # the wake epoch and the unscored epoch are untouched
        assert out.stages.tolist() == [0, 0, UNSCORED, 0]
        assert out.probabilities is None
        assert hyp.stages.tolist() == [1, 2, UNSCORED, 0]  # original unchanged

    def test_apply_override_noop_returns_same(self) -> None:
        hyp = _hyp([0, 0])
        out, n = apply_wake_override(hyp, np.array([True, False]))
        assert n == 0
        assert out is hyp

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="epochs"):
            apply_wake_override(_hyp([0, 0]), np.array([True]))
