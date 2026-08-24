"""Notebook helper tests: the two shortcuts that could silently diverge.

``notebooks/sleep_eda_core`` takes two shortcuts the app itself does not:

1. It reads a recording's start time and duration from the CSV's first and
   last rows instead of loading the file. If that ever disagreed with
   :func:`~app.ingest.loader.load_polar_csv`, the notebook would place a whole
   night at the wrong clock time — and the sleepecg classifier's circadian
   features would be fed a lie.
2. It rebuilds a :class:`~app.pipeline.quality.QualityResult` from the
   per-window arrays in the ``.npz`` cache. Staging reads only ``start_s`` and
   ``wander_rms_mv`` from it, so the reconstruction must preserve those
   exactly and must produce the same epoch features as the real object.

Both are pinned here against the committed synthetic fixture.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pytest

from app.ingest.loader import load_polar_csv
from app.pipeline.process import run_pipeline
from app.pipeline.quality import QualityWindow
from app.sleep.epochs import compute_epoch_features

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "notebooks"))

import sleep_eda_core as core  # noqa: E402

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "synthetic_supine.csv"
NOW = dt.datetime(2026, 8, 21, 12, 0, tzinfo=dt.UTC)


@pytest.fixture(scope="module")
def loaded():
    return load_polar_csv(FIXTURE, now=NOW)


class TestRecordingSpan:
    """The cheap span read must agree with the full loader."""

    def test_start_time_matches_loader(self, loaded) -> None:
        span = core.recording_span(FIXTURE, now=NOW)
        assert span.start_time == loaded.start_time

    def test_duration_matches_loader(self, loaded) -> None:
        span = core.recording_span(FIXTURE, now=NOW)
        # Both derive duration from the same first/last timestamps, so this is
        # an equality check, not an approximation.
        assert span.duration_s == pytest.approx(loaded.duration_s, abs=1e-6)

    def test_epoch_matches_loader(self, loaded) -> None:
        span = core.recording_span(FIXTURE, now=NOW)
        assert span.epoch_detected == loaded.epoch_detected

    def test_sampling_rate_estimate_is_close(self, loaded) -> None:
        span = core.recording_span(FIXTURE, now=NOW)
        assert span.sampling_rate_hz_estimate == pytest.approx(
            loaded.sampling_rate_hz, rel=0.02
        )

    def test_estimate_is_labelled_as_an_estimate(self) -> None:
        span = core.recording_span(FIXTURE, now=NOW)
        assert any("estimated" in note for note in span.notes)


class TestQualityFromCache:
    """The rebuilt QualityResult must drive staging identically."""

    @staticmethod
    def _cache_arrays(quality) -> dict:
        return {
            "window_start_s": np.array([w.start_s for w in quality.windows]),
            "window_sqi": np.array([w.sqi_mean for w in quality.windows]),
            "window_wander_mv": np.array([w.wander_rms_mv for w in quality.windows]),
            "window_excluded": np.array([w.excluded for w in quality.windows]),
        }

    def test_preserves_the_fields_staging_reads(self, loaded) -> None:
        result = run_pipeline(loaded)
        rebuilt = core._quality_from_cache(self._cache_arrays(result.quality))

        assert len(rebuilt.windows) == len(result.quality.windows)
        for got, want in zip(rebuilt.windows, result.quality.windows, strict=True):
            assert got.start_s == want.start_s
            assert got.wander_rms_mv == want.wander_rms_mv
            assert got.excluded == want.excluded

    def test_produces_identical_epoch_features(self, loaded) -> None:
        result = run_pipeline(loaded)
        rebuilt = core._quality_from_cache(self._cache_arrays(result.quality))

        real = compute_epoch_features(result.rr, result.quality, loaded.duration_s, None)
        from_cache = compute_epoch_features(result.rr, rebuilt, loaded.duration_s, None)

        np.testing.assert_array_equal(real.movement, from_cache.movement)
        assert real.movement_threshold == from_cache.movement_threshold
        assert real.movement_source == from_cache.movement_source

    def test_excluded_windows_carry_an_honest_reason(self) -> None:
        arrays = {
            "window_start_s": np.array([0.0, 5.0]),
            "window_sqi": np.array([0.9, 0.1]),
            "window_wander_mv": np.array([0.01, 0.4]),
            "window_excluded": np.array([False, True]),
        }
        rebuilt = core._quality_from_cache(arrays)
        excluded = [w for w in rebuilt.windows if w.excluded]
        assert len(excluded) == 1
        # The real reason is not cached; the placeholder must say so rather
        # than leave a blank that reads as "excluded for no reason".
        assert excluded[0].reasons and "not cached" in excluded[0].reasons[0]
        assert rebuilt.excluded_total_s == pytest.approx(5.0)

    def test_per_sample_traces_are_empty_not_faked(self) -> None:
        arrays = {
            "window_start_s": np.array([0.0]),
            "window_sqi": np.array([0.9]),
            "window_wander_mv": np.array([0.01]),
            "window_excluded": np.array([False]),
        }
        rebuilt = core._quality_from_cache(arrays)
        assert len(rebuilt.sqi) == 0
        assert len(rebuilt.wander_mv) == 0


class TestStagingGuard:
    """A recording too short to stage is refused, not guessed at."""

    def test_short_recording_produces_no_hypnogram(self) -> None:
        data = core.load_session(FIXTURE, prefer_cache=False)
        analysis, _ = core.stage_night(data)
        assert analysis.hypnograms == []
        assert any("minimum for sleep staging" in note for note in analysis.notes)

    def test_features_are_still_returned(self) -> None:
        """Emptiness lives on ``hypnograms``, never on ``features``.

        Both branches must agree on this: a caller that tested ``features is
        None`` would otherwise behave differently depending on whether a cache
        happened to exist.
        """
        data = core.load_session(FIXTURE, prefer_cache=False)
        _, features = core.stage_night(data)
        assert features is not None
        assert features.n_epochs == data.n_epochs


class TestSessionData:
    def test_full_branch_keeps_the_real_objects(self) -> None:
        data = core.load_session(FIXTURE, prefer_cache=False)
        assert data.source == "pipeline"
        assert data.recording is not None
        assert data.result is not None
        assert isinstance(data.quality.windows[0], QualityWindow)
        assert data.n_epochs == int(np.ceil(data.duration_s / 30.0))
