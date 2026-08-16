"""RR-construction tests — including the excluded-segment bridging bug.

Diffing R-peak times across an excluded window yields one spurious
multi-second "beat interval". These tests pin the fix: any interval whose
endpoints straddle an excluded segment is dropped and counted.
"""

from __future__ import annotations

import numpy as np

from app.pipeline.rr import (
    PHYSIOLOGICAL_RR_MAX_MS,
    build_rr,
)


class TestExclusionStraddling:
    def test_bridging_interval_dropped(self) -> None:
        # Beats every second except none between t=10 and t=15 (bad signal).
        peaks = np.array([float(t) for t in range(0, 11)] + [float(t) for t in range(15, 21)])
        rr = build_rr(peaks, excluded_segments=[(10.2, 14.8, "flat/dead signal")])
        # The 10→15 s "interval" (5000 ms) must not appear.
        assert np.all(rr.rr_ms <= PHYSIOLOGICAL_RR_MAX_MS)
        assert 5000.0 not in rr.rr_ms
        assert rr.n_dropped_excluded == 1
        # All the honest 1000 ms intervals survive.
        assert np.allclose(rr.rr_ms, 1000.0)

    def test_interval_inside_exclusion_dropped(self) -> None:
        peaks = np.arange(0.0, 30.0, 1.0)
        rr = build_rr(peaks, excluded_segments=[(10.0, 15.0, "clipping")])
        # Intervals with any part inside (10, 15) are gone: 9→10 ends ON the
        # boundary (kept); 10→11 … 14→15 overlap (dropped).
        assert rr.n_dropped_excluded == 5
        assert len(rr) == 29 - 5

    def test_interval_touching_boundary_kept(self) -> None:
        peaks = np.array([8.0, 9.0, 10.0, 15.0, 16.0, 17.0])
        rr = build_rr(peaks, excluded_segments=[(10.0, 15.0, "gap")])
        # 9→10 and 15→16 touch the boundary without crossing: kept.
        # 10→15 bridges: dropped (also beyond the ceiling).
        assert len(rr) == 4
        assert np.allclose(rr.rr_ms, 1000.0)

    def test_discontinuity_marked_after_drop(self) -> None:
        peaks = np.arange(0.0, 30.0, 1.0)
        rr = build_rr(peaks, excluded_segments=[(10.0, 15.0, "gap")])
        assert np.sum(rr.discontinuity) == 1  # exactly one seam
        seam = np.flatnonzero(rr.discontinuity)[0]
        assert rr.t_s[seam] == 16.0  # first interval after the exclusion


class TestPhysiologicalBounds:
    def test_ceiling_dropped(self) -> None:
        # A 3.5 s dropout bridge with no excluded segment reported.
        peaks = np.array([0.0, 1.0, 2.0, 5.5, 6.5, 7.5])
        rr = build_rr(peaks, excluded_segments=[])
        assert rr.n_dropped_ceiling == 1
        assert np.all(rr.rr_ms <= PHYSIOLOGICAL_RR_MAX_MS)

    def test_floor_dropped(self) -> None:
        # A 200 ms double-detection.
        peaks = np.array([0.0, 1.0, 1.2, 2.2, 3.2])
        rr = build_rr(peaks, excluded_segments=[])
        assert rr.n_dropped_floor == 1
        assert np.all(rr.rr_ms >= 300.0)

    def test_empty_and_single_peak(self) -> None:
        assert len(build_rr(np.array([]), [])) == 0
        assert len(build_rr(np.array([1.0]), [])) == 0
