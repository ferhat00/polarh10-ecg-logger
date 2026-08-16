"""RR-series construction with exclusion-aware filtering.

The bug this module exists to prevent: diffing R-peak times across an excluded
window produces one spurious multi-second "beat interval" that bridges the
gap. Any interval whose endpoints straddle (or fall inside) an excluded
segment is dropped, as is any interval beyond a physiological ceiling — and
every drop is counted and reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Physiological RR bounds. 300 ms ≈ 200 bpm (Tanaka max-HR territory);
#: 2200 ms ≈ 27 bpm, below any physiologic sinus rate — longer intervals are
#: detector dropouts or exclusion bridges, not beats.
PHYSIOLOGICAL_RR_MIN_MS = 300.0
PHYSIOLOGICAL_RR_MAX_MS = 2200.0


@dataclass
class RRSeries:
    """Beat-to-beat intervals that survived exclusion and ceiling filtering."""

    #: Interval lengths in milliseconds.
    rr_ms: np.ndarray
    #: Time (s, recording clock) of each interval's *ending* beat.
    t_s: np.ndarray
    #: True where this interval is NOT contiguous with the previous one
    #: (an interval was dropped between them) — spectral analysis must not
    #: bridge these boundaries.
    discontinuity: np.ndarray
    n_dropped_excluded: int
    n_dropped_ceiling: int
    n_dropped_floor: int

    def __len__(self) -> int:
        return len(self.rr_ms)


def build_rr(
    peak_times_s: np.ndarray,
    excluded_segments: list[tuple[float, float, str]],
) -> RRSeries:
    """Build the RR series from R-peak times, dropping compromised intervals.

    An interval is dropped when:

    * it overlaps an excluded segment (either endpoint inside, or the interval
      spanning the segment entirely — the "bridging" case), or
    * it exceeds :data:`PHYSIOLOGICAL_RR_MAX_MS` (dropout bridge) or falls
      below :data:`PHYSIOLOGICAL_RR_MIN_MS` (double-detection).
    """
    if len(peak_times_s) < 2:
        empty = np.array([])
        return RRSeries(empty, empty, np.array([], dtype=bool), 0, 0, 0)

    starts = peak_times_s[:-1]
    ends = peak_times_s[1:]
    rr_ms = (ends - starts) * 1000.0

    keep = np.ones(len(rr_ms), dtype=bool)
    n_excluded = 0
    for seg_start, seg_end, _reason in excluded_segments:
        # Open-interval overlap: an interval touching the boundary is fine,
        # one crossing into or over the segment is not.
        overlaps = (starts < seg_end) & (ends > seg_start)
        n_excluded += int(np.sum(overlaps & keep))
        keep &= ~overlaps

    too_long = rr_ms > PHYSIOLOGICAL_RR_MAX_MS
    too_short = rr_ms < PHYSIOLOGICAL_RR_MIN_MS
    n_ceiling = int(np.sum(too_long & keep))
    n_floor = int(np.sum(too_short & keep))
    keep &= ~too_long & ~too_short

    kept_idx = np.flatnonzero(keep)
    discontinuity = np.zeros(len(kept_idx), dtype=bool)
    if len(kept_idx) > 1:
        discontinuity[1:] = np.diff(kept_idx) > 1

    return RRSeries(
        rr_ms=rr_ms[kept_idx],
        t_s=ends[kept_idx],
        discontinuity=discontinuity,
        n_dropped_excluded=n_excluded,
        n_dropped_ceiling=n_ceiling,
        n_dropped_floor=n_floor,
    )
