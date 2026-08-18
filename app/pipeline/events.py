"""Ectopy event extraction: grouping confirmed ectopic beats into events.

Everything here is *derived* from the pipeline's existing outputs — the
correction step's ectopic class, the template stage's prematurity and motion
masks — and changes nothing upstream. The confirmation rule (correction-class
ectopic AND premature AND not motion-explained) lives in
:func:`confirmed_ectopic_indices` and is shared with the screening rules, so
the counts that reach flags, metrics, and trigger statistics can never drift
apart.

Consecutive confirmed beats group into events named with the classical
grading vocabulary (single / couplet / run), and each event carries a
compensatory-pause ratio. The pause ratio is a **timing descriptor only**: a
full pause is typical of ventricular origin in the literature and an
incomplete pause of atrial origin, but a single 130 Hz lead cannot determine
origin and nothing downstream may present it as one — user-facing wording
stays "ectopic beats" throughout. Background: docs/RESEARCH.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.pipeline.rr import PHYSIOLOGICAL_RR_MAX_MS, PHYSIOLOGICAL_RR_MIN_MS
from app.pipeline.template import LOCAL_WINDOW_BEATS, PREMATURITY_THRESHOLD_PCT
from app.screening import thresholds as th

#: npz-storable codes for EctopyEvent.kind.
EVENT_KIND_CODES = {"single": 0, "couplet": 1, "run": 2}


@dataclass(frozen=True)
class EctopyEvent:
    """One grouped ectopy event (consecutive confirmed ectopic beats)."""

    kind: str  # "single" | "couplet" | "run"
    #: Index range into the corrected-peak array, inclusive.
    start_beat: int
    end_beat: int
    n_ectopic: int
    t_start_s: float
    t_end_s: float
    #: (RR_pre + RR_post) / (2 × local reference RR); None when either
    #: neighbouring interval is missing, non-physiological, or excluded.
    pause_ratio: float | None
    #: pause_ratio >= th.PAUSE_COMPLETE_RATIO; None when undefined.
    pause_complete: bool | None


@dataclass
class EctopyEvents:
    """All ectopy events of one session, with the session-level aggregates."""

    #: Per corrected beat: True where the beat is a confirmed ectopic.
    confirmed_mask: np.ndarray
    events: list[EctopyEvent]
    n_confirmed: int
    n_singles: int
    n_couplets: int
    n_runs: int
    longest_run_beats: int
    bigeminy_episodes: int
    trigeminy_episodes: int
    #: Confirmed ectopics per hour of *analysed* time (None if no analysed time).
    per_hour: float | None
    #: Confirmed ectopics as a percentage of all beats (None if no beats).
    pct_of_beats: float | None
    n_pause_complete: int
    n_pause_incomplete: int


def confirmed_ectopic_indices(
    n_beats: int,
    ectopic_beat_indices: np.ndarray,
    prematurity_pct: np.ndarray,
    motion_explained: np.ndarray,
) -> np.ndarray:
    """Beats confirmed ectopic by the triple gate, in input order.

    The single shared implementation of the confirmation rule: a beat must be
    classified ectopic by the correction step AND premature against the local
    median RR AND not explained by the motion proxy. ``flag_ectopy`` and the
    event extraction both call this, so their counts are identical by
    construction.

    The correction step's ectopic indices come from neurokit2's Kubios
    implementation, which classifies over the *difference* series of RR
    intervals — its index for an ectopic can land on the beat after the
    premature one (the pause beat). For each candidate ``i`` the gate
    therefore examines ``i-1`` and ``i`` and confirms whichever shows the
    stronger prematurity, recording the premature beat's own index. Where the
    passed prematurity marks ``i`` itself (as the screening tests construct),
    this reduces exactly to the original per-index check.
    """
    confirmed: list[int] = []
    seen: set[int] = set()
    if n_beats == 0 or len(ectopic_beat_indices) == 0:
        return np.array(confirmed, dtype=np.int64)
    for i in np.asarray(ectopic_beat_indices, dtype=np.int64):
        candidates = [j for j in (int(i) - 1, int(i)) if 0 <= j < len(prematurity_pct)]
        candidates = [j for j in candidates if not np.isnan(prematurity_pct[j])]
        if not candidates:
            continue
        j = min(candidates, key=lambda k: prematurity_pct[k])
        premature = prematurity_pct[j] <= PREMATURITY_THRESHOLD_PCT
        motion = bool(motion_explained[j]) if j < len(motion_explained) else False
        if premature and not motion and j not in seen:
            seen.add(j)
            confirmed.append(j)
    return np.array(confirmed, dtype=np.int64)


def extract_events(
    confirmed_indices: np.ndarray,
    peak_times_s: np.ndarray,
    analysed_s: float,
    excluded_segments: list[tuple[float, float, str]],
) -> EctopyEvents:
    """Group confirmed ectopic beats into events and compute aggregates."""
    n_beats = len(peak_times_s)
    mask = np.zeros(n_beats, dtype=bool)
    idx = np.unique(np.asarray(confirmed_indices, dtype=np.int64))
    idx = idx[(idx >= 0) & (idx < n_beats)]
    mask[idx] = True

    events = [
        _build_event(int(start), int(end), peak_times_s, excluded_segments)
        for start, end in _consecutive_groups(idx)
    ]

    n_confirmed = int(len(idx))
    n_singles = sum(1 for e in events if e.kind == "single")
    n_couplets = sum(1 for e in events if e.kind == "couplet")
    n_runs = sum(1 for e in events if e.kind == "run")
    longest_run = max((e.n_ectopic for e in events), default=0)

    return EctopyEvents(
        confirmed_mask=mask,
        events=events,
        n_confirmed=n_confirmed,
        n_singles=n_singles,
        n_couplets=n_couplets,
        n_runs=n_runs,
        longest_run_beats=longest_run,
        bigeminy_episodes=_pattern_episodes(idx, th.BIGEMINY_SPACING),
        trigeminy_episodes=_pattern_episodes(idx, th.TRIGEMINY_SPACING),
        per_hour=(n_confirmed / (analysed_s / 3600.0)) if analysed_s > 0 else None,
        pct_of_beats=(100.0 * n_confirmed / n_beats) if n_beats > 0 else None,
        n_pause_complete=sum(1 for e in events if e.pause_complete is True),
        n_pause_incomplete=sum(1 for e in events if e.pause_complete is False),
    )


def _consecutive_groups(sorted_idx: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) inclusive index pairs of consecutive runs in sorted_idx."""
    if len(sorted_idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(sorted_idx) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(sorted_idx) - 1]))
    return [(int(sorted_idx[s]), int(sorted_idx[e])) for s, e in zip(starts, ends, strict=True)]


def _build_event(
    start: int,
    end: int,
    peak_times_s: np.ndarray,
    excluded_segments: list[tuple[float, float, str]],
) -> EctopyEvent:
    n_ectopic = end - start + 1
    if n_ectopic >= th.RUN_MIN_BEATS:
        kind = "run"
    elif n_ectopic == th.COUPLET_BEATS:
        kind = "couplet"
    else:
        kind = "single"

    ratio = _pause_ratio(start, end, peak_times_s, excluded_segments)
    return EctopyEvent(
        kind=kind,
        start_beat=start,
        end_beat=end,
        n_ectopic=n_ectopic,
        t_start_s=float(peak_times_s[start]),
        t_end_s=float(peak_times_s[end]),
        pause_ratio=ratio,
        pause_complete=(ratio >= th.PAUSE_COMPLETE_RATIO) if ratio is not None else None,
    )


def _pause_ratio(
    start: int,
    end: int,
    peak_times_s: np.ndarray,
    excluded_segments: list[tuple[float, float, str]],
) -> float | None:
    """Compensatory-pause ratio for one event; None whenever it isn't honest.

    The coupling interval (into the event's first beat) plus the post-event
    interval, against twice the local reference RR. Never fabricated across a
    detector gap, a non-physiological interval, or an excluded segment.
    """
    n = len(peak_times_s)
    if start - 1 < 0 or end + 1 >= n:
        return None

    rr_pre_ms = (peak_times_s[start] - peak_times_s[start - 1]) * 1000.0
    rr_post_ms = (peak_times_s[end + 1] - peak_times_s[end]) * 1000.0
    for rr in (rr_pre_ms, rr_post_ms):
        if not (PHYSIOLOGICAL_RR_MIN_MS <= rr <= PHYSIOLOGICAL_RR_MAX_MS):
            return None
    for a, b in ((start - 1, start), (end, end + 1)):
        if _interval_excluded(peak_times_s[a], peak_times_s[b], excluded_segments):
            return None

    reference = _local_reference_rr_ms(start, end, peak_times_s)
    if reference is None or reference <= 0:
        return None
    return float((rr_pre_ms + rr_post_ms) / (2.0 * reference))


def _interval_excluded(
    t_start: float,
    t_end: float,
    excluded_segments: list[tuple[float, float, str]],
) -> bool:
    """Open-interval overlap test, same convention as app.pipeline.rr."""
    return any(
        t_start < seg_end and t_end > seg_start
        for seg_start, seg_end, _ in excluded_segments
    )


def _local_reference_rr_ms(
    start: int, end: int, peak_times_s: np.ndarray
) -> float | None:
    """Median of nearby RR intervals, excluding those touching the event.

    Intervals touching the event are interval indices start-1 .. end (interval
    ``i`` runs from beat ``i`` to beat ``i+1``); up to LOCAL_WINDOW_BEATS
    surrounding intervals on each side feed the median, mirroring the
    prematurity computation in app.pipeline.template.
    """
    rr = np.diff(peak_times_s) * 1000.0
    lo = max(0, start - 1 - LOCAL_WINDOW_BEATS)
    hi = min(len(rr), end + 1 + LOCAL_WINDOW_BEATS)
    keep = [
        rr[i]
        for i in range(lo, hi)
        if (i < start - 1 or i > end)
        and PHYSIOLOGICAL_RR_MIN_MS <= rr[i] <= PHYSIOLOGICAL_RR_MAX_MS
    ]
    if len(keep) < 3:
        return None
    return float(np.median(keep))


def _pattern_episodes(sorted_idx: np.ndarray, spacing: int) -> int:
    """Count maximal chains of ectopics spaced exactly ``spacing`` beats apart
    with at least th.BIGEMINY_MIN_RUN pattern beats (bigeminy: spacing 2,
    trigeminy: spacing 3 — generalising the screening rule's bigeminy check).
    """
    episodes = 0
    run = 1
    for a, b in zip(sorted_idx, sorted_idx[1:], strict=False):
        if b - a == spacing:
            run += 1
        else:
            if run >= th.BIGEMINY_MIN_RUN:
                episodes += 1
            run = 1
    if len(sorted_idx) and run >= th.BIGEMINY_MIN_RUN:
        episodes += 1
    return episodes
