"""Sleep-architecture arithmetic on a hypnogram.

Definitions (each an epoch count × 0.5 min; all exactly testable):

* **TIB** (time in bed) — the whole scored grid. The recording start stands
  in for lights-off: the app cannot know when the user intended to sleep, so
  the README tells users to start recording at lights-off and this proxy is
  stated in the report.
* **Sleep onset** — the first epoch scored as any sleep stage (AASM Scoring
  Manual v2.6 sleep-onset rule: first epoch of any stage other than wake).
* **TST** — count of sleep-stage epochs. **SE** = TST/TIB × 100 (normative
  reference: Ohayon et al. 2017, Sleep Health 3(1):6-19,
  doi:10.1016/j.sleh.2016.11.006).
* **WASO** — wake epochs strictly between sleep onset and the final sleep
  epoch. Wake after the final sleep epoch is terminal wake, not WASO.
* **Awakenings** — contiguous wake runs inside the same span; a run is ≥ 1
  epoch, i.e. ≥ 30 s, which the report states (brief arousals shorter than
  an epoch are invisible at this resolution).
* **REM latency** — first REM epoch minus sleep onset (None when the
  vocabulary has no REM class or REM never occurred).

Unscored epochs count toward TIB (the strap was worn) but toward no stage;
their total is reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.pipeline.rr import RRSeries
from app.sleep.stages import (
    REM_CODE,
    UNSCORED,
    WAKE_CODE,
    Hypnogram,
    epoch_indices,
    stage_labels,
)


@dataclass
class SleepSummary:
    """Sleep-architecture numbers for one hypnogram."""

    engine: str
    vocab: str

    tib_min: float
    tst_min: float
    sleep_efficiency_pct: float
    unscored_min: float

    sol_min: float | None
    waso_min: float | None
    awakenings_n: int | None
    rem_latency_min: float | None

    #: Minutes per stage. Only the classes the vocabulary actually has are
    #: set; everything else stays None (a 3-class engine cannot report deep
    #: sleep, and no value is ever invented for it).
    wake_min: float | None = None
    sleep_min: float | None = None  # 2-class vocab only
    nrem_min: float | None = None  # 3-class vocab only
    light_min: float | None = None
    deep_min: float | None = None
    rem_min: float | None = None
    n1_min: float | None = None
    n2_min: float | None = None

    #: label -> % of TST, for sleep stages only.
    stage_pct_of_tst: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


#: SleepSummary attribute per (vocab label). WAKE is handled separately.
_STAGE_ATTR: dict[str, str] = {
    "Sleep": "sleep_min",
    "NREM": "nrem_min",
    "Light": "light_min",
    "Deep": "deep_min",
    "REM": "rem_min",
    "N1": "n1_min",
    "N2": "n2_min",
    "N3": "deep_min",  # AASM N3 *is* deep sleep — one queryable column.
}


def summarize(hyp: Hypnogram) -> SleepSummary:
    """Compute the sleep-architecture summary for one hypnogram."""
    stages = hyp.stages
    epoch_min = hyp.epoch_len_s / 60.0
    n_epochs = len(stages)
    notes: list[str] = []

    tib_min = n_epochs * epoch_min
    unscored_min = float(np.sum(stages == UNSCORED)) * epoch_min
    if unscored_min > 0:
        notes.append(
            f"{unscored_min:.1f} min of epochs could not be scored and count "
            "toward time in bed but no stage."
        )

    sleep_mask = stages >= 1
    tst_min = float(np.sum(sleep_mask)) * epoch_min
    se = (tst_min / tib_min * 100.0) if tib_min > 0 else 0.0

    summary = SleepSummary(
        engine=hyp.engine,
        vocab=hyp.vocab.value,
        tib_min=tib_min,
        tst_min=tst_min,
        sleep_efficiency_pct=se,
        unscored_min=unscored_min,
        sol_min=None,
        waso_min=None,
        awakenings_n=None,
        rem_latency_min=None,
        notes=notes,
    )

    labels = stage_labels(hyp.vocab)
    summary.wake_min = float(np.sum(stages == WAKE_CODE)) * epoch_min
    for code, label in enumerate(labels):
        if code == WAKE_CODE:
            continue
        attr = _STAGE_ATTR[label]
        setattr(summary, attr, float(np.sum(stages == code)) * epoch_min)

    if tst_min > 0:
        summary.stage_pct_of_tst = {
            label: float(np.sum(stages == code)) * epoch_min / tst_min * 100.0
            for code, label in enumerate(labels)
            if code != WAKE_CODE
        }

    sleep_idx = np.flatnonzero(sleep_mask)
    if len(sleep_idx) == 0:
        notes.append("No epoch was scored as sleep.")
        return summary

    onset = int(sleep_idx[0])
    last_sleep = int(sleep_idx[-1])
    summary.sol_min = onset * epoch_min

    between = stages[onset : last_sleep + 1]
    wake_between = between == WAKE_CODE
    summary.waso_min = float(np.sum(wake_between)) * epoch_min
    # Contiguous wake runs = rising edges of the wake mask.
    edges = np.diff(np.concatenate(([False], wake_between)).astype(np.int8)) == 1
    summary.awakenings_n = int(np.sum(edges))

    rem_code = REM_CODE[hyp.vocab]
    if rem_code is not None:
        rem_idx = np.flatnonzero(stages == rem_code)
        if len(rem_idx) > 0:
            summary.rem_latency_min = (int(rem_idx[0]) - onset) * epoch_min

    return summary


def stage_stats(hyp: Hypnogram, rr: RRSeries) -> list[dict]:
    """Per-stage mean HR and RMSSD for the report table.

    Epoch membership of an RR interval is decided by its ending-beat time.
    RMSSD uses successive differences only between intervals that are
    contiguous (``discontinuity`` false) *and* fall in epochs of the same
    stage — a difference across a stage boundary or a dropped interval would
    mix autonomic states.
    """
    if len(rr) == 0 or hyp.n_epochs == 0:
        return []

    epoch_of = epoch_indices(rr.t_s, hyp.n_epochs, hyp.epoch_len_s)
    valid = epoch_of >= 0
    stage_of_interval = np.full(len(rr), UNSCORED, dtype=np.int8)
    stage_of_interval[valid] = hyp.stages[epoch_of[valid]]

    diffs = np.diff(rr.rr_ms)
    # A successive difference is usable when the later interval is contiguous
    # with the earlier one and both carry the same stage.
    usable_diff = (~rr.discontinuity[1:]) & (
        stage_of_interval[1:] == stage_of_interval[:-1]
    )

    epoch_min = hyp.epoch_len_s / 60.0
    rows: list[dict] = []
    for code, label in enumerate(stage_labels(hyp.vocab)):
        in_stage = stage_of_interval == code
        n_epochs_stage = int(np.sum(hyp.stages == code))
        row: dict = {
            "stage": label,
            "minutes": n_epochs_stage * epoch_min,
            "n_epochs": n_epochs_stage,
            "mean_hr_bpm": None,
            "mean_rmssd_ms": None,
        }
        if np.any(in_stage):
            row["mean_hr_bpm"] = float(60000.0 / np.mean(rr.rr_ms[in_stage]))
            d = diffs[usable_diff & in_stage[1:]]
            if len(d) >= 10:
                row["mean_rmssd_ms"] = float(np.sqrt(np.mean(d**2)))
        rows.append(row)
    return rows
