"""SleepECG staging engine (optional dependency).

Uses the pre-trained ``wrn-gru-mesa-weighted`` GRU classifier shipped
*inside* the ``sleepecg`` wheel (BSD-3): Wake/REM/NREM per 30 s epoch from
heartbeat times, clock time, and (when known) age and sex. Trained on 1,971
MESA polysomnography nights and evaluated on SHHS (see the SleepECG
documentation and Brunner et al. 2023, JOSS 8(86):5411,
doi:10.21105/joss.05411). The weighted variant is chosen because plain
``wrn-gru-mesa`` under-recalls the minority classes (Wake, REM) that this
app's users most care about.

Verified against sleepecg 0.5.9 (see the mapping tests):

* ``load_classifier(name, "SleepECG")`` loads the classifier bundled in the
  wheel — **no network access, ever** (the app's offline stance).
* ``stage(..., return_mode="prob")`` returns ``(n_epochs, 4)`` with columns
  ``[UNDEFINED, NREM, REM, WAKE]`` for ``stages_mode='wake-rem-nrem'``
  (integer labels: 0=UNDEFINED, 1=NREM, 2=REM, 3=WAKE).
* ``recording_start_time`` is a clock ``datetime.time`` — the classifier's
  time features are circadian, so the recording's *local* clock time is
  passed (the app runs on the user's machine, so the system timezone is the
  recording timezone; a note records this assumption).

TensorFlow is required at inference. Everything TF-touching is imported
lazily inside functions so importing this module never pulls TF, and
:func:`status` reports exactly what is missing and how to install it.
"""

from __future__ import annotations

import datetime as dt
import importlib.util

import numpy as np

from app.activities.base import PersonContext
from app.sleep.engines import EngineStatus
from app.sleep.stages import (
    EPOCH_LEN_S,
    UNSCORED,
    Hypnogram,
    StageVocab,
    make_epoch_grid,
)

ENGINE_KEY = "sleepecg"
ENGINE_LABEL = "SleepECG GRU (wrn-gru-mesa-weighted)"

CLASSIFIER_NAME = "wrn-gru-mesa-weighted"

SLEEPECG_ACCURACY_NOTE = (
    "Pre-trained GRU (sleepecg 'wrn-gru-mesa-weighted'): Wake/REM/NREM from "
    "heartbeat times, trained on 1,971 MESA polysomnography nights and "
    "evaluated on SHHS (Brunner et al. 2023, JOSS 8(86):5411). Heart-beat-"
    "based staging tops out near ~77% / κ≈0.6 for comparable tasks "
    "(Radha et al. 2019, Sci Rep 9:14149) — an estimate, not a sleep study."
)

#: sleepecg 'wake-rem-nrem' integer label -> this app's WAKE_REM_NREM code
#: (0=WAKE 1=NREM 2=REM). Verified against sleepecg 0.5.9 _STAGE_NAMES.
_SLEEPECG_TO_VOCAB = {0: UNSCORED, 1: 1, 2: 2, 3: 0}

#: Probability-column order in this app's vocab: [WAKE, NREM, REM] taken
#: from sleepecg's columns [UNDEFINED, NREM, REM, WAKE].
_PROB_COLUMNS = (3, 1, 2)


def status() -> EngineStatus:
    """Availability without importing anything heavy."""
    missing = [
        name
        for name in ("sleepecg", "tensorflow")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        return EngineStatus(
            key=ENGINE_KEY,
            label=ENGINE_LABEL,
            available=False,
            unavailable_reason=(
                f"{' and '.join(missing)} not installed — "
                "pip install -r requirements-sleep.txt (CPU-only TensorFlow "
                "is enough; see docs/SLEEP.md)."
            ),
        )
    return EngineStatus(key=ENGINE_KEY, label=ENGINE_LABEL, available=True)


def stage_sleepecg(
    peak_times_s: np.ndarray,
    start_time: dt.datetime,
    duration_s: float,
    ctx: PersonContext | None = None,
    sex: str | None = None,
) -> Hypnogram:
    """Stage one night with the bundled sleepecg classifier.

    ``peak_times_s`` are the pipeline's corrected R-peak times (seconds on
    the recording clock). Output lands on the canonical epoch grid; epochs
    beyond what the classifier scored are UNSCORED.
    """
    import sleepecg  # lazy: pulls TensorFlow via the classifier below

    grid = make_epoch_grid(duration_s)
    n_epochs = len(grid)
    notes: list[str] = []

    local_start = start_time.astimezone()
    notes.append(
        f"Circadian features use the local clock time {local_start:%H:%M} "
        "(system timezone — assumed to be the recording's timezone)."
    )

    subject_data = _subject_data(ctx, sex, notes)

    clf = sleepecg.load_classifier(CLASSIFIER_NAME, "SleepECG")
    record = sleepecg.SleepRecord(
        sleep_stage_duration=int(EPOCH_LEN_S),
        heartbeat_times=np.asarray(peak_times_s, dtype=float),
        recording_start_time=local_start.time(),
        subject_data=subject_data,
    )
    probs = sleepecg.stage(clf, record, return_mode="prob")

    raw = np.argmax(probs, axis=1)
    mapped = np.array(
        [_SLEEPECG_TO_VOCAB.get(int(s), UNSCORED) for s in raw], dtype=np.int8
    )

    stages = np.full(n_epochs, UNSCORED, dtype=np.int8)
    probabilities = np.full((n_epochs, len(_PROB_COLUMNS)), np.nan)
    n_common = min(n_epochs, len(mapped))
    stages[:n_common] = mapped[:n_common]
    probabilities[:n_common] = probs[:n_common][:, _PROB_COLUMNS]
    if len(mapped) != n_epochs:
        notes.append(
            f"Classifier scored {len(mapped)} epoch(s) against a "
            f"{n_epochs}-epoch grid; the difference is unscored."
        )

    return Hypnogram(
        engine=ENGINE_KEY,
        engine_label=ENGINE_LABEL,
        vocab=StageVocab.WAKE_REM_NREM,
        epoch_len_s=EPOCH_LEN_S,
        epoch_start_s=grid,
        stages=stages,
        probabilities=probabilities,
        accuracy_note=SLEEPECG_ACCURACY_NOTE,
        notes=notes,
    )


def _subject_data(
    ctx: PersonContext | None, sex: str | None, notes: list[str]
):
    """SubjectData from what the person profile actually knows.

    The classifier was trained with age and sex features; missing values
    become NaN features the model was trained to tolerate (``max_nans``),
    at some accuracy cost — said out loud in the notes.
    """
    import sleepecg

    age = None
    if ctx is not None and ctx.age_years is not None:
        age = int(ctx.age_years)
    gender = None
    if sex == "male":
        gender = int(sleepecg.Gender.MALE)
    elif sex == "female":
        gender = int(sleepecg.Gender.FEMALE)

    missing = [name for name, v in (("age", age), ("sex", gender)) if v is None]
    if missing:
        notes.append(
            f"No {' or '.join(missing)} on the person profile — the "
            "classifier ran without those features (slightly lower accuracy). "
            "Add them on the person page."
        )
    return sleepecg.SubjectData(gender=gender, age=age)
