"""Built-in rule-based 4-class staging (Wake / Light / Deep / REM).

Zero optional dependencies and fully inspectable: every constant below is a
named threshold with its literature grounding, per-epoch class scores are a
weighted sum of logistic feature transforms, and the temporal structure is
imposed by a Viterbi pass over a sticky transition matrix. Deterministic by
construction (no sampling anywhere).

Physiological grounding:

* **Deep (SWS)**: the night's vagal maximum — lowest HR, highest
  beat-to-beat (HF/RMSSD) variability relative to LF, LF/HF minimum
  (Bušek et al. 2005, Physiol Res 54(4):369-376; Vanoli et al. 1995,
  Circulation 91(7):1918-1922).
* **REM**: sympathetic predominance — LF/HF maximum, HR elevated and
  irregular, near-total absence of gross body movement (same sources), and
  more prevalent in the later night (the report's circadian prior).
* **Wake**: movement above the night threshold, or HR well above the
  sustained nocturnal baseline.
* **Light (N1+N2)**: the default state between those poles — it is the most
  common stage of a normal night (~50 % of sleep).

These thresholds live here and not in :mod:`app.screening.thresholds`
because they are staging heuristics, not screening rules.

Accuracy honesty: cardio-only staging tops out near κ≈0.6 / ~77 % (4-class)
even with deep learning (Radha et al. 2019, Sci Rep 9:14149,
doi:10.1038/s41598-019-49703-y); the H10-validated Sleep² network reaches
80.3 % / κ≈0.69 (Topalidis et al. 2023, Sensors 23(5):2390). A transparent
rule set sits *below* both — roughly the 65-75 % band — and the hypnogram
panel says so.
"""

from __future__ import annotations

import numpy as np

from app.sleep.epochs import EpochFeatures
from app.sleep.stages import (
    EPOCH_LEN_S,
    UNSCORED,
    Hypnogram,
    StageVocab,
)

ENGINE_KEY = "heuristic"
ENGINE_LABEL = "Built-in rules (cardio-actigraphy)"
#: Declared here so the UI can name the stages this engine can tell apart
#: before it runs; the staging functions below use it as their vocab so
#: the declaration and the emitted hypnogram cannot drift.
ENGINE_VOCAB = StageVocab.WAKE_LIGHT_DEEP_REM

HEURISTIC_ACCURACY_NOTE = (
    "Rule-based estimate: expect roughly 65-75% epoch agreement with "
    "polysomnography for 4-class staging — below trained models such as "
    "Sleep² (80.3%, κ≈0.69 on this strap; Topalidis et al. 2023, "
    "Sensors 23(5):2390) and the κ≈0.6 ceiling of cardio-only deep "
    "learning (Radha et al. 2019, Sci Rep 9:14149)."
)

# --- Rule constants (each cited in the module docstring) -------------------

#: HR this far above the sustained nocturnal baseline suggests wake
#: (nocturnal HR dips 10-30% below waking rest; a return toward waking
#: levels is the wake signature).
WAKE_HR_DELTA_BPM = 12.0
#: Deep sleep sits near the baseline; above this delta it is unlikely.
DEEP_HR_DELTA_MAX_BPM = 4.0
#: LF/HF below this favours deep sleep (SWS LF/HF minimum, Bušek 2005).
DEEP_LFHF_MAX = 1.0
#: LF/HF must clearly exceed this to favour REM (REM shows the night's
#: LF/HF maximum — Bušek 2005; Vanoli 1995 — but light-sleep N2 values reach
#: ~2, so the REM threshold sits above the N2 range to avoid mislabelling
#: the night's most common stage).
REM_LFHF_MIN = 2.5
#: REM is rare in the first sleep cycle: fraction of the night before which
#: the REM score is damped (first REM latency is typically 70-100 min).
REM_CIRCADIAN_RAMP_FRAC = 0.25

#: Sticky transition matrix (rows: from W/L/D/R; cols: to W/L/D/R).
#: Self-transition ~0.9 per 30 s epoch reflects the minutes-long stage bouts
#: of normal sleep; transitions route through Light (direct Wake↔Deep and
#: Deep↔REM jumps are rare). Kishi et al. 2008, Am J Physiol Regul Integr
#: Comp Physiol 294(6):R1980-R1987 [verify — transition structure, not the
#: exact probabilities, which are smoothing parameters].
TRANSITIONS = np.array(
    [
        [0.900, 0.080, 0.005, 0.015],
        [0.030, 0.900, 0.045, 0.025],
        [0.010, 0.085, 0.900, 0.005],
        [0.025, 0.070, 0.005, 0.900],
    ]
)
#: Nights start awake far more often than not.
INITIAL = np.array([0.70, 0.28, 0.01, 0.01])

#: Weight of the emission (evidence) vs the transition prior in the Viterbi
#: pass; 1.0 = as-is. Kept explicit so the smoothing strength is visible.
EMISSION_WEIGHT = 1.0


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _epoch_scores(features: EpochFeatures) -> np.ndarray:
    """(n_epochs, 4) class scores in [~0, ~3]; NaN features contribute 0."""
    n = features.n_epochs
    hr_delta = features.hr_vs_baseline
    lfhf = features.lf_hf
    rmssd = features.rmssd_ms
    movement = features.movement

    # Night-relative RMSSD (robust z): deep sleep is *this night's* vagal
    # maximum, so the reference is the night itself, not population values.
    finite = np.isfinite(rmssd)
    if np.any(finite):
        med = np.median(rmssd[finite])
        mad = np.median(np.abs(rmssd[finite] - med))
        scale = mad if mad > 1e-9 else max(0.1 * med, 1e-9)
        rmssd_z = (rmssd - med) / scale
    else:
        rmssd_z = np.full(n, np.nan)

    moving = np.zeros(n)
    if np.isfinite(features.movement_threshold):
        valid_m = np.isfinite(movement)
        moving[valid_m] = (movement[valid_m] > features.movement_threshold).astype(
            float
        )

    def safe(x: np.ndarray) -> np.ndarray:
        return np.where(np.isfinite(x), x, 0.0)

    # Progress through the night, for the REM circadian prior.
    frac = np.linspace(0.0, 1.0, n) if n > 1 else np.zeros(1)
    rem_prior = np.clip(frac / REM_CIRCADIAN_RAMP_FRAC, 0.0, 1.0)

    s_wake = (
        2.0 * moving
        + 1.6 * safe(_sigmoid((hr_delta - WAKE_HR_DELTA_BPM) / 3.0))
        + 0.4 * safe(_sigmoid((-rmssd_z - 1.0) / 0.5))
    )
    s_deep = (
        1.2 * safe(_sigmoid((DEEP_LFHF_MAX - lfhf) / 0.25))
        + 0.9 * safe(_sigmoid((rmssd_z - 0.3) / 0.4))
        + 0.6 * safe(_sigmoid((DEEP_HR_DELTA_MAX_BPM - hr_delta) / 2.0))
        - 2.5 * moving
    )
    s_rem = (
        1.5 * safe(_sigmoid((lfhf - REM_LFHF_MIN) / 0.5))
        + 0.5 * safe(_sigmoid((hr_delta - 3.0) / 2.0))
        + 0.3 * rem_prior
        - 2.5 * moving
    )
    s_light = np.full(n, 1.1) - 1.5 * moving

    return np.stack([s_wake, s_light, s_deep, s_rem], axis=1)


def _viterbi(log_emit: np.ndarray) -> np.ndarray:
    """Most probable stage path under the sticky transition prior."""
    n, k = log_emit.shape
    log_a = np.log(TRANSITIONS)
    delta = np.log(INITIAL) + log_emit[0]
    backptr = np.zeros((n, k), dtype=np.int8)
    for i in range(1, n):
        candidates = delta[:, None] + log_a
        backptr[i] = np.argmax(candidates, axis=0)
        delta = candidates[backptr[i], np.arange(k)] + log_emit[i]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = int(np.argmax(delta))
    for i in range(n - 2, -1, -1):
        path[i] = backptr[i + 1, path[i + 1]]
    return path


def stage_heuristic(features: EpochFeatures) -> Hypnogram:
    """Stage one night with the rule set + Viterbi smoothing."""
    n = features.n_epochs
    if n == 0:
        return Hypnogram(
            engine=ENGINE_KEY,
            engine_label=ENGINE_LABEL,
            vocab=ENGINE_VOCAB,
            epoch_len_s=EPOCH_LEN_S,
            epoch_start_s=features.epoch_start_s,
            stages=np.array([], dtype=np.int8),
            probabilities=None,
            accuracy_note=HEURISTIC_ACCURACY_NOTE,
            notes=["Empty epoch grid — nothing to stage."],
        )

    scores = _epoch_scores(features)
    # Softmax → emission probabilities.
    shifted = scores - scores.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)

    # Unstageable epochs carry no evidence: uniform emissions let the Viterbi
    # path bridge them; they are masked to UNSCORED after decoding.
    stageable = features.stageable()
    probs[~stageable] = 0.25

    log_emit = EMISSION_WEIGHT * np.log(np.clip(probs, 1e-12, None))
    path = _viterbi(log_emit)
    stages = path.astype(np.int8)
    stages[~stageable] = UNSCORED

    notes = list(features.notes)
    n_unscored = int(np.sum(~stageable))
    if n_unscored:
        notes.append(
            f"{n_unscored} epoch(s) had too little usable signal to stage."
        )
    if features.movement_source == "wander_proxy":
        notes.append(
            "Wake detection used the ECG-wander movement proxy; an "
            "accelerometer file improves it."
        )

    return Hypnogram(
        engine=ENGINE_KEY,
        engine_label=ENGINE_LABEL,
        vocab=ENGINE_VOCAB,
        epoch_len_s=EPOCH_LEN_S,
        epoch_start_s=features.epoch_start_s,
        stages=stages,
        probabilities=probs,
        accuracy_note=HEURISTIC_ACCURACY_NOTE,
        notes=notes,
    )
