"""Trunk-posture classification from the H10's chest accelerometer.

Method: the gravity component of each axis (4th-order Butterworth low-pass
at 0.25 Hz — the exact complement of the 0.25–3 Hz movement band used by
``app.sleep.actigraphy``), averaged per 30 s epoch on the ECG clock, gives
a per-epoch gravity vector. Its orientation classifies the trunk:

* ``upright`` when gravity lies mostly along the torso's long axis (a chest
  strap cannot distinguish sitting from standing — never claimed);
* otherwise lying, split into supine / prone / left / right by the roll
  angle around the torso axis;
* ``moving`` when the actigraphy counts exceed the movement threshold (an
  orientation read during motion is unreliable);
* ``unknown`` when |g| is far from 1 g (bad contact / off-body) or ACC
  coverage is poor.

Axis frame: the only convention verified in this codebase (see
``tests/synth_util.py``) is that lying down puts gravity mostly on **+Z**
(chest-normal) with **Y** along the torso and **X** left–right along the
strap. The sign constants below encode that frame in one place. A strap worn
flipped or rotated swaps left/right and supine/prone — which is why every
posture surface carries :data:`POSTURE_DISCLAIMER` and the dominant gravity
vector is exposed for self-validation against a recording of known posture.

Chest-worn accelerometers classify lying postures with high accuracy in
research settings (docs/CONTEXT_METRICS.md §3.1); posture matters because it
is the largest within-subject HRV modifier (Schneider et al. 2024,
doi:10.1007/s00421-024-05601-4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sp_signal

from app.ingest.acc_loader import LoadedAcc
from app.sleep.actigraphy import AccEpochs

#: Epoch length; matches the sleep-staging grid so arrays align.
EPOCH_LEN_S = 30.0

#: Low-pass corner isolating gravity — complement of the movement band.
GRAVITY_LOWPASS_HZ = 0.25

#: |g| outside this window means bad contact/off-body → unknown.
G_MIN_MG, G_MAX_MG = 700.0, 1300.0

#: Epochs with less ACC coverage than this are unknown (mirrors actigraphy).
MIN_COVERAGE = 0.5

#: Gravity within this angle of the torso long axis → upright.
UPRIGHT_CONE_DEG = 40.0

#: The verified in-repo frame: supine ⇒ gravity ≈ +Z, torso long axis = Y,
#: left–right = X. One place to flip if a future strap mounts differently.
SUPINE_Z_SIGN = +1.0
LEFT_X_SIGN = +1.0

#: int8 codes for the cache and extras.
POSTURE_CODES: dict[str, int] = {
    "unknown": 0,
    "supine": 1,
    "prone": 2,
    "left": 3,
    "right": 4,
    "upright": 5,
    "moving": 6,
}
CODE_NAMES = {v: k for k, v in POSTURE_CODES.items()}

#: The four orientations the ACC can actually distinguish while lying.
LYING_POSTURES = ("supine", "prone", "left", "right")

#: Autofill guards: only a clearly dominant lying posture with enough valid
#: coverage ever auto-fills the session's body-position field.
AUTOFILL_MIN_DOMINANT_PCT = 60.0
AUTOFILL_MIN_VALID_PCT = 50.0

POSTURE_DISCLAIMER = (
    "Posture is estimated from the chest strap's orientation relative to "
    "gravity. It assumes the strap was worn in the usual position; a flipped "
    "or rotated strap swaps left/right and supine/prone. Upright cannot be "
    "split into sitting vs standing from a chest sensor. Orientation during "
    "movement is not classified."
)


@dataclass
class PostureResult:
    """Per-epoch posture on the ECG epoch grid, plus summary statistics."""

    epoch_start_s: np.ndarray
    #: int8 per epoch, see POSTURE_CODES.
    codes: np.ndarray
    #: Posture name -> % of *classifiable* epochs (moving/unknown excluded).
    pct_by_posture: dict[str, float]
    #: Dominant classifiable posture (None when nothing was classifiable).
    dominant: str | None
    dominant_pct: float
    #: Transitions between distinct classifiable postures.
    n_transitions: int
    #: % of epochs that were classifiable at all.
    valid_pct: float
    #: Mean gravity vector (mg) over classifiable epochs — for the wearer to
    #: sanity-check the axis convention against a known-posture recording.
    mean_gravity_mg: tuple[float, float, float] | None
    notes: list[str] = field(default_factory=list)

    def as_extras(self) -> dict:
        return {
            "pct_by_posture": {k: round(v, 1) for k, v in self.pct_by_posture.items()},
            "dominant": self.dominant,
            "dominant_pct": round(self.dominant_pct, 1),
            "n_transitions": self.n_transitions,
            "valid_pct": round(self.valid_pct, 1),
            "mean_gravity_mg": (
                None
                if self.mean_gravity_mg is None
                else [round(v, 0) for v in self.mean_gravity_mg]
            ),
            "epoch_len_s": EPOCH_LEN_S,
            "disclaimer": POSTURE_DISCLAIMER,
            "notes": list(self.notes),
        }


def classify_posture(
    acc: LoadedAcc,
    ecg_start_time,
    n_epochs: int,
    acc_epochs: AccEpochs | None = None,
    epoch_len_s: float = EPOCH_LEN_S,
) -> PostureResult:
    """Classify trunk posture per epoch, aligned to the ECG clock.

    ``acc_epochs`` (the actigraphy counts) supplies the movement veto; when
    absent, no epoch is vetoed for movement.
    """
    notes: list[str] = []
    offset_s = (acc.start_time - ecg_start_time).total_seconds()

    # Gravity per axis: zero-phase low-pass, mirroring the movement band-pass.
    nyq = acc.sampling_rate_hz / 2.0
    sos = sp_signal.butter(4, GRAVITY_LOWPASS_HZ / nyq, btype="lowpass", output="sos")
    gx = sp_signal.sosfiltfilt(sos, acc.x_mg)
    gy = sp_signal.sosfiltfilt(sos, acc.y_mg)
    gz = sp_signal.sosfiltfilt(sos, acc.z_mg)

    # Per-epoch mean gravity on the ECG epoch grid (same alignment arithmetic
    # as actigraphy.activity_counts).
    t_ecg = acc.time_s + offset_s
    epoch_idx = np.floor(t_ecg / epoch_len_s).astype(np.int64)
    in_grid = (epoch_idx >= 0) & (epoch_idx < n_epochs)
    idx = epoch_idx[in_grid]

    n_samples = np.bincount(idx, minlength=n_epochs)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_gx = np.bincount(idx, weights=gx[in_grid], minlength=n_epochs) / n_samples
        mean_gy = np.bincount(idx, weights=gy[in_grid], minlength=n_epochs) / n_samples
        mean_gz = np.bincount(idx, weights=gz[in_grid], minlength=n_epochs) / n_samples
    coverage = n_samples / (acc.sampling_rate_hz * epoch_len_s)

    g_mag = np.sqrt(mean_gx**2 + mean_gy**2 + mean_gz**2)
    codes = np.zeros(n_epochs, dtype=np.int8)  # unknown by default

    valid = (
        (coverage >= MIN_COVERAGE)
        & np.isfinite(g_mag)
        & (g_mag >= G_MIN_MG)
        & (g_mag <= G_MAX_MG)
    )

    # Movement veto from the actigraphy counts (same grid).
    moving = np.zeros(n_epochs, dtype=bool)
    if acc_epochs is not None and len(acc_epochs.counts) == n_epochs:
        if np.isfinite(acc_epochs.threshold):
            counts_ok = ~np.isnan(acc_epochs.counts)
            moving[counts_ok] = acc_epochs.counts[counts_ok] > acc_epochs.threshold

    upright_cos = np.cos(np.deg2rad(UPRIGHT_CONE_DEG))
    for i in np.flatnonzero(valid):
        if moving[i]:
            codes[i] = POSTURE_CODES["moving"]
            continue
        if abs(mean_gy[i]) >= upright_cos * g_mag[i]:
            codes[i] = POSTURE_CODES["upright"]
            continue
        # Lying: roll around the torso axis from the (x, z) gravity components.
        phi = np.arctan2(LEFT_X_SIGN * mean_gx[i], SUPINE_Z_SIGN * mean_gz[i])
        deg = np.degrees(phi)
        if abs(deg) <= 45.0:
            codes[i] = POSTURE_CODES["supine"]
        elif abs(deg) >= 135.0:
            codes[i] = POSTURE_CODES["prone"]
        elif deg > 0:
            codes[i] = POSTURE_CODES["left"]
        else:
            codes[i] = POSTURE_CODES["right"]

    n_unknown = int(np.sum(~valid))
    if n_unknown:
        notes.append(
            f"{n_unknown} epoch(s) unclassifiable (poor ACC coverage or |g| "
            f"outside {G_MIN_MG:.0f}–{G_MAX_MG:.0f} mg)."
        )

    # --- summary ----------------------------------------------------------
    classifiable = (codes != POSTURE_CODES["unknown"]) & (codes != POSTURE_CODES["moving"])
    n_classifiable = int(np.sum(classifiable))
    pct_by: dict[str, float] = {}
    dominant: str | None = None
    dominant_pct = 0.0
    mean_gravity: tuple[float, float, float] | None = None
    if n_classifiable:
        for name in (*LYING_POSTURES, "upright"):
            n = int(np.sum(codes == POSTURE_CODES[name]))
            if n:
                pct_by[name] = 100.0 * n / n_classifiable
        dominant = max(pct_by, key=pct_by.get)
        dominant_pct = pct_by[dominant]
        mean_gravity = (
            float(np.mean(mean_gx[classifiable])),
            float(np.mean(mean_gy[classifiable])),
            float(np.mean(mean_gz[classifiable])),
        )

    seq = codes[classifiable]
    n_transitions = int(np.sum(seq[1:] != seq[:-1])) if len(seq) > 1 else 0

    return PostureResult(
        epoch_start_s=np.arange(n_epochs) * epoch_len_s,
        codes=codes,
        pct_by_posture=pct_by,
        dominant=dominant,
        dominant_pct=dominant_pct,
        n_transitions=n_transitions,
        valid_pct=100.0 * n_classifiable / n_epochs if n_epochs else 0.0,
        mean_gravity_mg=mean_gravity,
        notes=notes,
    )


def autofill_position(posture: PostureResult) -> str | None:
    """The body-position value ACC evidence supports, or None.

    Only the four lying postures ever autofill (upright is ambiguous between
    sitting and standing), and only when clearly dominant with enough valid
    coverage.
    """
    if (
        posture.dominant in LYING_POSTURES
        and posture.dominant_pct >= AUTOFILL_MIN_DOMINANT_PCT
        and posture.valid_pct >= AUTOFILL_MIN_VALID_PCT
    ):
        return posture.dominant
    return None
