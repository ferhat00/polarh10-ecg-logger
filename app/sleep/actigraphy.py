"""Activity counts from chest-worn accelerometry, and the wake override.

Method: vector magnitude of the three axes, 4th-order Butterworth band-pass
0.25–3 Hz (removes the gravity/posture component below and sensor noise
above the human-movement band), rectify, and integrate per 30 s epoch —
the raw-accelerometry actigraphy convention of te Lindert & Van Someren
2013 (Sleep 36(5):781-789, doi:10.5665/sleep.2648). Counts are the integral
of |filtered acceleration| over the epoch (mg·s), so they are independent
of the ACC sampling rate.

Alignment: ACC and ECG are separate exports with their own clocks; both
loaders resolve wall-clock start times, and epochs live on the **ECG** grid.
Each epoch carries a coverage fraction so a mis-aligned or shorter ACC
stream is visible, never silently zero.

Wake override: sustained movement (≥ MIN_CONSECUTIVE_EPOCHS consecutive
epochs above a robust threshold) forces those epochs to WAKE in any
engine's hypnogram. The threshold is median + 5·MAD of the night's counts —
self-calibrating per night, because absolute Cole-Kripke-style thresholds
require per-device calibration this app cannot assume. Both the raw and the
overridden hypnograms are kept; the override count is surfaced in the
report.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy import signal as sp_signal

from app.ingest.acc_loader import LoadedAcc
from app.sleep.stages import EPOCH_LEN_S, WAKE_CODE, Hypnogram

#: Band-pass corners for the human-movement band (te Lindert & Van Someren).
MOVEMENT_BAND_HZ = (0.25, 3.0)

#: Epochs with less ACC coverage than this get NaN counts (unknown, not zero).
MIN_COVERAGE = 0.5

#: MAD multiplier for the per-night movement threshold.
THRESHOLD_MADS = 5.0

#: Consecutive above-threshold epochs required before the override fires —
#: a single 30 s twitch is a normal arousal, not confirmed wake.
MIN_CONSECUTIVE_EPOCHS = 2


@dataclass
class AccEpochs:
    """Per-epoch activity counts on the ECG epoch grid."""

    epoch_start_s: np.ndarray
    #: Integral of band-passed |acceleration| per epoch (mg·s); NaN where
    #: ACC coverage was insufficient.
    counts: np.ndarray
    #: Fraction of each epoch covered by ACC samples.
    coverage: np.ndarray
    #: The wake-override threshold used for this night (mg·s).
    threshold: float
    notes: list[str] = field(default_factory=list)


def activity_counts(
    acc: LoadedAcc,
    ecg_start_time,
    n_epochs: int,
    epoch_len_s: float = EPOCH_LEN_S,
) -> AccEpochs:
    """Band-passed activity counts per epoch, aligned to the ECG clock."""
    notes = list(acc.notes)
    offset_s = (acc.start_time - ecg_start_time).total_seconds()
    if abs(offset_s) > 1.0:
        notes.append(
            f"ACC stream starts {offset_s:+.1f} s relative to the ECG recording; "
            "epochs are aligned on the ECG clock."
        )

    mag = np.sqrt(acc.x_mg**2 + acc.y_mg**2 + acc.z_mg**2)
    nyq = acc.sampling_rate_hz / 2.0
    high = min(MOVEMENT_BAND_HZ[1], 0.9 * nyq)
    sos = sp_signal.butter(
        4, [MOVEMENT_BAND_HZ[0] / nyq, high / nyq], btype="bandpass", output="sos"
    )
    filtered = np.abs(sp_signal.sosfiltfilt(sos, mag))

    # Time of each ACC sample on the ECG clock, then its epoch index.
    t_ecg = acc.time_s + offset_s
    epoch_idx = np.floor(t_ecg / epoch_len_s).astype(np.int64)
    in_grid = (epoch_idx >= 0) & (epoch_idx < n_epochs)

    sums = np.bincount(
        epoch_idx[in_grid], weights=filtered[in_grid], minlength=n_epochs
    )
    n_samples = np.bincount(epoch_idx[in_grid], minlength=n_epochs)

    # Integral over the epoch: sum(|a|)·dt, dt = 1/fs.
    counts = sums / acc.sampling_rate_hz
    coverage = n_samples / (acc.sampling_rate_hz * epoch_len_s)
    counts[coverage < MIN_COVERAGE] = np.nan

    covered = counts[~np.isnan(counts)]
    if len(covered) == 0:
        threshold = np.nan
        notes.append("No epoch had sufficient ACC coverage; movement unusable.")
    else:
        med = float(np.median(covered))
        mad = float(np.median(np.abs(covered - med)))
        if mad > 0:
            threshold = med + THRESHOLD_MADS * mad
        else:
            # A perfectly quiet night can have MAD 0; require a clear
            # departure from the (near-constant) baseline instead.
            threshold = 2.0 * med + 1e-9
            notes.append(
                "Movement MAD is 0 (very quiet night); override threshold set "
                "to twice the median count."
            )

    low_cov = int(np.sum(coverage < MIN_COVERAGE))
    if low_cov:
        notes.append(
            f"{low_cov} epoch(s) lacked ACC coverage (<{MIN_COVERAGE:.0%}) and "
            "carry no movement value."
        )

    return AccEpochs(
        epoch_start_s=np.arange(n_epochs) * epoch_len_s,
        counts=counts,
        coverage=coverage,
        threshold=float(threshold),
        notes=notes,
    )


def wake_override_mask(
    acc_epochs: AccEpochs, min_consecutive: int = MIN_CONSECUTIVE_EPOCHS
) -> np.ndarray:
    """True where sustained movement justifies forcing the epoch to WAKE."""
    counts = acc_epochs.counts
    if np.isnan(acc_epochs.threshold):
        return np.zeros(len(counts), dtype=bool)
    above = np.zeros(len(counts), dtype=bool)
    valid = ~np.isnan(counts)
    above[valid] = counts[valid] > acc_epochs.threshold

    # Keep only runs of >= min_consecutive consecutive epochs.
    mask = np.zeros_like(above)
    run_start = None
    for i, flag in enumerate(above):
        if flag and run_start is None:
            run_start = i
        elif not flag and run_start is not None:
            if i - run_start >= min_consecutive:
                mask[run_start:i] = True
            run_start = None
    if run_start is not None and len(above) - run_start >= min_consecutive:
        mask[run_start:] = True
    return mask


def apply_wake_override(hyp: Hypnogram, mask: np.ndarray) -> tuple[Hypnogram, int]:
    """Force sustained-movement epochs to WAKE; returns (copy, n changed).

    Only scored epochs are overridden — movement cannot conjure a score for
    an epoch the engine refused to stage.
    """
    if len(mask) != hyp.n_epochs:
        raise ValueError(
            f"Override mask has {len(mask)} epochs, hypnogram has {hyp.n_epochs}."
        )
    change = mask & hyp.scored_mask() & (hyp.stages != WAKE_CODE)
    n_changed = int(np.sum(change))
    if n_changed == 0:
        return hyp, 0
    stages = hyp.stages.copy()
    stages[change] = WAKE_CODE
    out = replace(
        hyp,
        stages=stages,
        probabilities=None,  # engine probabilities no longer describe the result
        notes=list(hyp.notes)
        + [
            f"{n_changed} epoch(s) re-scored to Wake by sustained accelerometer "
            "movement (see movement trace)."
        ],
    )
    return out, n_changed
