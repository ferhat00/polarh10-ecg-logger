"""Synthetic ECG builders shared by the pipeline tests.

Signals are constructed from known RR series so every expectation is
checkable; no real health data is used anywhere in the test suite.
"""

from __future__ import annotations

import numpy as np

FS_HZ = 130.03


def beat_waveform(t_rel_s: np.ndarray, r_amp_mv: float = 0.60) -> np.ndarray:
    """A stylised PQRST complex (mV) as a sum of Gaussians around R."""

    def g(center_s: float, amp_mv: float, width_s: float) -> np.ndarray:
        return amp_mv * np.exp(-0.5 * ((t_rel_s - center_s) / width_s) ** 2)

    return (
        g(-0.170, 0.08, 0.030)
        + g(-0.030, -0.08, 0.012)
        + g(0.000, r_amp_mv, 0.012)
        + g(0.035, -0.15, 0.012)
        + g(0.250, 0.22, 0.060)
    )


def build_ecg_from_rr(
    rr_ms: np.ndarray,
    fs_hz: float = FS_HZ,
    noise_mv: float = 0.01,
    seed: int = 1,
    beat_amps: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(time_s, ecg_mv, beat_times_s) for a known RR series.

    ``beat_amps`` overrides the R amplitude of specific beats (by index into
    the beat train) — used to create morphology outliers.
    """
    beat_times = np.concatenate(([0.5], 0.5 + np.cumsum(rr_ms) / 1000.0))
    total_s = beat_times[-1] + 1.0
    n = int(total_s * fs_hz)
    t = np.arange(n) / fs_hz

    rng = np.random.default_rng(seed)
    ecg = rng.normal(0.0, noise_mv, n)
    for k, bt in enumerate(beat_times):
        amp = (beat_amps or {}).get(k, 0.60)
        lo = max(0, int((bt - 0.4) * fs_hz))
        hi = min(n, int((bt + 0.5) * fs_hz))
        ecg[lo:hi] += beat_waveform(t[lo:hi] - bt, r_amp_mv=amp)
    return t, ecg, beat_times


def peak_indices_for(beat_times_s: np.ndarray, fs_hz: float = FS_HZ) -> np.ndarray:
    """Sample indices of the known beat times (ground-truth 'detection')."""
    return np.round(beat_times_s * fs_hz).astype(np.int64)


def make_acc_csv_bytes(
    duration_s: float,
    fs_hz: float = 50.0,
    movement_bursts: list[tuple[float, float, float]] | None = None,
    start=None,
    delimiter: str = ";",
    decimal: str = ".",
    seed: int = 3,
) -> bytes:
    """A Polar Sensor Logger ACC export: still = gravity vector + noise.

    ``movement_bursts`` is a list of ``(t0_s, t1_s, amplitude_mg)`` spans in
    which an oscillation in the human-movement band (~1.5 Hz) is added, so
    tests know exactly which epochs contain movement.
    """
    import datetime as dt

    start = start or dt.datetime(2026, 8, 14, 10, 0, tzinfo=dt.UTC)
    n = int(duration_s * fs_hz)
    t = np.arange(n) / fs_hz
    rng = np.random.default_rng(seed)

    # Gravity mostly on Z (lying down), a little on X, plus sensor noise.
    x = 120.0 + rng.normal(0.0, 2.0, n)
    y = rng.normal(0.0, 2.0, n)
    z = 990.0 + rng.normal(0.0, 2.0, n)
    for t0, t1, amp in movement_bursts or []:
        span = (t >= t0) & (t < t1)
        wobble = amp * np.sin(2 * np.pi * 1.5 * t[span])
        x[span] += wobble
        y[span] += 0.7 * amp * np.sin(2 * np.pi * 2.1 * t[span])
        z[span] += 0.5 * wobble

    start_ns = int(start.timestamp() * 1e9)
    time_ns = start_ns + np.round(t * 1e9).astype(np.int64)

    def fmt(v: float) -> str:
        s = f"{v:.3f}"
        return s.replace(".", ",") if decimal == "," else s

    header = delimiter.join(
        ["Phone timestamp", "sensor timestamp [ns]", "X [mg]", "Y [mg]", "Z [mg]"]
    )
    lines = [header]
    for i in range(n):
        lines.append(
            delimiter.join(
                ["", str(time_ns[i]), fmt(x[i]), fmt(y[i]), fmt(z[i])]
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")
