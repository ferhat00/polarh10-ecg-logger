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


#: Per-stage RR modulation parameters for the overnight synthesizer:
#: (HR delta above base bpm, LF sine amplitude ms @0.10 Hz,
#:  HF/RSA sine amplitude ms @0.25 Hz, white jitter σ ms).
#: The orderings mirror the staging literature: deep = vagal maximum
#: (huge RSA, tiny LF), REM/wake = LF dominance, wake = highest HR.
STAGE_RR_PARAMS: dict[str, tuple[float, float, float, float]] = {
    "wake": (15.0, 30.0, 8.0, 10.0),
    "light": (5.0, 25.0, 18.0, 6.0),
    "deep": (0.0, 8.0, 35.0, 4.0),
    "rem": (8.0, 40.0, 10.0, 8.0),
}

STAGE_PLAN_DEFAULT: list[tuple[str, float]] = [
    ("wake", 10),
    ("light", 30),
    ("deep", 30),
    ("light", 20),
    ("rem", 20),
    ("light", 25),
    ("deep", 15),
    ("rem", 25),
    ("wake", 5),
]


def overnight_rr(
    stage_plan: list[tuple[str, float]] | None = None,
    base_hr_bpm: float = 55.0,
    seed: int = 11,
    epoch_len_s: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rr_ms, t_s, true_stage_per_epoch) with stage-programmed modulation.

    ``stage_plan`` is a list of (stage, minutes). Truth labels use the
    4-class codes 0=wake 1=light 2=deep 3=rem on 30 s epochs.
    """
    stage_plan = stage_plan or STAGE_PLAN_DEFAULT
    codes = {"wake": 0, "light": 1, "deep": 2, "rem": 3}
    rng = np.random.default_rng(seed)

    boundaries: list[tuple[float, str]] = []
    t_edge = 0.0
    for stage, minutes in stage_plan:
        boundaries.append((t_edge, stage))
        t_edge += minutes * 60.0
    total_s = t_edge

    def stage_at(t: float) -> str:
        current = boundaries[0][1]
        for edge, stage in boundaries:
            if t >= edge:
                current = stage
            else:
                break
        return current

    beat_times: list[float] = []
    rr_list: list[float] = []
    t = 0.0
    while t < total_s:
        delta, a_lf, a_hf, jitter = STAGE_RR_PARAMS[stage_at(t)]
        rr_ms = (
            60000.0 / (base_hr_bpm + delta)
            + a_lf * np.sin(2 * np.pi * 0.10 * t)
            + a_hf * np.sin(2 * np.pi * 0.25 * t)
            + rng.normal(0.0, jitter)
        )
        t += rr_ms / 1000.0
        beat_times.append(t)
        rr_list.append(rr_ms)

    n_epochs = int(np.ceil(total_s / epoch_len_s))
    truth = np.array(
        [
            codes[stage_at((k + 0.5) * epoch_len_s)]
            for k in range(n_epochs)
        ],
        dtype=np.int8,
    )
    return np.array(rr_list), np.array(beat_times), truth


def rr_series_from(rr_ms: np.ndarray, t_s: np.ndarray):
    """A contiguous RRSeries directly from arrays (engine-level tests)."""
    from app.pipeline.rr import RRSeries

    return RRSeries(
        rr_ms=np.asarray(rr_ms, dtype=float),
        t_s=np.asarray(t_s, dtype=float),
        discontinuity=np.zeros(len(rr_ms), dtype=bool),
        n_dropped_excluded=0,
        n_dropped_ceiling=0,
        n_dropped_floor=0,
    )


def make_acc_csv_bytes(
    duration_s: float,
    fs_hz: float = 50.0,
    movement_bursts: list[tuple[float, float, float]] | None = None,
    start=None,
    delimiter: str = ";",
    decimal: str = ".",
    seed: int = 3,
    gravity: tuple[float, float, float] | None = None,
) -> bytes:
    """A Polar Sensor Logger ACC export: still = gravity vector + noise.

    ``movement_bursts`` is a list of ``(t0_s, t1_s, amplitude_mg)`` spans in
    which an oscillation in the human-movement band (~1.5 Hz) is added, so
    tests know exactly which epochs contain movement. ``gravity`` overrides
    the resting (x, y, z) gravity vector in mg — default is the historical
    "lying down" vector (mostly +Z with a little +X).
    """
    import datetime as dt

    start = start or dt.datetime(2026, 8, 14, 10, 0, tzinfo=dt.UTC)
    n = int(duration_s * fs_hz)
    t = np.arange(n) / fs_hz
    rng = np.random.default_rng(seed)

    # Gravity mostly on Z (lying down), a little on X, plus sensor noise.
    gx, gy, gz = gravity if gravity is not None else (120.0, 0.0, 990.0)
    x = gx + rng.normal(0.0, 2.0, n)
    y = gy + rng.normal(0.0, 2.0, n)
    z = gz + rng.normal(0.0, 2.0, n)
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
