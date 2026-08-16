"""Generate the committed synthetic test fixture.

Builds a Polar-shaped CSV from a *known* RR series so the expected metrics are
analytically checkable, instead of committing real health data. Deterministic:
fixed seed, fixed start time. Regenerate with:

    .venv/Scripts/python scripts/make_fixture.py

The RR series is 800 ms with a 0.1 Hz (Mayer-wave-band) sinusoidal modulation
of ±30 ms plus seeded Gaussian jitter of σ=10 ms. Analytic expectations:
SDNN ≈ sqrt((30/√2)² + 10²) ≈ 23.5 ms; mean RR = 800 ms; mean HR = 75 bpm.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
START = dt.datetime(2026, 8, 14, 10, 0, tzinfo=dt.UTC)
FS_HZ = 130.03
DURATION_S = 300.0
SEED = 42

# Known RR series parameters.
RR_MEAN_MS = 800.0
RR_LF_AMP_MS = 30.0
RR_LF_HZ = 0.1
RR_JITTER_MS = 10.0


def known_rr_series() -> np.ndarray:
    """The ground-truth RR series (ms) the fixture is built from."""
    rng = np.random.default_rng(SEED)
    rr = []
    t = 0.0
    while t < DURATION_S:
        rr_ms = (
            RR_MEAN_MS
            + RR_LF_AMP_MS * math.sin(2 * math.pi * RR_LF_HZ * t)
            + rng.normal(0.0, RR_JITTER_MS)
        )
        rr.append(rr_ms)
        t += rr_ms / 1000.0
    return np.array(rr)


def _beat_waveform(t_rel_s: np.ndarray) -> np.ndarray:
    """A stylised PQRST complex (mV) as a sum of Gaussians around the R time."""

    def g(center_s: float, amp_mv: float, width_s: float) -> np.ndarray:
        return amp_mv * np.exp(-0.5 * ((t_rel_s - center_s) / width_s) ** 2)

    return (
        g(-0.170, 0.08, 0.030)  # P
        + g(-0.030, -0.08, 0.012)  # Q
        + g(0.000, 0.60, 0.012)  # R
        + g(0.035, -0.15, 0.012)  # S
        + g(0.250, 0.22, 0.060)  # T
    )


def build_ecg(rr_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample times (s), ECG (mV), and R-peak times (s) for the RR series."""
    beat_times = np.concatenate(([0.5], 0.5 + np.cumsum(rr_ms) / 1000.0))
    total_s = beat_times[-1] + 1.0
    n = int(total_s * FS_HZ)
    t = np.arange(n) / FS_HZ

    rng = np.random.default_rng(SEED + 1)
    ecg = rng.normal(0.0, 0.01, n)  # electrode noise floor
    for bt in beat_times:
        lo = max(0, int((bt - 0.4) * FS_HZ))
        hi = min(n, int((bt + 0.5) * FS_HZ))
        ecg[lo:hi] += _beat_waveform(t[lo:hi] - bt)
    return t, ecg, beat_times


def write_csv(path: Path) -> None:
    rr_ms = known_rr_series()
    t, ecg, beat_times = build_ecg(rr_ms)
    start_ns = int(START.timestamp() * 1e9)

    # Device streams, mimicking the ragged real export: the device's own rr
    # (ms, rounded) lands on the row nearest each beat; hr (1 Hz) on the row
    # nearest each whole second, skipping rr rows.
    rr_row: dict[int, int] = {}
    for k, bt in enumerate(beat_times[1:]):
        rr_row[int(round(bt * FS_HZ))] = int(round(rr_ms[k]))
    hr_row: dict[int, int] = {}
    inst_hr = 60000.0 / np.interp(np.arange(0, t[-1], 1.0), beat_times[1:], rr_ms)
    for sec, hr in enumerate(inst_hr):
        i = int(round(sec * FS_HZ))
        if i not in rr_row:
            hr_row[i] = int(round(hr))

    lines = ["time,ecg,hr,rr,marker"]
    for i, (ti, v) in enumerate(zip(t, ecg, strict=True)):
        fields = [str(start_ns + round(ti * 1e9)), f"{v:.4f}"]
        if i in rr_row:
            fields += ["", str(rr_row[i])]
        elif i in hr_row:
            fields.append(str(hr_row[i]))
        lines.append(",".join(fields))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {path} ({len(lines) - 1} rows, {len(rr_ms)} beats)")
    print(
        f"Known series: mean RR {np.mean(rr_ms):.2f} ms, "
        f"SDNN {np.std(rr_ms, ddof=1):.2f} ms, "
        f"RMSSD {np.sqrt(np.mean(np.diff(rr_ms) ** 2)):.2f} ms"
    )


if __name__ == "__main__":
    write_csv(FIXTURE_DIR / "synthetic_supine.csv")
