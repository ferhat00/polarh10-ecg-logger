"""Adapter for the optional external 5-class ECG staging tool.

The tool is `adammj/ecg-sleep-staging <https://github.com/adammj/ecg-sleep-staging>`_
("expert-level sleep scoring with ECG", 5-class AASM staging from raw
single-lead ECG, reported median κ≈0.73). It is **AGPL-3.0**, which is
incompatible with this repository's MIT license — so nothing from it is
imported, vendored, or copied. The separation is structural:

1. this adapter writes an ``input.h5`` file to the tool's *documented*
   interface,
2. runs the **user's own clone** with the **user's own Python environment**
   as a subprocess (``ECGLOG_SLEEP_EXTERNAL_DIR`` / ``_PYTHON``), and
3. parses the ``results.h5`` the tool writes.

Interface facts taken from the tool's README and requirements page
(cardiosomnography.com/requirements), pinned here:

* ``ecgs``: float array ``(epoch_count, 7680)`` — 30 s × 256 Hz epochs.
  Preprocessing: 0.5 Hz high-pass, mains notch (50/60 Hz), resample to
  256 Hz *after* filtering, median → 0, amplitude scaled so the 90th
  percentile of per-heartbeat peaks sits at ±0.5, clamped to [−1, 1].
* ``demographics``: ``[sex (0=female, 1=male), age/100]``. When the person
  profile lacks these, 0.5 is written for the missing value and a note says
  the model ran with defaulted demographics.
* ``midnight_offset``: recording start relative to local midnight, scaled
  to [−1, 1] (±12 h).
* Stage labels: ``0=Wake 1=N1 2=N2 3=N3 4=REM``.

The results dataset name is not documented; :func:`parse_results_h5`
accepts the first plausible dataset and otherwise reports exactly what the
file contained. All failure modes become engine notes — never session
errors — and the exchange directory is kept on failure for inspection.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import shutil
import subprocess
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import signal as sp_signal

from app.sleep.engines import EngineStatus
from app.sleep.stages import (
    EPOCH_LEN_S,
    UNSCORED,
    Hypnogram,
    StageVocab,
    make_epoch_grid,
)

ENGINE_KEY = "external-5class"
ENGINE_LABEL = "External 5-class deep net (adammj/ecg-sleep-staging)"

EXTERNAL_ACCURACY_NOTE = (
    "External deep network (adammj/ecg-sleep-staging): 5-class AASM staging "
    "from raw single-lead ECG, reported median kappa ~0.73 overall (per-stage "
    "medians: Wake 0.87, N1 0.33, N2 0.68, N3 0.63, REM 0.83) — run from "
    "your own AGPL-licensed clone, outside this app."
)

TARGET_FS_HZ = 256.0
EPOCH_SAMPLES = 7680  # 30 s × 256 Hz

#: Candidate dataset names for the (undocumented) results file.
RESULT_DATASET_CANDIDATES = ("stages", "predicted_stages", "predictions", "results")

#: 0=Wake 1=N1 2=N2 3=N3 4=REM → this app's AASM_5 (identical order).
_EXTERNAL_TO_VOCAB = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}


def status(config: Mapping) -> EngineStatus:
    """Availability from configuration — with the exact missing piece."""
    ext_dir = config.get("SLEEP_EXTERNAL_DIR")
    ext_python = config.get("SLEEP_EXTERNAL_PYTHON")
    if not ext_dir or not ext_python:
        return EngineStatus(
            key=ENGINE_KEY,
            label=ENGINE_LABEL,
            available=False,
            unavailable_reason=(
                "not configured — clone the AGPL tool separately and set "
                "ECGLOG_SLEEP_EXTERNAL_DIR and ECGLOG_SLEEP_EXTERNAL_PYTHON "
                "(see docs/SLEEP.md)."
            ),
        )
    problems = []
    if not Path(ext_dir).is_dir():
        problems.append(f"ECGLOG_SLEEP_EXTERNAL_DIR {ext_dir!r} is not a directory")
    elif not (Path(ext_dir) / "train.py").exists():
        problems.append(f"no train.py in ECGLOG_SLEEP_EXTERNAL_DIR {ext_dir!r}")
    if not Path(ext_python).exists():
        problems.append(f"ECGLOG_SLEEP_EXTERNAL_PYTHON {ext_python!r} does not exist")
    if importlib.util.find_spec("h5py") is None:
        problems.append("h5py not installed — pip install -r requirements-sleep.txt")
    if problems:
        return EngineStatus(
            key=ENGINE_KEY,
            label=ENGINE_LABEL,
            available=False,
            unavailable_reason="; ".join(problems) + ".",
        )
    return EngineStatus(key=ENGINE_KEY, label=ENGINE_LABEL, available=True)


def preprocess_ecg(
    ecg_mv: np.ndarray,
    fs_hz: float,
    peak_times_s: np.ndarray,
) -> np.ndarray:
    """Filter, resample, and scale the ECG to the tool's input contract."""
    nyq = fs_hz / 2.0
    sos = sp_signal.butter(4, 0.5 / nyq, btype="highpass", output="sos")
    x = sp_signal.sosfiltfilt(sos, np.asarray(ecg_mv, dtype=np.float64))
    for mains_hz in (50.0, 60.0):
        if mains_hz < 0.95 * nyq:
            b, a = sp_signal.iirnotch(mains_hz, Q=30.0, fs=fs_hz)
            x = sp_signal.filtfilt(b, a, x)

    frac = Fraction(TARGET_FS_HZ / fs_hz).limit_denominator(10000)
    x = sp_signal.resample_poly(x, frac.numerator, frac.denominator)

    x = x - np.median(x)

    # Per-heartbeat amplitude: |x| maximum in a ±0.4 s window around each
    # R peak (times are on the recording clock; convert to 256 Hz samples).
    if len(peak_times_s):
        half = int(0.4 * TARGET_FS_HZ)
        idx = np.round(np.asarray(peak_times_s) * TARGET_FS_HZ).astype(np.int64)
        idx = idx[(idx >= 0) & (idx < len(x))]
        peaks = np.array(
            [
                np.max(np.abs(x[max(0, i - half) : min(len(x), i + half)]))
                for i in idx
            ]
        )
        p90 = float(np.percentile(peaks, 90)) if len(peaks) else 0.0
    else:
        p90 = float(np.percentile(np.abs(x), 99))
    if p90 > 0:
        x = x * (0.5 / p90)
    return np.clip(x, -1.0, 1.0)


def export_input_h5(
    ecg_mv: np.ndarray,
    fs_hz: float,
    peak_times_s: np.ndarray,
    start_time: dt.datetime,
    age_years: float | None,
    sex: str | None,
    out_path: Path,
) -> tuple[int, list[str]]:
    """Write the tool's input file; returns (n complete epochs, notes)."""
    # Optional dependency (requirements-sleep.txt), guarded by status().
    import h5py  # pylint: disable=import-error

    notes: list[str] = []
    x = preprocess_ecg(ecg_mv, fs_hz, peak_times_s)
    n_epochs = len(x) // EPOCH_SAMPLES
    if n_epochs == 0:
        raise ValueError("Recording is shorter than one 30 s epoch at 256 Hz.")
    ecgs = x[: n_epochs * EPOCH_SAMPLES].reshape(n_epochs, EPOCH_SAMPLES)

    sex_value = {"female": 0.0, "male": 1.0}.get(sex or "", 0.5)
    age_value = (age_years / 100.0) if age_years is not None else 0.5
    if sex_value == 0.5 or (age_years is None):
        notes.append(
            "Age and/or sex missing from the person profile — the external "
            "model ran with defaulted demographics (0.5), which degrades its "
            "accuracy."
        )

    local = start_time.astimezone()
    seconds = local.hour * 3600 + local.minute * 60 + local.second
    # Signed offset from the nearest midnight, ±12 h → [-1, 1].
    if seconds > 43200:
        seconds -= 86400
    midnight_offset = seconds / 43200.0

    with h5py.File(out_path, "w") as f:
        f.create_dataset("ecgs", data=ecgs.astype(np.float32))
        f.create_dataset(
            "demographics",
            data=np.array([[sex_value], [age_value]], dtype=np.float32),
        )
        f.create_dataset("midnight_offset", data=np.float32(midnight_offset))
    return n_epochs, notes


def parse_results_h5(path: Path, n_epochs: int) -> np.ndarray:
    """Stage codes from the tool's results file (AASM_5 order)."""
    import h5py  # pylint: disable=import-error

    with h5py.File(path, "r") as f:
        available = list(f.keys())
        name = next((n for n in RESULT_DATASET_CANDIDATES if n in f), None)
        if name is None:
            raise ValueError(
                f"results file has no recognised stage dataset; found: {available}"
            )
        raw = np.asarray(f[name]).reshape(-1)
    if len(raw) != n_epochs:
        raise ValueError(
            f"results file has {len(raw)} epochs, expected {n_epochs}"
        )
    stages = np.array(
        [_EXTERNAL_TO_VOCAB.get(int(s), UNSCORED) for s in raw], dtype=np.int8
    )
    return stages


def stage_external(
    ecg_mv: np.ndarray,
    fs_hz: float,
    peak_times_s: np.ndarray,
    start_time: dt.datetime,
    duration_s: float,
    age_years: float | None,
    sex: str | None,
    stored_path: str,
    config: Mapping,
) -> Hypnogram:
    """Export → subprocess → parse. Any failure raises (the orchestrator
    converts it into an unavailable-engine status + note)."""
    grid = make_epoch_grid(duration_s)
    n_grid = len(grid)

    exchange = Path(f"{stored_path}.sleep_external")
    exchange.mkdir(parents=True, exist_ok=True)
    input_path = exchange / "input.h5"
    results_path = exchange / "results.h5"

    try:
        n_epochs, notes = export_input_h5(
            ecg_mv, fs_hz, peak_times_s, start_time, age_years, sex, input_path
        )

        proc = subprocess.run(  # noqa: S603 - paths come from the user's own config
            [str(config["SLEEP_EXTERNAL_PYTHON"]), "train.py", str(input_path)],
            cwd=str(config["SLEEP_EXTERNAL_DIR"]),
            capture_output=True,
            text=True,
            timeout=float(config.get("SLEEP_EXTERNAL_TIMEOUT_S", 1800)),
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-500:]
            raise RuntimeError(
                f"external scorer exited with {proc.returncode}: …{tail} "
                f"(exchange dir kept at {exchange})"
            )
        # The tool documents writing results.h5 "in the same folder".
        found = results_path if results_path.exists() else next(
            iter(sorted(exchange.glob("results*.h5"))), None
        )
        if found is None:
            raise RuntimeError(
                f"external scorer wrote no results.h5 (exchange dir kept at {exchange})"
            )
        scored = parse_results_h5(found, n_epochs)
    except Exception:
        # Keep the exchange dir for debugging; the note carries its path.
        raise
    else:
        shutil.rmtree(exchange, ignore_errors=True)

    stages = np.full(n_grid, UNSCORED, dtype=np.int8)
    n_common = min(n_grid, len(scored))
    stages[:n_common] = scored[:n_common]
    if n_common < n_grid:
        notes.append(
            f"The final {n_grid - n_common} epoch(s) were shorter than a full "
            "30 s at 256 Hz and are unscored."
        )

    return Hypnogram(
        engine=ENGINE_KEY,
        engine_label=ENGINE_LABEL,
        vocab=StageVocab.AASM_5,
        epoch_len_s=EPOCH_LEN_S,
        epoch_start_s=grid,
        stages=stages,
        probabilities=None,
        accuracy_note=EXTERNAL_ACCURACY_NOTE,
        notes=notes,
    )
