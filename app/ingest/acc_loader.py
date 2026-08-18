"""Loader for Polar Sensor Logger accelerometer (ACC) exports.

The verified shape is the Polar Sensor Logger per-stream export::

    Phone timestamp;sensor timestamp [ns];X [mg];Y [mg];Z [mg]

with a semicolon delimiter and, in some locales, decimal commas. A plain
``time,x,y,z`` shape (nanosecond timestamps, mg values) is also accepted.
Everything else raises :class:`~app.ingest.exceptions.LoaderError` quoting
what was observed versus what was expected — the interactive mapping-form
round-trip is wired to the ECG file's processing status and deliberately
does **not** extend to this optional companion file; an honest error beats
a guessed unit.

Detection mirrors the ECG loader:

* the timestamp epoch (Unix vs Polar-2000) is chosen by the shared
  plausible-date rule (:func:`app.ingest.loader.plausible_epoch_interpretations`),
* the sampling rate is derived from the timestamps (H10 ACC streams run at
  25/50/100/200 Hz nominal),
* decimal commas are handled — with a ``;`` delimiter they are unambiguous,
  so they are normalised with a note rather than raising.

Parsing uses pandas (already a core dependency): ACC files are regular
(never ragged) and an 8 h night at 50 Hz is ~1.4 M rows, where a Python
row loop would be the slowest step of the whole pipeline.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from app.ingest.exceptions import LoaderError
from app.ingest.loader import plausible_epoch_interpretations

#: Minimum usable samples (a few seconds at the slowest ACC rate).
MIN_ACC_SAMPLES = 100

#: Plausible ACC sampling-rate window (H10 streams 25–200 Hz nominal).
ACC_FS_MIN_HZ = 10.0
ACC_FS_MAX_HZ = 500.0

_EXPECTED_SHAPE = (
    "'Phone timestamp;sensor timestamp [ns];X [mg];Y [mg];Z [mg]' "
    "(Polar Sensor Logger ACC export) or 'time,x,y,z' with nanosecond "
    "timestamps and milli-g values"
)


@dataclass
class LoadedAcc:
    """A parsed, epoch-resolved accelerometer recording."""

    #: Sample times in seconds from the first ACC sample.
    time_s: np.ndarray
    #: Wall-clock start of the ACC stream (UTC), from its own timestamps.
    start_time: dt.datetime
    x_mg: np.ndarray
    y_mg: np.ndarray
    z_mg: np.ndarray
    #: Derived from the timestamps, never hardcoded.
    sampling_rate_hz: float
    epoch_detected: str  # "unix" | "polar2000"
    notes: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0]) if len(self.time_s) > 1 else 0.0


def _identify_columns(names: list[str]) -> tuple[int, int, int, int, float]:
    """(time_idx, x_idx, y_idx, z_idx, to_mg factor) or raise LoaderError."""
    lower = [n.strip().lower() for n in names]

    def find(predicate) -> int | None:
        hits = [i for i, n in enumerate(lower) if predicate(n)]
        return hits[0] if len(hits) == 1 else None

    time_idx = find(lambda n: "sensor timestamp" in n)
    if time_idx is None:
        time_idx = find(lambda n: n in ("time", "timestamp", "time [ns]"))

    def axis(ax: str) -> tuple[int | None, float]:
        i = find(lambda n: n in (ax, f"{ax} [mg]"))
        if i is not None:
            return i, 1.0
        i = find(lambda n: n == f"{ax} [g]")
        if i is not None:
            return i, 1000.0
        return None, 1.0

    x_idx, fx = axis("x")
    y_idx, fy = axis("y")
    z_idx, fz = axis("z")

    if None in (time_idx, x_idx, y_idx, z_idx) or len({fx, fy, fz}) != 1:
        raise LoaderError(
            "The accelerometer file's columns do not match the expected shape. "
            f"Found columns: {', '.join(names)}. Expected {_EXPECTED_SHAPE}."
        )
    return time_idx, x_idx, y_idx, z_idx, fx


def load_polar_acc_csv(path: str | Path, now: dt.datetime | None = None) -> LoadedAcc:
    """Load and validate a Polar accelerometer export.

    Raises :class:`LoaderError` with observed-vs-expected detail on any
    mismatch — never guesses a unit or epoch.
    """
    path = Path(path)
    now = now or dt.datetime.now(dt.UTC)
    notes: list[str] = []

    try:
        with path.open(encoding="utf-8-sig", errors="replace") as fh:
            header_line = fh.readline().strip()
    except OSError as exc:
        raise LoaderError(f"Could not read {path.name}: {exc}") from exc
    if not header_line:
        raise LoaderError(f"{path.name} is empty.")

    counts = {d: header_line.count(d) for d in (";", ",", "\t")}
    delimiter = max(counts, key=counts.get)
    if counts[delimiter] == 0:
        raise LoaderError(
            f"Could not determine the delimiter of {path.name}. "
            f"Header line: {header_line[:120]!r}. Expected {_EXPECTED_SHAPE}."
        )

    names = [n.strip() for n in header_line.split(delimiter)]
    if names and names[-1] == "":
        names = names[:-1]
    time_idx, x_idx, y_idx, z_idx, to_mg = _identify_columns(names)

    # Decimal commas inside a ';'-delimited file are unambiguous: normalise
    # with a note. (Inside a ','-delimited file they cannot occur intact.)
    decimal = "."
    if delimiter in (";", "\t"):
        with path.open(encoding="utf-8-sig", errors="replace") as fh:
            fh.readline()
            sample = [fh.readline() for _ in range(5)]
        if any("," in ln for ln in sample if ln):
            decimal = ","
            notes.append("Decimal commas detected and normalised.")

    try:
        frame = pd.read_csv(
            path,
            sep=delimiter,
            decimal=decimal,
            usecols=[names[i] for i in (time_idx, x_idx, y_idx, z_idx)],
            encoding="utf-8-sig",
            skip_blank_lines=True,
        )
    except (ValueError, pd.errors.ParserError) as exc:
        raise LoaderError(f"Could not parse {path.name}: {exc}") from exc

    time_col = frame[names[time_idx]]
    if not pd.api.types.is_numeric_dtype(time_col):
        raise LoaderError(
            f"The timestamp column {names[time_idx]!r} of {path.name} is not "
            "numeric — expected nanosecond integers."
        )
    axis_frames = [frame[names[i]] for i in (x_idx, y_idx, z_idx)]
    if not all(pd.api.types.is_numeric_dtype(c) for c in axis_frames):
        raise LoaderError(
            f"The X/Y/Z columns of {path.name} are not numeric — expected "
            "acceleration in milli-g."
        )

    keep = time_col.notna()
    for c in axis_frames:
        keep &= c.notna()
    time_ns = time_col[keep].to_numpy(dtype=np.int64)
    if len(time_ns) < MIN_ACC_SAMPLES:
        raise LoaderError(
            f"Only {len(time_ns)} usable accelerometer samples — too short "
            f"(minimum {MIN_ACC_SAMPLES})."
        )

    candidates = plausible_epoch_interpretations(int(time_ns[0]), now)
    if len(candidates) != 1:
        raise LoaderError(
            f"Could not determine the timestamp epoch of {path.name}: "
            f"first timestamp {int(time_ns[0])} lands in a plausible date window "
            f"under {len(candidates)} interpretation(s) (unix vs polar2000)."
        )
    epoch, start_time = next(iter(candidates.items()))
    notes.append(
        f"ACC epoch auto-detect: {epoch} interpretation lands at "
        f"{start_time:%Y-%m-%d %H:%M} (plausible)."
    )

    diffs = np.diff(time_ns)
    positive = diffs[diffs > 0]
    if len(positive) < MIN_ACC_SAMPLES - 1:
        raise LoaderError("Too few increasing ACC timestamps to derive a sampling rate.")
    fs = 1e9 / float(np.median(positive))
    if not (ACC_FS_MIN_HZ <= fs <= ACC_FS_MAX_HZ):
        raise LoaderError(
            f"Derived ACC sampling rate {fs:.1f} Hz is outside the plausible "
            f"range ({ACC_FS_MIN_HZ:g}–{ACC_FS_MAX_HZ:g} Hz) — the timestamp "
            "column may not be nanoseconds."
        )
    notes.append(f"ACC sampling rate derived from timestamps: {fs:.2f} Hz.")

    time_s = (time_ns - time_ns[0]) / 1e9
    return LoadedAcc(
        time_s=time_s,
        start_time=start_time,
        x_mg=axis_frames[0][keep].to_numpy(dtype=np.float64) * to_mg,
        y_mg=axis_frames[1][keep].to_numpy(dtype=np.float64) * to_mg,
        z_mg=axis_frames[2][keep].to_numpy(dtype=np.float64) * to_mg,
        sampling_rate_hz=fs,
        epoch_detected=epoch,
        notes=notes,
    )
