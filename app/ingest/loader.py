"""Loader for Polar H10 ECG CSV exports (Polar Sensor Logger format).

Verified facts about the real export this loader is written against (measured,
not assumed — see the project README for the data-format contract):

* Header ``time,ecg,hr,rr,marker``; **rows are ragged** (most have 2 fields,
  some 3–5). Every row is explicitly padded to the header width rather than
  trusting a CSV library's raggedness handling.
* ``time`` is **nanoseconds** — but the epoch varies. Polar documents a
  2000-01-01 sensor epoch; real exports have been observed using the Unix
  epoch. The epoch is *detected* (plausible-date check, cross-checked against
  any date in the filename), never hardcoded.
* ``ecg`` may be **mV or µV** depending on exporter version. Auto-detected by
  amplitude: a 99th-percentile |ecg| above ~20 is only physical in µV
  (a 0.6 µV R-wave would be below any electrode noise floor). The branch taken
  is logged and surfaced in the report.
* The sampling rate is **derived from the timestamps** (real files run
  ~130.030 Hz, not 130.000) — assuming exactly 130 introduces systematic
  timing error.
* ``hr``/``rr`` are the device's own sparse streams, kept as a cross-check
  against our R-peak detection, never as the primary source.

Anything that doesn't match this shape raises
:class:`~app.ingest.exceptions.AmbiguousFormatError` with the questions that
need answering; the answers come back via :class:`FormatOverrides`.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.ingest.exceptions import AmbiguousFormatError, FormatQuestion, LoaderError

#: Canonical column names in the verified export, in order.
CANONICAL_COLUMNS = ("time", "ecg", "hr", "rr", "marker")

#: 2000-01-01T00:00:00Z in Unix seconds — Polar's documented sensor epoch.
POLAR_EPOCH_2000_UNIX_S = 946_684_800

#: Recordings older than this are treated as implausible for epoch detection.
PLAUSIBLE_EARLIEST = dt.datetime(2010, 1, 1, tzinfo=dt.UTC)

#: 99th-percentile |ecg| above this is only physical if the unit is µV.
UV_DETECT_THRESHOLD = 20.0
#: Below this the amplitude is unambiguously mV territory. Between the two
#: bounds (impossible both as mV chest-strap signal and as µV R-waves) the
#: loader asks instead of guessing.
MV_DETECT_UPPER = 5.0

#: Minimum number of ECG samples for a recording to be analysable at all.
MIN_SAMPLES = 100

_DELIMITERS = (",", ";", "\t")
_DECIMAL_COMMA_RE = re.compile(r"^-?\d+,\d+$")
_FILENAME_DATE_RE = re.compile(r"(20\d{2})[-_.]?(\d{2})[-_.]?(\d{2})")


@dataclass(frozen=True)
class FormatOverrides:
    """User answers to :class:`FormatQuestion`s, applied on re-load."""

    delimiter: str | None = None
    decimal: str | None = None  # "." or ","
    #: Maps observed header names to canonical names ("Timestamp" -> "time").
    column_map: dict[str, str] | None = None
    ecg_unit: str | None = None  # "mV" | "uV"
    epoch: str | None = None  # "unix" | "polar2000"


@dataclass
class LoadedRecording:
    """A parsed, unit- and epoch-resolved recording, ready for the pipeline."""

    #: Sample times in seconds from the first sample.
    time_s: np.ndarray
    #: ECG in millivolts (converted if the file was µV).
    ecg_mv: np.ndarray
    #: Wall-clock start of the recording (UTC), from the CSV timestamps.
    start_time: dt.datetime
    #: Derived from the timestamps — never hardcoded.
    sampling_rate_hz: float
    #: Standard deviation of the sample interval, in milliseconds.
    sample_interval_std_ms: float

    #: Device-side sparse streams (cross-checks, not primary sources).
    device_hr_time_s: np.ndarray
    device_hr_bpm: np.ndarray
    device_rr_time_s: np.ndarray
    device_rr_ms: np.ndarray
    #: (time_s, text) for any populated marker fields.
    markers: list[tuple[float, str]]

    #: Detection provenance — stored on the session and shown in the report.
    ecg_unit_detected: str  # "mV" | "uV"
    epoch_detected: str  # "unix" | "polar2000"
    #: Evidence for the unit decision.
    p99_abs_amplitude: float
    #: Human-readable log of every detection decision taken.
    notes: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0]) if len(self.time_s) > 1 else 0.0


def load_polar_csv(
    path: str | Path,
    overrides: FormatOverrides | None = None,
    now: dt.datetime | None = None,
) -> LoadedRecording:
    """Load, validate, and normalise a Polar H10 ECG export.

    Parameters
    ----------
    path:
        CSV file on local disk.
    overrides:
        Answers to a previous :class:`AmbiguousFormatError`'s questions.
    now:
        Injectable "current time" for the epoch plausibility window (tests).

    Raises
    ------
    AmbiguousFormatError
        If the file shape, units, or epoch cannot be determined safely.
    LoaderError
        If the file is unusable (empty, too short, unreadable).
    """
    path = Path(path)
    overrides = overrides or FormatOverrides()
    now = now or dt.datetime.now(dt.UTC)
    notes: list[str] = []

    lines = _read_lines(path)
    delimiter = _detect_delimiter(lines[0], overrides, notes)
    columns = _parse_header(lines[0], delimiter, overrides)
    rows = _pad_rows(lines[1:], delimiter, len(columns))
    _check_decimal_separator(rows, columns, delimiter, overrides)

    parsed = _parse_columns(rows, columns, overrides, notes)
    time_ns = parsed["time_ns"]
    ecg_raw = parsed["ecg"]

    if len(time_ns) < MIN_SAMPLES:
        raise LoaderError(
            f"Only {len(time_ns)} usable samples — too short to analyse "
            f"(minimum {MIN_SAMPLES})."
        )

    epoch, start_time = _detect_epoch(time_ns[0], path.name, overrides, now, notes)
    ecg_mv, unit, p99 = _detect_unit(ecg_raw, overrides, notes)
    fs_hz, interval_std_ms = _derive_sampling_rate(time_ns, notes)

    time_s = (time_ns - time_ns[0]) / 1e9

    return LoadedRecording(
        time_s=time_s,
        ecg_mv=ecg_mv,
        start_time=start_time,
        sampling_rate_hz=fs_hz,
        sample_interval_std_ms=interval_std_ms,
        device_hr_time_s=parsed["hr_time_s"],
        device_hr_bpm=parsed["hr_bpm"],
        device_rr_time_s=parsed["rr_time_s"],
        device_rr_ms=parsed["rr_ms"],
        markers=parsed["markers"],
        ecg_unit_detected=unit,
        epoch_detected=epoch,
        p99_abs_amplitude=p99,
        notes=notes,
    )


# --------------------------------------------------------------------------
# Reading and structural checks
# --------------------------------------------------------------------------


def _read_lines(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        raise LoaderError(f"Could not read {path.name}: {exc}") from exc
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        raise LoaderError(f"{path.name} is empty.")
    if len(lines) < 2:
        raise LoaderError(f"{path.name} has a header but no data rows.")
    return lines


def _detect_delimiter(header_line: str, overrides: FormatOverrides, notes: list[str]) -> str:
    if overrides.delimiter:
        notes.append(f"Delimiter {overrides.delimiter!r} set by user override.")
        return overrides.delimiter
    counts = {d: header_line.count(d) for d in _DELIMITERS}
    present = [d for d, c in counts.items() if c > 0]
    if len(present) == 1:
        if present[0] != ",":
            # Non-comma delimiter is a departure from the verified shape:
            # surface it rather than silently accepting.
            raise AmbiguousFormatError(
                "The file does not use the expected comma delimiter.",
                [
                    FormatQuestion(
                        key="delimiter",
                        question="Which character separates the columns?",
                        observed=f"Header line: {header_line[:120]!r}",
                        options=(",", ";", "tab"),
                    )
                ],
            )
        return ","
    raise AmbiguousFormatError(
        "Could not determine the column delimiter.",
        [
            FormatQuestion(
                key="delimiter",
                question="Which character separates the columns?",
                observed=f"Header line: {header_line[:120]!r}",
                options=(",", ";", "tab"),
            )
        ],
    )


def _parse_header(
    header_line: str, delimiter: str, overrides: FormatOverrides
) -> list[str]:
    raw_names = [c.strip() for c in header_line.split(delimiter)]
    if raw_names and raw_names[-1] == "":
        raw_names = raw_names[:-1]  # tolerate a trailing delimiter

    if raw_names and _looks_numeric(raw_names[0]):
        raise AmbiguousFormatError(
            "The first line looks like data, not a header row.",
            [
                FormatQuestion(
                    key="columns",
                    question=(
                        "The file has no header row. Which columns are present, in order? "
                        "The expected shape is time,ecg,hr,rr,marker."
                    ),
                    observed=f"First line: {header_line[:120]!r}",
                )
            ],
        )

    column_map = {k.lower(): v for k, v in (overrides.column_map or {}).items()}
    names = [column_map.get(n.lower(), n.lower()) for n in raw_names]

    unrecognised = [n for n in names if n not in CANONICAL_COLUMNS]
    if unrecognised or names[:2] != ["time", "ecg"]:
        raise AmbiguousFormatError(
            "The file's columns do not match the expected Polar export shape.",
            [
                FormatQuestion(
                    key="columns",
                    question=(
                        "Map each of the file's columns to one of: time, ecg, hr, rr, "
                        "marker (or mark it as ignored)."
                    ),
                    observed=f"Found columns: {', '.join(raw_names)}",
                    options=CANONICAL_COLUMNS,
                )
            ],
        )
    return names


def _pad_rows(data_lines: list[str], delimiter: str, width: int) -> list[list[str]]:
    """Split every row and pad it to the header width explicitly.

    Rows in real exports are ragged (2–5 fields). Rather than trusting a CSV
    parser's raggedness handling, every row is padded with empty strings; a
    row *wider* than the header is a shape violation and raises.
    """
    rows: list[list[str]] = []
    for i, line in enumerate(data_lines, start=2):
        fields = [f.strip() for f in line.split(delimiter)]
        if len(fields) > width:
            raise AmbiguousFormatError(
                f"Row {i} has {len(fields)} fields but the header declares {width}.",
                [
                    FormatQuestion(
                        key="columns",
                        question=(
                            "Some rows have more fields than the header declares. "
                            "What do the extra fields contain?"
                        ),
                        observed=f"Row {i}: {line[:120]!r}",
                    )
                ],
            )
        fields.extend([""] * (width - len(fields)))
        rows.append(fields)
    return rows


def _check_decimal_separator(
    rows: list[list[str]],
    columns: list[str],
    delimiter: str,
    overrides: FormatOverrides,
) -> None:
    """Detect decimal-comma locales (e.g. ``-0,125`` with a ``;`` delimiter)."""
    if overrides.decimal == ",":
        return  # caller has told us; _parse_columns will normalise
    if delimiter == ",":
        return  # a decimal comma cannot survive a comma delimiter intact
    ecg_idx = columns.index("ecg")
    sample = [r[ecg_idx] for r in rows[:200] if r[ecg_idx]]
    if sample and any(_DECIMAL_COMMA_RE.match(v) for v in sample):
        raise AmbiguousFormatError(
            "The file appears to use decimal commas.",
            [
                FormatQuestion(
                    key="decimal",
                    question="Which decimal separator does the file use?",
                    observed=f"Example ECG values: {', '.join(sample[:3])}",
                    options=(".", ","),
                )
            ],
        )


def _looks_numeric(token: str) -> bool:
    try:
        float(token.replace(",", "."))
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------------
# Column parsing
# --------------------------------------------------------------------------


def _parse_columns(
    rows: list[list[str]],
    columns: list[str],
    overrides: FormatOverrides,
    notes: list[str],
) -> dict:
    idx = {name: columns.index(name) for name in columns}
    decimal_comma = overrides.decimal == ","

    def to_float(token: str) -> float:
        if decimal_comma:
            token = token.replace(",", ".")
        return float(token)

    time_ns: list[int] = []
    ecg: list[float] = []
    hr_rows: list[tuple[int, float]] = []  # (row index into kept samples, bpm)
    rr_rows: list[tuple[int, float]] = []
    markers: list[tuple[int, str]] = []
    bad_rows = 0
    first_bad: str | None = None

    for fields in rows:
        try:
            t = int(fields[idx["time"]])
            e = to_float(fields[idx["ecg"]])
        except (ValueError, IndexError):
            bad_rows += 1
            if first_bad is None:
                first_bad = ",".join(fields)[:120]
            continue
        row_i = len(time_ns)
        time_ns.append(t)
        ecg.append(e)

        if "hr" in idx and fields[idx["hr"]]:
            try:
                hr_rows.append((row_i, to_float(fields[idx["hr"]])))
            except ValueError:
                bad_rows += 1
        if "rr" in idx and fields[idx["rr"]]:
            try:
                rr_rows.append((row_i, to_float(fields[idx["rr"]])))
            except ValueError:
                bad_rows += 1
        if "marker" in idx and fields[idx["marker"]]:
            markers.append((row_i, fields[idx["marker"]]))

    total = len(rows)
    if total and bad_rows / total > 0.001:
        raise AmbiguousFormatError(
            f"{bad_rows} of {total} rows could not be parsed as numbers.",
            [
                FormatQuestion(
                    key="decimal",
                    question=(
                        "Many rows are not parseable — is the decimal separator or "
                        "delimiter different from the expected format?"
                    ),
                    observed=f"First unparseable row: {first_bad!r}",
                )
            ],
        )
    if bad_rows:
        notes.append(f"Skipped {bad_rows} unparseable row(s) (first: {first_bad!r}).")

    time_arr = np.array(time_ns, dtype=np.int64)
    t0 = time_arr[0] if len(time_arr) else 0
    to_s = lambda i: float((time_arr[i] - t0) / 1e9)  # noqa: E731

    hr_count, rr_count = len(hr_rows), len(rr_rows)
    notes.append(
        f"Parsed {len(time_arr)} samples; device streams: {hr_count} hr values, "
        f"{rr_count} rr values, {len(markers)} marker(s)."
    )

    return {
        "time_ns": time_arr,
        "ecg": np.array(ecg, dtype=np.float64),
        "hr_time_s": np.array([to_s(i) for i, _ in hr_rows]),
        "hr_bpm": np.array([v for _, v in hr_rows]),
        "rr_time_s": np.array([to_s(i) for i, _ in rr_rows]),
        "rr_ms": np.array([v for _, v in rr_rows]),
        "markers": [(to_s(i), text) for i, text in markers],
    }


# --------------------------------------------------------------------------
# Epoch, unit, and sampling-rate detection
# --------------------------------------------------------------------------


def _candidate_datetime(first_ns: int, epoch: str) -> dt.datetime | None:
    """Interpret a nanosecond timestamp under the given epoch, if representable."""
    seconds = first_ns / 1e9
    if epoch == "polar2000":
        seconds += POLAR_EPOCH_2000_UNIX_S
    try:
        return dt.datetime.fromtimestamp(seconds, tz=dt.UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _filename_date(filename: str) -> dt.date | None:
    for m in _FILENAME_DATE_RE.finditer(filename):
        try:
            return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue
    return None


def _detect_epoch(
    first_ns: int,
    filename: str,
    overrides: FormatOverrides,
    now: dt.datetime,
    notes: list[str],
) -> tuple[str, dt.datetime]:
    """Choose between Unix-epoch and Polar-2000-epoch nanosecond timestamps.

    Polar's documentation describes a 2000-01-01 sensor epoch, but real exports
    have been observed using the Unix epoch — so neither is hardcoded. The
    interpretation that lands in a plausible date window (2010 .. now+2 days)
    wins; a date embedded in the filename is used as a cross-check and as the
    tie-breaker.
    """
    if overrides.epoch:
        chosen = _candidate_datetime(first_ns, overrides.epoch)
        if chosen is None:
            raise LoaderError(
                f"Timestamps are not representable under the {overrides.epoch!r} epoch."
            )
        notes.append(
            f"Epoch {overrides.epoch!r} set by user override → starts {chosen:%Y-%m-%d %H:%M}."
        )
        return overrides.epoch, chosen

    latest = now + dt.timedelta(days=2)
    candidates: dict[str, dt.datetime] = {}
    for epoch in ("unix", "polar2000"):
        candidate = _candidate_datetime(first_ns, epoch)
        if candidate is not None and PLAUSIBLE_EARLIEST <= candidate <= latest:
            candidates[epoch] = candidate

    file_date = _filename_date(filename)

    if len(candidates) == 1:
        epoch, start = next(iter(candidates.items()))
        notes.append(
            f"Epoch auto-detect: {epoch} interpretation lands at "
            f"{start:%Y-%m-%d %H:%M} (plausible); "
            + (
                "no date found in filename to cross-check."
                if file_date is None
                else f"filename date {file_date} "
                + (
                    "agrees."
                    if abs((start.date() - file_date).days) <= 1
                    else f"differs by {abs((start.date() - file_date).days)} day(s) — "
                    "the filename date may be an export date rather than the recording date."
                )
            )
        )
        return epoch, start

    if len(candidates) == 2 and file_date is not None:
        # Extremely unlikely (interpretations are ~30 years apart) but if both
        # are plausible, let the filename decide.
        for epoch, start in candidates.items():
            if abs((start.date() - file_date).days) <= 1:
                notes.append(
                    f"Epoch auto-detect: both interpretations plausible; filename date "
                    f"{file_date} matches {epoch} → {start:%Y-%m-%d %H:%M}."
                )
                return epoch, start

    unix_dt = _candidate_datetime(first_ns, "unix")
    polar_dt = _candidate_datetime(first_ns, "polar2000")
    unix_desc = f"{unix_dt:%Y-%m-%d %H:%M}" if unix_dt else "unrepresentable"
    polar_desc = f"{polar_dt:%Y-%m-%d %H:%M}" if polar_dt else "unrepresentable"
    raise AmbiguousFormatError(
        "Could not determine the timestamp epoch.",
        [
            FormatQuestion(
                key="epoch",
                question="When did this recording actually start?",
                observed=(
                    f"First timestamp {first_ns}: as Unix-epoch nanoseconds → {unix_desc}; "
                    f"as 2000-epoch nanoseconds → {polar_desc}. Neither lands in a "
                    "plausible date window."
                ),
                options=("unix", "polar2000"),
            )
        ],
    )


def _detect_unit(
    ecg_raw: np.ndarray, overrides: FormatOverrides, notes: list[str]
) -> tuple[np.ndarray, str, float]:
    """Resolve mV vs µV by amplitude; convert to mV.

    A chest-strap R-wave is roughly 0.5–3 mV. If the 99th-percentile absolute
    value exceeds ~20 the numbers can only be µV (0.6 µV R-waves are below any
    electrode noise floor); clearly small values are mV. The zone between is
    physical under neither reading, so the loader asks.
    """
    p99 = float(np.percentile(np.abs(ecg_raw), 99))

    if overrides.ecg_unit:
        unit = overrides.ecg_unit
        notes.append(f"ECG unit {unit!r} set by user override (99th-pct |ecg| = {p99:.3g}).")
        return (ecg_raw / 1000.0 if unit == "uV" else ecg_raw), unit, p99

    if p99 > UV_DETECT_THRESHOLD:
        notes.append(
            f"ECG unit auto-detect: 99th-pct |ecg| = {p99:.3g} > {UV_DETECT_THRESHOLD:g} "
            "→ treated as microvolts and divided by 1000."
        )
        return ecg_raw / 1000.0, "uV", p99
    if p99 <= MV_DETECT_UPPER:
        notes.append(
            f"ECG unit auto-detect: 99th-pct |ecg| = {p99:.3g} ≤ {MV_DETECT_UPPER:g} "
            "→ treated as millivolts (no conversion)."
        )
        return ecg_raw, "mV", p99

    raise AmbiguousFormatError(
        "The ECG amplitude is implausible in both millivolts and microvolts.",
        [
            FormatQuestion(
                key="ecg_unit",
                question="Which unit is the ecg column in?",
                observed=(
                    f"99th-percentile absolute amplitude is {p99:.3g} — too large for a "
                    "chest-strap signal in mV, too small for R-waves in µV."
                ),
                options=("mV", "uV"),
            )
        ],
    )


def _derive_sampling_rate(time_ns: np.ndarray, notes: list[str]) -> tuple[float, float]:
    """Sampling rate from the median inter-sample interval — never hardcoded."""
    diffs = np.diff(time_ns)
    positive = diffs[diffs > 0]
    if len(positive) < MIN_SAMPLES - 1:
        raise LoaderError("Too few positive sample intervals to derive a sampling rate.")
    nonpositive = int(len(diffs) - len(positive))
    if nonpositive:
        notes.append(
            f"{nonpositive} non-increasing timestamp interval(s) ignored when deriving "
            "the sampling rate."
        )
    median_ns = float(np.median(positive))
    fs = 1e9 / median_ns
    std_ms = float(np.std(positive)) / 1e6
    if not (50.0 <= fs <= 2000.0):
        raise LoaderError(
            f"Derived sampling rate {fs:.1f} Hz is outside the plausible range for an "
            "ECG chest strap (50–2000 Hz) — the time column may not be nanoseconds."
        )
    notes.append(
        f"Sampling rate derived from timestamps: {fs:.3f} Hz "
        f"(median interval {median_ns / 1e6:.4f} ms, σ = {std_ms:.4f} ms)."
    )
    if math.isclose(fs, 130.0, abs_tol=1.0) and not math.isclose(fs, 130.0, abs_tol=1e-6):
        notes.append(
            f"Note: true rate {fs:.3f} Hz differs from nominal 130 Hz; the derived value "
            "is used throughout to avoid systematic timing error."
        )
    return fs, std_ms
