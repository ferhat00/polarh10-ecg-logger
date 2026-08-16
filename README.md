# Polar H10 ECG Logger

A local, offline Flask application for analysing raw ECG exports from a Polar H10
chest strap: upload sessions recorded during different activities, get a per-session
report on real ECG paper, screen for patterns worth noting, and compare sessions over
time.

## What this tool is — and is not

**It is a screening and fitness tool.** It computes HRV metrics, renders rhythm strips
at true clinical scale, applies rule-based screening (sustained high/low HR, confirmed
ectopic beats, RR irregularity), and tracks trends across sessions.

**It is not a diagnostic device.** Nothing it outputs is a diagnosis. Every screening
flag carries a disclaimer as part of the flag object itself, and "no flags raised"
means only that no rule-based threshold was crossed — never clinical clearance. It
deliberately does **not** compute QRS width, QT/QTc, PR interval, or ECG axis: at the
H10's ~130 Hz sampling rate one sample spans 7.7 ms, and interval delineation at that
resolution returns quantisation artifacts, not physiology. Interval measurement needs
500–1000 Hz and multiple leads.

All health data stays local: SQLite on disk, uploaded CSVs on the local filesystem, no
external API calls, no telemetry, no CDN assets. The app works fully offline (fonts
and all frontend assets are vendored).

## Setup

Requires Python 3.12 (the signal-processing stack is not yet reliable on 3.14).

```bash
py -3.12 -m venv .venv               # macOS/Linux: python3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m flask --app wsgi db upgrade
.venv/Scripts/python -m flask --app wsgi run
```

Then open http://127.0.0.1:5000, add a person, and upload a recording. Configuration
is environment-based with local defaults (`ECGLOG_DATA_DIR`, `ECGLOG_DATABASE_URI`,
`ECGLOG_SECRET_KEY`, `ECGLOG_MAX_UPLOAD_BYTES`) — see `app/config.py`.

Data layout (all under `data/`, which is gitignored):

```
data/app.db                                   SQLite database
data/uploads/<person>/<id>__<name>.csv        the original upload, never overwritten
data/uploads/<person>/<id>__<name>.npz        cached arrays (R-peaks, RR series, quality)
data/uploads/<person>/<id>__<name>.report.html  the rendered report
data/logs/<person>_history.md                 append-only decision log
```

Deleting a person removes their database rows; their files under `data/` are retained
on disk deliberately — delete those by hand if you mean to.

## The data-format contract

The loader is written against a verified Polar Sensor Logger export and **detects
rather than assumes**:

- Header `time,ecg,hr,rr,marker`; **rows are ragged** (2–5 fields) and every row is
  explicitly padded to the header width.
- `time` is nanoseconds, but the epoch is **detected** (Unix 1970 vs Polar's
  documented 2000-01-01), by checking which interpretation lands in a plausible date
  window, cross-checked against any date in the filename. Never hardcoded.
- `ecg` may be mV or µV depending on exporter version — **auto-detected by
  amplitude** (99th-percentile |ecg| > 20 is only physical as µV). The branch taken is
  logged and shown in the report.
- The sampling rate is **derived from the timestamps** (real straps run ~130.03 Hz,
  not 130.000).
- `hr`/`rr` are the device's own sparse streams, used only as a cross-check against
  our R-peak detection. `marker` is supported when populated, never required.

A file that doesn't match this shape — different delimiter, decimal commas, unknown
columns, ambiguous unit or epoch — raises a specific error carrying the questions
that need answering, and the UI shows a mapping form instead of guessing. A wrong
unit or epoch assumption would corrupt everything downstream, so nothing guesses.

## Adding an activity profile

Activities change what is computed, what is suppressed, and how results are read,
because posture and motion dominate HRV far more than fitness does.

**From the UI:** *Activities → Add activity*. A custom activity inherits its analysis
behaviour from a built-in profile and may override the expected HR range and the
comparison grouping key. One database row, no code.

**In code:** add one file to `app/activities/profiles/` with a
`@register`-decorated `ActivityProfile` subclass. A profile declares its expected HR
range and motion level, static suppressions/cautions (with the reason shown to the
user), dynamic suppressions (e.g. running suppresses the whole HRV suite above 70 %
heart-rate reserve), activity-specific derived metrics, screening HR limits, and its
comparison key. See `app/activities/profiles/standing.py` for a compact example.

## Development

```bash
.venv/Scripts/python -m pytest        # full suite
.venv/Scripts/python -m ruff check .  # lint
.venv/Scripts/python scripts/make_fixture.py   # regenerate the synthetic fixture
```

The test fixture is synthetic — generated from a known RR series so expected metrics
are analytically checkable. No real health data is committed to this repository, and
none should ever be.

Architecture, briefly: `app/ingest` (loader), `app/pipeline` (cleaning, R-peaks,
quality windowing, RR construction, Kubios artifact correction, beat-template
morphology, HRV), `app/screening` (rule-based flags, every threshold cited in
`screening/thresholds.py`), `app/activities` (profile registry), `app/report`
(matplotlib figures + self-contained HTML), `app/comparison.py` (cross-session
guardrails), `app/logbook` (append-only Markdown log), with Flask blueprints on top.
