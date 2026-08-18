# Polar H10 ECG Logger

A local, offline Flask application for analysing raw ECG exports from a Polar H10
chest strap: upload sessions recorded during different activities, get a per-session
report on real ECG paper, screen for patterns worth noting, analyse overnight
recordings into sleep stages, and compare sessions over time.

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

The literature review behind the ectopy-statistics feature — device validation,
detection algorithms at 130 Hz, burden variability, trigger epidemiology, and the
statistical design — lives in [`docs/RESEARCH.md`](docs/RESEARCH.md).

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
data/uploads/<person>/<id>__acc__<name>.csv   optional accelerometer companion file
data/uploads/<person>/<id>__<name>.npz        cached arrays (R-peaks, RR series, quality, sleep stages)
data/uploads/<person>/<id>__<name>.report.html  the rendered report
data/logs/<person>_history.md                 append-only decision log
```

The default upload cap is 512 MB (`ECGLOG_MAX_UPLOAD_BYTES`) — an 8-hour
overnight ECG export is ~135 MB.

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

## Trigger tags and ectopy statistics

Rare, benign ectopic beats are exactly what wrist-based spot-check wearables miss:
counting them needs a continuous denominator of analysed beats. This tool confirms
ectopic beats with a triple gate (artifact-correction class ∧ prematurity against the
local rhythm ∧ not explained by motion), groups them into events (singles, couplets,
runs, bigeminy/trigeminy patterns, each with a compensatory-pause descriptor), and
stores per-session burden so it can be compared across conditions.

**Trigger tags** mark exposures in the hours before or during a recording — caffeine,
alcohol, poor sleep, stress, and friends (the vocabulary follows the published
trigger trials; custom tags welcome). Tag sessions at upload or from the session
page, then open **Triggers** for the statistics: burden over time, tagged-vs-untagged
comparisons, an hour-of-day profile, and per-tag rate ratios from a
negative-binomial model with an analysed-time offset and a circadian covariate.

The statistics are deliberately guarded. Ectopic burden swings severalfold between
days within one person, so the dashboard refuses to print estimates below minimum
data thresholds, shows confidence intervals rather than verdicts, and spells out the
association-not-causation caveats. Sessions processed before this feature need one
re-analysis (button on the session page) to populate their event data. The full
literature grounding — device validation, algorithm choices, variability numbers,
and the experiment designs worth copying — is in [`docs/RESEARCH.md`](docs/RESEARCH.md).

## Sleep staging

Record a whole night (start the strap at lights-off, stop it on getting up),
upload it with the **Sleep (overnight)** activity, and the report gains a sleep
section: hypnograms per staging engine, time in each stage, sleep efficiency,
sleep-onset latency, WASO, awakenings, REM latency, per-stage HR/HRV, and —
when several engines ran — their epoch-by-epoch agreement (Cohen's κ).
Optionally attach the Polar Sensor Logger **accelerometer export** as a second
file: movement sharpens wake detection (the weakest class for heart-beat-based
staging) and draws a movement trace in the report.

Three engines, in increasing setup cost:

| Engine | Stages | Needs |
| --- | --- | --- |
| Built-in cited rules + smoothing | Wake/Light/Deep/REM | nothing — works offline out of the box |
| SleepECG pre-trained GRU (MESA/SHHS) | Wake/REM/NREM | `pip install -r requirements-sleep.txt` (classifiers ship in the wheel; still fully offline) |
| External 5-class deep net ([adammj/ecg-sleep-staging](https://github.com/adammj/ecg-sleep-staging)) | Wake/N1/N2/N3/REM | your own clone of that AGPL tool + two env vars — run over a subprocess boundary, never vendored |

**Honesty first**: heart-beat-based staging is an estimate. The best published
model validated on this exact strap — Sleep²/NUKKUAA (Topalidis et al. 2023,
Sensors 23(5):2390; proprietary, so it cannot run here, but its 4-class
convention and its numbers anchor this feature) — reaches 80.3 % epoch
agreement (κ≈0.69) with laboratory polysomnography. Every hypnogram in the
report carries its engine's accuracy note, every night carries a
non-diagnostic disclaimer, and sleep sessions screen bradycardia at
sleep-appropriate thresholds (nocturnal dipping is physiology, not a
finding). The full literature grounding, per-engine setup, and limitations
live in [`docs/SLEEP.md`](docs/SLEEP.md).

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
