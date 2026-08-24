"""Loading and staging helpers shared by ``sleep_eda.ipynb`` and its smoke test.

Only the boilerplate lives here — reading a session and running the staging
engines. Every number the notebook shows is still computed by ``app.sleep``
(:func:`~app.sleep.summary.summarize`,
:func:`~app.sleep.summary.stage_stats`,
:func:`~app.sleep.agreement.pairwise_agreement`); nothing in this module
reimplements sleep arithmetic.

Two ways in, one result:

* **cache branch** — the ``.npz`` written next to the CSV at processing time
  already holds the corrected RR series, the R-peak train, and the per-window
  quality summaries. Loading it takes milliseconds instead of the minutes a
  93 MB overnight CSV costs, and the staging engines need nothing else.
* **full branch** — ``load_polar_csv`` + ``run_pipeline`` from the raw CSV,
  which is what the app itself does.

Both produce a :class:`SessionData`, so the notebook's analysis cells never
branch on where the data came from.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def ensure_repo_on_path(repo_root: str | Path) -> Path:
    """Put the repository root on ``sys.path`` and return it.

    ``pyproject.toml`` has no ``[project]`` table, so the app is not an
    installable package — path insertion is the only way to ``import app``.
    """
    root = Path(repo_root).resolve()
    if not (root / "app").is_dir():
        raise FileNotFoundError(
            f"{root} does not look like the repository root (no 'app/' directory)."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


# Imports below are deferred to call time so this module can be imported
# before ``ensure_repo_on_path`` has run (the notebook's first cell).


def external_engine_config() -> dict:
    """The three config keys the external 5-class engine's status() reads."""
    from app.config import Config

    return {
        "SLEEP_EXTERNAL_DIR": Config.SLEEP_EXTERNAL_DIR,
        "SLEEP_EXTERNAL_PYTHON": Config.SLEEP_EXTERNAL_PYTHON,
        "SLEEP_EXTERNAL_TIMEOUT_S": Config.SLEEP_EXTERNAL_TIMEOUT_S,
    }


def engine_statuses() -> list:
    """EngineStatus for every engine, without importing anything heavy."""
    from app.sleep.engines import EngineStatus, external_ecg_staging, sleepecg_engine
    from app.sleep.engines import heuristic as heuristic_engine

    return [
        EngineStatus(
            key=heuristic_engine.ENGINE_KEY,
            label=heuristic_engine.ENGINE_LABEL,
            available=True,
        ),
        sleepecg_engine.status(),
        external_ecg_staging.status(external_engine_config()),
    ]


def find_candidate_csvs(repo_root: str | Path) -> list[Path]:
    """Uploaded CSVs under the configured data directory, largest first.

    An overnight recording is the big file in the directory, so size is a
    usable first sort key when the user has not said which night they mean.
    """
    root = Path(repo_root)
    data_dir = Path(os.environ.get("ECGLOG_DATA_DIR", root / "data"))
    uploads = data_dir / "uploads"
    if not uploads.is_dir():
        return []
    return sorted(uploads.rglob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)


# --------------------------------------------------------------------------
# Cheap recording span (start wall-clock + duration) without reading the file
# --------------------------------------------------------------------------


def _last_data_line(path: Path, tail_bytes: int = 65_536) -> str:
    """The final non-empty line, read from the tail rather than the whole file."""
    size = path.stat().st_size
    with path.open("rb") as fh:
        fh.seek(max(0, size - tail_bytes))
        chunk = fh.read()
    lines = [ln for ln in chunk.decode("utf-8", errors="replace").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"No data lines found in the tail of {path.name}.")
    return lines[-1]


@dataclass
class RecordingSpan:
    """When the recording started (UTC) and how long it ran."""

    start_time: dt.datetime
    duration_s: float
    epoch_detected: str
    #: Sampling rate estimated from the first samples only — a display value,
    #: not the loader's whole-file figure.
    sampling_rate_hz_estimate: float | None
    notes: list[str] = field(default_factory=list)


def recording_span(csv_path: str | Path, now: dt.datetime | None = None) -> RecordingSpan:
    """Start time and duration from the CSV's first and last rows only.

    Delimiter, header, and epoch resolution all go through the loader's own
    helpers, so this shortcut can never disagree with a full
    :func:`~app.ingest.loader.load_polar_csv` about when a night began.
    """
    # The loader's private helpers are used deliberately: reimplementing the
    # epoch tie-break here would be a second, divergable copy of that rule.
    from app.ingest.loader import (  # noqa: PLC2701
        FormatOverrides,
        _detect_delimiter,
        _detect_epoch,
        _parse_header,
    )

    path = Path(csv_path)
    now = now or dt.datetime.now(dt.UTC)
    overrides = FormatOverrides()
    notes: list[str] = []

    head: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip():
                head.append(line.rstrip("\n"))
            if len(head) >= 2001:
                break
    if len(head) < 2:
        raise ValueError(f"{path.name} has no data rows.")

    delimiter = _detect_delimiter(head[0], overrides, notes)
    columns = _parse_header(head[0], delimiter, overrides)
    time_idx = columns.index("time")

    def time_ns(line: str) -> int:
        return int(line.split(delimiter)[time_idx].strip())

    first_ns = time_ns(head[1])
    last_ns = time_ns(_last_data_line(path))
    epoch, start = _detect_epoch(first_ns, path.name, overrides, now, notes)

    sample_ns = np.array([time_ns(ln) for ln in head[1:]], dtype=np.int64)
    fs_estimate = None
    if len(sample_ns) > 10:
        median_dt_s = float(np.median(np.diff(sample_ns))) / 1e9
        if median_dt_s > 0:
            fs_estimate = 1.0 / median_dt_s
            notes.append(
                f"Sampling rate {fs_estimate:.2f} Hz estimated from the first "
                f"{len(sample_ns)} samples (the loader derives it over the whole file)."
            )

    return RecordingSpan(
        start_time=start,
        duration_s=float((last_ns - first_ns) / 1e9),
        epoch_detected=epoch,
        sampling_rate_hz_estimate=fs_estimate,
        notes=notes,
    )


# --------------------------------------------------------------------------
# One session, from either source
# --------------------------------------------------------------------------


@dataclass
class SessionData:
    """The inputs staging needs, however they were obtained."""

    csv_path: Path
    rr: object  # app.pipeline.rr.RRSeries
    quality: object  # app.pipeline.quality.QualityResult
    peak_times_s: np.ndarray
    start_time: dt.datetime
    duration_s: float
    sampling_rate_hz: float | None
    #: "cache" or "pipeline" — which branch produced this.
    source: str
    #: Present only on the full branch; None when loaded from the cache.
    recording: object | None = None
    result: object | None = None
    #: Hypnogram stage arrays stored in the cache at processing time, by
    #: engine key. Used to check the cache reconstruction, never as output.
    cached_stages: dict[str, np.ndarray] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def n_epochs(self) -> int:
        from app.sleep.stages import make_epoch_grid

        return len(make_epoch_grid(self.duration_s))


def cache_path_for_csv(csv_path: str | Path) -> Path:
    """The ``.npz`` the app writes next to an uploaded CSV."""
    return Path(csv_path).with_suffix(".npz")


def _quality_from_cache(data) -> object:
    """A QualityResult carrying exactly what the cache preserved.

    ``compute_epoch_features`` reads only ``quality.windows[*].start_s`` and
    ``.wander_rms_mv`` (app/sleep/epochs.py), so the per-window arrays in the
    cache are a complete input for staging. The per-sample ``sqi`` and
    ``wander_mv`` traces are *not* cached and stay empty here — this object is
    fit for staging and for nothing else, which is why the exclusion reasons
    are labelled rather than left blank.
    """
    from app.pipeline.quality import (  # noqa: PLC2701
        WINDOW_S,
        QualityResult,
        QualityWindow,
        _merge_excluded,
    )

    starts = data["window_start_s"]
    sqi = data["window_sqi"]
    wander = data["window_wander_mv"]
    excluded = data["window_excluded"]

    windows = [
        QualityWindow(
            start_s=float(s),
            end_s=float(s) + WINDOW_S,
            sqi_mean=float(q),
            wander_rms_mv=float(w),
            excluded=bool(x),
            reasons=("excluded at processing time (reason not cached)",) if x else (),
        )
        for s, q, w, x in zip(starts, sqi, wander, excluded, strict=True)
    ]
    segments = _merge_excluded(windows)
    excluded_total = float(sum(e - s for s, e, _ in segments))
    analysed = float(max(0.0, len(windows) * WINDOW_S - excluded_total))
    return QualityResult(
        windows=windows,
        excluded_segments=segments,
        sqi=np.array([]),
        wander_mv=np.array([]),
        excluded_total_s=excluded_total,
        analysed_total_s=analysed,
    )


#: Cache versions below this predate the sleep arrays; earlier caches still
#: carry everything staging needs, so they are usable — only the stored
#: hypnogram cross-check is skipped.
MIN_SLEEP_CACHE_VERSION = 3


def load_session(csv_path: str | Path, prefer_cache: bool = True) -> SessionData:
    """Load one night, from the ``.npz`` cache when possible."""
    from app.ingest.loader import load_polar_csv
    from app.pipeline.process import run_pipeline
    from app.pipeline.rr import RRSeries

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"No such CSV: {path}")

    cache = cache_path_for_csv(path)
    if prefer_cache and cache.exists():
        span = recording_span(path)
        notes = list(span.notes)
        with np.load(cache) as data:
            version = int(data["cache_version"][0]) if "cache_version" in data else 1
            rr = RRSeries(
                rr_ms=data["rr_ms"],
                t_s=data["rr_t_s"],
                discontinuity=data["rr_discontinuity"],
                n_dropped_excluded=0,
                n_dropped_ceiling=0,
                n_dropped_floor=0,
            )
            quality = _quality_from_cache(data)
            peak_times_s = data["peak_times_s"]
            cached_stages = {
                key.removeprefix("sleep_stages_").replace("_", "-"): data[key]
                for key in data.files
                if key.startswith("sleep_stages_")
            }
        notes.append(
            f"Loaded from the processing cache {cache.name} (version {version}); "
            "the raw ECG waveform was not read."
        )
        if version < MIN_SLEEP_CACHE_VERSION:
            notes.append(
                f"Cache version {version} predates the sleep arrays — staging still "
                "runs, but there is no stored hypnogram to cross-check against."
            )
        return SessionData(
            csv_path=path,
            rr=rr,
            quality=quality,
            peak_times_s=peak_times_s,
            start_time=span.start_time,
            duration_s=span.duration_s,
            sampling_rate_hz=span.sampling_rate_hz_estimate,
            source="cache",
            cached_stages=cached_stages,
            notes=notes,
        )

    rec = load_polar_csv(path)
    result = run_pipeline(rec)
    return SessionData(
        csv_path=path,
        rr=result.rr,
        quality=result.quality,
        peak_times_s=result.peak_times_s,
        start_time=rec.start_time,
        duration_s=rec.duration_s,
        sampling_rate_hz=rec.sampling_rate_hz,
        source="pipeline",
        recording=rec,
        result=result,
        notes=list(rec.notes) + list(result.notes),
    )


# --------------------------------------------------------------------------
# Staging
# --------------------------------------------------------------------------


def stage_night(
    data: SessionData,
    acc_path: str | Path | None = None,
    age_years: float | None = None,
    sex: str | None = None,
    engines: tuple[str, ...] = ("heuristic", "sleepecg"),
):
    """Run the requested engines and return a real :class:`SleepAnalysis`.

    On the full branch this delegates to
    :func:`~app.sleep.orchestrator.run_sleep_analysis` — the app's own
    contract. On the cache branch the same steps run here, because the
    orchestrator needs a ``LoadedRecording`` the cache does not contain.
    Either way, engine failures are downgraded to an ``EngineStatus`` and
    never raised: a broken engine must not end the analysis.

    Returns ``(analysis, features)``. The per-epoch features come back
    alongside the analysis because they are the notebook's main exploratory
    object — and they are returned even when the recording is too short to
    stage, since a night's HR and HRV per epoch are worth looking at whether
    or not stages were estimated. ``analysis.hypnograms`` is the emptiness to
    test for, not ``features``.
    """
    from app.activities.base import PersonContext
    from app.ingest.acc_loader import load_polar_acc_csv
    from app.ingest.exceptions import LoaderError
    from app.sleep.actigraphy import activity_counts, apply_wake_override, wake_override_mask
    from app.sleep.agreement import pairwise_agreement
    from app.sleep.engines import EngineStatus, sleepecg_engine
    from app.sleep.engines import heuristic as heuristic_engine
    from app.sleep.epochs import compute_epoch_features
    from app.sleep.orchestrator import (
        ENGINE_PRECEDENCE,
        MIN_STAGING_DURATION_S,
        SleepAnalysis,
        run_sleep_analysis,
    )
    from app.sleep.summary import stage_stats, summarize

    ctx = PersonContext(age_years=age_years, sex=sex)

    if data.source == "pipeline":
        analysis = run_sleep_analysis(
            data.recording,
            data.result,
            acc_path=str(acc_path) if acc_path else None,
            ctx=ctx,
            config=external_engine_config(),
        )
        features = compute_epoch_features(
            data.rr, data.quality, data.duration_s, analysis.acc_epochs
        )
        return analysis, features

    analysis = SleepAnalysis()
    n_epochs = data.n_epochs

    if acc_path:
        try:
            acc = load_polar_acc_csv(str(acc_path))
            analysis.acc_epochs = activity_counts(acc, data.start_time, n_epochs)
            analysis.notes.extend(analysis.acc_epochs.notes)
        except LoaderError as exc:
            analysis.acc_error = str(exc)
            analysis.notes.append(
                f"Accelerometer file could not be used: {exc} Staging ran "
                "without movement data."
            )

    features = compute_epoch_features(
        data.rr, data.quality, data.duration_s, analysis.acc_epochs
    )

    # Same refusal as the orchestrator: a sub-30-minute record cannot contain
    # the sleep architecture the numbers would claim to measure. The features
    # above are still returned — they are valid, the stages would not be.
    if data.duration_s < MIN_STAGING_DURATION_S:
        analysis.notes.append(
            f"Recording is {data.duration_s / 60.0:.0f} min long — below the "
            f"{MIN_STAGING_DURATION_S / 60.0:.0f} min minimum for sleep staging; "
            "no stages were estimated."
        )
        return analysis, features

    def run(status, fn):
        """Mirror of orchestrator._run_engine: failure downgrades, never raises."""
        if not status.available:
            analysis.engines.append(status)
            return
        try:
            analysis.hypnograms.append(fn())
            analysis.engines.append(status)
        except Exception as exc:  # noqa: BLE001 - a broken engine must not stop the notebook
            analysis.engines.append(
                EngineStatus(
                    key=status.key,
                    label=status.label,
                    available=False,
                    unavailable_reason=f"Engine failed: {exc}",
                )
            )
            analysis.notes.append(f"{status.label} failed and was skipped: {exc}")

    if "heuristic" in engines:
        run(
            EngineStatus(
                key=heuristic_engine.ENGINE_KEY,
                label=heuristic_engine.ENGINE_LABEL,
                available=True,
            ),
            lambda: heuristic_engine.stage_heuristic(features),
        )
    if "sleepecg" in engines:
        run(
            sleepecg_engine.status(),
            lambda: sleepecg_engine.stage_sleepecg(
                data.peak_times_s, data.start_time, data.duration_s, ctx, sex
            ),
        )

    if analysis.acc_epochs is not None and analysis.hypnograms:
        mask = wake_override_mask(analysis.acc_epochs)
        overridden = []
        for hyp in analysis.hypnograms:
            new_hyp, n_changed = (
                apply_wake_override(hyp, mask) if len(mask) == hyp.n_epochs else (hyp, 0)
            )
            overridden.append(new_hyp)
            analysis.override_epochs_n[hyp.engine] = n_changed
        analysis.hypnograms = overridden

    for hyp in analysis.hypnograms:
        analysis.summaries[hyp.engine] = summarize(hyp)
        analysis.stage_stats[hyp.engine] = stage_stats(hyp, data.rr)
    analysis.agreement = pairwise_agreement(analysis.hypnograms)

    produced = {h.engine for h in analysis.hypnograms}
    analysis.primary_engine = next(
        (key for key in ENGINE_PRECEDENCE if key in produced), None
    )
    return analysis, features


# --------------------------------------------------------------------------
# Tidy per-epoch table
# --------------------------------------------------------------------------


def epoch_table(data: SessionData, analysis, features):
    """One row per 30 s epoch: clock time, every engine's stage, and features."""
    import pandas as pd

    from app.sleep.stages import UNSCORED, stage_labels

    n = features.n_epochs
    clock = [
        data.start_time.astimezone() + dt.timedelta(seconds=float(s))
        for s in features.epoch_start_s
    ]
    table = {
        "epoch": np.arange(n),
        "clock_time": clock,
        "elapsed_min": features.epoch_start_s / 60.0,
        "mean_hr_bpm": features.mean_hr_bpm,
        "rmssd_ms": features.rmssd_ms,
        "lf_hf": features.lf_hf,
        "hr_vs_baseline_bpm": features.hr_vs_baseline,
        "movement": features.movement,
        "rr_coverage": features.coverage,
    }
    for hyp in analysis.hypnograms:
        labels = stage_labels(hyp.vocab)
        table[f"stage_{hyp.engine}"] = [
            "Unscored" if s == UNSCORED else labels[s] for s in hyp.stages
        ]
    return pd.DataFrame(table)
