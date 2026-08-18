"""Run every available staging engine over one session and assemble results.

The orchestrator is the only entry point processing calls. Its contract:

* Every engine is individually wrapped — an engine that is missing, broken,
  or unhappy becomes an :class:`EngineStatus` with a reason and a note.
  **Session processing never fails because of a staging engine.**
* The ACC file is optional and its loader errors are captured, not raised.
* All hypnograms are kept (the report shows each with its accuracy note);
  the *primary* engine — best-available by the precedence below — fills the
  queryable ``Metrics`` columns.

Primary-engine precedence: ``external-5class`` (raw-ECG deep network, when
the user has installed it) > ``sleepecg`` (pre-trained GRU, when the
optional extras are installed) > ``heuristic`` (always available).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

import numpy as np

from app.activities.base import PersonContext
from app.ingest.acc_loader import load_polar_acc_csv
from app.ingest.exceptions import LoaderError
from app.ingest.loader import LoadedRecording
from app.pipeline.process import PipelineResult
from app.sleep.actigraphy import AccEpochs, activity_counts, apply_wake_override, wake_override_mask
from app.sleep.agreement import AgreementRow, pairwise_agreement
from app.sleep.engines import EngineStatus
from app.sleep.engines import heuristic as heuristic_engine
from app.sleep.epochs import compute_epoch_features
from app.sleep.stages import Hypnogram, make_epoch_grid
from app.sleep.summary import SleepSummary, stage_stats, summarize

#: Below this recording length staging is skipped: a sub-30-minute record
#: cannot contain the sleep architecture the numbers would claim to measure.
MIN_STAGING_DURATION_S = 1800.0

#: Queryable-columns precedence (first match that produced a hypnogram wins).
ENGINE_PRECEDENCE = ("external-5class", "sleepecg", "heuristic")


@dataclass
class SleepAnalysis:
    """Everything the sleep feature learned about one night."""

    hypnograms: list[Hypnogram] = field(default_factory=list)
    summaries: dict[str, SleepSummary] = field(default_factory=dict)
    primary_engine: str | None = None
    agreement: list[AgreementRow] = field(default_factory=list)
    acc_epochs: AccEpochs | None = None
    acc_error: str | None = None
    #: Engine key -> epochs forced to Wake by the movement override.
    override_epochs_n: dict[str, int] = field(default_factory=dict)
    #: Engine key -> per-stage HR/RMSSD rows for the report table.
    stage_stats: dict[str, list[dict]] = field(default_factory=dict)
    engines: list[EngineStatus] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def hypnogram_for(self, engine: str) -> Hypnogram | None:
        for hyp in self.hypnograms:
            if hyp.engine == engine:
                return hyp
        return None

    def primary_summary(self) -> SleepSummary | None:
        if self.primary_engine is None:
            return None
        return self.summaries.get(self.primary_engine)

    def as_extras(self) -> dict:
        """JSON-safe representation for ``Metrics.extras['sleep']``."""
        return {
            "primary_engine": self.primary_engine,
            "engines": [asdict(s) for s in self.engines],
            "hypnograms": {
                h.engine: {
                    "engine_label": h.engine_label,
                    "vocab": h.vocab.value,
                    "epoch_len_s": h.epoch_len_s,
                    "stages": [int(s) for s in h.stages],
                    "accuracy_note": h.accuracy_note,
                    "notes": list(h.notes),
                }
                for h in self.hypnograms
            },
            "summaries": {k: asdict(v) for k, v in self.summaries.items()},
            "agreement": [
                {
                    "engine_a": row.engine_a,
                    "engine_b": row.engine_b,
                    "vocab": row.vocab.value,
                    "kappa": row.kappa,
                    "percent_agree": row.percent_agree,
                    "n_epochs": row.n_epochs,
                }
                for row in self.agreement
            ],
            "acc": {
                "present": self.acc_epochs is not None,
                "error": self.acc_error,
                "override_epochs_n": dict(self.override_epochs_n),
                "threshold": (
                    None
                    if self.acc_epochs is None
                    or not np.isfinite(self.acc_epochs.threshold)
                    else float(self.acc_epochs.threshold)
                ),
            },
            "stage_stats": {
                k: [
                    {kk: (None if vv is None else vv) for kk, vv in row.items()}
                    for row in rows
                ]
                for k, rows in self.stage_stats.items()
            },
            "notes": list(self.notes),
        }


def run_sleep_analysis(
    rec: LoadedRecording,
    result: PipelineResult,
    acc_path: str | None,
    ctx: PersonContext,
    config: Mapping,
    stored_path: str | None = None,
) -> SleepAnalysis:
    """Stage one night with every engine that can run here."""
    analysis = SleepAnalysis()
    duration_s = rec.duration_s

    if duration_s < MIN_STAGING_DURATION_S:
        analysis.notes.append(
            f"Recording is {duration_s / 60.0:.0f} min long — below the "
            f"{MIN_STAGING_DURATION_S / 60.0:.0f} min minimum for sleep staging; "
            "no stages were estimated."
        )
        return analysis

    n_epochs = len(make_epoch_grid(duration_s))

    # --- optional accelerometer ------------------------------------------
    if acc_path:
        try:
            acc = load_polar_acc_csv(acc_path)
            analysis.acc_epochs = activity_counts(acc, rec.start_time, n_epochs)
            analysis.notes.extend(analysis.acc_epochs.notes)
        except LoaderError as exc:
            analysis.acc_error = str(exc)
            analysis.notes.append(
                f"Accelerometer file could not be used: {exc} Staging ran "
                "without movement data."
            )

    features = compute_epoch_features(
        result.rr, result.quality, duration_s, analysis.acc_epochs
    )

    # --- engines (each individually wrapped) ------------------------------
    _run_engine(
        analysis,
        EngineStatus(
            key=heuristic_engine.ENGINE_KEY,
            label=heuristic_engine.ENGINE_LABEL,
            available=True,
        ),
        lambda: heuristic_engine.stage_heuristic(features),
    )

    _run_optional_engines(analysis, rec, result, ctx, config, stored_path)

    # --- movement wake-override (applies to every engine) -----------------
    if analysis.acc_epochs is not None and analysis.hypnograms:
        mask = wake_override_mask(analysis.acc_epochs)
        overridden: list[Hypnogram] = []
        for hyp in analysis.hypnograms:
            if len(mask) == hyp.n_epochs:
                new_hyp, n_changed = apply_wake_override(hyp, mask)
            else:  # pragma: no cover - grids always match today
                new_hyp, n_changed = hyp, 0
            overridden.append(new_hyp)
            analysis.override_epochs_n[hyp.engine] = n_changed
        analysis.hypnograms = overridden

    # --- summaries, agreement, per-stage stats ----------------------------
    for hyp in analysis.hypnograms:
        analysis.summaries[hyp.engine] = summarize(hyp)
        analysis.stage_stats[hyp.engine] = stage_stats(hyp, result.rr)
    analysis.agreement = pairwise_agreement(analysis.hypnograms)

    produced = {h.engine for h in analysis.hypnograms}
    analysis.primary_engine = next(
        (key for key in ENGINE_PRECEDENCE if key in produced), None
    )
    return analysis


def _run_engine(analysis: SleepAnalysis, status: EngineStatus, run) -> None:
    """Run one engine; failure downgrades its status instead of raising."""
    if not status.available:
        analysis.engines.append(status)
        return
    try:
        hyp = run()
        analysis.hypnograms.append(hyp)
        analysis.engines.append(status)
    except Exception as exc:  # noqa: BLE001 - a broken engine must not kill processing
        analysis.engines.append(
            EngineStatus(
                key=status.key,
                label=status.label,
                available=False,
                unavailable_reason=f"Engine failed: {exc}",
            )
        )
        analysis.notes.append(f"{status.label} failed and was skipped: {exc}")


def _run_optional_engines(
    analysis: SleepAnalysis,
    rec: LoadedRecording,
    result: PipelineResult,
    ctx: PersonContext,
    config: Mapping,
    stored_path: str | None,
) -> None:
    """Optional engines: each reports an EngineStatus even when absent."""
    from app.sleep.engines import external_ecg_staging, sleepecg_engine

    _run_engine(
        analysis,
        sleepecg_engine.status(),
        lambda: sleepecg_engine.stage_sleepecg(
            result.peak_times_s,
            rec.start_time,
            rec.duration_s,
            ctx,
            ctx.sex,
        ),
    )

    def _external():
        import tempfile

        base = stored_path or tempfile.mkdtemp(prefix="ecglog-sleep-")
        return external_ecg_staging.stage_external(
            rec.ecg_mv,
            rec.sampling_rate_hz,
            result.peak_times_s,
            rec.start_time,
            rec.duration_s,
            ctx.age_years,
            ctx.sex,
            base,
            config,
        )

    _run_engine(analysis, external_ecg_staging.status(config), _external)
