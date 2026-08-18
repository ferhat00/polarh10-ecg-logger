"""Background session processing.

Uploads never block the request: processing runs in a daemon thread (or
inline when ``PROCESS_SYNC`` is set, as in tests) and the upload page polls a
status endpoint. Results are persisted three ways:

* DB rows — session provenance/quality columns, one metrics row, flags,
  excluded segments;
* an ``.npz`` cache next to the CSV (R-peaks, corrected RR series, per-window
  quality) so comparison views never reprocess;
* the fully rendered report HTML next to the CSV, so opening a report is a
  file read, not a ~20 s pipeline run.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
from flask import Flask

from app.activities.base import ActivityInputs, PersonContext, apply_suppressions
from app.activities.metrics import lowest_sustained_hr
from app.activities.registry import resolve_profile
from app.extensions import db
from app.ingest.exceptions import AmbiguousFormatError, LoaderError
from app.ingest.loader import FormatOverrides, LoadedRecording, load_polar_csv
from app.models import ExcludedSegment, Flag, Metrics, ProcessingStatus, Session
from app.pipeline.engines import PipelineError
from app.pipeline.events import EVENT_KIND_CODES
from app.pipeline.hrv import HRVResult
from app.pipeline.process import PipelineResult, run_pipeline
from app.report.render import ReportMeta, build_report_html
from app.screening.rules import run_screening
from app.sleep.orchestrator import SleepAnalysis, run_sleep_analysis


def submit_processing(app: Flask, session_id: int) -> None:
    """Run processing inline (tests) or on a daemon thread (normal)."""
    if app.config.get("PROCESS_SYNC"):
        _process(app, session_id)
        return
    thread = threading.Thread(
        target=_process, args=(app, session_id), daemon=True, name=f"process-{session_id}"
    )
    thread.start()


def cache_path_for(session: Session) -> Path:
    return Path(session.stored_path).with_suffix(".npz")


def report_path_for(session: Session) -> Path:
    return Path(session.stored_path).with_suffix(".report.html")


def _process(app: Flask, session_id: int) -> None:
    with app.app_context():
        session = db.session.get(Session, session_id)
        if session is None:
            return
        session.processing_status = ProcessingStatus.RUNNING
        session.error_message = None
        db.session.commit()

        try:
            overrides = (
                FormatOverrides(**session.format_overrides)
                if session.format_overrides
                else None
            )
            rec = load_polar_csv(session.stored_path, overrides=overrides)
            result = run_pipeline(rec)

            # Sleep staging is a processing step, gated on the profile: it
            # needs the recording start time, the R-peak train, the raw ECG,
            # the optional ACC file, and app config — none of which belong
            # in the profiles' ActivityInputs slice.
            sleep: SleepAnalysis | None = None
            resolved_gate = (
                resolve_profile(session.activity_type) if session.activity_type else None
            )
            if resolved_gate is not None and resolved_gate.profile.requests_sleep_staging:
                sleep = run_sleep_analysis(
                    rec,
                    result,
                    session.acc_stored_path,
                    PersonContext.from_person(session.person),
                    app.config,
                )

            _persist(session, rec, result, sleep)
            session.processing_status = ProcessingStatus.DONE
            # Refresh relationships so the log entry sees this run's rows.
            db.session.flush()
            db.session.expire(session, ["flags", "metrics"])
            from app.logbook.writer import append_session_entry

            append_session_entry(session)
        except AmbiguousFormatError as exc:
            session.processing_status = ProcessingStatus.NEEDS_MAPPING
            session.error_message = json.dumps(exc.as_dict())
        except (LoaderError, PipelineError) as exc:
            session.processing_status = ProcessingStatus.ERROR
            session.error_message = str(exc)[:2000]
        except Exception as exc:  # noqa: BLE001 - a crashed thread must not strand "running"
            app.logger.exception("Unexpected processing failure for session %s", session_id)
            session.processing_status = ProcessingStatus.ERROR
            session.error_message = f"Unexpected error: {exc}"[:2000]
        db.session.commit()


def _persist(
    session: Session,
    rec: LoadedRecording,
    result: PipelineResult,
    sleep: SleepAnalysis | None = None,
) -> None:
    """Store DB rows, the .npz cache, and the rendered report."""
    person = session.person
    ctx = PersonContext.from_person(person)
    resolved = resolve_profile(session.activity_type) if session.activity_type else None
    profile = resolved.profile if resolved else None

    inputs = ActivityInputs.from_pipeline(result, markers=rec.markers)
    if profile is not None:
        analysis = profile.analyze(inputs, ctx)
    else:
        from app.activities import get_profile

        analysis = get_profile("sitting").analyze(inputs, ctx)
    hrv_censored, _ = apply_suppressions(result.hrv, analysis.suppressions)

    flags = (
        []
        if analysis.not_analysable
        else run_screening(
            result, limits=analysis.hr_limits, athlete_baseline=ctx.athlete_baseline
        )
    )

    # --- session provenance / quality columns -----------------------------
    session.recorded_at = rec.start_time
    session.duration_s = rec.duration_s
    session.sampling_rate_hz = rec.sampling_rate_hz
    session.engine_used = result.engine_used
    session.ecg_unit_detected = rec.ecg_unit_detected
    session.epoch_detected = rec.epoch_detected
    session.beats_corrected_pct = result.correction.pct_corrected
    session.reduced_confidence = result.correction.reduced_confidence
    session.excluded_s = result.quality.excluded_total_s
    session.analysed_s = result.quality.analysed_total_s

    # --- metrics (replace any previous run's row) -------------------------
    if session.metrics is not None:
        db.session.delete(session.metrics)
    for old_flag in session.flags:
        db.session.delete(old_flag)
    for old_seg in session.excluded_segments:
        db.session.delete(old_seg)
    db.session.flush()

    # Sleep results are dropped (with the reason recorded) when the profile
    # declared the whole session not analysable — stage fractions from a
    # mostly-excluded night would be misleading, not merely imprecise.
    sleep_extras: dict | None = None
    if sleep is not None and analysis.not_analysable:
        sleep_extras = {"skipped_reason": analysis.not_analysable_reason}
        sleep = None
    elif sleep is not None:
        sleep_extras = sleep.as_extras()
    primary_sleep = sleep.primary_summary() if sleep is not None else None

    h = hrv_censored
    ev = result.events
    metrics = Metrics(
        session_id=session.id,
        n_beats=h.n_beats,
        mean_rr_ms=h.mean_rr_ms,
        mean_hr_bpm=h.mean_hr_bpm,
        sdnn_ms=h.sdnn_ms,
        rmssd_ms=h.rmssd_ms,
        pnn50_pct=h.pnn50_pct,
        vlf_power_ms2=h.vlf_power_ms2,
        lf_power_ms2=h.lf_power_ms2,
        hf_power_ms2=h.hf_power_ms2,
        lf_hf_ratio=h.lf_hf_ratio,
        sd1_ms=h.sd1_ms,
        sd2_ms=h.sd2_ms,
        sd1_sd2_ratio=h.sd1_sd2_ratio,
        sample_entropy=h.sample_entropy,
        dfa_alpha1=h.dfa_alpha1,
        # Ectopy burden is a beat count over analysed time, not an HRV
        # family, so activity suppressions never censor it.
        ectopy_beats_n=ev.n_confirmed if ev else None,
        ectopy_per_hour=ev.per_hour if ev else None,
        ectopy_pct_beats=ev.pct_of_beats if ev else None,
        single_n=ev.n_singles if ev else None,
        couplet_n=ev.n_couplets if ev else None,
        run_n=ev.n_runs if ev else None,
        longest_run_beats=ev.longest_run_beats if ev else None,
        bigeminy_episode_n=ev.bigeminy_episodes if ev else None,
        trigeminy_episode_n=ev.trigeminy_episodes if ev else None,
        # Sleep architecture from the primary staging engine (sleep sessions
        # only). Stage minutes stay NULL when the engine's vocabulary cannot
        # distinguish them — never invented.
        tst_min=primary_sleep.tst_min if primary_sleep else None,
        sleep_efficiency_pct=(
            primary_sleep.sleep_efficiency_pct if primary_sleep else None
        ),
        sol_min=primary_sleep.sol_min if primary_sleep else None,
        waso_min=primary_sleep.waso_min if primary_sleep else None,
        light_min=primary_sleep.light_min if primary_sleep else None,
        deep_min=primary_sleep.deep_min if primary_sleep else None,
        rem_min=primary_sleep.rem_min if primary_sleep else None,
        awakenings_n=primary_sleep.awakenings_n if primary_sleep else None,
        sleep_engine=sleep.primary_engine if sleep is not None else None,
        extras=_build_extras(result, hrv_censored, analysis, sleep_extras),
    )
    db.session.add(metrics)

    for flag in flags:
        db.session.add(
            Flag(
                session_id=session.id,
                kind=flag.kind,
                severity=flag.severity,
                description=flag.description,
                evidence=flag.evidence,
                threshold=flag.threshold,
                disclaimer=flag.disclaimer,
            )
        )
    for seg_start, seg_end, reason in result.quality.excluded_segments:
        db.session.add(
            ExcludedSegment(
                session_id=session.id, start_s=seg_start, end_s=seg_end, reason=reason
            )
        )

    _write_cache(session, result, sleep)
    _write_report(session, rec, result, hrv_censored, analysis, resolved, flags)


def _build_extras(
    result: PipelineResult, hrv: HRVResult, analysis, sleep_extras: dict | None = None
) -> dict:
    sqi_values = [
        w.sqi_mean for w in result.quality.windows if not np.isnan(w.sqi_mean)
    ]
    return {
        "sleep": sleep_extras,
        "sqi_mean": float(np.mean(sqi_values)) if sqi_values else None,
        "resting_hr_bpm": lowest_sustained_hr(result.rr),
        "activity": analysis.extras,
        "profile_key": analysis.profile_key,
        "not_analysable": analysis.not_analysable,
        "not_analysable_reason": analysis.not_analysable_reason,
        "suppressions": [
            {"family": s.family, "reason": s.reason} for s in analysis.suppressions
        ],
        "cautions": [
            {"family": c.family, "reason": c.reason} for c in analysis.cautions
        ],
        "sdnn_per_window_ms": hrv.sdnn_per_window_ms,
        "sdnn_window_t_s": hrv.sdnn_window_t_s,
        "lf_peak_hz": hrv.lf_peak_hz,
        "psd_method": hrv.psd_method,
        "device_rr_median_abs_diff_ms": result.device_rr_median_abs_diff_ms,
        "ectopy": _ectopy_extras(result),
        "notes": result.notes + analysis.notes,
    }


def _ectopy_extras(result: PipelineResult) -> dict | None:
    """Descriptive ectopy-event detail for the extras blob (see events.py)."""
    ev = result.events
    if ev is None:
        return None
    return {
        "n_confirmed": ev.n_confirmed,
        "per_hour": ev.per_hour,
        "n_singles": ev.n_singles,
        "n_couplets": ev.n_couplets,
        "n_runs": ev.n_runs,
        "longest_run_beats": ev.longest_run_beats,
        "bigeminy_episodes": ev.bigeminy_episodes,
        "trigeminy_episodes": ev.trigeminy_episodes,
        "n_pause_complete": ev.n_pause_complete,
        "n_pause_incomplete": ev.n_pause_incomplete,
        "n_morphology_candidates": int(np.sum(result.morphology.ectopy_candidate)),
    }


#: Bumped when the cache gains arrays newer code depends on. Version 2 added
#: the morphology and ectopy-event arrays; version 3 adds the sleep-staging
#: and accelerometer arrays (present only on sleep sessions). Caches without
#: a version predate the events pipeline and load_cached_events() returns
#: None so the UI can offer a reprocess.
CACHE_VERSION = 3


def _write_cache(
    session: Session, result: PipelineResult, sleep: SleepAnalysis | None = None
) -> None:
    """Processed arrays next to the CSV — comparison views read these.

    Index conventions: ``morph_*`` arrays align with the *corrected* peak
    train (``peaks_corrected``); the ectopy confirmation/event arrays align
    with the *raw detected* train (``rpeak_indices``) — see PipelineResult.
    Sleep arrays (``sleep_*``, ``acc_*``) exist only for sleep sessions.
    """
    ev = result.events
    events = ev.events if ev else []

    sleep_arrays: dict[str, np.ndarray] = {}
    if sleep is not None and sleep.hypnograms:
        sleep_arrays["sleep_epoch_start_s"] = sleep.hypnograms[0].epoch_start_s
        for hyp in sleep.hypnograms:
            key = hyp.engine.replace("-", "_")
            sleep_arrays[f"sleep_stages_{key}"] = hyp.stages
            sleep_arrays[f"sleep_vocab_{key}"] = np.array([hyp.vocab.value])
            if hyp.probabilities is not None:
                sleep_arrays[f"sleep_probs_{key}"] = hyp.probabilities
    if sleep is not None and sleep.acc_epochs is not None:
        sleep_arrays["acc_epoch_start_s"] = sleep.acc_epochs.epoch_start_s
        sleep_arrays["acc_counts"] = sleep.acc_epochs.counts
        sleep_arrays["acc_coverage"] = sleep.acc_epochs.coverage

    np.savez_compressed(
        cache_path_for(session),
        **sleep_arrays,
        rpeak_indices=result.detection.rpeak_indices,
        peaks_corrected=result.correction.peaks_corrected,
        peak_times_s=result.peak_times_s,
        rr_ms=result.rr.rr_ms,
        rr_t_s=result.rr.t_s,
        rr_discontinuity=result.rr.discontinuity,
        window_start_s=np.array([w.start_s for w in result.quality.windows]),
        window_sqi=np.array([w.sqi_mean for w in result.quality.windows]),
        window_wander_mv=np.array([w.wander_rms_mv for w in result.quality.windows]),
        window_excluded=np.array([w.excluded for w in result.quality.windows]),
        cache_version=np.array([CACHE_VERSION]),
        morph_correlations=result.morphology.correlations,
        morph_prematurity_pct=result.morphology.prematurity_pct,
        morph_outlier_mask=result.morphology.outlier_mask,
        morph_motion_explained=result.morphology.motion_explained,
        morph_ectopy_candidate=result.morphology.ectopy_candidate,
        ectopy_prematurity_pct=result.ectopy_prematurity_pct,
        ectopy_motion_mask=result.ectopy_motion_mask,
        ectopy_confirmed_mask=(
            ev.confirmed_mask if ev else np.array([], dtype=bool)
        ),
        event_kind=np.array(
            [EVENT_KIND_CODES[e.kind] for e in events], dtype=np.int8
        ),
        event_start_beat=np.array([e.start_beat for e in events], dtype=np.int64),
        event_end_beat=np.array([e.end_beat for e in events], dtype=np.int64),
        event_t_start_s=np.array([e.t_start_s for e in events]),
        event_t_end_s=np.array([e.t_end_s for e in events]),
        event_pause_ratio=np.array(
            [np.nan if e.pause_ratio is None else e.pause_ratio for e in events]
        ),
    )


def load_cached_events(session: Session) -> dict | None:
    """Event arrays from the cache; None when absent or pre-events (v1)."""
    path = cache_path_for(session)
    if not path.exists():
        return None
    with np.load(path) as data:
        if "cache_version" not in data or int(data["cache_version"][0]) < 2:
            return None
        return {
            "event_kind": data["event_kind"],
            "event_start_beat": data["event_start_beat"],
            "event_end_beat": data["event_end_beat"],
            "event_t_start_s": data["event_t_start_s"],
            "event_t_end_s": data["event_t_end_s"],
            "event_pause_ratio": data["event_pause_ratio"],
            "ectopy_confirmed_mask": data["ectopy_confirmed_mask"],
        }


def _write_report(
    session: Session,
    rec: LoadedRecording,
    result: PipelineResult,
    hrv_censored: HRVResult,
    analysis,
    resolved,
    flags,
) -> None:
    meta = ReportMeta(
        person_name=session.person.name,
        activity_name=resolved.display_name if resolved else "Unspecified activity",
        recorded_at=rec.start_time,
        context_note=session.context_note,
        original_filename=session.original_filename,
        file_sha256=session.file_sha256,
        reduced_confidence=result.correction.reduced_confidence,
    )
    html = build_report_html(rec, result, hrv_censored, analysis, resolved, flags, meta)
    report_path_for(session).write_text(html, encoding="utf-8")
