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
from app.activities.registry import resolve_profile
from app.extensions import db
from app.ingest.exceptions import AmbiguousFormatError, LoaderError
from app.ingest.loader import FormatOverrides, LoadedRecording, load_polar_csv
from app.models import ExcludedSegment, Flag, Metrics, ProcessingStatus, Session
from app.pipeline.engines import PipelineError
from app.pipeline.hrv import HRVResult
from app.pipeline.process import PipelineResult, run_pipeline
from app.report.render import ReportMeta, build_report_html
from app.screening.rules import run_screening


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
            _persist(session, rec, result)
            session.processing_status = ProcessingStatus.DONE
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


def _persist(session: Session, rec: LoadedRecording, result: PipelineResult) -> None:
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

    h = hrv_censored
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
        extras=_build_extras(result, hrv_censored, analysis),
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

    _write_cache(session, result)
    _write_report(session, rec, result, hrv_censored, analysis, resolved, flags)


def _build_extras(
    result: PipelineResult, hrv: HRVResult, analysis
) -> dict:
    return {
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
        "notes": result.notes + analysis.notes,
    }


def _write_cache(session: Session, result: PipelineResult) -> None:
    """Processed arrays next to the CSV — comparison views read these."""
    np.savez_compressed(
        cache_path_for(session),
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
    )


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
