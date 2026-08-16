"""Development preview of the report renderer.

Renders the full report over the committed synthetic fixture so the template
and figures can be inspected before the upload flow exists (phase 7 wires
reports to real sessions). The pipeline run is cached per process.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from flask import Blueprint, Response

from app.activities import get_profile
from app.activities.base import ActivityInputs, PersonContext, apply_suppressions
from app.ingest.loader import load_polar_csv
from app.pipeline.process import run_pipeline
from app.report.render import ReportMeta, build_report_html
from app.screening.rules import run_screening

bp = Blueprint("report_preview", __name__)

_FIXTURE = (
    Path(__file__).resolve().parent.parent.parent
    / "tests" / "fixtures" / "synthetic_supine.csv"
)
_cache: dict[str, str] = {}


@bp.get("/report/preview")
def preview() -> Response:
    if "html" not in _cache:
        rec = load_polar_csv(_FIXTURE)
        result = run_pipeline(rec)
        profile = get_profile("supine")
        ctx = PersonContext()
        inputs = ActivityInputs.from_pipeline(result, markers=rec.markers)
        analysis = profile.analyze(inputs, ctx)
        hrv_censored, _ = apply_suppressions(result.hrv, analysis.suppressions)
        flags = run_screening(result)
        meta = ReportMeta(
            person_name="Preview (synthetic data)",
            activity_name="Lying down (supine)",
            recorded_at=rec.start_time,
            context_note="Synthetic fixture — committed test data, not a real recording.",
            original_filename=_FIXTURE.name,
            reduced_confidence=result.correction.reduced_confidence,
        )
        _cache["html"] = build_report_html(
            rec, result, hrv_censored, analysis, None, flags, meta
        )
    return Response(_cache["html"], mimetype="text/html")


@bp.get("/report/preview/download")
def preview_download() -> Response:
    resp = preview()
    resp.headers["Content-Disposition"] = (
        f"attachment; filename=ecg-report-preview-{dt.date.today().isoformat()}.html"
    )
    return resp
