"""Comparison and longitudinal-trend views."""

from __future__ import annotations

from collections import defaultdict

from flask import Blueprint, Response, flash, redirect, render_template, request, url_for
from sqlalchemy import select

from app.activities import metrics as activity_metrics
from app.comparison import (
    ComparisonError,
    build_comparison,
    build_trend,
    comparison_key_of,
    load_cached_rr,
)
from app.extensions import db
from app.models import Person, Session
from app.report import compare_figures

bp = Blueprint("compare", __name__, url_prefix="/compare")


@bp.get("/")
def index() -> str:
    people = db.session.scalars(select(Person).order_by(Person.name)).all()
    return render_template("compare/index.html", people=people)


@bp.get("/<int:person_id>")
def select_sessions(person_id: int) -> str:
    person = db.get_or_404(Person, person_id)
    groups: dict[str, list[Session]] = defaultdict(list)
    ungrouped: list[Session] = []
    for session in person.sessions:
        if session.processing_status != "done":
            continue
        key = comparison_key_of(session)
        (groups[key] if key else ungrouped).append(session)
    for sessions in groups.values():
        sessions.sort(key=lambda s: s.recorded_at or s.created_at)
    return render_template(
        "compare/select.html",
        person=person,
        groups=dict(sorted(groups.items())),
        ungrouped=ungrouped,
    )


@bp.get("/<int:person_id>/view")
def view(person_id: int) -> str | Response:
    person = db.get_or_404(Person, person_id)
    ids = request.args.getlist("sessions", type=int)
    mixed = request.args.get("mixed") == "on"
    sessions = [s for s in person.sessions if s.id in set(ids)]

    try:
        data = build_comparison(sessions, mixed=mixed)
    except ComparisonError as exc:
        flash(str(exc), "error")
        return redirect(url_for("compare.select_sessions", person_id=person.id))

    hr_series = []
    rmssd_series = []
    for s in data.sessions:
        rr = load_cached_rr(s)
        if rr is None:
            continue
        label = _session_label(s)
        starts, means = activity_metrics.per_minute_hr(rr)
        if means:
            hr_series.append((label, starts, means))
        r_starts, r_values = activity_metrics.per_minute_rmssd(rr)
        if r_values:
            rmssd_series.append((label, r_starts, r_values))

    figures = {
        "hr": compare_figures.overlay_series(
            hr_series, "bpm", "Per-minute mean HR, overlaid by session."
        ),
        "rmssd": compare_figures.overlay_series(
            rmssd_series, "RMSSD (ms)", "Per-minute RMSSD, overlaid by session."
        ),
    }
    return render_template(
        "compare/view.html", person=person, data=data, figures=figures
    )


@bp.get("/<int:person_id>/trends/<comparison_key>")
def trends(person_id: int, comparison_key: str) -> str:
    person = db.get_or_404(Person, person_id)
    sessions = [
        s for s in person.sessions if comparison_key_of(s) == comparison_key
    ]
    data = build_trend(sessions, comparison_key)

    dates = [p.recorded_at for p in data.points]
    figures = {}
    if data.resting:
        band = (
            (data.resting.band_mean, data.resting.band_lo, data.resting.band_hi)
            if data.resting.band_mean
            else None
        )
        figures["resting"] = compare_figures.trend_figure(
            dates,
            data.resting.values,
            band,
            "bpm",
            "Resting HR (lowest sustained 60 s) per session over time.",
        )
    if data.ln_rmssd:
        band = (
            (data.ln_rmssd.band_mean, data.ln_rmssd.band_lo, data.ln_rmssd.band_hi)
            if data.ln_rmssd.band_mean
            else None
        )
        figures["ln_rmssd"] = compare_figures.trend_figure(
            dates,
            data.ln_rmssd.values,
            band,
            "ln(RMSSD ms)",
            "ln(RMSSD) per session over time — the trend-tracking form; raw RMSSD "
            "is right-skewed.",
        )
    return render_template(
        "compare/trends.html", person=person, data=data, figures=figures
    )


def _session_label(session: Session) -> str:
    when = (
        session.recorded_at.strftime("%Y-%m-%d")
        if session.recorded_at
        else f"session {session.id}"
    )
    return f"#{session.id} {when}"
