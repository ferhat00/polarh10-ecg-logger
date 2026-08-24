"""Sleep history: every staged night, and re-staging them in bulk.

The session page is where one night is read; this is where a whole history
is worked through — "I just installed the SleepECG extras, re-run my last
three months with it". Re-analysis is delegated to the same reprocess path
the session page uses, so there is one validation rule and one queue.
"""

from __future__ import annotations

from flask import (
    Blueprint,
    Response,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from sqlalchemy import select

from app.activities.registry import requests_sleep_staging
from app.blueprints.sessions import apply_sleep_engine_pref
from app.extensions import db
from app.models import Person, ProcessingStatus, Session
from app.processing import queue_depth, submit_processing
from app.sleep.engines.registry import engine_statuses, vocab_summary

bp = Blueprint("sleep", __name__, url_prefix="/sleep")


#: Dropdown sentinel for "don't touch each night's stored preference" — the
#: form equivalent of omitting the field on the reprocess route.
KEEP = "keep"


def _nights(person_id: int | None) -> list[Session]:
    """Staged nights, newest first, optionally for one person.

    Includes sessions in every status, not just ``done``: a night that
    errored or is still queued is exactly the one a user wants to find here.
    """
    stmt = select(Session)
    if person_id is not None:
        stmt = stmt.where(Session.person_id == person_id)
    nights = [
        s for s in db.session.scalars(stmt) if requests_sleep_staging(s.activity_type)
    ]
    # Sorted here rather than in SQL: recorded_at is NULL until a session has
    # been processed once, and NULLS LAST ordering is not portable. Ranked on
    # a float timestamp so a naive and an aware datetime never get compared.
    nights.sort(key=_recency, reverse=True)
    return nights


def _recency(session: Session) -> tuple[int, float, int]:
    if session.recorded_at is None:
        return (0, 0.0, session.id)
    return (1, session.recorded_at.timestamp(), session.id)


@bp.get("/")
def history() -> str:
    person_id = request.args.get("person", type=int)
    people = db.session.scalars(select(Person).order_by(Person.name)).all()
    engines = engine_statuses(current_app.config)
    return render_template(
        "sleep/history.html",
        nights=_nights(person_id),
        people=people,
        person_id=person_id,
        sleep_engines=engines,
        engine_vocabs={s.key: vocab_summary(s.key) for s in engines},
        auto_label=next((s.label for s in engines if s.available), None),
        queued=queue_depth(),
    )


@bp.post("/reanalyse")
def reanalyse() -> Response:
    """Set one algorithm on the checked nights and queue them all.

    Validation runs once, against the first selected night, *before* any
    session is touched: a rejected algorithm must leave the history exactly
    as it was rather than half-applied. The queue in :mod:`app.processing`
    then drains them one at a time — with the external engine configured a
    night can hold a subprocess for half an hour, so N nights must never
    mean N concurrent runs.
    """
    person_id = request.form.get("person", type=int)
    back = redirect(url_for("sleep.history", person=person_id))

    ids = request.form.getlist("sessions", type=int)
    sessions = [s for s in _nights(person_id) if s.id in set(ids)]
    if not sessions:
        flash("No nights were selected.", "error")
        return back

    raw = request.form.get("sleep_engine", KEEP)
    if raw != KEEP:
        error = apply_sleep_engine_pref(sessions[0], raw)
        if error:
            db.session.rollback()
            flash(error, "error")
            return back
        for session in sessions[1:]:
            # Already validated above; it cannot fail for a sibling night.
            apply_sleep_engine_pref(session, raw)
    for session in sessions:
        session.processing_status = ProcessingStatus.PENDING
        session.error_message = None
    db.session.commit()

    app = current_app._get_current_object()
    for session in sessions:
        submit_processing(app, session.id)

    chosen = (
        "each night's own algorithm"
        if raw == KEEP
        else (sessions[0].sleep_engine_pref or "automatic (best available)")
    )
    flash(
        f"{len(sessions)} night(s) queued for re-analysis with {chosen} — "
        "they run one at a time.",
        "ok",
    )
    return back
