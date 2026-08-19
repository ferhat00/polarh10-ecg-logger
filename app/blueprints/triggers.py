"""Trigger dashboard: ectopy rates by tag, with honest inference and guards."""

from __future__ import annotations

import numpy as np
from flask import Blueprint, Response, flash, redirect, render_template, request, url_for
from sqlalchemy import select

from app.extensions import db
from app.models import Person, Session, TriggerTag, slugify
from app.processing import load_cached_events
from app.triggers import figures as trigger_figures
from app.triggers.seed import ensure_builtin_trigger_tags
from app.triggers.stats import (
    OUTCOMES,
    analyse_triggers,
    env_associations,
    hour_of_day_profile,
    rates_by_tag,
)

bp = Blueprint("triggers", __name__, url_prefix="/triggers")


@bp.get("/")
def index() -> str:
    people = db.session.scalars(select(Person).order_by(Person.name)).all()
    return render_template("triggers/index.html", people=people)


@bp.get("/<int:person_id>")
def dashboard(person_id: int) -> str:
    person = db.get_or_404(Person, person_id)
    outcome_key = request.args.get("outcome", "ectopy")
    if outcome_key not in OUTCOMES:
        flash(f"Unknown outcome {outcome_key!r} — showing ectopy burden.", "error")
        outcome_key = "ectopy"
    analysis = analyse_triggers(person, outcome=outcome_key)

    names = {
        t.slug: t.name
        for t in db.session.scalars(select(TriggerTag)).all()
    }
    # The burden timeline, tagged-vs-untagged strip, and hour profile are
    # ectopy-rate figures; other outcomes show the forest + table + env.
    figures = {
        "timeline": None,
        "by_tag": None,
        "hour": None,
        "forest": trigger_figures.forest(analysis.effects, analysis.outcome),
    }
    profile = None
    if outcome_key == "ectopy":
        figures["timeline"] = trigger_figures.burden_timeline(analysis.observations)
        figures["by_tag"] = trigger_figures.burden_by_tag(
            rates_by_tag(analysis.observations), names
        )
        profile = hour_of_day_profile(
            analysis.observations, _event_offsets(person, analysis)
        )
        figures["hour"] = trigger_figures.hour_profile_figure(profile)

    env_assocs = env_associations(analysis.observations, analysis.outcome)
    env_figures = {
        a.var_key: trigger_figures.env_scatter(a, analysis.outcome)
        for a in env_assocs
    }

    awaiting = (
        db.session.scalars(
            select(Session).where(Session.id.in_(analysis.notes.awaiting_ids))
        ).all()
        if analysis.notes.awaiting_ids
        else []
    )
    from app.screening.flags import DISCLAIMER

    return render_template(
        "triggers/dashboard.html",
        person=person,
        analysis=analysis,
        outcomes=OUTCOMES,
        outcome=analysis.outcome,
        figures=figures,
        env_assocs=env_assocs,
        env_figures=env_figures,
        profile=profile,
        awaiting=awaiting,
        disclaimer=DISCLAIMER,
    )


def _event_offsets(person: Person, analysis) -> dict[int, np.ndarray]:
    """session id → event start offsets (s), for sessions with a v2 cache."""
    by_id = {s.id: s for s in person.sessions}
    offsets: dict[int, np.ndarray] = {}
    for obs in analysis.observations:
        session = by_id.get(obs.session_id)
        if session is None:
            continue
        cached = load_cached_events(session)
        if cached is not None:
            offsets[obs.session_id] = cached["event_t_start_s"]
    return offsets


@bp.route("/tags", methods=["GET", "POST"])
def manage_tags() -> str | Response:
    ensure_builtin_trigger_tags()
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        if not name:
            flash("Give the tag a name.", "error")
        else:
            slug = slugify(name)
            existing = db.session.scalar(
                select(TriggerTag).where(TriggerTag.slug == slug)
            )
            if existing is not None:
                flash(f"A tag with this name already exists: {existing.name}.", "error")
            else:
                db.session.add(TriggerTag(name=name[:80], slug=slug))
                db.session.commit()
                flash("Tag added.", "ok")
        return redirect(url_for("triggers.manage_tags"))

    tags = db.session.scalars(
        select(TriggerTag).order_by(TriggerTag.is_builtin.desc(), TriggerTag.name)
    ).all()
    counts = {tag.id: len(tag.sessions) for tag in tags}
    return render_template("triggers/tags.html", tags=tags, counts=counts)


@bp.post("/tags/<int:tag_id>/delete")
def delete_tag(tag_id: int) -> Response:
    tag = db.get_or_404(TriggerTag, tag_id)
    in_use = len(tag.sessions)
    if tag.is_builtin:
        flash("Built-in tags cannot be deleted.", "error")
    elif in_use:
        flash(
            f"“{tag.name}” is attached to {in_use} session(s); untag them first.",
            "error",
        )
    else:
        db.session.delete(tag)
        db.session.commit()
        flash("Tag deleted.", "ok")
    return redirect(url_for("triggers.manage_tags"))
