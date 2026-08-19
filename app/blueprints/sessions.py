"""Session upload, status polling, format mapping, and report serving."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from flask import (
    Blueprint,
    Response,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from sqlalchemy import select

from app.activities.seed import ensure_builtin_activity_types
from app.extensions import db
from app.ingest.loader import CANONICAL_COLUMNS
from app.models import (
    BODY_POSITIONS,
    ActivityType,
    Person,
    ProcessingStatus,
    Session,
    TriggerTag,
    slugify,
)
from app.processing import report_path_for, submit_processing
from app.triggers.seed import ensure_builtin_trigger_tags

bp = Blueprint("sessions", __name__, url_prefix="/sessions")

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _all_trigger_tags() -> list[TriggerTag]:
    return db.session.scalars(
        select(TriggerTag).order_by(TriggerTag.is_builtin.desc(), TriggerTag.name)
    ).all()


def _selected_tags(form) -> list[TriggerTag]:
    """Resolve checked tag ids plus comma-separated custom names to rows.

    Custom names are get-or-create by slug so "Poor Sleep" and "poor sleep"
    land on one tag; brand-new names become non-builtin rows.
    """
    tags: dict[int, TriggerTag] = {}
    for raw_id in form.getlist("trigger_tags"):
        try:
            tag = db.session.get(TriggerTag, int(raw_id))
        except (TypeError, ValueError):
            continue
        if tag is not None:
            tags[tag.id] = tag
    for name in (form.get("new_tags") or "").split(","):
        name = name.strip()
        if not name:
            continue
        slug = slugify(name)
        tag = db.session.scalar(select(TriggerTag).where(TriggerTag.slug == slug))
        if tag is None:
            tag = TriggerTag(name=name[:80], slug=slug)
            db.session.add(tag)
            db.session.flush()
        tags[tag.id] = tag
    return list(tags.values())


def _apply_context_fields(session: Session, form) -> None:
    """Set the optional structured context columns from a submitted form.

    Invalid values are treated as absent rather than rejected — these fields
    are optional context, and a failed upload over a malformed drinks count
    would cost the user a re-upload of a half-gigabyte file.
    """
    session.context_note = (form.get("context_note") or "").strip()[:2000] or None

    position = form.get("body_position") or None
    if position in BODY_POSITIONS:
        session.body_position = position
        session.body_position_source = "user"
    elif not position:
        # Explicit "not recorded" clears a previous user value; an ACC-derived
        # value survives (the accelerometer evidence didn't change).
        if session.body_position_source == "user":
            session.body_position = None
            session.body_position_source = None

    drinks = form.get("alcohol_drinks_24h", type=int)
    session.alcohol_drinks_24h = drinks if drinks is not None and 0 <= drinks <= 30 else None

    quality = form.get("sleep_quality_1_5", type=int)
    session.sleep_quality_1_5 = quality if quality is not None and 1 <= quality <= 5 else None


@bp.route("/upload", methods=["GET", "POST"])
def upload() -> str | Response:
    ensure_builtin_activity_types()
    ensure_builtin_trigger_tags()
    people = db.session.scalars(select(Person).order_by(Person.name)).all()
    activities = db.session.scalars(
        select(ActivityType).order_by(ActivityType.is_builtin.desc(), ActivityType.name)
    ).all()
    trigger_tags = _all_trigger_tags()

    if request.method == "POST":
        errors: list[str] = []
        person = db.session.get(Person, request.form.get("person_id", type=int) or 0)
        activity = db.session.get(
            ActivityType, request.form.get("activity_type_id", type=int) or 0
        )
        upload_file = request.files.get("file")
        if person is None:
            errors.append("Choose the person this session belongs to.")
        if activity is None:
            errors.append("Choose the activity — it decides how the session is analysed.")
        if upload_file is None or not upload_file.filename:
            errors.append("Choose a CSV file to upload.")

        acc_file = request.files.get("acc_file")
        acc_payload: bytes | None = None
        if acc_file is not None and acc_file.filename:
            acc_payload = acc_file.read()
            if not acc_payload:
                errors.append("The accelerometer file is empty.")

        if not errors:
            payload = upload_file.read()
            if not payload:
                errors.append("The file is empty.")
            elif acc_payload is not None and acc_payload == payload:
                errors.append(
                    "The accelerometer file is identical to the ECG file — "
                    "it looks like the ECG export was attached twice."
                )

        if errors:
            for e in errors:
                flash(e, "error")
            return render_template(
                "sessions/upload.html", people=people, activities=activities,
                trigger_tags=trigger_tags, body_positions=BODY_POSITIONS,
                form=request.form,
                selected_tag_ids=request.form.getlist("trigger_tags"),
            )

        sha256 = hashlib.sha256(payload).hexdigest()
        existing = db.session.scalar(select(Session).where(Session.file_sha256 == sha256))
        if existing is not None:
            flash(
                f"This exact file was already uploaded as session {existing.id} "
                f"({existing.original_filename}) for {existing.person.name}.",
                "error",
            )
            return redirect(url_for("sessions.detail", session_id=existing.id))

        original_name = _SAFE_NAME_RE.sub("_", Path(upload_file.filename).name)
        session = Session(
            person_id=person.id,
            activity_type_id=activity.id,
            original_filename=original_name,
            stored_path="",
            file_sha256=sha256,
        )
        _apply_context_fields(session, request.form)
        db.session.add(session)
        db.session.flush()  # allocate session.id for the storage path
        session.trigger_tags = _selected_tags(request.form)

        upload_dir = Path(current_app.config["UPLOAD_DIR"]) / person.slug
        upload_dir.mkdir(parents=True, exist_ok=True)
        stored = upload_dir / f"{session.id}__{original_name}"
        stored.write_bytes(payload)  # path embeds the unique id: never overwrites
        session.stored_path = str(stored)

        if acc_payload is not None:
            acc_name = _SAFE_NAME_RE.sub("_", Path(acc_file.filename).name)
            acc_stored = upload_dir / f"{session.id}__acc__{acc_name}"
            acc_stored.write_bytes(acc_payload)
            session.acc_original_filename = acc_name
            session.acc_stored_path = str(acc_stored)
            session.acc_file_sha256 = hashlib.sha256(acc_payload).hexdigest()
        db.session.commit()

        submit_processing(current_app._get_current_object(), session.id)
        flash("Upload received — analysis is running.", "ok")
        return redirect(url_for("sessions.detail", session_id=session.id))

    return render_template(
        "sessions/upload.html", people=people, activities=activities,
        trigger_tags=trigger_tags, body_positions=BODY_POSITIONS,
        form={}, selected_tag_ids=[],
    )


@bp.get("/<int:session_id>")
def detail(session_id: int) -> str:
    session = db.get_or_404(Session, session_id)
    ensure_builtin_trigger_tags()
    questions = _mapping_questions(session)
    extras = session.metrics.extras if session.metrics else None
    return render_template(
        "sessions/detail.html",
        session=session,
        questions=questions,
        extras=extras or {},
        trigger_tags=_all_trigger_tags(),
        body_positions=BODY_POSITIONS,
        has_report=report_path_for(session).exists() if session.stored_path else False,
    )


@bp.post("/<int:session_id>/tags")
def edit_tags(session_id: int) -> Response:
    session = db.get_or_404(Session, session_id)
    session.trigger_tags = _selected_tags(request.form)
    db.session.commit()
    flash("Trigger tags updated.", "ok")
    return redirect(url_for("sessions.detail", session_id=session.id))


@bp.post("/<int:session_id>/context")
def edit_context(session_id: int) -> Response:
    """Edit the structured context fields and note after upload.

    Database-driven views reflect the change immediately; the stored report
    HTML reflects processing time and refreshes on re-analyse.
    """
    session = db.get_or_404(Session, session_id)
    _apply_context_fields(session, request.form)
    db.session.commit()
    flash("Context updated. The stored report refreshes on re-analyse.", "ok")
    return redirect(url_for("sessions.detail", session_id=session.id))


@bp.get("/<int:session_id>/status")
def status(session_id: int) -> Response:
    session = db.get_or_404(Session, session_id)
    payload: dict = {
        "id": session.id,
        "status": session.processing_status,
    }
    if session.processing_status == ProcessingStatus.ERROR:
        payload["error"] = session.error_message
    if session.processing_status == ProcessingStatus.NEEDS_MAPPING:
        parsed = _mapping_payload(session)
        payload["message"] = parsed.get("message") if parsed else None
        payload["questions"] = parsed.get("questions", []) if parsed else []
    return jsonify(payload)


@bp.route("/<int:session_id>/mapping", methods=["GET", "POST"])
def mapping(session_id: int) -> str | Response:
    session = db.get_or_404(Session, session_id)
    questions = _mapping_questions(session)
    if not questions:
        flash("This session isn't waiting on format answers.", "error")
        return redirect(url_for("sessions.detail", session_id=session.id))

    if request.method == "POST":
        overrides: dict = dict(session.format_overrides or {})
        if request.form.get("delimiter"):
            overrides["delimiter"] = (
                "\t" if request.form["delimiter"] == "tab" else request.form["delimiter"]
            )
        if request.form.get("decimal"):
            overrides["decimal"] = request.form["decimal"]
        if request.form.get("ecg_unit"):
            overrides["ecg_unit"] = request.form["ecg_unit"]
        if request.form.get("epoch"):
            overrides["epoch"] = request.form["epoch"]
        column_map = {}
        for key, value in request.form.items():
            if key.startswith("col__") and value and value != "ignore":
                column_map[key[len("col__") :]] = value
        if column_map:
            overrides["column_map"] = column_map

        session.format_overrides = overrides
        session.processing_status = ProcessingStatus.PENDING
        session.error_message = None
        db.session.commit()
        submit_processing(current_app._get_current_object(), session.id)
        flash("Answers saved — re-analysing with your mapping.", "ok")
        return redirect(url_for("sessions.detail", session_id=session.id))

    return render_template(
        "sessions/mapping.html",
        session=session,
        questions=questions,
        canonical_columns=CANONICAL_COLUMNS,
    )


@bp.post("/<int:session_id>/reprocess")
def reprocess(session_id: int) -> Response:
    session = db.get_or_404(Session, session_id)
    session.processing_status = ProcessingStatus.PENDING
    session.error_message = None
    db.session.commit()
    submit_processing(current_app._get_current_object(), session.id)
    flash("Re-analysis started.", "ok")
    return redirect(url_for("sessions.detail", session_id=session.id))


@bp.get("/<int:session_id>/report")
def report(session_id: int) -> Response:
    return _serve_report(session_id, download=False)


@bp.get("/<int:session_id>/report/download")
def report_download(session_id: int) -> Response:
    return _serve_report(session_id, download=True)


def _serve_report(session_id: int, download: bool) -> Response:
    session = db.get_or_404(Session, session_id)
    path = report_path_for(session)
    if not path.exists():
        flash("No report yet — the session hasn't finished processing.", "error")
        return redirect(url_for("sessions.detail", session_id=session.id))
    resp = Response(path.read_text(encoding="utf-8"), mimetype="text/html")
    if download:
        stamp = session.recorded_at.strftime("%Y%m%d") if session.recorded_at else "report"
        resp.headers["Content-Disposition"] = (
            f"attachment; filename=ecg-report-session{session.id}-{stamp}.html"
        )
    return resp


def _mapping_payload(session: Session) -> dict | None:
    if session.processing_status != ProcessingStatus.NEEDS_MAPPING:
        return None
    try:
        return json.loads(session.error_message or "")
    except (ValueError, TypeError):
        return None


def _mapping_questions(session: Session) -> list[dict]:
    payload = _mapping_payload(session)
    return payload.get("questions", []) if payload else []
