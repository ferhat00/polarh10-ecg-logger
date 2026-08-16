"""Decision-log timeline and user annotations."""

from __future__ import annotations

import re

from flask import Blueprint, Response, flash, redirect, render_template, request, url_for
from markupsafe import Markup, escape

from app.extensions import db
from app.logbook.writer import append_annotation_entry, read_entries
from app.models import Annotation, Person

bp = Blueprint("logbook", __name__, url_prefix="/log")

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def _entry_to_html(entry: str) -> Markup:
    """Minimal renderer for the log's constrained Markdown (no external lib)."""
    lines = entry.strip().splitlines()
    html: list[str] = []
    in_list = False
    for line in lines:
        text = str(escape(line))
        text = _BOLD_RE.sub(r"<strong>\1</strong>", text)
        if line.startswith("## "):
            html.append(f"<h3>{text[3:]}</h3>")
        elif line.startswith("- "):
            if not in_list:
                html.append("<ul>")
                in_list = True
            html.append(f"<li>{text[2:]}</li>")
        else:
            if in_list:
                html.append("</ul>")
                in_list = False
            if line.strip():
                html.append(f"<p>{text}</p>")
    if in_list:
        html.append("</ul>")
    return Markup("\n".join(html))


@bp.get("/<int:person_id>")
def timeline(person_id: int) -> str:
    person = db.get_or_404(Person, person_id)
    entries = [_entry_to_html(e) for e in reversed(read_entries(person))]
    return render_template("logbook/timeline.html", person=person, entries=entries)


@bp.post("/<int:person_id>/annotate")
def annotate(person_id: int) -> Response:
    person = db.get_or_404(Person, person_id)
    text = (request.form.get("text") or "").strip()
    session_id = request.form.get("session_id", type=int)
    if not text:
        flash("Write something first.", "error")
        return redirect(url_for("logbook.timeline", person_id=person.id))

    db.session.add(Annotation(person_id=person.id, session_id=session_id, text=text))
    db.session.commit()
    append_annotation_entry(person, text, session_id=session_id)
    flash("Annotation added to the log.", "ok")
    return redirect(url_for("logbook.timeline", person_id=person.id))
