"""Person CRUD.

People are profile separation, not accounts — several people can share one
chest strap. Deleting a person deletes their sessions (with confirmation).
"""

from __future__ import annotations

import datetime as dt

from flask import Blueprint, Response, flash, redirect, render_template, request, url_for
from sqlalchemy import select

from app.extensions import db
from app.models import Person, slugify

bp = Blueprint("people", __name__, url_prefix="/people")


def _unique_slug(name: str, exclude_id: int | None = None) -> str:
    base = slugify(name)
    slug = base
    n = 2
    while True:
        existing = db.session.scalar(select(Person).where(Person.slug == slug))
        if existing is None or existing.id == exclude_id:
            return slug
        slug = f"{base}-{n}"
        n += 1


def _apply_form(person: Person, form: dict[str, str]) -> list[str]:
    """Apply submitted form fields to a Person; return validation errors."""
    errors: list[str] = []

    name = (form.get("name") or "").strip()
    if not name:
        errors.append("Name is required.")
    else:
        clash = db.session.scalar(select(Person).where(Person.name == name))
        if clash is not None and clash.id != person.id:
            errors.append(f"A person named “{name}” already exists.")
        person.name = name

    dob_raw = (form.get("date_of_birth") or "").strip()
    if dob_raw:
        try:
            person.date_of_birth = dt.date.fromisoformat(dob_raw)
        except ValueError:
            errors.append("Date of birth must be YYYY-MM-DD.")
    else:
        person.date_of_birth = None

    for field, label, lo, hi in (
        ("max_hr_bpm", "Max HR", 100, 250),
        ("resting_hr_bpm", "Resting HR", 25, 120),
    ):
        raw = (form.get(field) or "").strip()
        if not raw:
            setattr(person, field, None)
            continue
        try:
            value = int(raw)
        except ValueError:
            errors.append(f"{label} must be a whole number.")
            continue
        if not lo <= value <= hi:
            errors.append(f"{label} must be between {lo} and {hi} bpm.")
        else:
            setattr(person, field, value)

    person.athlete_baseline = form.get("athlete_baseline") == "on"

    sex = (form.get("sex") or "").strip().lower()
    if sex in ("male", "female"):
        person.sex = sex
    elif sex == "":
        person.sex = None
    else:
        errors.append("Sex must be male, female, or left unset.")
    return errors


@bp.get("/")
def list_people() -> str:
    people = db.session.scalars(select(Person).order_by(Person.name)).all()
    return render_template("people/list.html", people=people)


@bp.route("/new", methods=["GET", "POST"])
def create() -> str | Response:
    person = Person(name="", slug="")
    if request.method == "POST":
        errors = _apply_form(person, request.form)
        if errors:
            for e in errors:
                flash(e, "error")
            return render_template("people/form.html", person=person, mode="new")
        person.slug = _unique_slug(person.name)
        db.session.add(person)
        db.session.commit()
        flash(f"Added {person.name}.", "ok")
        return redirect(url_for("people.detail", person_id=person.id))
    return render_template("people/form.html", person=person, mode="new")


@bp.get("/<int:person_id>")
def detail(person_id: int) -> str:
    person = db.get_or_404(Person, person_id)
    return render_template("people/detail.html", person=person)


@bp.route("/<int:person_id>/edit", methods=["GET", "POST"])
def edit(person_id: int) -> str | Response:
    person = db.get_or_404(Person, person_id)
    if request.method == "POST":
        errors = _apply_form(person, request.form)
        if errors:
            db.session.rollback()
            for e in errors:
                flash(e, "error")
            return render_template("people/form.html", person=person, mode="edit")
        # The slug is part of stored file paths; keep it stable once created.
        db.session.commit()
        flash("Saved.", "ok")
        return redirect(url_for("people.detail", person_id=person.id))
    return render_template("people/form.html", person=person, mode="edit")


@bp.post("/<int:person_id>/delete")
def delete(person_id: int) -> Response:
    person = db.get_or_404(Person, person_id)
    if request.form.get("confirm_name") != person.name:
        flash("Type the person's name exactly to confirm deletion.", "error")
        return redirect(url_for("people.detail", person_id=person.id))
    name = person.name
    db.session.delete(person)
    db.session.commit()
    flash(f"Deleted {name} and all their sessions.", "ok")
    return redirect(url_for("people.list_people"))
