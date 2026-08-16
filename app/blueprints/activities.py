"""Activity-type management: list the built-ins, add custom activities.

A custom activity inherits its behaviour (suppressions, derived metrics,
screening limits) from a chosen built-in profile and may override the
expected HR range and the comparison grouping key.
"""

from __future__ import annotations

from flask import Blueprint, Response, flash, redirect, render_template, request, url_for
from sqlalchemy import select

from app.activities import all_profiles
from app.activities.seed import ensure_builtin_activity_types
from app.extensions import db
from app.models import ActivityType

bp = Blueprint("activities", __name__, url_prefix="/activities")


@bp.get("/")
def list_activities() -> str:
    ensure_builtin_activity_types()
    activities = db.session.scalars(
        select(ActivityType).order_by(ActivityType.is_builtin.desc(), ActivityType.name)
    ).all()
    return render_template(
        "activities/list.html", activities=activities, profiles=all_profiles()
    )


@bp.route("/new", methods=["GET", "POST"])
def create() -> str | Response:
    ensure_builtin_activity_types()
    profiles = all_profiles()
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        profile_key = request.form.get("profile_key") or ""
        errors: list[str] = []

        if not name:
            errors.append("Name is required.")
        elif db.session.scalar(select(ActivityType).where(ActivityType.name == name)):
            errors.append(f"An activity named “{name}” already exists.")
        if profile_key not in profiles:
            errors.append("Choose which built-in profile this activity behaves like.")

        hr_min = hr_max = None
        for field, label in (("expected_hr_min", "min"), ("expected_hr_max", "max")):
            raw = (request.form.get(field) or "").strip()
            if raw:
                try:
                    value = int(raw)
                except ValueError:
                    errors.append(f"Expected HR {label} must be a whole number.")
                    continue
                if not 25 <= value <= 250:
                    errors.append(f"Expected HR {label} must be between 25 and 250.")
                elif field == "expected_hr_min":
                    hr_min = value
                else:
                    hr_max = value
        if hr_min is not None and hr_max is not None and hr_min >= hr_max:
            errors.append("Expected HR min must be below max.")

        if errors:
            for e in errors:
                flash(e, "error")
            return render_template("activities/form.html", profiles=profiles, form=request.form)

        comparison_key = (request.form.get("comparison_key") or "").strip() or None
        db.session.add(
            ActivityType(
                name=name,
                profile_key=profile_key,
                is_builtin=False,
                expected_hr_min=hr_min,
                expected_hr_max=hr_max,
                comparison_key=comparison_key,
            )
        )
        db.session.commit()
        flash(f"Added activity “{name}” (behaves like {profiles[profile_key].display_name}).", "ok")
        return redirect(url_for("activities.list_activities"))

    return render_template("activities/form.html", profiles=profiles, form={})
