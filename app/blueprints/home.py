"""Home page."""

from __future__ import annotations

from flask import Blueprint, render_template
from sqlalchemy import select

from app.extensions import db
from app.models import Person

bp = Blueprint("home", __name__)


@bp.get("/")
def index() -> str:
    people = db.session.scalars(select(Person).order_by(Person.name)).all()
    return render_template("index.html", people=people)
