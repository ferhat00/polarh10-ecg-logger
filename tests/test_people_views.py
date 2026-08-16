"""Smoke tests for the person CRUD views."""

from __future__ import annotations

from flask import Flask
from flask.testing import FlaskClient

from app.extensions import db
from app.models import Person


def test_home_renders_empty_state(client: FlaskClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"No people yet" in resp.data


def test_create_person_flow(client: FlaskClient, app: Flask) -> None:
    resp = client.post(
        "/people/new",
        data={
            "name": "Ferhat",
            "date_of_birth": "1980-05-01",
            "max_hr_bpm": "185",
            "resting_hr_bpm": "52",
            "athlete_baseline": "on",
        },
        follow_redirects=True,
    )
    assert resp.status_code == 200
    person = db.session.query(Person).one()
    assert person.slug == "ferhat"
    assert person.max_hr_bpm == 185
    assert person.athlete_baseline is True


def test_create_person_requires_name(client: FlaskClient, app: Flask) -> None:
    resp = client.post("/people/new", data={"name": "  "}, follow_redirects=True)
    assert b"Name is required" in resp.data
    assert db.session.query(Person).count() == 0


def test_duplicate_name_shows_error(client: FlaskClient, app: Flask) -> None:
    client.post("/people/new", data={"name": "Sam"})
    resp = client.post("/people/new", data={"name": "Sam"}, follow_redirects=True)
    assert b"already exists" in resp.data
    assert db.session.query(Person).count() == 1


def test_edit_person(client: FlaskClient, app: Flask) -> None:
    client.post("/people/new", data={"name": "Sam"})
    person = db.session.query(Person).one()
    client.post(
        f"/people/{person.id}/edit",
        data={"name": "Sam", "resting_hr_bpm": "58"},
    )
    db.session.expire_all()
    assert db.session.query(Person).one().resting_hr_bpm == 58


def test_delete_requires_name_confirmation(client: FlaskClient, app: Flask) -> None:
    client.post("/people/new", data={"name": "Sam"})
    person = db.session.query(Person).one()

    client.post(f"/people/{person.id}/delete", data={"confirm_name": "wrong"})
    assert db.session.query(Person).count() == 1

    client.post(f"/people/{person.id}/delete", data={"confirm_name": "Sam"})
    assert db.session.query(Person).count() == 0


def test_invalid_hr_rejected(client: FlaskClient, app: Flask) -> None:
    resp = client.post(
        "/people/new",
        data={"name": "Pat", "max_hr_bpm": "999"},
        follow_redirects=True,
    )
    assert b"between 100 and 250" in resp.data
    assert db.session.query(Person).count() == 0
