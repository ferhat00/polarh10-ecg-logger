"""Polish-phase tests: error states and chrome."""

from __future__ import annotations

from flask.testing import FlaskClient


def test_404_renders_friendly_page(client: FlaskClient) -> None:
    resp = client.get("/sessions/99999")
    assert resp.status_code == 404
    assert b"Nothing at this address" in resp.data


def test_404_unknown_route(client: FlaskClient) -> None:
    resp = client.get("/definitely/not/a/route")
    assert resp.status_code == 404


def test_home_quick_actions_present_with_people(client: FlaskClient, app) -> None:
    from app.extensions import db
    from app.models import Person

    db.session.add(Person(name="Homer", slug="homer"))
    db.session.commit()
    resp = client.get("/")
    assert b"Upload a recording" in resp.data
    assert b"Compare sessions" in resp.data


def test_disclaimer_strip_on_every_page(client: FlaskClient) -> None:
    for path in ("/", "/people/", "/activities/", "/compare/", "/sessions/upload"):
        resp = client.get(path)
        assert b"not a diagnostic device" in resp.data.lower(), path
