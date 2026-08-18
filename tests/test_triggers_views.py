"""Trigger dashboard and tag-management views."""

from __future__ import annotations

import datetime as dt

import pytest
from flask import Flask
from flask.testing import FlaskClient

from app.extensions import db
from app.models import Metrics, Person, ProcessingStatus, Session, TriggerTag
from app.screening.flags import BANNED_PHRASES
from app.triggers.seed import ensure_builtin_trigger_tags

BASE_T = dt.datetime(2026, 6, 1, 8, 0)


@pytest.fixture()
def person(app: Flask) -> Person:
    p = Person(name="Trig", slug="trig")
    db.session.add(p)
    db.session.commit()
    return p


def _fake_done_session(
    person: Person,
    i: int,
    ectopy_n: int | None,
    tags: list[TriggerTag] = (),
    analysed_s: float = 3600.0,
) -> Session:
    s = Session(
        person_id=person.id,
        original_filename=f"s{i}.csv",
        stored_path=f"/nonexistent/s{i}.csv",
        file_sha256=f"{i:064d}",
        processing_status=ProcessingStatus.DONE,
        analysed_s=analysed_s,
        duration_s=analysed_s,
        recorded_at=BASE_T + dt.timedelta(days=i, hours=i % 5),
    )
    db.session.add(s)
    db.session.flush()
    s.trigger_tags = list(tags)
    db.session.add(
        Metrics(
            session_id=s.id,
            ectopy_beats_n=ectopy_n,
            ectopy_per_hour=(
                ectopy_n / (analysed_s / 3600.0) if ectopy_n is not None else None
            ),
        )
    )
    db.session.commit()
    return s


class TestDashboard:
    def test_empty_person_renders_honestly(
        self, client: FlaskClient, person: Person
    ) -> None:
        page = client.get(f"/triggers/{person.id}")
        assert page.status_code == 200
        assert b"No usable sessions yet" in page.data

    def test_insufficient_data_row_is_honest(
        self, client: FlaskClient, app: Flask, person: Person
    ) -> None:
        ensure_builtin_trigger_tags()
        caffeine = db.session.query(TriggerTag).filter_by(slug="caffeine").one()
        for i in range(3):
            _fake_done_session(person, i, ectopy_n=4, tags=[caffeine])
        for i in range(3, 6):
            _fake_done_session(person, i, ectopy_n=2)

        page = client.get(f"/triggers/{person.id}")
        assert page.status_code == 200
        html = page.data.decode()
        assert "Caffeine" in html
        assert "insufficient" in html.lower() or "Needs at least" in html

    def test_full_inference_renders_rate_ratio(
        self, client: FlaskClient, app: Flask, person: Person
    ) -> None:
        ensure_builtin_trigger_tags()
        caffeine = db.session.query(TriggerTag).filter_by(slug="caffeine").one()
        for i in range(6):
            _fake_done_session(person, i, ectopy_n=12 + i, tags=[caffeine])
        for i in range(6, 12):
            _fake_done_session(person, i, ectopy_n=4 + (i % 3))

        page = client.get(f"/triggers/{person.id}")
        assert page.status_code == 200
        html = page.data.decode()
        assert "Rate ratio" in html
        assert "How to read this" in html
        # An actual estimate row rendered (bolded ratio value).
        assert "<strong>2." in html or "<strong>3." in html

    def test_no_diagnostic_language(
        self, client: FlaskClient, app: Flask, person: Person
    ) -> None:
        ensure_builtin_trigger_tags()
        caffeine = db.session.query(TriggerTag).filter_by(slug="caffeine").one()
        for i in range(6):
            _fake_done_session(person, i, ectopy_n=10, tags=[caffeine])
        for i in range(6, 12):
            _fake_done_session(person, i, ectopy_n=5)

        import re

        html = client.get(f"/triggers/{person.id}").data.decode()
        # Base64 figure payloads are random bytes — drop them before the scan.
        html = re.sub(r'src="data:image/png;base64,[^"]*"', "", html).lower()
        # "diagnos" appears only inside the sanctioned negation ("not a
        # diagnosis, and this tool cannot make one"); every other banned
        # phrase must be absent outright.
        for phrase in BANNED_PHRASES:
            if phrase == "diagnos":
                continue
            assert phrase not in html, f"banned phrase {phrase!r} in dashboard"
        assert "not a diagnosis" in html

    def test_awaiting_reprocess_card(
        self, client: FlaskClient, app: Flask, person: Person
    ) -> None:
        s = _fake_done_session(person, 0, ectopy_n=None)
        page = client.get(f"/triggers/{person.id}")
        html = page.data.decode()
        assert "await re-analysis" in html
        assert f"/sessions/{s.id}/reprocess" in html

    def test_index_lists_people(self, client: FlaskClient, person: Person) -> None:
        page = client.get("/triggers/")
        assert page.status_code == 200
        assert b"Trig" in page.data


class TestManageTags:
    def test_add_and_delete_custom_tag(self, client: FlaskClient, app: Flask) -> None:
        resp = client.post("/triggers/tags", data={"name": "Cold Shower"})
        assert resp.status_code == 302
        tag = db.session.query(TriggerTag).filter_by(slug="cold-shower").one()
        assert tag.is_builtin is False

        page = client.get("/triggers/tags")
        assert b"Cold Shower" in page.data

        resp = client.post(f"/triggers/tags/{tag.id}/delete")
        assert resp.status_code == 302
        assert db.session.query(TriggerTag).filter_by(slug="cold-shower").count() == 0

    def test_duplicate_name_rejected(self, client: FlaskClient, app: Flask) -> None:
        client.post("/triggers/tags", data={"name": "Cold Shower"})
        client.post("/triggers/tags", data={"name": "cold shower"})
        assert db.session.query(TriggerTag).filter_by(slug="cold-shower").count() == 1

    def test_builtin_not_deletable(self, client: FlaskClient, app: Flask) -> None:
        ensure_builtin_trigger_tags()
        tag = db.session.query(TriggerTag).filter_by(slug="caffeine").one()
        client.post(f"/triggers/tags/{tag.id}/delete")
        assert db.session.query(TriggerTag).filter_by(slug="caffeine").count() == 1

    def test_tag_in_use_not_deletable(
        self, client: FlaskClient, app: Flask, person: Person
    ) -> None:
        client.post("/triggers/tags", data={"name": "Cold Shower"})
        tag = db.session.query(TriggerTag).filter_by(slug="cold-shower").one()
        _fake_done_session(person, 0, ectopy_n=1, tags=[tag])
        client.post(f"/triggers/tags/{tag.id}/delete")
        assert db.session.query(TriggerTag).filter_by(slug="cold-shower").count() == 1
