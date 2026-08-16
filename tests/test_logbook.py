"""Decision-log tests: append-only semantics, entry format, timeline UI."""

from __future__ import annotations

from flask import Flask
from flask.testing import FlaskClient

from app.extensions import db
from app.logbook.writer import (
    append_annotation_entry,
    log_path_for,
    read_entries,
)
from app.models import Annotation, Person, Session
from tests.test_upload import make_csv_bytes


def _upload(client: FlaskClient, person_id: int, activity_id: int, payload: bytes,
            name: str = "log_2026-08-10.csv") -> None:
    import io

    client.post(
        "/sessions/upload",
        data={
            "person_id": str(person_id),
            "activity_type_id": str(activity_id),
            "context_note": "browsing library and shop",
            "file": (io.BytesIO(payload), name),
        },
        content_type="multipart/form-data",
    )


def _setup(client: FlaskClient) -> tuple[Person, int]:
    person = Person(name="Logged", slug="logged")
    db.session.add(person)
    db.session.commit()
    client.get("/sessions/upload")  # seed activities
    from app.models import ActivityType

    activity = db.session.query(ActivityType).filter_by(profile_key="sitting").one()
    return person, activity.id


class TestSessionEntries:
    def test_completed_analysis_appends_entry(
        self, client: FlaskClient, app: Flask
    ) -> None:
        person, activity_id = _setup(client)
        _upload(client, person.id, activity_id, make_csv_bytes())

        path = log_path_for(person)
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        session = db.session.query(Session).one()
        assert f"— Sitting — session {session.id}" in text
        assert "- **Duration:**" in text and "fs 130.03 Hz" in text
        assert "- **Quality:**" in text and "beats corrected" in text
        assert "- **Key metrics:**" in text and "trend-inclusive" in text
        assert "- **Flags:** none raised" in text
        assert "- **Context:** browsing library and shop" in text
        assert "- **Engine:** neurokit2 · Lipponen-Tarvainen correction" in text

    def test_entries_append_in_order_never_rewritten(
        self, client: FlaskClient, app: Flask
    ) -> None:
        person, activity_id = _setup(client)
        _upload(client, person.id, activity_id, make_csv_bytes(seed=1), name="a.csv")
        path = log_path_for(person)
        first_content = path.read_text(encoding="utf-8")

        _upload(client, person.id, activity_id, make_csv_bytes(seed=2), name="b.csv")
        second_content = path.read_text(encoding="utf-8")

        # Strictly append-only: the old content is an untouched prefix.
        assert second_content.startswith(first_content)
        assert len(read_entries(person)) == 2


class TestAnnotations:
    def test_annotation_appends_and_persists(
        self, client: FlaskClient, app: Flask
    ) -> None:
        person, _ = _setup(client)
        resp = client.post(
            f"/log/{person.id}/annotate",
            data={"text": "started a new training block"},
            follow_redirects=True,
        )
        assert resp.status_code == 200
        assert db.session.query(Annotation).count() == 1
        text = log_path_for(person).read_text(encoding="utf-8")
        assert "— Annotation" in text
        assert "started a new training block" in text

    def test_annotation_with_session_reference(
        self, client: FlaskClient, app: Flask
    ) -> None:
        person, activity_id = _setup(client)
        _upload(client, person.id, activity_id, make_csv_bytes())
        session = db.session.query(Session).one()
        client.post(
            f"/log/{person.id}/annotate",
            data={"text": "felt dizzy during this one", "session_id": str(session.id)},
        )
        text = log_path_for(person).read_text(encoding="utf-8")
        assert f"— Annotation — session {session.id}" in text

    def test_empty_annotation_rejected(self, client: FlaskClient, app: Flask) -> None:
        person, _ = _setup(client)
        client.post(f"/log/{person.id}/annotate", data={"text": "   "})
        assert db.session.query(Annotation).count() == 0
        assert not log_path_for(person).exists() or (
            "Annotation" not in log_path_for(person).read_text(encoding="utf-8")
        )


class TestTimelineView:
    def test_timeline_renders_entries_latest_first(
        self, client: FlaskClient, app: Flask
    ) -> None:
        person, activity_id = _setup(client)
        _upload(client, person.id, activity_id, make_csv_bytes(seed=3))
        append_annotation_entry(person, "note after the session")

        resp = client.get(f"/log/{person.id}")
        assert resp.status_code == 200
        body = resp.data.decode()
        assert "note after the session" in body
        assert "Sitting" in body
        # Latest entry (the annotation) renders before the session entry.
        assert body.index("note after the session") < body.index("Lipponen-Tarvainen")

    def test_timeline_escapes_html_in_annotations(
        self, client: FlaskClient, app: Flask
    ) -> None:
        person, _ = _setup(client)
        client.post(
            f"/log/{person.id}/annotate",
            data={"text": "<script>alert('x')</script>"},
        )
        resp = client.get(f"/log/{person.id}")
        assert b"<script>alert" not in resp.data
        assert b"&lt;script&gt;" in resp.data

    def test_empty_timeline_state(self, client: FlaskClient, app: Flask) -> None:
        person, _ = _setup(client)
        resp = client.get(f"/log/{person.id}")
        assert b"Nothing logged yet" in resp.data
