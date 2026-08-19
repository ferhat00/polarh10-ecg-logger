"""Trigger-tag model, seeding, tagging behaviour, and capture routes."""

from __future__ import annotations

import io

import pytest
from flask import Flask
from flask.testing import FlaskClient

from app.extensions import db
from app.models import ActivityType, Person, Session, TriggerTag, session_trigger_tag
from app.triggers.seed import ensure_builtin_trigger_tags
from app.triggers.vocabulary import BUILTIN_TRIGGER_TAGS
from tests.test_upload import make_csv_bytes


@pytest.fixture()
def person(app: Flask) -> Person:
    p = Person(name="Tagged", slug="tagged")
    db.session.add(p)
    db.session.commit()
    return p


def _session_for(person: Person, sha: str = "0" * 64) -> Session:
    s = Session(
        person_id=person.id,
        original_filename="x.csv",
        stored_path="/tmp/x.csv",
        file_sha256=sha,
    )
    db.session.add(s)
    db.session.commit()
    return s


class TestSeeding:
    def test_seeds_all_builtins(self, app: Flask) -> None:
        added = ensure_builtin_trigger_tags()
        assert added == len(BUILTIN_TRIGGER_TAGS)
        slugs = {t.slug for t in db.session.query(TriggerTag).all()}
        assert slugs == {slug for slug, _ in BUILTIN_TRIGGER_TAGS}

    def test_idempotent(self, app: Flask) -> None:
        ensure_builtin_trigger_tags()
        assert ensure_builtin_trigger_tags() == 0
        assert db.session.query(TriggerTag).count() == len(BUILTIN_TRIGGER_TAGS)

    def test_builtins_marked(self, app: Flask) -> None:
        ensure_builtin_trigger_tags()
        assert all(t.is_builtin for t in db.session.query(TriggerTag).all())

    def test_custom_tag_with_builtin_slug_is_promoted(self, app: Flask) -> None:
        # A user typed "nicotine" as a free-text tag before the vocabulary
        # grew to include it. Seeding must promote that row (keeping the
        # user's name and session links), not crash on the slug UNIQUE.
        db.session.add(TriggerTag(name="Nicotine gum", slug="nicotine"))
        db.session.commit()

        ensure_builtin_trigger_tags()
        rows = db.session.query(TriggerTag).filter_by(slug="nicotine").all()
        assert len(rows) == 1
        assert rows[0].is_builtin is True
        assert rows[0].name == "Nicotine gum"  # user's display name kept
        assert db.session.query(TriggerTag).count() == len(BUILTIN_TRIGGER_TAGS)


class TestTagging:
    def test_session_tags_roundtrip(self, app: Flask, person: Person) -> None:
        ensure_builtin_trigger_tags()
        s = _session_for(person)
        caffeine = db.session.query(TriggerTag).filter_by(slug="caffeine").one()
        alcohol = db.session.query(TriggerTag).filter_by(slug="alcohol").one()
        s.trigger_tags = [alcohol, caffeine]
        db.session.commit()

        loaded = db.session.get(Session, s.id)
        # order_by name: Alcohol before Caffeine.
        assert [t.slug for t in loaded.trigger_tags] == ["alcohol", "caffeine"]
        assert [x.id for x in caffeine.sessions] == [s.id]

    def test_tag_slug_unique(self, app: Flask) -> None:
        db.session.add(TriggerTag(name="Coffee", slug="caffeine"))
        db.session.commit()
        db.session.add(TriggerTag(name="Espresso", slug="caffeine"))
        with pytest.raises(Exception):  # noqa: B017 - IntegrityError wrapper varies
            db.session.commit()
        db.session.rollback()

    def test_selected_tags_parses_ids_and_custom_names(self, app: Flask) -> None:
        from werkzeug.datastructures import MultiDict

        from app.blueprints.sessions import _selected_tags

        ensure_builtin_trigger_tags()
        caffeine = db.session.query(TriggerTag).filter_by(slug="caffeine").one()
        form = MultiDict(
            [
                ("trigger_tags", str(caffeine.id)),
                ("trigger_tags", "not-a-number"),
                ("new_tags", "Cold Shower,  , caffeine"),
            ]
        )
        tags = _selected_tags(form)
        slugs = sorted(t.slug for t in tags)
        # "caffeine" typed as a custom name resolves to the existing tag.
        assert slugs == ["caffeine", "cold-shower"]
        custom = db.session.query(TriggerTag).filter_by(slug="cold-shower").one()
        assert custom.is_builtin is False

    def test_person_delete_cleans_association_rows(
        self, app: Flask, person: Person
    ) -> None:
        ensure_builtin_trigger_tags()
        s = _session_for(person)
        tag = db.session.query(TriggerTag).filter_by(slug="stress").one()
        s.trigger_tags = [tag]
        db.session.commit()

        db.session.delete(person)
        db.session.commit()

        assert db.session.query(Session).count() == 0
        assert db.session.execute(session_trigger_tag.select()).all() == []
        # The tag itself survives — vocabulary is global.
        assert db.session.query(TriggerTag).filter_by(slug="stress").count() == 1


class TestCaptureRoutes:
    @pytest.fixture()
    def uploaded(self, client: FlaskClient, app: Flask, person: Person) -> Session:
        client.get("/sessions/upload")  # seeds builtin activities and tags
        activity = db.session.query(ActivityType).filter_by(profile_key="sitting").one()
        caffeine = db.session.query(TriggerTag).filter_by(slug="caffeine").one()
        resp = client.post(
            "/sessions/upload",
            data={
                "person_id": str(person.id),
                "activity_type_id": str(activity.id),
                "context_note": "",
                "trigger_tags": [str(caffeine.id)],
                "new_tags": "Cold Shower",
                "file": (io.BytesIO(make_csv_bytes()), "ecg_2026-08-10.csv"),
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 302
        return db.session.query(Session).one()

    def test_upload_attaches_tags(self, uploaded: Session) -> None:
        assert sorted(t.slug for t in uploaded.trigger_tags) == [
            "caffeine",
            "cold-shower",
        ]

    def test_detail_shows_tags_and_edit_form(
        self, client: FlaskClient, uploaded: Session
    ) -> None:
        page = client.get(f"/sessions/{uploaded.id}")
        assert b"Cold Shower" in page.data
        assert b"Edit trigger tags" in page.data
        assert b"Ectopic beats" in page.data

    def test_edit_tags_replaces_set(
        self, client: FlaskClient, uploaded: Session
    ) -> None:
        alcohol = db.session.query(TriggerTag).filter_by(slug="alcohol").one()
        resp = client.post(
            f"/sessions/{uploaded.id}/tags",
            data={"trigger_tags": [str(alcohol.id)], "new_tags": ""},
        )
        assert resp.status_code == 302
        db.session.expire(uploaded, ["trigger_tags"])
        assert [t.slug for t in uploaded.trigger_tags] == ["alcohol"]

    def test_logbook_entry_records_tags(
        self, app: Flask, uploaded: Session
    ) -> None:
        from app.logbook.writer import log_path_for

        # Tags were attached at upload time, before processing logged the
        # session entry, so the entry carries them.
        text = log_path_for(uploaded.person).read_text(encoding="utf-8")
        assert "**Triggers:**" in text
        assert "Caffeine" in text
        assert "**Ectopic beats:**" in text
