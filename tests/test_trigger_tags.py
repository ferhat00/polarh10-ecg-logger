"""Trigger-tag model, seeding, and tagging behaviour."""

from __future__ import annotations

import pytest
from flask import Flask

from app.extensions import db
from app.models import Person, Session, TriggerTag, session_trigger_tag
from app.triggers.seed import ensure_builtin_trigger_tags
from app.triggers.vocabulary import BUILTIN_TRIGGER_TAGS


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
