"""Model-layer tests: constraints that protect data integrity and safety.

The flag-disclaimer CHECK constraint is the important one — the screening
disclaimer is welded to the flag row at the database level, so no template or
code path can persist a flag without one.
"""

from __future__ import annotations

import datetime as dt

import pytest
from flask import Flask
from sqlalchemy.exc import IntegrityError

from app.extensions import db
from app.models import (
    ActivityType,
    ExcludedSegment,
    Flag,
    Metrics,
    Person,
    ProcessingStatus,
    Session,
    slugify,
)


def _make_person(name: str = "Ferhat") -> Person:
    person = Person(name=name, slug=slugify(name))
    db.session.add(person)
    db.session.commit()
    return person


def _make_session(person: Person, sha: str = "a" * 64) -> Session:
    session = Session(
        person_id=person.id,
        original_filename="ecg.csv",
        stored_path=f"uploads/{person.slug}/1__ecg.csv",
        file_sha256=sha,
    )
    db.session.add(session)
    db.session.commit()
    return session


class TestSlugify:
    def test_basic(self) -> None:
        assert slugify("Ferhat Culfaz") == "ferhat-culfaz"

    def test_strips_symbols(self) -> None:
        assert slugify("  Anna-Maria O'Neill!  ") == "anna-maria-o-neill"

    def test_never_empty(self) -> None:
        assert slugify("!!!") == "person"


class TestPerson:
    def test_create_with_defaults(self, app: Flask) -> None:
        person = _make_person()
        assert person.id is not None
        assert person.athlete_baseline is False
        assert person.max_hr_bpm is None
        assert person.created_at is not None

    def test_name_unique(self, app: Flask) -> None:
        _make_person("Sam")
        with pytest.raises(IntegrityError):
            db.session.add(Person(name="Sam", slug="sam-2"))
            db.session.commit()

    def test_deleting_person_cascades_to_sessions(self, app: Flask) -> None:
        person = _make_person()
        session = _make_session(person)
        db.session.add(
            Flag(
                session_id=session.id,
                kind="tachycardia",
                severity="review",
                description="Sustained HR above screening threshold.",
                evidence={"hr": 130},
                threshold={"bpm": 100},
                disclaimer="Screening only, not a diagnosis.",
            )
        )
        db.session.add(
            ExcludedSegment(session_id=session.id, start_s=0, end_s=5, reason="clipping")
        )
        db.session.add(Metrics(session_id=session.id, rmssd_ms=20.0))
        db.session.commit()

        db.session.delete(person)
        db.session.commit()

        assert db.session.query(Session).count() == 0
        assert db.session.query(Flag).count() == 0
        assert db.session.query(Metrics).count() == 0
        assert db.session.query(ExcludedSegment).count() == 0


class TestSession:
    def test_duplicate_sha256_rejected(self, app: Flask) -> None:
        person = _make_person()
        _make_session(person, sha="b" * 64)
        with pytest.raises(IntegrityError):
            _make_session(person, sha="b" * 64)

    def test_status_check_constraint(self, app: Flask) -> None:
        person = _make_person()
        session = _make_session(person)
        session.processing_status = "definitely-not-a-status"
        with pytest.raises(IntegrityError):
            db.session.commit()
        db.session.rollback()

    def test_default_status_pending(self, app: Flask) -> None:
        person = _make_person()
        session = _make_session(person)
        assert session.processing_status == ProcessingStatus.PENDING
        assert session.reduced_confidence is False


class TestFlagDisclaimer:
    """The disclaimer travels with the flag — enforced at the DB level."""

    def _flag(self, session_id: int, disclaimer: str) -> Flag:
        return Flag(
            session_id=session_id,
            kind="rr_irregularity",
            severity="review",
            description="Irregularity indices crossed the screening threshold.",
            evidence={"cosen": 0.1, "cv": 0.2},
            threshold={"cosen": -1.0, "cv": 0.1},
            disclaimer=disclaimer,
        )

    def test_empty_disclaimer_rejected(self, app: Flask) -> None:
        person = _make_person()
        session = _make_session(person)
        with pytest.raises(IntegrityError):
            db.session.add(self._flag(session.id, ""))
            db.session.commit()
        db.session.rollback()

    def test_whitespace_disclaimer_rejected(self, app: Flask) -> None:
        person = _make_person()
        session = _make_session(person)
        with pytest.raises(IntegrityError):
            db.session.add(self._flag(session.id, "   "))
            db.session.commit()
        db.session.rollback()

    def test_null_disclaimer_rejected(self, app: Flask) -> None:
        person = _make_person()
        session = _make_session(person)
        with pytest.raises(IntegrityError):
            db.session.add(self._flag(session.id, None))  # type: ignore[arg-type]
            db.session.commit()
        db.session.rollback()

    def test_valid_flag_persists(self, app: Flask) -> None:
        person = _make_person()
        session = _make_session(person)
        db.session.add(self._flag(session.id, "Screening threshold exceeded; not a diagnosis."))
        db.session.commit()
        assert db.session.query(Flag).count() == 1


class TestActivityType:
    def test_builtin_and_custom(self, app: Flask) -> None:
        builtin = ActivityType(name="Lying down", profile_key="supine", is_builtin=True)
        custom = ActivityType(
            name="Yoga",
            profile_key="supine",
            is_builtin=False,
            expected_hr_min=45,
            expected_hr_max=90,
        )
        db.session.add_all([builtin, custom])
        db.session.commit()
        assert custom.profile_key == builtin.profile_key
        assert custom.expected_hr_min == 45

    def test_name_unique(self, app: Flask) -> None:
        db.session.add(ActivityType(name="Walking", profile_key="walking"))
        db.session.commit()
        with pytest.raises(IntegrityError):
            db.session.add(ActivityType(name="Walking", profile_key="walking"))
            db.session.commit()


class TestMetrics:
    def test_one_row_per_session(self, app: Flask) -> None:
        person = _make_person()
        session = _make_session(person)
        db.session.add(Metrics(session_id=session.id, rmssd_ms=19.4))
        db.session.commit()
        with pytest.raises(IntegrityError):
            db.session.add(Metrics(session_id=session.id, rmssd_ms=25.0))
            db.session.commit()
        db.session.rollback()

    def test_extras_json_roundtrip(self, app: Flask) -> None:
        person = _make_person()
        session = _make_session(person)
        extras = {"sdnn_per_window_ms": [29.1, 41.3, 60.2], "hr_trend_bpm_per_min": -0.59}
        db.session.add(Metrics(session_id=session.id, extras=extras))
        db.session.commit()
        db.session.expire_all()
        stored = db.session.query(Metrics).one()
        assert stored.extras["sdnn_per_window_ms"] == [29.1, 41.3, 60.2]


class TestPersonDates:
    def test_date_of_birth_roundtrip(self, app: Flask) -> None:
        person = Person(name="Dated", slug="dated", date_of_birth=dt.date(1980, 5, 1))
        db.session.add(person)
        db.session.commit()
        db.session.expire_all()
        assert db.session.query(Person).filter_by(slug="dated").one().date_of_birth == dt.date(
            1980, 5, 1
        )
