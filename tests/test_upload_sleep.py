"""End-to-end sleep session: upload → pipeline → staging → persistence.

Uses a compressed "night" (~37 min — above the staging minimum, fast enough
for CI) built from the stage-programmed RR synthesizer, with a matching
accelerometer file whose movement burst lands in the opening wake block.
"""

from __future__ import annotations

import datetime as dt
import io

import numpy as np
import pytest
from flask import Flask
from flask.testing import FlaskClient

from app.extensions import db
from app.models import ActivityType, Metrics, Person, ProcessingStatus, Session
from app.processing import cache_path_for
from tests.synth_util import build_ecg_from_rr, make_acc_csv_bytes, overnight_rr

START = dt.datetime(2026, 8, 10, 22, 30, tzinfo=dt.UTC)

#: Compressed night: opening wake (with movement), two NREM cycles, REM.
STAGE_PLAN = [
    ("wake", 3),
    ("light", 10),
    ("deep", 9),
    ("rem", 8),
    ("light", 7),
]


def _night_csv_bytes(seed: int = 21) -> bytes:
    rr_ms, _t, _truth = overnight_rr(stage_plan=STAGE_PLAN, seed=seed)
    t, ecg, _beats = build_ecg_from_rr(rr_ms, seed=seed)
    start_ns = int(START.timestamp() * 1e9)
    lines = ["time,ecg,hr,rr,marker"]
    for ti, v in zip(t, ecg, strict=True):
        lines.append(f"{start_ns + round(ti * 1e9)},{v:.4f}")
    return ("\n".join(lines) + "\n").encode()


def _night_acc_bytes() -> bytes:
    total_s = sum(minutes for _stage, minutes in STAGE_PLAN) * 60.0
    return make_acc_csv_bytes(
        duration_s=total_s,
        fs_hz=25.0,
        start=START,
        # Sustained movement inside the programmed opening wake block.
        movement_bursts=[(30.0, 150.0, 300.0)],
    )


@pytest.fixture(autouse=True)
def heuristic_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin these tests to the always-available engine so they are identical
    with and without the optional sleepecg/TensorFlow extras installed."""
    from app.sleep.engines import EngineStatus, sleepecg_engine

    monkeypatch.setattr(
        sleepecg_engine,
        "status",
        lambda: EngineStatus(
            sleepecg_engine.ENGINE_KEY,
            sleepecg_engine.ENGINE_LABEL,
            False,
            "disabled in this test",
        ),
    )


@pytest.fixture()
def person(app: Flask) -> Person:
    p = Person(name="Sleeper", slug="sleeper")
    db.session.add(p)
    db.session.commit()
    return p


@pytest.fixture()
def sleep_activity_id(client: FlaskClient, app: Flask) -> int:
    client.get("/sessions/upload")  # triggers built-in seeding
    return db.session.query(ActivityType).filter_by(profile_key="sleep").one().id


def _upload(
    client: FlaskClient,
    person: Person,
    activity_id: int,
    payload: bytes,
    acc_payload: bytes | None = None,
):
    data = {
        "person_id": str(person.id),
        "activity_type_id": str(activity_id),
        "context_note": "",
        "file": (io.BytesIO(payload), "ecg_2026-08-10.csv"),
    }
    if acc_payload is not None:
        data["acc_file"] = (io.BytesIO(acc_payload), "acc_2026-08-10.csv")
    return client.post(
        "/sessions/upload", data=data, content_type="multipart/form-data"
    )


class TestSleepSessionEndToEnd:
    def test_full_night_with_acc(
        self, client: FlaskClient, app: Flask, person: Person, sleep_activity_id: int
    ) -> None:
        resp = _upload(
            client, person, sleep_activity_id, _night_csv_bytes(), _night_acc_bytes()
        )
        assert resp.status_code == 302

        session = db.session.query(Session).one()
        assert session.processing_status == ProcessingStatus.DONE
        metrics = db.session.query(Metrics).one()

        # Queryable sleep columns from the primary engine (heuristic — the
        # optional engines are not installed in CI).
        assert metrics.sleep_engine == "heuristic"
        assert metrics.tst_min is not None and metrics.tst_min > 15.0
        assert metrics.sleep_efficiency_pct is not None
        assert 0.0 < metrics.sleep_efficiency_pct <= 100.0
        assert metrics.sol_min is not None
        assert metrics.waso_min is not None
        assert metrics.deep_min is not None and metrics.deep_min > 3.0
        assert metrics.rem_min is not None and metrics.rem_min > 3.0

        # Extras carry the full detail, consistent with the columns.
        sleep_extras = metrics.extras["sleep"]
        assert sleep_extras["primary_engine"] == "heuristic"
        summary = sleep_extras["summaries"]["heuristic"]
        assert summary["tst_min"] == pytest.approx(metrics.tst_min)
        hyp = sleep_extras["hypnograms"]["heuristic"]
        assert hyp["vocab"] == "wldr_4"
        assert len(hyp["stages"]) == len(
            np.arange(0, session.duration_s, 30.0)
        )
        assert "accuracy_note" in hyp and "polysomnography" in hyp["accuracy_note"]
        assert sleep_extras["acc"]["present"] is True
        assert sleep_extras["acc"]["error"] is None

        # The movement burst forces wake at the night's start.
        stages = np.array(hyp["stages"])
        assert np.any(stages[2:5] == 0)

        # The rendered report carries the sleep section and its disclaimer.
        from app.processing import report_path_for

        report_html = report_path_for(session).read_text(encoding="utf-8")
        assert "Hypnogram" in report_html
        assert "substitute for a sleep study" in report_html

        # Cache v3 arrays.
        with np.load(cache_path_for(session)) as cache:
            assert int(cache["cache_version"][0]) >= 3
            assert "sleep_stages_heuristic" in cache
            assert "sleep_probs_heuristic" in cache
            assert "acc_counts" in cache
            assert len(cache["sleep_stages_heuristic"]) == len(stages)

    def test_night_without_acc_still_stages(
        self, client: FlaskClient, app: Flask, person: Person, sleep_activity_id: int
    ) -> None:
        _upload(client, person, sleep_activity_id, _night_csv_bytes(seed=8))
        session = db.session.query(Session).one()
        assert session.processing_status == ProcessingStatus.DONE
        metrics = db.session.query(Metrics).one()
        assert metrics.sleep_engine == "heuristic"
        sleep_extras = metrics.extras["sleep"]
        assert sleep_extras["acc"]["present"] is False
        # The wander-proxy note is surfaced.
        joined = " ".join(sleep_extras["hypnograms"]["heuristic"]["notes"])
        assert "proxy" in joined

    def test_corrupt_acc_survives_with_error_surfaced(
        self, client: FlaskClient, app: Flask, person: Person, sleep_activity_id: int
    ) -> None:
        _upload(
            client, person, sleep_activity_id, _night_csv_bytes(seed=9),
            b"not;a;real;acc\n1;2;3;4\n",
        )
        session = db.session.query(Session).one()
        assert session.processing_status == ProcessingStatus.DONE
        metrics = db.session.query(Metrics).one()
        sleep_extras = metrics.extras["sleep"]
        assert sleep_extras["acc"]["error"]
        assert metrics.sleep_engine == "heuristic"  # staging still ran

    def test_short_recording_skips_staging_with_note(
        self, client: FlaskClient, app: Flask, person: Person, sleep_activity_id: int
    ) -> None:
        # A 10-minute "nap" — below the 30 min staging minimum.
        rr_ms, _t, _truth = overnight_rr(stage_plan=[("light", 10)], seed=4)
        t, ecg, _beats = build_ecg_from_rr(rr_ms, seed=4)
        start_ns = int(START.timestamp() * 1e9)
        lines = ["time,ecg,hr,rr,marker"] + [
            f"{start_ns + round(ti * 1e9)},{v:.4f}"
            for ti, v in zip(t, ecg, strict=True)
        ]
        _upload(client, person, sleep_activity_id, ("\n".join(lines)).encode())
        session = db.session.query(Session).one()
        assert session.processing_status == ProcessingStatus.DONE
        metrics = db.session.query(Metrics).one()
        assert metrics.sleep_engine is None
        assert metrics.tst_min is None
        notes = " ".join(metrics.extras["sleep"]["notes"])
        assert "minimum for sleep staging" in notes


class TestEngineChoiceUI:
    """The staging-engine dropdown on a finished night, and its validation.

    The autouse ``heuristic_only`` fixture pins sleepecg to unavailable, so
    these run identically with and without the optional extras installed.
    """

    @pytest.fixture()
    def night(
        self, client: FlaskClient, app: Flask, person: Person, sleep_activity_id: int
    ) -> Session:
        _upload(client, person, sleep_activity_id, _night_csv_bytes())
        session = db.session.query(Session).one()
        assert session.processing_status == ProcessingStatus.DONE
        return session

    def test_dropdown_offers_engines_with_their_reasons(
        self, client: FlaskClient, night: Session
    ) -> None:
        html = client.get(f"/sessions/{night.id}").get_data(as_text=True)
        assert 'name="sleep_engine"' in html
        assert 'value="heuristic"' in html
        # Unavailable engines stay visible, disabled, with their exact reason.
        assert 'value="sleepecg"' in html
        assert "disabled" in html
        assert "disabled in this test" in html
        assert "ECGLOG_SLEEP_EXTERNAL_DIR" in html
        # And the dropdown says what each engine can tell apart.
        assert "Wake/Light/Deep/REM" in html

    def test_no_dropdown_for_a_non_sleep_session(
        self, client: FlaskClient, app: Flask, person: Person
    ) -> None:
        client.get("/sessions/upload")
        sitting = (
            db.session.query(ActivityType).filter_by(profile_key="sitting").one().id
        )
        _upload(client, person, sitting, _night_csv_bytes(seed=33))
        session = db.session.query(Session).one()
        html = client.get(f"/sessions/{session.id}").get_data(as_text=True)
        assert 'name="sleep_engine"' not in html

    def test_choosing_an_available_engine_sticks_and_reruns(
        self, client: FlaskClient, night: Session
    ) -> None:
        resp = client.post(
            f"/sessions/{night.id}/reprocess", data={"sleep_engine": "heuristic"}
        )
        assert resp.status_code == 302
        # Processing ran in its own app context, so this long-lived test
        # session is holding a pre-reprocess copy; a real request reads fresh.
        db.session.expire_all()
        session = db.session.query(Session).one()
        assert session.sleep_engine_pref == "heuristic"
        assert session.processing_status == ProcessingStatus.DONE
        metrics = db.session.query(Metrics).one()
        assert metrics.extras["sleep"]["engine_pref"] == "heuristic"
        assert metrics.sleep_engine == "heuristic"

    def test_unavailable_engine_is_refused_and_changes_nothing(
        self, client: FlaskClient, night: Session
    ) -> None:
        before = db.session.query(Metrics).one().tst_min
        resp = client.post(
            f"/sessions/{night.id}/reprocess",
            data={"sleep_engine": "sleepecg"},
            follow_redirects=True,
        )
        html = resp.get_data(as_text=True)
        assert "disabled in this test" in html
        session = db.session.query(Session).one()
        assert session.sleep_engine_pref is None
        assert session.processing_status == ProcessingStatus.DONE
        assert db.session.query(Metrics).one().tst_min == before

    def test_unknown_engine_is_refused(
        self, client: FlaskClient, night: Session
    ) -> None:
        resp = client.post(
            f"/sessions/{night.id}/reprocess",
            data={"sleep_engine": "totally-made-up"},
            follow_redirects=True,
        )
        assert "Unknown sleep algorithm" in resp.get_data(as_text=True)
        assert db.session.query(Session).one().sleep_engine_pref is None

    def test_empty_value_clears_the_preference(
        self, client: FlaskClient, night: Session
    ) -> None:
        client.post(
            f"/sessions/{night.id}/reprocess", data={"sleep_engine": "heuristic"}
        )
        assert db.session.query(Session).one().sleep_engine_pref == "heuristic"
        client.post(f"/sessions/{night.id}/reprocess", data={"sleep_engine": ""})
        assert db.session.query(Session).one().sleep_engine_pref is None

    def test_plain_reprocess_leaves_the_preference_alone(
        self, client: FlaskClient, night: Session
    ) -> None:
        """The error-retry and ectopy-backfill buttons send no engine field."""
        client.post(
            f"/sessions/{night.id}/reprocess", data={"sleep_engine": "heuristic"}
        )
        client.post(f"/sessions/{night.id}/reprocess")
        session = db.session.query(Session).one()
        assert session.sleep_engine_pref == "heuristic"
        assert session.processing_status == ProcessingStatus.DONE

    def test_report_records_whose_choice_it_was(
        self, client: FlaskClient, night: Session
    ) -> None:
        client.post(
            f"/sessions/{night.id}/reprocess", data={"sleep_engine": "heuristic"}
        )
        html = client.get(f"/sessions/{night.id}/report").get_data(as_text=True)
        assert "chosen by you" in html
        assert "(your choice)" in html


class TestSleepHistory:
    @pytest.fixture()
    def two_nights(
        self, client: FlaskClient, app: Flask, person: Person, sleep_activity_id: int
    ) -> list[Session]:
        _upload(client, person, sleep_activity_id, _night_csv_bytes(seed=21))
        _upload(client, person, sleep_activity_id, _night_csv_bytes(seed=22))
        sessions = db.session.query(Session).all()
        assert len(sessions) == 2
        return sessions

    def test_history_lists_nights_with_their_engine(
        self, client: FlaskClient, two_nights: list[Session]
    ) -> None:
        html = client.get("/sleep/").get_data(as_text=True)
        for s in two_nights:
            assert f"/sessions/{s.id}" in html
        assert "heuristic" in html
        assert 'name="sleep_engine"' in html

    def test_history_excludes_non_sleep_sessions(
        self, client: FlaskClient, app: Flask, person: Person
    ) -> None:
        client.get("/sessions/upload")
        sitting = (
            db.session.query(ActivityType).filter_by(profile_key="sitting").one().id
        )
        _upload(client, person, sitting, _night_csv_bytes(seed=34))
        session = db.session.query(Session).one()
        html = client.get("/sleep/").get_data(as_text=True)
        assert f"/sessions/{session.id}" not in html
        assert "No sleep sessions yet" in html

    def test_batch_sets_the_engine_on_every_selected_night(
        self, client: FlaskClient, two_nights: list[Session]
    ) -> None:
        resp = client.post(
            "/sleep/reanalyse",
            data={
                "sessions": [str(s.id) for s in two_nights],
                "sleep_engine": "heuristic",
            },
            follow_redirects=True,
        )
        assert "2 night(s) queued" in resp.get_data(as_text=True)
        db.session.expire_all()
        for s in db.session.query(Session).all():
            assert s.sleep_engine_pref == "heuristic"
            assert s.processing_status == ProcessingStatus.DONE

    def test_batch_keep_leaves_each_nights_choice_alone(
        self, client: FlaskClient, two_nights: list[Session]
    ) -> None:
        first, second = two_nights
        client.post(
            f"/sessions/{first.id}/reprocess", data={"sleep_engine": "heuristic"}
        )
        client.post(
            "/sleep/reanalyse",
            data={
                "sessions": [str(s.id) for s in two_nights],
                "sleep_engine": "keep",
            },
        )
        prefs = {s.id: s.sleep_engine_pref for s in db.session.query(Session).all()}
        assert prefs[first.id] == "heuristic"
        assert prefs[second.id] is None

    def test_batch_rejects_an_unavailable_engine_without_touching_anything(
        self, client: FlaskClient, two_nights: list[Session]
    ) -> None:
        resp = client.post(
            "/sleep/reanalyse",
            data={
                "sessions": [str(s.id) for s in two_nights],
                "sleep_engine": "sleepecg",
            },
            follow_redirects=True,
        )
        assert "disabled in this test" in resp.get_data(as_text=True)
        for s in db.session.query(Session).all():
            assert s.sleep_engine_pref is None

    def test_batch_with_nothing_selected_says_so(
        self, client: FlaskClient, two_nights: list[Session]
    ) -> None:
        resp = client.post(
            "/sleep/reanalyse",
            data={"sleep_engine": "heuristic"},
            follow_redirects=True,
        )
        assert "No nights were selected" in resp.get_data(as_text=True)
