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
