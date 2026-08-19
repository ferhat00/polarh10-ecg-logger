"""Upload-flow tests: storage layout, dedup, processing, mapping, reports."""

from __future__ import annotations

import datetime as dt
import io
import json
from pathlib import Path

import numpy as np
import pytest
from flask import Flask
from flask.testing import FlaskClient

from app.extensions import db
from app.models import ActivityType, Metrics, Person, ProcessingStatus, Session
from tests.synth_util import build_ecg_from_rr

START = dt.datetime(2026, 8, 10, 9, 0, tzinfo=dt.UTC)


def make_csv_bytes(
    duration_s: float = 150.0,
    delim: str = ",",
    decimal: str = ".",
    seed: int = 21,
) -> bytes:
    """A Polar-shaped CSV built from a known RR series (~75 bpm)."""
    rng = np.random.default_rng(seed)
    n_beats = int(duration_s / 0.8) + 2
    t_beat = np.arange(n_beats) * 0.8
    rr_ms = 800.0 + 25.0 * np.sin(2 * np.pi * 0.1 * t_beat) + rng.normal(0, 8.0, n_beats)
    t, ecg, _beats = build_ecg_from_rr(rr_ms, seed=seed)
    keep = t <= duration_s
    t, ecg = t[keep], ecg[keep]

    start_ns = int(START.timestamp() * 1e9)
    header = delim.join(["time", "ecg", "hr", "rr", "marker"])
    lines = [header]
    for ti, v in zip(t, ecg, strict=True):
        val = f"{v:.4f}"
        if decimal == ",":
            val = val.replace(".", ",")
        lines.append(delim.join([str(start_ns + round(ti * 1e9)), val]))
    return ("\n".join(lines) + "\n").encode()


@pytest.fixture()
def person(app: Flask) -> Person:
    p = Person(name="Uploader", slug="uploader")
    db.session.add(p)
    db.session.commit()
    return p


@pytest.fixture()
def activity_id(client: FlaskClient, app: Flask) -> int:
    client.get("/sessions/upload")  # triggers built-in seeding
    return db.session.query(ActivityType).filter_by(profile_key="sitting").one().id


def _upload(
    client: FlaskClient,
    person: Person,
    activity_id: int,
    payload: bytes,
    filename: str = "ecg_2026-08-10.csv",
    note: str = "",
    extra: dict | None = None,
):
    data = {
        "person_id": str(person.id),
        "activity_type_id": str(activity_id),
        "context_note": note,
        "file": (io.BytesIO(payload), filename),
    }
    if extra:
        data.update(extra)
    return client.post(
        "/sessions/upload",
        data=data,
        content_type="multipart/form-data",
    )


class TestUploadHappyPath:
    def test_upload_processes_and_persists(
        self, client: FlaskClient, app: Flask, person: Person, activity_id: int
    ) -> None:
        resp = _upload(client, person, activity_id, make_csv_bytes(), note="after coffee")
        assert resp.status_code == 302

        session = db.session.query(Session).one()
        assert session.processing_status == ProcessingStatus.DONE
        assert session.context_note == "after coffee"
        assert session.ecg_unit_detected == "mV"
        assert session.epoch_detected == "unix"
        assert session.sampling_rate_hz == pytest.approx(130.03, abs=0.01)
        assert session.recorded_at is not None
        assert session.recorded_at.strftime("%Y-%m-%d %H:%M") == "2026-08-10 09:00"
        assert session.engine_used == "neurokit2"
        assert session.excluded_s == 0.0

        metrics = db.session.query(Metrics).one()
        assert metrics.rmssd_ms is not None and metrics.rmssd_ms > 0
        assert metrics.extras["profile_key"] == "sitting"

        # Storage layout: data/uploads/<slug>/<id>__<name>.csv + caches.
        stored = Path(session.stored_path)
        assert stored.exists()
        assert stored.parent.name == "uploader"
        assert stored.name == f"{session.id}__ecg_2026-08-10.csv"
        assert stored.with_suffix(".npz").exists()
        assert stored.with_suffix(".report.html").exists()

        cache = np.load(stored.with_suffix(".npz"))
        assert len(cache["rr_ms"]) > 100
        assert "peak_times_s" in cache

        # Cache v2 added morphology + ectopy arrays; v3 added sleep arrays.
        assert int(cache["cache_version"][0]) >= 2
        assert "morph_correlations" in cache
        assert "ectopy_confirmed_mask" in cache
        assert "event_t_start_s" in cache
        # Ectopy metrics columns populated (zero on a clean synthetic file).
        assert metrics.ectopy_beats_n is not None
        assert metrics.ectopy_per_hour is not None
        assert metrics.extras["ectopy"]["n_confirmed"] == metrics.ectopy_beats_n

    def test_detail_page_and_report_served(
        self, client: FlaskClient, app: Flask, person: Person, activity_id: int
    ) -> None:
        _upload(client, person, activity_id, make_csv_bytes())
        session = db.session.query(Session).one()

        detail = client.get(f"/sessions/{session.id}")
        assert b"Excluded time" in detail.data
        assert b"Open report" in detail.data
        assert b"clinical clearance" in detail.data

        report = client.get(f"/sessions/{session.id}/report")
        assert report.status_code == 200
        assert b"ECG session report" in report.data
        assert b"data:image/png;base64," in report.data

        download = client.get(f"/sessions/{session.id}/report/download")
        assert "attachment" in download.headers["Content-Disposition"]
        assert f"session{session.id}" in download.headers["Content-Disposition"]


class TestCacheVersioning:
    def test_v2_cache_loads_events(
        self, client: FlaskClient, app: Flask, person: Person, activity_id: int
    ) -> None:
        from app.processing import load_cached_events

        _upload(client, person, activity_id, make_csv_bytes())
        session = db.session.query(Session).one()
        events = load_cached_events(session)
        assert events is not None
        assert "event_t_start_s" in events
        assert len(events["ectopy_confirmed_mask"]) > 100

    def test_pre_events_cache_reports_none(
        self, client: FlaskClient, app: Flask, person: Person, activity_id: int
    ) -> None:
        from app.processing import cache_path_for, load_cached_events

        _upload(client, person, activity_id, make_csv_bytes())
        session = db.session.query(Session).one()
        # Rewrite the cache in the pre-events (v1) shape: no cache_version.
        np.savez_compressed(
            cache_path_for(session), rr_ms=np.array([800.0, 810.0])
        )
        assert load_cached_events(session) is None


class TestContextFields:
    def test_context_fields_persist_from_upload(
        self, client: FlaskClient, app: Flask, person: Person, activity_id: int
    ) -> None:
        _upload(
            client, person, activity_id, make_csv_bytes(),
            extra={
                "body_position": "supine",
                "alcohol_drinks_24h": "2",
                "sleep_quality_1_5": "4",
            },
        )
        session = db.session.query(Session).one()
        assert session.body_position == "supine"
        assert session.body_position_source == "user"
        assert session.alcohol_drinks_24h == 2
        assert session.sleep_quality_1_5 == 4

        detail = client.get(f"/sessions/{session.id}")
        assert b"supine" in detail.data
        assert b"alcohol 2 drink(s)/24 h" in detail.data
        assert b"sleep quality 4/5" in detail.data

    def test_context_fields_default_to_null(
        self, client: FlaskClient, app: Flask, person: Person, activity_id: int
    ) -> None:
        _upload(client, person, activity_id, make_csv_bytes())
        session = db.session.query(Session).one()
        assert session.body_position is None
        assert session.body_position_source is None
        assert session.alcohol_drinks_24h is None
        assert session.sleep_quality_1_5 is None
        # Environment columns stay NULL by default (feature is opt-in).
        assert session.env_fetched_at is None
        assert session.env_temp_c is None
        assert session.env_pm25_ugm3 is None

    def test_invalid_context_values_treated_as_absent(
        self, client: FlaskClient, app: Flask, person: Person, activity_id: int
    ) -> None:
        _upload(
            client, person, activity_id, make_csv_bytes(),
            extra={
                "body_position": "handstand",
                "alcohol_drinks_24h": "99",
                "sleep_quality_1_5": "0",
            },
        )
        session = db.session.query(Session).one()
        assert session.body_position is None
        assert session.alcohol_drinks_24h is None
        assert session.sleep_quality_1_5 is None
        # The upload itself must never fail over malformed optional context.
        assert session.processing_status == ProcessingStatus.DONE

    def test_edit_context_updates_and_clears(
        self, client: FlaskClient, app: Flask, person: Person, activity_id: int
    ) -> None:
        _upload(
            client, person, activity_id, make_csv_bytes(),
            extra={"body_position": "sitting", "alcohol_drinks_24h": "1"},
        )
        session = db.session.query(Session).one()

        resp = client.post(
            f"/sessions/{session.id}/context",
            data={
                "context_note": "edited later",
                "body_position": "supine",
                "alcohol_drinks_24h": "",
                "sleep_quality_1_5": "5",
            },
        )
        assert resp.status_code == 302
        db.session.expire_all()
        session = db.session.get(Session, session.id)
        assert session.context_note == "edited later"
        assert session.body_position == "supine"
        assert session.body_position_source == "user"
        assert session.alcohol_drinks_24h is None
        assert session.sleep_quality_1_5 == 5

        # Clearing the position ("not recorded") removes a user-set value.
        client.post(f"/sessions/{session.id}/context", data={"body_position": ""})
        db.session.expire_all()
        session = db.session.get(Session, session.id)
        assert session.body_position is None
        assert session.body_position_source is None

    def test_acc_derived_position_survives_blank_edit(
        self, client: FlaskClient, app: Flask, person: Person, activity_id: int
    ) -> None:
        _upload(client, person, activity_id, make_csv_bytes())
        session = db.session.query(Session).one()
        session.body_position = "left"
        session.body_position_source = "acc"
        db.session.commit()

        # A form submitted with position blank must not wipe ACC evidence...
        client.post(f"/sessions/{session.id}/context", data={"body_position": ""})
        db.session.expire_all()
        session = db.session.get(Session, session.id)
        assert session.body_position == "left"
        assert session.body_position_source == "acc"

        # ...but an explicit user choice overrides it.
        client.post(f"/sessions/{session.id}/context", data={"body_position": "prone"})
        db.session.expire_all()
        session = db.session.get(Session, session.id)
        assert session.body_position == "prone"
        assert session.body_position_source == "user"


class TestDuplicateRejection:
    def test_exact_duplicate_points_at_existing_session(
        self, client: FlaskClient, app: Flask, person: Person, activity_id: int
    ) -> None:
        payload = make_csv_bytes()
        _upload(client, person, activity_id, payload)
        first = db.session.query(Session).one()

        resp = _upload(client, person, activity_id, payload, filename="renamed.csv")
        assert resp.status_code == 302
        assert resp.headers["Location"].endswith(f"/sessions/{first.id}")
        assert db.session.query(Session).count() == 1

        followed = client.get(resp.headers["Location"])
        assert f"already uploaded as session {first.id}".encode() in followed.data


class TestMappingFlow:
    def test_ambiguous_file_needs_mapping_then_resolves(
        self, client: FlaskClient, app: Flask, person: Person, activity_id: int
    ) -> None:
        payload = make_csv_bytes(delim=";", decimal=",")
        _upload(client, person, activity_id, payload, filename="semi.csv")
        session = db.session.query(Session).one()
        assert session.processing_status == ProcessingStatus.NEEDS_MAPPING

        status = client.get(f"/sessions/{session.id}/status").get_json()
        assert status["status"] == "needs_mapping"
        assert status["questions"][0]["key"] == "delimiter"

        page = client.get(f"/sessions/{session.id}/mapping")
        assert b"delimiter" in page.data.lower()

        # Answer delimiter; the decimal-comma question surfaces on rerun.
        client.post(f"/sessions/{session.id}/mapping", data={"delimiter": ";"})
        db.session.expire_all()
        session = db.session.get(Session, session.id)
        assert session.processing_status == ProcessingStatus.NEEDS_MAPPING
        assert _questions(session)[0]["key"] == "decimal"

        client.post(
            f"/sessions/{session.id}/mapping", data={"delimiter": ";", "decimal": ","}
        )
        db.session.expire_all()
        session = db.session.get(Session, session.id)
        assert session.processing_status == ProcessingStatus.DONE
        assert session.format_overrides == {"delimiter": ";", "decimal": ","}
        assert db.session.query(Metrics).count() == 1

    def test_reprocess_after_mapping_does_not_duplicate_rows(
        self, client: FlaskClient, app: Flask, person: Person, activity_id: int
    ) -> None:
        _upload(client, person, activity_id, make_csv_bytes())
        session = db.session.query(Session).one()
        client.post(f"/sessions/{session.id}/reprocess")
        assert db.session.query(Metrics).count() == 1


class TestErrorPath:
    def test_unusable_file_reports_error(
        self, client: FlaskClient, app: Flask, person: Person, activity_id: int
    ) -> None:
        garbage = b"time,ecg\n" + b"\n".join(
            f"{1_700_000_000_000_000_000 + i * 7_690_000},0.4".encode() for i in range(50)
        )
        _upload(client, person, activity_id, garbage, filename="short.csv")
        session = db.session.query(Session).one()
        assert session.processing_status == ProcessingStatus.ERROR
        assert "too short" in (session.error_message or "")

        detail = client.get(f"/sessions/{session.id}")
        assert b"Processing failed" in detail.data
        assert b"Try again" in detail.data


def _questions(session: Session) -> list[dict]:
    return json.loads(session.error_message or "{}").get("questions", [])


class TestAccUpload:
    def _upload_with_acc(
        self, client: FlaskClient, person: Person, activity_id: int,
        payload: bytes, acc_payload: bytes,
    ):
        return client.post(
            "/sessions/upload",
            data={
                "person_id": str(person.id),
                "activity_type_id": str(activity_id),
                "context_note": "",
                "file": (io.BytesIO(payload), "ecg_2026-08-10.csv"),
                "acc_file": (io.BytesIO(acc_payload), "acc_2026-08-10.csv"),
            },
            content_type="multipart/form-data",
        )

    def test_acc_file_stored_next_to_ecg(
        self, client: FlaskClient, app: Flask, person: Person, activity_id: int
    ) -> None:
        from tests.synth_util import make_acc_csv_bytes

        acc_payload = make_acc_csv_bytes(duration_s=10.0, start=START)
        resp = self._upload_with_acc(
            client, person, activity_id, make_csv_bytes(), acc_payload
        )
        assert resp.status_code == 302
        session = db.session.query(Session).one()
        assert session.acc_original_filename == "acc_2026-08-10.csv"
        assert session.acc_stored_path is not None
        assert Path(session.acc_stored_path).exists()
        assert Path(session.acc_stored_path).read_bytes() == acc_payload
        assert session.acc_file_sha256 is not None
        # A non-sleep activity: the ACC file is stored but staging is not run.
        assert session.processing_status == ProcessingStatus.DONE

    def test_identical_acc_and_ecg_rejected(
        self, client: FlaskClient, app: Flask, person: Person, activity_id: int
    ) -> None:
        payload = make_csv_bytes()
        resp = self._upload_with_acc(client, person, activity_id, payload, payload)
        assert resp.status_code == 200  # re-rendered form with the error
        assert b"attached twice" in resp.data
        assert db.session.query(Session).count() == 0

    def test_upload_without_acc_leaves_columns_null(
        self, client: FlaskClient, app: Flask, person: Person, activity_id: int
    ) -> None:
        _upload(client, person, activity_id, make_csv_bytes())
        session = db.session.query(Session).one()
        assert session.acc_stored_path is None
        assert session.acc_file_sha256 is None
