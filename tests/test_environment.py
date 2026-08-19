"""Opt-in environment capture: client parsing, gating, failure, backfill.

No test here (or anywhere) touches the network: the client takes an
injectable ``opener`` and the app-level tests monkeypatch the fetch.
"""

from __future__ import annotations

import datetime as dt
import io
import json
from collections.abc import Iterator

import pytest
from flask import Flask
from flask.testing import FlaskClient

from app import create_app
from app.config import TestConfig
from app.environment import format_environment_line
from app.environment.client import (
    AIR_QUALITY_URL,
    WEATHER_ARCHIVE_URL,
    WEATHER_FORECAST_URL,
    EnvironmentSample,
    fetch_environment,
)
from app.extensions import db
from app.models import ActivityType, Person, ProcessingStatus, Session
from tests.test_upload import make_csv_bytes

NOW = dt.datetime(2026, 8, 19, 12, 0, tzinfo=dt.UTC)
RECENT_START = dt.datetime(2026, 8, 10, 9, 0, tzinfo=dt.UTC)
OLD_START = dt.datetime(2025, 1, 5, 22, 30, tzinfo=dt.UTC)


def _hourly_body(start: dt.datetime, hours: int, fields: dict[str, list]) -> dict:
    times = [
        (start + dt.timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(hours)
    ]
    return {"hourly": {"time": times, **fields}}


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _FakeOpener:
    """Routes each URL prefix to a canned JSON body (or an exception)."""

    def __init__(self, routes: dict[str, dict | Exception]) -> None:
        self.routes = routes
        self.calls: list[str] = []

    def __call__(self, url: str, timeout: float = 0.0) -> _FakeResponse:
        self.calls.append(url)
        for prefix, body in self.routes.items():
            if url.startswith(prefix):
                if isinstance(body, Exception):
                    raise body
                return _FakeResponse(json.dumps(body).encode())
        raise AssertionError(f"unexpected URL fetched: {url}")


class TestClient:
    def test_recent_recording_uses_forecast_api(self) -> None:
        weather = _hourly_body(
            RECENT_START.replace(hour=0),
            24,
            {
                "temperature_2m": [20.0] * 24,
                "relative_humidity_2m": [50.0] * 24,
                "apparent_temperature": [19.0] * 24,
                "surface_pressure": [1013.0] * 24,
            },
        )
        weather["daily"] = {
            "time": [RECENT_START.date().isoformat()],
            "daylight_duration": [14 * 3600.0],
        }
        air = _hourly_body(
            RECENT_START.replace(hour=0),
            24,
            {
                "pm2_5": [8.0] * 24,
                "pm10": [15.0] * 24,
                "ozone": [60.0] * 24,
                "nitrogen_dioxide": [12.0] * 24,
                "european_aqi": [22.0] * 24,
            },
        )
        opener = _FakeOpener({WEATHER_FORECAST_URL: weather, AIR_QUALITY_URL: air})
        sample = fetch_environment(
            51.5, -0.1, RECENT_START, 3600.0, opener=opener, now=NOW
        )
        assert sample is not None
        assert sample.source == "open-meteo-forecast"
        assert sample.temp_c == pytest.approx(20.0)
        assert sample.humidity_pct == pytest.approx(50.0)
        assert sample.pressure_hpa == pytest.approx(1013.0)
        assert sample.pm25_ugm3 == pytest.approx(8.0)
        assert sample.aqi == pytest.approx(22.0)
        assert sample.daylight_h == pytest.approx(14.0)
        assert sample.notes == []
        # past_days covers the 9-day-old recording; both APIs called once.
        assert len(opener.calls) == 2
        assert "past_days=10" in opener.calls[0]

    def test_old_recording_uses_archive_api(self) -> None:
        weather = _hourly_body(
            OLD_START.replace(hour=0), 48, {"temperature_2m": [3.0] * 48}
        )
        opener = _FakeOpener(
            {
                WEATHER_ARCHIVE_URL: weather,
                AIR_QUALITY_URL: OSError("no AQ coverage that far back"),
            }
        )
        sample = fetch_environment(51.5, -0.1, OLD_START, 7200.0, opener=opener, now=NOW)
        assert sample is not None
        assert sample.source == "open-meteo-archive"
        assert sample.temp_c == pytest.approx(3.0)
        # Missing air quality is a supported partial outcome, with a note.
        assert sample.pm25_ugm3 is None
        assert any("air quality lookup failed" in n for n in sample.notes)
        assert "start_date=2025-01-05" in opener.calls[0]

    def test_window_averaging_selects_overlapping_hours(self) -> None:
        # 22:30–00:30 recording: hours 22, 23 (day 1) and 00 (day 2) overlap.
        temps = [float(i) for i in range(48)]  # temp == hour index
        weather = _hourly_body(
            OLD_START.replace(hour=0, minute=0), 48, {"temperature_2m": temps}
        )
        opener = _FakeOpener(
            {WEATHER_ARCHIVE_URL: weather, AIR_QUALITY_URL: OSError("skip")}
        )
        sample = fetch_environment(51.5, -0.1, OLD_START, 7200.0, opener=opener, now=NOW)
        assert sample is not None
        assert sample.temp_c == pytest.approx((22 + 23 + 24) / 3)

    def test_null_values_and_total_failure(self) -> None:
        weather = _hourly_body(
            OLD_START.replace(hour=0), 24, {"temperature_2m": [None] * 24}
        )
        opener = _FakeOpener(
            {WEATHER_ARCHIVE_URL: weather, AIR_QUALITY_URL: OSError("down")}
        )
        assert fetch_environment(51.5, -0.1, OLD_START, 60.0, opener=opener, now=NOW) is None

        opener = _FakeOpener(
            {
                WEATHER_ARCHIVE_URL: OSError("down"),
                AIR_QUALITY_URL: OSError("down"),
            }
        )
        assert fetch_environment(51.5, -0.1, OLD_START, 60.0, opener=opener, now=NOW) is None

    def test_api_error_body_is_a_note_not_a_crash(self) -> None:
        opener = _FakeOpener(
            {
                WEATHER_FORECAST_URL: {"error": True, "reason": "invalid coordinates"},
                AIR_QUALITY_URL: _hourly_body(
                    RECENT_START.replace(hour=0), 24, {"pm2_5": [7.0] * 24}
                ),
            }
        )
        sample = fetch_environment(
            99.0, 0.0, RECENT_START, 600.0, opener=opener, now=NOW
        )
        assert sample is not None
        assert sample.temp_c is None
        assert sample.pm25_ugm3 == pytest.approx(7.0)
        assert any("invalid coordinates" in n for n in sample.notes)


class TestFormatting:
    def test_format_line_partial_values(self) -> None:
        class Stub:
            env_temp_c = 21.3
            env_apparent_temp_c = 20.1
            env_humidity_pct = 46.0
            env_pressure_hpa = None
            env_pm25_ugm3 = 8.0
            env_aqi = None
            env_fetched_at = NOW
            env_source = "open-meteo-forecast"

        line = format_environment_line(Stub())
        assert line == (
            "21.3 °C (feels 20.1) · RH 46% · PM2.5 8 µg/m³ "
            "(open-meteo-forecast, window-averaged)"
        )

    def test_format_line_none_when_unfetched(self) -> None:
        class Stub:
            env_temp_c = None
            env_apparent_temp_c = None
            env_humidity_pct = None
            env_pressure_hpa = None
            env_pm25_ugm3 = None
            env_aqi = None
            env_fetched_at = None
            env_source = None

        assert format_environment_line(Stub()) is None


@pytest.fixture()
def enabled_app(tmp_path) -> Iterator[Flask]:
    config = TestConfig()
    config.DATA_DIR = tmp_path / "data"
    config.WEATHER_ENABLED = True
    config.HOME_LAT = 51.5
    config.HOME_LON = -0.1
    app = create_app(config)
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


def _upload_one(app: Flask, note: str = "") -> Session:
    client = app.test_client()
    client.get("/sessions/upload")  # seed builtins
    person = Person(name="Env", slug="env")
    db.session.add(person)
    db.session.commit()
    activity = db.session.query(ActivityType).filter_by(profile_key="sitting").one()
    resp = client.post(
        "/sessions/upload",
        data={
            "person_id": str(person.id),
            "activity_type_id": str(activity.id),
            "context_note": note,
            "file": (io.BytesIO(make_csv_bytes()), "ecg_2026-08-10.csv"),
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 302
    # Processing ran inline on nested app contexts (their own db sessions);
    # drop any stale identity-map state before reading the outcome.
    db.session.expire_all()
    return db.session.query(Session).one()


class TestProcessingIntegration:
    def test_disabled_by_default_never_calls_out(
        self, app: Flask, client: FlaskClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*args: object, **kwargs: object) -> None:
            raise AssertionError("network fetch attempted with weather disabled")

        import app.environment.client as env_client

        monkeypatch.setattr(env_client, "fetch_environment", _boom)
        session = _upload_one(app)
        assert session.processing_status == ProcessingStatus.DONE
        assert session.env_fetched_at is None
        assert session.metrics.extras["environment"] is None

    def test_enabled_fetch_populates_columns_and_report(
        self, enabled_app: Flask, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sample = EnvironmentSample(
            temp_c=21.3,
            apparent_temp_c=20.1,
            humidity_pct=46.0,
            pressure_hpa=1013.0,
            pm25_ugm3=8.0,
            aqi=22.0,
            daylight_h=14.2,
            source="open-meteo-forecast",
        )
        import app.environment.client as env_client

        monkeypatch.setattr(
            env_client, "fetch_environment", lambda *a, **kw: sample
        )
        session = _upload_one(enabled_app)
        assert session.processing_status == ProcessingStatus.DONE
        assert session.env_temp_c == pytest.approx(21.3)
        assert session.env_pm25_ugm3 == pytest.approx(8.0)
        assert session.env_source == "open-meteo-forecast"
        assert session.env_fetched_at is not None
        assert session.metrics.extras["environment"]["status"] == "ok"

        page = enabled_app.test_client().get(f"/sessions/{session.id}")
        assert "21.3 °C".encode() in page.data
        report = enabled_app.test_client().get(f"/sessions/{session.id}/report")
        assert b"Environment" in report.data
        assert "21.3 °C".encode() in report.data

        from app.logbook.writer import log_path_for

        text = log_path_for(session.person).read_text(encoding="utf-8")
        assert "**Environment:**" in text
        assert "21.3 °C" in text

    def test_fetch_failure_never_fails_the_session(
        self, enabled_app: Flask, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import app.environment.client as env_client

        def _explode(*args: object, **kwargs: object) -> None:
            raise OSError("network unreachable")

        monkeypatch.setattr(env_client, "fetch_environment", _explode)
        session = _upload_one(enabled_app)
        assert session.processing_status == ProcessingStatus.DONE
        assert session.env_fetched_at is None
        env = session.metrics.extras["environment"]
        assert env["status"] == "failed"
        assert any("network unreachable" in n for n in env["notes"])

    def test_reprocess_does_not_refetch(
        self, enabled_app: Flask, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import app.environment.client as env_client

        calls: list[int] = []

        def _fetch(*args: object, **kwargs: object) -> EnvironmentSample:
            calls.append(1)
            return EnvironmentSample(temp_c=10.0, source="open-meteo-forecast")

        monkeypatch.setattr(env_client, "fetch_environment", _fetch)
        session = _upload_one(enabled_app)
        assert len(calls) == 1
        client = enabled_app.test_client()
        client.post(f"/sessions/{session.id}/reprocess")
        assert len(calls) == 1  # kept from the previous fetch
        db.session.expire_all()
        session = db.session.get(Session, session.id)
        assert session.env_temp_c == pytest.approx(10.0)


class TestBackfillCli:
    def test_disabled_message(self, app: Flask) -> None:
        runner = app.test_cli_runner()
        result = runner.invoke(args=["env-backfill"])
        assert "off" in result.output

    def test_backfill_fills_and_refetches(
        self, enabled_app: Flask, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import app.environment.client as env_client

        monkeypatch.setattr(
            env_client,
            "fetch_environment",
            lambda *a, **kw: EnvironmentSample(temp_c=5.5, source="open-meteo-archive"),
        )
        session = _upload_one(enabled_app)
        # Wipe what processing fetched (simulate an opt-in after upload).
        session.env_fetched_at = None
        session.env_temp_c = None
        session.env_source = None
        db.session.commit()

        runner = enabled_app.test_cli_runner()
        result = runner.invoke(args=["env-backfill"])
        assert "1 session(s) updated" in result.output
        db.session.expire_all()
        session = db.session.get(Session, session.id)
        assert session.env_temp_c == pytest.approx(5.5)

        # Without --refetch the second run skips; with it, it refetches.
        result = runner.invoke(args=["env-backfill"])
        assert "skipped" in result.output
        monkeypatch.setattr(
            env_client,
            "fetch_environment",
            lambda *a, **kw: EnvironmentSample(temp_c=7.7, source="open-meteo-archive"),
        )
        result = runner.invoke(args=["env-backfill", "--refetch"])
        assert "1 session(s) updated" in result.output
        db.session.expire_all()
        session = db.session.get(Session, session.id)
        assert session.env_temp_c == pytest.approx(7.7)
