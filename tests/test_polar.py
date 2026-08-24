"""Opt-in Polar Flow sync: allowlist, parsing, gating, failure, idempotency.

No test here (or anywhere) touches the network: the client takes an injectable
``opener`` and every fixture routes requests to canned JSON.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import urllib.error
from collections.abc import Iterator

import pytest
from flask import Flask

from app import create_app
from app.config import TestConfig
from app.extensions import db
from app.models import ActivityType, FlowNight, Person, PolarAccount, Session
from app.polar import client as polar_client
from app.polar import format_night_line, night_before
from app.polar.sync import sync_person
from tests.test_upload import make_csv_bytes

TODAY = dt.date(2026, 8, 21)
NIGHT = dt.date(2026, 8, 20)

SLEEP_ITEM = {
    "date": "2026-08-20",
    "device_id": "AAA1",
    "sleep_start_time": "2026-08-19T23:10:00+02:00",
    "sleep_end_time": "2026-08-20T07:05:00+02:00",
    "light_sleep": 14000,
    "deep_sleep": 5000,
    "rem_sleep": 6000,
    "unrecognized_sleep_stage": 100,
    "total_interruption_duration": 900,
    "sleep_score": 82,
    "sleep_charge": 4,
    "continuity": 2.1,
    "continuity_class": 2,
    "sleep_cycles": 5,
}

RECHARGE_ITEM = {
    "date": "2026-08-20",
    "heart_rate_avg": 52,
    "beat_to_beat_avg": 1150,
    "heart_rate_variability_avg": 41,
    "breathing_rate_avg": 13.2,
    "nightly_recharge_status": 5,
    "ans_charge": 1.4,
    "ans_charge_status": 4,
    "hrv_samples": {"00:41": 40, "00:46": 42},
    "breathing_samples": {"00:39": 13.1},
}

DEFAULT_ROUTES: dict[str, object] = {
    "/v3/users/sleep": {"nights": [SLEEP_ITEM]},
    "/v3/users/nightly-recharge": {"recharges": [RECHARGE_ITEM]},
    "/v3/users/continuous-heart-rate": [
        {
            "date": "2026-08-20",
            "heart_rate_samples": [{"heart_rate": 63, "sample_time": "00:02:08"}],
        }
    ],
}


class _FakeResponse:
    def __init__(self, payload: bytes, status: int = 200) -> None:
        self._payload = payload
        self.status = status

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _FakeOpener:
    """Routes each request's path to canned JSON, an HTTPError, or an exception."""

    def __init__(self, routes: dict[str, object]) -> None:
        self.routes = routes
        self.calls: list[str] = []

    def __call__(self, request, timeout: float = 0.0) -> _FakeResponse:
        url = getattr(request, "full_url", request)
        self.calls.append(url)
        path = url.replace(polar_client.API_BASE, "").split("?", 1)[0]
        for prefix, body in self.routes.items():
            if path.startswith(prefix):
                if isinstance(body, Exception):
                    raise body
                return _FakeResponse(json.dumps(body).encode())
        raise AssertionError(f"unexpected URL fetched: {url}")


def _http_error(status: int, headers: dict | None = None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://www.polaraccesslink.com/x", status, "nope", headers or {}, None
    )


class TestAllowlist:
    """Transaction endpoints delete data on commit — they must be unreachable."""

    @pytest.mark.parametrize(
        "path",
        [
            "/v3/users/12/exercise-transactions",
            "/v3/users/12/activity-transactions",
            "/v3/users/12/activity-transactions/9/activities/3",
            "/v3/exercises",
            "/v3/users/physical-info",
        ],
    )
    def test_transaction_and_other_paths_are_refused(self, path: str) -> None:
        with pytest.raises(polar_client.DisallowedEndpointError):
            polar_client._api_request(path, "token")

    @pytest.mark.parametrize(
        "path",
        [
            "/v3/users",  # registration
            "/v3/users/sleep",
            "/v3/users/sleep/2026-08-20",
            "/v3/users/nightly-recharge/2026-08-20",
            "/v3/users/continuous-heart-rate?from=a&to=b",
        ],
    )
    def test_allowlisted_paths_build(self, path: str) -> None:
        request = polar_client._api_request(path, "token")
        assert request.full_url.startswith(polar_client.API_BASE)
        assert request.headers["Authorization"] == "Bearer token"


class TestClientParsing:
    def test_list_window_merges_sleep_and_recharge(self) -> None:
        opener = _FakeOpener(DEFAULT_ROUTES)
        result = polar_client.fetch_nights(
            "tok", NIGHT - dt.timedelta(days=3), TODAY, today=TODAY, opener=opener
        )
        assert result.notes == []
        night = result.nights[NIGHT]
        assert night.source_device_id == "AAA1"
        assert night.sleep_score == 82
        assert night.light_sleep_s == 14000
        assert night.hrv_rmssd_ms == 41
        assert night.nightly_recharge_status == 5
        assert night.ans_charge == pytest.approx(1.4)
        assert night.hrv_samples == {"00:41": 40, "00:46": 42}
        # Two list calls cover the window — no per-date requests.
        assert len(opener.calls) == 2

    def test_offset_timestamps_become_naive_utc(self) -> None:
        opener = _FakeOpener(DEFAULT_ROUTES)
        result = polar_client.fetch_nights(
            "tok", NIGHT, TODAY, today=TODAY, opener=opener
        )
        night = result.nights[NIGHT]
        # 23:10+02:00 is 21:10 UTC, stored naive like every other datetime here.
        assert night.sleep_start == dt.datetime(2026, 8, 19, 21, 10)
        assert night.sleep_start.tzinfo is None

    def test_continuous_hr_merges_into_the_same_night(self) -> None:
        opener = _FakeOpener(DEFAULT_ROUTES)
        result = polar_client.fetch_nights(
            "tok", NIGHT, TODAY, today=TODAY, opener=opener
        )
        polar_client.fetch_continuous_hr("tok", NIGHT, TODAY, result, opener=opener)
        assert result.nights[NIGHT].hr_samples == [
            {"heart_rate": 63, "sample_time": "00:02:08"}
        ]

    def test_dates_outside_the_range_are_dropped(self) -> None:
        opener = _FakeOpener(DEFAULT_ROUTES)
        result = polar_client.fetch_nights(
            "tok", TODAY, TODAY, today=TODAY, opener=opener
        )
        assert NIGHT not in result.nights

    def test_older_dates_use_available_then_by_date(self) -> None:
        old = dt.date(2026, 6, 1)
        opener = _FakeOpener(
            {
                "/v3/users/sleep/available": {"available": [{"date": "2026-06-01"}]},
                "/v3/users/sleep/2026-06-01": dict(SLEEP_ITEM, date="2026-06-01"),
                "/v3/users/nightly-recharge/2026-06-01": dict(
                    RECHARGE_ITEM, date="2026-06-01"
                ),
                "/v3/users/sleep": {"nights": []},
                "/v3/users/nightly-recharge": {"recharges": []},
            }
        )
        result = polar_client.fetch_nights(
            "tok", old, TODAY, today=TODAY, opener=opener
        )
        assert result.nights[old].sleep_score == 82
        assert result.nights[old].hrv_rmssd_ms == 41
        # Only the one date Polar said it holds was requested by date.
        by_date = [c for c in opener.calls if "2026-06-01" in c]
        assert len(by_date) == 2

    def test_missing_fields_are_none_not_errors(self) -> None:
        opener = _FakeOpener(
            {
                "/v3/users/sleep": {"nights": [{"date": "2026-08-20"}]},
                "/v3/users/nightly-recharge": {"recharges": []},
            }
        )
        result = polar_client.fetch_nights(
            "tok", NIGHT, TODAY, today=TODAY, opener=opener
        )
        night = result.nights[NIGHT]
        assert night.sleep_score is None
        assert night.hrv_rmssd_ms is None
        assert result.notes == []


class TestClientFailures:
    """Every failure is a note. Nothing raises."""

    def test_401_flags_auth_and_notes_it(self) -> None:
        opener = _FakeOpener({"/v3/users/": _http_error(401)})
        result = polar_client.fetch_nights(
            "tok", NIGHT, TODAY, today=TODAY, opener=opener
        )
        assert result.auth_failed is True
        assert any("401" in n for n in result.notes)
        assert result.nights == {}

    def test_429_flags_rate_limit_with_the_reset_hint(self) -> None:
        opener = _FakeOpener({"/v3/users/": _http_error(429, {"RateLimit-Reset": "600"})})
        result = polar_client.fetch_nights(
            "tok", NIGHT, TODAY, today=TODAY, opener=opener
        )
        assert result.rate_limited is True
        assert any("600" in n for n in result.notes)

    def test_network_failure_is_a_note(self) -> None:
        opener = _FakeOpener({"/v3/users/": OSError("network unreachable")})
        result = polar_client.fetch_nights(
            "tok", NIGHT, TODAY, today=TODAY, opener=opener
        )
        assert any("network unreachable" in n for n in result.notes)
        assert result.auth_failed is False

    def test_unreadable_body_is_a_note(self) -> None:
        class Junk(_FakeOpener):
            def __call__(self, request, timeout: float = 0.0) -> _FakeResponse:
                return _FakeResponse(b"<html>not json</html>")

        result = polar_client.fetch_nights(
            "tok", NIGHT, TODAY, today=TODAY, opener=Junk({})
        )
        assert any("unreadable" in n for n in result.notes)

    def test_rate_limit_stops_the_by_date_walk(self) -> None:
        old = dt.date(2026, 6, 1)
        opener = _FakeOpener(
            {
                "/v3/users/sleep/available": {
                    "available": [{"date": "2026-06-01"}, {"date": "2026-06-02"}]
                },
                "/v3/users/sleep/2026-06-01": _http_error(429),
                "/v3/users/nightly-recharge/2026-06-01": _http_error(429),
                "/v3/users/sleep": {"nights": []},
                "/v3/users/nightly-recharge": {"recharges": []},
            }
        )
        result = polar_client.fetch_nights(
            "tok", old, TODAY, today=TODAY, opener=opener
        )
        assert result.rate_limited is True
        assert not any("2026-06-02" in c for c in opener.calls)
        assert any("stopped at 2026-06-02" in n for n in result.notes)


class TestRegistration:
    def test_409_already_registered_is_success(self) -> None:
        opener = _FakeOpener({"/v3/users": _http_error(409)})
        linked, notes = polar_client.register_user("tok", "ferhat", opener=opener)
        assert linked is True
        assert notes == []

    def test_200_is_success(self) -> None:
        opener = _FakeOpener({"/v3/users": {"polar-user-id": 1, "member-id": "x"}})
        linked, _notes = polar_client.register_user("tok", "x", opener=opener)
        assert linked is True

    def test_403_missing_consent_is_a_refusal_with_a_reason(self) -> None:
        opener = _FakeOpener({"/v3/users": _http_error(403)})
        linked, notes = polar_client.register_user("tok", "x", opener=opener)
        assert linked is False
        assert any("consent" in n for n in notes)

    def test_token_exchange_returns_the_grant(self) -> None:
        opener = _FakeOpener(
            {"": {"access_token": "abc123", "x_user_id": 4242, "token_type": "bearer"}}
        )
        grant, notes = polar_client.exchange_code(
            "code", "cid", "secret", "http://localhost:5000/polar/callback",
            opener=opener,
        )
        assert grant is not None
        assert grant.access_token == "abc123"
        assert grant.polar_user_id == "4242"
        assert notes == []

    def test_token_exchange_without_a_token_is_a_note(self) -> None:
        opener = _FakeOpener({"": {"error": "invalid_grant"}})
        grant, notes = polar_client.exchange_code(
            "code", "cid", "secret", "http://localhost:5000/polar/callback",
            opener=opener,
        )
        assert grant is None
        assert notes


@pytest.fixture()
def enabled_app(tmp_path) -> Iterator[Flask]:
    config = TestConfig()
    config.DATA_DIR = tmp_path / "data"
    config.POLAR_ENABLED = True
    config.POLAR_CLIENT_ID = "test-client"
    config.POLAR_CLIENT_SECRET = "test-secret"
    app = create_app(config)
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


def _linked_person(name: str = "Flo", slug: str = "flo") -> tuple[Person, PolarAccount]:
    person = Person(name=name, slug=slug)
    db.session.add(person)
    db.session.commit()
    account = PolarAccount(
        person_id=person.id,
        polar_user_id="4242",
        access_token="tok",
        member_id=slug,
    )
    db.session.add(account)
    db.session.commit()
    return person, account


class TestSync:
    def test_upsert_writes_every_column(self, enabled_app: Flask) -> None:
        person, account = _linked_person()
        opener = _FakeOpener(DEFAULT_ROUTES)
        result = sync_person(
            person, account, NIGHT, TODAY, today=TODAY, opener=opener
        )
        assert result.created == 1
        assert result.updated == 0

        night = db.session.query(FlowNight).one()
        assert night.date == NIGHT
        assert night.source_device_id == "AAA1"
        assert night.sleep_score == 82
        assert night.deep_sleep_s == 5000
        assert night.total_sleep_s == 25100
        assert night.hrv_rmssd_ms == 41
        assert night.beat_to_beat_avg_ms == 1150
        assert night.breathing_rate_avg == pytest.approx(13.2)
        assert night.ans_charge_status == 4
        assert night.hrv_samples == {"00:41": 40, "00:46": 42}
        assert night.hr_samples == [{"heart_rate": 63, "sample_time": "00:02:08"}]
        assert night.has_sleep() and night.has_recharge()
        assert account.last_sync_at is not None
        assert "1 new" in (account.last_sync_note or "")

    def test_second_sync_updates_rather_than_duplicating(
        self, enabled_app: Flask
    ) -> None:
        person, account = _linked_person()
        opener = _FakeOpener(DEFAULT_ROUTES)
        sync_person(person, account, NIGHT, TODAY, today=TODAY, opener=opener)

        changed = dict(RECHARGE_ITEM, heart_rate_variability_avg=55)
        opener2 = _FakeOpener(
            dict(DEFAULT_ROUTES, **{"/v3/users/nightly-recharge": {"recharges": [changed]}})
        )
        result = sync_person(
            person, account, NIGHT, TODAY, today=TODAY, opener=opener2
        )
        assert result.created == 0
        assert result.updated == 1
        assert db.session.query(FlowNight).count() == 1
        db.session.expire_all()
        assert db.session.query(FlowNight).one().hrv_rmssd_ms == 55

    def test_dates_with_nothing_behind_them_are_not_stored(
        self, enabled_app: Flask
    ) -> None:
        person, account = _linked_person()
        opener = _FakeOpener(
            {
                "/v3/users/sleep": {"nights": [{"date": "2026-08-20"}]},
                "/v3/users/nightly-recharge": {"recharges": []},
                "/v3/users/continuous-heart-rate": [],
            }
        )
        result = sync_person(
            person, account, NIGHT, TODAY, today=TODAY, opener=opener
        )
        assert result.created == 0
        assert result.empty == 1
        assert db.session.query(FlowNight).count() == 0

    def test_revoked_token_is_dropped_so_it_is_not_retried(
        self, enabled_app: Flask
    ) -> None:
        person, account = _linked_person()
        opener = _FakeOpener({"/v3/users/": _http_error(401)})
        result = sync_person(
            person, account, NIGHT, TODAY, today=TODAY, opener=opener
        )
        assert result.auth_failed is True
        assert db.session.query(PolarAccount).count() == 0

    def test_partial_window_is_kept_when_rate_limited(
        self, enabled_app: Flask
    ) -> None:
        """A wall mid-sync must not discard what already arrived."""
        person, account = _linked_person()
        opener = _FakeOpener(
            {
                "/v3/users/sleep/available": {"available": [{"date": "2026-06-01"}]},
                "/v3/users/sleep/2026-06-01": _http_error(429),
                "/v3/users/nightly-recharge/2026-06-01": _http_error(429),
                "/v3/users/sleep": {"nights": [SLEEP_ITEM]},
                "/v3/users/nightly-recharge": {"recharges": [RECHARGE_ITEM]},
                "/v3/users/continuous-heart-rate": [],
            }
        )
        result = sync_person(
            person, account, dt.date(2026, 6, 1), TODAY, today=TODAY, opener=opener
        )
        assert result.rate_limited is True
        assert result.created == 1  # the recent night survived
        assert "rate limited" in result.summary()


class TestGating:
    def test_routes_absent_when_disabled(self, app: Flask) -> None:
        rules = {r.rule for r in app.url_map.iter_rules()}
        assert not any(r.startswith("/polar") for r in rules)
        assert app.test_client().get("/polar/link/1").status_code == 404

    def test_routes_present_when_enabled(self, enabled_app: Flask) -> None:
        rules = {r.rule for r in enabled_app.url_map.iter_rules()}
        assert "/polar/callback" in rules
        assert "/polar/sync/<int:person_id>" in rules

    def test_person_page_hides_the_section_when_disabled(self, app: Flask) -> None:
        person = Person(name="Nope", slug="nope")
        db.session.add(person)
        db.session.commit()
        page = app.test_client().get(f"/people/{person.id}")
        assert page.status_code == 200
        assert b"Polar Flow" not in page.data

    def test_person_page_offers_linking_when_enabled(
        self, enabled_app: Flask
    ) -> None:
        person = Person(name="Yes", slug="yes")
        db.session.add(person)
        db.session.commit()
        page = enabled_app.test_client().get(f"/people/{person.id}")
        assert b"Link Polar Flow" in page.data

    def test_person_page_shows_the_linked_state(self, enabled_app: Flask) -> None:
        person, account = _linked_person()
        account.last_sync_note = "3 new, 0 updated"
        db.session.commit()
        page = enabled_app.test_client().get(f"/people/{person.id}")
        assert b"3 new, 0 updated" in page.data
        assert b"Unlink" in page.data

    def test_unlink_deletes_the_token_but_keeps_the_nights(
        self, enabled_app: Flask
    ) -> None:
        person, _account = _linked_person()
        db.session.add(FlowNight(person_id=person.id, date=NIGHT, sleep_score=70))
        db.session.commit()
        resp = enabled_app.test_client().post(f"/polar/unlink/{person.id}")
        assert resp.status_code == 302
        assert db.session.query(PolarAccount).count() == 0
        assert db.session.query(FlowNight).count() == 1

    def test_callback_rejects_a_state_it_did_not_issue(
        self, enabled_app: Flask
    ) -> None:
        resp = enabled_app.test_client().get("/polar/callback?code=x&state=forged")
        assert resp.status_code == 302
        assert "/people" in resp.headers["Location"]
        assert db.session.query(PolarAccount).count() == 0


class TestPresentation:
    def test_line_names_the_source_and_omits_missing_values(self) -> None:
        night = FlowNight(
            person_id=1, date=NIGHT, light_sleep_s=3600, sleep_score=82, hrv_rmssd_ms=41
        )
        line = format_night_line(night)
        assert "sleep score 82" in line
        assert "overnight RMSSD 41 ms" in line
        assert "Polar Flow" in line
        assert "recharge" not in line  # absent, so not printed

    def test_line_is_none_without_content(self) -> None:
        assert format_night_line(None) is None
        assert format_night_line(FlowNight(person_id=1, date=NIGHT)) is None

    def test_night_before_matches_the_recording_date(self, enabled_app: Flask) -> None:
        person, _ = _linked_person()
        db.session.add(FlowNight(person_id=person.id, date=NIGHT, sleep_score=70))
        db.session.commit()
        db.session.refresh(person)
        assert night_before(person, dt.datetime(2026, 8, 20, 7, 30)) is not None
        assert night_before(person, dt.datetime(2026, 8, 19, 7, 30)) is None
        assert night_before(person, None) is None


#: The trigger statistics drop sessions under MIN_ANALYSED_S (600 s), so
#: anything feeding assemble_observations has to clear that bar.
LONG_ENOUGH_S = 700.0


def _upload(
    app: Flask,
    person: Person,
    seed: int,
    name: str,
    duration_s: float = 150.0,
) -> Session:
    client = app.test_client()
    client.get("/sessions/upload")  # seed builtins
    activity = db.session.query(ActivityType).filter_by(profile_key="sitting").one()
    resp = client.post(
        "/sessions/upload",
        data={
            "person_id": str(person.id),
            "activity_type_id": str(activity.id),
            "file": (
                io.BytesIO(make_csv_bytes(duration_s=duration_s, seed=seed)),
                name,
            ),
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 302
    db.session.expire_all()
    return db.session.query(Session).order_by(Session.id.desc()).first()


class TestSessionAndTrendViews:
    def test_session_page_without_a_night_is_unchanged(
        self, enabled_app: Flask
    ) -> None:
        person, _ = _linked_person()
        session = _upload(enabled_app, person, 21, "ecg_a.csv")
        page = enabled_app.test_client().get(f"/sessions/{session.id}")
        assert page.status_code == 200
        assert b"Prior night" not in page.data

    def test_session_page_shows_the_prior_night(self, enabled_app: Flask) -> None:
        person, _ = _linked_person()
        session = _upload(enabled_app, person, 21, "ecg_a.csv")
        db.session.add(
            FlowNight(
                person_id=person.id,
                date=session.recorded_at.date(),
                light_sleep_s=25200,
                sleep_score=82,
                hrv_rmssd_ms=41,
                nightly_recharge_status=5,
                hr_avg_bpm=52,
            )
        )
        db.session.commit()
        page = enabled_app.test_client().get(f"/sessions/{session.id}")
        assert b"Prior night" in page.data
        assert b"sleep score 82" in page.data
        assert b"overnight RMSSD 41 ms" in page.data
        assert b"recharge good" in page.data
        # The caveat travels with the number, never separately.
        assert b"not comparable" in page.data

    def test_trends_draws_a_separate_flow_figure(self, enabled_app: Flask) -> None:
        person, _ = _linked_person()
        first = _upload(enabled_app, person, 21, "ecg_a.csv")
        second = _upload(enabled_app, person, 22, "ecg_b.csv")
        # Spread the two sessions so several nights fall inside their span.
        first.recorded_at = dt.datetime(2026, 8, 10, 9, 0)
        second.recorded_at = dt.datetime(2026, 8, 20, 9, 0)
        for offset, rmssd in ((0, 38), (4, 44), (10, 41)):
            db.session.add(
                FlowNight(
                    person_id=person.id,
                    date=dt.date(2026, 8, 10) + dt.timedelta(days=offset),
                    hrv_rmssd_ms=rmssd,
                )
            )
        db.session.commit()

        page = enabled_app.test_client().get(f"/compare/{person.id}/trends/sitting")
        assert page.status_code == 200
        assert b"Overnight recovery (Polar Flow)" in page.data
        assert b"do not read the two as one series" in page.data

    def test_trends_omits_the_figure_without_nights(self, enabled_app: Flask) -> None:
        person, _ = _linked_person()
        _upload(enabled_app, person, 21, "ecg_a.csv")
        _upload(enabled_app, person, 22, "ecg_b.csv")
        page = enabled_app.test_client().get(f"/compare/{person.id}/trends/sitting")
        assert page.status_code == 200
        assert b"Overnight recovery (Polar Flow)" not in page.data


class TestCli:
    def test_disabled_message(self, app: Flask) -> None:
        result = app.test_cli_runner().invoke(args=["polar-sync"])
        assert "off" in result.output

    def test_no_linked_accounts_says_so(self, enabled_app: Flask) -> None:
        result = enabled_app.test_cli_runner().invoke(args=["polar-sync"])
        assert "No linked Polar accounts" in result.output

    def test_sync_reports_per_person(
        self, enabled_app: Flask, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        person, _account = _linked_person()
        import app.polar.sync as sync_module

        # The CLI calls sync_person without an opener, so the fake is injected
        # here rather than by patching urlopen — the ``opener`` default is
        # bound at definition time and patching the module attribute would
        # silently let a real request through.
        opener = _FakeOpener(DEFAULT_ROUTES)
        real = sync_module.sync_person
        monkeypatch.setattr(
            sync_module,
            "sync_person",
            lambda *a, **kw: real(*a, **{**kw, "opener": opener, "today": TODAY}),
        )
        result = enabled_app.test_cli_runner().invoke(
            args=["polar-sync", "--from", "2026-08-19", "--to", "2026-08-21"]
        )
        assert result.exit_code == 0
        assert "Flo" in result.output
        assert "1 new" in result.output
        assert db.session.query(FlowNight).count() == 1

    def test_from_after_to_is_refused(self, enabled_app: Flask) -> None:
        _linked_person()
        result = enabled_app.test_cli_runner().invoke(
            args=["polar-sync", "--from", "2026-08-21", "--to", "2026-08-19"]
        )
        assert "nothing to do" in result.output


class TestTriggerOutcome:
    def test_overnight_rmssd_is_a_selectable_outcome(self) -> None:
        from app.triggers.stats import OUTCOMES

        spec = OUTCOMES["flow_rmssd"]
        assert spec.kind == "pct_change"
        assert spec.group_unit == "ms"

    def test_observations_carry_the_matching_night(self, enabled_app: Flask) -> None:
        from app.triggers.stats import assemble_observations

        person, _ = _linked_person()
        session = _upload(
            enabled_app, person, 21, "ecg_a.csv", duration_s=LONG_ENOUGH_S
        )
        db.session.add(
            FlowNight(
                person_id=person.id,
                date=session.recorded_at.date(),
                hrv_rmssd_ms=40,
            )
        )
        db.session.commit()
        db.session.refresh(person)

        observations, _notes = assemble_observations(person)
        assert observations
        assert observations[0].flow_rmssd_ms == pytest.approx(40.0)
        assert observations[0].ln_flow_rmssd == pytest.approx(3.6889, abs=1e-3)

    def test_observations_without_a_night_leave_the_outcome_missing(
        self, enabled_app: Flask
    ) -> None:
        from app.triggers.stats import assemble_observations

        person, _ = _linked_person()
        _upload(enabled_app, person, 21, "ecg_a.csv", duration_s=LONG_ENOUGH_S)
        db.session.refresh(person)
        observations, _notes = assemble_observations(person)
        assert observations
        assert observations[0].ln_flow_rmssd is None
