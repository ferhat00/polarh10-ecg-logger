"""Comparison and longitudinal-trend tests.

Sessions, metrics, and .npz caches are constructed directly (no pipeline
runs) — comparison is contractually a cache reader.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import pytest
from flask import Flask
from flask.testing import FlaskClient

from app.activities.seed import ensure_builtin_activity_types
from app.comparison import (
    BASELINE_MIN_SESSIONS,
    ComparisonError,
    build_comparison,
    build_trend,
    load_cached_rr,
    resting_hr_bpm,
)
from app.extensions import db
from app.models import ActivityType, Metrics, Person, Session
from app.pipeline.rr import RRSeries


@pytest.fixture()
def person(app: Flask) -> Person:
    ensure_builtin_activity_types()
    p = Person(name="Trender", slug="trender")
    db.session.add(p)
    db.session.commit()
    return p


def _activity(profile_key: str) -> ActivityType:
    return db.session.query(ActivityType).filter_by(profile_key=profile_key).one()


def _make_session(
    app: Flask,
    person: Person,
    profile_key: str = "supine",
    recorded: dt.datetime | None = None,
    rmssd: float = 20.0,
    sdnn: float = 30.0,
    duration_s: float = 600.0,
    mean_hr: float = 62.0,
    with_cache: bool = True,
    n: int | None = None,
) -> Session:
    n = n if n is not None else (db.session.query(Session).count() + 1)
    session = Session(
        person_id=person.id,
        activity_type_id=_activity(profile_key).id,
        recorded_at=recorded or dt.datetime(2026, 7, 1, tzinfo=dt.UTC) + dt.timedelta(days=n),
        duration_s=duration_s,
        analysed_s=duration_s,
        excluded_s=0.0,
        beats_corrected_pct=0.3,
        original_filename=f"s{n}.csv",
        stored_path="",
        file_sha256=f"{n:064d}",
        processing_status="done",
    )
    db.session.add(session)
    db.session.flush()
    db.session.add(
        Metrics(
            session_id=session.id,
            n_beats=int(duration_s / 0.9),
            mean_hr_bpm=mean_hr,
            mean_rr_ms=60000.0 / mean_hr,
            rmssd_ms=rmssd,
            sdnn_ms=sdnn,
            pnn50_pct=5.0,
            sd1_ms=rmssd / math.sqrt(2),
            sd2_ms=sdnn * 1.3,
        )
    )
    upload_dir = Path(app.config["DATA_DIR"]) / "uploads" / person.slug
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored = upload_dir / f"{session.id}__s{n}.csv"
    stored.write_text("placeholder")
    session.stored_path = str(stored)
    if with_cache:
        rr_ms_value = 60000.0 / mean_hr
        count = int(duration_s * 1000 / rr_ms_value)
        rng = np.random.default_rng(n)
        rr = np.full(count, rr_ms_value) + rng.normal(0, 8.0, count)
        t = np.cumsum(rr) / 1000.0
        np.savez_compressed(
            stored.with_suffix(".npz"),
            rr_ms=rr,
            rr_t_s=t,
            rr_discontinuity=np.zeros(count, dtype=bool),
        )
    db.session.commit()
    return session


class TestComparisonGuardrails:
    def test_needs_two_sessions(self, app: Flask, person: Person) -> None:
        s1 = _make_session(app, person)
        with pytest.raises(ComparisonError, match="at least two"):
            build_comparison([s1])

    def test_mixed_activities_rejected_without_override(
        self, app: Flask, person: Person
    ) -> None:
        s1 = _make_session(app, person, "supine")
        s2 = _make_session(app, person, "standing")
        with pytest.raises(ComparisonError, match="different activity types"):
            build_comparison([s1, s2], mixed=False)

    def test_mixed_override_attaches_posture_warning(
        self, app: Flask, person: Person
    ) -> None:
        s1 = _make_session(app, person, "supine")
        s2 = _make_session(app, person, "standing")
        data = build_comparison([s1, s2], mixed=True)
        assert data.mixed is True
        assert any("Posture and motion dominate" in w for w in data.warnings)

    def test_same_activity_no_posture_warning(self, app: Flask, person: Person) -> None:
        s1 = _make_session(app, person)
        s2 = _make_session(app, person)
        data = build_comparison([s1, s2])
        assert data.mixed is False
        assert not any("Posture" in w for w in data.warnings)

    def test_sdnn_duration_guard(self, app: Flask, person: Person) -> None:
        s1 = _make_session(app, person, duration_s=300.0)
        s2 = _make_session(app, person, duration_s=1110.0)  # 5 vs 18.5 min
        data = build_comparison([s1, s2])
        sdnn_row = next(r for r in data.rows if r.label.startswith("SDNN"))
        assert sdnn_row.note is not None and "length dependent" in sdnn_row.note
        assert any("materially" in w for w in data.warnings)

    def test_similar_durations_no_guard(self, app: Flask, person: Person) -> None:
        s1 = _make_session(app, person, duration_s=600.0)
        s2 = _make_session(app, person, duration_s=660.0)
        data = build_comparison([s1, s2])
        sdnn_row = next(r for r in data.rows if r.label.startswith("SDNN"))
        assert sdnn_row.note is None

    def test_reduced_confidence_warning(self, app: Flask, person: Person) -> None:
        s1 = _make_session(app, person)
        s2 = _make_session(app, person)
        s2.reduced_confidence = True
        db.session.commit()
        data = build_comparison([s1, s2])
        assert any("reduced-confidence" in w for w in data.warnings)


class TestComparisonRows:
    def test_deltas_vs_earliest(self, app: Flask, person: Person) -> None:
        s1 = _make_session(app, person, rmssd=20.0)
        s2 = _make_session(app, person, rmssd=25.0)
        data = build_comparison([s2, s1])  # order given shouldn't matter
        rmssd_row = next(r for r in data.rows if r.label == "RMSSD")
        assert rmssd_row.values == [20.0, 25.0]  # sorted by recorded_at
        assert rmssd_row.deltas[0] is None
        assert rmssd_row.deltas[1] == pytest.approx(5.0)
        assert rmssd_row.formatted_delta(1) == "+5.0"

    def test_ln_rmssd_row(self, app: Flask, person: Person) -> None:
        s1 = _make_session(app, person, rmssd=20.0)
        s2 = _make_session(app, person, rmssd=25.0)
        data = build_comparison([s1, s2])
        ln_row = next(r for r in data.rows if r.label == "ln(RMSSD)")
        assert ln_row.values[0] == pytest.approx(math.log(20.0))
        assert ln_row.values[1] == pytest.approx(math.log(25.0))

    def test_missing_metrics_render_as_none(self, app: Flask, person: Person) -> None:
        s1 = _make_session(app, person)
        s2 = _make_session(app, person)
        s2.metrics.rmssd_ms = None  # e.g. suppressed at intensity
        db.session.commit()
        data = build_comparison([s1, s2])
        rmssd_row = next(r for r in data.rows if r.label == "RMSSD")
        assert rmssd_row.values[1] is None
        assert rmssd_row.formatted(1) == "—"


class TestCacheReading:
    def test_load_cached_rr_roundtrip(self, app: Flask, person: Person) -> None:
        s = _make_session(app, person, mean_hr=60.0)
        rr = load_cached_rr(s)
        assert rr is not None
        assert float(np.mean(rr.rr_ms)) == pytest.approx(1000.0, abs=5.0)

    def test_missing_cache_returns_none(self, app: Flask, person: Person) -> None:
        s = _make_session(app, person, with_cache=False)
        assert load_cached_rr(s) is None

    def test_resting_hr_is_lowest_sustained(self) -> None:
        # 10 min at 70 bpm with 2 min at 55 in the middle.
        rr = np.concatenate(
            [np.full(280, 857.0), np.full(110, 1091.0), np.full(280, 857.0)]
        )
        series = RRSeries(
            rr_ms=rr,
            t_s=np.cumsum(rr) / 1000.0,
            discontinuity=np.zeros(len(rr), dtype=bool),
            n_dropped_excluded=0,
            n_dropped_ceiling=0,
            n_dropped_floor=0,
        )
        resting = resting_hr_bpm(series)
        assert resting == pytest.approx(55.0, abs=1.0)


class TestTrends:
    def test_provisional_below_five_sessions(self, app: Flask, person: Person) -> None:
        sessions = [_make_session(app, person) for _ in range(3)]
        data = build_trend(sessions, "supine")
        assert data.provisional is True
        assert data.ln_rmssd.band_mean == []

    def test_band_appears_at_five_sessions(self, app: Flask, person: Person) -> None:
        sessions = [
            _make_session(app, person, rmssd=18.0 + i) for i in range(BASELINE_MIN_SESSIONS + 1)
        ]
        data = build_trend(sessions, "supine")
        assert data.provisional is False
        assert len(data.ln_rmssd.band_mean) == len(sessions)
        # Band exists only once the window is full.
        assert data.ln_rmssd.band_mean[BASELINE_MIN_SESSIONS - 2] is None
        assert data.ln_rmssd.band_mean[BASELINE_MIN_SESSIONS - 1] is not None
        assert (
            data.ln_rmssd.band_lo[BASELINE_MIN_SESSIONS - 1]
            < data.ln_rmssd.band_mean[BASELINE_MIN_SESSIONS - 1]
            < data.ln_rmssd.band_hi[BASELINE_MIN_SESSIONS - 1]
        )

    def test_trend_tracks_ln_rmssd(self, app: Flask, person: Person) -> None:
        sessions = [_make_session(app, person, rmssd=20.0), _make_session(app, person, rmssd=40.0)]
        data = build_trend(sessions, "supine")
        assert data.points[0].ln_rmssd == pytest.approx(math.log(20.0))
        assert data.points[1].ln_rmssd == pytest.approx(math.log(40.0))


class TestCompareViews:
    def test_select_page_groups_by_activity(
        self, client: FlaskClient, app: Flask, person: Person
    ) -> None:
        _make_session(app, person, "supine")
        _make_session(app, person, "standing")
        resp = client.get(f"/compare/{person.id}")
        assert b"supine" in resp.data
        assert b"standing" in resp.data
        assert b"Allow cross-activity comparison" in resp.data

    def test_view_rejects_mixed_without_override(
        self, client: FlaskClient, app: Flask, person: Person
    ) -> None:
        s1 = _make_session(app, person, "supine")
        s2 = _make_session(app, person, "standing")
        resp = client.get(
            f"/compare/{person.id}/view?sessions={s1.id}&sessions={s2.id}",
            follow_redirects=True,
        )
        assert b"different activity types" in resp.data
        assert b"Metric" not in resp.data

    def test_view_renders_table_and_overlays(
        self, client: FlaskClient, app: Flask, person: Person
    ) -> None:
        s1 = _make_session(app, person)
        s2 = _make_session(app, person)
        resp = client.get(f"/compare/{person.id}/view?sessions={s1.id}&sessions={s2.id}")
        assert resp.status_code == 200
        assert b"RMSSD" in resp.data
        assert resp.data.count(b"data:image/png;base64,") >= 2

    def test_trends_page_shows_provisional_notice(
        self, client: FlaskClient, app: Flask, person: Person
    ) -> None:
        for _ in range(3):
            _make_session(app, person)
        resp = client.get(f"/compare/{person.id}/trends/supine")
        assert b"Provisional baseline" in resp.data
        assert b"ln(RMSSD)" in resp.data
