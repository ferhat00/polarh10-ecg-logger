"""ECG-derived respiration: rate recovery, gating, paced-breathing caution.

Ground truth is synthetic: RR intervals carry a known RSA modulation and the
R-wave amplitudes carry a matching (or deliberately mismatched) modulation,
so the two-channel fusion logic is tested directly.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest
from flask import Flask
from flask.testing import FlaskClient

from app.extensions import db
from app.models import ActivityType, Metrics, Person, ProcessingStatus, Session
from app.pipeline.respiration import (
    MIN_WINDOWS,
    RespirationResult,
    estimate_respiration,
)
from app.pipeline.rr import RRSeries
from tests.synth_util import build_ecg_from_rr

FS_HZ = 130.0


def _beat_times(duration_s: float, resp_hz: float, rr_base_ms: float = 800.0,
                rsa_ms: float = 30.0) -> np.ndarray:
    """Beat times whose RR intervals oscillate at the given resp frequency."""
    times = [0.5]
    while times[-1] < duration_s:
        rr = rr_base_ms + rsa_ms * np.sin(2 * np.pi * resp_hz * times[-1])
        times.append(times[-1] + rr / 1000.0)
    return np.array(times)


def _inputs(
    duration_s: float,
    rr_resp_hz: float,
    amp_resp_hz: float | None = None,
    amp_mod: float = 0.08,
):
    """(rr, ecg_clean, peaks, peak_times, quality) with known modulations.

    ``amp_resp_hz`` defaults to the RR modulation frequency (agreeing
    channels); pass a different value to make the channels disagree.
    """
    beat_times = _beat_times(duration_s, rr_resp_hz)
    rr_ms = np.diff(beat_times) * 1000.0
    rr = RRSeries(
        rr_ms=rr_ms,
        t_s=beat_times[1:],
        discontinuity=np.zeros(len(rr_ms), dtype=bool),
        n_dropped_excluded=0,
        n_dropped_ceiling=0,
        n_dropped_floor=0,
    )
    peaks = np.round(beat_times * FS_HZ).astype(np.int64)
    ecg_clean = np.zeros(int((duration_s + 2) * FS_HZ))
    f_amp = amp_resp_hz if amp_resp_hz is not None else rr_resp_hz
    ecg_clean[peaks] = 0.60 + amp_mod * np.sin(2 * np.pi * f_amp * beat_times)
    quality = SimpleNamespace(excluded_segments=[])
    return rr, ecg_clean, peaks, beat_times, quality


class TestRateRecovery:
    def test_recovers_15_brpm(self) -> None:
        result = estimate_respiration(*_inputs(360.0, rr_resp_hz=0.25))
        assert result.median_brpm == pytest.approx(15.0, abs=1.5)
        assert result.n_windows_used >= MIN_WINDOWS
        assert result.paced_breathing is False

    def test_slow_breathing_flags_paced(self) -> None:
        result = estimate_respiration(*_inputs(360.0, rr_resp_hz=0.1))
        assert result.median_brpm == pytest.approx(6.0, abs=1.0)
        assert result.paced_breathing is True
        assert result.paced_rate_brpm == pytest.approx(6.0, abs=1.0)

    def test_channel_disagreement_drops_windows(self) -> None:
        # RR says 15 brpm, amplitude says 27 brpm — no window may fuse.
        result = estimate_respiration(
            *_inputs(360.0, rr_resp_hz=0.25, amp_resp_hz=0.45)
        )
        assert result.median_brpm is None
        assert result.n_windows_used < MIN_WINDOWS
        assert any("gates" in n for n in result.notes)

    def test_short_record_reports_nothing(self) -> None:
        result = estimate_respiration(*_inputs(60.0, rr_resp_hz=0.25))
        assert result.median_brpm is None
        assert result.notes


class TestExtras:
    def test_as_extras_roundtrip(self) -> None:
        result = estimate_respiration(*_inputs(360.0, rr_resp_hz=0.25))
        extras = result.as_extras()
        assert extras["median_brpm"] == result.median_brpm
        assert len(extras["window_brpm"]) == result.n_windows_used
        assert extras["paced_breathing"] is False
        assert "charlton2016" in extras["method"]


def _paced_csv_bytes(duration_s: float = 330.0) -> bytes:
    """An ECG export whose RR *and* R-amplitude oscillate at 6 brpm."""
    beat_times = _beat_times(duration_s, resp_hz=0.1, rr_base_ms=900.0, rsa_ms=60.0)
    rr_ms = np.diff(beat_times) * 1000.0
    beat_amps = {
        k: 0.60 + 0.08 * np.sin(2 * np.pi * 0.1 * t)
        for k, t in enumerate(beat_times)
    }
    t, ecg, _ = build_ecg_from_rr(rr_ms, seed=11, beat_amps=beat_amps)
    keep = t <= duration_s
    t, ecg = t[keep], ecg[keep]
    import datetime as dt

    start_ns = int(dt.datetime(2026, 8, 12, 7, 0, tzinfo=dt.UTC).timestamp() * 1e9)
    lines = ["time,ecg,hr,rr,marker"]
    for ti, v in zip(t, ecg, strict=True):
        lines.append(f"{start_ns + round(ti * 1e9)},{v:.4f}")
    return ("\n".join(lines) + "\n").encode()


class TestUploadIntegration:
    @pytest.fixture()
    def person(self, app: Flask) -> Person:
        p = Person(name="Breather", slug="breather")
        db.session.add(p)
        db.session.commit()
        return p

    def _upload(self, client: FlaskClient, person: Person, payload: bytes) -> Session:
        client.get("/sessions/upload")
        activity = db.session.query(ActivityType).filter_by(profile_key="sitting").one()
        resp = client.post(
            "/sessions/upload",
            data={
                "person_id": str(person.id),
                "activity_type_id": str(activity.id),
                "context_note": "",
                "file": (io.BytesIO(payload), "ecg_2026-08-12.csv"),
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 302
        db.session.expire_all()
        return db.session.query(Session).one()

    def test_paced_breathing_caution_end_to_end(
        self, client: FlaskClient, app: Flask, person: Person
    ) -> None:
        session = self._upload(client, person, _paced_csv_bytes())
        assert session.processing_status == ProcessingStatus.DONE

        metrics = db.session.query(Metrics).one()
        assert metrics.resp_rate_median_brpm == pytest.approx(6.0, abs=1.0)
        resp_extras = metrics.extras["respiration"]
        assert resp_extras["paced_breathing"] is True

        cautions = metrics.extras["cautions"]
        assert any("inflates RMSSD" in c["reason"] for c in cautions)

        # The caution and the estimate both reach the user-facing surfaces.
        detail = client.get(f"/sessions/{session.id}")
        assert b"slow breathing" in detail.data.lower().replace(b"paced ", b"")
        report = client.get(f"/sessions/{session.id}/report")
        assert b"Resp rate (EDR)" in report.data
        assert b"estimated" in report.data
        assert b"brpm" in report.data

        from app.logbook.writer import log_path_for

        text = log_path_for(person).read_text(encoding="utf-8")
        assert "**Respiration (EDR estimate):**" in text

    def test_normal_breathing_no_caution(
        self, client: FlaskClient, app: Flask, person: Person
    ) -> None:
        # The standard upload fixture: RR modulated at 0.1 Hz (LF/Mayer
        # band) but amplitude constant — channels disagree, no rate.
        from tests.test_upload import make_csv_bytes

        session = self._upload(client, person, make_csv_bytes(duration_s=200.0))
        metrics = db.session.query(Metrics).one()
        cautions = metrics.extras.get("cautions") or []
        assert not any("inflates RMSSD" in c["reason"] for c in cautions)
        assert session.processing_status == ProcessingStatus.DONE
