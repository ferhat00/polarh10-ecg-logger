"""Posture classification from the chest accelerometer.

The synthetic ACC builder writes a known gravity vector, so each class is
tested against ground truth. Frame (verified in-repo convention): supine ⇒
gravity ≈ +Z, torso long axis = Y, left–right = X.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from flask import Flask
from flask.testing import FlaskClient

from app.extensions import db
from app.ingest.acc_loader import load_polar_acc_csv
from app.models import ActivityType, Person, ProcessingStatus, Session
from app.pipeline.posture import (
    POSTURE_CODES,
    PostureResult,
    autofill_position,
    classify_posture,
)
from app.sleep.actigraphy import activity_counts
from tests.synth_util import make_acc_csv_bytes
from tests.test_upload import START, make_csv_bytes

#: Gravity vectors (x, y, z) in mg per expected class, H10 frame.
GRAVITY_CASES = {
    "supine": (0.0, 0.0, 1000.0),
    "prone": (0.0, 0.0, -1000.0),
    "left": (900.0, 0.0, 430.0),
    "right": (-900.0, 0.0, 430.0),
    "upright": (0.0, -990.0, 140.0),
}


def _load(payload: bytes):
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp()) / "acc.csv"
    path.write_bytes(payload)
    return load_polar_acc_csv(str(path))


def _classify(payload: bytes, duration_s: float = 120.0):
    acc = _load(payload)
    n_epochs = int(np.ceil(duration_s / 30.0))
    counts = activity_counts(acc, acc.start_time, n_epochs)
    return classify_posture(acc, acc.start_time, n_epochs, acc_epochs=counts)


class TestClassification:
    @pytest.mark.parametrize(("expected", "gravity"), sorted(GRAVITY_CASES.items()))
    def test_known_gravity_vectors(self, expected: str, gravity) -> None:
        posture = _classify(make_acc_csv_bytes(duration_s=120.0, gravity=gravity))
        assert posture.dominant == expected
        assert posture.dominant_pct == pytest.approx(100.0)
        assert posture.n_transitions == 0

    def test_default_fixture_vector_is_supine(self) -> None:
        # The historical synth vector (120, 0, 990) leans slightly left but
        # is within the ±45° supine cone.
        posture = _classify(make_acc_csv_bytes(duration_s=120.0))
        assert posture.dominant == "supine"

    def test_movement_bursts_become_moving(self) -> None:
        payload = make_acc_csv_bytes(
            duration_s=180.0,
            movement_bursts=[(60.0, 120.0, 400.0)],
            gravity=(0.0, 0.0, 1000.0),
        )
        posture = _classify(payload, duration_s=180.0)
        codes = posture.codes
        assert codes[0] == POSTURE_CODES["supine"]
        # The burst epochs (2 and 3) are vetoed as moving, not classified.
        assert codes[2] == POSTURE_CODES["moving"]
        assert codes[3] == POSTURE_CODES["moving"]
        # Moving epochs are excluded from the posture percentages.
        assert posture.pct_by_posture["supine"] == pytest.approx(100.0)

    def test_implausible_gravity_magnitude_is_unknown(self) -> None:
        posture = _classify(
            make_acc_csv_bytes(duration_s=120.0, gravity=(0.0, 0.0, 200.0))
        )
        assert posture.dominant is None
        assert posture.valid_pct == 0.0
        assert all(c == POSTURE_CODES["unknown"] for c in posture.codes)

    def test_transitions_counted(self) -> None:
        # Two recordings stitched: supine 60 s then right-lateral 60 s.
        supine = make_acc_csv_bytes(duration_s=60.0, gravity=(0.0, 0.0, 1000.0))
        acc = _load(supine)
        # Build arrays directly: 2 epochs supine, 2 epochs right.
        n = len(acc.time_s)
        acc.x_mg = np.concatenate([acc.x_mg, -900.0 + acc.x_mg[:n]])
        acc.y_mg = np.concatenate([acc.y_mg, acc.y_mg[:n]])
        acc.z_mg = np.concatenate([acc.z_mg, 430.0 + 0 * acc.z_mg[:n]])
        acc.time_s = np.concatenate([acc.time_s, acc.time_s + 60.0])
        posture = classify_posture(acc, acc.start_time, 4)
        assert posture.n_transitions == 1
        assert posture.pct_by_posture["supine"] == pytest.approx(50.0)
        assert posture.pct_by_posture["right"] == pytest.approx(50.0)


class TestAutofill:
    def _result(self, dominant: str, dominant_pct: float, valid_pct: float) -> PostureResult:
        return PostureResult(
            epoch_start_s=np.array([0.0]),
            codes=np.array([1], dtype=np.int8),
            pct_by_posture={dominant: dominant_pct},
            dominant=dominant,
            dominant_pct=dominant_pct,
            n_transitions=0,
            valid_pct=valid_pct,
            mean_gravity_mg=(0.0, 0.0, 1000.0),
        )

    def test_lying_dominant_fills(self) -> None:
        assert autofill_position(self._result("supine", 85.0, 90.0)) == "supine"

    def test_upright_never_fills(self) -> None:
        # A chest strap cannot split sitting from standing.
        assert autofill_position(self._result("upright", 100.0, 100.0)) is None

    def test_weak_dominance_or_coverage_refrains(self) -> None:
        assert autofill_position(self._result("left", 55.0, 90.0)) is None
        assert autofill_position(self._result("left", 90.0, 30.0)) is None


class TestUploadIntegration:
    @pytest.fixture()
    def person(self, app: Flask) -> Person:
        p = Person(name="Posture", slug="posture")
        db.session.add(p)
        db.session.commit()
        return p

    def _upload_with_acc(
        self,
        client: FlaskClient,
        person: Person,
        payload: bytes,
        acc_payload: bytes,
        body_position: str = "",
    ) -> Session:
        client.get("/sessions/upload")
        activity = db.session.query(ActivityType).filter_by(profile_key="sitting").one()
        resp = client.post(
            "/sessions/upload",
            data={
                "person_id": str(person.id),
                "activity_type_id": str(activity.id),
                "context_note": "",
                "body_position": body_position,
                "file": (io.BytesIO(payload), "ecg_2026-08-10.csv"),
                "acc_file": (io.BytesIO(acc_payload), "acc_2026-08-10.csv"),
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 302
        db.session.expire_all()
        return db.session.query(Session).one()

    def test_acc_autofills_lying_position(
        self, client: FlaskClient, app: Flask, person: Person
    ) -> None:
        session = self._upload_with_acc(
            client,
            person,
            make_csv_bytes(),
            make_acc_csv_bytes(
                duration_s=150.0, start=START, gravity=(0.0, 0.0, 1000.0)
            ),
        )
        assert session.processing_status == ProcessingStatus.DONE
        assert session.body_position == "supine"
        assert session.body_position_source == "acc"
        posture = session.metrics.extras["posture"]
        assert posture["dominant"] == "supine"
        assert posture["disclaimer"]
        assert posture["mean_gravity_mg"] is not None

        report = client.get(f"/sessions/{session.id}/report")
        assert b"Posture per 30 s epoch" in report.data

        from app.logbook.writer import log_path_for

        text = log_path_for(person).read_text(encoding="utf-8")
        assert "**Posture (ACC):**" in text

    def test_user_position_survives_acc(
        self, client: FlaskClient, app: Flask, person: Person
    ) -> None:
        session = self._upload_with_acc(
            client,
            person,
            make_csv_bytes(),
            make_acc_csv_bytes(
                duration_s=150.0, start=START, gravity=(0.0, 0.0, 1000.0)
            ),
            body_position="sitting",
        )
        assert session.body_position == "sitting"
        assert session.body_position_source == "user"
        # Measured posture still recorded in extras for comparison.
        assert session.metrics.extras["posture"]["dominant"] == "supine"

    def test_upright_acc_never_autofills(
        self, client: FlaskClient, app: Flask, person: Person
    ) -> None:
        session = self._upload_with_acc(
            client,
            person,
            make_csv_bytes(),
            make_acc_csv_bytes(
                duration_s=150.0, start=START, gravity=(0.0, -990.0, 140.0)
            ),
        )
        assert session.body_position is None
        assert session.metrics.extras["posture"]["dominant"] == "upright"

    def test_cache_gains_posture_arrays(
        self, client: FlaskClient, app: Flask, person: Person
    ) -> None:
        from app.processing import cache_path_for

        session = self._upload_with_acc(
            client,
            person,
            make_csv_bytes(),
            make_acc_csv_bytes(
                duration_s=150.0, start=START, gravity=(0.0, 0.0, 1000.0)
            ),
        )
        cache = np.load(cache_path_for(session))
        assert int(cache["cache_version"][0]) >= 4
        assert "posture_codes" in cache
        assert len(cache["posture_codes"]) == len(cache["posture_epoch_start_s"])
