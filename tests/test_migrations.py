"""Alembic migrations run against a real file-backed SQLite database."""

from __future__ import annotations

from pathlib import Path

import sqlalchemy as sa
from flask_migrate import downgrade, upgrade

from app import create_app
from app.config import TestConfig

MIGRATIONS_DIR = str(Path(__file__).parent.parent / "migrations")


def _app(tmp_path):
    config = TestConfig()
    config.DATA_DIR = tmp_path / "data"
    config.SQLALCHEMY_DATABASE_URI = f"sqlite:///{tmp_path / 'mig.db'}"
    return create_app(config)


def _inspector(app):
    from app.extensions import db

    return sa.inspect(db.engine)


class TestMigrations:
    def test_upgrade_builds_full_schema(self, tmp_path) -> None:
        app = _app(tmp_path)
        with app.app_context():
            upgrade(directory=MIGRATIONS_DIR)
            insp = _inspector(app)
            tables = set(insp.get_table_names())
            assert {"person", "session", "metrics", "trigger_tag", "session_trigger_tag"} <= tables
            metrics_cols = {c["name"] for c in insp.get_columns("metrics")}
            assert {
                "ectopy_beats_n",
                "ectopy_per_hour",
                "ectopy_pct_beats",
                "single_n",
                "couplet_n",
                "run_n",
                "longest_run_beats",
                "bigeminy_episode_n",
                "trigeminy_episode_n",
            } <= metrics_cols
            tag_cols = {c["name"] for c in insp.get_columns("trigger_tag")}
            assert {"id", "name", "slug", "is_builtin", "created_at"} <= tag_cols
            assert {
                "tst_min",
                "sleep_efficiency_pct",
                "sol_min",
                "waso_min",
                "light_min",
                "deep_min",
                "rem_min",
                "awakenings_n",
                "sleep_engine",
            } <= metrics_cols
            session_cols = {c["name"] for c in insp.get_columns("session")}
            assert {
                "acc_original_filename",
                "acc_stored_path",
                "acc_file_sha256",
            } <= session_cols
            person_cols = {c["name"] for c in insp.get_columns("person")}
            assert "sex" in person_cols
            # Context / environment / respiration revision.
            assert {
                "body_position",
                "body_position_source",
                "alcohol_drinks_24h",
                "sleep_quality_1_5",
                "env_temp_c",
                "env_apparent_temp_c",
                "env_humidity_pct",
                "env_pressure_hpa",
                "env_pm25_ugm3",
                "env_pm10_ugm3",
                "env_ozone_ugm3",
                "env_no2_ugm3",
                "env_aqi",
                "env_daylight_h",
                "env_source",
                "env_fetched_at",
            } <= session_cols
            assert {
                "resp_rate_median_brpm",
                "resp_rate_p5_brpm",
                "resp_rate_p95_brpm",
            } <= metrics_cols
            # Polar Flow revision.
            assert {"polar_account", "flow_night"} <= tables
            account_cols = {c["name"] for c in insp.get_columns("polar_account")}
            assert {
                "person_id",
                "polar_user_id",
                "access_token",
                "member_id",
                "linked_at",
                "last_sync_at",
                "last_sync_note",
            } <= account_cols
            night_cols = {c["name"] for c in insp.get_columns("flow_night")}
            assert {
                "person_id",
                "date",
                "source_device_id",
                "sleep_start",
                "sleep_end",
                "light_sleep_s",
                "deep_sleep_s",
                "rem_sleep_s",
                "sleep_score",
                "hr_avg_bpm",
                "beat_to_beat_avg_ms",
                "hrv_rmssd_ms",
                "nightly_recharge_status",
                "ans_charge",
                "hrv_samples",
                "fetched_at",
            } <= night_cols

    def test_downgrade_removes_new_schema(self, tmp_path) -> None:
        app = _app(tmp_path)
        with app.app_context():
            upgrade(directory=MIGRATIONS_DIR)
            # Step back over the Polar Flow revision first.
            downgrade(directory=MIGRATIONS_DIR, revision="-1")
            insp = _inspector(app)
            tables = set(insp.get_table_names())
            assert "polar_account" not in tables
            assert "flow_night" not in tables
            assert "session" in tables  # earlier revisions untouched
            # Then over the context/env/respiration revision.
            downgrade(directory=MIGRATIONS_DIR, revision="-1")
            insp = _inspector(app)
            session_cols = {c["name"] for c in insp.get_columns("session")}
            assert "body_position" not in session_cols
            assert "env_temp_c" not in session_cols
            assert "env_fetched_at" not in session_cols
            metrics_cols = {c["name"] for c in insp.get_columns("metrics")}
            assert "resp_rate_median_brpm" not in metrics_cols
            assert "tst_min" in metrics_cols  # earlier revisions untouched
            # Then over the sleep/ACC revision.
            downgrade(directory=MIGRATIONS_DIR, revision="-1")
            insp = _inspector(app)
            metrics_cols = {c["name"] for c in insp.get_columns("metrics")}
            assert "tst_min" not in metrics_cols
            session_cols = {c["name"] for c in insp.get_columns("session")}
            assert "acc_stored_path" not in session_cols
            person_cols = {c["name"] for c in insp.get_columns("person")}
            assert "sex" not in person_cols
            # Then over the trigger/ectopy revision.
            downgrade(directory=MIGRATIONS_DIR, revision="-1")
            insp = _inspector(app)
            tables = set(insp.get_table_names())
            assert "trigger_tag" not in tables
            assert "session_trigger_tag" not in tables
            metrics_cols = {c["name"] for c in insp.get_columns("metrics")}
            assert "ectopy_beats_n" not in metrics_cols
            # And back up again.
            upgrade(directory=MIGRATIONS_DIR)
            insp = _inspector(app)
            tables = set(insp.get_table_names())
            assert "trigger_tag" in tables
            assert {"polar_account", "flow_night"} <= tables
            assert "tst_min" in {c["name"] for c in insp.get_columns("metrics")}
            assert "body_position" in {c["name"] for c in insp.get_columns("session")}
