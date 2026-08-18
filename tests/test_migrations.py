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

    def test_downgrade_removes_new_schema(self, tmp_path) -> None:
        app = _app(tmp_path)
        with app.app_context():
            upgrade(directory=MIGRATIONS_DIR)
            # Step back over the sleep/ACC revision first.
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
            assert "trigger_tag" in set(insp.get_table_names())
            assert "tst_min" in {c["name"] for c in insp.get_columns("metrics")}
