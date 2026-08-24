"""Application factory for the Polar H10 ECG logger.

A local, offline screening and fitness tool. It is not a diagnostic device:
nothing it outputs is a diagnosis, and "no flags raised" is not clinical
clearance.
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask

from app.config import Config
from app.extensions import db, migrate


def create_app(config_object: object | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_object or Config())

    _ensure_data_dirs(app)

    db.init_app(app)
    migrate.init_app(app, db)

    from app import models  # noqa: F401 - register models with the metadata
    from app.blueprints.activities import bp as activities_bp
    from app.blueprints.compare import bp as compare_bp
    from app.blueprints.home import bp as home_bp
    from app.blueprints.logbook import bp as logbook_bp
    from app.blueprints.people import bp as people_bp
    from app.blueprints.report_preview import bp as report_preview_bp
    from app.blueprints.sessions import bp as sessions_bp
    from app.blueprints.sleep import bp as sleep_bp
    from app.blueprints.triggers import bp as triggers_bp

    app.register_blueprint(home_bp)
    app.register_blueprint(people_bp)
    app.register_blueprint(activities_bp)
    app.register_blueprint(report_preview_bp)
    app.register_blueprint(sessions_bp)
    app.register_blueprint(sleep_bp)
    app.register_blueprint(compare_bp)
    app.register_blueprint(logbook_bp)
    app.register_blueprint(triggers_bp)

    # Opt-in only: with ECGLOG_POLAR_ENABLED unset these URLs do not exist,
    # so no code path can reach Polar by accident.
    if app.config.get("POLAR_ENABLED"):
        from app.blueprints.polar import bp as polar_bp

        app.register_blueprint(polar_bp)

    from app.cli import register_cli

    register_cli(app)

    from flask import render_template

    @app.errorhandler(404)
    def not_found(_error: object):
        return render_template("errors/404.html"), 404

    @app.errorhandler(500)
    def server_error(_error: object):
        return render_template("errors/500.html"), 500

    return app


def _ensure_data_dirs(app: Flask) -> None:
    """Create the local data directories if missing (skipped for in-memory tests)."""
    if app.config.get("TESTING"):
        return
    data_dir = Path(app.config["DATA_DIR"])
    for sub in (data_dir, data_dir / "uploads", data_dir / "logs"):
        sub.mkdir(parents=True, exist_ok=True)
