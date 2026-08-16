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
    from app.blueprints.people import bp as people_bp
    from app.blueprints.report_preview import bp as report_preview_bp
    from app.blueprints.sessions import bp as sessions_bp

    app.register_blueprint(home_bp)
    app.register_blueprint(people_bp)
    app.register_blueprint(activities_bp)
    app.register_blueprint(report_preview_bp)
    app.register_blueprint(sessions_bp)
    app.register_blueprint(compare_bp)

    return app


def _ensure_data_dirs(app: Flask) -> None:
    """Create the local data directories if missing (skipped for in-memory tests)."""
    if app.config.get("TESTING"):
        return
    data_dir = Path(app.config["DATA_DIR"])
    for sub in (data_dir, data_dir / "uploads", data_dir / "logs"):
        sub.mkdir(parents=True, exist_ok=True)
