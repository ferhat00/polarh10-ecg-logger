"""Application configuration.

Everything is configurable via environment variables but defaults to sane
local values. All health data stays on the local filesystem — there is no
cloud storage, telemetry, or external API of any kind.
"""

from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Config:
    """Default (local development) configuration."""

    SECRET_KEY: str = os.environ.get("ECGLOG_SECRET_KEY", "local-only-not-a-secret")

    #: Root directory for all persisted health data (SQLite DB, uploaded CSVs,
    #: processed caches, decision logs). Local disk only.
    DATA_DIR: Path = Path(os.environ.get("ECGLOG_DATA_DIR", _PROJECT_ROOT / "data"))

    SQLALCHEMY_DATABASE_URI: str = os.environ.get(
        "ECGLOG_DATABASE_URI", f"sqlite:///{DATA_DIR / 'app.db'}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False

    #: An 18-minute 130 Hz export is ~5 MB and a full 8 h night ~135 MB;
    #: 512 MB leaves headroom for overnight ECG + accelerometer uploads.
    MAX_CONTENT_LENGTH: int = int(os.environ.get("ECGLOG_MAX_UPLOAD_BYTES", 512 * 1024 * 1024))

    # --- Optional external sleep-staging engine (adammj/ecg-sleep-staging) --
    #: The external 5-class engine is AGPL-licensed and therefore never
    #: vendored or imported: when these point at the user's own clone and its
    #: Python interpreter, the engine runs as a subprocess (see
    #: app/sleep/engines/external_ecg_staging.py). Unset = engine unavailable.
    SLEEP_EXTERNAL_DIR: str | None = os.environ.get("ECGLOG_SLEEP_EXTERNAL_DIR")
    SLEEP_EXTERNAL_PYTHON: str | None = os.environ.get("ECGLOG_SLEEP_EXTERNAL_PYTHON")
    #: Hard wall-clock limit for one external scoring run (an 8 h night on
    #: CPU takes minutes, not hours).
    SLEEP_EXTERNAL_TIMEOUT_S: int = int(
        os.environ.get("ECGLOG_SLEEP_EXTERNAL_TIMEOUT_S", 1800)
    )

    @property
    def UPLOAD_DIR(self) -> Path:  # noqa: N802 - Flask config naming convention
        return self.DATA_DIR / "uploads"

    @property
    def LOG_DIR(self) -> Path:  # noqa: N802
        return self.DATA_DIR / "logs"


class TestConfig(Config):
    """Configuration for the pytest suite: in-memory DB, temp data dir."""

    TESTING: bool = True
    SQLALCHEMY_DATABASE_URI: str = "sqlite://"
    WTF_CSRF_ENABLED: bool = False
    #: Run session processing inline instead of on a thread (deterministic).
    PROCESS_SYNC: bool = True
