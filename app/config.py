"""Application configuration.

Everything is configurable via environment variables but defaults to sane
local values. All health data stays on the local filesystem — there is no
cloud storage or telemetry, and by default no external API call of any kind.
The single, deliberate exception is the opt-in weather lookup: when
``ECGLOG_WEATHER_ENABLED`` is set together with a home latitude/longitude,
each processed session fetches historical weather/air quality from
Open-Meteo. Only a date and the configured coordinates are ever sent; no
health data leaves the machine, and with the flag unset (the default) the
app never opens a network connection.
"""

from __future__ import annotations

import os
from pathlib import Path


def _optional_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        return None

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

    # --- Opt-in weather/air-quality lookup (docs/CONTEXT_METRICS.md §4) ----
    #: OFF by default: the app makes no network call unless this is set to
    #: 1/true/yes AND both coordinates are configured.
    WEATHER_ENABLED: bool = os.environ.get("ECGLOG_WEATHER_ENABLED", "").lower() in (
        "1",
        "true",
        "yes",
    )
    #: Home location used for every lookup (recordings made elsewhere should
    #: say so in the context note). Decimal degrees.
    HOME_LAT: float | None = _optional_float("ECGLOG_HOME_LAT")
    HOME_LON: float | None = _optional_float("ECGLOG_HOME_LON")
    #: Per-request timeout; a slow lookup delays only the processing thread,
    #: never the upload request, and a failed one never fails the session.
    WEATHER_TIMEOUT_S: int = int(os.environ.get("ECGLOG_WEATHER_TIMEOUT_S", 10))

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
    #: Tests never touch the network, whatever the developer's env says.
    WEATHER_ENABLED: bool = False
    HOME_LAT: float | None = None
    HOME_LON: float | None = None
