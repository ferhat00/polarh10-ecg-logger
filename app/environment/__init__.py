"""Environment context: opt-in weather/air-quality capture and formatting.

The client (``app.environment.client``) talks to Open-Meteo **only** when the
wearer has explicitly opted in (``ECGLOG_WEATHER_ENABLED=1`` plus a home
latitude/longitude) — with the flag unset the app never makes a network call.
This module also holds the shared presentation helper so the report, the
session page, and the logbook print the same line.
"""

from __future__ import annotations

from typing import Protocol


class _HasEnvColumns(Protocol):  # pragma: no cover - typing aid only
    env_temp_c: float | None
    env_apparent_temp_c: float | None
    env_humidity_pct: float | None
    env_pressure_hpa: float | None
    env_pm25_ugm3: float | None
    env_aqi: float | None
    env_fetched_at: object
    env_source: str | None


def format_environment_line(session: _HasEnvColumns) -> str | None:
    """One human-readable line from a session's env columns; None if unfetched.

    Partial data is normal (regional air-quality gaps): only present values
    are printed, and the source is always named.
    """
    if session.env_fetched_at is None:
        return None
    bits: list[str] = []
    if session.env_temp_c is not None:
        temp = f"{session.env_temp_c:.1f} °C"
        if session.env_apparent_temp_c is not None:
            temp += f" (feels {session.env_apparent_temp_c:.1f})"
        bits.append(temp)
    if session.env_humidity_pct is not None:
        bits.append(f"RH {session.env_humidity_pct:.0f}%")
    if session.env_pressure_hpa is not None:
        bits.append(f"{session.env_pressure_hpa:.0f} hPa")
    if session.env_pm25_ugm3 is not None:
        bits.append(f"PM2.5 {session.env_pm25_ugm3:.0f} µg/m³")
    if session.env_aqi is not None:
        bits.append(f"AQI {session.env_aqi:.0f}")
    if not bits:
        return None
    return " · ".join(bits) + f" ({session.env_source or 'unknown source'}, window-averaged)"
