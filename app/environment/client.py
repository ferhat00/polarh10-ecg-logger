"""Open-Meteo client for the opt-in environment lookup.

Design constraints (docs/CONTEXT_METRICS.md §4):

* **Opt-in only** — callers gate on config; this module never reads config.
* **stdlib only** — ``urllib`` keeps the dependency surface unchanged; the
  ``opener`` parameter is the injection seam so tests never touch the
  network.
* **Never raises** — a failed or partial lookup returns what it got (or
  ``None``) with human-readable notes. Missing air-quality coverage is a
  normal outcome, not an error.
* **Minimal data out** — the request carries a date range and coordinates,
  nothing else.

Endpoint choice: the historical *archive* API (ERA5) lags realtime by
several days, so recent recordings use the *forecast* API's ``past_days``
window (up to 92 days back) and older ones fall back to the archive.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

#: Recordings at most this many days old use the forecast API (past_days
#: covers 92; 85 leaves slack for long recordings crossing midnight).
FORECAST_MAX_AGE_DAYS = 85

_HOURLY_WEATHER = "temperature_2m,relative_humidity_2m,apparent_temperature,surface_pressure"
_HOURLY_AIR = "pm2_5,pm10,ozone,nitrogen_dioxide,european_aqi"

WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


@dataclass
class EnvironmentSample:
    """Window-averaged environment values for one recording."""

    temp_c: float | None = None
    apparent_temp_c: float | None = None
    humidity_pct: float | None = None
    pressure_hpa: float | None = None
    pm25_ugm3: float | None = None
    pm10_ugm3: float | None = None
    ozone_ugm3: float | None = None
    no2_ugm3: float | None = None
    aqi: float | None = None
    daylight_h: float | None = None
    source: str = ""
    notes: list[str] = field(default_factory=list)

    def has_any_value(self) -> bool:
        return any(
            getattr(self, name) is not None
            for name in (
                "temp_c",
                "apparent_temp_c",
                "humidity_pct",
                "pressure_hpa",
                "pm25_ugm3",
                "pm10_ugm3",
                "ozone_ugm3",
                "no2_ugm3",
                "aqi",
                "daylight_h",
            )
        )


def fetch_environment(
    lat: float,
    lon: float,
    start_utc: dt.datetime,
    duration_s: float,
    timeout_s: float = 10.0,
    opener=urllib.request.urlopen,
    now: dt.datetime | None = None,
) -> EnvironmentSample | None:
    """Fetch and window-average weather + air quality for one recording.

    Returns ``None`` only when *nothing* was retrievable; otherwise a sample
    whose missing fields are ``None`` with the reason in ``notes``.
    """
    if start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=dt.UTC)
    end_utc = start_utc + dt.timedelta(seconds=max(duration_s, 0.0))
    now = now or dt.datetime.now(dt.UTC)

    sample = EnvironmentSample()
    age_days = (now.date() - start_utc.date()).days

    # --- weather ----------------------------------------------------------
    if age_days < 0:
        sample.notes.append("recording start is in the future — weather skipped")
    else:
        if age_days <= FORECAST_MAX_AGE_DAYS:
            url = WEATHER_FORECAST_URL + "?" + urllib.parse.urlencode(
                {
                    "latitude": f"{lat:.4f}",
                    "longitude": f"{lon:.4f}",
                    "past_days": min(age_days + 1, 92),
                    "forecast_days": 1,
                    "hourly": _HOURLY_WEATHER,
                    "daily": "daylight_duration",
                    "timezone": "UTC",
                }
            )
            sample.source = "open-meteo-forecast"
        else:
            url = WEATHER_ARCHIVE_URL + "?" + urllib.parse.urlencode(
                {
                    "latitude": f"{lat:.4f}",
                    "longitude": f"{lon:.4f}",
                    "start_date": start_utc.date().isoformat(),
                    "end_date": end_utc.date().isoformat(),
                    "hourly": _HOURLY_WEATHER,
                    "daily": "daylight_duration",
                    "timezone": "UTC",
                }
            )
            sample.source = "open-meteo-archive"

        body = _get_json(url, timeout_s, opener, sample.notes, what="weather")
        if body is not None:
            hourly = body.get("hourly") or {}
            sample.temp_c = _window_mean(hourly, "temperature_2m", start_utc, end_utc)
            sample.apparent_temp_c = _window_mean(
                hourly, "apparent_temperature", start_utc, end_utc
            )
            sample.humidity_pct = _window_mean(
                hourly, "relative_humidity_2m", start_utc, end_utc
            )
            sample.pressure_hpa = _window_mean(
                hourly, "surface_pressure", start_utc, end_utc
            )
            sample.daylight_h = _daylight_hours(body.get("daily") or {}, start_utc)

    # --- air quality (separate host; regional gaps are normal) ------------
    air_url = AIR_QUALITY_URL + "?" + urllib.parse.urlencode(
        {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "start_date": start_utc.date().isoformat(),
            "end_date": end_utc.date().isoformat(),
            "hourly": _HOURLY_AIR,
            "timezone": "UTC",
        }
    )
    air = _get_json(air_url, timeout_s, opener, sample.notes, what="air quality")
    if air is not None:
        hourly = air.get("hourly") or {}
        sample.pm25_ugm3 = _window_mean(hourly, "pm2_5", start_utc, end_utc)
        sample.pm10_ugm3 = _window_mean(hourly, "pm10", start_utc, end_utc)
        sample.ozone_ugm3 = _window_mean(hourly, "ozone", start_utc, end_utc)
        sample.no2_ugm3 = _window_mean(hourly, "nitrogen_dioxide", start_utc, end_utc)
        sample.aqi = _window_mean(hourly, "european_aqi", start_utc, end_utc)

    if not sample.has_any_value():
        return None
    return sample


def _get_json(url: str, timeout_s: float, opener, notes: list[str], what: str) -> dict | None:
    """One GET → parsed JSON; failures become a note, never an exception."""
    try:
        with opener(url, timeout=timeout_s) as resp:
            payload = resp.read()
        body = json.loads(payload)
        if not isinstance(body, dict):
            raise ValueError("unexpected response shape")
        if body.get("error"):
            notes.append(f"{what} lookup rejected: {body.get('reason', 'unknown reason')}")
            return None
        return body
    except Exception as exc:  # noqa: BLE001 - any failure is a supported outcome
        notes.append(f"{what} lookup failed: {type(exc).__name__}: {exc}")
        return None


def _window_mean(
    hourly: dict,
    key: str,
    start_utc: dt.datetime,
    end_utc: dt.datetime,
) -> float | None:
    """Mean of the hourly slots overlapping the recording window.

    Open-Meteo returns parallel arrays with ISO-hour timestamps (UTC, because
    every request pins ``timezone=UTC``) and ``null`` for missing values. A
    short session inside one hour gets the nearest slot.
    """
    times = hourly.get("time") or []
    values = hourly.get(key) or []
    if not times or not values:
        return None

    parsed: list[tuple[dt.datetime, float]] = []
    for stamp, value in zip(times, values, strict=False):
        if value is None:
            continue
        try:
            t = dt.datetime.fromisoformat(stamp).replace(tzinfo=dt.UTC)
        except ValueError:
            continue
        parsed.append((t, float(value)))
    if not parsed:
        return None

    # Slots overlapping [start, end): slot t covers [t, t+1h).
    hour = dt.timedelta(hours=1)
    in_window = [v for t, v in parsed if t < end_utc and (t + hour) > start_utc]
    if not in_window:
        nearest = min(parsed, key=lambda tv: abs((tv[0] - start_utc).total_seconds()))
        # Only accept a nearest-slot fallback within 2 h of the recording.
        if abs((nearest[0] - start_utc).total_seconds()) > 7200:
            return None
        return nearest[1]
    return sum(in_window) / len(in_window)


def _daylight_hours(daily: dict, start_utc: dt.datetime) -> float | None:
    """Daylight duration (h) for the recording's start date."""
    times = daily.get("time") or []
    values = daily.get("daylight_duration") or []
    target = start_utc.date().isoformat()
    for stamp, value in zip(times, values, strict=False):
        if stamp == target and value is not None:
            return float(value) / 3600.0
    return None
